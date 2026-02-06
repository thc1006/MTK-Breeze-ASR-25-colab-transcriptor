"""
Core audio enhancement engine with GPU acceleration.
"""

import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
import noisereduce as nr
import pyloudnorm as pyln
from tqdm import tqdm

from .config import EnhanceConfig


class AudioEnhancer:
    """GPU-accelerated audio enhancer.
    
    This class provides methods to enhance audio files by:
    - Normalizing loudness (LUFS)
    - Reducing noise
    - Compressing dynamic range
    - Applying EQ adjustments
    
    Example:
        >>> config = EnhanceConfig(target_loudness=-14.0)
        >>> enhancer = AudioEnhancer(config)
        >>> stats = enhancer.process("input.wav", "output.wav")
        >>> print(f"Gain applied: {stats['gain']:.1f} dB")
    """
    
    # Supported audio formats
    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
    
    def __init__(self, config: Optional[EnhanceConfig] = None):
        """Initialize the audio enhancer.
        
        Args:
            config: Enhancement configuration. Uses defaults if None.
        """
        self.config = config or EnhanceConfig()
        self.device = self._setup_device()
        self.meter = pyln.Meter(self.config.sample_rate)
        
    def _setup_device(self) -> torch.device:
        """Setup compute device (GPU or CPU).
        
        Returns:
            torch.device: The device to use for processing.
        """
        use_gpu = self.config.use_gpu
        
        if use_gpu == "auto":
            use_gpu = torch.cuda.is_available()
        
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] {gpu_name} ({gpu_mem:.1f} GB)")
            
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
        else:
            device = torch.device("cpu")
            if use_gpu:
                print("[WARN] GPU requested but not available, using CPU")
            else:
                print("[CPU] Using CPU")
        
        return device
    
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        file_path = Path(file_path)
        
        try:
            # Try torchaudio first (faster, GPU-compatible)
            waveform, sr = torchaudio.load(str(file_path))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.config.sample_rate
                ).to(self.device)
                waveform = waveform.to(self.device)
                waveform = resampler(waveform)
                waveform = waveform.cpu()
            
            return waveform.numpy().flatten(), self.config.sample_rate
            
        except Exception:
            # Fallback to librosa
            audio, sr = librosa.load(
                str(file_path), 
                sr=self.config.sample_rate, 
                mono=True
            )
            return audio, sr
    
    def denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction.
        
        Args:
            audio: Audio data.
            sr: Sample rate.
            
        Returns:
            Denoised audio data.
        """
        if not self.config.enable_denoise:
            return audio
        
        return nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=self.config.denoise_strength,
            stationary=self.config.denoise_stationary,
            n_jobs=-1  # Use all CPU cores
        )
    
    def _gpu_denoise(self, audio_tensor: torch.Tensor, sr: int) -> torch.Tensor:
        """GPU 頻譜閘門降噪

        用 torch.stft 做頻譜分析，取前 0.5 秒估算噪音底噪，
        對低於底噪 * threshold 的頻率分量做衰減。
        比 noisereduce (CPU) 快很多。

        Args:
            audio_tensor: 1-D float tensor，已在 self.device 上
            sr: 取樣率

        Returns:
            降噪後的 tensor，同裝置
        """
        if not self.config.enable_denoise:
            return audio_tensor

        n_fft = 2048
        hop_length = 512
        # 用 Hann window 做 STFT
        window = torch.hann_window(n_fft, device=audio_tensor.device)

        # STFT -> 複數頻譜
        stft = torch.stft(
            audio_tensor, n_fft=n_fft, hop_length=hop_length,
            window=window, return_complex=True,
        )
        magnitude = torch.abs(stft)

        # 取前 0.5 秒的 frame 來估算噪音頻譜
        noise_frames = max(1, int(0.5 * sr / hop_length))
        noise_frames = min(noise_frames, magnitude.shape[-1])
        noise_profile = magnitude[:, :noise_frames].mean(dim=-1, keepdim=True)

        # 頻譜閘門：低於噪音底噪的部分做衰減
        # denoise_strength 控制衰減程度（0=不衰減, 1=完全消除）
        strength = self.config.denoise_strength
        threshold_mult = 1.5  # 底噪的倍數作為門檻
        gate = torch.clamp(
            (magnitude - noise_profile * threshold_mult)
            / (magnitude + 1e-10),
            min=1.0 - strength,
            max=1.0,
        )

        # 套用 gate 到複數頻譜
        filtered = stft * gate

        # ISTFT 重建波形
        result = torch.istft(
            filtered, n_fft=n_fft, hop_length=hop_length,
            window=window, length=len(audio_tensor),
        )
        return result

    def _compress_tensor(self, audio_tensor: torch.Tensor, sr: int) -> torch.Tensor:
        """在 GPU tensor 上執行動態範圍壓縮（不搬移裝置）

        Args:
            audio_tensor: 1-D float tensor，已經在 self.device 上
            sr: 取樣率

        Returns:
            壓縮後的 tensor，同裝置
        """
        if not self.config.enable_compression:
            return audio_tensor

        abs_audio = torch.abs(audio_tensor)

        # 用 attack time 算平滑 envelope
        window_size = int(sr * self.config.compression_attack / 1000)
        if window_size > 1:
            kernel = torch.ones(window_size, device=audio_tensor.device) / window_size
            envelope = torch.nn.functional.conv1d(
                abs_audio.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=window_size // 2
            ).squeeze()

            if len(envelope) != len(audio_tensor):
                envelope = torch.nn.functional.interpolate(
                    envelope.unsqueeze(0).unsqueeze(0),
                    size=len(audio_tensor),
                    mode="linear"
                ).squeeze()
        else:
            envelope = abs_audio

        envelope_db = 20 * torch.log10(envelope + 1e-10)

        threshold = self.config.compression_threshold
        ratio = self.config.compression_ratio
        gain_db = torch.where(
            envelope_db > threshold,
            threshold + (envelope_db - threshold) / ratio - envelope_db,
            torch.zeros_like(envelope_db)
        )

        gain_linear = 10 ** (gain_db / 20)
        return audio_tensor * gain_linear

    def _apply_eq_tensor(self, audio_tensor: torch.Tensor, sr: int) -> torch.Tensor:
        """在 GPU tensor 上執行 EQ 調整（不搬移裝置）

        Args:
            audio_tensor: 1-D float tensor，已經在 self.device 上
            sr: 取樣率

        Returns:
            EQ 調整後的 tensor，同裝置
        """
        if not self.config.enable_eq:
            return audio_tensor

        # 低頻增強 (< 250 Hz)
        if self.config.bass_boost != 0:
            low_shelf = torchaudio.functional.lowpass_biquad(
                audio_tensor.unsqueeze(0), sr, 250
            ).squeeze(0)
            bass_gain = 10 ** (self.config.bass_boost / 20)
            audio_tensor = audio_tensor + (low_shelf - audio_tensor) * (bass_gain - 1) * 0.5

        # 中頻增強 (250-4000 Hz) - 人聲頻段
        if self.config.mid_boost != 0:
            mid_gain = 10 ** (self.config.mid_boost / 20)
            mid_low = torchaudio.functional.highpass_biquad(
                audio_tensor.unsqueeze(0), sr, 250
            ).squeeze(0)
            mid_band = torchaudio.functional.lowpass_biquad(
                mid_low.unsqueeze(0), sr, 4000
            ).squeeze(0)
            audio_tensor = audio_tensor + mid_band * (mid_gain - 1) * 0.3

        # 高頻增強 (> 4000 Hz)
        if self.config.treble_boost != 0:
            high_shelf = torchaudio.functional.highpass_biquad(
                audio_tensor.unsqueeze(0), sr, 4000
            ).squeeze(0)
            treble_gain = 10 ** (self.config.treble_boost / 20)
            audio_tensor = audio_tensor + high_shelf * (treble_gain - 1) * 0.5

        return audio_tensor

    def _gpu_pipeline(
        self, audio: np.ndarray, sr: int, include_denoise: bool = False,
    ) -> np.ndarray:
        """在 GPU 上連續執行 denoise + compress + EQ，只搬移一次

        整合所有 GPU 可加速的步驟到單一 pass：
        - denoise: 用 torch.stft 頻譜閘門（取代 noisereduce CPU 瓶頸）
        - compress: 動態範圍壓縮
        - EQ: 頻段增強

        資料只搬進搬出 GPU 各一次（2 次 transfer）。

        Args:
            audio: numpy array
            sr: 取樣率
            include_denoise: 是否在 GPU pipeline 內執行降噪
                True = 用 GPU STFT 降噪（快，但效果與 noisereduce 略有不同）
                False = 不在此做降噪（由呼叫端自行用 denoise() 處理）

        Returns:
            處理後的 numpy array
        """
        need_denoise = include_denoise and self.config.enable_denoise
        need_compress = self.config.enable_compression
        need_eq = self.config.enable_eq

        if not need_denoise and not need_compress and not need_eq:
            return audio

        # 搬進 GPU（1 次 transfer）
        audio_tensor = torch.from_numpy(audio).float().to(self.device)

        if need_denoise:
            audio_tensor = self._gpu_denoise(audio_tensor, sr)
        if need_compress:
            audio_tensor = self._compress_tensor(audio_tensor, sr)
        if need_eq:
            audio_tensor = self._apply_eq_tensor(audio_tensor, sr)

        # 搬回 CPU（1 次 transfer）
        return audio_tensor.cpu().numpy()

    def compress(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply dynamic range compression.

        Args:
            audio: Audio data.
            sr: Sample rate.

        Returns:
            Compressed audio data.
        """
        if not self.config.enable_compression:
            return audio

        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        result = self._compress_tensor(audio_tensor, sr)
        return result.cpu().numpy()

    def apply_eq(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply EQ adjustments.

        Args:
            audio: Audio data.
            sr: Sample rate.

        Returns:
            EQ-adjusted audio data.
        """
        if not self.config.enable_eq:
            return audio

        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        result = self._apply_eq_tensor(audio_tensor, sr)
        return result.cpu().numpy()
    
    def normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target loudness.
        
        Args:
            audio: Audio data.
            
        Returns:
            Loudness-normalized audio data.
        """
        # Measure current loudness
        current_loudness = self.meter.integrated_loudness(audio)
        
        if np.isinf(current_loudness):
            # Audio is too quiet, use peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                ceiling = 10 ** (self.config.peak_ceiling / 20)
                return audio * (ceiling / peak)
            return audio
        
        # Calculate and apply gain
        gain_db = self.config.target_loudness - current_loudness
        gain_linear = 10 ** (gain_db / 20)
        normalized = audio * gain_linear
        
        # Apply peak ceiling (limiter)
        ceiling = 10 ** (self.config.peak_ceiling / 20)
        normalized = np.clip(normalized, -ceiling, ceiling)
        
        return normalized
    
    def _get_meter(self, sr: int) -> pyln.Meter:
        """取得對應取樣率的 loudness meter

        pyloudnorm.Meter 綁定取樣率，不同取樣率需要不同的 Meter 實例。
        self.meter 對應 config.sample_rate（通常 44100）；
        若傳入的 sr 不同，就另外建一個。

        Args:
            sr: 取樣率

        Returns:
            對應取樣率的 Meter 實例
        """
        if sr == self.config.sample_rate:
            return self.meter
        return pyln.Meter(sr)

    def process_array(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """直接處理 numpy array（不經過磁碟 I/O）

        跟 process() 走一樣的增強流程（denoise -> compress -> EQ -> loudness），
        但省掉讀寫檔案的步驟，適合當作 ASR 前處理。

        Args:
            audio: mono numpy array，任意取樣率
            sr: 取樣率

        Returns:
            (enhanced_audio, stats)
            stats 包含 original_loudness / final_loudness / gain
        """
        meter = self._get_meter(sr)
        original_loudness = meter.integrated_loudness(audio)

        # GPU 可用時：denoise + compress + EQ 全在 GPU 上做（單一 pass）
        # GPU 不可用時：denoise 走 noisereduce (CPU)，compress + EQ 走 CPU tensor
        use_gpu_denoise = (self.device.type == "cuda")

        if use_gpu_denoise:
            audio = self._gpu_pipeline(audio, sr, include_denoise=True)
        else:
            audio = self.denoise(audio, sr)
            audio = self._gpu_pipeline(audio, sr, include_denoise=False)

        # 響度標準化（用對應取樣率的 meter）
        current_loudness = meter.integrated_loudness(audio)
        if not np.isinf(current_loudness):
            gain_db = self.config.target_loudness - current_loudness
            gain_linear = 10 ** (gain_db / 20)
            audio = audio * gain_linear
            ceiling = 10 ** (self.config.peak_ceiling / 20)
            audio = np.clip(audio, -ceiling, ceiling)
        else:
            # 太安靜，做 peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                ceiling = 10 ** (self.config.peak_ceiling / 20)
                audio = audio * (ceiling / peak)

        final_loudness = meter.integrated_loudness(audio)

        stats = {
            "original_loudness": float(original_loudness),
            "final_loudness": float(final_loudness),
            "gain": float(final_loudness - original_loudness),
        }
        return audio, stats

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> Dict[str, float]:
        """Process a single audio file.
        
        Args:
            input_path: Path to input audio file.
            output_path: Path to save enhanced audio.
            
        Returns:
            Dictionary with processing statistics:
                - original_loudness: Original loudness in LUFS
                - final_loudness: Final loudness in LUFS
                - gain: Total gain applied in dB
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        audio, sr = self.load_audio(input_path)
        original_loudness = self.meter.integrated_loudness(audio)
        
        # 增強流程：denoise (CPU) -> compress + EQ (GPU pipeline) -> loudness
        audio = self.denoise(audio, sr)
        audio = self._gpu_pipeline(audio, sr)
        audio = self.normalize_loudness(audio)
        
        # Get final stats
        final_loudness = self.meter.integrated_loudness(audio)
        
        # Save output
        sf.write(str(output_path), audio, sr)
        
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "original_loudness": original_loudness,
            "final_loudness": final_loudness,
            "gain": final_loudness - original_loudness
        }
    
    def process_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        recursive: bool = False,
        show_progress: bool = True
    ) -> List[Dict]:
        """Process all audio files in a directory.
        
        Args:
            input_dir: Input directory path.
            output_dir: Output directory path.
            recursive: Process subdirectories recursively.
            show_progress: Show progress bar.
            
        Returns:
            List of results for each file.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        if recursive:
            files = [
                f for f in input_dir.rglob("*")
                if f.suffix.lower() in self.SUPPORTED_FORMATS
            ]
        else:
            files = [
                f for f in input_dir.iterdir()
                if f.suffix.lower() in self.SUPPORTED_FORMATS
            ]
        
        if not files:
            print(f"No audio files found in {input_dir}")
            return []
        
        print(f"Found {len(files)} audio files")
        
        results = []
        iterator = tqdm(files, desc="Processing") if show_progress else files
        
        for input_path in iterator:
            # Generate output path
            rel_path = input_path.relative_to(input_dir)
            output_path = output_dir / rel_path.parent / f"{rel_path.stem}_enhanced.{self.config.output_format}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                stats = self.process(input_path, output_path)
                results.append({
                    "file": str(input_path.name),
                    "status": "success",
                    "output": str(output_path),
                    **stats
                })
                
                if show_progress:
                    tqdm.write(
                        f"  [OK] {input_path.name}: "
                        f"{stats['original_loudness']:.1f} -> {stats['final_loudness']:.1f} LUFS "
                        f"({stats['gain']:+.1f} dB)"
                    )
                    
            except Exception as e:
                results.append({
                    "file": str(input_path.name),
                    "status": "failed",
                    "error": str(e)
                })
                if show_progress:
                    tqdm.write(f"  [FAIL] {input_path.name}: {e}")
        
        # Summary
        success = sum(1 for r in results if r["status"] == "success")
        print(f"\n[Done] {success}/{len(results)} files")

        if success > 0:
            avg_gain = np.mean([r["gain"] for r in results if r["status"] == "success"])
            print(f"[Avg gain] {avg_gain:+.1f} dB")
        
        return results
