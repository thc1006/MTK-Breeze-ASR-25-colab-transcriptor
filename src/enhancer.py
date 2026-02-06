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
        
        # Move to GPU
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        # Calculate envelope
        abs_audio = torch.abs(audio_tensor)
        
        # Smooth envelope with attack time
        window_size = int(sr * self.config.compression_attack / 1000)
        if window_size > 1:
            kernel = torch.ones(window_size, device=self.device) / window_size
            envelope = torch.nn.functional.conv1d(
                abs_audio.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=window_size // 2
            ).squeeze()
            
            # Ensure same length
            if len(envelope) != len(audio_tensor):
                envelope = torch.nn.functional.interpolate(
                    envelope.unsqueeze(0).unsqueeze(0),
                    size=len(audio_tensor),
                    mode="linear"
                ).squeeze()
        else:
            envelope = abs_audio
        
        # Convert to dB
        envelope_db = 20 * torch.log10(envelope + 1e-10)
        
        # Calculate gain reduction
        threshold = self.config.compression_threshold
        ratio = self.config.compression_ratio
        
        gain_db = torch.where(
            envelope_db > threshold,
            threshold + (envelope_db - threshold) / ratio - envelope_db,
            torch.zeros_like(envelope_db)
        )
        
        # Apply gain
        gain_linear = 10 ** (gain_db / 20)
        compressed = audio_tensor * gain_linear
        
        return compressed.cpu().numpy()
    
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
        
        # Move to GPU
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        # Bass boost (< 250 Hz)
        if self.config.bass_boost != 0:
            low_shelf = torchaudio.functional.lowpass_biquad(
                audio_tensor.unsqueeze(0), sr, 250
            ).squeeze(0)
            bass_gain = 10 ** (self.config.bass_boost / 20)
            audio_tensor = audio_tensor + (low_shelf - audio_tensor) * (bass_gain - 1) * 0.5
        
        # Mid boost (250-4000 Hz) - voice range
        if self.config.mid_boost != 0:
            mid_gain = 10 ** (self.config.mid_boost / 20)
            mid_low = torchaudio.functional.highpass_biquad(
                audio_tensor.unsqueeze(0), sr, 250
            ).squeeze(0)
            mid_band = torchaudio.functional.lowpass_biquad(
                mid_low.unsqueeze(0), sr, 4000
            ).squeeze(0)
            audio_tensor = audio_tensor + mid_band * (mid_gain - 1) * 0.3
        
        # Treble boost (> 4000 Hz)
        if self.config.treble_boost != 0:
            high_shelf = torchaudio.functional.highpass_biquad(
                audio_tensor.unsqueeze(0), sr, 4000
            ).squeeze(0)
            treble_gain = 10 ** (self.config.treble_boost / 20)
            audio_tensor = audio_tensor + high_shelf * (treble_gain - 1) * 0.5
        
        return audio_tensor.cpu().numpy()
    
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
        
        # Process pipeline
        audio = self.denoise(audio, sr)
        audio = self.compress(audio, sr)
        audio = self.apply_eq(audio, sr)
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
