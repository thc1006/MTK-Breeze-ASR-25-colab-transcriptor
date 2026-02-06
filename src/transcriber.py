"""
Breeze-ASR-25 語音轉文字轉錄器

使用 HuggingFace Transformers 載入官方 MediaTek-Research/Breeze-ASR-25 模型，
搭配 Silero-VAD 做語音區段偵測，支援 bitsandbytes 量化降級。
"""

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# ============================================================================
# RTX 3050 4GB 專用優化配置
# ============================================================================

@dataclass
class RTX3050Config:
    """
    RTX 3050 Laptop 4GB VRAM 專屬優化配置

    使用 HuggingFace Transformers 搭配 bitsandbytes NF4 量化
    """

    # --- 計算精度 ---
    dtype: str = "float16"
    quant_mode: str = "bnb_4bit"  # NF4 量化省 VRAM

    # --- Beam Search ---
    beam_size: int = 1  # 4GB VRAM 建議用 1，速度優先

    # --- VAD 語音活動偵測 ---
    vad_filter: bool = True
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 100
    vad_speech_pad_ms: int = 30
    vad_max_speech_duration: float = 30

    # --- Pipeline 分段 ---
    chunk_length_s: int = 30  # pipeline 每次處理的秒數

    # --- 速度優化 ---
    condition_on_previous_text: bool = False  # 關閉上下文條件，加速
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

    # --- 輸出控制 ---
    word_timestamps: bool = False
    without_timestamps: bool = False

    # --- 音頻前處理 ---
    enhance_audio: bool = False  # 啟用降噪+人聲增強前處理


@dataclass
class QualityConfig:
    """品質優先配置（較慢但更準確）"""
    dtype: str = "float16"
    quant_mode: str = "none"

    beam_size: int = 3
    vad_filter: bool = True
    vad_threshold: float = 0.4
    vad_min_speech_ms: int = 200
    vad_min_silence_ms: int = 150
    vad_speech_pad_ms: int = 50
    vad_max_speech_duration: float = 30
    chunk_length_s: int = 30
    condition_on_previous_text: bool = True  # 開啟提升準確度
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    word_timestamps: bool = False
    without_timestamps: bool = False
    enhance_audio: bool = False


@dataclass
class GPUConfig:
    """
    自適應 GPU 配置

    由 _auto_config() 根據偵測到的 VRAM 和 compute capability 自動產生，
    也可手動建立覆蓋特定參數。
    """

    # --- 計算精度 ---
    dtype: str = "float16"
    quant_mode: str = "none"  # "none" / "bnb_4bit" / "bnb_8bit"

    # --- Beam Search ---
    beam_size: int = 1

    # --- VAD 語音活動偵測 ---
    vad_filter: bool = True
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 100
    vad_speech_pad_ms: int = 30
    vad_max_speech_duration: float = 30

    # --- Pipeline 分段 ---
    chunk_length_s: int = 30  # 0 = 不分段（高 VRAM），30 = 分段處理（低 VRAM）

    # --- 速度優化 ---
    condition_on_previous_text: bool = False
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

    # --- 輸出控制 ---
    word_timestamps: bool = False
    without_timestamps: bool = False

    # --- 音頻前處理 ---
    enhance_audio: bool = False

    # --- GPU 偵測資訊（唯讀紀錄） ---
    gpu_name: str = ""
    vram_mb: int = 0
    compute_capability: float = 0.0
    tier_name: str = "unknown"


# ============================================================================
# VRAM 分級表
# 各層對應 (tier_name, vram_upper_mb, dtype, quant_mode, beam, cond_prev, chunk_s)
# ============================================================================

_VRAM_TIERS = [
    # tier_name,  vram_upper,  dtype,      quant_mode,  beam, cond_prev, chunk_s
    ("low",       4096,        "float16",  "bnb_4bit",  1,    False,     30),
    ("mid_low",   6144,        "float16",  "bnb_8bit",  2,    False,     30),
    ("mid",       8192,        "float16",  "none",      3,    True,      30),
    ("mid_high",  11264,       "float16",  "none",      5,    True,      0),
    ("high",      24576,       "float16",  "none",      5,    True,      0),
    ("very_high", 49152,       "float16",  "none",      5,    True,      0),
]

# 超過 49152 MB 的 GPU 用這組
_ULTRA_TIER = ("ultra", 0, "float16", "none", 5, True, 0)


# ============================================================================
# 資料結構
# ============================================================================

@dataclass
class TranscriptSegment:
    """帶時間戳記的轉錄片段"""
    start: float
    end: float
    text: str

    def to_timestamp_line(self) -> str:
        """輸出時間戳記格式"""
        start_str = self._format_time(self.start)
        end_str = self._format_time(self.end)
        return f"[{start_str} - {end_str}] {self.text}"

    def to_srt_block(self, index: int) -> str:
        """輸出 SRT 字幕格式"""
        start_str = self._format_time(self.start, srt=True)
        end_str = self._format_time(self.end, srt=True)
        return f"{index}\n{start_str} --> {end_str}\n{self.text}\n"

    def _format_time(self, seconds: float, srt: bool = False) -> str:
        """格式化時間"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if srt:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


# ============================================================================
# 轉錄器
# ============================================================================

class BreezeTranscriber:
    """
    Breeze-ASR-25 轉錄器

    使用 HuggingFace Transformers 載入官方模型，
    搭配 Silero-VAD 做語音區段偵測。
    """

    # 官方 MediaTek Breeze-ASR-25 模型
    MODEL_ID = "MediaTek-Research/Breeze-ASR-25"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "auto",
        config: Optional[Union[RTX3050Config, QualityConfig, GPUConfig]] = None,
        download_root: Optional[str] = None,
    ):
        """
        初始化轉錄器

        Args:
            model_id: HuggingFace 模型 ID
            device: 裝置 ("auto", "cuda", "cpu")
            config: 優化配置，None = 自動偵測硬體
            download_root: 模型下載目錄（對應 HF cache_dir）
        """
        self.model_id = model_id or self.MODEL_ID
        self.download_root = download_root

        self.device = self._detect_device(device)

        if config is None:
            self.config = self._auto_config()
        else:
            self.config = config

        if self.device == "cuda":
            self._setup_cuda_optimizations()

        # 延遲載入
        self._pipeline = None
        self._processor = None
        self._vad_model = None
        self._enhancer = None

    def _detect_device(self, device: str) -> str:
        """偵測最佳裝置"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def _auto_config(self) -> GPUConfig:
        """根據 VRAM 和 compute capability 自動選擇最佳配置"""
        if self.device != "cuda":
            print("[CPU] 使用 float32 模式")
            return GPUConfig(dtype="float32", quant_mode="none", tier_name="cpu")

        props = torch.cuda.get_device_properties(0)
        vram_mb = props.total_memory // (1024 * 1024)
        gpu_name = torch.cuda.get_device_name(0)
        cc = props.major + props.minor / 10

        print(f"[GPU] {gpu_name}")
        print(f"[VRAM] {vram_mb} MB")
        print(f"[CC] {props.major}.{props.minor}")

        # 查表
        matched = None
        for tier in _VRAM_TIERS:
            if vram_mb <= tier[1]:
                matched = tier
                break

        if matched is None:
            matched = _ULTRA_TIER

        tier_name, _, dtype, quant_mode, beam, cond_prev, chunk_s = matched

        # Ampere+ (CC >= 8.0) 且 high 以上：升級到 bfloat16
        if cc >= 8.0 and dtype == "float16" and tier_name in ("high", "very_high", "ultra"):
            dtype = "bfloat16"

        cfg = GPUConfig(
            dtype=dtype,
            quant_mode=quant_mode,
            beam_size=beam,
            condition_on_previous_text=cond_prev,
            chunk_length_s=chunk_s,
            gpu_name=gpu_name,
            vram_mb=vram_mb,
            compute_capability=cc,
            tier_name=tier_name,
        )

        quant_label = f", quant={quant_mode}" if quant_mode != "none" else ""
        chunk_label = f", chunk={chunk_s}s" if chunk_s > 0 else ""
        print(f"[Profile] {tier_name} - {dtype}{quant_label}, beam={beam}{chunk_label}")

        return cfg

    def _setup_cuda_optimizations(self):
        """設定 CUDA 優化"""
        torch.backends.cudnn.benchmark = True

        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print("[CUDA] 優化設定完成")

    # ------------------------------------------------------------------
    # dtype 解析
    # ------------------------------------------------------------------

    def _resolve_dtype(self, dtype_str: str) -> torch.dtype:
        """把字串 dtype 轉成 torch.dtype"""
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(dtype_str, torch.float16)

    # ------------------------------------------------------------------
    # 模型載入（含量化降級鏈）
    # ------------------------------------------------------------------

    @property
    def pipeline(self):
        """延遲載入 pipeline"""
        if self._pipeline is None:
            self._load_model()
        return self._pipeline

    def _load_model(self):
        """載入 HF Whisper 模型 + 建立 ASR pipeline"""
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )

        print(f"[Model] {self.model_id}")
        print(f"[Device] {self.device}")

        dtype = getattr(self.config, "dtype", "float16")
        quant_mode = getattr(self.config, "quant_mode", "none")
        print(f"[dtype] {dtype}, quant={quant_mode}")

        # 載入 processor
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.download_root,
        )

        # 嘗試量化載入，失敗就降級
        model = self._try_load_quantized(quant_mode, dtype)

        if model is None:
            # 量化失敗，降級為 float16 + device_map auto
            print("[Fallback] 量化載入失敗，嘗試 float16 + device_map=auto")
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=self.download_root,
                )
            except Exception as e:
                # 最後手段：純 CPU float32
                print(f"[Fallback] GPU 載入失敗 ({e})，降級為 CPU")
                self.device = "cpu"
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    cache_dir=self.download_root,
                )

        # 如果沒有用 device_map，手動搬到指定裝置
        if not hasattr(model, "hf_device_map"):
            if self.device == "cuda":
                model = model.to("cuda")

        # 建立 ASR pipeline
        chunk_length_s = getattr(self.config, "chunk_length_s", 30)
        pipe_kwargs = {
            "model": model,
            "tokenizer": self._processor.tokenizer,
            "feature_extractor": self._processor.feature_extractor,
            "torch_dtype": self._resolve_dtype(dtype) if self.device != "cpu" else torch.float32,
        }

        # chunk_length_s > 0 表示啟用分段處理（低 VRAM 情境）
        if chunk_length_s > 0:
            pipe_kwargs["chunk_length_s"] = chunk_length_s
            pipe_kwargs["batch_size"] = 1
            print(f"[Pipeline] chunk_length_s={chunk_length_s}")
        else:
            print("[Pipeline] 整段處理模式")

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            **pipe_kwargs,
        )

        # 顯示 VRAM 使用量
        if self.device == "cuda":
            torch.cuda.synchronize()
            used_mb = torch.cuda.memory_allocated() // (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved() // (1024 * 1024)
            print(f"[VRAM] allocated={used_mb}MB, reserved={reserved_mb}MB")

        print("[Ready]")

    def _try_load_quantized(self, quant_mode: str, dtype: str):
        """
        嘗試用 bitsandbytes 量化載入模型

        降級鏈: bnb_4bit -> bnb_8bit -> None（由呼叫者處理）
        """
        if quant_mode == "none":
            # 不量化，直接載入
            from transformers import AutoModelForSpeechSeq2Seq
            torch_dtype = self._resolve_dtype(dtype)
            return AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                cache_dir=self.download_root,
            )

        # 準備量化嘗試順序
        attempts = []
        if quant_mode == "bnb_4bit":
            attempts = ["bnb_4bit", "bnb_8bit"]
        elif quant_mode == "bnb_8bit":
            attempts = ["bnb_8bit"]

        for mode in attempts:
            model = self._load_with_bnb(mode)
            if model is not None:
                return model

        return None

    def _load_with_bnb(self, mode: str):
        """嘗試用指定的 bitsandbytes 模式載入"""
        try:
            from transformers import AutoModelForSpeechSeq2Seq, BitsAndBytesConfig

            if mode == "bnb_4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                print("[Quantize] 嘗試 NF4 4-bit 量化...")
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                print("[Quantize] 嘗試 8-bit 量化...")

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=self.download_root,
            )
            print(f"[Quantize] {mode} 載入成功")
            return model

        except (ImportError, RuntimeError, ValueError) as e:
            print(f"[Quantize] {mode} 失敗: {e}")
            return None

    # ------------------------------------------------------------------
    # 音檔載入
    # ------------------------------------------------------------------

    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        載入音檔並轉換為 16kHz mono numpy array

        優先用 torchaudio，若格式不支援（如 m4a）則用 librosa。

        Returns:
            (audio_array, sample_rate) -- sample_rate 固定 16000
        """
        # 先嘗試 torchaudio（wav/flac 等格式較快）
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(audio_path))

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            audio_np = waveform.squeeze(0).numpy()
            return audio_np, 16000

        except Exception:
            pass

        # torchaudio 失敗就用 librosa（支援 m4a 等格式）
        import librosa
        audio_np, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        return audio_np, 16000

    # ------------------------------------------------------------------
    # Silero-VAD 整合
    # ------------------------------------------------------------------

    def _run_vad(self, audio_array: np.ndarray, sample_rate: int) -> Optional[List[Tuple[float, float]]]:
        """
        用 Silero-VAD 偵測語音區段

        Args:
            audio_array: 16kHz mono numpy array
            sample_rate: 取樣率（應該是 16000）

        Returns:
            語音區段列表 [(start_sec, end_sec), ...] 或 None（VAD 不可用時）
        """
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
        except ImportError:
            print("[VAD] silero-vad 套件未安裝，跳過 VAD")
            return None

        if self._vad_model is None:
            self._vad_model = load_silero_vad()
            print("[VAD] Silero-VAD 模型已載入")

        # 轉成 torch tensor 給 VAD
        audio_tensor = torch.from_numpy(audio_array).float()

        timestamps = get_speech_timestamps(
            audio_tensor,
            self._vad_model,
            sampling_rate=sample_rate,
            threshold=self.config.vad_threshold,
            min_speech_duration_ms=self.config.vad_min_speech_ms,
            min_silence_duration_ms=self.config.vad_min_silence_ms,
            speech_pad_ms=self.config.vad_speech_pad_ms,
            max_speech_duration_s=self.config.vad_max_speech_duration,
        )

        if not timestamps:
            print("[VAD] 沒有偵測到語音")
            return []

        # 轉成秒數
        segments = []
        for ts in timestamps:
            start_sec = ts["start"] / sample_rate
            end_sec = ts["end"] / sample_rate
            segments.append((start_sec, end_sec))

        print(f"[VAD] 偵測到 {len(segments)} 個語音區段")
        return segments

    # ------------------------------------------------------------------
    # 音頻前處理
    # ------------------------------------------------------------------

    def _preprocess_audio(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        """ASR 前處理：降噪 + 壓縮 + 中頻增強 + 響度標準化

        延遲載入 AudioEnhancer，用保守的 ASR 前處理參數：
        - 降噪強度 0.5（避免過度消除語音細節）
        - mid_boost=2.0（增強人聲頻段 250-4000Hz）
        - 不做低頻/高頻增強（ASR 不需要）

        Args:
            audio_array: 16kHz mono numpy array
            sr: 取樣率

        Returns:
            增強後的 numpy array
        """
        if self._enhancer is None:
            from .enhancer import AudioEnhancer
            from .config import EnhanceConfig

            self._enhancer = AudioEnhancer(EnhanceConfig(
                sample_rate=sr,
                use_gpu=True if self.device == "cuda" else False,
                enable_denoise=True,
                denoise_strength=0.5,
                enable_compression=True,
                enable_eq=True,
                bass_boost=0.0,
                mid_boost=2.0,
                treble_boost=0.0,
            ))

        enhanced, stats = self._enhancer.process_array(audio_array, sr)
        gain = stats.get("gain", 0.0)
        print(f"[Enhance] gain={gain:+.1f} dB")
        return enhanced

    # ------------------------------------------------------------------
    # 轉錄
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: str = "zh",
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
    ) -> Tuple[List[TranscriptSegment], Dict]:
        """
        轉錄音檔

        Args:
            audio_path: 音檔路徑
            language: 語言代碼 ("zh", "en")
            task: "transcribe" 或 "translate"
            initial_prompt: 初始提示詞

        Returns:
            (segments, stats)
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"找不到音檔: {audio_path}")

        print(f"\n[Transcribe] {audio_path.name}")

        # 載入音檔
        audio_array, sr = self._load_audio(audio_path)
        duration = len(audio_array) / sr
        print(f"[Audio] {duration:.1f}s, {sr}Hz")

        # 組裝 generate_kwargs 給 pipeline
        generate_kwargs = {
            "language": language,
            "task": task,
            "num_beams": self.config.beam_size,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "logprob_threshold": self.config.log_prob_threshold,
            "no_speech_threshold": self.config.no_speech_threshold,
            "condition_on_prev_tokens": self.config.condition_on_previous_text,
            "temperature": 0.0,  # greedy decoding (beam search 時必須為 0)
        }

        # initial_prompt 透過 prompt_ids 傳入
        if initial_prompt and self._processor is not None:
            prompt_ids = self._processor.tokenizer.encode(
                initial_prompt, add_special_tokens=False
            )
            # Whisper prompt 上限 224 tokens
            if len(prompt_ids) > 224:
                prompt_ids = prompt_ids[:224]
            generate_kwargs["prompt_ids"] = torch.tensor([prompt_ids], dtype=torch.long)

        # 音頻前處理（降噪 + 壓縮 + 人聲增強），在 VAD 之前做
        if getattr(self.config, "enhance_audio", False):
            print("[Enhance] 執行音頻前處理...")
            audio_array = self._preprocess_audio(audio_array, sr)
            print("[Enhance] 前處理完成")

        # VAD 前處理
        vad_segments = None
        if self.config.vad_filter:
            vad_segments = self._run_vad(audio_array, sr)

        # 根據 VAD 結果選擇轉錄策略
        if vad_segments is not None and len(vad_segments) > 0:
            segments = self._transcribe_with_vad(audio_array, sr, vad_segments, generate_kwargs)
        else:
            segments = self._transcribe_full(audio_array, generate_kwargs)

        # 統計
        total_chars = sum(len(seg.text) for seg in segments)
        stats = {
            "duration": duration,
            "language": language,
            "language_probability": 0.0,  # HF pipeline 不直接回傳此值
            "segments": len(segments),
            "total_chars": total_chars,
        }

        print(f"[Done] {len(segments)} segments, {total_chars} chars")
        self._cleanup_memory()

        return segments, stats

    def _transcribe_full(
        self, audio_array: np.ndarray, generate_kwargs: Dict
    ) -> List[TranscriptSegment]:
        """整段音檔丟給 pipeline 處理"""
        pipe = self.pipeline

        result = pipe(
            audio_array.copy(),
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )

        return self._parse_pipeline_result(result)

    def _transcribe_with_vad(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        vad_segments: List[Tuple[float, float]],
        generate_kwargs: Dict,
    ) -> List[TranscriptSegment]:
        """把 VAD 區段逐段送進 pipeline，偏移時間戳"""
        pipe = self.pipeline
        all_segments = []

        for i, (start_sec, end_sec) in enumerate(vad_segments):
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            chunk = audio_array[start_sample:end_sample]

            if len(chunk) < 1600:  # 太短就跳過（<0.1s）
                continue

            result = pipe(
                chunk.copy(),
                return_timestamps=True,
                generate_kwargs=generate_kwargs,
            )

            # 解析結果並偏移時間戳
            for seg in self._parse_pipeline_result(result):
                seg.start += start_sec
                seg.end += start_sec
                all_segments.append(seg)

        return all_segments

    def _parse_pipeline_result(self, result: Dict) -> List[TranscriptSegment]:
        """把 HF pipeline 的輸出轉成 TranscriptSegment 列表"""
        segments = []

        # pipeline 回傳格式: {"text": "...", "chunks": [{"text": ..., "timestamp": (start, end)}]}
        chunks = result.get("chunks", [])

        if not chunks:
            # 沒有 chunks（可能 return_timestamps 沒生效），用整段 text
            text = result.get("text", "").strip()
            if text:
                segments.append(TranscriptSegment(start=0.0, end=0.0, text=text))
            return segments

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            ts = chunk.get("timestamp", (0.0, 0.0))
            start = ts[0] if ts[0] is not None else 0.0
            end = ts[1] if ts[1] is not None else start
            segments.append(TranscriptSegment(start=start, end=end, text=text))

        return segments

    def _cleanup_memory(self):
        """清理記憶體"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def transcribe_to_text(self, audio_path: Union[str, Path], **kwargs) -> str:
        """轉錄並返回純文字"""
        segments, _ = self.transcribe(audio_path, **kwargs)
        return "\n".join(seg.text for seg in segments)

    def transcribe_to_timestamps(self, audio_path: Union[str, Path], **kwargs) -> str:
        """轉錄並返回時間戳記格式"""
        segments, _ = self.transcribe(audio_path, **kwargs)
        return "\n".join(seg.to_timestamp_line() for seg in segments)

    def transcribe_to_srt(self, audio_path: Union[str, Path], **kwargs) -> str:
        """轉錄並返回 SRT 字幕格式"""
        segments, _ = self.transcribe(audio_path, **kwargs)
        lines = []
        for i, seg in enumerate(segments, 1):
            lines.append(seg.to_srt_block(i))
        return "\n".join(lines)

    def get_config_summary(self) -> str:
        """取得配置摘要"""
        dtype = getattr(self.config, "dtype", "unknown")
        quant = getattr(self.config, "quant_mode", "none")
        chunk_s = getattr(self.config, "chunk_length_s", 30)

        lines = [
            f"Device: {self.device}",
            f"dtype: {dtype}",
            f"Beam: {self.config.beam_size}",
            f"VAD: {self.config.vad_filter} (threshold={self.config.vad_threshold})",
            f"Chunk: {chunk_s}s",
        ]

        if quant != "none":
            lines.insert(2, f"Quantize: {quant}")

        tier = getattr(self.config, "tier_name", "")
        if tier and tier != "unknown":
            lines.insert(0, f"Tier: {tier}")
        gpu = getattr(self.config, "gpu_name", "")
        if gpu:
            lines.insert(0, f"GPU: {gpu}")

        return "\n".join(lines) + "\n"


# ============================================================================
# 工具函數
# ============================================================================

def save_transcript(
    segments: List[TranscriptSegment],
    output_path: Union[str, Path],
    format: str = "txt",
) -> List[Path]:
    """
    儲存轉錄結果

    Args:
        segments: 轉錄片段列表
        output_path: 輸出路徑（不含副檔名）
        format: 格式 ("txt", "srt", "both")

    Returns:
        輸出檔案路徑列表
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_files = []

    if format in ("txt", "both"):
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(seg.to_timestamp_line() + "\n")
        print(f"[Saved] {txt_path}")
        saved_files.append(txt_path)

    if format in ("srt", "both"):
        srt_path = output_path.with_suffix(".srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                f.write(seg.to_srt_block(i) + "\n")
        print(f"[Saved] {srt_path}")
        saved_files.append(srt_path)

    return saved_files


def create_transcriber(
    mode: str = "speed",
    model_id: Optional[str] = None,
) -> BreezeTranscriber:
    """
    建立轉錄器的便捷函數

    Args:
        mode: "speed" 速度優先, "quality" 品質優先, "auto" 自動
        model_id: 模型 ID

    Returns:
        BreezeTranscriber 實例
    """
    if mode == "speed":
        config = RTX3050Config()
    elif mode == "quality":
        config = QualityConfig()
    else:
        config = None  # 自動偵測

    return BreezeTranscriber(model_id=model_id, config=config)
