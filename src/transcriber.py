"""
Breeze-ASR-25 語音轉文字轉錄器

針對 RTX 3050 Laptop 4GB VRAM 深度優化版本
"""

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch


# ============================================================================
# RTX 3050 4GB 專用優化配置
# ============================================================================

@dataclass
class RTX3050Config:
    """
    RTX 3050 Laptop 4GB VRAM 專屬優化配置

    基於 faster-whisper + CTranslate2 的最佳實踐
    參考: https://github.com/SYSTRAN/faster-whisper
    """

    # --- 計算精度 ---
    # int8_float16: 權重 INT8 + 計算 FP16，VRAM ~2.9GB（最佳平衡）
    compute_type: str = "int8_float16"

    # --- Beam Search ---
    # beam_size=1: 貪婪解碼，最快但略降品質
    # beam_size=3: 平衡速度與品質（推薦）
    # beam_size=5: 預設，最佳品質但較慢
    beam_size: int = 1  # 4GB VRAM 建議用 1，速度優先

    # --- VAD 語音活動偵測 ---
    # 過濾靜音段落，大幅提升速度
    vad_filter: bool = True
    vad_threshold: float = 0.5           # 語音偵測閾值 (0-1)
    vad_min_speech_ms: int = 250         # 最短語音段落 (ms)
    vad_min_silence_ms: int = 100        # 觸發分段的最短靜音 (ms)
    vad_speech_pad_ms: int = 30          # 語音前後填充 (ms)
    vad_max_speech_duration: float = 30  # 單段最長秒數（避免 OOM）

    # --- 記憶體管理 ---
    # 長音檔分段處理，避免 VRAM 爆掉
    chunk_length: int = 30       # 每次處理秒數
    cpu_threads: int = 4         # CPU 執行緒數
    num_workers: int = 1         # GPU worker 數（4GB 只用 1 個）

    # --- 速度優化 ---
    condition_on_previous_text: bool = False  # 關閉上下文條件，加速
    compression_ratio_threshold: float = 2.4  # 壓縮比閾值
    log_prob_threshold: float = -1.0          # log 機率閾值
    no_speech_threshold: float = 0.6          # 無語音閾值

    # --- 輸出控制 ---
    word_timestamps: bool = False  # 詞級時間戳（開啟會變慢）
    without_timestamps: bool = False  # 完全不要時間戳


@dataclass
class QualityConfig:
    """品質優先配置（較慢但更準確）"""
    compute_type: str = "int8_float16"
    beam_size: int = 3
    vad_filter: bool = True
    vad_threshold: float = 0.4
    vad_min_speech_ms: int = 200
    vad_min_silence_ms: int = 150
    vad_speech_pad_ms: int = 50
    vad_max_speech_duration: float = 30
    chunk_length: int = 30
    cpu_threads: int = 4
    num_workers: int = 1
    condition_on_previous_text: bool = True  # 開啟提升準確度
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    word_timestamps: bool = False
    without_timestamps: bool = False


@dataclass
class GPUConfig:
    """
    自適應 GPU 配置

    由 _auto_config() 根據偵測到的 VRAM 和 compute capability 自動產生，
    也可手動建立覆蓋特定參數。
    """

    # --- 計算精度 ---
    compute_type: str = "int8_float16"

    # --- Beam Search ---
    beam_size: int = 1

    # --- VAD 語音活動偵測 ---
    vad_filter: bool = True
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 100
    vad_speech_pad_ms: int = 30
    vad_max_speech_duration: float = 30

    # --- 記憶體管理 ---
    chunk_length: int = 30
    cpu_threads: int = 4
    num_workers: int = 1

    # --- 速度優化 ---
    condition_on_previous_text: bool = False
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

    # --- 輸出控制 ---
    word_timestamps: bool = False
    without_timestamps: bool = False

    # --- 批次推論 ---
    batch_size: int = 0  # 0 = 不使用批次，>0 = BatchedInferencePipeline

    # --- GPU 偵測資訊（唯讀紀錄） ---
    gpu_name: str = ""
    vram_mb: int = 0
    compute_capability: float = 0.0
    tier_name: str = "unknown"


# ============================================================================
# VRAM 分級表
# 各層對應 (tier_name, vram_upper_mb, compute_type, beam, batch, cond_prev, workers)
# ============================================================================

_VRAM_TIERS = [
    # tier_name,  vram_upper,  compute_type,     beam, batch, cond_prev, workers
    ("low",       4096,        "int8_float16",   1,    0,     False,     1),
    ("mid_low",   6144,        "int8_float16",   3,    0,     False,     1),
    ("mid",       8192,        "int8_float16",   5,    0,     True,      1),
    ("mid_high",  11264,       "float16",        5,    0,     True,      1),
    ("high",      24576,       "float16",        5,    12,    True,      2),
    ("very_high", 49152,       "float16",        5,    24,    True,      4),
]

# 超過 49152 MB 的 GPU 用這組
_ULTRA_TIER = ("ultra", 0, "float16", 5, 32, True, 4)


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

    針對 RTX 3050 Laptop 4GB VRAM 深度優化
    """

    # 預轉換的 faster-whisper 模型
    MODEL_ID = "SoybeanMilk/faster-whisper-Breeze-ASR-25"

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
            model_id: 模型 ID，預設使用 Breeze-ASR-25
            device: 裝置 ("auto", "cuda", "cpu")
            config: 優化配置，預設自動偵測硬體 (None = 自動)
            download_root: 模型下載目錄
        """
        self.model_id = model_id or self.MODEL_ID
        self.download_root = download_root

        # 設定裝置
        self.device = self._detect_device(device)

        # 自動或手動配置
        if config is None:
            self.config = self._auto_config()
        else:
            self.config = config

        # 套用 CUDA 優化
        if self.device == "cuda":
            self._setup_cuda_optimizations()

        # 延遲載入模型
        self._model = None

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
            print("[CPU] 使用 INT8 量化")
            return GPUConfig(compute_type="int8", tier_name="cpu")

        props = torch.cuda.get_device_properties(0)
        vram_mb = props.total_memory // (1024 * 1024)
        gpu_name = torch.cuda.get_device_name(0)
        cc = props.major + props.minor / 10  # 例如 8.6, 8.9, 9.0

        print(f"[GPU] {gpu_name}")
        print(f"[VRAM] {vram_mb} MB")
        print(f"[CC] {props.major}.{props.minor}")

        # 查表：找到 VRAM 所屬的層級
        matched = None
        for tier in _VRAM_TIERS:
            if vram_mb <= tier[1]:
                matched = tier
                break

        # 超過所有層級就用 ultra
        if matched is None:
            matched = _ULTRA_TIER

        tier_name, _, compute_type, beam, batch, cond_prev, workers = matched

        # Ampere+ (CC >= 8.0) 且 high 以上層級：升級到 bfloat16
        if cc >= 8.0 and compute_type == "float16" and tier_name in ("high", "very_high", "ultra"):
            compute_type = "bfloat16"

        cfg = GPUConfig(
            compute_type=compute_type,
            beam_size=beam,
            batch_size=batch,
            condition_on_previous_text=cond_prev,
            num_workers=workers,
            gpu_name=gpu_name,
            vram_mb=vram_mb,
            compute_capability=cc,
            tier_name=tier_name,
        )

        batch_label = f", batch={batch}" if batch > 0 else ""
        print(f"[Profile] {tier_name} - {compute_type}, beam={beam}{batch_label}")

        return cfg

    def _setup_cuda_optimizations(self):
        """設定 CUDA 優化"""
        # cuDNN benchmark：自動選擇最快的卷積演算法
        torch.backends.cudnn.benchmark = True

        # 允許 TF32（Ampere 架構加速）
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

        # 記憶體分配策略
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        print("[CUDA] 優化設定完成")

    @property
    def model(self):
        """延遲載入模型"""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """載入 faster-whisper 模型（含優化參數）"""
        from faster_whisper import WhisperModel

        print(f"[Model] {self.model_id}")
        print(f"[Device] {self.device}")
        print(f"[Compute] {self.config.compute_type}")
        print(f"[Beam] {self.config.beam_size}")

        self._model = WhisperModel(
            self.model_id,
            device=self.device,
            compute_type=self.config.compute_type,
            download_root=self.download_root,
            cpu_threads=self.config.cpu_threads,
            num_workers=self.config.num_workers,
        )

        # 顯示 VRAM 使用量
        if self.device == "cuda":
            torch.cuda.synchronize()
            used_mb = torch.cuda.memory_allocated() // (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved() // (1024 * 1024)
            print(f"[VRAM] allocated={used_mb}MB, reserved={reserved_mb}MB")

        print("[Ready]")

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: str = "zh",
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
    ) -> Tuple[List[TranscriptSegment], Dict]:
        """
        轉錄音檔（使用優化參數）

        Args:
            audio_path: 音檔路徑
            language: 語言代碼 ("zh" 中文, "en" 英文)
            task: 任務類型 ("transcribe" 轉錄, "translate" 翻譯)
            initial_prompt: 初始提示詞（可提升特定領域準確度）

        Returns:
            (segments, info) - 轉錄片段列表和統計資訊
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"找不到音檔: {audio_path}")

        print(f"\n[Transcribe] {audio_path.name}")

        # 建構 VAD 參數
        vad_params = None
        if self.config.vad_filter:
            vad_params = {
                "threshold": self.config.vad_threshold,
                "min_speech_duration_ms": self.config.vad_min_speech_ms,
                "min_silence_duration_ms": self.config.vad_min_silence_ms,
                "speech_pad_ms": self.config.vad_speech_pad_ms,
                "max_speech_duration_s": self.config.vad_max_speech_duration,
            }

        # 執行轉錄（使用所有優化參數）
        segments_raw, info = self.model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            vad_parameters=vad_params,
            word_timestamps=self.config.word_timestamps,
            condition_on_previous_text=self.config.condition_on_previous_text,
            compression_ratio_threshold=self.config.compression_ratio_threshold,
            log_prob_threshold=self.config.log_prob_threshold,
            no_speech_threshold=self.config.no_speech_threshold,
            initial_prompt=initial_prompt,
            without_timestamps=self.config.without_timestamps,
        )

        # 轉換為 TranscriptSegment
        segments = []
        total_chars = 0

        for seg in segments_raw:
            text = seg.text.strip()
            if text:  # 跳過空白段落
                segment = TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=text,
                )
                segments.append(segment)
                total_chars += len(text)

        # 統計資訊
        duration = info.duration if hasattr(info, "duration") else 0
        stats = {
            "duration": duration,
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": len(segments),
            "total_chars": total_chars,
        }

        print(f"[Done] {len(segments)} segments, {total_chars} chars")

        # 清理 GPU 記憶體
        self._cleanup_memory()

        return segments, stats

    def _cleanup_memory(self):
        """清理記憶體"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

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
        lines = [
            f"Device: {self.device}",
            f"Compute: {self.config.compute_type}",
            f"Beam: {self.config.beam_size}",
            f"VAD: {self.config.vad_filter} (threshold={self.config.vad_threshold})",
            f"Chunk: {self.config.chunk_length}s",
        ]

        # GPUConfig 才有偵測資訊
        tier = getattr(self.config, "tier_name", "")
        if tier and tier != "unknown":
            lines.insert(0, f"Tier: {tier}")
        gpu = getattr(self.config, "gpu_name", "")
        if gpu:
            lines.insert(0, f"GPU: {gpu}")
        batch = getattr(self.config, "batch_size", 0)
        if batch > 0:
            lines.append(f"Batch: {batch}")

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
