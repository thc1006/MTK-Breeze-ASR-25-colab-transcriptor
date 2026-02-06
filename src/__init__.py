"""
MTK Breeze-ASR-25 Taiwan Mandarin Transcriptor
==============================================

語音轉文字 + 音頻增強工具，針對 RTX 3050 4GB VRAM 深度優化

語音轉文字範例:
    from src.transcriber import BreezeTranscriber

    transcriber = BreezeTranscriber()
    segments, info = transcriber.transcribe("audio.m4a")
    for seg in segments:
        print(seg.to_timestamp_line())

音頻增強範例:
    from src.enhancer import AudioEnhancer
    from src.config import EnhanceConfig

    config = EnhanceConfig(target_loudness=-14.0)
    enhancer = AudioEnhancer(config)
    enhancer.process("input.wav", "output.wav")
"""

# 音頻增強
from .config import EnhanceConfig
from .enhancer import AudioEnhancer

# 語音轉文字
from .transcriber import (
    BreezeTranscriber,
    GPUConfig,
    RTX3050Config,
    QualityConfig,
    TranscriptSegment,
    save_transcript,
    create_transcriber,
)

__version__ = "2.0.0"
__all__ = [
    # 音頻增強
    "EnhanceConfig",
    "AudioEnhancer",
    # 語音轉文字
    "BreezeTranscriber",
    "GPUConfig",
    "RTX3050Config",
    "QualityConfig",
    "TranscriptSegment",
    "save_transcript",
    "create_transcriber",
]
