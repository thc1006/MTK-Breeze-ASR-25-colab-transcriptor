#!/usr/bin/env python3
"""
Breeze-ASR-25 語音轉文字 CLI

自適應 GPU 優化：自動偵測 VRAM / compute capability，選擇最佳配置

Usage:
    python transcribe_cli.py audio.m4a
    python transcribe_cli.py audio.wav -o output.txt
    python transcribe_cli.py audio.mp3 --srt
    python transcribe_cli.py audio.m4a --mode auto
"""

import argparse
import sys
import time
from pathlib import Path

from src.transcriber import (
    BreezeTranscriber,
    GPUConfig,
    RTX3050Config,
    QualityConfig,
    save_transcript,
)


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="Breeze-ASR-25 Taiwan Mandarin Speech-to-Text (RTX 3050 Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.m4a                    # 速度優先模式（預設）
  %(prog)s audio.m4a --mode quality     # 品質優先模式
  %(prog)s audio.wav -o result.txt      # 指定輸出檔案
  %(prog)s audio.mp3 --srt              # 同時輸出 SRT 字幕
  %(prog)s audio.m4a --lang en          # 英文轉錄
  %(prog)s audio.m4a --cpu              # 強制使用 CPU

Optimization Modes:
  speed   - beam=1, VAD aggressive (RTX 3050 4GB 專用)
  quality - beam=3, VAD balanced (較慢但更準確)
  auto    - 根據 GPU VRAM 自動選擇
        """
    )

    # 輸入輸出
    parser.add_argument(
        "input",
        type=str,
        help="輸入音檔路徑"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="輸出檔案路徑 (預設: {input}_transcript.txt)"
    )

    # 輸出格式
    parser.add_argument(
        "--srt",
        action="store_true",
        help="同時輸出 SRT 字幕檔"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="僅輸出純文字 (不含時間戳記)"
    )

    # 語言設定
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default="zh",
        help="語言代碼 (zh=中文, en=英文, 預設: zh)"
    )

    # 效能設定
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["speed", "quality", "auto"],
        default="speed",
        help="優化模式: speed=速度優先, quality=品質優先, auto=自動 (預設: speed)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="強制使用 CPU"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam search 大小 (預設: 由 mode 決定)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size for BatchedInferencePipeline (需要足夠 VRAM)"
    )

    # VAD 設定
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="停用 VAD 過濾"
    )

    # 其他
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="安靜模式"
    )

    return parser.parse_args()


def _build_profile_line(config) -> str:
    """從 config 物件組裝一行 profile 描述"""
    gpu = getattr(config, "gpu_name", "")
    tier = getattr(config, "tier_name", "")
    if gpu and tier:
        return f"{gpu} [{tier}]"
    elif isinstance(config, RTX3050Config):
        return "Manual: speed mode (RTX3050Config)"
    elif isinstance(config, QualityConfig):
        return "Manual: quality mode"
    return "Unknown config"


def main():
    """主程式"""
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 找不到檔案: {input_path}", file=sys.stderr)
        sys.exit(1)

    # 決定輸出路徑
    if args.output:
        output_path = Path(args.output).with_suffix("")
    else:
        output_path = input_path.parent / f"{input_path.stem}_transcript"

    # 根據模式選擇配置
    if args.mode == "speed":
        config = RTX3050Config()
    elif args.mode == "quality":
        config = QualityConfig()
    else:
        config = None  # 自動偵測

    # 覆蓋 beam_size（如果有指定）
    if args.beam_size is not None and config is not None:
        config.beam_size = args.beam_size

    # 覆蓋 VAD 設定
    if args.no_vad and config is not None:
        config.vad_filter = False

    # 初始化轉錄器（auto 模式會在這裡偵測 GPU）
    device = "cpu" if args.cpu else "auto"
    transcriber = BreezeTranscriber(
        device=device,
        config=config,
    )

    # --batch 覆蓋：套用到已經建好的 config 上
    if args.batch is not None:
        if hasattr(transcriber.config, "batch_size"):
            transcriber.config.batch_size = args.batch
        else:
            # RTX3050Config / QualityConfig 沒有 batch_size 欄位，
            # 建立一個 GPUConfig 來取代
            old = transcriber.config
            transcriber.config = GPUConfig(
                compute_type=old.compute_type,
                beam_size=old.beam_size,
                vad_filter=old.vad_filter,
                vad_threshold=old.vad_threshold,
                vad_min_speech_ms=old.vad_min_speech_ms,
                vad_min_silence_ms=old.vad_min_silence_ms,
                vad_speech_pad_ms=old.vad_speech_pad_ms,
                vad_max_speech_duration=old.vad_max_speech_duration,
                chunk_length=old.chunk_length,
                cpu_threads=old.cpu_threads,
                num_workers=old.num_workers,
                condition_on_previous_text=old.condition_on_previous_text,
                batch_size=args.batch,
            )

    # 顯示標題（在偵測完 GPU 之後，可以顯示實際 profile）
    if not args.quiet:
        profile_line = _build_profile_line(transcriber.config)
        print("=" * 50)
        print("Breeze-ASR-25 Taiwan Mandarin Transcriber")
        print(profile_line)
        print("=" * 50)
        print(f"[Input] {input_path}")
        print(f"[Language] {args.lang}")
        print(f"[Mode] {args.mode}")
        batch_val = getattr(transcriber.config, "batch_size", 0)
        if batch_val > 0:
            print(f"[Batch] {batch_val}")
        print()

    start_time = time.time()

    # 執行轉錄
    segments, stats = transcriber.transcribe(
        input_path,
        language=args.lang,
    )

    if not segments:
        print("[WARN] 沒有轉錄到任何內容")
        sys.exit(0)

    # 儲存結果
    if args.text_only:
        # 純文字輸出
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(seg.text + "\n")
        print(f"[Saved] {txt_path}")
    else:
        # 時間戳記輸出
        fmt = "both" if args.srt else "txt"
        save_transcript(segments, output_path, format=fmt)

    elapsed = time.time() - start_time

    # 顯示結果摘要
    if not args.quiet:
        print()
        print("=" * 50)
        print("[Result]")
        print(f"  Segments: {stats['segments']}")
        print(f"  Characters: {stats['total_chars']}")
        print(f"  Language: {stats['language']} ({stats['language_probability']:.1%})")
        print(f"  Time: {elapsed:.1f}s")

        # GPU 配置資訊
        tier = getattr(transcriber.config, "tier_name", "")
        if tier and tier not in ("unknown", "cpu"):
            vram = getattr(transcriber.config, "vram_mb", 0)
            cc = getattr(transcriber.config, "compute_capability", 0)
            print(f"  Profile: {tier} (VRAM={vram}MB, CC={cc})")

        # 顯示前幾段預覽
        print()
        print("[Preview]")
        for seg in segments[:5]:
            print(f"  {seg.to_timestamp_line()}")
        if len(segments) > 5:
            print(f"  ... ({len(segments) - 5} more segments)")

        print("=" * 50)


if __name__ == "__main__":
    main()
