#!/usr/bin/env python3
"""
Command-line interface for Audio Enhancer.

Usage:
    audio-enhance input.wav -o output.wav
    audio-enhance ./input_folder/ -o ./output_folder/
    audio-enhance input.wav --loudness -14 --denoise 0.8
"""

import argparse
import sys
import time
from pathlib import Path

from src.config import EnhanceConfig
from src.enhancer import AudioEnhancer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GPU-accelerated audio enhancement tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.wav -o output.wav
  %(prog)s ./recordings/ -o ./enhanced/
  %(prog)s input.wav -o output.wav --loudness -14 --denoise 0.8
  %(prog)s input.wav -o output.wav --config custom.yaml
        """
    )
    
    # Input/Output
    parser.add_argument(
        "input",
        type=str,
        help="Input audio file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file or directory"
    )
    
    # Config file
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    
    # Loudness
    parser.add_argument(
        "--loudness",
        type=float,
        default=None,
        help="Target loudness in LUFS (default: -14.0)"
    )
    
    # Compression
    parser.add_argument(
        "--compress",
        action="store_true",
        default=None,
        help="Enable compression (default: enabled)"
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable compression"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Compression threshold in dB (default: -20)"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Compression ratio (default: 4.0)"
    )
    
    # Denoise
    parser.add_argument(
        "--denoise",
        type=float,
        default=None,
        help="Denoise strength 0-1 (default: 0.75, 0 to disable)"
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable denoising"
    )
    
    # EQ
    parser.add_argument(
        "--eq",
        action="store_true",
        default=None,
        help="Enable EQ (default: enabled)"
    )
    parser.add_argument(
        "--no-eq",
        action="store_true",
        help="Disable EQ"
    )
    parser.add_argument(
        "--eq-bass",
        type=float,
        default=None,
        help="Bass boost in dB (default: 2.0)"
    )
    parser.add_argument(
        "--eq-mid",
        type=float,
        default=None,
        help="Mid boost in dB (default: 2.0)"
    )
    parser.add_argument(
        "--eq-treble",
        type=float,
        default=None,
        help="Treble boost in dB (default: 1.0)"
    )
    
    # Output format
    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["wav", "mp3", "flac"],
        default=None,
        help="Output format (default: wav)"
    )
    
    # Processing options
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU processing (disable GPU)"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process directories recursively"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    return parser.parse_args()


def build_config(args) -> EnhanceConfig:
    """Build configuration from arguments."""
    # Start with config file or defaults
    if args.config:
        config = EnhanceConfig.from_yaml(args.config)
    else:
        # Try to load default config
        default_config = Path(__file__).parent / "config" / "default.yaml"
        if default_config.exists():
            config = EnhanceConfig.from_yaml(default_config)
        else:
            config = EnhanceConfig()
    
    # Override with command line arguments
    if args.loudness is not None:
        config.target_loudness = args.loudness
    
    if args.no_compress:
        config.enable_compression = False
    elif args.compress:
        config.enable_compression = True
    
    if args.threshold is not None:
        config.compression_threshold = args.threshold
    if args.ratio is not None:
        config.compression_ratio = args.ratio
    
    if args.no_denoise:
        config.enable_denoise = False
    elif args.denoise is not None:
        if args.denoise == 0:
            config.enable_denoise = False
        else:
            config.enable_denoise = True
            config.denoise_strength = args.denoise
    
    if args.no_eq:
        config.enable_eq = False
    elif args.eq:
        config.enable_eq = True
    
    if args.eq_bass is not None:
        config.bass_boost = args.eq_bass
    if args.eq_mid is not None:
        config.mid_boost = args.eq_mid
    if args.eq_treble is not None:
        config.treble_boost = args.eq_treble
    
    if args.format is not None:
        config.output_format = args.format
    
    if args.cpu:
        config.use_gpu = False
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build configuration
    config = build_config(args)
    
    # Print config
    if not args.quiet:
        print("=" * 50)
        print("Audio Enhancer")
        print("=" * 50)
        print(config)
        print()
    
    # Initialize enhancer
    enhancer = AudioEnhancer(config)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    start_time = time.time()
    
    if input_path.is_file():
        # Single file processing
        if not args.quiet:
            print(f"\n[Processing] {input_path.name}")
        
        # If output is a directory, generate filename
        if output_path.is_dir() or not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / f"{input_path.stem}_enhanced.{config.output_format}"
        
        try:
            stats = enhancer.process(input_path, output_path)
            
            if not args.quiet:
                print(f"\n[OK] Success!")
                print(f"   Original: {stats['original_loudness']:.1f} LUFS")
                print(f"   Enhanced: {stats['final_loudness']:.1f} LUFS")
                print(f"   Gain: {stats['gain']:+.1f} dB")
                print(f"\n[Output] {output_path}")
                
        except Exception as e:
            print(f"\n[ERROR] {e}", file=sys.stderr)
            sys.exit(1)
            
    elif input_path.is_dir():
        # Batch processing
        if not args.quiet:
            print(f"\n[Processing directory] {input_path}")
        
        results = enhancer.process_batch(
            input_path,
            output_path,
            recursive=args.recursive,
            show_progress=not args.quiet
        )
        
        # Check for failures
        failures = [r for r in results if r["status"] == "failed"]
        if failures and not args.quiet:
            print(f"\n[WARN] {len(failures)} file(s) failed:")
            for f in failures[:5]:
                print(f"   - {f['file']}: {f.get('error', 'Unknown error')}")
            if len(failures) > 5:
                print(f"   ... and {len(failures) - 5} more")
    else:
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    elapsed = time.time() - start_time
    if not args.quiet:
        print(f"\n[Time] {elapsed:.1f}s")


if __name__ == "__main__":
    main()
