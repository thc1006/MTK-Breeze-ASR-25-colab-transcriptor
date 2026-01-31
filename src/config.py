"""
Configuration module for Audio Enhancer.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import yaml


@dataclass
class EnhanceConfig:
    """Audio enhancement configuration.
    
    Attributes:
        target_loudness: Target loudness in LUFS (default: -14.0)
        peak_ceiling: Peak ceiling in dB (default: -1.0)
        enable_compression: Enable dynamic range compression
        compression_threshold: Compression threshold in dB
        compression_ratio: Compression ratio (e.g., 4.0 = 4:1)
        compression_attack: Attack time in ms
        compression_release: Release time in ms
        enable_denoise: Enable noise reduction
        denoise_strength: Noise reduction strength (0-1)
        denoise_stationary: Use stationary noise reduction
        enable_eq: Enable EQ
        bass_boost: Bass boost in dB (< 250 Hz)
        mid_boost: Mid boost in dB (250-4000 Hz)
        treble_boost: Treble boost in dB (> 4000 Hz)
        output_format: Output format (wav, mp3, flac)
        sample_rate: Output sample rate
        use_gpu: Use GPU acceleration ('auto', True, False)
        chunk_mode: Process in chunks (for long files)
        chunk_size: Chunk size in seconds
    """
    
    # Loudness
    target_loudness: float = -14.0
    peak_ceiling: float = -1.0
    
    # Compression
    enable_compression: bool = True
    compression_threshold: float = -20.0
    compression_ratio: float = 4.0
    compression_attack: float = 5.0
    compression_release: float = 50.0
    
    # Denoise
    enable_denoise: bool = True
    denoise_strength: float = 0.75
    denoise_stationary: bool = False
    
    # EQ
    enable_eq: bool = True
    bass_boost: float = 2.0
    mid_boost: float = 2.0
    treble_boost: float = 1.0
    
    # Output
    output_format: str = "wav"
    sample_rate: int = 44100
    
    # Processing
    use_gpu: Union[str, bool] = "auto"
    chunk_mode: bool = False
    chunk_size: int = 300
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EnhanceConfig":
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            EnhanceConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        return cls(
            # Loudness
            target_loudness=data.get("loudness", {}).get("target", -14.0),
            peak_ceiling=data.get("loudness", {}).get("peak_ceiling", -1.0),
            # Compression
            enable_compression=data.get("compression", {}).get("enabled", True),
            compression_threshold=data.get("compression", {}).get("threshold", -20.0),
            compression_ratio=data.get("compression", {}).get("ratio", 4.0),
            compression_attack=data.get("compression", {}).get("attack", 5.0),
            compression_release=data.get("compression", {}).get("release", 50.0),
            # Denoise
            enable_denoise=data.get("denoise", {}).get("enabled", True),
            denoise_strength=data.get("denoise", {}).get("strength", 0.75),
            denoise_stationary=data.get("denoise", {}).get("stationary", False),
            # EQ
            enable_eq=data.get("eq", {}).get("enabled", True),
            bass_boost=data.get("eq", {}).get("bass", 2.0),
            mid_boost=data.get("eq", {}).get("mid", 2.0),
            treble_boost=data.get("eq", {}).get("treble", 1.0),
            # Output
            output_format=data.get("output", {}).get("format", "wav"),
            sample_rate=data.get("output", {}).get("sample_rate", 44100),
            # Processing
            use_gpu=data.get("processing", {}).get("use_gpu", "auto"),
            chunk_mode=data.get("processing", {}).get("chunk_mode", False),
            chunk_size=data.get("processing", {}).get("chunk_size", 300),
        )
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        data = {
            "loudness": {
                "target": self.target_loudness,
                "peak_ceiling": self.peak_ceiling,
            },
            "compression": {
                "enabled": self.enable_compression,
                "threshold": self.compression_threshold,
                "ratio": self.compression_ratio,
                "attack": self.compression_attack,
                "release": self.compression_release,
            },
            "denoise": {
                "enabled": self.enable_denoise,
                "strength": self.denoise_strength,
                "stationary": self.denoise_stationary,
            },
            "eq": {
                "enabled": self.enable_eq,
                "bass": self.bass_boost,
                "mid": self.mid_boost,
                "treble": self.treble_boost,
            },
            "output": {
                "format": self.output_format,
                "sample_rate": self.sample_rate,
            },
            "processing": {
                "use_gpu": self.use_gpu,
                "chunk_mode": self.chunk_mode,
                "chunk_size": self.chunk_size,
            },
        }
        
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"EnhanceConfig(\n"
            f"  loudness={self.target_loudness} LUFS,\n"
            f"  compression={'ON' if self.enable_compression else 'OFF'} "
            f"(threshold={self.compression_threshold}dB, ratio={self.compression_ratio}:1),\n"
            f"  denoise={'ON' if self.enable_denoise else 'OFF'} "
            f"(strength={self.denoise_strength}),\n"
            f"  eq={'ON' if self.enable_eq else 'OFF'} "
            f"(bass={self.bass_boost:+.1f}, mid={self.mid_boost:+.1f}, treble={self.treble_boost:+.1f} dB),\n"
            f"  output={self.output_format} @ {self.sample_rate}Hz\n"
            f")"
        )
