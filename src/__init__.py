"""
Audio Enhancer GPU
==================

GPU-accelerated audio enhancement tool for amplifying quiet recordings.

Example:
    from src.enhancer import AudioEnhancer
    from src.config import EnhanceConfig
    
    config = EnhanceConfig(target_loudness=-14.0)
    enhancer = AudioEnhancer(config)
    enhancer.process("input.wav", "output.wav")
"""

from .config import EnhanceConfig
from .enhancer import AudioEnhancer

__version__ = "1.0.0"
__all__ = ["EnhanceConfig", "AudioEnhancer"]
