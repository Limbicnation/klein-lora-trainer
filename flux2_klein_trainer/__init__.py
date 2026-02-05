"""FLUX.2-klein-4B LoRA Trainer

A dedicated, memory-efficient trainer for fine-tuning FLUX.2-klein-4B with LoRA.
Optimized for consumer GPUs (RTX 3090/4070+ with 13GB+ VRAM).
"""

__version__ = "0.1.0"
__author__ = "Limbicnation"

from .trainer import KleinLoRATrainer
from .config import TrainingConfig

__all__ = ["KleinLoRATrainer", "TrainingConfig"]
