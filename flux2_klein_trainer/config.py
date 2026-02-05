"""Training configuration for FLUX.2-klein-4B LoRA."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    pretrained_model_name: str = "black-forest-labs/FLUX.2-klein-4B"
    dtype: str = "bfloat16"  # FLUX.2-klein works best with bfloat16
    enable_cpu_offload: bool = False  # Set True for <16GB VRAM
    
    
@dataclass
class LoRAConfig:
    """LoRA configuration."""
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_rslora: bool = True  # Rank-stabilized LoRA for better training


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    data_dir: str = "./training_data"
    caption_ext: str = "txt"
    resolution: int = 512  # FLUX.2-klein can handle 512-1024
    center_crop: bool = False
    random_flip: bool = True
    
    
@dataclass
class TrainingConfig:
    """Main training configuration."""
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Dataset
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Training
    output_dir: str = "./output/flux2-klein-lora"
    num_train_steps: int = 2000
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine_with_restarts"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw_8bit"  # 8-bit Adam for memory efficiency
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    
    # Logging & Saving
    save_every: int = 500
    sample_every: int = 500
    log_every: int = 10
    
    # Sampling
    sample_prompts: List[str] = field(default_factory=lambda: [
        "pixel art sprite, a brave knight, game asset",
        "pixel art sprite, a wizard with staff, game asset",
    ])
    sample_steps: int = 4  # FLUX.2-klein uses 4 steps
    sample_guidance_scale: float = 1.0  # FLUX.2-klein uses cfg=1.0
    
    # HuggingFace Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private: bool = False
    
    # Trigger word
    trigger_word: str = "pixel art sprite"
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        from omegaconf import OmegaConf
        
        cfg = OmegaConf.load(path)
        return cls(**OmegaConf.to_container(cfg, resolve=True))
    
    def to_yaml(self, path: str):
        """Save config to YAML file."""
        from omegaconf import OmegaConf
        
        cfg = OmegaConf.structured(self)
        OmegaConf.save(cfg, path)
