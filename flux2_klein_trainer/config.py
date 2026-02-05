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
    target_modules: List[str] = field(default_factory=lambda: ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.v_proj", "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj", "model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.k_proj", "model.layers.1.self_attn.v_proj", "model.layers.1.self_attn.o_proj", "model.layers.1.mlp.gate_proj", "model.layers.1.mlp.up_proj", "model.layers.1.mlp.down_proj", "model.layers.2.self_attn.q_proj", "model.layers.2.self_attn.k_proj", "model.layers.2.self_attn.v_proj", "model.layers.2.self_attn.o_proj", "model.layers.2.mlp.gate_proj", "model.layers.2.mlp.up_proj", "model.layers.2.mlp.down_proj", "model.layers.3.self_attn.q_proj", "model.layers.3.self_attn.k_proj", "model.layers.3.self_attn.v_proj", "model.layers.3.self_attn.o_proj", "model.layers.3.mlp.gate_proj", "model.layers.3.mlp.up_proj", "model.layers.3.mlp.down_proj", "model.layers.4.self_attn.q_proj", "model.layers.4.self_attn.k_proj", "model.layers.4.self_attn.v_proj", "model.layers.4.self_attn.o_proj", "model.layers.4.mlp.gate_proj", "model.layers.4.mlp.up_proj", "model.layers.4.mlp.down_proj", "model.layers.5.self_attn.q_proj", "model.layers.5.self_attn.k_proj", "model.layers.5.self_attn.v_proj", "model.layers.5.self_attn.o_proj", "model.layers.5.mlp.gate_proj", "model.layers.5.mlp.up_proj", "model.layers.5.mlp.down_proj", "model.layers.6.self_attn.q_proj", "model.layers.6.self_attn.k_proj", "model.layers.6.self_attn.v_proj", "model.layers.6.self_attn.o_proj", "model.layers.6.mlp.gate_proj", "model.layers.6.mlp.up_proj", "model.layers.6.mlp.down_proj", "model.layers.7.self_attn.q_proj", "model.layers.7.self_attn.k_proj", "model.layers.7.self_attn.v_proj", "model.layers.7.self_attn.o_proj", "model.layers.7.mlp.gate_proj", "model.layers.7.mlp.up_proj", "model.layers.7.mlp.down_proj", "model.layers.8.self_attn.q_proj", "model.layers.8.self_attn.k_proj", "model.layers.8.self_attn.v_proj", "model.layers.8.self_attn.o_proj", "model.layers.8.mlp.gate_proj", "model.layers.8.mlp.up_proj", "model.layers.8.mlp.down_proj", "model.layers.9.self_attn.q_proj", "model.layers.9.self_attn.k_proj", "model.layers.9.self_attn.v_proj", "model.layers.9.self_attn.o_proj", "model.layers.9.mlp.gate_proj", "model.layers.9.mlp.up_proj", "model.layers.9.mlp.down_proj", "model.layers.10.self_attn.q_proj", "model.layers.10.self_attn.k_proj", "model.layers.10.self_attn.v_proj", "model.layers.10.self_attn.o_proj", "model.layers.10.mlp.gate_proj", "model.layers.10.mlp.up_proj", "model.layers.10.mlp.down_proj", "model.layers.11.self_attn.q_proj", "model.layers.11.self_attn.k_proj", "model.layers.11.self_attn.v_proj", "model.layers.11.self_attn.o_proj", "model.layers.11.mlp.gate_proj", "model.layers.11.mlp.up_proj", "model.layers.11.mlp.down_proj", "model.layers.12.self_attn.q_proj", "model.layers.12.self_attn.k_proj", "model.layers.12.self_attn.v_proj", "model.layers.12.self_attn.o_proj", "model.layers.12.mlp.gate_proj", "model.layers.12.mlp.up_proj", "model.layers.12.mlp.down_proj", "model.layers.13.self_attn.q_proj", "model.layers.13.self_attn.k_proj", "model.layers.13.self_attn.v_proj", "model.layers.13.self_attn.o_proj", "model.layers.13.mlp.gate_proj", "model.layers.13.mlp.up_proj", "model.layers.13.mlp.down_proj", "model.layers.14.self_attn.q_proj", "model.layers.14.self_attn.k_proj", "model.layers.14.self_attn.v_proj", "model.layers.14.self_attn.o_proj", "model.layers.14.mlp.gate_proj", "model.layers.14.mlp.up_proj", "model.layers.14.mlp.down_proj", "model.layers.15.self_attn.q_proj", "model.layers.15.self_attn.k_proj", "model.layers.15.self_attn.v_proj", "model.layers.15.self_attn.o_proj", "model.layers.15.mlp.gate_proj", "model.layers.15.mlp.up_proj", "model.layers.15.mlp.down_proj", "model.layers.16.self_attn.q_proj", "model.layers.16.self_attn.k_proj", "model.layers.16.self_attn.v_proj", "model.layers.16.self_attn.o_proj", "model.layers.16.mlp.gate_proj", "model.layers.16.mlp.up_proj", "model.layers.16.mlp.down_proj", "model.layers.17.self_attn.q_proj", "model.layers.17.self_attn.k_proj", "model.layers.17.self_attn.v_proj", "model.layers.17.self_attn.o_proj", "model.layers.17.mlp.gate_proj", "model.layers.17.mlp.up_proj", "model.layers.17.mlp.down_proj", "model.layers.18.self_attn.q_proj", "model.layers.18.self_attn.k_proj", "model.layers.18.self_attn.v_proj", "model.layers.18.self_attn.o_proj", "model.layers.18.mlp.gate_proj", "model.layers.18.mlp.up_proj", "model.layers.18.mlp.down_proj", "model.layers.19.self_attn.q_proj", "model.layers.19.self_attn.k_proj", "model.layers.19.self_attn.v_proj", "model.layers.19.self_attn.o_proj", "model.layers.19.mlp.gate_proj", "model.layers.19.mlp.up_proj", "model.layers.19.mlp.down_proj", "model.layers.20.self_attn.q_proj", "model.layers.20.self_attn.k_proj", "model.layers.20.self_attn.v_proj", "model.layers.20.self_attn.o_proj", "model.layers.20.mlp.gate_proj", "model.layers.20.mlp.up_proj", "model.layers.20.mlp.down_proj", "model.layers.21.self_attn.q_proj", "model.layers.21.self_attn.k_proj", "model.layers.21.self_attn.v_proj", "model.layers.21.self_attn.o_proj", "model.layers.21.mlp.gate_proj", "model.layers.21.mlp.up_proj", "model.layers.21.mlp.down_proj", "model.layers.22.self_attn.q_proj", "model.layers.22.self_attn.k_proj", "model.layers.22.self_attn.v_proj", "model.layers.22.self_attn.o_proj", "model.layers.22.mlp.gate_proj", "model.layers.22.mlp.up_proj", "model.layers.22.mlp.down_proj", "model.layers.23.self_attn.q_proj", "model.layers.23.self_attn.k_proj", "model.layers.23.self_attn.v_proj", "model.layers.23.self_attn.o_proj", "model.layers.23.mlp.gate_proj", "model.layers.23.mlp.up_proj", "model.layers.23.mlp.down_proj", "model.layers.24.self_attn.q_proj", "model.layers.24.self_attn.k_proj", "model.layers.24.self_attn.v_proj", "model.layers.24.self_attn.o_proj", "model.layers.24.mlp.gate_proj", "model.layers.24.mlp.up_proj", "model.layers.24.mlp.down_proj", "model.layers.25.self_attn.q_proj", "model.layers.25.self_attn.k_proj", "model.layers.25.self_attn.v_proj", "model.layers.25.self_attn.o_proj", "model.layers.25.mlp.gate_proj", "model.layers.25.mlp.up_proj", "model.layers.25.mlp.down_proj", "model.layers.26.self_attn.q_proj", "model.layers.26.self_attn.k_proj", "model.layers.26.self_attn.v_proj", "model.layers.26.self_attn.o_proj", "model.layers.26.mlp.gate_proj", "model.layers.26.mlp.up_proj", "model.layers.26.mlp.down_proj", "model.layers.27.self_attn.q_proj", "model.layers.27.self_attn.k_proj", "model.layers.27.self_attn.v_proj", "model.layers.27.self_attn.o_proj", "model.layers.27.mlp.gate_proj", "model.layers.27.mlp.up_proj", "model.layers.27.mlp.down_proj", "model.layers.28.self_attn.q_proj", "model.layers.28.self_attn.k_proj", "model.layers.28.self_attn.v_proj", "model.layers.28.self_attn.o_proj", "model.layers.28.mlp.gate_proj", "model.layers.28.mlp.up_proj", "model.layers.28.mlp.down_proj", "model.layers.29.self_attn.q_proj", "model.layers.29.self_attn.k_proj", "model.layers.29.self_attn.v_proj", "model.layers.29.self_attn.o_proj", "model.layers.29.mlp.gate_proj", "model.layers.29.mlp.up_proj", "model.layers.29.mlp.down_proj", "model.layers.30.self_attn.q_proj", "model.layers.30.self_attn.k_proj", "model.layers.30.self_attn.v_proj", "model.layers.30.self_attn.o_proj", "model.layers.30.mlp.gate_proj", "model.layers.30.mlp.up_proj", "model.layers.30.mlp.down_proj", "model.layers.31.self_attn.q_proj", "model.layers.31.self_attn.k_proj", "model.layers.31.self_attn.v_proj", "model.layers.31.self_attn.o_proj", "model.layers.31.mlp.gate_proj", "model.layers.31.mlp.up_proj", "model.layers.31.mlp.down_proj", "model.layers.32.self_attn.q_proj", "model.layers.32.self_attn.k_proj", "model.layers.32.self_attn.v_proj", "model.layers.32.self_attn.o_proj", "model.layers.32.mlp.gate_proj", "model.layers.32.mlp.up_proj", "model.layers.32.mlp.down_proj", "model.layers.33.self_attn.q_proj", "model.layers.33.self_attn.k_proj", "model.layers.33.self_attn.v_proj", "model.layers.33.self_attn.o_proj", "model.layers.33.mlp.gate_proj", "model.layers.33.mlp.up_proj", "model.layers.33.mlp.down_proj", "model.layers.34.self_attn.q_proj", "model.layers.34.self_attn.k_proj", "model.layers.34.self_attn.v_proj", "model.layers.34.self_attn.o_proj", "model.layers.34.mlp.gate_proj", "model.layers.34.mlp.up_proj", "model.layers.34.mlp.down_proj", "model.layers.35.self_attn.q_proj", "model.layers.35.self_attn.k_proj", "model.layers.35.self_attn.v_proj", "model.layers.35.self_attn.o_proj", "model.layers.35.mlp.gate_proj", "model.layers.35.mlp.up_proj", "model.layers.35.mlp.down_proj"])
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
