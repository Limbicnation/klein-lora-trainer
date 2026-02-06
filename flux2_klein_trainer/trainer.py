"""FLUX.2-klein-4B LoRA Trainer implementation."""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
from PIL import Image

from diffusers import Flux2KleinPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, T5EncoderModel, CLIPTextModel
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict
from safetensors.torch import load_file
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder, HfApi

from .config import TrainingConfig
from .dataset import ImageCaptionDataset


class KleinLoRATrainer:
    """Trainer for FLUX.2-klein-4B LoRA fine-tuning."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision="bf16" if config.model.dtype == "bfloat16" else "fp16",
        )
        
        self.device = self.accelerator.device
        self.global_step = 0
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Models will be loaded in setup()
        self.pipe = None
        self.optimizer = None
        self.dataset = None
        self.dataloader = None
        
    def setup(self):
        """Initialize models, optimizer, and dataset."""
        self.accelerator.print("🚀 Initializing FLUX.2-klein-4B LoRA Trainer")
        
        # Load pipeline with bfloat16 (optimal for FLUX.2-klein)
        dtype = torch.bfloat16 if self.config.model.dtype == "bfloat16" else torch.float16
        
        self.accelerator.print(f"📥 Loading FLUX.2-klein-4B with {self.config.model.dtype}...")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.config.model.pretrained_model_name,
            torch_dtype=dtype,
        )
        
        # Enable CPU offloading for low VRAM
        if self.config.model.enable_cpu_offload:
            self.accelerator.print("💾 Enabling CPU offloading for low VRAM")
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)
        
        # Apply LoRA to transformer
        self._apply_lora()
        
        if self.config.resume_from_checkpoint:
            self._resume_from_checkpoint()
        
        # Setup dataset
        self.accelerator.print("📂 Loading dataset...")
        self.dataset = ImageCaptionDataset(
            data_dir=self.config.dataset.data_dir,
            caption_ext=self.config.dataset.caption_ext,
            resolution=self.config.dataset.resolution,
            center_crop=self.config.dataset.center_crop,
            random_flip=self.config.dataset.random_flip,
        )
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        
        # Setup optimizer (8-bit AdamW for memory efficiency)
        self.accelerator.print("⚙️  Setting up optimizer...")
        trainable_params = [p for p in self.pipe.transformer.parameters() if p.requires_grad]
        
        if self.config.optimizer == "adamw_8bit":
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
            )
        
        # Prepare with accelerator
        self.pipe.transformer, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.pipe.transformer, self.optimizer, self.dataloader
        )
        
        self.accelerator.print(f"✅ Setup complete! Dataset size: {len(self.dataset)}")
        
    def _apply_lora(self):
        """Apply LoRA to the transformer."""
        from peft import LoraConfig
        
        lora_config = LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            use_rslora=self.config.lora.use_rslora,
        )
        
        # Add LoRA to transformer
        self.pipe.transformer = get_peft_model(self.pipe.transformer, lora_config)
        self.pipe.transformer.print_trainable_parameters()
        
    def _resume_from_checkpoint(self):
        """Resume training from a checkpoint."""
        checkpoint_path = Path(self.config.resume_from_checkpoint)
        if not checkpoint_path.exists():
            self.accelerator.print(f"⚠️  Checkpoint path does not exist: {checkpoint_path}")
            return
            
        self.accelerator.print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        
        # Load weights into the PeftModel
        weights_path = checkpoint_path / "pytorch_lora_weights.safetensors"
        if not weights_path.exists():
            self.accelerator.print(f"⚠️  Weights file not found: {weights_path}")
            return
            
        state_dict = load_file(weights_path)
        
        # Peft expects the state dict with specific prefixes if saved via save_lora_weights
        # Our save fix ensures correct naming, but the current 8GB file might be different
        # Let's try to load it into the's Peft wrapper
        set_peft_model_state_dict(self.pipe.transformer, state_dict)
        
        # Infer global_step from path name (e.g., 'step_500')
        try:
            name = checkpoint_path.name
            if name.startswith("step_"):
                self.global_step = int(name.replace("step_", ""))
                self.accelerator.print(f"   Detected starting step: {self.global_step}")
        except Exception:
            self.accelerator.print("   Could not detect starting step from path name, starting from 0.")
        
    def train(self):
        """Main training loop."""
        self.setup()
        
        if self.accelerator.is_main_process:
            self.accelerator.print(f"\n🏋️  Starting training for {self.config.num_train_steps} steps")
            progress_bar = tqdm(
                total=self.config.num_train_steps,
                desc="Training",
                disable=not self.accelerator.is_local_main_process,
            )
        
        self.pipe.transformer.train()
        
        while self.global_step < self.config.num_train_steps:
            for batch in self.dataloader:
                with self.accelerator.accumulate(self.pipe.transformer):
                    loss = self._training_step(batch)
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.pipe.transformer.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update progress
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    
                    if self.accelerator.is_main_process:
                        progress_bar.update(1)
                        progress_bar.set_postfix({"loss": loss.item()})
                        
                        # Logging
                        if self.global_step % self.config.log_every == 0:
                            self.accelerator.log({"loss": loss.item()}, step=self.global_step)
                        
                        # Save checkpoint
                        if self.global_step % self.config.save_every == 0:
                            self.save_checkpoint()
                        
                        # Generate samples
                        if self.global_step % self.config.sample_every == 0:
                            self.generate_samples()
                    
                    # Check if training is complete
                    if self.global_step >= self.config.num_train_steps:
                        break
        
        if self.accelerator.is_main_process:
            progress_bar.close()
            self.accelerator.print("\n✅ Training complete!")
            self.save_checkpoint(is_final=True)
            
            if self.config.push_to_hub:
                self.push_to_hub()
    
    @staticmethod
    def _patchify_latents(latents):
        """Turn latent image patches into flattened features (2x2 patches)."""
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _pack_latents(latents):
        """Flatten 2D patches into a 1D sequence."""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @staticmethod
    def _prepare_latent_ids(latents):
        """Generate 4D IDs (T, H, W, L) for positional embeddings."""
        batch_size, _, height, width = latents.shape
        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0] - layer dimension
        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_ids

    def _training_step(self, batch) -> torch.Tensor:
        """Single training step using flow matching (for FLUX models)."""
        images = batch["images"].to(self.device).to(dtype=self.pipe.vae.dtype)
        captions = batch["captions"]
        
        # Add trigger word if specified
        if self.config.trigger_word:
            captions = [f"{self.config.trigger_word}, {c}" for c in captions]
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.pipe.vae.encode(images).latent_dist.sample()
            # FLUX.2-klein uses 32 latent channels. Patchification turns 2x2 patches into 128 features.
            latents = self._patchify_latents(latents)
        
        # Sample noise and timesteps for flow matching
        noise = torch.randn_like(latents)
        
        # Map [0, 1] range to scheduler steps
        u = torch.rand(latents.shape[0], device=self.device)
        timesteps = u * 1000.0  # FLUX models usually trained with 0-1000 range internally
        
        # Flow matching: nominate x_t = (1 - u) * x_0 + u * epsilon
        # u = 0 (data), u = 1 (noise)
        sigmas_t = u.view(-1, 1, 1, 1)
        noisy_latents = (1 - sigmas_t) * latents + sigmas_t * noise
        
        # Target for rectified flow is the velocity (v = epsilon - x_0)
        target = noise - latents
        
        # Pack latents for transformer (B, H*W, 128)
        packed_noisy_latents = self._pack_latents(noisy_latents)
        packed_target = self._pack_latents(target)
        img_ids = self._prepare_latent_ids(noisy_latents).to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            prompt_embeds, text_ids = self.pipe.encode_prompt(
                captions,
            )
        
        # Predict velocity
        model_output = self.pipe.transformer(
            hidden_states=packed_noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
        
        # Compute flow matching loss
        loss = F.mse_loss(model_output.float(), packed_target.float(), reduction="mean")
        
        return loss
    
    @torch.no_grad()
    def generate_samples(self):
        """Generate sample images."""
        if not self.accelerator.is_main_process:
            return
        
        self.accelerator.print(f"\n🎨 Generating samples at step {self.global_step}...")
        self.pipe.transformer.eval()
        
        sample_dir = self.output_dir / f"samples_step_{self.global_step}"
        sample_dir.mkdir(exist_ok=True)
        
        for i, prompt in enumerate(self.config.sample_prompts):
            # Add trigger word
            if self.config.trigger_word:
                prompt = f"{self.config.trigger_word}, {prompt}"
            
            image = self.pipe(
                prompt=prompt,
                height=self.config.dataset.resolution,
                width=self.config.dataset.resolution,
                num_inference_steps=self.config.sample_steps,
                guidance_scale=self.config.sample_guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(42),
            ).images[0]
            
            image.save(sample_dir / f"sample_{i:02d}.png")
        
        self.accelerator.print(f"   Saved samples to {sample_dir}")
        self.pipe.transformer.train()
    
    def save_checkpoint(self, is_final: bool = False):
        """Save LoRA checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        step_str = "final" if is_final else f"step_{self.global_step}"
        self.accelerator.print(f"\n💾 Saving checkpoint: {step_str}")
        
        save_dir = self.output_dir / step_str
        save_dir.mkdir(exist_ok=True)
        
        # Save LoRA weights (Only save the Peft weights to keep size small)
        self.pipe.save_lora_weights(
            save_directory=save_dir,
            transformer_lora_layers=get_peft_model_state_dict(
                self.accelerator.unwrap_model(self.pipe.transformer)
            ),
        )
        
        # Save config
        self.config.to_yaml(save_dir / "config.yaml")
        
        self.accelerator.print(f"   Saved to {save_dir}")
    
    def push_to_hub(self):
        """Push to HuggingFace Hub."""
        if not self.accelerator.is_main_process:
            return
        
        hub_model_id = self.config.hub_model_id or "Limbicnation/pixel-art-lora"
        self.accelerator.print(f"\n🚀 Pushing to HuggingFace Hub: {hub_model_id}")
        
        # Create repo if it doesn't exist
        api = HfApi()
        try:
            create_repo(hub_model_id, exist_ok=True, private=self.config.hub_private)
        except Exception as e:
            self.accelerator.print(f"   Repo check: {e}")
        
        # Upload
        api.upload_folder(
            folder_path=self.output_dir / "final",
            repo_id=hub_model_id,
            repo_type="model",
        )
        
        self.accelerator.print(f"   ✅ Uploaded to https://huggingface.co/{hub_model_id}")
