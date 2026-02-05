"""Command-line interface for FLUX.2-klein LoRA trainer."""

import click
from pathlib import Path
from typing import Optional
from flux2_klein_trainer.trainer import KleinLoRATrainer
from flux2_klein_trainer.config import TrainingConfig, ModelConfig, LoRAConfig, DatasetConfig


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """FLUX.2-klein-4B LoRA Trainer CLI
    
    Fast, memory-efficient fine-tuning for FLUX.2-klein-4B.
    Optimized for consumer GPUs (RTX 3090/4070+ with 13GB+ VRAM).
    """
    pass


@cli.command()
@click.option("--data-dir", "-d", required=True, type=click.Path(exists=True), 
              help="Directory containing training images")
@click.option("--output-dir", "-o", default="./output/flux2-klein-lora", 
              help="Output directory for checkpoints")
@click.option("--resolution", "-r", default=512, type=int,
              help="Training resolution (512-1024)")
@click.option("--steps", "-s", default=2000, type=int,
              help="Number of training steps")
@click.option("--batch-size", "-b", default=1, type=int,
              help="Batch size (use 1 for 16GB VRAM)")
@click.option("--learning-rate", "-lr", default=1e-4, type=float,
              help="Learning rate")
@click.option("--lora-rank", default=16, type=int,
              help="LoRA rank (4-128)")
@click.option("--lora-alpha", default=16, type=int,
              help="LoRA alpha")
@click.option("--trigger-word", "-t", default="pixel art sprite",
              help="Trigger word to prepend to captions")
@click.option("--hub-model-id", default=None,
              help="HuggingFace Hub model ID (e.g., 'username/model-name')")
@click.option("--push-to-hub", is_flag=True,
              help="Push to HuggingFace Hub after training")
@click.option("--low-vram", is_flag=True,
              help="Enable CPU offloading for low VRAM GPUs")
@click.option("--resume-from-checkpoint", default=None, type=click.Path(exists=True),
              help="Path to a checkpoint folder to resume from")
@click.option("--caption-ext", default="txt",
              help="Caption file extension")
def train(
    data_dir: str,
    output_dir: str,
    resolution: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    lora_rank: int,
    lora_alpha: int,
    trigger_word: str,
    hub_model_id: Optional[str],
    push_to_hub: bool,
    low_vram: bool,
    resume_from_checkpoint: Optional[str],
    caption_ext: str,
):
    """Train a LoRA model on FLUX.2-klein-4B."""
    
    click.echo("🚀 FLUX.2-klein-4B LoRA Trainer v0.1.0")
    click.echo()
    
    # Build config
    config = TrainingConfig(
        model=ModelConfig(
            pretrained_model_name="black-forest-labs/FLUX.2-klein-4B",
            dtype="bfloat16",
            enable_cpu_offload=low_vram,
        ),
        lora=LoRAConfig(
            rank=lora_rank,
            alpha=lora_alpha,
        ),
        dataset=DatasetConfig(
            data_dir=data_dir,
            caption_ext=caption_ext,
            resolution=resolution,
        ),
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        num_train_steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        trigger_word=trigger_word,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )
    
    # Print config
    click.echo("📋 Training Configuration:")
    click.echo(f"   Data dir: {data_dir}")
    click.echo(f"   Output dir: {output_dir}")
    click.echo(f"   Resolution: {resolution}x{resolution}")
    click.echo(f"   Steps: {steps}")
    click.echo(f"   Batch size: {batch_size}")
    click.echo(f"   LoRA rank: {lora_rank}")
    click.echo(f"   LoRA alpha: {lora_alpha}")
    click.echo(f"   Learning rate: {learning_rate}")
    click.echo(f"   Trigger word: '{trigger_word}'")
    click.echo(f"   Low VRAM mode: {'Yes' if low_vram else 'No'}")
    click.echo()
    
    # Start training
    trainer = KleinLoRATrainer(config)
    trainer.train()


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to YAML config file")
def train_config(config: str):
    """Train using a YAML configuration file."""
    click.echo(f"📂 Loading config from {config}")
    
    cfg = TrainingConfig.from_yaml(config)
    trainer = KleinLoRATrainer(cfg)
    trainer.train()


@cli.command()
@click.option("--output", "-o", default="config.yaml",
              help="Output config file path")
def init_config(output: str):
    """Create a sample training configuration file."""
    config = TrainingConfig()
    config.to_yaml(output)
    click.echo(f"✅ Created sample config: {output}")
    click.echo("   Edit this file and run: flux2-train train-config -c config.yaml")


@cli.command()
@click.option("--model-path", "-m", required=True, type=click.Path(exists=True),
              help="Path to trained LoRA weights")
@click.option("--prompt", "-p", required=True,
              help="Prompt for generation")
@click.option("--output", "-o", default="output.png",
              help="Output image path")
@click.option("--steps", "-s", default=4, type=int,
              help="Number of inference steps (FLUX.2-klein uses 4)")
@click.option("--guidance-scale", "-g", default=1.0, type=float,
              help="Guidance scale (FLUX.2-klein uses 1.0)")
@click.option("--resolution", "-r", default=512, type=int,
              help="Image resolution")
def generate(
    model_path: str,
    prompt: str,
    output: str,
    steps: int,
    guidance_scale: float,
    resolution: int,
):
    """Generate an image using a trained LoRA."""
    import torch
    from diffusers import Flux2KleinPipeline
    
    click.echo("🎨 Generating image...")
    
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(model_path)
    pipe = pipe.to("cuda")
    
    image = pipe(
        prompt,
        height=resolution,
        width=resolution,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    ).images[0]
    
    image.save(output)
    click.echo(f"✅ Saved to {output}")


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
