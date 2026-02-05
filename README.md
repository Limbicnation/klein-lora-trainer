# FLUX.2-klein-4B LoRA Trainer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

A dedicated, memory-efficient LoRA trainer for **FLUX.2-klein-4B**. Built specifically for fine-tuning the fastest FLUX model on consumer GPUs.

## 🌟 Features

- **Optimized for FLUX.2-klein-4B** - Native support for the 4B parameter model
- **Consumer GPU Ready** - Runs on RTX 3090/4070+ with 13GB+ VRAM
- **Memory Efficient** - 8-bit AdamW optimizer + gradient checkpointing
- **Fast Training** - bf16 precision + optimized data loading
- **Easy CLI** - Simple commands for training and inference
- **Hub Integration** - Push to HuggingFace Hub with one flag

## 📋 Requirements

- NVIDIA GPU with 13GB+ VRAM (RTX 3090/4070/4090)
- Python 3.10+
- CUDA 11.8+ or 12.1+

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Limbicnation/flux2-klein-lora.git
cd flux2-klein-lora

# Install
pip install -e .

# Or install from PyPI (when published)
pip install flux2-klein-lora
```

## 📚 Quick Start

### 1. Prepare Your Dataset

Organize your training images with caption files:

```
training_data/
├── image_001.png
├── image_001.txt
├── image_002.png
├── image_002.txt
└── ...
```

**Caption files** (`.txt`) should contain descriptions of each image:
```
A pixel art character sprite, warrior in red armor, transparent background
```

### 2. Train Your LoRA

**Basic training:**
```bash
flux2-train train \
  --data-dir ./training_data \
  --output-dir ./output/my-lora \
  --steps 2000 \
  --lora-rank 16
```

**With trigger word and Hub upload:**
```bash
flux2-train train \
  --data-dir ./training_data \
  --output-dir ./output/pixel-art-lora \
  --steps 2000 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --trigger-word "pixel art sprite" \
  --hub-model-id "your-username/pixel-art-lora" \
  --push-to-hub
```

**Low VRAM mode (for 12GB GPUs):**
```bash
flux2-train train \
  --data-dir ./training_data \
  --low-vram \
  --batch-size 1 \
  --resolution 512
```

### 3. Generate Images

```bash
flux2-train generate \
  --model-path ./output/my-lora/final \
  --prompt "pixel art sprite, a brave knight in golden armor" \
  --output ./knight.png
```

## ⚙️ Configuration

Create a YAML config file for advanced settings:

```bash
flux2-train init-config --output my_config.yaml
```

Edit `my_config.yaml` and run:
```bash
flux2-train train-config -c my_config.yaml
```

### Example Config

```yaml
model:
  pretrained_model_name: "black-forest-labs/FLUX.2-klein-4B"
  dtype: "bfloat16"
  enable_cpu_offload: false

lora:
  rank: 16
  alpha: 16
  dropout: 0.0
  use_rslora: true

dataset:
  data_dir: "./training_data"
  caption_ext: "txt"
  resolution: 512
  random_flip: true

output_dir: "./output/my-lora"
num_train_steps: 2000
batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 0.0001
optimizer: "adamw_8bit"

trigger_word: "pixel art sprite"

sample_prompts:
  - "pixel art sprite, a brave knight"
  - "pixel art sprite, a wizard with staff"

push_to_hub: true
hub_model_id: "your-username/my-lora"
```

## 🎨 Training Tips

### For Pixel Art (like SpriteForge)

```bash
flux2-train train \
  --data-dir ./training_data/processed/images \
  --resolution 512 \
  --steps 2000 \
  --lora-rank 16 \
  --trigger-word "pixel art sprite" \
  --hub-model-id "Limbicnation/pixel-art-lora" \
  --push-to-hub
```

### Recommended Settings

| VRAM | Resolution | Batch Size | Rank | Steps |
|------|------------|------------|------|-------|
| 24GB | 1024 | 2 | 32-64 | 2000-4000 |
| 16GB | 768 | 1 | 16-32 | 2000 |
| 12GB | 512 | 1 | 16 | 1000-2000 |

## 🔧 Advanced Usage

### Custom Dataset Loading

```python
from flux2_klein_trainer import KleinLoRATrainer, TrainingConfig

config = TrainingConfig.from_yaml("my_config.yaml")
trainer = KleinLoRATrainer(config)
trainer.train()
```

### Programmatic Generation

```python
import torch
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.load_lora_weights("./output/my-lora/final")

image = pipe(
    "pixel art sprite, a brave knight",
    num_inference_steps=4,
    guidance_scale=1.0,
).images[0]

image.save("output.png")
```

## 📊 Monitoring Training

Training progress is displayed with `tqdm` progress bars. The trainer will:
- Save checkpoints every 500 steps (configurable)
- Generate sample images every 500 steps
- Log loss values every 10 steps

Sample outputs are saved to `output/samples_step_XXXX/`.

## 🤝 Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

## 📄 License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) file.

FLUX.2-klein-4B is licensed under Apache 2.0 by Black Forest Labs.

## 🙏 Acknowledgments

- [Black Forest Labs](https://bfl.ai/) for FLUX.2-klein-4B
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) team
- [PEFT](https://github.com/huggingface/peft) library

## 📞 Support

- GitHub Issues: [github.com/Limbicnation/flux2-klein-lora/issues](https://github.com/Limbicnation/flux2-klein-lora/issues)
- Discussions: [GitHub Discussions](https://github.com/Limbicnation/flux2-klein-lora/discussions)

---

Made with ❤️ for the SpriteForge project by [Limbicnation](https://github.com/Limbicnation)
