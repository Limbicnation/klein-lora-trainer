#!/bin/bash
# Installation script for FLUX.2-klein-4B LoRA Trainer

set -e

echo "🚀 Installing FLUX.2-klein-4B LoRA Trainer"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo "❌ Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"

# Check CUDA
echo "🔍 Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found. Make sure CUDA is installed."
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate peft huggingface-hub
pip install Pillow numpy tqdm safetensors omegaconf click
pip install bitsandbytes  # For 8-bit optimizer

# Install package
echo ""
echo "🔧 Installing flux2-klein-lora package..."
pip install -e .

echo ""
echo "✅ Installation complete!"
echo ""
echo "Quick start:"
echo "  flux2-train init-config"
echo "  flux2-train train --data-dir ./your_images --steps 2000"
