"""Dataset module for FLUX.2-klein training."""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ImageCaptionDataset(Dataset):
    """Dataset for image-caption pairs."""
    
    def __init__(
        self,
        data_dir: str,
        caption_ext: str = "txt",
        resolution: int = 512,
        center_crop: bool = False,
        random_flip: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.caption_ext = caption_ext
        self.resolution = resolution
        
        # Find all images
        self.image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            self.image_paths.extend(list(self.data_dir.glob(ext)))
        
        self.image_paths = sorted(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        # Image transforms
        transform_list = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        
        if center_crop:
            transform_list.append(transforms.CenterCrop(resolution))
        else:
            transform_list.append(transforms.RandomCrop(resolution))
        
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1] range
        ])
        
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Load caption
        caption_path = image_path.with_suffix(f".{self.caption_ext}")
        if caption_path.exists():
            caption = caption_path.read_text().strip()
        else:
            # Use filename without extension as caption
            caption = image_path.stem.replace("_", " ").replace("-", " ")
        
        return {
            "images": image,
            "captions": caption,
        }
