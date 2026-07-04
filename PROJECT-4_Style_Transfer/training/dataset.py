"""
Dataset loaders for AdaIN style transfer training.

Content images: COCO val2017 (5000 images)
Style images:   WikiArt subset (3000 images)

Each batch yields (content_batch, style_batch) independently sampled.
Images are loaded as float tensors in [0, 1] range, size = image_size x image_size.
"""
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(root_dir):
    """Recursively collect all valid image paths under root_dir."""
    root = Path(root_dir)
    images = []
    for f in root.rglob("*"):
        if f.suffix.lower() in VALID_EXTENSIONS:
            images.append(str(f))
    return sorted(images)


class StyleTransferDataset(Dataset):
    """Combined dataset: samples (content, style) pairs independently.
    
    Each __getitem__ returns a random content image and a random style image.
    The same content image can be paired with different styles across epochs.
    """

    def __init__(self, content_dir, style_dir, image_size=256):
        self.content_paths = collect_images(content_dir)
        self.style_paths = collect_images(style_dir)

        if not self.content_paths:
            raise RuntimeError(f"No content images found in {content_dir}")
        if not self.style_paths:
            raise RuntimeError(f"No style images found in {style_dir}")

        print(f"  Content images: {len(self.content_paths)}")
        print(f"  Style images:   {len(self.style_paths)}")

        self.image_size = image_size

        # Training transforms: resize → random crop → jitter → tensor
        self.content_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

        self.style_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def __len__(self):
        # Length = max of two datasets (one full pass per epoch)
        return max(len(self.content_paths), len(self.style_paths))

    def __getitem__(self, idx):
        # Content: sequential indexing (wraps around)
        c_path = self.content_paths[idx % len(self.content_paths)]
        # Style: random sample each time
        s_path = random.choice(self.style_paths)

        try:
            content_img = Image.open(c_path).convert("RGB")
        except Exception:
            # Fallback: next image
            content_img = Image.open(
                self.content_paths[(idx + 1) % len(self.content_paths)]
            ).convert("RGB")

        try:
            style_img = Image.open(s_path).convert("RGB")
        except Exception:
            style_img = Image.open(
                self.style_paths[(idx + 1) % len(self.style_paths)]
            ).convert("RGB")

        content = self.content_transform(content_img)
        style = self.style_transform(style_img)

        return content, style


def get_train_loader(content_dir, style_dir, image_size=256,
                     batch_size=8, num_workers=4, shuffle=True):
    """Create training DataLoader."""
    dataset = StyleTransferDataset(
        content_dir=content_dir,
        style_dir=style_dir,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    return loader, dataset


# ─── Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing dataset loader...")
    loader, ds = get_train_loader(
        content_dir=r"D:\project-4_style_transfer\data\coco_content\val2017",
        style_dir=r"D:\project-4_style_transfer\data\wikiart_styles",
        image_size=256,
        batch_size=4,
        num_workers=0,
    )
    content, style = next(iter(loader))
    print(f"Content batch: {content.shape}, range [{content.min():.2f}, {content.max():.2f}]")
    print(f"Style batch:   {style.shape}, range [{style.min():.2f}, {style.max():.2f}]")
    print(f"Dataset length: {len(ds)}")
    print("✓ Dataset test passed")
