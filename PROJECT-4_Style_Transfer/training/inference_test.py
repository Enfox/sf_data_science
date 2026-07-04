"""
Inference test: run trained model on magenta test images.

Generates stylized outputs for multiple (content, style) pairs,
saves comparison grids for README.
"""
import os
import sys
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from model import StyleTransferNet


def load_image(path, size=512):
    """Load image as tensor [1, 3, H, W] in [0, 1]."""
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def save_image(tensor, path):
    """Save tensor [1, 3, H, W] or [3, H, W] as image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0, 1).cpu()
    img = T.ToPILImage()(tensor)
    img.save(path)


def run_inference(
    checkpoint_path=None,
    content_dir=None,
    style_dir=None,
    output_dir=None,
    image_size=512,
    alpha=1.0,
):
    """Run inference on all content × style combinations."""
    project_root = Path(__file__).parent.parent

    if checkpoint_path is None:
        checkpoint_path = project_root / "checkpoints" / "decoder_final.pth"
    if content_dir is None:
        content_dir = project_root / "test_images" / "magenta_content"
    if style_dir is None:
        style_dir = project_root / "test_images" / "magenta_styles"
    if output_dir is None:
        output_dir = project_root / "test_images" / "results"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    net = StyleTransferNet().to(device)
    
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        net.decoder.load_state_dict(ckpt["decoder"])
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"  Epoch: {ckpt.get('epoch', '?')}, Loss: {ckpt.get('loss', '?'):.4f}")
    else:
        print(f"⚠ No checkpoint found at {checkpoint_path}")
        print("  Using randomly initialized decoder (output will be untrained)")
    
    net.eval()

    # Collect images
    valid_ext = {".jpg", ".jpeg", ".png"}
    content_images = sorted([f for f in Path(content_dir).iterdir() if f.suffix.lower() in valid_ext])
    style_images = sorted([f for f in Path(style_dir).iterdir() if f.suffix.lower() in valid_ext])

    print(f"\nContent images: {len(content_images)}")
    print(f"Style images:   {len(style_images)}")
    print(f"Output dir:     {output_dir}\n")

    # Process each content × style pair
    with torch.no_grad():
        for ci, content_path in enumerate(content_images):
            content = load_image(str(content_path), size=image_size).to(device)
            content_name = content_path.stem

            for si, style_path in enumerate(style_images):
                style = load_image(str(style_path), size=image_size).to(device)
                style_name = style_path.stem

                output = net(content, style, alpha=alpha)

                out_name = f"{content_name}_x_{style_name}.jpg"
                save_image(output, str(output_dir / out_name))
                print(f"  {out_name}", end="\r")

    print(f"\n\n✓ Inference complete! Results in {output_dir}")
    print(f"  Total images: {len(content_images) * len(style_images)}")

    # Also save a combined grid for the best pairs
    try:
        from torchvision.utils import make_grid
        grid_dir = output_dir / "grids"
        grid_dir.mkdir(exist_ok=True)
        
        # Create grid: content | style | result for first 4 styles on first content
        for content_path in content_images[:2]:
            content = load_image(str(content_path), size=image_size).to(device)
            c_name = content_path.stem
            
            for style_path in style_images[:4]:
                style = load_image(str(style_path), size=image_size).to(device)
                s_name = style_path.stem
                
                with torch.no_grad():
                    output = net(content, style, alpha=alpha)
                
                # Hstack content, style, output
                row = torch.cat([content, style, output], dim=0)
                grid = make_grid(row, nrow=3, padding=4)
                save_image(grid, str(grid_dir / f"grid_{c_name}_x_{s_name}.jpg"))
        
        print(f"  Comparison grids: {grid_dir}")
    except Exception as e:
        print(f"  Grid generation skipped: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run style transfer inference")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--content_dir", type=str, default=None)
    parser.add_argument("--style_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    
    run_inference(
        checkpoint_path=args.checkpoint,
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        output_dir=args.output_dir,
        image_size=args.size,
        alpha=args.alpha,
    )
