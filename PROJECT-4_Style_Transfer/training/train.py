"""
Training script for AdaIN style transfer.

Loss = L_content + lambda * L_style

  L_content = MSE( VGG(output)[:relu4_1], AdaIN(content_feat, style_feat) )
  L_style   = sum over layers (relu1_1..relu4_1):
              MSE( mean(feat_out) - mean(feat_style) ) + MSE( std(feat_out) - std(feat_style) )

Optimizer: Adam (lr=1e-4), CosineAnnealingLR
"""
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Add training dir to path
sys.path.insert(0, str(Path(__file__).parent))

from model import StyleTransferNet, adaptive_instance_normalization, calc_mean_std
from dataset import get_train_loader


def load_config(config_path=None):
    """Load training config from YAML or use defaults."""
    defaults = {
        "image_size": 256,
        "batch_size": 8,
        "epochs": 20,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "content_weight": 1.0,
        "style_weight": 1.0,
        "num_workers": 4,
        "save_every": 2,
        "checkpoint_dir": "checkpoints",
        "content_dir": "data/coco_content/val2017",
        "style_dir": "data/wikiart_styles",
    }
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            defaults.update(yaml.safe_load(f))
    return defaults


def train(config_path=None):
    cfg = load_config(config_path)
    print("=" * 60)
    print("  AdaIN Style Transfer — Training")
    print("=" * 60)

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    content_dir = project_root / cfg["content_dir"]
    style_dir = project_root / cfg["style_dir"]
    ckpt_dir = project_root / cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConfig:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # ─── Device ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # ─── Data ────────────────────────────────────────────────────
    print(f"\nLoading datasets...")
    loader, dataset = get_train_loader(
        content_dir=str(content_dir),
        style_dir=str(style_dir),
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    steps_per_epoch = len(loader)
    print(f"  Steps per epoch: {steps_per_epoch}")

    # ─── Model ───────────────────────────────────────────────────
    print(f"\nBuilding model...")
    net = StyleTransferNet().to(device)
    encoder = net.encoder
    decoder = net.decoder

    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Only train decoder
    decoder.train()
    dec_params = list(decoder.parameters())
    print(f"  Trainable parameters: {sum(p.numel() for p in dec_params) / 1e6:.2f}M")

    # ─── Optimizer + Scheduler ───────────────────────────────────
    optimizer = torch.optim.Adam(
        dec_params,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] * steps_per_epoch
    )

    # ─── Style layers for loss ───────────────────────────────────
    style_layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1"]
    content_layer = "relu4_1"

    # ─── Training loop ───────────────────────────────────────────
    print(f"\nStarting training: {cfg['epochs']} epochs\n")
    best_loss = float("inf")
    log_file = ckpt_dir / "training_log.csv"
    with open(log_file, "w") as f:
        f.write("epoch,step,total_loss,content_loss,style_loss,lr,time_per_step\n")

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        epoch_total_loss = 0.0
        epoch_content_loss = 0.0
        epoch_style_loss = 0.0

        for step, (content, style) in enumerate(loader):
            content = content.to(device, non_blocking=True)
            style = style.to(device, non_blocking=True)

            optimizer.zero_grad()

            # ── Forward ──────────────────────────────────────────
            with torch.no_grad():
                # Encode content and style
                content_feat = encoder(content, layers=[content_layer])
                style_feats = encoder(style, layers=style_layers)

            # AdaIN target
            t = adaptive_instance_normalization(content_feat, style_feats[content_layer])

            # Decode
            output = decoder(t)

            # ── Loss ─────────────────────────────────────────────
            # Encode output for loss computation
            output_feats = encoder(output, layers=style_layers + [content_layer])

            # Content loss: MSE between output features and AdaIN target
            # (content_layer is the last in style_layers, so it's also in output_feats)
            # But t uses relu4_1 which = content_layer
            loss_content = nn.functional.mse_loss(
                output_feats[content_layer], t.detach()
            )

            # Style loss: match mean/std of output features to style features
            loss_style = 0.0
            for layer_name in style_layers:
                out_feat = output_feats[layer_name]
                s_feat = style_feats[layer_name]
                out_mean, out_std = calc_mean_std(out_feat)
                s_mean, s_std = calc_mean_std(s_feat)
                loss_style += nn.functional.mse_loss(out_mean, s_mean)
                loss_style += nn.functional.mse_loss(out_std, s_std)
            loss_style /= len(style_layers)

            # Total loss
            loss = cfg["content_weight"] * loss_content + cfg["style_weight"] * loss_style

            # ── Backward ─────────────────────────────────────────
            loss.backward()
            optimizer.step()
            scheduler.step()

            # ── Logging ──────────────────────────────────────────
            epoch_total_loss += loss.item()
            epoch_content_loss += loss_content.item()
            epoch_style_loss += loss_style.item()

            if (step + 1) % 50 == 0 or step == 0:
                avg_loss = epoch_total_loss / (step + 1)
                elapsed = time.time() - epoch_start
                speed = (step + 1) / max(elapsed, 0.001)
                eta_sec = (steps_per_epoch - step - 1) / max(speed, 0.001)
                print(
                    f"  Epoch {epoch+1:2d}/{cfg['epochs']} | "
                    f"Step {step+1:4d}/{steps_per_epoch} | "
                    f"Loss: {loss.item():.4f} "
                    f"(c: {loss_content.item():.4f}, s: {loss_style.item():.4f}) | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                    f"{speed:.1f} step/s | "
                    f"ETA: {eta_sec/60:.1f} min",
                    end="\r",
                )
                with open(log_file, "a") as f:
                    f.write(
                        f"{epoch+1},{step+1},{loss.item():.6f},"
                        f"{loss_content.item():.6f},{loss_style.item():.6f},"
                        f"{scheduler.get_last_lr()[0]:.8f},{1.0/max(speed,0.001):.4f}\n"
                    )

        # ─── Epoch summary ──────────────────────────────────────
        epoch_time = time.time() - epoch_start
        avg_total = epoch_total_loss / steps_per_epoch
        avg_content = epoch_content_loss / steps_per_epoch
        avg_style = epoch_style_loss / steps_per_epoch
        print(
            f"\n  Epoch {epoch+1:2d} done | "
            f"Avg loss: {avg_total:.4f} "
            f"(c: {avg_content:.4f}, s: {avg_style:.4f}) | "
            f"Time: {epoch_time/60:.1f} min"
        )

        # ─── Checkpoint ─────────────────────────────────────────
        if avg_total < best_loss:
            best_loss = avg_total
            best_path = ckpt_dir / "decoder_best.pth"
            torch.save({
                "decoder": decoder.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_total,
            }, best_path)
            print(f"  ★ New best model saved: {best_path}")

        if (epoch + 1) % cfg["save_every"] == 0 or (epoch + 1) == cfg["epochs"]:
            ckpt_path = ckpt_dir / f"decoder_epoch_{epoch+1:02d}.pth"
            torch.save({
                "decoder": decoder.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_total,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        print()

    # ─── Final ──────────────────────────────────────────────────
    final_path = ckpt_dir / "decoder_final.pth"
    torch.save({
        "decoder": decoder.state_dict(),
        "epoch": cfg["epochs"],
        "loss": best_loss,
    }, final_path)
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Final model: {final_path}")
    print(f"  Log: {log_file}")
    print(f"{'='*60}")

    return final_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train(config_path=args.config)
