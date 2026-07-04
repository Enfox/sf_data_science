"""
AdaIN Style Transfer — Architecture
VGG19 encoder (frozen) + AdaIN layer + trainable decoder

Based on: Huang & Belongie, "Arbitrary Style Transfer in Real-time
with Adaptive Instance Normalization" (ICCV 2017)
"""
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


# ─── VGG19 Encoder ───────────────────────────────────────────────

# Layer indices in vgg19.features corresponding to ReLU outputs
VGG19_LAYER_INDICES = {
    "relu1_1": 0,   # after conv1_1 + relu
    "relu2_1": 5,   # after conv2_1 + relu
    "relu3_1": 10,  # after conv3_1 + relu
    "relu3_4": 16,  # after conv3_4 + relu (used by some implementations)
    "relu4_1": 25,  # after conv4_1 + relu
    "relu4_4": 34,  # after conv4_4 + relu
    "relu5_1": 40,  # after conv5_1 + relu
}


class VGGEncoder(nn.Module):
    """Pretrained VGG19 feature extractor (frozen).
    
    Extracts features at specified ReLU layers.
    Output of relu4_1 is used as the content/style feature space.
    """

    def __init__(self, checkpoint_dir=None):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Normalization (ImageNet stats)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def normalize(self, x):
        """Normalize input from [0,1] to ImageNet-normalized."""
        return (x - self.mean) / self.std

    def forward(self, x, return_layers=None, layers=None):
        """Extract features at specified layers.
        
        Args:
            x: input image in [0, 1] range, shape (B, 3, H, W)
            return_layers: list of layer names (e.g. ['relu1_1', 'relu4_1']).
                           'layers' is an alias for return_layers.
                           If None, returns only relu4_1.
        
        Returns:
            dict of layer_name -> feature tensor (if multiple layers)
            or single tensor (if single/None)
        """
        x = self.normalize(x)

        if layers is not None:
            return_layers = layers
        if return_layers is None:
            return_layers = ["relu4_1"]

        if isinstance(return_layers, (list, tuple)):
            return_layers = {name: VGG19_LAYER_INDICES[name] for name in return_layers}

        features = {}
        max_idx = max(return_layers.values())

        for i, layer in enumerate(self.features):
            x = layer(x)
            for name, idx in return_layers.items():
                if i == idx:
                    features[name] = x
            if i >= max_idx:
                break

        if len(features) == 1:
            return list(features.values())[0]
        return features


# ─── AdaIN Layer ─────────────────────────────────────────────────


def calc_mean_std(feat, eps=1e-5):
    """Calculate spatial mean and std per channel.
    
    Args:
        feat: (B, C, H, W)
    Returns:
        mean: (B, C, 1, 1)
        std:  (B, C, 1, 1)
    """
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """AdaIN: align mean/std of content to style.
    
    t = σ(style) * (content − μ(content)) / σ(content) + μ(style)
    
    Args:
        content_feat: (B, C, H, W) — content features from VGG
        style_feat:   (B, C, H', W') — style features from VGG
    
    Returns:
        (B, C, H, W) — stylized features (same spatial size as content)
    """
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


# ─── Decoder ─────────────────────────────────────────────────────


class Decoder(nn.Module):
    """Decoder network — mirror of VGG19 layers relu4_1 → input.
    
    Takes AdaIN output (512-dim features at 32×32 for 256×256 input)
    and reconstructs a 3-channel image.
    
    Architecture is the symmetric inverse of VGG19 encoder up to relu4_1.
    Uses nearest-neighbor upsampling + conv (not transposed conv).
    """

    def __init__(self):
        super().__init__()

        # Reverse of VGG19 features[:26] (up to relu4_1, index 25)
        # Input: 512 channels (relu4_1 output)
        
        # Block 4: 512→512 (no pooling, just convs — reverse of layers 19-25)
        self.block4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=False),
        )

        # Upsample to block 3 size
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")

        # Block 3: 256→256→256→128 (reverse of layers 10-18)
        self.block3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=False),
        )

        # Upsample to block 2 size
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")

        # Block 2: 128→128→64 (reverse of layers 5-9)
        self.block2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=False),
        )

        # Upsample to block 1 size
        self.upsample3 = nn.Upsample(scale_factor=2, mode="nearest")

        # Block 1: 64→64→3 (reverse of layers 0-4)
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.block4(x)
        x = self.upsample1(x)
        x = self.block3(x)
        x = self.upsample2(x)
        x = self.block2(x)
        x = self.upsample3(x)
        x = self.block1(x)
        return x


# ─── Full Style Transfer Net ─────────────────────────────────────


class StyleTransferNet(nn.Module):
    """Complete AdaIN style transfer network.
    
    Combines frozen VGG encoder, AdaIN layer, and trainable decoder.
    """

    def __init__(self):
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = Decoder()
        self.num_adain_input = 512  # channels at relu4_1

    def encode(self, x, layers=None):
        """Extract VGG features."""
        if layers is None:
            return self.encoder(x, layers=["relu4_1"])
        return self.encoder(x, layers=layers)

    def forward(self, content, style, alpha=1.0):
        """Forward pass: content + style → stylized image.
        
        Args:
            content: (B, 3, H, W) in [0, 1]
            style:   (B, 3, H', W') in [0, 1]
            alpha:   style interpolation factor (1.0 = full style)
        
        Returns:
            stylized: (B, 3, H, W) in [0, 1] (approximately)
        """
        # Encode
        content_feat = self.encoder(content, layers=["relu4_1"])
        style_feat = self.encoder(style, layers=["relu4_1"])

        # AdaIN
        t = adaptive_instance_normalization(content_feat, style_feat)

        # Interpolate between content and stylized features
        if alpha < 1.0:
            t = alpha * t + (1 - alpha) * content_feat

        # Decode
        out = self.decoder(t)
        return out


# ─── Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    net = StyleTransferNet().cuda()
    decoder_params = sum(p.numel() for p in net.decoder.parameters())
    print(f"Decoder parameters: {decoder_params / 1e6:.2f}M")

    content = torch.randn(2, 3, 256, 256).cuda()
    style = torch.randn(2, 3, 256, 256).cuda()

    with torch.no_grad():
        out = net(content, style)
    print(f"Input:  content={content.shape}, style={style.shape}")
    print(f"Output: {out.shape}")
    print(f"Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print("✓ Architecture test passed")
