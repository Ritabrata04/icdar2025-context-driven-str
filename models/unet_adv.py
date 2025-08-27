# models/unet_adv.py
"""
AdvancedUNet: Attention-aided U-Net with a ResNet-50 encoder and BatchNorm bottleneck.

This module implements a strong, training-free (at runtime) segmentation backbone that
you can drop into the pipeline for scene text foreground extraction. It follows the
"Attention U-Net" idea (gated skip connections) while leveraging a ResNet-50 encoder
(pretrained on ImageNet by default) and a clean, BN-regularized bottleneck.

Key design points
-----------------
• Encoder: torchvision.models.resnet50 (weights optional). We expose the following
  feature maps with progressively reduced spatial resolutions:
    - e0: stem output after conv1/bn1/relu            (H/2,   64 ch)
    - e1: layer1 output                                (H/4,  256 ch)
    - e2: layer2 output                                (H/8,  512 ch)
    - e3: layer3 output                                (H/16, 1024 ch)
    - e4: layer4 output                                (H/32, 2048 ch)  ← bottleneck in U-Net terms

• Decoder: five upsampling stages (nearest+1×1 conv) each followed by an Attention Gate
  on the corresponding encoder feature and a 2×ConvBlock (Conv→BN→ReLU×2).
  Channel plan (out_channels after each stage):
     e4 (2048) → d3 (1024) → d2 (512) → d1 (256) → d0 (128) → d_ (64)
  The final 1×1 Conv maps 64 → num_classes (default 1). Output is *logits* (no sigmoid).

• Attention Gate (Oktay et al.): computes an additive attention map ψ = σ(W( ReLU(W_g g + W_x x) )).
  We multiply the skip x by ψ and concatenate with the upsampled decoder feature g before ConvBlock.

• Bottleneck BN: the bottleneck (on top of ResNet layer4) uses Conv→BN→ReLU→Conv→BN→ReLU
  to stabilize the very deep feature tensor.

• Robust shape handling: if spatial sizes ever mismatch due to odd input sizes,
  we interpolate encoder features to the decoder feature size before attention.

Usage
-----
    from models.unet_adv import AdvancedUNet
    net = AdvancedUNet(num_classes=1, in_channels=3, encoder_pretrained=True).eval()
    logits = net(torch.randn(1,3,512,512))  # -> [1, 1, 512, 512]

Notes
-----
- You typically apply `.sigmoid()` on the logits when using a single foreground class.
- Inputs need not be multiples of 32; the module internally interpolates to align skips.
- If you require strict determinism, set cudnn flags at process start (outside this file).

Author: ChatGPT (for research reproducibility)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- Building Blocks -------------------------------

class ConvBlock(nn.Module):
    """
    Two 3×3 Conv layers with BN and ReLU.
    Conv(in_ch → out_ch) → BN → ReLU → Conv(out_ch → out_ch) → BN → ReLU.
    """

    def __init__(self, in_ch: int, out_ch: int, bn_eps: float = 1e-5, bn_momentum: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum)
        self.act = nn.ReLU(inplace=True)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class AttentionGate(nn.Module):
    """
    Additive attention gate for skip connections.

    Given:
        - g: gating signal from decoder (coarse, low-res feature)
        - x: skip connection from encoder (finer, higher-res feature)

    We first align channel dims (1×1 convs) and spatial dims (interpolate if needed),
    then compute:
        att = σ( ψ( ReLU( W_g * g + W_x * x ) ) )
    and return x * att.

    inter_ch controls the internal reduction. By default we pick max(out_ch // 2, 16).
    """

    def __init__(self, g_ch: int, x_ch: int, out_ch: int, inter_ch: Optional[int] = None):
        super().__init__()
        inter_ch = inter_ch or max(out_ch // 2, 16)

        self.W_g = nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_ch, out_ch, kernel_size=1, bias=False)

        self.bn_g = nn.BatchNorm2d(inter_ch)
        self.bn_x = nn.BatchNorm2d(inter_ch)
        self.bn_psi = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="nearest")

        g1 = self.bn_g(self.W_g(g))
        x1 = self.bn_x(self.W_x(x))
        att = self.act(g1 + x1)
        att = self.bn_psi(self.psi(att))
        att = self.sig(att)
        return x * att


class UpBlock(nn.Module):
    """
    One decoder stage:
      - Upsample the decoder feature (nearest) + 1×1 conv to reduce channels,
      - Attention gate on the encoder skip,
      - Concatenate and pass through a ConvBlock.

    Args:
        dec_in_ch:  channels coming from previous decoder stage
        enc_skip_ch:channels in the encoder skip at this stage
        out_ch:     output channels after ConvBlock
    """

    def __init__(self, dec_in_ch: int, enc_skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dec_in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Attention gate brings enc skip to out_ch channels
        self.att = AttentionGate(g_ch=out_ch, x_ch=enc_skip_ch, out_ch=out_ch)
        # After concat: out_ch (up) + out_ch (gated skip) → 2*out_ch → out_ch
        self.block = ConvBlock(in_ch=2 * out_ch, out_ch=out_ch)

    def forward(self, dec_in: torch.Tensor, enc_skip: torch.Tensor) -> torch.Tensor:
        up = self.up(dec_in)
        if up.shape[2:] != enc_skip.shape[2:]:
            enc_skip = F.interpolate(enc_skip, size=up.shape[2:], mode="bilinear", align_corners=False)
        gated = self.att(up, enc_skip)
        x = torch.cat([up, gated], dim=1)
        return self.block(x)


# ------------------------------- Encoder (ResNet-50) -------------------------------

def _make_resnet50(encoder_pretrained: bool) -> nn.Module:
    """
    Constructs a torchvision ResNet-50 and returns the full model.
    We'll manually walk its layers in the encoder forward pass.
    """
    try:
        from torchvision.models import resnet50
        # Newer torchvision uses weights enums; we support both APIs.
        if encoder_pretrained:
            try:
                from torchvision.models import ResNet50_Weights
                model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            except Exception:
                model = resnet50(pretrained=True)
        else:
            model = resnet50(weights=None)  # torchvision >= 0.13
    except Exception:
        # Fallback to legacy signature
        from torchvision.models import resnet50
        model = resnet50(pretrained=encoder_pretrained)
    return model


# ------------------------------- Advanced U-Net -------------------------------

@dataclass
class AdvancedUNetConfig:
    num_classes: int = 1
    in_channels: int = 3
    encoder_pretrained: bool = True
    freeze_encoder: bool = False
    # decoder channel plan: 2048→1024→512→256→128→64
    dec_channels: Tuple[int, int, int, int, int] = (1024, 512, 256, 128, 64)
    bn_eps: float = 1e-5
    bn_momentum: float = 0.1


class AdvancedUNet(nn.Module):
    """
    Attention-aided U-Net with ResNet-50 encoder and BN bottleneck.

    Forward input:
        x: [B, in_channels, H, W]

    Forward output:
        logits: [B, num_classes, H, W]
    """

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 3,
        encoder_pretrained: bool = True,
        freeze_encoder: bool = False,
        dec_channels: Tuple[int, int, int, int, int] = (1024, 512, 256, 128, 64),
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ):
        super().__init__()
        self.cfg = AdvancedUNetConfig(
            num_classes=num_classes,
            in_channels=in_channels,
            encoder_pretrained=encoder_pretrained,
            freeze_encoder=freeze_encoder,
            dec_channels=dec_channels,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )

        # ---- Encoder (ResNet-50) ----
        self.encoder = _make_resnet50(encoder_pretrained)
        if in_channels != 3:
            # Replace the first conv if a different #input channels is required
            old = self.encoder.conv1
            self.encoder.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                                           stride=old.stride, padding=old.padding, bias=False)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ---- Bottleneck block (BN) on top of layer4 ----
        self.bottleneck = ConvBlock(2048, 2048, bn_eps=bn_eps, bn_momentum=bn_momentum)

        # ---- Decoder stages ----
        dch = dec_channels  # (1024, 512, 256, 128, 64)
        # stage d3: 2048 → 1024 (skip from e3: 1024)
        self.up3 = UpBlock(dec_in_ch=2048, enc_skip_ch=1024, out_ch=dch[0])
        # stage d2: 1024 → 512 (skip from e2: 512)
        self.up2 = UpBlock(dec_in_ch=dch[0], enc_skip_ch=512, out_ch=dch[1])
        # stage d1: 512 → 256 (skip from e1: 256)
        self.up1 = UpBlock(dec_in_ch=dch[1], enc_skip_ch=256, out_ch=dch[2])
        # stage d0: 256 → 128 (skip from e0: 64)
        self.up0 = UpBlock(dec_in_ch=dch[2], enc_skip_ch=64, out_ch=dch[3])
        # stage d_: 128 → 64 (no encoder skip at this scale)
        self.up_ = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dch[3], dch[4], kernel_size=1, bias=False),
            nn.BatchNorm2d(dch[4], eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            ConvBlock(dch[4], dch[4], bn_eps=bn_eps, bn_momentum=bn_momentum),
        )

        # ---- Prediction head ----
        self.head = nn.Conv2d(dch[4], num_classes, kernel_size=1, bias=True)
        nn.init.zeros_(self.head.bias)

    # Helper: extract features from ResNet-50
    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns [e0, e1, e2, e3, e4] feature maps at scales:
          H/2, H/4, H/8, H/16, H/32
        """
        # ResNet stem
        x = self.encoder.conv1(x)     # /2
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        e0 = x                        # (H/2, 64)

        x = self.encoder.maxpool(x)   # /4
        x = self.encoder.layer1(x)    # /4 → 256 ch
        e1 = x

        x = self.encoder.layer2(x)    # /8 → 512 ch
        e2 = x

        x = self.encoder.layer3(x)    # /16 → 1024 ch
        e3 = x

        x = self.encoder.layer4(x)    # /32 → 2048 ch
        e4 = x

        return [e0, e1, e2, e3, e4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing logits at input resolution.
        """
        B, C, H, W = x.shape
        e0, e1, e2, e3, e4 = self._encode(x)

        # Bottleneck refinement
        b = self.bottleneck(e4)  # [B, 2048, H/32, W/32]

        # Decoder
        d3 = self.up3(b, e3)     # [B, dec[0], H/16, W/16]
        d2 = self.up2(d3, e2)    # [B, dec[1], H/8,  W/8 ]
        d1 = self.up1(d2, e1)    # [B, dec[2], H/4,  W/4 ]
        d0 = self.up0(d1, e0)    # [B, dec[3], H/2,  W/2 ]
        d_ = self.up_(d0)        # [B, dec[4], H,    W   ]

        logits = self.head(d_)   # [B, num_classes, H, W]
        # Ensure exact spatial size (guard against any rounding from odd inputs)
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


# ------------------------------- Quick Self-Test -------------------------------

if __name__ == "__main__":
    # Smoke test for shapes; run: python -m models.unet_adv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = AdvancedUNet(num_classes=1, in_channels=3, encoder_pretrained=False).to(device).eval()
    x = torch.randn(2, 3, 321, 513, device=device)   # deliberately odd sizes
    with torch.inference_mode():
        y = net(x)
    print("Input :", tuple(x.shape))
    print("Logits:", tuple(y.shape))  # expected (2, 1, 321, 513)
