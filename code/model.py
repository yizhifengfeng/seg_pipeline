# -*- coding: utf-8 -*-
"""
Stage 5+7: Multi-Channel DeepLabV3+ with Texture/Spectral Branches + CBAM
============================================================================
Architecture (input -> output):
  Input: (B, 6, H, W)  [b1, b2, b3, b4, b5, pan_resampled]  z-scored

  1. EdgePriorModule (TEXTURE branch):
     - Compute Sobel edges on pan (channel 5) and b2 (channel 1)
     - Concatenate: (B, 6, H, W) -> (B, 8, H, W)
     - These 2 edge channels serve as explicit texture priors (no extra params)
     - The backbone CNN then automatically LEARNS texture features from these + raw bands

  2. ResNet-34 backbone (modified first conv for 8-channel input, ImageNet pretrained):
     - stem -> layer1 -> layer2 -> layer3 -> layer4
     - SPECTRAL branch: 5 multispectral bands (b1-b5) are fed directly; the CNN
       learns spectral patterns (band ratios, vegetation indices implicitly)

  3. CBAM attention (BACKGROUND SUPPRESSION) after each backbone stage:
     - Channel attention: which feature channels matter (suppress irrelevant)
     - Spatial attention: which spatial regions matter (suppress background)
     - Applied 4 times (after layer1-4)

  4. ASPP (multi-scale context): atrous rates 6/12/18 + image pooling
     - Captures defects at different scales (thin cracks vs large flaking)

  5. Decoder: fuse ASPP features + low-level (layer1) features
     - Low-level preserves spatial detail for crack boundary precision

  6. Output: (B, 5, H, W) logits  [background, crack, bio, flaking, hollow]

Importable: from model import DeepLabV3Plus
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# 1. Edge Prior Module (Texture branch - explicit Sobel priors)
# ---------------------------------------------------------------------------
class EdgePriorModule(nn.Module):
    """Compute Sobel edges on pan and b2, concatenate as extra channels.

    Input: (B, 6, H, W) -> Output: (B, 8, H, W)
    Adds 2 explicit texture edge channels (pan edge + b2 edge).
    Fixed filters (not learned) - acts as hand-crafted prior.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _sobel(self, x):
        gx = F.conv2d(F.pad(x, (1, 1, 1, 1)), self.sobel_x)
        gy = F.conv2d(F.pad(x, (1, 1, 1, 1)), self.sobel_y)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, x):
        # x: (B, 6, H, W); channel 5 = pan, channel 1 = b2
        edge_pan = self._sobel(x[:, 5:6])
        edge_b2 = self._sobel(x[:, 1:2])
        return torch.cat([x, edge_pan, edge_b2], dim=1)  # (B, 8, H, W)


# ---------------------------------------------------------------------------
# 2. CBAM Attention (Background suppression)
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        r = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
        )

    def forward(self, x):
        return torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1))) * x


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018).
    Channel + Spatial attention. Suppresses background, highlights defects.
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


# ---------------------------------------------------------------------------
# 3. ASPP (Atrous Spatial Pyramid Pooling)
# ---------------------------------------------------------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
        super().__init__()
        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))]
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate,
                          dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(  # image-level pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.5))

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res[-1] = F.interpolate(res[-1], size=x.shape[2:],
                                mode="bilinear", align_corners=False)
        return self.project(torch.cat(res, dim=1))


# ---------------------------------------------------------------------------
# 4. DeepLabV3+ Main Model
# ---------------------------------------------------------------------------
class DeepLabV3Plus(nn.Module):
    """Multi-channel DeepLabV3+ for building facade defect segmentation.

    Args:
        num_classes: 5 (background + 4 defect classes)
        backbone_pretrained: use ImageNet pretrained ResNet-34 weights
        output_stride: 16 (default) or 32. OS=16 uses dilated convolutions in
                       layer3 (dilation=2) and layer4 (dilation=4) to preserve
                       spatial resolution. Critical for thin cracks (2-3px wide):
                       at OS=32 a 2px crack becomes 0.06px in the bottleneck
                       feature map -- essentially lost. At OS=16 it keeps 0.12px,
                       which combined with the decoder upsampling recovers
                       usable boundary information.

    Input:  (B, 6, H, W) float32 z-scored
    Output: (B, num_classes, H, W) logits
    """

    def __init__(self, num_classes=5, backbone_pretrained=True, output_stride=16):
        super().__init__()
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.edge_prior = EdgePriorModule()

        # ResNet-34 backbone
        weights = tvm.ResNet34_Weights.DEFAULT if backbone_pretrained else None
        resnet = tvm.resnet34(weights=weights)

        # Modify first conv: 3ch -> 8ch (keep pretrained RGB, init extra channels)
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if backbone_pretrained:
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                # Initialize 5 extra channels (b4,b5,edge_pan,edge_b2) from RGB mean
                mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:] = mean_rgb.expand(-1, 5, -1, -1) / 5.0
        resnet.conv1 = new_conv

        # ── Output Stride 16: convert layer3/4 to dilated convolutions ──
        # Standard ResNet-34: layer3 /16 (stride 2), layer4 /32 (stride 2)
        # OS=16: layer3 stride 2->1 + dilation 2, layer4 stride 2->1 + dilation 4
        # This keeps the receptive field intact while preserving spatial resolution.
        if output_stride == 16:
            self._convert_to_dilated(resnet.layer3, dilation=2)
            self._convert_to_dilated(resnet.layer4, dilation=4)

        # Split backbone
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 64ch,  /4
        self.layer2 = resnet.layer2  # 128ch, /8
        self.layer3 = resnet.layer3  # 256ch, /16 (dilated if OS=16)
        self.layer4 = resnet.layer4  # 512ch, /16 (dilated if OS=16, was /32)

        # CBAM after each stage
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        # ASPP (rates 6/12/18 for OS=16: effective receptive field 96/192/288 px)
        self.aspp = ASPP(512, out_channels=256, atrous_rates=(6, 12, 18))

        # Decoder
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1))

    @staticmethod
    def _convert_to_dilated(layer, dilation):
        """Convert a ResNet layer to use stride=1 + dilated 3x3 convolutions.

        Works for both BasicBlock (ResNet-18/34, stride on conv1) and
        Bottleneck (ResNet-50+, stride on conv2). Sets the stride-1
        modification on whichever conv carries the block stride, and
        applies dilation+padding to all 3x3 convs.
        """
        block = layer[0]
        # ResNet-34 BasicBlock: stride is on conv1 (3x3)
        # ResNet-50 Bottleneck: stride is on conv2 (3x3)
        for conv_attr in ("conv1", "conv2"):
            conv = getattr(block, conv_attr, None)
            if conv is not None and isinstance(conv, nn.Conv2d) and conv.stride != (1, 1):
                conv.stride = (1, 1)
                break
        # downsample: 1x1 conv -> stride 1
        if hasattr(block, 'downsample') and block.downsample is not None:
            block.downsample[0].stride = (1, 1)
        # All 3x3 convs in the layer: set dilation + padding
        for m in layer.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation = (dilation, dilation)
                m.padding = (dilation, dilation)

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.edge_prior(x)                 # (B, 8, H, W)
        x = self.stem(x)                       # (B, 64, H/4)
        low_level = self.cbam1(self.layer1(x)) # (B, 64, H/4)
        x = self.cbam2(self.layer2(low_level)) # (B, 128, H/8)
        x = self.cbam3(self.layer3(x))         # (B, 256, H/16)
        x = self.cbam4(self.layer4(x))         # (B, 512, H/16) -- was H/32
        x = self.aspp(x)                       # (B, 256, H/16)
        x = F.interpolate(x, size=low_level.shape[2:],
                          mode="bilinear", align_corners=False)
        x = torch.cat([x, self.low_level_proj(low_level)], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 5. Baseline Model (pan-only, for comparison experiment)
# ---------------------------------------------------------------------------
class EdgePriorModulePanOnly(nn.Module):
    """仅 pan 的 Sobel 边缘先验.

    Input: (B, 1, H, W) -> Output: (B, 2, H, W)  [pan + edge_pan]
    与 EdgePriorModule 唯一区别: 只算 pan 一个边缘通道.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _sobel(self, x):
        gx = F.conv2d(F.pad(x, (1, 1, 1, 1)), self.sobel_x)
        gy = F.conv2d(F.pad(x, (1, 1, 1, 1)), self.sobel_y)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, x):
        # x: (B, 1, H, W); channel 0 = pan
        edge_pan = self._sobel(x[:, 0:1])
        return torch.cat([x, edge_pan], dim=1)  # (B, 2, H, W)


class DeepLabV3PlusBaseline(nn.Module):
    """基线模型: 仅 pan 输入的 DeepLabV3+ (对比实验用).

    与 DeepLabV3Plus 的唯一差异:
      - 输入: 1 通道 pan (vs 6 通道 b1-b5+pan)
      - EdgePrior: 1 个 Sobel 边缘 (pan only) -> 2 通道 (vs 2 个边缘 -> 8 通道)
      - conv1: 2 通道输入 (vs 8 通道)

    其余完全一致: ResNet-34 backbone, OS=16, CBAM x4, ASPP(6,12,18), Decoder.
    保证对比公平 -- 唯一变量是输入模态(可见光 vs 多光谱融合).

    Input:  (B, 1, H, W) float32 z-scored pan
    Output: (B, num_classes, H, W) logits
    """

    def __init__(self, num_classes=5, backbone_pretrained=True, output_stride=16):
        super().__init__()
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.input_mode = "pan_only"
        self.edge_prior = EdgePriorModulePanOnly()

        # ResNet-34 backbone
        weights = tvm.ResNet34_Weights.DEFAULT if backbone_pretrained else None
        resnet = tvm.resnet34(weights=weights)

        # Modify first conv: 3ch -> 2ch (pan + edge_pan)
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if backbone_pretrained:
            with torch.no_grad():
                # 用预训练 RGB 的灰度均值初始化 2 个通道
                mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                new_conv.weight[:, 0:1] = mean_rgb          # pan
                new_conv.weight[:, 1:2] = mean_rgb * 0.1    # edge: 小初始化(边缘值小)
        resnet.conv1 = new_conv

        # OS=16 dilated backbone (与 DeepLabV3Plus 完全一致)
        if output_stride == 16:
            self._convert_to_dilated(resnet.layer3, dilation=2)
            self._convert_to_dilated(resnet.layer4, dilation=4)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # CBAM (与 DeepLabV3Plus 完全一致)
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        # ASPP (与 DeepLabV3Plus 完全一致)
        self.aspp = ASPP(512, out_channels=256, atrous_rates=(6, 12, 18))

        # Decoder (与 DeepLabV3Plus 完全一致)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1))

    @staticmethod
    def _convert_to_dilated(layer, dilation):
        """与 DeepLabV3Plus._convert_to_dilated 完全一致."""
        block = layer[0]
        for conv_attr in ("conv1", "conv2"):
            conv = getattr(block, conv_attr, None)
            if conv is not None and isinstance(conv, nn.Conv2d) and conv.stride != (1, 1):
                conv.stride = (1, 1)
                break
        if hasattr(block, 'downsample') and block.downsample is not None:
            block.downsample[0].stride = (1, 1)
        for m in layer.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation = (dilation, dilation)
                m.padding = (dilation, dilation)

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.edge_prior(x)                 # (B, 2, H, W)
        x = self.stem(x)                       # (B, 64, H/4)
        low_level = self.cbam1(self.layer1(x)) # (B, 64, H/4)
        x = self.cbam2(self.layer2(low_level)) # (B, 128, H/8)
        x = self.cbam3(self.layer3(x))         # (B, 256, H/16)
        x = self.cbam4(self.layer4(x))         # (B, 512, H/16)
        x = self.aspp(x)                       # (B, 256, H/16)
        x = F.interpolate(x, size=low_level.shape[2:],
                          mode="bilinear", align_corners=False)
        x = torch.cat([x, self.low_level_proj(low_level)], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
if __name__ == "__main__":
    print("=" * 64)
    print("Stage 7: DeepLabV3+ Model Construction & Forward Test")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DeepLabV3Plus(num_classes=5, backbone_pretrained=True,
                          output_stride=16).to(device)
    n_params = model.count_parameters()
    print(f"Trainable parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Forward pass test (keep grad for backward test)
    torch.manual_seed(42)
    dummy_input = torch.randn(2, 6, 512, 512).to(device)
    print(f"Input shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 5, 512, 512), f"Expected (2,5,512,512), got {output.shape}"
    print("Forward pass: OK")

    # Test backward
    target = torch.randint(0, 5, (2, 512, 512)).to(device)
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "06_loss"))
    from losses import CombinedLoss, compute_focal_alpha
    alpha = compute_focal_alpha([501586446, 9719840, 10129403, 5624990, 2038073]).to(device)
    criterion = CombinedLoss(alpha=alpha).to(device)
    loss, parts = criterion(output, target)
    loss.backward()
    print(f"Backward pass: OK (loss={parts})")

    # Memory check
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"Peak GPU memory: {mem:.2f} GB")

    print()
    print("Model architecture summary:")
    print("  Input: 6 channels (b1-b5 + pan) z-scored")
    print("  EdgePrior: +2 Sobel edge channels -> 8 channels")
    print("  Backbone: ResNet-34 (ImageNet pretrained, modified conv1 for 8ch)")
    print("  Output Stride: 16 (dilated layer3=2, layer4=4) -- preserves thin-crack detail")
    print("  CBAM x4: channel + spatial attention after each stage")
    print("  ASPP: atrous rates 6/12/18 + image pooling -> 256ch")
    print("  Decoder: fuse ASPP + low-level -> 5-class logits")
    print("  Output: 5 classes (background, crack, bio, flaking, hollow)")

    # ── 基线模型测试 (对比实验) ──────────────────────────────
    print("\n" + "=" * 64)
    print("Baseline Model (pan-only) Test")
    print("=" * 64)

    baseline = DeepLabV3PlusBaseline(num_classes=5, backbone_pretrained=True,
                                     output_stride=16).to(device)
    n_base = baseline.count_parameters()
    print(f"Baseline trainable parameters: {n_base:,} ({n_base / 1e6:.2f}M)")
    print(f"Multispectral parameters:      {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"Parameter difference:          {n_params - n_base:,} "
          f"({(n_params - n_base) / 1e6:.2f}M, "
          f"{(n_params - n_base) / n_params * 100:.1f}% of full model)")

    # Baseline forward + backward
    torch.manual_seed(42)
    dummy_pan = torch.randn(2, 1, 512, 512).to(device)
    print(f"\nBaseline input shape: {dummy_pan.shape} (pan only)")
    out_base = baseline(dummy_pan)
    print(f"Baseline output shape: {out_base.shape}")
    assert out_base.shape == (2, 5, 512, 512), \
        f"Expected (2,5,512,512), got {out_base.shape}"
    print("Baseline forward pass: OK")

    target = torch.randint(0, 5, (2, 512, 512)).to(device)
    import sys as _sys
    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from losses import CombinedLoss, compute_focal_alpha
    alpha = compute_focal_alpha([501586446, 9719840, 10129403, 5624990, 2038073]).to(device)
    criterion = CombinedLoss(alpha=alpha).to(device)
    loss_b, parts_b = criterion(out_base, target)
    loss_b.backward()
    print(f"Baseline backward pass: OK (loss={parts_b})")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = baseline(dummy_pan)
        mem_b = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"Baseline peak GPU memory: {mem_b:.2f} GB")

    print("\nBaseline architecture summary:")
    print("  Input: 1 channel (pan) z-scored")
    print("  EdgePrior: +1 Sobel edge channel -> 2 channels")
    print("  Backbone: ResNet-34 (ImageNet pretrained, modified conv1 for 2ch)")
    print("  Output Stride: 16 (dilated layer3=2, layer4=4)")
    print("  CBAM x4 + ASPP(6,12,18) + Decoder")
    print("  Output: 5 classes (background, crack, bio, flaking, hollow)")
    print("\n  >> Comparison: baseline (2ch) vs full (8ch) isolates the")
    print("     contribution of 5 multispectral bands to defect detection.")
