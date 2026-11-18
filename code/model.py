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

  6. Output: (B, 4, H, W) logits  [background, crack, bio, flaking]

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

    def __init__(self, num_classes=4, backbone_pretrained=True, output_stride=16):
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

    def __init__(self, num_classes=4, backbone_pretrained=True, output_stride=16):
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


# ===========================================================================
# 5.5. RGB 基线 (方案A 新增) — b1/b2/b3 三通道可见光 DeepLabV3+
#   与 DeepLabV3PlusBaseline 的唯一差异 = 输入模态 (3ch 彩色 vs 1ch 灰度).
#   用于四模型对比中的 RGB 基线, 保证与 code/ 目录的 RGB 基线完全一致.
# ===========================================================================
class EdgePriorModuleRGB(nn.Module):
    """RGB 基线的 Sobel 边缘先验 (方案A 新增).

    Input: (B, 3, H, W) [b1, b2, b3] -> Output: (B, 4, H, W)
    用 BT.601 灰度权重 (0.299R + 0.587G + 0.114B) 合成灰度, 再算 Sobel 边缘.
    与 pan 基线的 EdgePriorModulePanOnly 语义对称: 都加"1 个灰度边缘通道",
    使 RGB 基线 vs pan 基线的唯一变量 = 彩色分谱 vs 全谱灰度.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        # BT.601 灰度权重 (与 torchvision/常见视觉库一致)
        self.register_buffer("gray_w", torch.tensor([0.299, 0.587, 0.114],
                                                     dtype=torch.float32).view(1, 3, 1, 1))

    def _sobel(self, x):
        gx = F.conv2d(F.pad(x, (1, 1, 1, 1)), self.sobel_x)
        gy = F.conv2d(F.pad(x, (1, 1, 1, 1)), self.sobel_y)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, x):
        # x: (B, 3, H, W) [b1, b2, b3]; 合成灰度再算边缘
        gray = F.conv2d(x, self.gray_w)  # (B, 1, H, W)
        edge = self._sobel(gray)          # (B, 1, H, W)
        return torch.cat([x, edge], dim=1)  # (B, 4, H, W)


class DeepLabV3PlusRGB(nn.Module):
    """RGB 基线模型: b1/b2/b3 三通道可见光输入的 DeepLabV3+ (方案A 新增).

    与 DeepLabV3PlusBaseline 的唯一差异:
      - 输入: 3 通道 RGB (b1/b2/b3, 反射率) vs 1 通道 pan (DN)
      - EdgePrior: RGB 灰度 Sobel 边缘 -> 4 通道 (vs pan 边缘 -> 2 通道)
      - conv1: 4 通道输入 (vs 2 通道)

    关键初始化优势: 前 3 通道可直接用 ImageNet 预训练 RGB 权重对齐
    (b1/b2/b3 即蓝/绿/红), 比 pan 基线的"灰度均值初始化"更合理.

    其余完全一致: ResNet-34 backbone, OS=16, CBAM x4, ASPP(6,12,18), Decoder.

    Input:  (B, 3, H, W) float32 z-scored [b1, b2, b3]
    Output: (B, num_classes, H, W) logits
    """

    def __init__(self, num_classes=4, backbone_pretrained=True, output_stride=16):
        super().__init__()
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.input_mode = "rgb"
        self.edge_prior = EdgePriorModuleRGB()

        # ResNet-34 backbone
        weights = tvm.ResNet34_Weights.DEFAULT if backbone_pretrained else None
        resnet = tvm.resnet34(weights=weights)

        # Modify first conv: 3ch -> 4ch (RGB + gray edge)
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if backbone_pretrained:
            with torch.no_grad():
                # 前 3 通道直接用 ImageNet 预训练 RGB 权重 (b1/b2/b3 = B/G/R)
                new_conv.weight[:, 0:3] = old_conv.weight
                # 第 4 通道 (灰度边缘) 用 RGB 灰度均值小初始化
                mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                new_conv.weight[:, 3:4] = mean_rgb * 0.1
        resnet.conv1 = new_conv

        # OS=16 dilated backbone (与 DeepLabV3Plus / Baseline 完全一致)
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
        x = self.edge_prior(x)                 # (B, 4, H, W)
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


# ===========================================================================
# 6. Intermediate Fusion (feature-level) — 中期融合模型
#    设计文档: lab/INTERMEDIATE_FUSION_PLAN.md
#    原则: 不修改上方任何现有类; 早期融合/基线代码冻结, 保证可精确复现.
# ===========================================================================
class EdgePriorModuleSplit(nn.Module):
    """中期融合专用: 模态拆分 + 各自 Sobel 边缘先验.

    Input:  (B, 6, H, W)  [b1, b2, b3, b4, b5, pan]  (与早期融合完全相同的输入)
    Output: tuple (vis, spec)
      vis  = (B, 2, H, W)  [pan, edge_pan]          -- 与基线模型相同的边缘定义
      spec = (B, 6, H, W)  [b1..b5, edge_b2]         -- 与早期融合相同的 b2 边缘定义

    Sobel 索引与早期融合版严格一致: vis 用 channel 5 (pan), spec 用 channel 1 (b2).
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
        # x: (B, 6, H, W); ch5=pan, ch1=b2 (索引与早期融合 EdgePriorModule 一致)
        pan = x[:, 5:6]
        bands = x[:, 0:5]
        edge_pan = self._sobel(pan)
        edge_b2 = self._sobel(x[:, 1:2])
        vis = torch.cat([pan, edge_pan], dim=1)        # (B, 2, H, W)
        spec = torch.cat([bands, edge_b2], dim=1)      # (B, 6, H, W)
        return vis, spec


class FusionBlock(nn.Module):
    """逐级特征融合: concat -> 1x1 conv 压缩 -> BN -> ReLU -> CBAM.

    融合算子选 concat+1x1 (可学习加权, 稳定, 参数可控);
    CBAM 施加在融合后的联合特征上, 与早期融合版"每级一个 CBAM"数量一致.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)

    def forward(self, x_vis, x_spec):
        return self.cbam(self.relu(self.bn(self.conv(
            torch.cat([x_vis, x_spec], dim=1)))))


class NarrowBasicBlock(nn.Module):
    """ResNet BasicBlock, 通道宽度可缩放 (仅用于 spec_width_scale<1 的消融实验).

    结构与 torchvision BasicBlock 完全一致, 仅 planes 由调用方给定.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class _ResNet34Twin(nn.Module):
    """最小 ResNet-34 骨干, 属性命名与 torchvision 一致 (conv1/bn1/layer1-4).

    用途: spec_width_scale<1 的消融实验. 拓扑与 torchvision resnet34 完全相同,
    仅各层通道按 width_scale 缩窄 (四舍五入到 8 的倍数); 预训练权重由
    _load_pretrained_sliced() 沿输出通道维切片加载 (channel pruning 常规做法).
    """
    def __init__(self, in_chans, width_scale):
        super().__init__()
        assert 0.0 < width_scale <= 1.0, "width_scale 应在 (0, 1]"

        def ch(c):  # 缩放并取整到 8 的倍数
            return max(8, int(round(c * width_scale / 8)) * 8)

        c1, c2, c3, c4 = ch(64), ch(128), ch(256), ch(512)
        self.conv1 = nn.Conv2d(in_chans, c1, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)

        def make_layer(in_p, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or in_p != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(in_p, planes, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes))
            layers = [NarrowBasicBlock(in_p, planes, stride, downsample)]
            for _ in range(1, blocks):
                layers.append(NarrowBasicBlock(planes, planes))
            return nn.Sequential(*layers)

        self.layer1 = make_layer(c1, c1, 3)
        self.layer2 = make_layer(c1, c2, 4, stride=2)
        self.layer3 = make_layer(c2, c3, 6, stride=2)
        self.layer4 = make_layer(c3, c4, 3, stride=2)
        self.out_channels = c4  # 供融合层计算通道

    def forward(self, x):
        # 骨干由 DeepLabV3PlusIntermediateFusion 逐层拆用 (stem_s/layer1_s-4_s),
        # 本类自身不做端到端前向.
        raise NotImplementedError(
            "_ResNet34Twin is a backbone container; use its stem/layer modules")


def _load_pretrained_sliced(narrow_model, scale, in_chans, mean_init):
    """从 ImageNet 预训练 ResNet-34 权重做通道维切片, 初始化窄版骨干.

    规则: conv weight (O,I,k,k) -> 前 ceil(O*scale) 行 / 前 ceil(I*scale) 列;
          BN 向量取前 c 个. conv1/bn1 不加载 (输入通道数不同), 由调用方手动初始化.
    说明: 切片初始化保留了预训练特征的大部分能量 (channel pruning 常规做法),
          但消融版仍属"参数量敏感性参考", 不作为主结论 (见计划文档 2.3 节).
    """
    resnet = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT)
    std_sd = resnet.state_dict()
    narrow_sd = narrow_model.state_dict()
    new_sd = {}
    for k, v in std_sd.items():
        if k.startswith("conv1") or k.startswith("bn1") or "fc" in k:
            continue  # conv1/bn1 由调用方初始化; resnet34 无 fc 需求
        if k not in narrow_sd:
            continue
        tgt_shape = narrow_sd[k].shape
        if v.ndim == 4:
            new_sd[k] = v[:tgt_shape[0], :tgt_shape[1]].clone()
        elif v.ndim == 1:
            new_sd[k] = v[:tgt_shape[0]].clone()
        # num_batches_tracked (0-dim) 跳过
    missing = narrow_model.load_state_dict(new_sd, strict=False)
    return missing


class DeepLabV3PlusIntermediateFusion(nn.Module):
    """中期融合 (特征级) DeepLabV3+ : 对称双塔 ResNet-34 + 逐级特征融合.

    与早期融合版 (DeepLabV3Plus) 的唯一差异 = 融合方式:
      - 早期融合: 6ch 输入端拼接 (+2 Sobel = 8ch) -> 单塔 ResNet-34
      - 中期融合 (本类): 输入 (B,6,H,W) 不变, 模型内部拆分模态 ->
          vis 塔 (pan + edge_pan, 2ch) 与 spec 塔 (b1-b5 + edge_b2, 6ch)
          各自独立提特征 -> layer1-4 末端逐级 concat+1x1 压缩融合 ->
          融合特征 (含 CBAM) 进入共享 ASPP + Decoder.

    其余全部与早期融合版一致: ResNet-34 OS=16, ASPP(6,12,18),
    balance score 兼容的输出接口 (B, num_classes, H, W) logits.

    Args:
        num_classes: 5
        backbone_pretrained: ImageNet 预训练
        output_stride: 16
        spec_width_scale: 光谱塔通道宽度缩放 (1.0=满宽主实验; 0.5=消融实验,
            将总参数量拉回与早期融合 ~26.9M 相近, 回应"容量混淆"质疑;
            <1.0 时预训练权重沿通道维切片初始化).

    Input:  (B, 6, H, W) float32 z-scored  -- 数据管道与早期融合完全相同
    Output: (B, num_classes, H, W) logits
    """

    def __init__(self, num_classes=4, backbone_pretrained=True, output_stride=16,
                 spec_width_scale=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.input_mode = "multispectral_intermediate_fusion"
        self.spec_width_scale = spec_width_scale
        self.edge_prior = EdgePriorModuleSplit()

        weights = tvm.ResNet34_Weights.DEFAULT if backbone_pretrained else None

        # ── 可见光塔 (2ch: pan + edge_pan), 满宽, 初始化与基线一致 ──
        resnet_v = tvm.resnet34(weights=weights)
        old_conv = resnet_v.conv1
        conv_v = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if backbone_pretrained:
            with torch.no_grad():
                mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
                conv_v.weight[:, 0:1] = mean_rgb       # pan
                conv_v.weight[:, 1:2] = mean_rgb * 0.1  # edge: 小初始化 (同基线)
        resnet_v.conv1 = conv_v

        # ── 光谱塔 (6ch: b1-b5 + edge_b2) ──
        if spec_width_scale == 1.0:
            # 满宽: 与早期融合相同的构建路径 (torchvision), 初始化策略一致
            resnet_s = tvm.resnet34(weights=weights)
            old_conv_s = resnet_s.conv1
            conv_s = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
            if backbone_pretrained:
                with torch.no_grad():
                    mean_rgb_s = old_conv_s.weight.mean(dim=1, keepdim=True)
                    # b1-b5: RGB 均值缩放 (同早期融合 8ch 版对额外 5 通道的 /5 策略)
                    conv_s.weight[:, :5] = mean_rgb_s.expand(-1, 5, -1, -1) / 5.0
                    # edge_b2: 小初始化 (同基线 edge 通道的 *0.1 策略)
                    conv_s.weight[:, 5:6] = mean_rgb_s * 0.1
            resnet_s.conv1 = conv_s
            s_ch = (64, 128, 256, 512)
        else:
            # 消融: 窄版 + 预训练切片初始化
            resnet_s = _ResNet34Twin(6, spec_width_scale)
            if backbone_pretrained:
                _load_pretrained_sliced(resnet_s, spec_width_scale, 6, None)
                with torch.no_grad():
                    # conv1/bn1 手动初始化 (切片函数明确跳过这两层);
                    # 注意切输出通道维到窄版宽度
                    std_resnet = tvm.resnet34(weights=weights)
                    o = resnet_s.conv1.weight.shape[0]  # 窄版输出通道数
                    mean_rgb_s = std_resnet.conv1.weight[:o].mean(
                        dim=1, keepdim=True)              # (o, 1, 7, 7)
                    resnet_s.conv1.weight[:, :5] = \
                        mean_rgb_s.expand(-1, 5, -1, -1) / 5.0
                    resnet_s.conv1.weight[:, 5:6] = mean_rgb_s * 0.1
                    resnet_s.bn1.weight.data.copy_(std_resnet.bn1.weight[:o])
                    resnet_s.bn1.bias.data.copy_(std_resnet.bn1.bias[:o])
                    resnet_s.bn1.running_mean.data.copy_(
                        std_resnet.bn1.running_mean[:o])
                    resnet_s.bn1.running_var.data.copy_(
                        std_resnet.bn1.running_var[:o])
            c1 = resnet_s.layer1[0].conv1.out_channels if hasattr(
                resnet_s.layer1[0], "conv1") else 64
            s_ch = (c1,
                    resnet_s.layer2[0].conv2.out_channels,
                    resnet_s.layer3[0].conv2.out_channels,
                    resnet_s.layer4[0].conv2.out_channels)

        # ── OS=16: 两塔都转 dilated ──
        if output_stride == 16:
            self._convert_to_dilated(resnet_v.layer3, dilation=2)
            self._convert_to_dilated(resnet_v.layer4, dilation=4)
            self._convert_to_dilated(resnet_s.layer3, dilation=2)
            self._convert_to_dilated(resnet_s.layer4, dilation=4)

        # ── 拆分骨干 (两塔) ──
        self.stem_v = nn.Sequential(resnet_v.conv1, resnet_v.bn1,
                                    resnet_v.relu, resnet_v.maxpool)
        self.layer1_v = resnet_v.layer1  # 64,  /4
        self.layer2_v = resnet_v.layer2  # 128, /8
        self.layer3_v = resnet_v.layer3  # 256, /16 (dilated)
        self.layer4_v = resnet_v.layer4  # 512, /16 (dilated)

        self.stem_s = nn.Sequential(resnet_s.conv1, resnet_s.bn1,
                                    resnet_s.relu if hasattr(resnet_s, "relu")
                                    else nn.ReLU(inplace=True),
                                    nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1_s = resnet_s.layer1
        self.layer2_s = resnet_s.layer2
        self.layer3_s = resnet_s.layer3
        self.layer4_s = resnet_s.layer4

        # ── 逐级融合 (layer1-4 末端; CBAM 在融合后, 全模型共 4 个, 与早期融合一致) ──
        v_ch = (64, 128, 256, 512)
        self.fuse1 = FusionBlock(v_ch[0] + s_ch[0], 64)
        self.fuse2 = FusionBlock(v_ch[1] + s_ch[1], 128)
        self.fuse3 = FusionBlock(v_ch[2] + s_ch[2], 256)
        self.fuse4 = FusionBlock(v_ch[3] + s_ch[3], 512)

        # ── 共享头: ASPP + Decoder (与早期融合版完全一致) ──
        self.aspp = ASPP(512, out_channels=256, atrous_rates=(6, 12, 18))
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
        """与 DeepLabV3Plus._convert_to_dilated 完全一致 (对两塔复用)."""
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
        vis, spec = self.edge_prior(x)          # (B,2,H,W), (B,6,H,W)
        v = self.stem_v(vis)
        s = self.stem_s(spec)
        l1v, l1s = self.layer1_v(v), self.layer1_s(s)
        f1 = self.fuse1(l1v, l1s)               # (B, 64, H/4)
        l2v, l2s = self.layer2_v(l1v), self.layer2_s(l1s)
        f2 = self.fuse2(l2v, l2s)               # (B, 128, H/8)
        l3v, l3s = self.layer3_v(l2v), self.layer3_s(l2s)
        f3 = self.fuse3(l3v, l3s)               # (B, 256, H/16)
        l4v, l4s = self.layer4_v(l3v), self.layer4_s(l3s)
        f4 = self.fuse4(l4v, l4s)               # (B, 512, H/16)
        x = self.aspp(f4)                       # (B, 256, H/16)
        x = F.interpolate(x, size=f1.shape[2:],
                          mode="bilinear", align_corners=False)
        x = torch.cat([x, self.low_level_proj(f1)], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 8. 光谱指数增强模型 (E2/E3, 阶段二新增)
# ---------------------------------------------------------------------------
class EdgePriorModuleIndices(nn.Module):
    """E2 (ms + indices) 的 Sobel 边缘先验.

    Input: (B, 10, H, W) [b1,b2,b3,b4,b5,pan, ndi_b4b5, ndi_b2b3, ndi_b2b4,
            ndi_b2b5] -> Output: (B, 12, H, W)
    Sobel 语义与 EdgePriorModule 完全一致 (pan 边缘 + b2 边缘),
    指数通道不参与灰度合成 (指数是对比度信息, 不是亮度).
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
        # x: (B, 10, H, W); ch5 = pan, ch1 = b2 (与 EdgePriorModule 相同语义)
        edge_pan = self._sobel(x[:, 5:6])
        edge_b2 = self._sobel(x[:, 1:2])
        return torch.cat([x, edge_pan, edge_b2], dim=1)  # (B, 12, H, W)


class DeepLabV3PlusIndices(nn.Module):
    """E2: ms 6 通道 + 4 光谱指数 = 10 通道输入的 DeepLabV3+ (阶段二新增).

    科学问题: 4 个精选归一化差值指数作为显式通道, 是否在原始波段之上
    带来额外增益? (对比对象: train.py 的 DeepLabV3Plus, test mIoU 0.6214)
    (ndi_b1b5 已因与 b2b5/b2b4 冗余剔除, 见 build_input_tensor.py 注释)

    conv1 通道映射 (12ch): [b1,b2,b3,b4,b5,pan, ndi x4, edge_pan, edge_b2]
      - ch0-2: ImageNet RGB 预训练权重 (与 DeepLabV3Plus 相同)
      - ch3-5: mean_rgb/5 (b4/b5/pan, 与 DeepLabV3Plus 的额外通道初始化相同)
      - ch6-9: 指数通道 **零初始化** → t=0 时模型函数与 DeepLabV3Plus 完全
        一致, 训练后的任何性能差异均可干净归因于学到的指数通道权重
      - ch10-11: mean_rgb/5 (边缘通道, 与 DeepLabV3Plus 相同)

    Input:  (B, 10, H, W) float32 z-scored [b1..b5, pan, ndi x4]
    Output: (B, num_classes, H, W) logits
    """

    def __init__(self, num_classes=4, backbone_pretrained=True, output_stride=16):
        super().__init__()
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.input_mode = "ms_indices"
        self.edge_prior = EdgePriorModuleIndices()

        weights = tvm.ResNet34_Weights.DEFAULT if backbone_pretrained else None
        resnet = tvm.resnet34(weights=weights)

        # Modify first conv: 3ch -> 12ch
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if backbone_pretrained:
            with torch.no_grad():
                new_conv.weight[:, 0:3] = old_conv.weight
                mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
                # b4, b5, pan (ch3-5) 与基座模型相同: mean/5
                new_conv.weight[:, 3:6] = mean_rgb.expand(-1, 3, -1, -1) / 5.0
                # 边缘通道 (ch10-11) 与基座模型相同: mean/5
                new_conv.weight[:, 10:12] = mean_rgb.expand(-1, 2, -1, -1) / 5.0
        # 4 个指数通道 (ch6-9): 零初始化 (无条件执行 — 因果零起点, 见类 docstring)
        with torch.no_grad():
            new_conv.weight[:, 6:10] = 0.0
        resnet.conv1 = new_conv

        if output_stride == 16:
            self._convert_to_dilated(resnet.layer3, dilation=2)
            self._convert_to_dilated(resnet.layer4, dilation=4)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        self.aspp = ASPP(512, out_channels=256, atrous_rates=(6, 12, 18))

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1))

    _convert_to_dilated = staticmethod(DeepLabV3Plus._convert_to_dilated)

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.edge_prior(x)                 # (B, 12, H, W)
        x = self.stem(x)
        low_level = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(low_level))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.aspp(x)
        x = F.interpolate(x, size=low_level.shape[2:],
                          mode="bilinear", align_corners=False)
        x = torch.cat([x, self.low_level_proj(low_level)], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EdgePriorModuleRGBIndices(nn.Module):
    """E3 (RGB + indices) 的 Sobel 边缘先验.

    Input: (B, 7, H, W) [b1,b2,b3, ndi x4] -> Output: (B, 8, H, W)
    灰度 Sobel 只从 b1/b2/b3 合成 (BT.601), 与 EdgePriorModuleRGB 语义一致.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("gray_w", torch.tensor([0.299, 0.587, 0.114],
                                                     dtype=torch.float32).view(1, 3, 1, 1))

    def _sobel(self, x):
        gx = F.conv2d(F.pad(x, (1, 1, 1, 1)), self.sobel_x)
        gy = F.conv2d(F.pad(x, (1, 1, 1, 1)), self.sobel_y)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, x):
        # x: (B, 7, H, W); 前 3 通道 = b1/b2/b3
        gray = F.conv2d(x[:, 0:3], self.gray_w)  # (B, 1, H, W)
        edge = self._sobel(gray)
        return torch.cat([x[:, 0:3], edge, x[:, 3:7]], dim=1)  # (B, 8, H, W)


class DeepLabV3PlusRGBIndices(nn.Module):
    """E3: RGB 3 通道 + 4 光谱指数 = 7 通道输入的 DeepLabV3+ (阶段二新增).

    科学问题: 可见光 RGB + 4 个光谱指数能否逼近完整多光谱 (6ch ms 模型)?
    若能, 说明光谱增益主要由少数归一化差值指数承载 (传感器可降配卖点).
    (ndi_b1b5 已因与 b2b5/b2b4 冗余剔除)

    conv1 通道映射 (8ch): [b1,b2,b3, edge_gray, ndi x4]
      - ch0-2: ImageNet RGB 预训练 (与 DeepLabV3PlusRGB 相同)
      - ch3: mean_rgb*0.1 (灰度边缘, 与 DeepLabV3PlusRGB 相同)
      - ch4-7: 指数通道 **零初始化** → t=0 与 DeepLabV3PlusRGB 函数一致

    Input:  (B, 7, H, W) float32 z-scored [b1, b2, b3, ndi x4]
    Output: (B, num_classes, H, W) logits
    """

    def __init__(self, num_classes=4, backbone_pretrained=True, output_stride=16):
        super().__init__()
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.input_mode = "rgb_indices"
        self.edge_prior = EdgePriorModuleRGBIndices()

        weights = tvm.ResNet34_Weights.DEFAULT if backbone_pretrained else None
        resnet = tvm.resnet34(weights=weights)

        old_conv = resnet.conv1
        new_conv = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if backbone_pretrained:
            with torch.no_grad():
                new_conv.weight[:, 0:3] = old_conv.weight
                mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:4] = mean_rgb * 0.1   # 灰度边缘, 与 RGB 基线相同
        # 4 个指数通道 (ch4-7): 零初始化 (无条件执行 — 因果零起点, 见类 docstring)
        with torch.no_grad():
            new_conv.weight[:, 4:8] = 0.0
        resnet.conv1 = new_conv

        if output_stride == 16:
            self._convert_to_dilated(resnet.layer3, dilation=2)
            self._convert_to_dilated(resnet.layer4, dilation=4)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        self.aspp = ASPP(512, out_channels=256, atrous_rates=(6, 12, 18))

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1))

    _convert_to_dilated = staticmethod(DeepLabV3Plus._convert_to_dilated)

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.edge_prior(x)                 # (B, 9, H, W)
        x = self.stem(x)
        low_level = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(low_level))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.aspp(x)
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

    # v5.2: 自检块的类别数一律从 config 取, 不再写死 5 (v4 hollow 时代残留)
    from config import NUM_CLASSES, CLASS_NAMES, CLASS_PIXEL_COUNTS

    model = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone_pretrained=True,
                          output_stride=16).to(device)
    n_params = model.count_parameters()
    print(f"Trainable parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Forward pass test (keep grad for backward test)
    torch.manual_seed(42)
    dummy_input = torch.randn(2, 6, 512, 512).to(device)
    print(f"Input shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, NUM_CLASSES, 512, 512), \
        f"Expected (2,{NUM_CLASSES},512,512), got {output.shape}"
    print("Forward pass: OK")

    # Test backward
    target = torch.randint(0, NUM_CLASSES, (2, 512, 512)).to(device)
    import sys
    from losses import CombinedLoss, compute_focal_alpha
    alpha = compute_focal_alpha(
        [CLASS_PIXEL_COUNTS[c] for c in CLASS_NAMES]).to(device)
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
    print(f"  Decoder: fuse ASPP + low-level -> {NUM_CLASSES}-class logits")
    print("  Output: 4 classes (background, crack, bio, flaking)")

    # ── 基线模型测试 (对比实验) ──────────────────────────────
    print("\n" + "=" * 64)
    print("Baseline Model (pan-only) Test")
    print("=" * 64)

    baseline = DeepLabV3PlusBaseline(num_classes=NUM_CLASSES, backbone_pretrained=True,
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
    assert out_base.shape == (2, NUM_CLASSES, 512, 512), \
        f"Expected (2,{NUM_CLASSES},512,512), got {out_base.shape}"
    print("Baseline forward pass: OK")

    target = torch.randint(0, NUM_CLASSES, (2, 512, 512)).to(device)
    import sys as _sys
    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from losses import CombinedLoss, compute_focal_alpha
    alpha = compute_focal_alpha(
        [CLASS_PIXEL_COUNTS[c] for c in CLASS_NAMES]).to(device)
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
    print("  Output: 4 classes (background, crack, bio, flaking)")
    print("\n  >> Comparison: baseline (2ch) vs full (8ch) isolates the")
    print("     contribution of 5 multispectral bands to defect detection.")

    # ── 中期融合模型测试 (特征级融合, 对比实验用) ────────────
    print("\n" + "=" * 64)
    print("Intermediate Fusion Model (dual-tower, feature-level) Test")
    print("=" * 64)

    inter = DeepLabV3PlusIntermediateFusion(
        num_classes=NUM_CLASSES, backbone_pretrained=True, output_stride=16,
        spec_width_scale=1.0).to(device)
    n_inter = inter.count_parameters()
    print(f"Intermediate fusion params: {n_inter:,} ({n_inter / 1e6:.2f}M)")
    print(f"Early fusion params:        {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"Baseline params:            {n_base:,} ({n_base / 1e6:.2f}M)")

    # 前向 + 反向 (输入接口与早期融合完全相同: 6ch)
    torch.manual_seed(42)
    dummy_6ch = torch.randn(2, 6, 512, 512).to(device)
    out_inter = inter(dummy_6ch)
    print(f"\nIntermediate input:  {tuple(dummy_6ch.shape)} (6ch, same as early fusion)")
    print(f"Intermediate output: {tuple(out_inter.shape)}")
    assert out_inter.shape == (2, NUM_CLASSES, 512, 512), \
        f"Expected (2,{NUM_CLASSES},512,512), got {out_inter.shape}"
    print("Intermediate forward pass: OK")

    target_i = torch.randint(0, NUM_CLASSES, (2, 512, 512)).to(device)
    loss_i, parts_i = criterion(out_inter, target_i)
    loss_i.backward()
    # 双塔梯度检查: 两塔都应有梯度流
    g_v = inter.layer4_v[0].conv1.weight.grad
    g_s = inter.layer4_s[0].conv1.weight.grad
    assert g_v is not None and g_v.abs().sum() > 0, "vis tower has no grad"
    assert g_s is not None and g_s.abs().sum() > 0, "spec tower has no grad"
    print(f"Intermediate backward pass: OK (loss={parts_i['total']:.4f}, "
          f"dual-tower grads OK)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = inter(dummy_6ch)
        mem_i = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"Intermediate peak GPU memory (fwd only): {mem_i:.2f} GB")

    # 消融模式: spec_width_scale=0.5 (参数量拉回早期融合附近)
    inter_half = DeepLabV3PlusIntermediateFusion(
        num_classes=NUM_CLASSES, backbone_pretrained=True, output_stride=16,
        spec_width_scale=0.5).to(device)
    n_half = inter_half.count_parameters()
    print(f"\nAblation (spec_width_scale=0.5) params: {n_half:,} "
          f"({n_half / 1e6:.2f}M)  [early fusion: {n_params / 1e6:.2f}M]")
    out_half = inter_half(dummy_6ch)
    assert out_half.shape == (2, NUM_CLASSES, 512, 512), \
        f"Expected (2,{NUM_CLASSES},512,512), got {out_half.shape}"
    print("Ablation forward pass: OK")

    print("\nIntermediate fusion architecture summary:")
    print("  Input: 6 channels (b1-b5 + pan) z-scored  -- identical pipeline to early fusion")
    print("  Split: vis tower (pan + edge_pan = 2ch) | spec tower (b1-b5 + edge_b2 = 6ch)")
    print("  Towers: dual ResNet-34 (ImageNet pretrained, OS=16 dilated)")
    print("  Fusion: layer1-4 逐级 concat + 1x1 conv 压缩 + CBAM (feature-level)")
    print("  Head:   ASPP(6,12,18) 吃 fused_4; decoder low-level 吃 fused_1")
    print("  Output: 4 classes (background, crack, bio, flaking)")
    print("\n  >> Comparison: early (8ch concat input) vs intermediate (dual-tower")
    print("     feature fusion) isolates the effect of FUSION STRATEGY, with")
    print("     identical data pipeline, loss, training schedule and seed.")

    # ── RGB 基线模型测试 (方案A 新增, 四模型对比用) ──────────
    print("\n" + "=" * 64)
    print("RGB Baseline Model (b1/b2/b3 visible-light) Test")
    print("=" * 64)

    rgb = DeepLabV3PlusRGB(num_classes=NUM_CLASSES, backbone_pretrained=True,
                           output_stride=16).to(device)
    n_rgb = rgb.count_parameters()
    print(f"RGB baseline params: {n_rgb:,} ({n_rgb / 1e6:.2f}M)")
    print(f"pan baseline params: {n_base:,} ({n_base / 1e6:.2f}M)  "
          f"(diff {(n_rgb - n_base) / 1e6:.2f}M)")

    torch.manual_seed(42)
    dummy_rgb = torch.randn(2, 3, 512, 512).to(device)
    print(f"\nRGB input shape: {tuple(dummy_rgb.shape)} (b1/b2/b3)")
    out_rgb = rgb(dummy_rgb)
    assert out_rgb.shape == (2, NUM_CLASSES, 512, 512), \
        f"Expected (2,{NUM_CLASSES},512,512), got {out_rgb.shape}"
    print("RGB forward pass: OK")

    target_rgb = torch.randint(0, NUM_CLASSES, (2, 512, 512)).to(device)
    loss_rgb, parts_rgb = criterion(out_rgb, target_rgb)
    loss_rgb.backward()
    print(f"RGB backward pass: OK (loss={parts_rgb['total']:.4f})")

    print("\nRGB baseline architecture summary:")
    print("  Input: 3 channels (b1/b2/b3 = B/G/R, reflectance) z-scored")
    print("  EdgePrior: +1 gray Sobel edge channel -> 4 channels")
    print("  Backbone: ResNet-34 (ImageNet pretrained, conv1 3ch->4ch, RGB weights aligned)")
    print("  CBAM x4 + ASPP(6,12,18) + Decoder")
    print("  Output: 4 classes (background, crack, bio, flaking)")
    print("\n  >> Four-model comparison (pan / rgb / early / intermediate):")
    print("     pan (2ch)  vs rgb (4ch)  isolates COLOR vs GRAYSCALE within visible light")
    print("     rgb (4ch)  vs early(8ch) isolates SPECTRAL gain (b4/b5 beyond visible)")
    print("     early (8ch) vs inter(dual) isolates FUSION STRATEGY effect")
