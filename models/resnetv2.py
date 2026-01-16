# models/resnetv2.py
# ResNet-50 v2 (pre-activation) implementation in PyTorch.
# We expose a stride-16 feature map (layer3 output) to match patch size 16 on 224x224:
# 224/16 = 14 tokens per side => 14x14 = 196 tokens.

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, mid_ch * self.expansion, kernel_size=1, bias=False)

        self.downsample = downsample

    def forward(self, x):
        # Pre-activation: BN -> ReLU happens before conv
        out = F.relu(self.bn1(x), inplace=True)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(out)  # downsample uses the pre-activated tensor

        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.conv3(F.relu(self.bn3(out), inplace=True))

        out = out + shortcut
        return out


class ResNetV2(nn.Module):
    def __init__(self, block, layers, in_channels=3):
        super().__init__()
        # Sarada-style stem: 7x7 conv stride 2 + 3x3 maxpool stride 2
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-50 stage config: [3,4,6,3]
        self.in_ch = 64
        self.layer1 = self._make_layer(block, mid_ch=64,  blocks=layers[0], stride=1)  # out 256
        self.layer2 = self._make_layer(block, mid_ch=128, blocks=layers[1], stride=2)  # out 512
        self.layer3 = self._make_layer(block, mid_ch=256, blocks=layers[2], stride=2)  # out 1024 (stride 16)
        self.layer4 = self._make_layer(block, mid_ch=512, blocks=layers[3], stride=2)  # out 2048 (stride 32)

        # Final BN for v2 style (not used directly here, but kept for completeness)
        self.bn_final = nn.BatchNorm2d(2048)

    def _make_layer(self, block, mid_ch, blocks, stride):
        out_ch = mid_ch * block.expansion

        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            # v2 uses projection on skip when shape changes
            downsample = nn.Conv2d(self.in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(self.in_ch, mid_ch, stride=stride, downsample=downsample))
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(block(self.in_ch, mid_ch, stride=1, downsample=None))

        return nn.Sequential(*layers)

    @staticmethod
    def resnet50v2(in_channels=3):
        return ResNetV2(PreActBottleneck, [3, 4, 6, 3], in_channels=in_channels)

    def forward_features_stride16(self, x):
        # Returns feature map at stride 16 (layer3 output): (B, 1024, 14, 14) for 224x224 input
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x):
        # Full forward (unused by hybrid, but provided)
        x = self.forward_features_stride16(x)
        x = self.layer4(x)
        x = F.relu(self.bn_final(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x
