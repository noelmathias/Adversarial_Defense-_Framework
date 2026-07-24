# models/resnet.py

"""
CIFAR-10 optimized ResNet-18.

Key differences from torchvision.models.resnet18():
  - Stem: Conv2d(3→64, 3×3, stride=1)  instead of 7×7 stride=2
  - No MaxPool after stem
  - These two changes preserve 32×32 spatial resolution through the stem,
    which is critical for CIFAR-10.  The ImageNet variant would reduce
    32×32 → 4×4 before the first residual block.

Interface is identical to DefenseCNN:
  - forward(x)       → logits (B, 10)
  - get_features(x)  → (B, 512) pre-classifier embedding
  - __init__(num_classes=10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3×3 conv layers with a residual skip connection."""
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                                padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                                padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Projection shortcut — needed when stride > 1 or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """
    CIFAR-10 ResNet-18.

    Spatial flow (32×32 input):
      Stem    : 32×32 → 32×32  (stride-1 conv, no pool)
      Layer1  : 32×32 → 32×32  (stride 1)
      Layer2  : 32×32 → 16×16  (stride 2)
      Layer3  : 16×16 →  8×8   (stride 2)
      Layer4  :  8×8  →  4×4   (stride 2)
      AvgPool :  4×4  →  1×1
      FC      : 512   → num_classes
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # CIFAR-10 stem — 3×3, stride=1, no maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64,  64,  n=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n=2, stride=2)

        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, in_ch, out_ch, n, stride) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, num_classes). No softmax applied."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns 512-dim pre-classifier embedding.
        Same contract as DefenseCNN.get_features().
        Used by the detection module and any feature-based experiments.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


# Alias — lets code that was written for DefenseCNN continue working
DefenseCNN = ResNet18


if __name__ == "__main__":
    m = ResNet18()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        logits = m(x)
        feats  = m.get_features(x)
    n = sum(p.numel() for p in m.parameters())
    print(f"logits : {tuple(logits.shape)}   expected (4, 10)")
    print(f"feats  : {tuple(feats.shape)}  expected (4, 512)")
    print(f"params : {n:,}")
    assert logits.shape == (4, 10)
    assert feats.shape  == (4, 512)
    print("All assertions passed ✓")