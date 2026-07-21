import torch
import torch.nn as nn
import torch.nn.functional as F

class DefenseCNN(nn.Module):
    """
    Baseline CNN for CIFAR-10. Designed to be:
    - Strong enough to be a meaningful attack target (target: >75% clean acc)
    - Simple enough to be transparent for research explanation
    - Upgradeable to ResNet in Phase 2 without changing surrounding code
    """
    def __init__(self, num_classes=10):
        super(DefenseCNN, self).__init__()

        # Block 1: 3 → 64 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        # Block 2: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        # Block 3: 128 → 256 channels
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)

        # Classifier head: 256 * 4 * 4 = 4096
        self.fc1  = nn.Linear(256 * 4 * 4, 512)
        self.bn7  = nn.BatchNorm1d(512)
        self.drop4 = nn.Dropout(0.5)
        self.fc2  = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.drop3(self.pool3(x))

        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.drop4(x)
        x = self.fc2(x)              # raw logits (no softmax — CrossEntropyLoss handles it)
        return x

    def get_features(self, x):
    # adjust based on your architecture
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.drop3(self.pool3(x))

        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.bn7(self.fc1(x))) #penultimate layer features
        return x                    # (B, D)
