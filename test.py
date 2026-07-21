'''from utils.data_loader import get_cifar10_loaders

train_loader, test_loader = get_cifar10_loaders()

for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break '''

import torch
from models.cnn import DefenseCNN

model = DefenseCNN()
model.eval()  # set to evaluation mode
x = torch.randn(1, 3, 32, 32)  # batch of 1 image
output = model(x)
print("Output shape:", output.shape)  # should be [1, 10]