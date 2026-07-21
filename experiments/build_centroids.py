# experiments/build_centroids.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.cnn import DefenseCNN
from utils.data_loader import get_cifar10_loaders

def build_centroids():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_cifar10_loaders(batch_size=128)

    model = DefenseCNN(num_classes=10).to(device)
    model.load_state_dict(
        torch.load("checkpoints/cnn_best.pth", map_location=device)
    )
    model.eval()

    # Accumulate features per class
    class_feats = {c: [] for c in range(10)}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            feats  = model.get_features(images)   # (B, D)
            for feat, label in zip(feats, labels):
                class_feats[label.item()].append(feat.cpu())

    # Compute per-class centroid
    centroids = {}
    for c in range(10):
        stacked       = torch.stack(class_feats[c])   # (N, D)
        centroids[c]  = stacked.mean(dim=0)            # (D,)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(centroids, "checkpoints/clean_centroids.pt")
    print(f"[SAVED] checkpoints/clean_centroids.pt")
    print(f"Feature dim: {centroids[0].shape[0]}")

    # Sanity: intra-class vs inter-class distance
    intra, inter = [], []
    for c in range(10):
        for c2 in range(10):
            dist = (centroids[c] - centroids[c2]).norm().item()
            if c == c2:
                intra.append(dist)
            else:
                inter.append(dist)
    print(f"Intra-class centroid dist (should be 0): {np.mean(intra):.4f}")
    print(f"Inter-class centroid dist (should be >0): {np.mean(inter):.4f}")

if __name__ == "__main__":
    build_centroids()