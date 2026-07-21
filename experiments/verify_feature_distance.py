# experiments/verify_feature_distance.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.cnn import DefenseCNN
from utils.data_loader import get_cifar10_loaders
from attacks import ATTACK_REGISTRY

CLIP_VALUE = 60.0

def get_feat_dist(model, loader, device, centroids,
                  attack_fn=None, epsilon=None,
                  n_batches=10, **kwargs):
    model.eval()
    dists = []
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches: break
        images = images.to(device)
        if attack_fn:
            images, _ = attack_fn(model, images,
                                   labels.to(device),
                                   epsilon, device, **kwargs)
        with torch.no_grad():
            feats = model.get_features(images)
            preds = model(images).argmax(dim=1)
        for j in range(images.size(0)):
            c    = preds[j].item()
            dist = (feats[j] - centroids[c].to(device)).norm().item()
            dists.append(dist)
    raw    = float(np.mean(dists))
    normed = float(np.clip(raw / CLIP_VALUE, 0, 1))
    return raw, normed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loader = get_cifar10_loaders(batch_size=64)
    model = DefenseCNN(num_classes=10).to(device)
    model.load_state_dict(
        torch.load("checkpoints/cnn_best.pth", map_location=device)
    )
    model.eval()
    centroids = torch.load("checkpoints/clean_centroids.pt",
                           map_location=device)

    conditions = [
        ("Clean",        None,   None,  {}),
        ("FGSM ε=0.03",  "fgsm", 0.03, {}),
        ("FGSM ε=0.10",  "fgsm", 0.10, {}),
        ("PGD  ε=0.03",  "pgd",  0.03, {"num_steps": 20}),
        ("PGD  ε=0.10",  "pgd",  0.10, {"num_steps": 20}),
    ]

    print(f"\n{'Condition':<16} {'Raw dist':>10} "
          f"{'Normed':>10} {'Monotonic':>10}")
    print("-" * 50)

    prev, results = 0, {}
    for name, atk, eps, kw in conditions:
        fn = ATTACK_REGISTRY.get(atk) if atk else None
        raw, normed = get_feat_dist(
            model, loader, device, centroids,
            attack_fn=fn, epsilon=eps, **kw
        )
        results[name] = normed
        ok = "✓" if normed >= prev else "✗"
        prev = normed
        print(f"{name:<16} {raw:>10.3f} {normed:>10.4f} {ok:>10}")

    ratio = results["PGD  ε=0.10"] / max(results["Clean"], 1e-6)
    print(f"\nSeparation (PGD ε=0.10 / Clean): {ratio:.1f}×")
    print("✓ Signal healthy" if ratio > 5 else "✗ Check get_features() output")

if __name__ == "__main__":
    main()