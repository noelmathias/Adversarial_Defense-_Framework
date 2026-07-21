# experiments/calibrate_grad_magnitude.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.cnn import DefenseCNN
from utils.data_loader import get_cifar10_loaders
from attacks import ATTACK_REGISTRY

CLIP_VALUE = 100.0

def get_p90_grad(model, loader, device,
                 attack_fn=None, epsilon=None,
                 n_batches=10, **kwargs):
    model.eval()
    p90s = []
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)
        if attack_fn:
            images, _ = attack_fn(model, images, labels,
                                   epsilon, device, **kwargs)
        images_req = images.clone().detach().requires_grad_(True)
        logits     = model(images_req)
        top1_vals  = logits.max(dim=1).values
        loss       = -top1_vals.sum()
        model.zero_grad()
        loss.backward()
        grad      = images_req.grad.data
        grad_norm = grad.view(grad.size(0), -1).norm(dim=1)
        p90s.append(torch.quantile(grad_norm, 0.90).item())

    raw_p90   = float(np.mean(p90s))
    normed    = float(np.clip(raw_p90 / CLIP_VALUE, 0, 1))
    return raw_p90, normed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loader = get_cifar10_loaders(batch_size=64)
    model = DefenseCNN(num_classes=10).to(device)
    model.load_state_dict(
        torch.load("checkpoints/cnn_best.pth", map_location=device)
    )
    model.eval()

    conditions = [
        ("Clean",        None,   None,  {}),
        ("FGSM ε=0.03",  "fgsm", 0.03, {}),
        ("FGSM ε=0.10",  "fgsm", 0.10, {}),
        ("PGD  ε=0.03",  "pgd",  0.03, {"num_steps": 20}),
        ("PGD  ε=0.10",  "pgd",  0.10, {"num_steps": 20}),
    ]

    print(f"\n{'Condition':<16} {'Raw p90':>10} {'Normed':>10} {'Pass':>6}")
    print("-" * 46)

    normed_vals = {}
    for name, atk, eps, kw in conditions:
        fn = ATTACK_REGISTRY.get(atk) if atk else None
        raw, normed = get_p90_grad(
            model, loader, device,
            attack_fn=fn, epsilon=eps, **kw
        )
        normed_vals[name] = normed
        ok = "✓" if normed > 0.05 else "✗"
        print(f"{name:<16} {raw:>10.3f} {normed:>10.4f} {ok:>6}")

    clean  = normed_vals["Clean"]
    pgd10  = normed_vals["PGD  ε=0.10"]
    ratio  = pgd10 / max(clean, 1e-6)
    print(f"\nSeparation ratio (PGD ε=0.10 / Clean): {ratio:.1f}×")
    print(f"{'✓ Signal healthy' if ratio > 5 else '✗ Still weak — check model confidence range'}")

if __name__ == "__main__":
    main()