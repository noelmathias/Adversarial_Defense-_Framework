import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from models.cnn import DefenseCNN
from utils.data_loader import get_cifar10_loaders
from attacks import ATTACK_REGISTRY
from detection.detector import AdversarialDetector


def collect(detector, model, loader, device,
            attack_type=None, epsilon=None, n_batches=15):
    scores = {k: [] for k in
              ["confidence","sensitivity","instability",
               "grad_magnitude","risk_score"]}
    model.eval()
    attack_fn = ATTACK_REGISTRY.get(attack_type) if attack_type else None

    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)
        if attack_fn:
            kwargs = {"num_steps": 20} if attack_type == "pgd" else {}
            images, _ = attack_fn(model, images, labels,
                                   epsilon, device, **kwargs)
        result = detector.analyze_batch(images)
        for k in scores:
            scores[k].append(result[k])

    return {k: float(np.mean(v)) for k, v in scores.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_cifar10_loaders(batch_size=32)

    model = DefenseCNN(num_classes=10).to(device)
    model.load_state_dict(
        torch.load("checkpoints/cnn_best.pth", map_location=device)
    )
    model.eval()

    detector = AdversarialDetector(model=model, device=device)

    conditions = [
        ("Clean",        None,   None),
        ("FGSM ε=0.03",  "fgsm", 0.03),
        ("FGSM ε=0.10",  "fgsm", 0.10),
        ("PGD  ε=0.03",  "pgd",  0.03),
        ("PGD  ε=0.10",  "pgd",  0.10),
    ]

    results = {}
    print(f"\n{'Condition':<16} {'Conf':>7} {'Sens':>7} "
          f"{'Inst':>7} {'GradMag':>9} {'Risk':>7}")
    print("-" * 60)

    for name, atk, eps in conditions:
        r = collect(detector, model, test_loader, device,
                    attack_type=atk, epsilon=eps)
        results[name] = r
        print(f"{name:<16} {r['confidence']:>7.3f} {r['sensitivity']:>7.3f} "
              f"{r['instability']:>7.3f} {r['grad_magnitude']:>9.3f} "
              f"{r['risk_score']:>7.3f}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/detection_pgd_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[SAVED] logs/detection_pgd_analysis.json")

    # Sanity check: PGD ε=0.10 must have higher risk than FGSM ε=0.10
    pgd_risk  = results["PGD  ε=0.10"]["risk_score"]
    fgsm_risk = results["FGSM ε=0.10"]["risk_score"]
    clean_risk = results["Clean"]["risk_score"]
    print(f"\n=== Sanity Checks ===")
    print(f"Clean risk < FGSM risk:        "
          f"{'✓' if clean_risk < fgsm_risk else '✗ FAIL'}")
    print(f"FGSM ε=0.10 risk < PGD ε=0.10 risk: "
          f"{'✓' if fgsm_risk < pgd_risk else '✗ CHECK — may be close'}")
    pgd_grad = results["PGD  ε=0.10"]["grad_magnitude"]
    clean_grad = results["Clean"]["grad_magnitude"]

    print(
    f"PGD ε=0.10 grad_mag > Clean: "
    f"{'✓' if pgd_grad > clean_grad else '✗ FAIL'}"
    )


if __name__ == "__main__":
    main()