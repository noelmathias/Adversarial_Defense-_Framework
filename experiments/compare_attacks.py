import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from models.cnn import DefenseCNN
from utils.data_loader import get_cifar10_loaders
from attacks import ATTACK_REGISTRY

def evaluate(model, loader, device, attack_fn,
             epsilon, n_batches=20, **kwargs):
    model.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)
        adv, _  = attack_fn(model, images, labels,
                             epsilon, device, **kwargs)
        with torch.no_grad():
            preds = model(adv).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return 100. * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_cifar10_loaders(batch_size=64)

    model = DefenseCNN(num_classes=10).to(device)
    model.load_state_dict(
        torch.load("checkpoints/cnn_best.pth", map_location=device)
    )
    model.eval()

    epsilons = [0.03, 0.06, 0.10]
    results  = {}

    print(f"\n{'ε':<8} {'Clean':>8} {'FGSM':>8} {'PGD-20':>10} {'PGD stronger':>14}")
    print("-" * 52)

    # Clean baseline
    correct, total = 0, 0
    for i, (images, labels) in enumerate(test_loader):
        if i >= 20: break
        with torch.no_grad():
            preds = model(images.to(device)).argmax(dim=1)
        correct += preds.eq(labels.to(device)).sum().item()
        total   += labels.size(0)
    clean_acc = 100. * correct / total

    for eps in epsilons:
        fgsm_acc = evaluate(
            model, test_loader, device,
            ATTACK_REGISTRY["fgsm"], eps
        )
        pgd_acc  = evaluate(
            model, test_loader, device,
            ATTACK_REGISTRY["pgd"], eps,
            num_steps=40,
            alpha=eps/2
        )
        stronger = "✓" if pgd_acc < fgsm_acc else "✗ CHECK"
        results[eps] = {
            "clean": clean_acc,
            "fgsm":  fgsm_acc,
            "pgd":   pgd_acc
        }
        print(f"{eps:<8.2f} {clean_acc:>7.2f}% "
              f"{fgsm_acc:>7.2f}% {pgd_acc:>9.2f}% {stronger:>14}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/attack_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[SAVED] logs/attack_comparison.json")

if __name__ == "__main__":
    main()