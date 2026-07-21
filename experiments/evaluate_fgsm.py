import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json

from models.cnn import DefenseCNN
from utils.data_loader import get_cifar10_loaders
from attacks.fgsm import fgsm_attack


def evaluate_accuracy(model, loader, device, epsilon=0.0, attack_fn=None):
    """
    Evaluate model accuracy.
    If attack_fn provided, evaluates under attack at given epsilon.
    """
    model.eval()
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if attack_fn is not None and epsilon > 0:
            images, _ = attack_fn(model, images, labels, epsilon, device)

        with torch.no_grad():
            outputs   = model(images)
            _, predicted = outputs.max(1)
            correct  += predicted.eq(labels).sum().item()
            total    += labels.size(0)

    return 100. * correct / total


def run_epsilon_sweep(model, test_loader, device, epsilons):
    """
    Sweep epsilon values and record accuracy at each.
    This produces your key vulnerability curve.
    """
    results = {}

    # Baseline: clean accuracy
    clean_acc = evaluate_accuracy(model, test_loader, device)
    results["clean"] = clean_acc
    print(f"  Clean accuracy:          {clean_acc:.2f}%")

    # FGSM at each epsilon
    for eps in epsilons:
        adv_acc = evaluate_accuracy(
            model, test_loader, device,
            epsilon=eps, attack_fn=fgsm_attack
        )
        results[f"fgsm_eps_{eps:.3f}"] = adv_acc
        print(f"  FGSM ε={eps:.3f} accuracy:   {adv_acc:.2f}%")

    return results


def plot_vulnerability_curve(results, epsilons, save_path="results/fgsm_vulnerability.png"):
    os.makedirs("results", exist_ok=True)

    clean_acc = results["clean"]
    adv_accs  = [results[f"fgsm_eps_{eps:.3f}"] for eps in epsilons]

    plt.figure(figsize=(8, 5))
    plt.plot([0] + epsilons, [clean_acc] + adv_accs,
             marker='o', linewidth=2, color='crimson', label='FGSM Attack')
    plt.axhline(y=clean_acc, linestyle='--', color='steelblue',
                linewidth=1.5, label=f'Clean Accuracy ({clean_acc:.1f}%)')
    plt.axhline(y=10, linestyle=':', color='gray',
                linewidth=1, label='Random Guess (10%)')

    plt.xlabel("Perturbation Budget ε", fontsize=13)
    plt.ylabel("Test Accuracy (%)", fontsize=13)
    plt.title("Model Vulnerability Under FGSM Attack\n(CIFAR-10, DefenseCNN)", fontsize=14)
    plt.legend(fontsize=11)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n[SAVED] Vulnerability curve → {save_path}")


def visualize_adversarial_examples(model, test_loader, device,
                                   epsilon=0.03,
                                   save_path="results/fgsm_examples.png"):
    """
    Visual sanity check: show clean vs. adversarial images side-by-side.
    Critical for demo — proves perturbation is humanly imperceptible.
    """
    os.makedirs("results", exist_ok=True)

    classes = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

    # Get one batch
    images, labels = next(iter(test_loader))
    images, labels = images[:8].to(device), labels[:8].to(device)

    adv_images, _ = fgsm_attack(model, images, labels, epsilon, device)

    model.eval()
    with torch.no_grad():
        clean_preds = model(images).argmax(dim=1)
        adv_preds   = model(adv_images).argmax(dim=1)

    # Denormalize for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1).to(device)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1).to(device)

    def denorm(t):
        return torch.clamp(t * std + mean, 0, 1)

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    fig.suptitle(f"Clean vs. FGSM Adversarial Examples (ε={epsilon})", fontsize=13)

    for i in range(8):
        # Clean
        img_clean = denorm(images[i]).cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(img_clean)
        axes[0, i].set_title(f"True: {classes[labels[i]]}\nPred: {classes[clean_preds[i]]}",
                             fontsize=7)
        axes[0, i].axis('off')

        # Adversarial
        img_adv = denorm(adv_images[i]).cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(img_adv)
        color = 'red' if adv_preds[i] != labels[i] else 'green'
        axes[1, i].set_title(f"Adv Pred: {classes[adv_preds[i]]}",
                             fontsize=7, color=color)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Clean", fontsize=10)
    axes[1, 0].set_ylabel("Adversarial", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SAVED] Adversarial examples → {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    _, test_loader = get_cifar10_loaders(batch_size=128)

    model = DefenseCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load("checkpoints/cnn_best.pth", map_location=device))
    model.eval()
    print("[INFO] Model loaded from checkpoints/cnn_best.pth\n")

    # --- Epsilon sweep ---
    epsilons = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    print("=== FGSM Epsilon Sweep ===")
    results = run_epsilon_sweep(model, test_loader, device, epsilons)

    # Save results
    os.makedirs("logs", exist_ok=True)
    with open("logs/fgsm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Plots ---
    plot_vulnerability_curve(results, epsilons)
    visualize_adversarial_examples(model, test_loader, device, epsilon=0.03)

    # --- Print summary table ---
    print("\n=== Summary Table ===")
    print(f"{'Attack':<25} {'Accuracy':>10} {'Drop':>10}")
    print("-" * 47)
    clean = results["clean"]
    print(f"{'Clean (No Attack)':<25} {clean:>9.2f}%  {'—':>9}")
    for eps in epsilons:
        acc  = results[f"fgsm_eps_{eps:.3f}"]
        drop = clean - acc
        print(f"{'FGSM ε=' + str(eps):<25} {acc:>9.2f}%  {drop:>8.2f}%↓")


if __name__ == "__main__":
    main()