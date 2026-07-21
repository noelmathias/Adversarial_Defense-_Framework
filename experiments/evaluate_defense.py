import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import time

from models.cnn import DefenseCNN
from utils.data_loader import get_cifar10_loaders
from attacks.fgsm import fgsm_attack
from defense.defenses import (
    apply_light_defense,
    apply_medium_defense,
    apply_strong_defense,
    plot_defense_entropy,
    validate_defense_ordering
)


# ─────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────

def evaluate_clean(model, loader, device, n_batches=30):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i >= n_batches:
                break
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
    return 100. * correct / total


def evaluate_fgsm_only(model, loader, device, epsilon, n_batches=30):
    model.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)
        adv_images, _ = fgsm_attack(model, images, labels, epsilon, device)
        with torch.no_grad():
            preds = model(adv_images).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return 100. * correct / total


# In experiments/evaluate_defense.py
# Replace the three evaluate_ functions with these:

def evaluate_fgsm_light(model, loader, device, epsilon, n_batches=30):
    model.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)
        adv, _  = fgsm_attack(model, images, labels, epsilon, device)
        defended = apply_light_defense(adv, noise_std=0.05)
        with torch.no_grad():
            preds = model(defended).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return 100. * correct / total


def evaluate_fgsm_medium(model, loader, device, epsilon, n_batches=30):
    model.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)
        adv, _    = fgsm_attack(model, images, labels, epsilon, device)
        avg_probs = apply_medium_defense(
            adv, model, n_samples=5, noise_std=0.08
        )
        preds = avg_probs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return 100. * correct / total


def evaluate_fgsm_strong(model, loader, device, epsilon, n_batches=30):
    model.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)
        adv, _    = fgsm_attack(model, images, labels, epsilon, device)
        avg_probs = apply_strong_defense(
            adv, model, n_samples=20, noise_std=0.12
        )
        preds = avg_probs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return 100. * correct / total


# ─────────────────────────────────────────────
# EPSILON SWEEP (key research figure)
# ─────────────────────────────────────────────

def run_defense_sweep(model, loader, device, epsilons):
    """
    For each epsilon, evaluate all 5 conditions.
    Produces the core Phase 1 results table.
    """
    results = {
        "clean":        [],
        "fgsm":         [],
        "fgsm_light":   [],
        "fgsm_medium":  [],
        "fgsm_strong":  [],
    }

    clean_acc = evaluate_clean(model, loader, device, n_batches=30)
    for eps in epsilons:
        results["clean"].append(clean_acc)

    for eps in epsilons:
        print(f"\n  ε = {eps:.3f}")

        t0  = time.time()
        acc = evaluate_fgsm_only(model, loader, device, eps, n_batches=30)
        results["fgsm"].append(acc)
        print(f"    FGSM only:    {acc:.2f}%  ({time.time()-t0:.1f}s)")

        t0  = time.time()
        acc = evaluate_fgsm_light(model, loader, device, eps, n_batches=30)
        results["fgsm_light"].append(acc)
        print(f"    + Light:      {acc:.2f}%  ({time.time()-t0:.1f}s)")

        t0  = time.time()
        acc = evaluate_fgsm_medium(model, loader, device, eps, n_batches=30)
        results["fgsm_medium"].append(acc)
        print(f"    + Medium:     {acc:.2f}%  ({time.time()-t0:.1f}s)")

        t0  = time.time()
        acc = evaluate_fgsm_strong(model, loader, device, eps, n_batches=30)
        results["fgsm_strong"].append(acc)
        print(f"    + Strong:     {acc:.2f}%  ({time.time()-t0:.1f}s)")

    return results


# ─────────────────────────────────────────────
# OUTPUT: TABLE + PLOTS
# ─────────────────────────────────────────────

def print_results_table(results, epsilons):
    print("\n" + "=" * 65)
    print(f"{'Attack / Defense':<28}", end="")
    for eps in epsilons:
        print(f"  ε={eps:.2f}", end="")
    print()
    print("-" * 65)

    rows = [
        ("Clean",           "clean"),
        ("FGSM (no defense)","fgsm"),
        ("FGSM + Light",    "fgsm_light"),
        ("FGSM + Medium",   "fgsm_medium"),
        ("FGSM + Strong",   "fgsm_strong"),
    ]
    for label, key in rows:
        print(f"{label:<28}", end="")
        for acc in results[key]:
            print(f"  {acc:>6.2f}%", end="")
        print()
    print("=" * 65)


def plot_defense_comparison(results, epsilons,
                            save_path="results/defense_comparison.png"):
    os.makedirs("results", exist_ok=True)

    styles = {
        "clean":       ("steelblue",  "--", "Clean (no attack)"),
        "fgsm":        ("crimson",    "-",  "FGSM (no defense)"),
        "fgsm_light":  ("orange",     "-",  "FGSM + Light Defense"),
        "fgsm_medium": ("green",      "-",  "FGSM + Medium Defense"),
        "fgsm_strong": ("purple",     "-",  "FGSM + Strong Defense"),
    }

    plt.figure(figsize=(9, 5))
    for key, (color, ls, label) in styles.items():
        plt.plot(epsilons, results[key],
                 color=color, linestyle=ls, linewidth=2,
                 marker='o', label=label)

    plt.xlabel("Perturbation Budget ε", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Defense Effectiveness vs. Attack Strength\n(CIFAR-10, FGSM)", fontsize=13)
    plt.legend(fontsize=9, loc="upper right")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n[SAVED] Defense comparison → {save_path}")


def plot_recovery_bars(results, epsilon_idx=2,
                       epsilons=None,
                       save_path="results/defense_recovery_bar.png"):
    """
    Bar chart at a single epsilon showing recovery from attack.
    Good for demo slides — single clear figure.
    """
    os.makedirs("results", exist_ok=True)

    eps_val = epsilons[epsilon_idx]
    labels  = ["Clean", "FGSM", "+Light", "+Medium", "+Strong"]
    keys    = ["clean", "fgsm", "fgsm_light", "fgsm_medium", "fgsm_strong"]
    accs    = [results[k][epsilon_idx] for k in keys]
    colors  = ["steelblue", "crimson", "orange", "green", "purple"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, accs, color=colors, edgecolor='black', linewidth=0.7)

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{acc:.1f}%",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.axhline(y=10, linestyle=':', color='gray', linewidth=1, label='Random (10%)')
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title(f"Accuracy Recovery Under FGSM (ε={eps_val})\n"
              f"Clean vs. Attack vs. Defenses", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SAVED] Recovery bar chart → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    _, test_loader = get_cifar10_loaders(batch_size=64)

    model = DefenseCNN(num_classes=10).to(device)
    model.load_state_dict(
        torch.load("checkpoints/cnn_best.pth", map_location=device)
    )
    model.eval()
    print("[INFO] Model loaded.\n")

    '''# Sanity check before full sweep — runs once, costs nothing
    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images[:32].to(device)
    sample_labels = sample_labels[:32].to(device)
    adv_sample, _ = fgsm_attack(model, sample_images, sample_labels, 0.03, device)
    run_defense_sanity_check(sample_images, adv_sample, model, device)'''




    epsilons = [0.01, 0.02, 0.03, 0.05, 0.08]

    print("=== Running Defense Evaluation Sweep ===")
    results = run_defense_sweep(model, test_loader, device, epsilons)
    print_results_table(results, epsilons)
    
    plot_defense_comparison(results, epsilons)
    plot_recovery_bars(results, epsilon_idx=2, epsilons=epsilons)

    # After run_defense_sweep completes:
    validate_defense_ordering(results, epsilons)

    plot_defense_entropy(model, test_loader, device, epsilon=0.03)

    os.makedirs("logs", exist_ok=True)
    with open("logs/defense_results.json", "w") as f:
        json.dump({"epsilons": epsilons, "results": results}, f, indent=2)
    print("[SAVED] Logs → logs/defense_results.json")


if __name__ == "__main__":
    main()