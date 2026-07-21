import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import json

from models.cnn import DefenseCNN
from utils.data_loader import get_cifar10_loaders
from attacks.fgsm import fgsm_attack
from detection.detector import AdversarialDetector


def collect_scores(detector, model, loader, device, epsilon=None, n_batches=20):
    """
    Collect per-image risk scores for clean or adversarial inputs.
    Limits to n_batches for speed during development.
    """
    all_scores = {
        "confidence": [],
        "sensitivity": [],
        "instability": [],
        "risk_score": [],
    }

    model.eval()
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break

        images, labels = images.to(device), labels.to(device)

        if epsilon is not None:
            images, _ = fgsm_attack(model, images, labels, epsilon, device)

        _, context = detector.compute_risk_score(images)

        all_scores["confidence"].extend(context["confidence"].cpu().tolist())
        all_scores["sensitivity"].extend(context["sensitivity"].cpu().tolist())
        all_scores["instability"].extend(context["instability"].cpu().tolist())
        all_scores["risk_score"].extend(context["risk_score"].cpu().tolist())

    return {k: np.array(v) for k, v in all_scores.items()}


def print_comparison_table(clean_scores, adv_scores, epsilon):
    metrics = ["confidence", "sensitivity", "instability", "risk_score"]
    print(f"\n{'Metric':<20} {'Clean Mean':>12} {'Adv Mean (ε='+str(epsilon)+')':>16} {'Separation':>12}")
    print("-" * 62)
    for m in metrics:
        c_mean = clean_scores[m].mean()
        a_mean = adv_scores[m].mean()
        sep    = abs(a_mean - c_mean)
        print(f"{m:<20} {c_mean:>12.4f} {a_mean:>16.4f} {sep:>11.4f}↑")


def plot_risk_distributions(clean_scores, adv_scores, epsilon,
                            save_path="results/detection_distributions.png"):
    os.makedirs("results", exist_ok=True)

    metrics = ["confidence", "sensitivity", "instability", "risk_score"]
    titles  = ["Confidence", "Sensitivity", "Instability", "Risk Score"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"Detection Signal Distributions — Clean vs. FGSM (ε={epsilon})", fontsize=13)

    for ax, m, title in zip(axes, metrics, titles):
        ax.hist(clean_scores[m], bins=40, alpha=0.6, color='steelblue',
                label='Clean', density=True)
        ax.hist(adv_scores[m],   bins=40, alpha=0.6, color='crimson',
                label=f'FGSM ε={epsilon}', density=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Score Value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SAVED] Distribution plot → {save_path}")


def plot_risk_score_sweep(detector, model, loader, device,
                          epsilons=[0.0, 0.01, 0.02, 0.03, 0.05, 0.08],
                          save_path="results/risk_score_vs_epsilon.png"):
    """
    Shows how risk score increases as attack strength increases.
    This is a key research figure — proves the detector tracks attack severity.
    """
    mean_risks = []

    for eps in epsilons:
        scores = collect_scores(
            detector, model, loader, device,
            epsilon=(eps if eps > 0 else None),
            n_batches=10
        )
        mean_risks.append(scores["risk_score"].mean())
        print(f"  ε={eps:.3f} → mean risk score: {mean_risks[-1]:.4f}")

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(epsilons, mean_risks, marker='o', linewidth=2, color='darkorange')
    plt.xlabel("Perturbation Budget ε", fontsize=12)
    plt.ylabel("Mean Risk Score", fontsize=12)
    plt.title("Detection Risk Score vs. Attack Strength", fontsize=13)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SAVED] Risk vs epsilon → {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}\n")

    _, test_loader = get_cifar10_loaders(batch_size=64)

    model = DefenseCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load("checkpoints/cnn_best.pth", map_location=device))
    model.eval()

    detector = AdversarialDetector(
        model=model,
        device=device,
        noise_std=0.05,
        n_samples=20,
        weights=(0.3, 0.4, 0.3)
    )

    EVAL_EPSILON = 0.03

    # --- Collect scores ---
    print("=== Collecting clean input scores ===")
    clean_scores = collect_scores(detector, model, test_loader, device,
                                  epsilon=None, n_batches=20)

    print(f"\n=== Collecting adversarial scores (FGSM ε={EVAL_EPSILON}) ===")
    adv_scores = collect_scores(detector, model, test_loader, device,
                                epsilon=EVAL_EPSILON, n_batches=20)

    # --- Table ---
    print_comparison_table(clean_scores, adv_scores, EVAL_EPSILON)

    # --- Plots ---
    plot_risk_distributions(clean_scores, adv_scores, EVAL_EPSILON)

    print("\n=== Risk Score vs. Epsilon Sweep ===")
    plot_risk_score_sweep(detector, model, test_loader, device)

    # --- Save results ---
    os.makedirs("logs", exist_ok=True)
    results = {
        "clean": {k: float(v.mean()) for k, v in clean_scores.items()},
        "adversarial": {k: float(v.mean()) for k, v in adv_scores.items()},
    }
    with open("logs/detection_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[SAVED] Detection results → logs/detection_results.json")


if __name__ == "__main__":
    main()