"""
experiments/pgd_analysis.py

Phase 2 Stage 2 — PGD Analysis

Sweeps ε values under PGD and records per-tier accuracy, flip rate,
recovery rate, and detection signals (including the updated noise_entropy).

The grad_magnitude column in printed output now reflects noise_entropy.
All log keys use "noise_entropy"; "grad_magnitude" is kept as an alias
in the returned dict for any downstream code that still reads the old key.

Run:
    python experiments/pgd_analysis.py
"""

import sys
import os
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import DefenseCNN
from utils.data_loader import get_cifar10_loaders
from attacks import ATTACK_REGISTRY
from defense.defenses import (
    apply_light_defense,
    apply_medium_defense,
    apply_strong_defense,
)

# ── Config ────────────────────────────────────────────────────────────────────
EPSILONS   = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
N_BATCHES  = 25
BATCH_SIZE = 64
PGD_STEPS  = 20

# Noise-entropy detection parameters
NE_SAMPLES  = 50    # number of noisy forward passes per detection
NE_NOISE    = 0.05  # Gaussian noise std

DEFENSE_PARAMS = {
    "none":   {},
    "light":  {"noise_std": 0.05},
    "medium": {"n_samples": 10, "noise_std": 0.08},
    "strong": {"n_samples": 30, "noise_std": 0.12},
}


# ─────────────────────────────────────────────────────────────────────────────
# Detection signals (batch-level)
# ─────────────────────────────────────────────────────────────────────────────

def get_detection_signals(model, images: torch.Tensor, device) -> dict:
    """
    Compute confidence, entropy, and noise_entropy for a batch.

    noise_entropy replaces the previous gradient-magnitude signal.
    "grad_magnitude" is returned as an alias of noise_entropy so any
    downstream code that reads the old key continues to work.
    """
    import math
    NUM_CLASSES = 10
    MAX_ENTROPY = math.log(NUM_CLASSES)

    model.eval()

    # Confidence & softmax entropy
    with torch.no_grad():
        logits = model(images)
        probs  = F.softmax(logits, dim=1)

    confidence   = probs.max(dim=1).values.mean().item()
    raw_ent      = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    entropy_norm = (raw_ent / MAX_ENTROPY).mean().item()

    # Noise entropy — run NE_SAMPLES noisy forward passes per batch image
    B           = images.size(0)
    vote_counts = torch.zeros(B, NUM_CLASSES).to(device)

    with torch.no_grad():
        for _ in range(NE_SAMPLES):
            noise = torch.randn_like(images) * NE_NOISE
            noisy = torch.clamp(images + noise, -3.0, 3.0)
            preds = model(noisy).argmax(dim=1)    # (B,)
            for i in range(B):
                vote_counts[i, preds[i]] += 1

    probs_v   = torch.clamp(vote_counts / NE_SAMPLES, 1e-10, 1.0)
    ne_per    = -(probs_v * torch.log(probs_v)).sum(dim=1) / MAX_ENTROPY  # (B,)
    noise_entropy = ne_per.mean().item()

    return {
        "confidence":    confidence,
        "entropy_norm":  entropy_norm,
        "noise_entropy": noise_entropy,
        "grad_magnitude": noise_entropy,   # legacy alias
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single (ε, defense_tier) evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_condition(model, loader, device, epsilon, defense_tier, n_batches):
    attack_fn = ATTACK_REGISTRY["pgd"]
    correct_adv, correct_def, total = 0, 0, 0
    flips, recoveries               = 0, 0
    det_signals                     = []
    t_start                         = time.time()

    model.eval()
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)

        # Clean predictions
        with torch.no_grad():
            clean_preds = model(images).argmax(dim=1)

        # PGD attack
        adv_images, _ = attack_fn(
            model, images, labels, epsilon, device,
            num_steps=PGD_STEPS,
        )

        # Detection signals on adversarial batch
        sigs = get_detection_signals(model, adv_images, device)
        det_signals.append(sigs)

        # Adversarial accuracy
        with torch.no_grad():
            adv_preds = model(adv_images).argmax(dim=1)
        correct_adv += adv_preds.eq(labels).sum().item()
        flips       += (adv_preds != clean_preds).sum().item()

        # Defense application
        params = DEFENSE_PARAMS[defense_tier]
        if defense_tier == "none":
            def_preds = adv_preds
        elif defense_tier == "light":
            defended = apply_light_defense(adv_images, noise_std=params["noise_std"])
            with torch.no_grad():
                def_preds = model(defended).argmax(dim=1)
        elif defense_tier == "medium":
            avg_probs = apply_medium_defense(
                adv_images, model,
                n_samples=params["n_samples"],
                noise_std=params["noise_std"],
            )
            def_preds = avg_probs.argmax(dim=1)
        else:   # strong
            avg_probs = apply_strong_defense(
                adv_images, model,
                n_samples=params["n_samples"],
                noise_std=params["noise_std"],
            )
            def_preds = avg_probs.argmax(dim=1)

        correct_def += def_preds.eq(labels).sum().item()

        # Recovery: flipped by attack AND restored by defense
        flipped_mask  = adv_preds != labels
        restored_mask = def_preds == labels
        recoveries   += (flipped_mask & restored_mask).sum().item()
        total        += labels.size(0)

    elapsed       = time.time() - t_start
    adv_acc       = 100. * correct_adv / total
    def_acc       = 100. * correct_def / total
    flip_rate     = 100. * flips / total
    recovery_rate = 100. * recoveries / max(flips, 1)

    avg_sigs = {
        k: float(np.mean([s[k] for s in det_signals]))
        for k in det_signals[0]
    }

    return {
        "adv_accuracy":   round(adv_acc, 2),
        "def_accuracy":   round(def_acc, 2),
        "flip_rate":      round(flip_rate, 2),
        "recovery_rate":  round(recovery_rate, 2),
        "elapsed_s":      round(elapsed, 1),
        **avg_sigs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_accuracy_curves(results, epsilons):
    styles = {
        "none":   ("crimson",        "--", "No Defense"),
        "light":  ("orange",         "-",  "Light Defense"),
        "medium": ("mediumseagreen", "-",  "Medium Defense"),
        "strong": ("steelblue",      "-",  "Strong Defense"),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for tier, (color, ls, label) in styles.items():
        accs = [results[eps][tier]["def_accuracy"] for eps in epsilons]
        ax.plot(epsilons, accs, color=color, linestyle=ls,
                linewidth=2, marker="o", label=label)
    ax.axhline(10, linestyle=":", color="gray",  linewidth=1, alpha=0.6, label="Random (10%)")
    ax.axhline(30, linestyle=":", color="red",   linewidth=1, alpha=0.4, label="Failure threshold (30%)")
    ax.set_xlabel("Perturbation Budget ε (PGD)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("PGD: Defense Accuracy vs. Attack Strength", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/pgd_accuracy_curves.png", dpi=150)
    plt.close()
    print("[SAVED] results/pgd_accuracy_curves.png")


def _plot_detection_signals(results, epsilons):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("PGD Detection Signals vs. ε (no-defense condition)", fontsize=12)
    signal_cfg = [
        ("confidence",    "Confidence",       "steelblue"),
        ("entropy_norm",  "Entropy (norm)",   "crimson"),
        ("noise_entropy", "Noise Entropy",     "darkorange"),  # updated key
    ]
    for ax, (key, label, color) in zip(axes, signal_cfg):
        vals = [results[eps]["none"][key] for eps in epsilons]
        ax.plot(epsilons, vals, color=color, linewidth=2, marker="o")
        ax.set_xlabel("ε", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/pgd_detection_signals.png", dpi=150)
    plt.close()
    print("[SAVED] results/pgd_detection_signals.png")


def _plot_recovery_rates(results, epsilons):
    colors = {"light": "orange", "medium": "mediumseagreen", "strong": "steelblue"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for tier in ["light", "medium", "strong"]:
        rates = [results[eps][tier]["recovery_rate"] for eps in epsilons]
        ax.plot(epsilons, rates, color=colors[tier], linewidth=2,
                marker="o", label=f"{tier.capitalize()} Defense")
    ax.set_xlabel("Perturbation Budget ε (PGD)", fontsize=12)
    ax.set_ylabel("Recovery Rate (%)", fontsize=12)
    ax.set_title("PGD: Recovery Rate vs. Attack Strength\n"
                 "(% of flipped predictions restored by defense)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/pgd_recovery_rates.png", dpi=150)
    plt.close()
    print("[SAVED] results/pgd_recovery_rates.png")


def _build_findings(results, epsilons, failure_eps):
    findings = {}
    max_eps = max(epsilons)
    findings["strong_vs_none_at_max_eps"] = (
        f"ε={max_eps}: strong={results[max_eps]['strong']['def_accuracy']:.1f}% "
        f"vs no-defense={results[max_eps]['none']['adv_accuracy']:.1f}%"
    )
    ne_low  = results[min(epsilons)]["none"]["noise_entropy"]
    ne_high = results[max_eps]["none"]["noise_entropy"]
    findings["noise_entropy_separation"] = (
        f"ε={min(epsilons)}: {ne_low:.3f} → ε={max_eps}: {ne_high:.3f} "
        f"(ratio: {ne_high / max(ne_low, 1e-6):.1f}×)"
    )
    findings["failure_thresholds"] = {
        tier: (f"ε={eps:.2f}" if eps else "does not fail")
        for tier, eps in failure_eps.items()
    }
    mid_eps = epsilons[len(epsilons) // 2]
    findings["recovery_at_mid_eps"] = {
        tier: f"{results[mid_eps][tier]['recovery_rate']:.1f}%"
        for tier in ["light", "medium", "strong"]
    }
    findings["compute_cost_at_max_eps"] = {
        tier: results[max_eps][tier]["elapsed_s"]
        for tier in ["light", "medium", "strong"]
    }
    findings["bandit_baseline_note"] = (
        "These results define the static defense baseline. "
        "Stage 5 bandit must exceed the best static defense "
        "while using lower average compute cost."
    )
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} · PGD steps: {PGD_STEPS} · "
          f"Batches: {N_BATCHES} · NE samples: {NE_SAMPLES}")

    _, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)

    model = DefenseCNN(num_classes=10).to(device)
    model.load_state_dict(
        torch.load("checkpoints/cnn_best.pth", map_location=device)
    )
    model.eval()

    results = {}

    for eps in EPSILONS:
        results[eps] = {}
        print(f"\n{'═'*70}")
        print(f"ε = {eps:.2f}")
        print(f"{'─'*70}")
        print(f"  {'Defense':<10} {'AtkAcc':>8} {'DefAcc':>8} "
              f"{'Flip%':>7} {'Rec%':>7} {'Conf':>7} "
              f"{'Entr':>7} {'NoiseEnt':>10} {'Time':>6}")
        print(f"  {'─'*80}")

        for tier in ["none", "light", "medium", "strong"]:
            r = evaluate_condition(
                model, test_loader, device,
                epsilon=eps,
                defense_tier=tier,
                n_batches=N_BATCHES,
            )
            results[eps][tier] = r
            print(
                f"  {tier:<10} "
                f"{r['adv_accuracy']:>7.2f}% "
                f"{r['def_accuracy']:>7.2f}% "
                f"{r['flip_rate']:>6.2f}% "
                f"{r['recovery_rate']:>6.2f}% "
                f"{r['confidence']:>7.3f} "
                f"{r['entropy_norm']:>7.3f} "
                f"{r['noise_entropy']:>10.3f} "
                f"{r['elapsed_s']:>5.1f}s"
            )

    # ── Save raw results ───────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    serializable = {str(k): v for k, v in results.items()}
    with open("logs/pgd_analysis.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[SAVED] logs/pgd_analysis.json")

    # ── Failure thresholds ────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("FAILURE THRESHOLD ANALYSIS  (def_accuracy < 30%)")
    print(f"{'─'*70}")
    failure_eps = {}
    for tier in ["light", "medium", "strong"]:
        failed_at = None
        for eps in EPSILONS:
            if results[eps][tier]["def_accuracy"] < 30.0:
                failed_at = eps
                break
        failure_eps[tier] = failed_at
        status = f"fails at ε={failed_at:.2f}" if failed_at else "holds across all ε"
        print(f"  {tier:<10} → {status}")

    # ── Plots ──────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    _plot_accuracy_curves(results, EPSILONS)
    _plot_detection_signals(results, EPSILONS)
    _plot_recovery_rates(results, EPSILONS)

    # ── Findings ───────────────────────────────────────────────────────────
    findings = _build_findings(results, EPSILONS, failure_eps)
    with open("logs/pgd_analysis_findings.json", "w") as f:
        json.dump(findings, f, indent=2)
    print(f"[SAVED] logs/pgd_analysis_findings.json")

    print(f"\n{'═'*70}")
    print("KEY FINDINGS")
    print(f"{'─'*70}")
    for k, v in findings.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()