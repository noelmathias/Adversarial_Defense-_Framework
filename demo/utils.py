"""
demo/utils.py

Shared helpers for the Streamlit demo.

Change log (noise-entropy update)
----------------------------------
- compute_risk_score():
    • Removed gradient computation block (cross-entropy / margin / top-1 loss).
    • Added compute_noise_entropy_single() helper (50 noisy forward passes).
    • Risk formula updated: 5-signal equal-weight combination.
    • Return dict keeps key "grad_magnitude" as alias of noise_entropy so
      Section 3 of app.py requires zero changes.
- run_attack() — unchanged.
- apply_selected_defense() — unchanged.
- compute_recovery_status() — unchanged.
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants ────────────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

RISK_LOW_THRESHOLD    = 0.35
RISK_MEDIUM_THRESHOLD = 0.55
RISK_HIGH_FLOOR       = 0.56   # minimum score when label flip confirmed

# ── DEFENSE_CONFIG ───────────────────────────────────────────────────────────
DEFENSE_CONFIG = {
    "light": {
        "label":       "Light Defense",
        "icon":        "🟢",
        "description": "Single noisy forward pass. Fast, minimal accuracy cost. "
                       "Sufficient for weak perturbations.",
        "params":      {"noise_std": 0.05},
        "cost":        "Low  (1 forward pass)",
        "risk_range":  "Risk < 0.35, no flip",
    },
    "medium": {
        "label":       "Medium Defense",
        "icon":        "🟡",
        "description": "Randomized smoothing over 10 noisy samples. "
                       "Handles mid-strength attacks and borderline flips.",
        "params":      {"n_samples": 10, "noise_std": 0.08},
        "cost":        "Medium (10 forward passes)",
        "risk_range":  "0.35 ≤ Risk < 0.55, no flip",
    },
    "strong": {
        "label":       "Strong Defense",
        "icon":        "🔴",
        "description": "Randomized smoothing over 30 noisy samples. "
                       "Maximum robustness. Applied on all confirmed label flips.",
        "params":      {"n_samples": 30, "noise_std": 0.12},
        "cost":        "High (30 forward passes)",
        "risk_range":  "Risk ≥ 0.55 or confirmed label flip",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """PIL → (1, 3, 32, 32) normalized tensor."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transform(pil_image.convert("RGB")).unsqueeze(0)


def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """Normalized (1,3,32,32) tensor → RGB uint8 numpy array (256×256)."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img  = torch.clamp(tensor.squeeze(0).cpu() * std + mean, 0, 1)
    img  = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return np.array(Image.fromarray(img).resize((256, 256), Image.NEAREST))


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, tensor: torch.Tensor, device):
    """Single forward pass → (label_str, confidence_float, probs_ndarray)."""
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(tensor.to(device)), dim=1).squeeze(0)
    conf, idx = probs.max(dim=0)
    return CIFAR10_CLASSES[idx.item()], conf.item(), probs.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Attack dispatch
# ─────────────────────────────────────────────────────────────────────────────

def run_attack(model, tensor: torch.Tensor, device,
               epsilon: float, attack_type: str = "fgsm"):
    """
    Unified attack entry point.  Returns the same four values regardless of
    attack type so the UI never needs to branch on attack_type.

    Returns
    -------
    adv_tensor  : (1, 3, 32, 32) CPU tensor
    adv_label   : predicted class string
    adv_conf    : float confidence
    adv_probs   : (10,) numpy array
    """
    from attacks import ATTACK_REGISTRY

    attack_fn = ATTACK_REGISTRY[attack_type]
    model.eval()
    with torch.no_grad():
        label_tensor = model(tensor.to(device)).argmax(dim=1)

    kwargs = {"num_steps": 20} if attack_type == "pgd" else {}
    adv_tensor, _ = attack_fn(
        model, tensor.to(device), label_tensor, epsilon, device, **kwargs
    )
    adv_tensor  = adv_tensor.detach().cpu()
    adv_label, adv_conf, adv_probs = predict(model, adv_tensor, device)
    return adv_tensor, adv_label, adv_conf, adv_probs


def confidence_delta_color(delta: float) -> str:
    if delta < -0.3:  return "🔴"
    if delta < -0.1:  return "🟠"
    return "🟡"


# ─────────────────────────────────────────────────────────────────────────────
# Noise-entropy helper (single-image / small-batch, demo-side)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_noise_entropy_single(
    model,
    tensor: torch.Tensor,
    device,
    n_samples: int = 50,
    noise_std:  float = 0.05,
) -> float:
    """
    Monte Carlo noise entropy for a single input tensor (1, C, H, W).

    Steps
    -----
    1. Run n_samples forward passes on noisy copies x_i = clip(x+δ_i, -3, 3)
       where δ_i ~ N(0, noise_std² I).
    2. Record the predicted class ŷ_i for each copy.
    3. Compute the empirical class-frequency distribution p̂_k.
    4. Return H_noise = −Σ_k p̂_k log(p̂_k) / log(K)  ∈ [0, 1].

    Separation property
    -------------------
    Clean inputs:        same class every time → H_noise ≈ 0.02–0.08
    FGSM ε=0.03 inputs:  occasional flips     → H_noise ≈ 0.20–0.40
    PGD  ε=0.10 inputs:  frequent flips        → H_noise ≈ 0.65–0.90
    """
    NUM_CLASSES = 10
    MAX_ENTROPY = math.log(NUM_CLASSES)

    vote_counts = torch.zeros(1, NUM_CLASSES).to(device)
    inp = tensor.to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            noise = torch.randn_like(inp) * noise_std
            noisy = torch.clamp(inp + noise, -3.0, 3.0)
            pred  = model(noisy).argmax(dim=1).item()
            vote_counts[0, pred] += 1

    probs_v   = torch.clamp(vote_counts / n_samples, 1e-10, 1.0)
    raw_ent   = -(probs_v * torch.log(probs_v)).sum(dim=1).item()
    return float(np.clip(raw_ent / MAX_ENTROPY, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Detection — risk score  (updated to use noise entropy)
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_score(
    model,
    tensor: torch.Tensor,
    device,
    n_samples: int = 50,     # noise-entropy samples
    noise_std:  float = 0.05,
    clean_label: str = None,
    clean_confidence: float = None,
):
    """
    Five-signal risk score for a single input tensor.

    Formula
    -------
    r = 0.20·(1−c) + 0.20·H_norm + 0.20·Δc + 0.20·m + 0.20·H_noise

    where
        c       = max softmax probability (adversarial input)
        H_norm  = normalized Shannon entropy of softmax distribution
        Δc      = max(0, c_clean − c_adv)   [0 if clean_confidence not given]
        m       = 1 if predicted label ≠ clean_label, else 0
        H_noise = Monte Carlo noise entropy  (NEW — replaces grad_magnitude)

    Hard floor: if m==1  →  r = max(r, RISK_HIGH_FLOOR)

    Returns
    -------
    dict with keys:
        confidence, entropy_raw, entropy_norm, conf_drop,
        pred_mismatch, unstable, noise_entropy, grad_magnitude (alias),
        risk_score
    """
    model.eval()

    # ── Base softmax signals ─────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs  = F.softmax(logits, dim=1).squeeze(0)   # (10,)

    adv_confidence = probs.max().item()
    adv_label      = CIFAR10_CLASSES[probs.argmax().item()]

    MAX_ENTROPY  = math.log(10)
    raw_entropy  = -(probs * torch.log(probs + 1e-8)).sum().item()
    entropy_norm = raw_entropy / MAX_ENTROPY

    # ── Confidence drop ──────────────────────────────────────────────────
    if clean_confidence is not None:
        conf_drop = float(np.clip(clean_confidence - adv_confidence, 0.0, 1.0))
    else:
        conf_drop = 0.0

    # ── Prediction mismatch ──────────────────────────────────────────────
    pred_mismatch = int(
        clean_label is not None and adv_label != clean_label
    )
    unstable = int(adv_confidence < 0.80)

    # ── Noise entropy (primary adversarial signal) ───────────────────────
    noise_entropy_val = _compute_noise_entropy_single(
        model, tensor, device,
        n_samples=n_samples,
        noise_std=noise_std,
    )

    # ── Combined risk ────────────────────────────────────────────────────
    risk_score = (
        0.20 * (1.0 - adv_confidence)
        + 0.20 * entropy_norm
        + 0.20 * conf_drop
        + 0.20 * float(pred_mismatch)
        + 0.20 * noise_entropy_val
    )

    if pred_mismatch:
        risk_score = max(risk_score, RISK_HIGH_FLOOR)

    risk_score = float(np.clip(risk_score, 0.0, 1.0))

    return {
        "confidence":    adv_confidence,
        "entropy_raw":   raw_entropy,
        "entropy_norm":  entropy_norm,
        "conf_drop":     conf_drop,
        "pred_mismatch": pred_mismatch,
        "unstable":      unstable,
        "noise_entropy": noise_entropy_val,
        "grad_magnitude": noise_entropy_val,   # legacy key — app.py reads this
        "risk_score":    risk_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Risk level label
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_level(risk_score: float):
    """Returns (level_str, colour_str, icon_str)."""
    if risk_score < RISK_LOW_THRESHOLD:
        return "LOW",    "success", "🟢"
    if risk_score < RISK_MEDIUM_THRESHOLD:
        return "MEDIUM", "warning", "🟡"
    return "HIGH",   "error",   "🔴"


# ─────────────────────────────────────────────────────────────────────────────
# Decision engine
# ─────────────────────────────────────────────────────────────────────────────

def select_defense(
    risk_score:   float,
    pred_mismatch: int   = 0,
    unstable:      int   = 0,
    entropy_norm:  float = 0.0,
):
    """
    Rule-based defense selector with mismatch and instability escalation.

    Priority order
    --------------
    1. Mismatch + risk ≥ 0.35  → STRONG
    2. Mismatch + risk < 0.35  → MEDIUM
    3. No mismatch, (unstable OR entropy_norm > 0.20):
           escalate base tier by one level
    4. Standard threshold bands
    """
    # Rule 1 & 2 — confirmed flip
    if pred_mismatch:
        if risk_score >= RISK_LOW_THRESHOLD:
            tier   = "strong"
            reason = (
                f"Label flip confirmed + risk {risk_score:.3f} ≥ {RISK_LOW_THRESHOLD}. "
                "Escalated to Strong — confirmed attack requires maximum recovery."
            )
        else:
            tier   = "medium"
            reason = (
                f"Label flip confirmed (risk={risk_score:.3f} < {RISK_LOW_THRESHOLD}). "
                "Escalated to Medium — mismatch overrides low-risk band."
            )
        return tier, DEFENSE_CONFIG[tier], reason

    # Rule 3 — instability escalation (no flip)
    if unstable or entropy_norm > 0.20:
        if risk_score < RISK_LOW_THRESHOLD:
            base_tier = "light"
        elif risk_score < RISK_MEDIUM_THRESHOLD:
            base_tier = "medium"
        else:
            base_tier = "strong"

        escalation_map = {"light": "medium", "medium": "strong", "strong": "strong"}
        tier = escalation_map[base_tier]

        triggers = []
        if unstable:              triggers.append("adv_conf < 0.80")
        if entropy_norm > 0.20:   triggers.append(f"entropy_norm={entropy_norm:.3f} > 0.20")
        reason = (
            f"No label flip. Instability detected: {', '.join(triggers)}. "
            f"Base tier '{base_tier}' escalated to '{tier}'."
        )
        return tier, DEFENSE_CONFIG[tier], reason

    # Rule 4 — threshold bands
    if risk_score < RISK_LOW_THRESHOLD:
        tier   = "light"
        reason = (
            f"Risk {risk_score:.3f} < {RISK_LOW_THRESHOLD}. "
            "No flip, no instability. Light defense sufficient."
        )
    elif risk_score < RISK_MEDIUM_THRESHOLD:
        tier   = "medium"
        reason = (
            f"Risk {risk_score:.3f} in [{RISK_LOW_THRESHOLD}, "
            f"{RISK_MEDIUM_THRESHOLD}). Medium smoothing applied."
        )
    else:
        tier   = "strong"
        reason = (
            f"Risk {risk_score:.3f} ≥ {RISK_MEDIUM_THRESHOLD}. "
            "Strong ensemble required."
        )
    return tier, DEFENSE_CONFIG[tier], reason


# ─────────────────────────────────────────────────────────────────────────────
# Defense application
# ─────────────────────────────────────────────────────────────────────────────

def apply_selected_defense(model, adv_tensor: torch.Tensor, device, tier: str, config: dict):
    """
    Execute defense strictly from config["params"].  No internal defaults.

    Returns
    -------
    defended_img  : (1,3,32,32) CPU tensor  (for display)
    def_label     : predicted class string after defense
    def_conf      : float confidence after defense
    def_probs     : (10,) numpy probability array
    """
    from defense.defenses import (
        apply_light_defense,
        apply_medium_defense,
        apply_strong_defense,
    )

    model.eval()
    adv_tensor = adv_tensor.to(device)
    params     = config["params"]

    if tier == "light":
        defended_tensor = apply_light_defense(adv_tensor, noise_std=params["noise_std"])
        with torch.no_grad():
            probs = F.softmax(model(defended_tensor), dim=1).squeeze(0)
    else:
        defense_fn = apply_medium_defense if tier == "medium" else apply_strong_defense
        probs = defense_fn(
            adv_tensor, model,
            n_samples=params["n_samples"],
            noise_std=params["noise_std"],
        ).squeeze(0)
        defended_tensor = apply_light_defense(adv_tensor, noise_std=params["noise_std"])

    probs_np  = probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else probs
    def_conf  = float(probs_np.max())
    def_label = CIFAR10_CLASSES[int(probs_np.argmax())]
    def_img   = defended_tensor.detach().cpu()

    # Stability safeguard — don't return a worse result than the attack gave
    with torch.no_grad():
        adv_probs_np = F.softmax(model(adv_tensor), dim=1).squeeze(0).cpu().numpy()
    adv_conf_check = float(adv_probs_np.max())

    if def_conf < adv_conf_check:
        return (
            adv_tensor.cpu(),
            CIFAR10_CLASSES[int(adv_probs_np.argmax())],
            adv_conf_check,
            adv_probs_np,
        )

    return def_img, def_label, def_conf, probs_np


# ─────────────────────────────────────────────────────────────────────────────
# Recovery status
# ─────────────────────────────────────────────────────────────────────────────

def compute_recovery_status(
    clean_label:   str,
    clean_conf:    float,
    adv_conf:      float,
    def_label:     str,
    def_conf:      float,
    pred_mismatch: int,
) -> dict:
    """
    Logically correct recovery metrics.

    conf_recovery_pct is only meaningful when the label is restored.
    failure_explanation is populated when flip is confirmed but label not restored.
    """
    recovered = def_label == clean_label

    if recovered and (clean_conf - adv_conf) > 0.01:
        conf_recovery_pct = float(
            np.clip(
                (def_conf - adv_conf) / (clean_conf - adv_conf) * 100,
                0.0, 100.0,
            )
        )
    else:
        conf_recovery_pct = 0.0

    if pred_mismatch and not recovered:
        failure_explanation = (
            "Attack likely crossed the decision boundary significantly. "
            "Current defense (randomized smoothing) cannot recover "
            "deep adversarial shifts."
        )
    else:
        failure_explanation = None

    return {
        "recovered":            recovered,
        "conf_recovery_pct":    conf_recovery_pct,
        "failure_explanation":  failure_explanation,
    }