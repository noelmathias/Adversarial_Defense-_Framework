"""
detection/detector.py

AdversarialDetector — five-signal behavioral detection module.

Signal set (Phase 2, updated):
    1. confidence       — max softmax probability
    2. sensitivity      — mean softmax shift under noise (L2)
    3. instability      — prediction flip rate under noise
    4. noise_entropy    — Monte Carlo prediction entropy (REPLACES grad_magnitude)
    5. risk_score       — weighted combination of signals 1–4

Why noise_entropy replaces gradient magnitude
---------------------------------------------
PGD converges to an approximate critical point of the loss surface, so
∇_x L ≈ 0 at convergence — every gradient-based objective (cross-entropy,
margin loss, negative top-1 logit) collapses to near-zero for strongly
attacked inputs, producing no useful separation.

Noise entropy exploits a different property: adversarial examples occupy
geometrically thin regions in the smoothed input distribution (Cohen et al.
2019). Even when the base classifier is highly confident in the wrong class,
adding small Gaussian noise frequently pushes the input back across the
decision boundary, producing inconsistent predictions. The empirical Shannon
entropy of the predicted-class distribution over n noisy copies quantifies
this instability and increases monotonically with attack strength.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdversarialDetector:
    """
    Behavioral detection module.  Operates entirely on model outputs —
    no separate detector network, no ground-truth labels required.

    Parameters
    ----------
    model       : trained classifier (eval mode)
    device      : torch device
    noise_std   : std of Gaussian noise used for sensitivity / instability
                  probing (default 0.05 — matches the defense module)
    n_samples   : number of noisy copies per detection call (default 50)
                  Higher values give more reliable noise-entropy estimates.
    weights     : dict mapping signal names to scalar weights (must sum ~1).
                  Default distributes weight toward noise_entropy because it
                  is the only signal that reliably separates clean from
                  strongly-adversarial PGD inputs.
    """

    def __init__(
        self,
        model,
        device,
        noise_std: float = 0.05,
        n_samples: int = 50,
        weights: dict = None,
    ):
        self.model      = model
        self.device     = device
        self.noise_std  = noise_std
        self.n_samples  = n_samples

        # Default weights — noise_entropy carries the most weight because
        # it is the only signal that does not degrade under confident-wrong PGD.
        if weights is None:
            self.weights = {
                "confidence":    0.20,
                "sensitivity":   0.15,
                "instability":   0.15,
                "noise_entropy": 0.50,
            }
        else:
            self.weights = weights

    # ------------------------------------------------------------------ #
    # Signal 1: Confidence                                                 #
    # ------------------------------------------------------------------ #
    def compute_confidence(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns max softmax probability for each image.
        Shape: (B,)
        """
        self.model.eval()
        with torch.no_grad():
            probs = F.softmax(self.model(images), dim=1)
        confidence, _ = probs.max(dim=1)
        return confidence

    # ------------------------------------------------------------------ #
    # Signal 2: Sensitivity                                                #
    # ------------------------------------------------------------------ #
    def compute_sensitivity(self, images: torch.Tensor) -> torch.Tensor:
        """
        Mean L2 distance between base softmax and noisy-copy softmax,
        averaged over n_samples copies.  Shape: (B,)
        """
        self.model.eval()
        with torch.no_grad():
            base_probs = F.softmax(self.model(images), dim=1)   # (B, C)
            distances  = torch.zeros(images.size(0)).to(self.device)
            for _ in range(self.n_samples):
                noise       = torch.randn_like(images) * self.noise_std
                noisy_probs = F.softmax(self.model(images + noise), dim=1)
                distances  += (noisy_probs - base_probs).pow(2).sum(dim=1).sqrt()
        return distances / self.n_samples

    # ------------------------------------------------------------------ #
    # Signal 3: Instability                                                #
    # ------------------------------------------------------------------ #
    def compute_instability(self, images: torch.Tensor) -> torch.Tensor:
        """
        Fraction of noisy copies whose predicted class differs from the
        base prediction.  Shape: (B,)
        """
        self.model.eval()
        with torch.no_grad():
            base_preds = self.model(images).argmax(dim=1)
            flip_count = torch.zeros(images.size(0)).to(self.device)
            for _ in range(self.n_samples):
                noise       = torch.randn_like(images) * self.noise_std
                noisy_preds = self.model(images + noise).argmax(dim=1)
                flip_count += (noisy_preds != base_preds).float()
        return flip_count / self.n_samples

    # ------------------------------------------------------------------ #
    # Signal 4: Noise Entropy  (replaces gradient magnitude)              #
    # ------------------------------------------------------------------ #
    def compute_noise_entropy(
        self,
        images: torch.Tensor,
        n_samples: int = None,
        noise_std:  float = None,
    ) -> torch.Tensor:
        """
        Monte Carlo prediction entropy under Gaussian noise.

        Algorithm
        ---------
        1.  Generate n_samples noisy copies:
                x_i = clip(x + δ_i, -3, 3),  δ_i ~ N(0, σ²I)
        2.  Record predicted class for each copy:
                ŷ_i = argmax f(x_i)
        3.  Build empirical class-frequency distribution over the B inputs:
                p̂_k  = (1/n) Σ_i  1[ŷ_i = k],    k = 0..K-1
        4.  Compute normalized Shannon entropy per image:
                H_noise = −Σ_k p̂_k log(p̂_k) / log(K)  ∈ [0, 1]

        Returns
        -------
        noise_entropy : (B,) tensor in [0, 1]
            Low  → same class predicted consistently  → clean-like
            High → predictions flip across copies     → adversarial-like
        """
        n   = n_samples if n_samples is not None else self.n_samples
        std = noise_std  if noise_std  is not None else self.noise_std

        NUM_CLASSES = 10
        MAX_ENTROPY = math.log(NUM_CLASSES)  # normalisation constant
        B           = images.size(0)

        vote_counts = torch.zeros(B, NUM_CLASSES).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for _ in range(n):
                noise = torch.randn_like(images) * std
                noisy = torch.clamp(images + noise, -3.0, 3.0)
                preds = self.model(noisy).argmax(dim=1)          # (B,)
                for i in range(B):
                    vote_counts[i, preds[i]] += 1

        probs_v   = torch.clamp(vote_counts / n, 1e-10, 1.0)
        raw_ent   = -(probs_v * torch.log(probs_v)).sum(dim=1)  # (B,)
        return raw_ent / MAX_ENTROPY                              # (B,) in [0,1]

    # kept as alias so any code that still calls compute_grad_magnitude
    # continues to work without modification
    def compute_grad_magnitude(self, images: torch.Tensor) -> torch.Tensor:
        """Deprecated alias → delegates to compute_noise_entropy."""
        return self.compute_noise_entropy(images)

    # ------------------------------------------------------------------ #
    # Combined risk score                                                  #
    # ------------------------------------------------------------------ #
    def compute_risk_score(self, images: torch.Tensor):
        """
        Combine all four signals into a single risk score in [0, 1].

        Returns
        -------
        risk_score : (B,) tensor
        context    : dict with individual signal tensors
        """
        confidence    = self.compute_confidence(images)
        sensitivity   = self.compute_sensitivity(images)
        instability   = self.compute_instability(images)
        noise_entropy = self.compute_noise_entropy(images)

        sensitivity_norm = torch.clamp(sensitivity / 0.5, 0.0, 1.0)
        conf_risk        = 1.0 - confidence

        risk_score = (
            self.weights["confidence"]    * conf_risk
            + self.weights["sensitivity"] * sensitivity_norm
            + self.weights["instability"] * instability
            + self.weights["noise_entropy"] * noise_entropy
        )
        risk_score = torch.clamp(risk_score, 0.0, 1.0)

        context = {
            "confidence":        confidence,
            "sensitivity":       sensitivity,
            "sensitivity_norm":  sensitivity_norm,
            "instability":       instability,
            "noise_entropy":     noise_entropy,   # key used everywhere downstream
            "grad_magnitude":    noise_entropy,   # legacy alias for old code
            "risk_score":        risk_score,
        }
        return risk_score, context

    def analyze_batch(self, images: torch.Tensor) -> dict:
        """Convenience wrapper — returns scalar means for logging."""
        risk_score, context = self.compute_risk_score(images)
        return {
            "confidence":    context["confidence"].mean().item(),
            "sensitivity":   context["sensitivity"].mean().item(),
            "instability":   context["instability"].mean().item(),
            "noise_entropy": context["noise_entropy"].mean().item(),
            "grad_magnitude": context["noise_entropy"].mean().item(),  # legacy
            "risk_score":    risk_score.mean().item(),
        }