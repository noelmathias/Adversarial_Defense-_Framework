# defense/defenses.py

import torch
import torch.nn.functional as F


def apply_light_defense(images, noise_std):
    """
    Single noisy forward pass.
    noise_std must be passed explicitly — no default.
    """
    noise    = torch.randn_like(images) * noise_std
    defended = torch.clamp(images + noise, -3.0, 3.0)
    return defended


def apply_medium_defense(images, model, n_samples, noise_std):
    """
    Randomized smoothing — small ensemble.
    Both n_samples and noise_std must be passed explicitly.
    """
    model.eval()
    B           = images.size(0)
    accumulated = torch.zeros(B, 10).to(images.device)

    with torch.no_grad():
        for _ in range(n_samples):
            noise  = torch.randn_like(images) * noise_std
            noisy  = torch.clamp(images + noise, -3.0, 3.0)
            probs  = F.softmax(model(noisy), dim=1)
            accumulated += probs

    return accumulated / n_samples


def apply_strong_defense(images, model, n_samples, noise_std):
    """
    Randomized smoothing — large ensemble.
    Both n_samples and noise_std must be passed explicitly.
    """
    model.eval()
    B           = images.size(0)
    accumulated = torch.zeros(B, 10).to(images.device)

    with torch.no_grad():
        for _ in range(n_samples):
            noise  = torch.randn_like(images) * noise_std
            noisy  = torch.clamp(images + noise, -3.0, 3.0)
            probs  = F.softmax(model(noisy), dim=1)
            accumulated += probs

    return accumulated / n_samples


# ── Utility: feature squeezing (ablation only) ─────
CIFAR_NORM_MIN = torch.tensor([-2.429, -2.418, -2.221])
CIFAR_NORM_MAX = torch.tensor([ 2.515,  2.597,  2.751])

def apply_feature_squeezing(images, n_bits=5):
    device   = images.device
    norm_min = CIFAR_NORM_MIN.view(1, 3, 1, 1).to(device)
    norm_max = CIFAR_NORM_MAX.view(1, 3, 1, 1).to(device)
    levels   = 2 ** n_bits
    img_01   = torch.clamp(
        (images - norm_min) / (norm_max - norm_min + 1e-8), 0.0, 1.0
    )
    img_q    = torch.round(img_01 * levels) / levels
    return img_q * (norm_max - norm_min) + norm_min