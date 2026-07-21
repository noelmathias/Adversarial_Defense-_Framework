import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon, device):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).

    Args:
        model:    trained classifier (in eval mode)
        images:   clean input batch, shape (B, C, H, W), already normalized
        labels:   true labels, shape (B,)
        epsilon:  perturbation budget (same scale as normalized input space)
        device:   torch device

    Returns:
        adv_images:  perturbed images, clamped to valid input range
        loss.item(): scalar loss on clean inputs (for logging)
    """
    images  = images.to(device)
    labels  = labels.to(device)

    # --- Critical: require grad on INPUT, not weights ---
    adv_images = images.clone().detach().requires_grad_(True)

    criterion = nn.CrossEntropyLoss()

    # Forward pass
    outputs = model(adv_images)
    loss    = criterion(outputs, labels)

    # Backward pass — gradients w.r.t. input
    model.zero_grad()
    loss.backward()

    # Gradient sign perturbation
    gradient_sign = adv_images.grad.data.sign()
    adv_images    = adv_images + epsilon * gradient_sign

    # Clamp to valid normalized range
    # CIFAR-10 normalized: roughly [-2.4, 2.7] per channel
    # Safest approach: clamp to [-3.0, 3.0] or match clean image range
    adv_images = torch.clamp(adv_images, -3.0, 3.0).detach()

    return adv_images, loss.item()