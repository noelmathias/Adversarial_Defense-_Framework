import torch
import torch.nn as nn


def pgd_attack(model, images, labels, epsilon, device,
               alpha=None, num_steps=40, random_start=True):
    """
    Projected Gradient Descent attack (Madry et al., 2018).

    Iterative extension of FGSM. At each step:
        1. Take gradient sign step of size alpha
        2. Project back into epsilon-ball around original image
        3. Clamp to valid input range

    Args:
        model:        trained classifier (eval mode)
        images:       clean input batch (B, C, H, W), normalized
        labels:       true labels (B,)
        epsilon:      perturbation budget — same scale as FGSM
        device:       torch device
        alpha:        step size per iteration (default: epsilon / 4)
        num_steps:    number of PGD iterations (default: 40)
        random_start: randomize starting point within epsilon-ball

    Returns:
        adv_images:   perturbed batch, same shape as input
        final_loss:   scalar loss on final iterate (for logging)

    Interface matches fgsm_attack() exactly — drop-in replacement.
    """
    if alpha is None:
        alpha = epsilon / 2.0

    model.eval()
    images  = images.to(device)
    labels  = labels.to(device)

    criterion = nn.CrossEntropyLoss()

    # ── Random start — improves attack strength ────────
    if random_start:
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(images + delta, -3.0, 3.0).detach()
    else:
        adv_images = images.clone().detach()

    # ── Iterative gradient steps ───────────────────────
    for _ in range(num_steps):
        adv_images.requires_grad_(True)

        outputs = model(adv_images)
        loss    = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        # Gradient sign step
        grad_sign  = adv_images.grad.data.sign()
        adv_images = adv_images.detach() + alpha * grad_sign

        # Project back into epsilon-ball around original
        delta      = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + delta, -3.0, 3.0).detach()

    final_loss = criterion(model(adv_images), labels).item()
    return adv_images, final_loss