# experiments/train_resnet_colab.py
"""
CIFAR-10 ResNet-18 training script — Google Colab / GPU.

Usage in Colab
--------------
1. Mount Drive:
       from google.colab import drive
       drive.mount('/content/drive')

2. Clone / upload project, then run this file:
       !python experiments/train_resnet_colab.py

3. After training, copy checkpoint into your project:
       import shutil
       shutil.copy('/content/drive/MyDrive/agentic_defense/resnet_best.pth',
                   'checkpoints/cnn_best.pth')

Training recipe (achieves >90% clean accuracy)
-----------------------------------------------
Optimizer  : SGD, lr=0.1, momentum=0.9, weight_decay=5e-4, Nesterov=True
Scheduler  : CosineAnnealingLR, T_max=200, eta_min=1e-4
LR warmup  : 5-epoch linear warmup (prevents early divergence)
Augment    : RandomCrop(32,pad=4) + RandomHorizontalFlip + Cutout(16)
Label smooth: 0.1
Batch size : 128
Mixed prec : torch.cuda.amp (2× throughput on Colab GPU)
Early stop : patience=30 epochs
Checkpoints: saved to Drive + local checkpoints/
"""

import os
import sys
import json
import time
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── allow running from project root or experiments/ ─────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.resnet import ResNet18


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    epochs       = 200,
    batch_size   = 128,
    lr           = 0.1,
    weight_decay = 5e-4,
    warmup_epochs= 5,
    patience     = 30,          # early stopping
    num_classes  = 10,
    cutout_size  = 16,
    label_smooth = 0.1,
    # Colab Drive path — change if needed
    drive_dir    = "/content/drive/MyDrive/agentic_defense",
    local_ckpt   = "checkpoints",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# ─────────────────────────────────────────────────────────────────────────────
# Cutout augmentation
# ─────────────────────────────────────────────────────────────────────────────

class Cutout:
    """
    Randomly zeros a square patch.
    Adds ~0.8–1.2% accuracy on CIFAR-10 (DeVries & Taylor 2017).
    Applied after ToTensor so the patch is zeroed in normalized space.
    """
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1 = max(0, cy - self.size // 2)
        y2 = min(h, cy + self.size // 2)
        x1 = max(0, cx - self.size // 2)
        x2 = min(w, cx + self.size // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(batch_size: int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        Cutout(CFG["cutout_size"]),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    n_workers = 2   # stable on Colab

    train_ds = datasets.CIFAR10("./data", train=True,  download=True,
                                 transform=train_tf)
    test_ds  = datasets.CIFAR10("./data", train=False, download=True,
                                 transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False,
                              num_workers=n_workers, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosine:
    """
    Linear warmup for `warmup` epochs, then CosineAnnealingLR.
    Prevents loss spikes from large random-init gradients at epoch 1.
    """
    def __init__(self, opt, base_lr, warmup, total):
        self.opt      = opt
        self.base_lr  = base_lr
        self.warmup   = warmup
        self.cosine   = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total - warmup, eta_min=1e-4
        )
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup:
            lr = self.base_lr * self._step / self.warmup
            for pg in self.opt.param_groups:
                pg["lr"] = lr
        else:
            self.cosine.step()

    def current_lr(self):
        return self.opt.param_groups[0]["lr"]


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, opt, criterion, device, scaler):
    model.train()
    loss_sum = correct = total = 0

    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        with autocast():
            out  = model(imgs)
            loss = criterion(out, lbls)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(lbls).sum().item()
        total    += imgs.size(0)

    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(lbls).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, 100.0 * correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    if device.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")

    os.makedirs(CFG["local_ckpt"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Try to create Drive directory; skip silently if not mounted
    try:
        os.makedirs(CFG["drive_dir"], exist_ok=True)
        use_drive = True
        print(f"Drive   : {CFG['drive_dir']}")
    except Exception:
        use_drive = False
        print("Drive   : not mounted — saving locally only")

    train_loader, test_loader = build_loaders(CFG["batch_size"])

    model     = ResNet18(num_classes=CFG["num_classes"]).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Params  : {n_params:,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smooth"])
    optimizer = optim.SGD(
        model.parameters(),
        lr=CFG["lr"],
        momentum=0.9,
        weight_decay=CFG["weight_decay"],
        nesterov=True,
    )
    scheduler = WarmupCosine(
        optimizer,
        base_lr=CFG["lr"],
        warmup=CFG["warmup_epochs"],
        total=CFG["epochs"],
    )
    scaler = GradScaler()

    best_acc     = 0.0
    patience_cnt = 0
    history      = {"train_loss":[], "train_acc":[],
                    "test_loss":[],  "test_acc":[]}

    print(f"{'Ep':>4} {'TrLoss':>8} {'TrAcc':>8} "
          f"{'TeLoss':>8} {'TeAcc':>8} {'LR':>9} {'s':>5}")
    print("─" * 58)

    for ep in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        lr = scheduler.current_lr()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        flag = " ◀" if te_acc > best_acc else ""
        print(f"{ep:>4} {tr_loss:>8.4f} {tr_acc:>7.2f}% "
              f"{te_loss:>8.4f} {te_acc:>7.2f}% "
              f"{lr:>9.6f} {time.time()-t0:>4.0f}s{flag}")

        if te_acc > best_acc:
            best_acc     = te_acc
            patience_cnt = 0

            # Save locally under both canonical names
            local_best = os.path.join(CFG["local_ckpt"], "resnet_best.pth")
            local_cnn  = os.path.join(CFG["local_ckpt"], "cnn_best.pth")
            torch.save(model.state_dict(), local_best)
            shutil.copy(local_best, local_cnn)

            # Mirror to Drive if available
            if use_drive:
                drive_best = os.path.join(CFG["drive_dir"], "resnet_best.pth")
                drive_cnn  = os.path.join(CFG["drive_dir"], "cnn_best.pth")
                shutil.copy(local_best, drive_best)
                shutil.copy(local_best, drive_cnn)
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                print(f"\nEarly stopping at epoch {ep} "
                      f"(no improvement for {CFG['patience']} epochs).")
                break

    # Save training history
    hist_path = "logs/resnet_training_history.json"
    with open(hist_path, "w") as f:
        json.dump({"best_test_acc": best_acc, "history": history}, f, indent=2)

    print(f"\n{'─'*58}")
    print(f"Best test accuracy : {best_acc:.2f}%")
    if best_acc >= 90.0:
        print("Target ≥90%        : ✓ REACHED")
    else:
        print(f"Target ≥90%        : ✗ not reached — try more epochs")
    print(f"Checkpoint (local) : {local_best}")
    if use_drive:
        print(f"Checkpoint (Drive) : {drive_best}")
    print(f"History            : {hist_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Colab notebook cell — paste everything below into one cell
# ─────────────────────────────────────────────────────────────────────────────
"""
# ── PASTE INTO COLAB CELL ──────────────────────────────────────────────────
# Step 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Clone your repo (or upload files)
# !git clone https://github.com/your-repo/agentic-defense.git
# %cd agentic-defense

# Step 3: Install dependencies
!pip install torch torchvision -q

# Step 4: Train
!python experiments/train_resnet_colab.py

# Step 5 (after training): confirm checkpoint exists
import os
print("resnet_best.pth exists:", os.path.exists("checkpoints/resnet_best.pth"))
print("cnn_best.pth    exists:", os.path.exists("checkpoints/cnn_best.pth"))
# ──────────────────────────────────────────────────────────────────────────
"""

if __name__ == "__main__":
    train()