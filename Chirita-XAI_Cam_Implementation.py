import json
import os
import tarfile
import pickle
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt



CIFAR10_TAR_GZ = r"C:\Users\User1\PycharmProjects\PhD-Emergent topics\cifar-10-python.tar.gz"

OUT_DIR = "./cam_outputs_cifar10"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 0
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
NUM_WORKERS = 2
PRINT_EVERY = 100

TOPK_VIS = 8
THRESHOLD_RATIO = 0.20

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)



def _read_cifar_batch_from_tar(tar: tarfile.TarFile, member_name: str):
    f = tar.extractfile(member_name)
    if f is None:
        raise FileNotFoundError(f"Could not extract member: {member_name}")
    data = pickle.load(f, encoding="bytes")
    X = data[b"data"]  # shape [N, 3072]
    y = data[b"labels"]
    X = X.reshape(-1, 3, 32, 32)
    return X, np.array(y, dtype=np.int64)


def load_cifar10_from_local_tar(tar_gz_path: str):
    """
    Loads CIFAR-10 from the standard 'cifar-10-python.tar.gz' archive.
    Returns:
      X_train [50000, 3,32,32], y_train [50000]
      X_test  [10000, 3,32,32], y_test  [10000]
    """
    if not os.path.exists(tar_gz_path):
        raise FileNotFoundError(f"Tar.gz not found: {tar_gz_path}")

    with tarfile.open(tar_gz_path, "r:gz") as tar:
        names = tar.getnames()

        base = "cifar-10-batches-py"
        train_batches = [f"{base}/data_batch_{i}" for i in range(1, 6)]
        test_batch = f"{base}/test_batch"

        for m in train_batches + [test_batch]:
            if m not in names:
                raise FileNotFoundError(f"Missing member inside tar: {m}")

        Xs, ys = [], []
        for m in train_batches:
            Xb, yb = _read_cifar_batch_from_tar(tar, m)
            Xs.append(Xb)
            ys.append(yb)

        X_train = np.concatenate(Xs, axis=0)
        y_train = np.concatenate(ys, axis=0)

        X_test, y_test = _read_cifar_batch_from_tar(tar, test_batch)

    return X_train, y_train, X_test, y_test


class CIFAR10Numpy(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, train: bool = True):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        img = self.X[idx]  # [3,32,32], uint8-ish
        label = int(self.y[idx])

        x = torch.tensor(img, dtype=torch.float32) / 255.0

        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
        x = (x - mean) / std

        if self.train:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
            x = F.pad(x.unsqueeze(0), (4,4,4,4), mode="reflect").squeeze(0)
            i = random.randint(0, 8)
            j = random.randint(0, 8)
            x = x[:, i:i+32, j:j+32]

        return x, label


# -------------------------
# CAM-compatible CNN: conv -> conv -> conv -> GAP -> Linear
# -------------------------
class CAM_CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),  # last conv maps
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        fmaps = self.features(x)           # [B,256,H,W] (H=W=8)
        pooled = self.gap(fmaps).flatten(1)  # [B,256]
        logits = self.classifier(pooled)   # [B,10]
        return logits, fmaps


# -------------------------
# Training / evaluation
# -------------------------
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def train_one_epoch(model, loader, opt, epoch: int):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for step, (x, y) in enumerate(loader, start=1):
        x, y = x.to(DEVICE), y.to(DEVICE)

        opt.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        running_loss += float(loss.item())
        running_acc += accuracy(logits, y)

        if step % PRINT_EVERY == 0:
            print(f"[train] epoch={epoch} step={step}/{len(loader)} "
                  f"loss={running_loss/step:.4f} acc={running_acc/step:.4f}")

    return running_loss / len(loader), running_acc / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses = []
    accs = []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, _ = model(x)
        losses.append(float(F.cross_entropy(logits, y).item()))
        accs.append(accuracy(logits, y))
    return float(np.mean(losses)), float(np.mean(accs))


# -------------------------
# CAM computation + visualization
# -------------------------
@dataclass
class CamResult:
    class_idx: int
    class_name: str
    prob: float
    cam: np.ndarray
    bbox: Optional[Tuple[int, int, int, int]]


def cam_from_maps(fmaps: torch.Tensor, classifier_w: torch.Tensor, class_idx: int) -> np.ndarray:
    """
    fmaps: [1,C,H,W]
    classifier_w: [num_classes, C]
    CAM: sum_k w_k^c * f_k
    """
    w = classifier_w[class_idx].view(-1, 1, 1)         # [C,1,1]
    cam = (fmaps[0] * w).sum(dim=0)                    # [H,W]
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    return cam.detach().cpu().numpy()


def cam_to_bbox(cam: np.ndarray, threshold_ratio: float = 0.20) -> Optional[Tuple[int, int, int, int]]:
    thr = float(cam.max()) * float(threshold_ratio)
    mask = (cam >= thr).astype(np.uint8)

    try:
        import cv2
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        x = int(stats[largest, cv2.CC_STAT_LEFT])
        y = int(stats[largest, cv2.CC_STAT_TOP])
        w = int(stats[largest, cv2.CC_STAT_WIDTH])
        h = int(stats[largest, cv2.CC_STAT_HEIGHT])
        return (x, y, x + w, y + h)
    except Exception:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return (x0, y0, x1 + 1, y1 + 1)


def denorm_cifar10(x_norm: torch.Tensor) -> np.ndarray:
    """
    x_norm: [3,32,32] tensor normalized with CIFAR-10 mean/std
    returns uint8 HWC
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
    x = x_norm.cpu() * std + mean
    x = torch.clamp(x, 0.0, 1.0)
    img = (x.permute(1,2,0).numpy() * 255.0).astype(np.uint8)
    return img


def save_overlay(img_uint8: np.ndarray, cam: np.ndarray, bbox, out_path: str, title: str):
    plt.figure(figsize=(5, 5))
    plt.imshow(img_uint8)
    plt.imshow(cam, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title(title)

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        ax = plt.gca()
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2)
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


@torch.no_grad()
def visualize_cams(model: CAM_CNN, loader: DataLoader, out_dir: str, topk: int = 8):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    saved = 0
    summary = []

    for x, y in loader:
        x = x.to(DEVICE)
        logits, fmaps = model(x)
        probs = torch.softmax(logits, dim=1)

        for i in range(x.size(0)):
            if saved >= topk:
                with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                print(f"[cam] saved {saved} overlays to {os.path.abspath(out_dir)}")
                return

            p = probs[i].detach().cpu().numpy()
            pred = int(np.argmax(p))
            prob = float(p[pred])

            cam = cam_from_maps(fmaps[i:i+1], model.classifier.weight.detach(), pred)

            # Upsample CAM from 8x8 -> 32x32
            cam_up = torch.tensor(cam)[None, None, ...]
            cam_up = F.interpolate(cam_up, size=(32, 32), mode="bilinear", align_corners=False)[0, 0].numpy()
            cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-6)

            bbox = cam_to_bbox(cam_up, threshold_ratio=THRESHOLD_RATIO)

            img_uint8 = denorm_cifar10(x[i])

            out_path = os.path.join(out_dir, f"cam_{saved:03d}_pred_{CLASSES[pred]}_true_{CLASSES[int(y[i])]}.png")
            title = f"pred={CLASSES[pred]}({prob:.2f}) true={CLASSES[int(y[i])]}"
            save_overlay(img_uint8, cam_up, bbox, out_path, title)

            summary.append({
                "idx": saved,
                "pred_class": CLASSES[pred],
                "pred_prob": prob,
                "true_class": CLASSES[int(y[i])],
                "bbox": bbox
            })

            saved += 1


def main():
    print("Device:", DEVICE)
    print("Loading CIFAR-10 from local tar.gz:")
    print(" ", CIFAR10_TAR_GZ)

    X_train, y_train, X_test, y_test = load_cifar10_from_local_tar(CIFAR10_TAR_GZ)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    train_ds = CIFAR10Numpy(X_train, y_train, train=True)
    test_ds = CIFAR10Numpy(X_test, y_test, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = CAM_CNN(num_classes=10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("\nTraining...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, epoch)
        te_loss, te_acc = evaluate(model, test_loader)
        print(f"[epoch {epoch}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | test loss={te_loss:.4f} acc={te_acc:.4f}")

    print("\nGenerating CAM overlays on test set...")
    os.makedirs(OUT_DIR, exist_ok=True)
    visualize_cams(model, test_loader, OUT_DIR, topk=TOPK_VIS)

    print("Done. Outputs:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
