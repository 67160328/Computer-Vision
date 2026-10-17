# -*- coding: utf-8 -*-
"""
Phase 4B Training Script: Handwritten Digit Recognition (0-9) for Header ID Boxes
ฝึกสอนโมเดล HCRResNet28 (10 คลาส 0-9) จากชุดข้อมูล `data/Number Handwriter` (21,555 ภาพ)
โดยแบ่งข้อมูลแบบ Stratified 80% Train / 10% Val / 10% Test
พร้อมเทคนิค Exam-Header Domain Augmentation (สุ่มความหนาเส้นปากกา, หมุน/ย่อขยายเล็กน้อย)
และบันทึกน้ำหนักโมเดลที่ `models/hcr_resnet28_digits.pt`
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Tuple, List, Dict
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.char_classifier import HCRResNet28, CanonicalGlyphNormalizer
from src.recognition.digit_classifier import HeaderDigitNormalizer, DIGITS


class DigitHandwrittenDataset(Dataset):
    """Dataset สำหรับเทรนและทดสอบภาพตัวเลข 0-9 ขนาด 28x28"""
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        mean_norm: float = 0.1307,
        std_norm: float = 0.3081,
        augment: bool = False,
    ):
        self.images = images
        self.labels = labels
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx].copy()  # (28, 28) uint8
        label = int(self.labels[idx])

        if self.augment:
            # 1. สุ่มปรับความหนาเส้น (Erosion / Dilation) เลียนแบบปากกาหัวแหลมและหัวหมึกซึม
            r_morph = np.random.rand()
            if r_morph < 0.22:
                img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)
            elif r_morph < 0.48:
                img = cv2.dilate(img, np.ones((2, 2), np.uint8), iterations=1)

            # 2. สุ่มหมุน (-9 ถึง +9 องศา), ย่อ/ขยาย (0.88 ถึง 1.12), และเลื่อนตำแหน่ง (-2 ถึง +2 px)
            angle = np.random.uniform(-9.0, 9.0)
            scale = np.random.uniform(0.88, 1.12)
            tx = np.random.uniform(-2.0, 2.0)
            ty = np.random.uniform(-2.0, 2.0)
            M = cv2.getRotationMatrix2D((13.5, 13.5), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(
                img, M, (28, 28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

            # 3. สุ่มปรับความเข้มหมึกและเติม Gaussian noise เล็กน้อย
            gain = np.random.uniform(0.80, 1.15)
            noise = np.random.normal(0.0, 5.0, img.shape)
            img = np.clip(img.astype(np.float32) * gain + noise, 0.0, 255.0).astype(np.uint8)

        x_norm = (img.astype(np.float32) / 255.0 - self.mean_norm) / self.std_norm
        return torch.from_numpy(x_norm[None, :, :]), torch.tensor(label, dtype=torch.long)


def load_and_split_digit_dataset(
    dataset_dir: Path, cache_npz_path: Path, seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    โหลดภาพจาก `data/Number Handwriter/0..9`, แปลงเป็น 28x28 ด้วย HeaderDigitNormalizer + CanonicalGlyphNormalizer
    และแบ่ง Stratified 80% Train / 10% Val / 10% Test พร้อมบันทึก Cache .npz เพื่อความรวดเร็ว
    """
    if cache_npz_path.exists():
        print(f"[Cache] โหลดข้อมูลตัวเลข 28x28 จาก {cache_npz_path} ...")
        data = np.load(cache_npz_path)
        return {k: data[k] for k in data.files}

    print(f"[Dataset] กำลังอ่านและแปลงภาพตัวเลขจาก {dataset_dir} สู่ขนาดมาตรฐาน 28x28 ...")
    rng = np.random.default_rng(seed)

    train_imgs, train_lbls = [], []
    val_imgs, val_lbls = [], []
    test_imgs, test_lbls = [], []

    for digit_idx in range(10):
        folder = dataset_dir / str(digit_idx)
        if not folder.exists():
            raise FileNotFoundError(f"ไม่พบโฟลเดอร์ตัวเลข: {folder}")

        img_files = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )
        glyphs_cls = []
        for p in img_files:
            # รองรับ path ภาษาไทย/ช่องว่างบน Windows
            raw_bytes = np.fromfile(str(p), dtype=np.uint8)
            gray = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            cleaned = HeaderDigitNormalizer.clean_border_artifacts(gray)
            glyph = CanonicalGlyphNormalizer.to_canonical_28x28(cleaned)
            # หากภาพว่างเปล่าหลังแปลง ให้ fallback ใช้การ invert โดยตรงแล้ว resize
            if np.max(glyph) == 0:
                inv = np.clip(255.0 - cleaned.astype(np.float32), 0.0, 255.0).astype(np.uint8)
                glyph = cv2.resize(inv, (28, 28), interpolation=cv2.INTER_AREA)
            glyphs_cls.append(glyph)

        glyphs_arr = np.stack(glyphs_cls, axis=0)
        n_total = len(glyphs_arr)
        perm = rng.permutation(n_total)

        n_train = int(round(n_total * 0.80))
        n_val = int(round(n_total * 0.10))

        idx_tr = perm[:n_train]
        idx_va = perm[n_train : n_train + n_val]
        idx_te = perm[n_train + n_val :]

        train_imgs.append(glyphs_arr[idx_tr])
        train_lbls.append(np.full(len(idx_tr), digit_idx, dtype=np.int64))

        val_imgs.append(glyphs_arr[idx_va])
        val_lbls.append(np.full(len(idx_va), digit_idx, dtype=np.int64))

        test_imgs.append(glyphs_arr[idx_te])
        test_lbls.append(np.full(len(idx_te), digit_idx, dtype=np.int64))

        print(
            f"  คลาส '{digit_idx}': รวม {n_total:4d} ภาพ -> Train {len(idx_tr):4d} | Val {len(idx_va):3d} | Test {len(idx_te):3d}"
        )

    splits = {
        "train_images": np.concatenate(train_imgs, axis=0),
        "train_labels": np.concatenate(train_lbls, axis=0),
        "val_images": np.concatenate(val_imgs, axis=0),
        "val_labels": np.concatenate(val_lbls, axis=0),
        "test_images": np.concatenate(test_imgs, axis=0),
        "test_labels": np.concatenate(test_lbls, axis=0),
    }

    cache_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_npz_path, **splits)
    print(f"[Cache] บันทึกไฟล์แคช 28x28 เรียบร้อยที่ {cache_npz_path}")
    return splits


@torch.no_grad()
def evaluate_split(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """ประเมิน Top-1 Accuracy, Top-3 Accuracy, และ Loss"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top3_correct = 0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs

        preds = torch.argmax(logits, dim=1)
        top1_correct += int((preds == y).sum().item())

        _, top3 = torch.topk(logits, k=3, dim=1)
        top3_correct += int((top3 == y.unsqueeze(1)).any(dim=1).sum().item())

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    return (
        total_loss / max(1, total_samples),
        100.0 * top1_correct / max(1, total_samples),
        100.0 * top3_correct / max(1, total_samples),
        np.concatenate(all_preds),
        np.concatenate(all_targets),
    )


def compute_macro_f1(preds: np.ndarray, targets: np.ndarray, num_classes: int = 10) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """คำนวณ Macro-F1 และ Per-Class Precision/Recall/F1"""
    per_class = {}
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((preds == c) & (targets == c)))
        fp = int(np.sum((preds == c) & (targets != c)))
        fn = int(np.sum((preds != c) & (targets == c)))
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        per_class[str(c)] = {
            "precision_pct": round(prec * 100.0, 2),
            "recall_pct": round(rec * 100.0, 2),
            "f1_score": round(f1, 4),
            "support": int(np.sum(targets == c)),
        }
    return float(np.mean(f1s)), per_class


def train_digit_model():
    t_start = time.perf_counter()
    dataset_dir = PROJECT_ROOT / "data" / "Number Handwriter"
    cache_npz_path = PROJECT_ROOT / "data" / "dataset_splits" / "number_handwriter_28x28_splits.npz"
    weights_out_path = PROJECT_ROOT / "models" / "hcr_resnet28_digits.pt"

    splits = load_and_split_digit_dataset(dataset_dir, cache_npz_path, seed=42)

    train_x = splits["train_images"]
    train_y = splits["train_labels"]
    val_x = splits["val_images"]
    val_y = splits["val_labels"]
    test_x = splits["test_images"]
    test_y = splits["test_labels"]

    mean_norm = float(np.mean(train_x.astype(np.float32) / 255.0))
    std_norm = float(np.std(train_x.astype(np.float32) / 255.0)) + 1e-6
    print(f"[Stats] Train={len(train_y)}, Val={len(val_y)}, Test={len(test_y)} | mean={mean_norm:.4f}, std={std_norm:.4f}")

    train_ds = DigitHandwrittenDataset(train_x, train_y, mean_norm, std_norm, augment=True)
    val_ds = DigitHandwrittenDataset(val_x, val_y, mean_norm, std_norm, augment=False)
    test_ds = DigitHandwrittenDataset(test_x, test_y, mean_norm, std_norm, augment=False)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HCRResNet28(num_classes=10, dropout=0.20).to(device)

    # โหลด Backbone เบื้องต้นจากโมเดล A-Z (Transfer Learning จากลายมือเขียน A-Z สู่ตัวเลข 0-9)
    az_weights = PROJECT_ROOT / "models" / "hcr_resnet28_az.pt"
    if az_weights.exists():
        state_az = torch.load(az_weights, map_location=device)
        sd_az = state_az.get("model_state_dict", state_az)
        filtered_sd = {k: v for k, v in sd_az.items() if not k.startswith("fc.")}
        model.load_state_dict(filtered_sd, strict=False)
        print("[Transfer] โหลด Conv Backbone จาก hcr_resnet28_az.pt สำเร็จ")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    epochs = 18
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    history = []

    print(f"\n[Train] เริ่มเทรนโมเดลตัวเลข 0-9 จำนวน {epochs} Epochs บน {device}...")
    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            tr_loss += float(loss.item()) * bs
            tr_correct += int((torch.argmax(logits, dim=1) == y).sum().item())
            tr_total += bs

        scheduler.step()
        tr_loss /= max(1, tr_total)
        tr_acc = 100.0 * tr_correct / max(1, tr_total)

        val_loss, val_top1, val_top3, val_preds, val_targets = evaluate_split(model, val_loader, device)
        val_f1, _ = compute_macro_f1(val_preds, val_targets)
        ep_ms = (time.perf_counter() - ep_t0) * 1000.0

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss, 4),
                "train_acc": round(tr_acc, 2),
                "val_loss": round(val_loss, 4),
                "val_top1_acc": round(val_top1, 2),
                "val_top3_acc": round(val_top3, 2),
                "val_macro_f1": round(val_f1, 4),
            }
        )

        marker = ""
        if val_top1 >= best_val_acc:
            best_val_acc = val_top1
            weights_out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": 10,
                    "digits": DIGITS,
                    "mean_norm": mean_norm,
                    "std_norm": std_norm,
                    "best_val_top1_acc": round(val_top1, 4),
                    "best_val_top3_acc": round(val_top3, 4),
                },
                weights_out_path,
            )
            marker = " * [BEST]"

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | Train Acc: {tr_acc:6.2f}% | Val Top-1: {val_top1:6.2f}% | Val Top-3: {val_top3:6.2f}% | F1: {val_f1:.4f} ({ep_ms:.0f} ms){marker}"
        )

    # โหลดโมเดลที่ดีที่สุดมาวัดผลบนชุด Test Set (Unseen 10%)
    best_ckpt = torch.load(weights_out_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_top1, test_top3, test_preds, test_targets = evaluate_split(model, test_loader, device)
    test_f1, per_class_metrics = compute_macro_f1(test_preds, test_targets)

    total_sec = time.perf_counter() - t_start
    print("\n" + "=" * 72)
    print(f"[เสร็จสมบูรณ์] ผลการทดสอบบน Unseen Test Split ({len(test_y)} ภาพ):")
    print(f"  - Top-1 Accuracy : {test_top1:.2f}%")
    print(f"  - Top-3 Accuracy : {test_top3:.2f}%")
    print(f"  - Macro F1-Score : {test_f1:.4f}")
    print(f"  - บันทึกโมเดลไว้ที่ : {weights_out_path}")
    print("=" * 72)

    metrics_out = {
        "dataset": "data/Number Handwriter (0-9)",
        "total_images": int(len(train_y) + len(val_y) + len(test_y)),
        "train_samples": int(len(train_y)),
        "val_samples": int(len(val_y)),
        "test_samples": int(len(test_y)),
        "best_val_top1_accuracy_pct": round(best_val_acc, 2),
        "test_top1_accuracy_pct": round(test_top1, 2),
        "test_top3_accuracy_pct": round(test_top3, 2),
        "test_macro_f1": round(test_f1, 4),
        "training_time_sec": round(total_sec, 2),
        "per_class_test_metrics": per_class_metrics,
        "epoch_history": history,
    }

    metrics_json_path = PROJECT_ROOT / "metrics" / "digit_recognition_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    train_digit_model()
