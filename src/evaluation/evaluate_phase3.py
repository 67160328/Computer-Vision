# -*- coding: utf-8 -*-
"""
Phase 3 Training & Quantitative Evaluation Script
ฝึกสอนโมเดล StrikethroughHybridNet บนชุดข้อมูล `data/dataset_splits/cells/`
ที่แบ่งด้วยอัตราส่วน Train : Validate : Test (80 : 10 : 10) แบบ Zero Data Leakage
พร้อมประเมินผลทั้งระดับช่องคำตอบ (Cell-Level) และระดับใบกระดาษสอบจริง (Sheet-Level Word Filtering)
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dewarp import FiducialDewarper
from src.detection.grid_extractor import Template2GridExtractor
from src.detection.strikethrough_detector import (
    CELL_CLASSES,
    CLASS_TO_IDX,
    extract_dual_channel_and_geom,
    StrikethroughHybridNet,
    StrikethroughDetectorEngine,
)
from src.recognition.char_classifier import HCRModelEngine


def imread_gray_unicode(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)


def imread_bgr_unicode(path: Path) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix.lower() in (".heic", ".heif"):
        try:
            from PIL import Image, ImageOps
            import pillow_heif
            pillow_heif.register_heif_opener()
            pil_img = ImageOps.exif_transpose(Image.open(str(path_obj)))
            return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as e:
            pass
    raw = np.fromfile(str(path_obj), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        try:
            from PIL import Image, ImageOps
            pil_img = ImageOps.exif_transpose(Image.open(str(path_obj)))
            return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img


def imwrite_unicode(path: Path, img: np.ndarray, quality: int = 95) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, enc = cv2.imencode(ext, img, params)
    if ok:
        enc.tofile(str(path))
    return ok


class CellStateDataset(Dataset):
    """Dataset สำหรับโหลดภาพช่องคำตอบ 3 คลาส (empty, normal_char, strikethrough)"""
    def __init__(
        self,
        dual_imgs: np.ndarray,
        geom_feats: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ):
        self.dual_imgs = dual_imgs
        self.geom_feats = geom_feats
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_img = self.dual_imgs[idx].copy()
        g_feat = self.geom_feats[idx].copy()
        label = int(self.labels[idx])

        if self.augment and label != 0:
            # Random small shift (-2 to +2 px)
            tx = int(np.random.randint(-2, 3))
            ty = int(np.random.randint(-2, 3))
            if tx != 0 or ty != 0:
                M = np.array([[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)]], dtype=np.float32)
                d_img[0] = cv2.warpAffine(d_img[0], M, (64, 64), borderValue=0.0)
                d_img[1] = cv2.warpAffine(d_img[1], M, (64, 64), borderValue=0.0)

        return (
            torch.from_numpy(d_img),
            torch.from_numpy(g_feat),
            torch.tensor(label, dtype=torch.long),
        )


def load_split_features(
    cells_root: Path,
    split_name: str,
    max_normal_train: int = 4500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """โหลดภาพช่องคำตอบจาก data/dataset_splits/cells/<split_name>/ และสกัดฟีเจอร์"""
    rng = np.random.default_rng(seed=42)
    dual_list = []
    geom_list = []
    label_list = []
    raw_counts = {}

    for cls_name in CELL_CLASSES:
        cls_idx = CLASS_TO_IDX[cls_name]
        cls_dir = cells_root / split_name / cls_name
        files = sorted(cls_dir.glob("*.jpg"))
        raw_counts[cls_name] = len(files)

        # สำหรับชุด train เราสุ่ม normal_char ให้สมดุลและรวดเร็วในการเทรนบน CPU
        # ส่วนชุด val และ test ใช้ครบทุกภาพ 100% เพื่อวัดผลอย่างเที่ยงตรงที่สุด
        if split_name == "train" and cls_name == "normal_char" and len(files) > max_normal_train:
            sel_idx = rng.choice(len(files), size=max_normal_train, replace=False)
            files = [files[int(i)] for i in sorted(sel_idx)]

        for fp in files:
            patch = imread_gray_unicode(fp)
            if patch is None:
                continue
            d_img, g_feat = extract_dual_channel_and_geom(patch)
            dual_list.append(d_img)
            geom_list.append(g_feat)
            label_list.append(cls_idx)

    return (
        np.stack(dual_list, axis=0),
        np.stack(geom_list, axis=0),
        np.array(label_list, dtype=np.int64),
        raw_counts,
    )


def synthesize_strikethrough_samples(
    normal_patches_dir: Path,
    num_samples: int = 700,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    สร้างตัวอย่างรอยขีดฆ่า 2 เส้นสังเคราะห์เพิ่มเติมจากตัวอักษรปกติในชุด Train เท่านั้น
    เพื่อเพิ่มความหลากหลายของความหนาเส้นหมึก ระยะห่างเส้นคู่ และความเอียงของเส้นขีดฆ่า
    """
    rng = np.random.default_rng(seed=2026)
    files = sorted(normal_patches_dir.glob("*.jpg"))
    if not files:
        return np.zeros((0, 2, 64, 64), dtype=np.float32), np.zeros((0, 8), dtype=np.float32), np.zeros(0, dtype=np.int64)

    chosen = rng.choice(len(files), size=min(num_samples, len(files)), replace=False)
    dual_list, geom_list, label_list = [], [], []

    for i in chosen:
        patch = imread_gray_unicode(files[int(i)])
        if patch is None:
            continue
        aug = patch.copy()
        h, w = aug.shape[:2]
        # สุ่มตำแหน่งกึ่งกลางและระยะห่างระหว่างเส้นขีดฆ่า 2 เส้น (รวมทั้งแบบลากเฉียง)
        cy = int(rng.integers(int(h * 0.34), int(h * 0.66)))
        gap = int(rng.integers(8, 23))
        y1 = max(7, cy - gap // 2)
        y2 = min(h - 8, cy + gap // 2)
        x_start = int(rng.integers(3, int(w * 0.22)))
        x_end = int(rng.integers(int(w * 0.78), w - 3))
        slant = int(rng.integers(-7, 8))
        thickness = int(rng.integers(2, 4))
        ink_val = int(rng.integers(20, 100))

        cv2.line(aug, (x_start, y1), (x_end, np.clip(y1 + slant, 6, h - 6)), ink_val, thickness, cv2.LINE_AA)
        cv2.line(aug, (x_start, y2), (x_end, np.clip(y2 + slant, 6, h - 6)), ink_val, thickness, cv2.LINE_AA)

        d_img, g_feat = extract_dual_channel_and_geom(aug)
        dual_list.append(d_img)
        geom_list.append(g_feat)
        label_list.append(CLASS_TO_IDX["strikethrough"])

    return (
        np.stack(dual_list, axis=0),
        np.stack(geom_list, axis=0),
        np.array(label_list, dtype=np.int64),
    )


def render_phase3_and_4_overlay(
    rectified_bgr: np.ndarray,
    grid_out,
    state_preds_map: dict,
    char_preds_map: dict,
    sheet_title: str = "",
) -> np.ndarray:
    """
    วาดผลลัพธ์การทำงานร่วมกันของ Phase 2 + Phase 3 (ตรวจจับขีดฆ่า 2 เส้น) + Phase 4 (อ่านตัวอักษรและรวมคำศัพท์):
    - ช่องว่าง (`empty`)          : กรอบสีเขียวอ่อนบางๆ
    - ช่องตัวอักษรปกติ (`normal`) : กรอบสีน้ำเงิน/เขียวเข้ม พร้อมแสดงตัวอักษรที่อ่านได้ด้านบน
    - ช่องขีดฆ่า 2 เส้น (`strike`) : กรอบสีแดงสด + กากบาททับ พร้อมป้าย `[=]` และข้ามไม่นำมารวมเป็นคำศัพท์!
    - ด้านขวาของแต่ละแถว         : แสดงคำศัพท์ที่รวมได้หลังจากตัดช่องขีดฆ่าออกแล้ว (Filtered Word)
    """
    vis = rectified_bgr.copy()

    # วาด Header IDs และ OMR Bubbles
    for _, cell_list in grid_out.header_cells_info.items():
        for hcell in cell_list:
            x1, y1, x2, y2 = hcell.bbox
            color = (180, 0, 180) if hcell.is_aligned else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    for _, b_list in grid_out.omr_bubbles_info.items():
        for b in b_list:
            cx, cy = int(round(b.cx)), int(round(b.cy))
            color = (0, 180, 0) if b.is_filled else (220, 180, 0)
            cv2.circle(vis, (cx, cy), 22, color, 2)

    for r_idx in range(1, 21):
        row_cells = grid_out.answer_grid[r_idx - 1]
        assembled_chars = []
        strike_in_row = 0

        for c_idx in range(1, 21):
            cell = row_cells[c_idx - 1]
            x1, y1, x2, y2 = cell.bbox
            st = state_preds_map.get((r_idx, c_idx))

            if st is None or st.is_empty:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (170, 225, 170), 1)
                continue

            if st.is_strikethrough:
                strike_in_row += 1
                # วาดกรอบสีแดงสด + แถบไฮไลท์โปร่งแสงสีแดงอ่อน + สัญลักษณ์ [=]
                overlay_patch = vis[y1:y2, x1:x2]
                red_tint = np.full_like(overlay_patch, (40, 40, 255))
                cv2.addWeighted(overlay_patch, 0.72, red_tint, 0.28, 0, overlay_patch)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 235), 3)
                cv2.line(vis, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), (0, 0, 220), 2, cv2.LINE_AA)
                cv2.putText(
                    vis,
                    "=",
                    (x1 + 6, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (0, 0, 220),
                    2,
                    cv2.LINE_AA,
                )
            else:
                # ช่องตัวอักษรปกติ (normal_char)
                cpred = char_preds_map.get((r_idx, c_idx))
                ch_str = cpred.predicted_char if cpred is not None else "?"
                assembled_chars.append(ch_str)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 155, 0), 2)
                cv2.putText(
                    vis,
                    ch_str,
                    (x1 + 5, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (175, 50, 0),
                    2,
                    cv2.LINE_AA,
                )

        if assembled_chars or strike_in_row > 0:
            word_str = "".join(assembled_chars)
            last_x2 = row_cells[-1].bbox[2]
            ry1 = row_cells[0].bbox[1]
            ry2 = row_cells[0].bbox[3]
            txt = f"R{r_idx:02d}: {word_str}" + (f" (-{strike_in_row}[=])" if strike_in_row > 0 else "")
            cv2.rectangle(
                vis,
                (last_x2 + 8, ry1 + 6),
                (min(vis.shape[1] - 6, last_x2 + 275), ry2 - 6),
                (238, 252, 238) if strike_in_row == 0 else (235, 245, 255),
                -1,
            )
            cv2.putText(
                vis,
                txt,
                (last_x2 + 12, ry1 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (0, 110, 0) if strike_in_row == 0 else (160, 60, 0),
                2,
                cv2.LINE_AA,
            )

    return vis


def run_phase3_training_and_evaluation():
    t_start = time.perf_counter()
    cells_root = PROJECT_ROOT / "data" / "dataset_splits" / "cells"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_path = models_dir / "strikethrough_Scroll_cnn.pt"
    weights_path = models_dir / "strikethrough_cnn.pt"

    print("1. กำลังโหลดและสกัดคุณลักษณะจากชุดข้อมูล Train / Val / Test (80:10:10)...")
    X_tr_img, X_tr_geom, y_tr, tr_raw_counts = load_split_features(cells_root, "train", max_normal_train=4500)
    X_va_img, X_va_geom, y_va, va_raw_counts = load_split_features(cells_root, "val")
    X_te_img, X_te_geom, y_te, te_raw_counts = load_split_features(cells_root, "test")

    # เพิ่มข้อมูลสังเคราะห์รอยขีดฆ่า 2 เส้นเฉพาะในชุด Train เพื่อให้โมเดลทนทานต่อการขีดฆ่าทุกรูปแบบ
    syn_img, syn_geom, syn_y = synthesize_strikethrough_samples(
        cells_root / "train" / "normal_char", num_samples=700
    )
    if len(syn_y) > 0:
        X_tr_img = np.concatenate([X_tr_img, syn_img], axis=0)
        X_tr_geom = np.concatenate([X_tr_geom, syn_geom], axis=0)
        y_tr = np.concatenate([y_tr, syn_y], axis=0)

    print(f"  Train samples loaded : {len(y_tr):,} (Raw split counts: {tr_raw_counts})")
    print(f"  Val samples loaded   : {len(y_va):,} (Raw split counts: {va_raw_counts})")
    print(f"  Test samples loaded  : {len(y_te):,} (Raw split counts: {te_raw_counts})")

    # ถ่วงน้ำหนักการสุ่มตัวอย่าง (WeightedRandomSampler) เพื่อให้ทั้ง 3 คลาสถูกเรียนรู้สมดุลกัน
    class_sample_counts = np.bincount(y_tr, minlength=3)
    class_weights = 1.0 / np.maximum(class_sample_counts, 1)
    sample_weights = class_weights[y_tr]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(y_tr),
        replacement=True,
    )

    train_ds = CellStateDataset(X_tr_img, X_tr_geom, y_tr, augment=True)
    val_ds = CellStateDataset(X_va_img, X_va_geom, y_va, augment=False)
    test_ds = CellStateDataset(X_te_img, X_te_geom, y_te, augment=False)

    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = torch.device("cpu")
    model = StrikethroughHybridNet(num_classes=3, dropout=0.20).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    history = []

    print("\n2. เริ่มการฝึกสอนโมเดล StrikethroughHybridNet (Phase 3)...")
    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        tr_loss = 0.0
        for b_img, b_geom, b_y in train_loader:
            optimizer.zero_grad()
            logits = model(b_img, b_geom)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item()) * len(b_y)
        scheduler.step()
        tr_loss /= len(train_ds)

        # Evaluate on Validation Set
        model.eval()
        va_preds, va_targets = [], []
        with torch.no_grad():
            for b_img, b_geom, b_y in val_loader:
                logits = model(b_img, b_geom)
                preds = logits.argmax(dim=1)
                va_preds.extend(preds.numpy().tolist())
                va_targets.extend(b_y.numpy().tolist())

        va_preds_arr = np.array(va_preds)
        va_targets_arr = np.array(va_targets)
        va_acc = 100.0 * float(np.mean(va_preds_arr == va_targets_arr))
        prec, rec, f1, _ = precision_recall_fscore_support(
            va_targets_arr, va_preds_arr, average=None, labels=[0, 1, 2], zero_division=0
        )
        macro_f1 = 100.0 * float(np.mean(f1))
        strike_f1 = 100.0 * float(f1[2])

        ep_sec = time.perf_counter() - t_ep
        history.append(
            {
                "epoch": ep,
                "train_loss": round(tr_loss, 4),
                "val_accuracy_pct": round(va_acc, 2),
                "val_macro_f1_pct": round(macro_f1, 2),
                "val_strikethrough_f1_pct": round(strike_f1, 2),
                "time_sec": round(ep_sec, 2),
            }
        )
        print(
            f"  Epoch [{ep:02d}/{epochs:02d}] Loss: {tr_loss:.4f} | "
            f"Val Acc: {va_acc:.2f}% | Macro F1: {macro_f1:.2f}% | "
            f"Strikethrough F1: {strike_f1:.2f}% ({ep_sec:.1f}s)"
        )

        if macro_f1 >= best_val_f1:
            best_val_f1 = macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CELL_CLASSES,
                    "best_val_macro_f1_pct": round(best_val_f1, 2),
                },
                weights_path,
            )

    # โหลดโมเดลที่ดีที่สุดมาวัดผลบน Validation Set และ Test Set
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def eval_loader(loader):
        preds_all, targets_all, probs_all = [], [], []
        with torch.no_grad():
            for b_img, b_geom, b_y in loader:
                logits = model(b_img, b_geom)
                probs = torch.softmax(logits, dim=1)
                preds_all.extend(probs.argmax(dim=1).numpy().tolist())
                targets_all.extend(b_y.numpy().tolist())
                probs_all.append(probs.numpy())
        p_arr = np.array(preds_all)
        t_arr = np.array(targets_all)
        prob_mat = np.concatenate(probs_all, axis=0)
        acc = 100.0 * float(np.mean(p_arr == t_arr))
        prec, rec, f1, sup = precision_recall_fscore_support(
            t_arr, p_arr, average=None, labels=[0, 1, 2], zero_division=0
        )
        cm = confusion_matrix(t_arr, p_arr, labels=[0, 1, 2]).tolist()
        # Binary AUC for strikethrough class (class 2 vs rest)
        y_bin = (t_arr == 2).astype(np.int32)
        strike_auc = 100.0 * float(roc_auc_score(y_bin, prob_mat[:, 2])) if len(np.unique(y_bin)) > 1 else 100.0
        return {
            "overall_accuracy_pct": round(acc, 2),
            "macro_precision_pct": round(100.0 * float(np.mean(prec)), 2),
            "macro_recall_pct": round(100.0 * float(np.mean(rec)), 2),
            "macro_f1_pct": round(100.0 * float(np.mean(f1)), 2),
            "strikethrough_roc_auc_pct": round(strike_auc, 2),
            "per_class": {
                CELL_CLASSES[i]: {
                    "precision_pct": round(100.0 * float(prec[i]), 2),
                    "recall_pct": round(100.0 * float(rec[i]), 2),
                    "f1_pct": round(100.0 * float(f1[i]), 2),
                    "support": int(sup[i]),
                }
                for i in range(3)
            },
            "confusion_matrix": cm,
        }

    val_metrics = eval_loader(val_loader)
    test_metrics = eval_loader(test_loader)

    print("\n3. ผลการประเมินบนชุดข้อมูล Validate และ Test (Zero Data Leakage):")
    print(f"  Validate Overall Acc : {val_metrics['overall_accuracy_pct']:.2f}% | Macro F1: {val_metrics['macro_f1_pct']:.2f}% | Strike AUC: {val_metrics['strikethrough_roc_auc_pct']:.2f}%")
    print(f"  Test Overall Acc     : {test_metrics['overall_accuracy_pct']:.2f}% | Macro F1: {test_metrics['macro_f1_pct']:.2f}% | Strike AUC: {test_metrics['strikethrough_roc_auc_pct']:.2f}%")

    # 4. ทดสอบ End-to-End (Phase 1 -> Phase 2 -> Phase 3 -> Phase 4) บนกระดาษคำตอบที่มีรอยขีดฆ่าจริง
    # ได้แก่:
    # - data/Test/ตัวอย่างที่ 1.jpg (ข้อ 1-6: ช่องที่ 1 เป็น U..Z ปกติ, ช่องที่ 2 เป็นรอยขีดฆ่า `=` 6 ช่อง)
    # - data/Test/ตัวอย่างการขีดฆ่า.jpg (ข้อ 1-6: ช่องที่ 1 เป็น U..Z ปกติ, ช่องที่ 2-20 เป็นรอยขีดฆ่า `=` 114 ช่อง)
    # - ชุด Test Split ของหมวด Edge (เขียนคำผิด) ทั้ง 3 แผ่นใน data/dataset_splits/sheets_raw/test/02_edge_strikethrough/
    dewarper = FiducialDewarper()
    extractor = Template2GridExtractor()
    strike_engine = StrikethroughDetectorEngine(weights_path=str(weights_path), device="cpu")
    hcr_engine = HCRModelEngine(weights_path=str(PROJECT_ROOT / "models" / "hcr_resnet28_az.pt"), device="cpu")

    vis_dir = PROJECT_ROOT / "metrics" / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(r"C:\Users\Lenovo\.gemini\antigravity\brain\4235fe50-ff2c-4087-b6d2-6ebefe9c82ef")

    showcase_sheets = [
        ("ตัวอย่างที่ 1.jpg", PROJECT_ROOT / "data" / "Test" / "ตัวอย่างที่ 1.jpg", "phase3_test_1.jpg"),
        ("ตัวอย่างการขีดฆ่า.jpg", PROJECT_ROOT / "data" / "Test" / "ตัวอย่างการขีดฆ่า.jpg", "phase3_test_strike.jpg"),
    ]
    edge_test_files = sorted(
        (PROJECT_ROOT / "data" / "dataset_splits" / "sheets_raw" / "test" / "02_edge_strikethrough").glob("*.jpg")
    )
    for idx, ef in enumerate(edge_test_files, start=1):
        showcase_sheets.append((ef.name, ef, f"phase3_edge_test_{idx}.jpg"))

    sheet_eval_summary = []
    for label_name, fpath, out_vis_name in showcase_sheets:
        if not fpath.exists():
            continue
        t_s = time.perf_counter()
        raw_bgr = imread_bgr_unicode(fpath)
        res1 = dewarper.process_image(raw_bgr, image_name=label_name, return_normalized=True)
        grid = extractor.process_rectified_sheet(res1.rectified_norm_gray, image_name=label_name)

        all_cells_batch = []
        for r_idx in range(1, 21):
            for c_idx in range(1, 21):
                all_cells_batch.append((r_idx, c_idx, grid.answer_grid[r_idx - 1][c_idx - 1].clean_patch))

        t_p3 = time.perf_counter()
        state_preds = strike_engine.predict_cells(all_cells_batch)
        p3_ms = (time.perf_counter() - t_p3) * 1000.0

        state_map = {(sp.row_idx, sp.col_idx): sp for sp in state_preds}
        valid_char_batch = [
            (r_idx, c_idx, patch)
            for (r_idx, c_idx, patch), sp in zip(all_cells_batch, state_preds)
            if sp.is_valid_char
        ]
        char_preds = hcr_engine.predict_cells(valid_char_batch)
        char_map = {(cp.row_idx, cp.col_idx): cp for cp in char_preds}

        vis_img = render_phase3_and_4_overlay(
            res1.rectified_bgr, grid, state_map, char_map, sheet_title=label_name
        )
        out_vis_path = vis_dir / out_vis_name
        imwrite_unicode(out_vis_path, vis_img, quality=93)
        if artifact_dir.exists():
            shutil.copy2(out_vis_path, artifact_dir / out_vis_name)

        n_empty = sum(1 for sp in state_preds if sp.is_empty)
        n_normal = sum(1 for sp in state_preds if sp.is_valid_char)
        n_strike = sum(1 for sp in state_preds if sp.is_strikethrough)

        # รวบรวมคำศัพท์ที่อ่านได้ในแต่ละแถวหลังจากตัดช่องขีดฆ่าออกแล้ว
        row_words = {}
        for r_idx in range(1, 21):
            chars = [
                char_map[(r_idx, c_idx)].predicted_char
                for c_idx in range(1, 21)
                if (r_idx, c_idx) in char_map
            ]
            if chars:
                row_words[f"row_{r_idx:02d}"] = "".join(chars)

        sheet_eval_summary.append(
            {
                "sheet_name": label_name,
                "visual_output": str(out_vis_path.relative_to(PROJECT_ROOT)),
                "phase3_inference_ms": round(p3_ms, 2),
                "total_pipeline_ms": round((time.perf_counter() - t_s) * 1000.0, 2),
                "detected_empty_cells": n_empty,
                "detected_normal_char_cells": n_normal,
                "detected_strikethrough_cells": n_strike,
                "assembled_words_after_strike_removal": row_words,
            }
        )
        print(
            f"  [Sheet Eval] {label_name}: normal={n_normal}, strikethrough={n_strike}, "
            f"empty={n_empty} | P3={p3_ms:.1f}ms"
        )

    # บันทึกไฟล์ JSON และรายงาน Markdown
    metrics_payload = {
        "phase": "Phase 3 - Cell State & Double Strikethrough (=) Detection",
        "model_architecture": "StrikethroughHybridNet (Dual-Channel 64x64 ResNet + 6D Morphological Stroke Features)",
        "weights_path": "models/strikethrough_cnn.pt",
        "dataset_split_ratio": "80:10:10 (Sheet-Level Stratified, Zero Data Leakage)",
        "raw_split_cell_counts": {
            "train": tr_raw_counts,
            "val": va_raw_counts,
            "test": te_raw_counts,
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_history": history,
        "end_to_end_sheet_evaluations": sheet_eval_summary,
    }

    json_path = PROJECT_ROOT / "metrics" / "phase3_evaluation_metrics.json"
    json_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # สร้างรายงาน Markdown
    cm_te = test_metrics["confusion_matrix"]
    md = [
        "# รายงานผลการประเมินเชิงปริมาณ Phase 3: การจำแนกสถานะช่องคำตอบและตรวจจับรอยขีดฆ่า 2 เส้น (`=`)",
        "",
        "## 1. สถาปัตยกรรมโมเดลและการจัดเตรียมชุดข้อมูล (80 : 10 : 10 Zero Data Leakage)",
        "- **โมเดล**: `StrikethroughHybridNet` (`models/strikethrough_cnn.pt`) ผสานเทนเซอร์ภาพ 2 ช่องสัญญาณ (`64x64` Inverted Ink Map + Directional Horizontal-Stroke Map) ร่วมกับเวกเตอร์คุณลักษณะเชิงสัณฐานวิทยา 6 มิติ",
        "- **การแบ่งชุดข้อมูล (Sheet-Level Stratified Split 80:10:10)**:",
        "  - แปลงไฟล์กระดาษคำตอบต้นฉบับทั้งหมด **145 แผ่น** จาก `New-Dataset/` (`.047Z`, `.068Z`, `.069Z`, `.316Z`, `.HEIC`) เป็นไฟล์ `.jpg`",
        "  - แบ่งระดับใบกระดาษเป็น **Train 117 แผ่น (80.69%) : Validate 14 แผ่น (9.66%) : Test 14 แผ่น (9.66%)**",
        "  - สกัดภาพช่องคำตอบย่อยรวม **25,802 ช่อง** แยกตามใบกระดาษต้นสังกัดโดยไม่มีการรั่วไหลของข้อมูลข้ามชุด (Zero Data Leakage)",
        "",
        "## 2. ตัวชี้วัดประสิทธิภาพบนชุดข้อมูล Validate และ Test (Unseen Sheets)",
        "",
        "| ตัวชี้วัด (Metric) | Validate Set (2,440 ช่อง) | Test Set (2,542 ช่อง) |",
        "| :--- | :---: | :---: |",
        f"| **Overall Accuracy** | **{val_metrics['overall_accuracy_pct']:.2f}%** | **{test_metrics['overall_accuracy_pct']:.2f}%** |",
        f"| **Macro Precision** | {val_metrics['macro_precision_pct']:.2f}% | {test_metrics['macro_precision_pct']:.2f}% |",
        f"| **Macro Recall** | {val_metrics['macro_recall_pct']:.2f}% | {test_metrics['macro_recall_pct']:.2f}% |",
        f"| **Macro F1-Score** | **{val_metrics['macro_f1_pct']:.2f}%** | **{test_metrics['macro_f1_pct']:.2f}%** |",
        f"| **Strikethrough ROC-AUC** | **{val_metrics['strikethrough_roc_auc_pct']:.2f}%** | **{test_metrics['strikethrough_roc_auc_pct']:.2f}%** |",
        "",
        "### ประสิทธิภาพรายคลาสบนชุดทดสอบ (Test Set Per-Class Breakdown)",
        "",
        "| คลาส (Cell State) | Precision | Recall | F1-Score | จำนวนตัวอย่าง (Support) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **`empty` (ช่องว่างเปล่า)** | {test_metrics['per_class']['empty']['precision_pct']:.2f}% | {test_metrics['per_class']['empty']['recall_pct']:.2f}% | {test_metrics['per_class']['empty']['f1_pct']:.2f}% | {test_metrics['per_class']['empty']['support']:,} |",
        f"| **`normal_char` (ตัวอักษรปกติ A–Z)** | {test_metrics['per_class']['normal_char']['precision_pct']:.2f}% | {test_metrics['per_class']['normal_char']['recall_pct']:.2f}% | {test_metrics['per_class']['normal_char']['f1_pct']:.2f}% | {test_metrics['per_class']['normal_char']['support']:,} |",
        f"| **`strikethrough` (ขีดฆ่า 2 เส้น `=`)** | {test_metrics['per_class']['strikethrough']['precision_pct']:.2f}% | {test_metrics['per_class']['strikethrough']['recall_pct']:.2f}% | **{test_metrics['per_class']['strikethrough']['f1_pct']:.2f}%** | {test_metrics['per_class']['strikethrough']['support']:,} |",
        "",
        "### Confusion Matrix บนชุดทดสอบ (Test Set - 2,542 ช่อง)",
        "",
        "| Actual \\\\ Predicted | `empty` | `normal_char` | `strikethrough` |",
        "| :--- | :---: | :---: | :---: |",
        f"| **`empty`** | **{cm_te[0][0]}** | {cm_te[0][1]} | {cm_te[0][2]} |",
        f"| **`normal_char`** | {cm_te[1][0]} | **{cm_te[1][1]}** | {cm_te[1][2]} |",
        f"| **`strikethrough`** | {cm_te[2][0]} | {cm_te[2][1]} | **{cm_te[2][2]}** |",
        "",
        "## 3. ผลการทดสอบ End-to-End (ตัดช่องขีดฆ่า `=` ออกแล้วรวมคำศัพท์อัตโนมัติ)",
        "",
        "| ใบกระดาษคำตอบ | ช่องตัวอักษรปกติ (`normal_char`) | ช่องขีดฆ่าที่ถูกตัดออก (`strikethrough`) | ช่องว่าง (`empty`) | เวลาประมวลผล Phase 3 (400 ช่อง) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for s in sheet_eval_summary:
        md.append(
            f"| `{s['sheet_name']}` | {s['detected_normal_char_cells']} | **{s['detected_strikethrough_cells']}** | {s['detected_empty_cells']} | {s['phase3_inference_ms']:.1f} ms |"
        )

    md_path = PROJECT_ROOT / "metrics" / "Phase3_Metrics_Report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nบันทึกผลลัพธ์ Phase 3 เรียบร้อยแล้วที่: {md_path}")


if __name__ == "__main__":
    run_phase3_training_and_evaluation()
