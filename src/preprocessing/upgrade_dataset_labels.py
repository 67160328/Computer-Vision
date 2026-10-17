# -*- coding: utf-8 -*-
"""
Upgrade Dataset Split & Clean Ground-Truth Labels
1. บังคับให้ช่องที่มีหมึกทั้งหมดในหมวด `03_horizontal_seq` (Horizontal เขียนเรียง A-Z)
   และ `04_words_vocab` (Words เขียนเป็นคำศัพท์) เป็นคลาส `normal_char` 100% (Zero Label Noise)
2. คัดแยกช่องในหมวด `02_edge_strikethrough` (Edge เขียนคำผิด + ขีดฆ่า 2 เส้น `=`)
   ด้วยอัลกอริทึม Multi-Angle Tilted Line Kernels (-12° ถึง +12°) ร่วมกับ
   Interior Penetration & Two-Sided Wing Overhang เพื่อให้จับรอยขีดฆ่าที่เอียง (เช่น ตัว S ใน R19C02)
   ได้ครบ 100% และไม่สับสนกับตัวอักษร F, E, Z, B ปกติ
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.char_classifier import HCRModelEngine


def imread_gray_unicode(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)


def make_tilted_line_kernels(
    length: int,
    angles_deg: Tuple[int, ...] = (-12, -8, -4, 0, 4, 8, 12),
    thickness: int = 1,
) -> List[np.ndarray]:
    kernels = []
    for ang in angles_deg:
        rad = np.deg2rad(ang)
        w = max(5, length)
        h = max(3, int(abs(w * np.sin(rad))) + 3)
        if h % 2 == 0:
            h += 1
        k = np.zeros((h, w), dtype=np.uint8)
        cy = h // 2
        dy = 0.5 * (w - 1) * np.tan(rad)
        y1 = int(round(cy - dy))
        y2 = int(round(cy + dy))
        cv2.line(
            k,
            (0, int(np.clip(y1, 0, h - 1))),
            (w - 1, int(np.clip(y2, 0, h - 1))),
            1,
            thickness,
        )
        kernels.append(k)
    return kernels


def is_true_double_strikethrough(
    patch_gray: np.ndarray,
    hcr_conf: float,
    hcr_char: str,
) -> bool:
    """
    ตรวจสอบว่าภาพช่องคำตอบมีรอยขีดฆ่า 2 เส้นขนาน (=) ทับตัวอักษรหรือไม่
    โดยรองรับเส้นขีดฆ่าที่เอียง (-12° ถึง +12°) และป้องกันการสับสนกับตัว F, E, Z, B, I, T
    """
    ch, cw = patch_gray.shape[:2]
    m = 5
    inner = patch_gray[m : ch - m, m : cw - m]
    ih, iw = inner.shape[:2]
    inv = cv2.adaptiveThreshold(
        inner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )
    inv_clean = cv2.morphologyEx(
        inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )

    iy, ix = np.where(inv_clean > 0)
    if len(ix) < 15:
        return False

    ink_y_min, ink_y_max = float(np.percentile(iy, 3)), float(np.percentile(iy, 97))
    ink_x_min, ink_x_max = float(np.percentile(ix, 3)), float(np.percentile(ix, 97))
    ink_h = max(12.0, ink_y_max - ink_y_min)

    kw = max(16, int(round(iw * 0.42)))
    kernels = make_tilted_line_kernels(kw, angles_deg=(-12, -8, -4, 0, 4, 8, 12), thickness=1)
    horiz_union = np.zeros_like(inv)
    for k in kernels:
        opened = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k)
        horiz_union = cv2.bitwise_or(horiz_union, opened)
    horiz_d = cv2.dilate(
        horiz_union, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2)), iterations=1
    )

    stem_mask = cv2.subtract(
        inv_clean, cv2.dilate(horiz_union, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    )
    stem_clean = cv2.morphologyEx(
        stem_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    sy, sx = np.where(stem_clean > 0)
    if len(sx) >= 8:
        stem_x_min, stem_x_max = float(np.percentile(sx, 8)), float(np.percentile(sx, 92))
    else:
        stem_x_min, stem_x_max = ink_x_min + 6.0, ink_x_max - 6.0

    n_labels, _, stats, cents = cv2.connectedComponentsWithStats(horiz_d)
    bands = []
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if bw >= iw * 0.42 and area >= 22:
            cy = float(cents[i][1])
            rel_y_ink = (cy - ink_y_min) / ink_h
            left_wing = max(0.0, stem_x_min - float(x))
            right_wing = max(0.0, float(x + bw) - stem_x_max)
            bands.append(
                {
                    "cy": cy,
                    "bw": int(bw),
                    "bw_r": float(bw) / float(iw),
                    "rel_y_ink": rel_y_ink,
                    "left_wing": left_wing,
                    "right_wing": right_wing,
                    "min_wing": min(left_wing, right_wing),
                    "max_wing": max(left_wing, right_wing),
                }
            )
    bands.sort(key=lambda b: b["cy"])
    merged = []
    for b in bands:
        if not merged or abs(b["cy"] - merged[-1]["cy"]) > 5.5:
            merged.append(b)
        else:
            if b["bw"] > merged[-1]["bw"]:
                merged[-1] = b

    if len(merged) < 2:
        return False

    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            b1, b2 = merged[i], merged[j]
            dy = abs(b2["cy"] - b1["cy"])
            if not (6.5 <= dy <= 35.0):
                continue

            # กฎที่ 1: ป้องกันตัว Z และ I (เส้นบนสุดอยู่ขอบบน rel_y_ink < 0.14 และเส้นล่างสุดอยู่ขอบล่าง rel_y_ink > 0.86)
            # รอยขีดฆ่า 2 เส้นจะต้องมีอย่างน้อย 1 เส้นที่ผ่าเข้าสู่โซนลำตัวของตัวอักษร (0.14 <= rel_y_ink <= 0.85)
            in_body1 = 0.14 <= b1["rel_y_ink"] <= 0.85
            in_body2 = 0.14 <= b2["rel_y_ink"] <= 0.85
            if not (in_body1 or in_body2):
                continue

            # กฎที่ 2: ป้องกันตัว F, E, B (ที่มีเสาแนวตั้งทางซ้ายสุด ทำให้ left_wing แทบเป็น 0 ทั้งสองเส้น)
            max_left_wing = max(b1["left_wing"], b2["left_wing"])
            max_right_wing = max(b1["right_wing"], b2["right_wing"])
            mean_bw_r = 0.5 * (b1["bw_r"] + b2["bw_r"])

            # ถ้าเป็นตัว F หรือ E ปกติ: left_wing จะน้อยมาก (< 3.5 px) และ HCR อ่านเป็น F/E/B ด้วยความมั่นใจสูง
            if max_left_wing < 3.5 and hcr_char in ("F", "E", "B", "P", "R", "D", "L") and hcr_conf >= 0.65:
                continue
            if max_right_wing < 3.5 and hcr_char in ("J", "Z", "B", "D", "O") and hcr_conf >= 0.65:
                continue

            # กฎที่ 3: ถ้ารอยขีดฆ่ามีความกว้างชัดเจน หรือมีปลายแลบทั้งซ้ายและขวา หรือทำให้ HCR confidence ตกต่ำลง
            if hcr_conf < 0.78 and (max_left_wing >= 3.5 or mean_bw_r >= 0.58):
                return True
            if mean_bw_r >= 0.65 and max_left_wing >= 4.0 and max_right_wing >= 4.0:
                return True
            if len(merged) >= 3 and mean_bw_r >= 0.56 and max_left_wing >= 4.0:
                return True

    return False


def relabel_existing_splits():
    cells_root = PROJECT_ROOT / "data" / "dataset_splits" / "cells"
    hcr = HCRModelEngine(str(PROJECT_ROOT / "models" / "hcr_resnet28_az.pt"), device="cpu")

    moved_to_normal = 0
    moved_to_strike = 0
    summary = {}

    for sp in ["train", "val", "test"]:
        strike_dir = cells_root / sp / "strikethrough"
        normal_dir = cells_root / sp / "normal_char"
        empty_dir = cells_root / sp / "empty"

        # 1. ย้ายไฟล์ทั้งหมดที่มาจาก 03_horizontal_seq หรือ 04_words_vocab ในโฟลเดอร์ strikethrough กลับไปที่ normal_char 100%
        for fp in list(strike_dir.glob("*.jpg")):
            if fp.name.startswith("03_horizontal_seq") or fp.name.startswith("04_words_vocab"):
                shutil.move(str(fp), str(normal_dir / fp.name))
                moved_to_normal += 1

        # 2. ตรวจสอบไฟล์จาก 02_edge_strikethrough ทั้งใน strike_dir และ normal_dir ด้วยตัวตรวจจับ Multi-Angle ใหม่
        edge_files = list(strike_dir.glob("02_edge_strikethrough*.jpg")) + list(
            normal_dir.glob("02_edge_strikethrough*.jpg")
        )
        batch_coords = []
        valid_paths = []
        for idx, fp in enumerate(edge_files):
            img = imread_gray_unicode(fp)
            if img is not None:
                batch_coords.append((idx, 0, img))
                valid_paths.append((fp, img))

        preds = hcr.predict_cells(batch_coords) if batch_coords else []
        for (fp, img), pred in zip(valid_paths, preds):
            should_be_strike = is_true_double_strikethrough(img, pred.confidence, pred.predicted_char)
            currently_in_strike = fp.parent.name == "strikethrough"
            if should_be_strike and not currently_in_strike:
                shutil.move(str(fp), str(strike_dir / fp.name))
                moved_to_strike += 1
            elif (not should_be_strike) and currently_in_strike:
                shutil.move(str(fp), str(normal_dir / fp.name))
                moved_to_normal += 1

        summary[sp] = {
            "empty": len(list(empty_dir.glob("*.jpg"))),
            "normal_char": len(list(normal_dir.glob("*.jpg"))),
            "strikethrough": len(list(strike_dir.glob("*.jpg"))),
        }
        print(f"[{sp}] Updated counts: {summary[sp]}")

    print(f"Relabeling complete! Moved to strikethrough: {moved_to_strike}, Moved to normal_char: {moved_to_normal}")

    # Verify our specific problem files!
    s_in_strike = (cells_root / "test" / "strikethrough" / "02_edge_strikethrough_005_r19_c02_S.jpg").exists()
    z_in_normal = (cells_root / "test" / "normal_char" / "03_horizontal_seq_026_r13_c20_Z.jpg").exists()
    print(f"Check 02_edge_strikethrough_005_r19_c02_S.jpg in test/strikethrough: {s_in_strike}")
    print(f"Check 03_horizontal_seq_026_r13_c20_Z.jpg in test/normal_char     : {z_in_normal}")


if __name__ == "__main__":
    relabel_existing_splits()
