# -*- coding: utf-8 -*-
"""
Data Preparation & Zero-Leakage Dataset Splitting Pipeline (80 : 10 : 10)
สำหรับชุดข้อมูลกระดาษคำตอบ Rayong MARCO ชุดใหม่ (New-Dataset จำนวน 145 แผ่น)

การทำงานหลัก:
1. แปลงไฟล์ภาพต้นฉบับทุกนามสกุล (.047Z, .068Z, .069Z, .316Z, .HEIC) จากโฟลเดอร์ New-Dataset/
   ให้เป็นไฟล์มาตรฐาน .jpg (JPEG Quality 96)
2. แบ่งชุดข้อมูลในระดับใบกระดาษคำตอบ (Sheet-Level Stratified Split) ด้วยอัตราส่วน Train : Val : Test = 80 : 10 : 10
   แยกตามหมวดหมู่ทั้ง 4 โฟลเดอร์ เพื่อป้องกันปัญหา Data Leakage 100%:
   - 0.กระดาษเปล่า (50 แผ่น)        -> Train: 40 | Val: 5 | Test: 5
   - Edge (เขียนคำผิด) (31 แผ่น)    -> Train: 25 | Val: 3 | Test: 3
   - Horizontal (เขียนเรียง) (32 แผ่น)-> Train: 26 | Val: 3 | Test: 3
   - Words (เขียนเป็นคำ) (32 แผ่น)   -> Train: 26 | Val: 3 | Test: 3
   รวมทั้งสิ้น 145 แผ่น              -> Train: 117 (80.69%) | Val: 14 (9.66%) | Test: 14 (9.66%)
3. บันทึกภาพกระดาษคำตอบทั้ง 2 รูปแบบ:
   - data/dataset_splits/sheets_raw/{train,val,test}/<category>/*.jpg
     (คงมุมกล้องและทิศทางภาพต้นฉบับไว้ 100% เพื่อใช้ทดสอบความยืดหยุ่นของระบบ Auto-Orientation)
   - data/dataset_splits/sheets_rectified/{train,val,test}/<category>/*.jpg
     (ภาพที่ผ่านการหมุนตั้งตรงอัตโนมัติและดัดระนาบ 4-Point Projective Homography ขนาด 2480x3508)
4. ตัดแบ่งภาพช่องคำตอบย่อย (Cropped Cell Patches) ออกเป็น 3 คลาสตามใบกระดาษต้นสังกัด:
   - strikethrough : ช่องที่มีการขีดฆ่า 2 เส้นขนานแนวนอน (=) ทับตัวอักษรที่เขียนผิด
   - normal_char   : ช่องตัวอักษรภาษาอังกฤษเขียนด้วยลายมือปกติ (A-Z)
   - empty         : ช่องว่างเปล่า (ไม่มีการเขียน)
5. สร้างไฟล์สรุปรายงานและ Manifest:
   - data/dataset_splits/dataset_manifest.json
   - data/dataset_splits/dataset_summary.md
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dewarp import FiducialDewarper
from src.detection.grid_extractor import Template2GridExtractor
from src.recognition.char_classifier import HCRModelEngine


CATEGORY_MAP = {
    "0.กระดาษเปล่า": "01_blank_sheets",
    "Edge (เขียนคำผิด)": "02_edge_strikethrough",
    "Horizontal (เขียนเรียง)": "03_horizontal_seq",
    "Words (เขียนเป็นคำ)": "04_words_vocab",
}

# อัตราส่วน 80 : 10 : 10 แบบ Stratified ตามจำนวนแผ่นจริงในแต่ละโฟลเดอร์
SPLIT_COUNTS = {
    "0.กระดาษเปล่า": (40, 5, 5),          # รวม 50 แผ่น
    "Edge (เขียนคำผิด)": (25, 3, 3),      # รวม 31 แผ่น
    "Horizontal (เขียนเรียง)": (26, 3, 3), # รวม 32 แผ่น
    "Words (เขียนเป็นคำ)": (26, 3, 3),     # รวม 32 แผ่น
}


def load_any_image_bgr(path: Path) -> np.ndarray:
    """อ่านไฟล์ภาพได้ทุกนามสกุล (.047Z, .068Z, .069Z, .316Z, .HEIC, .jpg, .png)"""
    raw_bytes = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if img is None:
        pil_img = Image.open(str(path)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError(f"ไม่สามารถอ่านไฟล์ภาพได้: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray, quality: int = 96) -> bool:
    """บันทึกไฟล์ภาพรองรับพาธภาษาไทย"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, encoded = cv2.imencode(ext, img, params)
    if ok:
        encoded.tofile(str(path))
    return ok


def analyze_double_strikethrough_features(
    crop_gray: np.ndarray,
    hcr_conf: float,
    hcr_char: str,
) -> Dict[str, Any]:
    """
    สกัดคุณลักษณะเชิงสัณฐานวิทยา (Morphological Horizontal Stroke Profile) ร่วมกับ
    ความมั่นใจของโมเดลตัวอักษร (HCR Confidence) เพื่อตรวจจับรอยขีดฆ่า 2 เส้น (=)
    """
    ch, cw = crop_gray.shape[:2]
    m = 5
    inner = crop_gray[m : ch - m, m : cw - m]
    ih, iw = inner.shape[:2]
    inv = cv2.adaptiveThreshold(
        inner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    kw = max(18, int(round(iw * 0.45)))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kh)
    horiz_d = cv2.dilate(
        horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2)), iterations=1
    )

    n_labels, _, stats, cents = cv2.connectedComponentsWithStats(horiz_d)
    bands = []
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if bw >= iw * 0.45 and area >= 24:
            bands.append(
                {
                    "cy": float(cents[i][1]),
                    "bw": int(bw),
                    "area": int(area),
                    "x1": int(x),
                    "x2": int(x + bw),
                    "bw_ratio": float(bw) / float(iw),
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

    best_pair = None
    best_score = -1.0
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            b1, b2 = merged[i], merged[j]
            dy = abs(b2["cy"] - b1["cy"])
            if 6.5 <= dy <= 34.0:
                in_mid1 = 9.0 <= b1["cy"] <= ih - 9.0
                in_mid2 = 9.0 <= b2["cy"] <= ih - 9.0
                if not (in_mid1 or in_mid2):
                    continue
                min_bw_r = min(b1["bw_ratio"], b2["bw_ratio"])
                mean_bw_r = 0.5 * (b1["bw_ratio"] + b2["bw_ratio"])
                score = mean_bw_r * 2.0 + (1.0 - hcr_conf) * 1.5
                if score > best_score:
                    best_score = score
                    best_pair = (b1, b2, dy, min_bw_r, mean_bw_r)

    is_strike = False
    pair_dy = 0.0
    mean_bw_ratio = 0.0
    if best_pair is not None:
        b1, b2, pair_dy, min_bw_r, mean_bw_ratio = best_pair
        if hcr_conf < 0.78:
            is_strike = True
        elif mean_bw_ratio >= 0.66 and min_bw_r >= 0.55:
            is_strike = True
        elif len(merged) >= 3 and mean_bw_ratio >= 0.56 and hcr_conf < 0.92:
            is_strike = True

    return {
        "is_strike": is_strike,
        "num_horiz_bands": len(merged),
        "pair_dy": round(float(pair_dy), 2),
        "mean_bw_ratio": round(float(mean_bw_ratio), 4),
        "hcr_conf": round(float(hcr_conf), 4),
        "hcr_char": hcr_char,
    }


def run_dataset_preparation():
    t0 = time.perf_counter()
    new_ds_dir = PROJECT_ROOT / "New-Dataset"
    out_root = PROJECT_ROOT / "data" / "dataset_splits"

    if out_root.exists():
        shutil.rmtree(out_root)

    dewarper = FiducialDewarper()
    extractor = Template2GridExtractor()
    hcr_engine = HCRModelEngine(
        weights_path=str(PROJECT_ROOT / "models" / "hcr_resnet28_az.pt"),
        device="cpu",
    )

    rng = np.random.default_rng(seed=20261017)

    manifest: Dict[str, Any] = {
        "dataset_name": "Rayong MARCO New-Dataset (Converted JPG & 80:10:10 Split)",
        "split_ratio_target": "80:10:10 (Train:Val:Test)",
        "zero_data_leakage": True,
        "categories": {},
        "splits_summary": {
            "train": {"sheets": 0, "cells": {"strikethrough": 0, "normal_char": 0, "empty": 0}},
            "val": {"sheets": 0, "cells": {"strikethrough": 0, "normal_char": 0, "empty": 0}},
            "test": {"sheets": 0, "cells": {"strikethrough": 0, "normal_char": 0, "empty": 0}},
        },
        "sheets": [],
    }

    total_sheets_processed = 0

    for orig_cat_name, safe_cat_name in CATEGORY_MAP.items():
        cat_dir = new_ds_dir / orig_cat_name
        files = sorted([p for p in cat_dir.iterdir() if p.is_file()])
        n_total = len(files)
        n_train, n_val, n_test = SPLIT_COUNTS[orig_cat_name]
        assert n_train + n_val + n_test == n_total, f"Count mismatch for {orig_cat_name}"

        indices = np.arange(n_total)
        rng.shuffle(indices)

        split_assignments = {}
        for rank, idx in enumerate(indices):
            if rank < n_train:
                split_assignments[idx] = "train"
            elif rank < n_train + n_val:
                split_assignments[idx] = "val"
            else:
                split_assignments[idx] = "test"

        manifest["categories"][orig_cat_name] = {
            "safe_name": safe_cat_name,
            "total_sheets": n_total,
            "train_sheets": n_train,
            "val_sheets": n_val,
            "test_sheets": n_test,
        }

        print(f"\n=== หมวดหมู่: {orig_cat_name} ({n_total} แผ่น -> Train:{n_train}, Val:{n_val}, Test:{n_test}) ===")

        for idx, fpath in enumerate(files):
            split_name = split_assignments[idx]
            sheet_id = f"{safe_cat_name}_{idx + 1:03d}"
            jpg_filename = f"{sheet_id}.jpg"

            # 1. โหลดภาพต้นฉบับและบันทึกเป็น .jpg ใน sheets_raw (คงมุมกล้องและ Orientation เดิมไว้ 100%)
            raw_bgr = load_any_image_bgr(fpath)
            raw_out_path = out_root / "sheets_raw" / split_name / safe_cat_name / jpg_filename
            imwrite_unicode(raw_out_path, raw_bgr, quality=96)

            # 2. หมุนภาพอัตโนมัติ (Auto-Orientation) + ดัดระนาบ 4-Point Projective Homography
            res1 = dewarper.process_image(raw_bgr, image_name=sheet_id, return_normalized=True)
            rect_out_path = out_root / "sheets_rectified" / split_name / safe_cat_name / jpg_filename
            imwrite_unicode(rect_out_path, res1.rectified_bgr, quality=96)

            # 3. สกัดตาราง 400 ช่องคำตอบ
            grid = extractor.process_rectified_sheet(res1.rectified_norm_gray, image_name=sheet_id)

            occupied_batch = []
            empty_cells_list = []
            for r_idx in range(1, 21):
                for c_idx in range(1, 21):
                    cell = grid.answer_grid[r_idx - 1][c_idx - 1]
                    if orig_cat_name == "0.กระดาษเปล่า" or cell.is_empty:
                        empty_cells_list.append((r_idx, c_idx, cell.clean_patch))
                    else:
                        occupied_batch.append((r_idx, c_idx, cell.clean_patch))

            preds = hcr_engine.predict_cells(occupied_batch) if occupied_batch else []

            sheet_strike_count = 0
            sheet_normal_count = 0
            sheet_empty_saved = 0

            # 4. จำแนกและบันทึกช่องที่มีหมึก (strikethrough vs normal_char)
            from src.preprocessing.upgrade_dataset_labels import is_true_double_strikethrough
            for (r_idx, c_idx, crop_patch), pred in zip(occupied_batch, preds):
                # ในหมวด Words (เขียนเป็นคำ) และ Horizontal (เขียนเรียง A-Z) เป็นตัวอักษรปกติล้วน 100%
                if orig_cat_name in ("Words (เขียนเป็นคำ)", "Horizontal (เขียนเรียง)"):
                    cell_class = "normal_char"
                else:
                    is_strike = is_true_double_strikethrough(
                        crop_patch, pred.confidence, pred.predicted_char
                    )
                    cell_class = "strikethrough" if is_strike else "normal_char"

                cell_fname = f"{sheet_id}_r{r_idx:02d}_c{c_idx:02d}_{pred.predicted_char}.jpg"
                cell_out_path = out_root / "cells" / split_name / cell_class / cell_fname
                imwrite_unicode(cell_out_path, crop_patch, quality=96)

                manifest["splits_summary"][split_name]["cells"][cell_class] += 1
                if cell_class == "strikethrough":
                    sheet_strike_count += 1
                else:
                    sheet_normal_count += 1

            # 5. บันทึกช่องว่างตัวแทน (empty) จากใบกระดาษนี้ลงใน split เดียวกัน
            max_empty_per_sheet = 40 if orig_cat_name == "0.กระดาษเปล่า" else 20
            if len(empty_cells_list) > max_empty_per_sheet:
                step = len(empty_cells_list) / float(max_empty_per_sheet)
                selected_empty = [empty_cells_list[int(i * step)] for i in range(max_empty_per_sheet)]
            else:
                selected_empty = empty_cells_list

            for r_idx, c_idx, crop_patch in selected_empty:
                cell_fname = f"{sheet_id}_r{r_idx:02d}_c{c_idx:02d}_empty.jpg"
                cell_out_path = out_root / "cells" / split_name / "empty" / cell_fname
                imwrite_unicode(cell_out_path, crop_patch, quality=95)
                manifest["splits_summary"][split_name]["cells"]["empty"] += 1
                sheet_empty_saved += 1

            manifest["splits_summary"][split_name]["sheets"] += 1
            total_sheets_processed += 1

            manifest["sheets"].append(
                {
                    "sheet_id": sheet_id,
                    "original_file": str(fpath.relative_to(PROJECT_ROOT)),
                    "original_ext": fpath.suffix,
                    "category": orig_cat_name,
                    "split": split_name,
                    "raw_jpg_path": str(raw_out_path.relative_to(PROJECT_ROOT)),
                    "rectified_jpg_path": str(rect_out_path.relative_to(PROJECT_ROOT)),
                    "raw_resolution": f"{raw_bgr.shape[1]}x{raw_bgr.shape[0]}",
                    "rotation_applied_deg": res1.metrics.orientation_angle_deg,
                    "fiducials_detected": res1.metrics.markers_found_count,
                    "timing_bars_left": grid.metrics.left_bars_detected,
                    "timing_bars_right": grid.metrics.right_bars_detected,
                    "header_align_pct": grid.metrics.header_id_exact_align_pct,
                    "omr_align_pct": grid.metrics.omr_bubbles_exact_align_pct,
                    "cells_strikethrough": sheet_strike_count,
                    "cells_normal_char": sheet_normal_count,
                    "cells_empty_total": len(empty_cells_list),
                    "cells_empty_saved": sheet_empty_saved,
                }
            )

            if (idx + 1) % 10 == 0 or (idx + 1) == n_total:
                print(
                    f"  [{idx + 1:02d}/{n_total:02d}] {sheet_id} ({split_name}) | "
                    f"rot={res1.metrics.orientation_angle_deg} deg | "
                    f"strike={sheet_strike_count}, normal={sheet_normal_count}, empty_saved={sheet_empty_saved}"
                )

    elapsed_s = time.perf_counter() - t0
    manifest["total_sheets_processed"] = total_sheets_processed
    manifest["processing_time_sec"] = round(elapsed_s, 2)

    manifest_path = out_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    tr_s = manifest["splits_summary"]["train"]
    va_s = manifest["splits_summary"]["val"]
    te_s = manifest["splits_summary"]["test"]

    tot_strike = tr_s["cells"]["strikethrough"] + va_s["cells"]["strikethrough"] + te_s["cells"]["strikethrough"]
    tot_normal = tr_s["cells"]["normal_char"] + va_s["cells"]["normal_char"] + te_s["cells"]["normal_char"]
    tot_empty = tr_s["cells"]["empty"] + va_s["cells"]["empty"] + te_s["cells"]["empty"]
    tot_cells = tot_strike + tot_normal + tot_empty

    md_lines = [
        "# รายงานการจัดเตรียมชุดข้อมูล (JPG Conversion & 80:10:10 Stratified Split)",
        "",
        "## 1. สรุปการแปลงไฟล์และแบ่งชุดข้อมูลระดับใบกระดาษ (Sheet-Level Split)",
        "- **แหล่งข้อมูลต้นฉบับ**: `New-Dataset/` (รวม 145 แผ่น จากไฟล์ `.047Z`, `.068Z`, `.069Z`, `.316Z`, `.HEIC`)",
        "- **การป้องกัน Data Leakage**: แบ่งชุดข้อมูลในระดับใบกระดาษ (Sheet-Level Stratified Split) ก่อนตัดภาพช่องย่อย ทำให้ช่องตัวอักษรจากกระดาษแผ่นเดียวกันอยู่ในชุดเดียวกัน 100%",
        "- **ความยืดหยุ่นต่อมุมกล้อง (Orientation Invariance)**:",
        "  - `data/dataset_splits/sheets_raw/`: เก็บภาพ `.jpg` ตามมุมกล้องเดิม (ตะแคง 270°, เอียง, แนวตั้ง)",
        "  - `data/dataset_splits/sheets_rectified/`: เก็บภาพ `.jpg` ที่ผ่านการหมุนตั้งตรงอัตโนมัติและดัดระนาบ 4-Point Projective Homography (`2480 x 3508` px)",
        "",
        "| หมวดหมู่ (Category) | จำนวนรวม (แผ่น) | Train (~80%) | Validate (~10%) | Test (~10%) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for orig_cat, info in manifest["categories"].items():
        md_lines.append(
            f"| `{orig_cat}` | {info['total_sheets']} | {info['train_sheets']} | {info['val_sheets']} | {info['test_sheets']} |"
        )
    md_lines.extend(
        [
            f"| **รวมทั้งหมด (Total Sheets)** | **{total_sheets_processed} (100%)** | **{tr_s['sheets']} (80.69%)** | **{va_s['sheets']} (9.66%)** | **{te_s['sheets']} (9.66%)** |",
            "",
            "## 2. สรุปจำนวนภาพระดับช่องคำตอบ (Cell-Level Cropped Patches)",
            "- ที่อยู่ไฟล์: `data/dataset_splits/cells/{train,val,test}/{strikethrough,normal_char,empty}/`",
            "",
            "| คลาสของช่องคำตอบ (Cell Class) | Train Set | Validate Set | Test Set | รวมทั้งหมด (Total Patches) |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| **`strikethrough` (ขีดฆ่า 2 เส้น `=`)** | {tr_s['cells']['strikethrough']:,} | {va_s['cells']['strikethrough']:,} | {te_s['cells']['strikethrough']:,} | **{tot_strike:,}** |",
            f"| **`normal_char` (ตัวอักษรปกติ A–Z)** | {tr_s['cells']['normal_char']:,} | {va_s['cells']['normal_char']:,} | {te_s['cells']['normal_char']:,} | **{tot_normal:,}** |",
            f"| **`empty` (ช่องว่างเปล่า)** | {tr_s['cells']['empty']:,} | {va_s['cells']['empty']:,} | {te_s['cells']['empty']:,} | **{tot_empty:,}** |",
            f"| **รวมทุกคลาส (Total Cell Patches)** | **{sum(tr_s['cells'].values()):,}** | **{sum(va_s['cells'].values()):,}** | **{sum(te_s['cells'].values()):,}** | **{tot_cells:,}** |",
            "",
        ]
    )

    summary_md_path = out_root / "dataset_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nดำเนินการเสร็จสมบูรณ์ใน {elapsed_s:.2f} วินาที!")
    print(f"Manifest: {manifest_path}")
    print(f"Summary : {summary_md_path}")


if __name__ == "__main__":
    run_dataset_preparation()
