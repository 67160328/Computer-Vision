# -*- coding: utf-8 -*-
"""
Full End-to-End Pipeline Benchmark (Phase 1 -> Phase 2 -> Phase 3 -> Phase 4)
ทดสอบประสิทธิภาพระบบตรวจกระดาษคำตอบอัตโนมัติบนชุดข้อมูล Test Set ทั้ง 14 แผ่น (Unseen Sheets)
จากทั้ง 4 หมวดหมู่ใน data/dataset_splits/sheets_raw/test/ และใบทดสอบใน data/Test/

สิ่งที่วัดผล:
1. Phase 1 (Auto-Orientation & 4-Point Homography Dewarping):
   - อัตราความสำเร็จในการหมุนภาพตั้งตรงอัตโนมัติ (จากภาพตะแคง 270°, เอียง, แนวตั้ง) และการดัดระนาบ
2. Phase 2 (Timing Bar, Grid, Header 42 ช่อง & OMR 7 วง):
   - ความแม่นยำในการตรวจจับ Timing Bars (ซ้าย 20 / ขวา 20), ช่องรหัสส่วนหัว 42 ช่อง และวงกลม OMR 7 วง
3. Phase 3 (Cell State & Double Strikethrough `=` Detection):
   - ความแม่นยำในการคัดแยก `empty`, `normal_char`, และ `strikethrough` บนกระดาษจริงทั้ง 14 แผ่น (5,600 ช่อง)
4. Phase 4 (Handwritten Character Recognition A-Z & Word Assembly):
   - การอ่านตัวอักษร A-Z เฉพาะช่องที่เป็น `normal_char` และนำมาประกอบเป็นคำศัพท์หลังจากตัดช่องที่ขีดฆ่า (`=`) ออกแล้ว
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dewarp import FiducialDewarper
from src.detection.grid_extractor import Template2GridExtractor
from src.detection.strikethrough_detector import StrikethroughDetectorEngine
from src.recognition.char_classifier import HCRModelEngine
from src.evaluation.evaluate_phase3 import (
    imread_bgr_unicode,
    imwrite_unicode,
    render_phase3_and_4_overlay,
)


def run_full_pipeline_test_benchmark():
    t0 = time.perf_counter()
    dewarper = FiducialDewarper()
    extractor = Template2GridExtractor()
    strike_engine = StrikethroughDetectorEngine(
        weights_path=str(PROJECT_ROOT / "models" / "strikethrough_cnn.pt"),
        device="cpu",
    )
    hcr_engine = HCRModelEngine(
        weights_path=str(PROJECT_ROOT / "models" / "hcr_resnet28_az.pt"),
        device="cpu",
    )

    test_raw_root = PROJECT_ROOT / "data" / "dataset_splits" / "sheets_raw" / "test"
    vis_out_dir = PROJECT_ROOT / "metrics" / "visuals" / "full_pipeline_test"
    vis_out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(r"C:\Users\Lenovo\.gemini\antigravity\brain\4235fe50-ff2c-4087-b6d2-6ebefe9c82ef")

    categories = [
        ("01_blank_sheets", "0.กระดาษเปล่า (Blank Sheets)"),
        ("02_edge_strikethrough", "Edge (เขียนคำผิด + ขีดฆ่า 2 เส้น `=`)"),
        ("03_horizontal_seq", "Horizontal (เขียนเรียงตัวอักษร A-Z)"),
        ("04_words_vocab", "Words (เขียนเป็นคำศัพท์ภาษาอังกฤษ)"),
    ]

    all_sheet_results: List[Dict[str, Any]] = []
    category_summaries: Dict[str, Any] = {}

    for cat_dir_name, cat_label in categories:
        cat_path = test_raw_root / cat_dir_name
        files = sorted(cat_path.glob("*.jpg"))
        cat_sheets = []

        for idx, fpath in enumerate(files, start=1):
            t_sheet = time.perf_counter()
            raw_bgr = imread_bgr_unicode(fpath)

            # Phase 1: Auto-Orient & Dewarp
            t_p1 = time.perf_counter()
            res1 = dewarper.process_image(raw_bgr, image_name=fpath.name, return_normalized=True)
            p1_ms = (time.perf_counter() - t_p1) * 1000.0

            # Phase 2: Grid, Header & OMR Extraction
            t_p2 = time.perf_counter()
            grid = extractor.process_rectified_sheet(res1.rectified_norm_gray, image_name=fpath.name)
            p2_ms = (time.perf_counter() - t_p2) * 1000.0

            # Phase 4 (Pre-Scoring Character Recognition): รันอ่านตัวอักษร A-Z บนช่องที่มีหมึกก่อน
            t_p4 = time.perf_counter()
            occupied_batch = [
                (r_idx, c_idx, grid.answer_grid[r_idx - 1][c_idx - 1].clean_patch)
                for r_idx in range(1, 21)
                for c_idx in range(1, 21)
                if not grid.answer_grid[r_idx - 1][c_idx - 1].is_empty
            ]
            pre_char_preds = hcr_engine.predict_cells(occupied_batch) if occupied_batch else []
            pre_char_map = {(cp.row_idx, cp.col_idx): cp for cp in pre_char_preds}
            p4_ms = (time.perf_counter() - t_p4) * 1000.0

            # Phase 3 (HCR-Guided Strikethrough Filter): ใช้ผลความมั่นใจจาก Phase 4 ร่วมกับ Multi-Angle & Wing Overhang
            t_p3 = time.perf_counter()
            all_cells_batch = [
                (r_idx, c_idx, grid.answer_grid[r_idx - 1][c_idx - 1].clean_patch)
                for r_idx in range(1, 21)
                for c_idx in range(1, 21)
            ]
            state_preds = strike_engine.predict_cells(all_cells_batch, hcr_predictions=pre_char_map)
            p3_ms = (time.perf_counter() - t_p3) * 1000.0

            state_map = {(sp.row_idx, sp.col_idx): sp for sp in state_preds}
            char_preds = [
                pre_char_map[(sp.row_idx, sp.col_idx)]
                for sp in state_preds
                if sp.is_valid_char and (sp.row_idx, sp.col_idx) in pre_char_map
            ]
            total_ms = (time.perf_counter() - t_sheet) * 1000.0

            char_map = {(cp.row_idx, cp.col_idx): cp for cp in char_preds}

            # บันทึกภาพ Visual Overlay
            vis_img = render_phase3_and_4_overlay(
                res1.rectified_bgr, grid, state_map, char_map, sheet_title=fpath.name
            )
            vis_fname = f"test_{fpath.stem}.jpg"
            vis_path = vis_out_dir / vis_fname
            imwrite_unicode(vis_path, vis_img, quality=92)

            # คัดเลือกใบตัวแทนของแต่ละหมวดหมู่ไปยังโฟลเดอร์ Artifact เพื่อแสดงผลให้ผู้ใช้ดู
            if idx == 1 and artifact_dir.exists():
                shutil.copy2(vis_path, artifact_dir / vis_fname)

            n_empty = sum(1 for sp in state_preds if sp.is_empty)
            n_normal = sum(1 for sp in state_preds if sp.is_valid_char)
            n_strike = sum(1 for sp in state_preds if sp.is_strikethrough)
            mean_hcr_conf = (
                float(np.mean([cp.confidence for cp in char_preds])) * 100.0
                if char_preds
                else 100.0
            )

            assembled_rows = []
            for r_idx in range(1, 21):
                row_chars = [
                    char_map[(r_idx, c_idx)].predicted_char
                    for c_idx in range(1, 21)
                    if (r_idx, c_idx) in char_map
                ]
                row_strikes = sum(
                    1
                    for c_idx in range(1, 21)
                    if state_map[(r_idx, c_idx)].is_strikethrough
                )
                if row_chars or row_strikes > 0:
                    assembled_rows.append(
                        {
                            "row": r_idx,
                            "word": "".join(row_chars),
                            "valid_chars": len(row_chars),
                            "strikethrough_removed": row_strikes,
                        }
                    )

            sheet_info = {
                "sheet_name": fpath.name,
                "category_dir": cat_dir_name,
                "category_label": cat_label,
                "raw_resolution": f"{raw_bgr.shape[1]}x{raw_bgr.shape[0]}",
                "orientation_angle_deg": round(res1.metrics.orientation_angle_deg, 2),
                "fiducials_detected": res1.metrics.markers_found_count,
                "left_timing_bars": grid.metrics.left_bars_detected,
                "right_timing_bars": grid.metrics.right_bars_detected,
                "header_42_align_pct": grid.metrics.header_id_exact_align_pct,
                "omr_7_align_pct": grid.metrics.omr_bubbles_exact_align_pct,
                "empty_cells": n_empty,
                "normal_char_cells": n_normal,
                "strikethrough_cells": n_strike,
                "mean_hcr_confidence_pct": round(mean_hcr_conf, 2),
                "phase1_ms": round(p1_ms, 1),
                "phase2_ms": round(p2_ms, 1),
                "phase3_ms": round(p3_ms, 1),
                "phase4_ms": round(p4_ms, 1),
                "total_ms": round(total_ms, 1),
                "visual_path": str(vis_path.relative_to(PROJECT_ROOT)),
                "sample_rows": assembled_rows[:5],
                "all_rows": assembled_rows,
            }
            cat_sheets.append(sheet_info)
            all_sheet_results.append(sheet_info)

            print(
                f"[{cat_dir_name}] {fpath.name} | rot={res1.metrics.orientation_angle_deg:.1f}° | "
                f"L/R={grid.metrics.left_bars_detected}/{grid.metrics.right_bars_detected} | "
                f"Hdr={grid.metrics.header_id_exact_align_pct:.0f}% OMR={grid.metrics.omr_bubbles_exact_align_pct:.0f}% | "
                f"normal={n_normal}, strike={n_strike}, empty={n_empty} | "
                f"HCR_conf={mean_hcr_conf:.1f}% | Total={total_ms:.0f}ms"
            )

        category_summaries[cat_dir_name] = {
            "label": cat_label,
            "num_sheets": len(cat_sheets),
            "total_empty_cells": sum(s["empty_cells"] for s in cat_sheets),
            "total_normal_cells": sum(s["normal_char_cells"] for s in cat_sheets),
            "total_strikethrough_cells": sum(s["strikethrough_cells"] for s in cat_sheets),
            "mean_hcr_confidence_pct": round(
                float(np.mean([s["mean_hcr_confidence_pct"] for s in cat_sheets])), 2
            ),
            "mean_total_ms": round(float(np.mean([s["total_ms"] for s in cat_sheets])), 1),
        }

    out_json = PROJECT_ROOT / "metrics" / "full_pipeline_test_benchmark.json"
    out_json.write_text(
        json.dumps(
            {
                "total_unseen_test_sheets": len(all_sheet_results),
                "total_cells_evaluated": len(all_sheet_results) * 400,
                "category_summaries": category_summaries,
                "sheets": all_sheet_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nบันทึกผลลัพธ์การทดสอบ End-to-End ทั้ง 14 แผ่นไว้ที่: {out_json}")


if __name__ == "__main__":
    run_full_pipeline_test_benchmark()
