# -*- coding: utf-8 -*-
"""
Test script for evaluating Phase 1 (Dewarping) and Phase 2 (Grid & Cell Localization)
on the 5 handwritten test sheets in data/Test/.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dewarp import FiducialDewarper
from src.detection.grid_extractor import Template2GridExtractor


def imread_unicode(path: Path) -> np.ndarray:
    stream = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    ext = path.suffix
    result, encoded = cv2.imencode(ext, img)
    if result:
        encoded.tofile(str(path))
    return result


def main():
    test_dir = PROJECT_ROOT / "data" / "Test"
    out_dir = PROJECT_ROOT / "data" / "processed" / "test_sheets_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    dewarper = FiducialDewarper()
    extractor = Template2GridExtractor()

    test_files = [
        ("ตัวอย่างที่ 1.jpg", "test_1_grid.jpg"),
        ("ตัวอย่างที่ 2.jpg", "test_2_grid.jpg"),
        ("ตัวอย่างที่ 3.jpg", "test_3_grid.jpg"),
        ("ตัวอย่างที่ 4.jpg", "test_4_grid.jpg"),
        ("ตัวอย่างการขีดฆ่า.jpg", "test_strike_grid.jpg"),
    ]

    for fname, out_name in test_files:
        fpath = test_dir / fname
        img_bgr = imread_unicode(fpath)
        if img_bgr is None:
            print(f"[ERROR] Could not read {fname}")
            continue

        res1 = dewarper.process_image(img_bgr, image_name=fname, return_normalized=True)
        m1 = res1.metrics
        print(f"=== {fname} -> {out_name} (Size: {img_bgr.shape[1]}x{img_bgr.shape[0]}) ===")
        print(f"  Phase 1 Dewarp : Success={m1.success} | Markers={m1.markers_found_count}/3 | RMSE={m1.reprojection_rmse_px:.3f} px | Fallback={m1.used_fallback}")

        out2 = extractor.process_rectified_sheet(res1.rectified_norm_gray, image_name=fname)
        m2 = out2.metrics
        vis = extractor.render_visualization(res1.rectified_bgr, out2)
        vis_small = cv2.resize(vis, (1240, 1754), interpolation=cv2.INTER_AREA)

        out_path = out_dir / out_name
        imwrite_unicode(out_path, vis_small)

        written_cells = sum(1 for row in out2.answer_grid for c in row if not c.is_empty)
        empty_cells = sum(1 for row in out2.answer_grid for c in row if c.is_empty)

        print(f"  Timing Bars    : {m2.left_bars_detected}/20 Left | {m2.right_bars_detected}/20 Right ({m2.timing_bar_recall_pct}%)")
        print(f"  Answer Cells   : 400 total | Written(Orange)={written_cells} | Empty(Green)={empty_cells}")
        print(f"  Header IDs     : {m2.header_id_exact_align_pct}% Purple")
        print(f"  OMR Bubbles    : {m2.omr_bubbles_exact_align_pct}% Aligned (Mean NCC={m2.omr_mean_ncc_score:.3f})")


if __name__ == "__main__":
    main()
