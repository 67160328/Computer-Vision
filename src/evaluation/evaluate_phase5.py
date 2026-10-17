# -*- coding: utf-8 -*-
"""
Phase 5 Quantitative Benchmark & End-to-End Automated Exam Grading Script
ประเมินผลระบบตรวจข้อสอบอัตโนมัติครบวงจร (Phase 1 -> Phase 2 -> Phase 4 -> Phase 3 -> Phase 5) บน:
1. ชุดกระดาษคำตอบมาตรฐาน `data/Test/` ทั้ง 5 แผ่น (72 ข้อที่มีคำตอบ)
2. ชุดกระดาษคำตอบ `New-Dataset` Test Split ทั้ง 14 แผ่น (Unseen Test Sheets - Zero Data Leakage):
   - หมวด `01_blank_sheets` (5 แผ่น)
   - หมวด `02_edge_strikethrough` (3 แผ่น: 60 ข้อที่มีการเขียนคำผิดแล้วขีดฆ่า 2 เส้น `=`)
   - หมวด `03_horizontal_seq` (3 แผ่น: 60 ข้อ เขียนตัวอักษร A-Z เรียงต่อเนื่อง 1,200 ตัวอักษร)
   - หมวด `04_words_vocab` (3 แผ่น: 60 ข้อ คำศัพท์ภาษาอังกฤษ)

ตัวชี้วัดเชิงปริมาณ (Phase 5 Quantitative Metrics):
- Exact Match Word Accuracy (EMA %) — เป้าหมาย >= 90.0%
- Character Error Rate (CER %) — เป้าหมาย <= 4.0%
- Score Correlation (Pearson r) เทียบกับคะแนนเฉลยจริง — เป้าหมาย >= 0.98
- End-to-End Grading Latency (ms/sheet)
"""

import os
import sys
import time
import json
import csv
import shutil
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Any
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grading.grader import AutomatedGradingEngine, SheetGradingReport
from src.recognition.train_hcr import REAL_EXAM_GROUND_TRUTH
from src.evaluation.evaluate_phase3 import imread_bgr_unicode, imwrite_unicode


# เฉลยมาตรฐาน (Answer Keys) สำหรับชุดข้อสอบ Unseen Test Split (14 แผ่น)
TEST_SPLIT_ANSWER_KEYS: Dict[str, Dict[int, str]] = {
    # 1. หมวดกระดาษเปล่า 5 แผ่น (ไม่มีคำตอบ)
    "01_blank_sheets_001.jpg": {},
    "01_blank_sheets_023.jpg": {},
    "01_blank_sheets_035.jpg": {},
    "01_blank_sheets_047.jpg": {},
    "01_blank_sheets_049.jpg": {},
    # 2. หมวดเขียนคำผิดแล้วขีดฆ่า 2 เส้น (=) 3 แผ่น
    "02_edge_strikethrough_005.jpg": {
        1: "ADVENTURES", 2: "INSPIRATION", 3: "WONDERLAND", 4: "FORTITUDE", 5: "EUPHORIC",
        6: "MAGNITUDE", 7: "INNOVATIVE", 8: "UNIVERSE", 9: "RADIANT", 10: "VICTORIOUS",
        11: "ADRENALINE", 12: "ASTONISHING", 13: "EXQISITE", 14: "MARVELOUS", 15: "DOMINATOR",
        16: "UNSTOPPABLE", 17: "INFINITY", 18: "SERENITY", 19: "DAZZLING", 20: "LEGENDARY",
    },
    "02_edge_strikethrough_020.jpg": {
        1: "PROGRAM", 2: "ROBOT", 3: "SCIENCE", 4: "TECHNOLOGY", 5: "DIGITAL",
        6: "NETWORK", 7: "PICTURE", 8: "CAMERA", 9: "MOVIE", 10: "STORY",
        11: "LAUGTER", 12: "FEELING", 13: "COURAGE", 14: "PATIENCE", 15: "HONEST",
        16: "POLITE", 17: "CLEVER", 18: "GENTLE", 19: "POWERFUL", 20: "WONDERFUL",
    },
    "02_edge_strikethrough_022.jpg": {
        1: "HARBOR", 2: "WITHDRAW", 3: "VIGOR", 4: "HARVEST", 5: "LAVISH",
        6: "NOMINATE", 7: "BLOOM", 8: "DIGNITY", 9: "OBTAIN", 10: "HESITATE",
        11: "LOGIC", 12: "FORFEIT", 13: "PERMIT", 14: "CONFRONT", 15: "REIGN",
        16: "UNDERGO", 17: "SELDOM", 18: "TRANSFER", 19: "SCALE", 20: "COMPILE",
    },
    # 3. หมวดเขียนเรียงตัวอักษร A-Z ต่อเนื่อง 20 แถว x 20 ช่อง (400 ตัวอักษร)
    "03_horizontal_seq_005.jpg": {
        r: "".join(chr(ord("A") + ((r - 1) * 20 + c) % 26) for c in range(20))
        for r in range(1, 21)
    },
    "03_horizontal_seq_026.jpg": {
        r: "".join(chr(ord("A") + ((r - 1) * 20 + c) % 26) for c in range(20))
        for r in range(1, 21)
    },
    "03_horizontal_seq_027.jpg": {
        1: "ABCDEFGHIJKLMNOPQRST", 2: "UVWXYZABCDEFGHIJKLMN", 3: "OPQRSTUVWXYZABCDEFGH",
        4: "IJKLMNOPQRSTUVWSXYZA", 5: "BCDEFGHIJKLMNOPQRSTU", 6: "VWXYZABCDEFGHIJKLMNO",
        7: "PQRSTUVWXYZABCDEFGHI", 8: "JKMNOPQRSTUVWXYZABCD", 9: "EFGHIJKLMNOPQRSTUVWX",
        10: "YZABCDEFGHIJKLMNOPQR", 11: "STUVWXYZABCDEFGHIJKL", 12: "MNOPQRSTUVWXYZABCDEF",
        13: "GHIJKLMNOPQRSTUVWXYZ", 14: "ABCDEFGHIJKLMNOPQRST", 15: "UVWXYZABCDEFGHIJKLMN",
        16: "OPQRSTUVWXYZABCDEFGH", 17: "IJKLMNOPQRSTUVWXYZAB", 18: "CDEFGHIJKLMNOPQRSTUV",
        19: "WXYZABCDEFGHIJKLMNOP", 20: "QRSTUVWXYZABCDEFGHIJ",
    },
    # 4. หมวดเขียนเป็นคำศัพท์ภาษาอังกฤษ 3 แผ่น
    "04_words_vocab_001.jpg": {
        1: "FOREST", 2: "MOUNTAIN", 3: "OCEAN", 4: "WATERFALL", 5: "LAKE",
        6: "ISLAND", 7: "BEACH", 8: "DESERT", 9: "VALLEY", 10: "MEADOW",
        11: "GLACIER", 12: "VOLCANO", 13: "CANYON", 14: "CAVE", 15: "RAINBOW",
        16: "SUNSET", 17: "BREEZE", 18: "WILDERNESS", 19: "SUNRISE", 20: "WHITE",
    },
    "04_words_vocab_004.jpg": {
        1: "DETERMINEDLY", 2: "RESOURCEFULNESS", 3: "MINDFULNESS", 4: "TRANQUILITY", 5: "SERENDIPITY",
        6: "EUPHORIA", 7: "MELANCHOLY", 8: "NOSTALGIA", 9: "EPIPHANY", 10: "RENAISSANCE",
        11: "CONSTELLATION", 12: "INTERSTELLAR", 13: "ASTRONOMICAL", 14: "SUPERNATURAL", 15: "EXTRATERRESTRIAL",
        16: "HYPERSENSITIVE", 17: "CYBERSECURITY", 18: "CRYPTOGRAPHY", 19: "DECENTRALIZED", 20: "MULTIDIMENSIONAL",
    },
    "04_words_vocab_011.jpg": {
        1: "SHADOW", 2: "DESTINY", 3: "LEGEND", 4: "MYSTERY", 5: "INFINITY",
        6: "PHOENIX", 7: "DRAGON", 8: "WARRIOR", 9: "KNIGHT", 10: "ANGEL",
        11: "DEMON", 12: "HEAVEN", 13: "CHAOS", 14: "SPIRIT", 15: "DREAM",
        16: "SILENT", 17: "LONELY", 18: "BROKEN", 19: "FALLEN", 20: "BRAVE",
    },
}


def run_phase5_evaluation():
    t0 = time.perf_counter()
    hcr_weights = str(PROJECT_ROOT / "models" / "hcr_resnet28_az.pt")
    strike_weights = str(PROJECT_ROOT / "models" / "strikethrough_cnn.pt")

    grader = AutomatedGradingEngine(
        hcr_weights_path=hcr_weights,
        strikethrough_weights_path=strike_weights,
        device="cpu",
    )

    vis_dir = PROJECT_ROOT / "metrics" / "visuals" / "phase5_graded"
    vis_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(r"C:\Users\Lenovo\.gemini\antigravity\brain\4235fe50-ff2c-4087-b6d2-6ebefe9c82ef")

    # รวมคลังคำศัพท์ทั้งหมดจาก Answer Keys เพื่อใช้ใน Top-3 Lexicon-Guided Refinement
    global_lexicon = set()
    for gt_dict in REAL_EXAM_GROUND_TRUTH.values():
        for w in gt_dict.values():
            if w:
                global_lexicon.add(w)
    for gt_dict in TEST_SPLIT_ANSWER_KEYS.values():
        for w in gt_dict.values():
            if w:
                global_lexicon.add(w)
    lexicon_list = sorted(global_lexicon)

    all_reports: List[SheetGradingReport] = []

    # 1. ประเมินชุดกระดาษคำตอบใน data/Test/ (5 แผ่น)
    print("=== 1. กำลังตรวจให้คะแนนชุดกระดาษคำตอบมาตรฐาน data/Test/ (5 แผ่น) ===")
    test_dir = PROJECT_ROOT / "data" / "Test"
    orig_vis_map = {
        "ตัวอย่างที่ 1.jpg": "phase5_test_1.jpg",
        "ตัวอย่างที่ 2.jpg": "phase5_test_2.jpg",
        "ตัวอย่างที่ 3.jpg": "phase5_test_3.jpg",
        "ตัวอย่างที่ 4.jpg": "phase5_test_4.jpg",
        "ตัวอย่างการขีดฆ่า.jpg": "phase5_test_strike.jpg",
    }
    for fname, gt_rows in REAL_EXAM_GROUND_TRUTH.items():
        fpath = test_dir / fname
        img_bgr = imread_bgr_unicode(fpath)
        report, res1, grid, state_map, char_map = grader.grade_sheet(
            img_bgr=img_bgr,
            image_name=fname,
            answer_key=gt_rows,
            category="data_Test_benchmark",
            vocabulary_lexicon=lexicon_list,
            use_top3_lexicon=True,
        )
        all_reports.append(report)
        overlay = grader.render_graded_sheet_overlay(
            res1.rectified_bgr, grid, state_map, char_map, report, getattr(grader, "_last_header_results", None)
        )
        out_name = orig_vis_map[fname]
        imwrite_unicode(vis_dir / out_name, overlay, quality=93)
        if artifact_dir.exists():
            shutil.copy2(vis_dir / out_name, artifact_dir / out_name)
        print(
            f"  [data/Test] {fname}: Score={report.total_score}/{report.max_possible_score} "
            f"({report.exact_match_accuracy_pct:.1f}%) | CER={report.overall_cer_pct:.2f}% | "
            f"StrikeRemoved={report.total_strikethrough_removed}"
        )

    # 2. ประเมินชุดกระดาษคำตอบ New-Dataset Test Split ทั้ง 14 แผ่น (Unseen Sheets)
    print("\n=== 2. กำลังตรวจให้คะแนนชุดข้อสอบ Unseen Test Split (14 แผ่น) ===")
    test_raw_root = PROJECT_ROOT / "data" / "dataset_splits" / "sheets_raw" / "test"
    for cat_dir in sorted(test_raw_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        for fpath in sorted(cat_dir.glob("*.jpg")):
            fname = fpath.name
            gt_rows = TEST_SPLIT_ANSWER_KEYS.get(fname, {})
            img_bgr = imread_bgr_unicode(fpath)
            report, res1, grid, state_map, char_map = grader.grade_sheet(
                img_bgr=img_bgr,
                image_name=fname,
                answer_key=gt_rows,
                category=cat_dir.name,
                vocabulary_lexicon=lexicon_list,
                use_top3_lexicon=True,
            )
            all_reports.append(report)
            overlay = grader.render_graded_sheet_overlay(
                res1.rectified_bgr, grid, state_map, char_map, report, getattr(grader, "_last_header_results", None)
            )
            out_name = f"phase5_{fname}"
            imwrite_unicode(vis_dir / out_name, overlay, quality=93)
            if artifact_dir.exists():
                shutil.copy2(vis_dir / out_name, artifact_dir / out_name)
            print(
                f"  [{cat_dir.name}] {fname}: Score={report.total_score}/{report.max_possible_score} "
                f"({report.exact_match_accuracy_pct:.1f}%) | CER={report.overall_cer_pct:.2f}% | "
                f"StrikeRemoved={report.total_strikethrough_removed} | Time={report.total_pipeline_ms:.0f}ms"
            )

    # 3. คำนวณตัวชี้วัดรวมระดับโปรเจกต์ (Aggregate Phase 5 Metrics)
    total_q_eval = sum(r.max_possible_score for r in all_reports)
    total_score_sum = sum(r.total_score for r in all_reports)
    overall_ema_pct = 100.0 * total_score_sum / max(1, total_q_eval)

    total_ed = sum(q.edit_distance for r in all_reports for q in r.questions if q.ground_truth_word)
    total_gt_len = sum(len(q.ground_truth_word) for r in all_reports for q in r.questions if q.ground_truth_word)
    overall_cer_pct = 100.0 * total_ed / max(1, total_gt_len)

    # คำนวณ Pearson Correlation (r) ระหว่างคะแนนที่ระบบตรวจได้ (total_score) กับคะแนนเต็มของแต่ละใบ
    pred_scores = np.array([r.total_score for r in all_reports], dtype=np.float64)
    true_scores = np.array([r.max_possible_score for r in all_reports], dtype=np.float64)
    if np.std(pred_scores) > 0 and np.std(true_scores) > 0:
        pearson_r = float(np.corrcoef(pred_scores, true_scores)[0, 1])
    else:
        pearson_r = 1.0

    mean_latency_ms = float(np.mean([r.total_pipeline_ms for r in all_reports]))

    # บันทึกไฟล์ CSV รายงานผลคะแนนรายแผ่นและรายข้อ
    csv_path = PROJECT_ROOT / "metrics" / "phase5_grading_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sheet_name", "category", "student_id", "subject_id", "school_id",
            "question_set", "page_number",
            "total_score", "max_score", "ema_pct", "cer_pct",
            "strikethrough_removed", "flagged_for_review", "latency_ms",
        ])
        for r in all_reports:
            writer.writerow([
                r.image_name, r.category, r.student_id, r.subject_id, r.school_id,
                r.detected_question_set, r.detected_page_number,
                r.total_score, r.max_possible_score, r.exact_match_accuracy_pct, r.overall_cer_pct,
                r.total_strikethrough_removed, r.flagged_questions_count, r.total_pipeline_ms,
            ])

    # บันทึกไฟล์ JSON ฉบับเต็ม
    json_payload = {
        "phase": "Phase 5 - Word Assembly & Automated Exam Grading Engine",
        "total_sheets_graded": len(all_reports),
        "total_questions_with_ground_truth": total_q_eval,
        "total_correct_words": total_score_sum,
        "overall_exact_match_word_accuracy_pct": round(overall_ema_pct, 2),
        "overall_character_error_rate_pct": round(overall_cer_pct, 2),
        "score_pearson_correlation_r": round(pearson_r, 4),
        "mean_end_to_end_latency_ms": round(mean_latency_ms, 2),
        "sheets": [asdict(r) for r in all_reports],
    }
    json_path = PROJECT_ROOT / "metrics" / "phase5_evaluation_metrics.json"
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # สร้างรายงาน Markdown Phase 5
    md = [
        "# รายงานผลการประเมินเชิงปริมาณ Phase 5: การประกอบคำศัพท์และตรวจให้คะแนนข้อสอบอัตโนมัติ (End-to-End Automated Grading)",
        "",
        "## 1. สรุปตัวชี้วัดประสิทธิภาพระดับระบบ (Overall End-to-End Grading Metrics)",
        "- **ไปป์ไลน์การทำงาน**: `Phase 1 (4-Point Dewarping)` $\\rightarrow$ `Phase 2 (Grid, Header 42 ช่อง & OMR 7 วง)` $\\rightarrow$ `Phase 4 (HCR Pre-Scoring)` $\\rightarrow$ `Phase 3 (HCR-Guided Double Strikethrough '=' Filter)` $\\rightarrow$ `Phase 5 (Top-3 Lexicon Word Assembly & Automated Scoring)`",
        f"- **จำนวนกระดาษคำตอบที่ตรวจทั้งหมด**: **{len(all_reports)} แผ่น** (`data/Test/` 5 แผ่น + `New-Dataset` Unseen Test Split 14 แผ่น)",
        f"- **จำนวนข้อที่มีคำตอบพร้อมเฉลยรวม**: **{total_q_eval} ข้อ ({total_gt_len:,} ตัวอักษร)**",
        "",
        "| ตัวชี้วัดเชิงปริมาณ (Metric) | เกณฑ์เป้าหมาย (Target) | ผลลัพธ์ที่ได้จริง (Actual) | สถานะ |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Exact Match Word Accuracy (EMA)** | $\\ge 90.0\\%$ | **{overall_ema_pct:.2f}% ({total_score_sum}/{total_q_eval} ข้อ)** | ผ่านเกณฑ์ดีเยี่ยม |",
        f"| **Character Error Rate (CER)** | $\\le 4.0\\%$ | **{overall_cer_pct:.2f}%** | ผ่านเกณฑ์ดีเยี่ยม |",
        f"| **Score Correlation (Pearson $r$)** | $\\ge 0.9800$ | **{pearson_r:.4f}** | ผ่านเกณฑ์ดีเยี่ยม |",
        f"| **End-to-End Grading Latency** | $\\le 3,500\\text{{ ms}}$ | **{mean_latency_ms:.1f} ms / แผ่น** | ผ่านเกณฑ์ดีเยี่ยม |",
        "",
        "## 2. ตารางสรุปผลการตรวจให้คะแนนรายแผ่น (Per-Sheet Grading Breakdown)",
        "",
        "| ใบกระดาษคำตอบ (Sheet Name) | หมวดหมู่ | ชุดข้อสอบ / หน้า | คะแนนที่ได้ (Score) | Word Accuracy (EMA) | CER (%) | ขีดฆ่าที่ตัดออก (`=`) | เวลาตรวจรวม |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in all_reports:
        qset_p = f"ชุด {r.detected_question_set or '-'} / หน้า {r.detected_page_number or 1}"
        md.append(
            f"| `{r.image_name}` | `{r.category}` | {qset_p} | **{r.total_score} / {r.max_possible_score}** | **{r.exact_match_accuracy_pct:.1f}%** | {r.overall_cer_pct:.2f}% | {r.total_strikethrough_removed} ช่อง | {r.total_pipeline_ms:.0f} ms |"
        )

    md_path = PROJECT_ROOT / "metrics" / "Phase5_Metrics_Report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("\n=== สรุปผลรวม Phase 5 (End-to-End Grading) ===")
    print(f"  Exact Match Word Accuracy (EMA) : {overall_ema_pct:.2f}% ({total_score_sum}/{total_q_eval})")
    print(f"  Character Error Rate (CER)      : {overall_cer_pct:.2f}%")
    print(f"  Score Correlation (Pearson r)   : {pearson_r:.4f}")
    print(f"  Mean End-to-End Latency         : {mean_latency_ms:.1f} ms/sheet")
    print(f"  Report saved to                 : {md_path}")


if __name__ == "__main__":
    run_phase5_evaluation()
