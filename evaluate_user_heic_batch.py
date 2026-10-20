# -*- coding: utf-8 -*-
"""
Batch Evaluation Script for User's 16 HEIC Vocabulary Exam Sheets
Directory: C:\\Users\\Lenovo\\Downloads\\Test (คําศัพท์)-20261020T082544Z-1-001\\Test (คําศัพท์)
"""

import os
import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grading.grader import AutomatedGradingEngine
from src.evaluation.evaluate_phase3 import imread_bgr_unicode, imwrite_unicode

SET_1_ANSWER_KEY = {
    1: "ELECTROENCEPHALOGRAM",
    2: "UNCHARACTERISTICALLY",
    3: "COUNTERREVOLUTIONARY",
    4: "INCOMPREHENSIBILITY",
    5: "DISPROPORTIONATELY",
    6: "PSYCHOPHARMACOLOGY",
    7: "TRANSUBSTANTIATION",
    8: "OVERSIMPLIFICATION",
    9: "CONSTITUTIONALITY",
    10: "TELECOMMUNICATION",
    11: "ENVIRONMENTALISTS",
    12: "STRAIGHTFORWARDLY",
    13: "INDIVIDUALIZATION",
    14: "MISUNDERSTANDINGS",
    15: "DISQUALIFICATION",
    16: "EXTRATERRESTRIAL",
    17: "BIOTECHNOLOGICAL",
    18: "ACKNOWLEDGEMENTS",
    19: "JURISDICTIONAL",
    20: "CYBERSECURITY",
}

SET_2_ANSWER_KEY = {
    1: "QUIZ",
    2: "JUMP",
    3: "WOLF",
    4: "ZEBRA",
    5: "BRAVE",
    6: "SPARK",
    7: "OXYGEN",
    8: "KNIGHT",
    9: "FREEZE",
    10: "SHADOW",
    11: "QUANTUM",
    12: "DOLPHIN",
    13: "JOURNEY",
    14: "VICTORY",
    15: "TREASURE",
    16: "CHAMPION",
    17: "ADVENTURE",
    18: "LIGHTNING",
    19: "FRIENDSHIP",
    20: "WONDERLAND",
}


def main():
    test_dir = Path(r"C:\Users\Lenovo\Downloads\Test (คําศัพท์)-20261020T082544Z-1-001\Test (คําศัพท์)")
    if not test_dir.exists():
        print(f"[ERROR] Directory not found: {test_dir}")
        return

    output_vis_dir = PROJECT_ROOT / "metrics" / "visuals" / "test_vocab_heic"
    output_vis_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in test_dir.iterdir() if f.suffix.upper() in (".HEIC", ".JPG", ".PNG")])
    print(f"=======================================================")
    print(f"  Batch Grading for 16 Vocabulary Test Sheets (HEIC)")
    print(f"  Found {len(files)} files in {test_dir.name}")
    print(f"=======================================================\n")

    hcr_weights = str(PROJECT_ROOT / "models" / "hcr_resnet28_az.pt")
    strike_weights = str(PROJECT_ROOT / "models" / "strikethrough_cnn.pt")
    digit_weights = str(PROJECT_ROOT / "models" / "hcr_resnet28_digits.pt")

    grader = AutomatedGradingEngine(
        hcr_weights_path=hcr_weights,
        strikethrough_weights_path=strike_weights,
        digit_weights_path=digit_weights,
        device="cpu",
    )

    all_results = []

    for idx, fpath in enumerate(files, 1):
        t0 = time.perf_counter()
        img_bgr = imread_bgr_unicode(fpath)
        if img_bgr is None:
            print(f"[{idx:2d}/{len(files)}] {fpath.name}: FAILED to load image!")
            continue

        # Initial fast check to determine question set via dewarp + OMR
        # Or let grader.grade_sheet run with a dummy key first, then re-grade with the exact key
        # To be clean, first run dewarp & grid extraction to know the OMR question set
        dewarp_res = grader.dewarper.process_image(img_bgr, fpath.name)
        if not dewarp_res.metrics.success:
            print(f"[{idx:2d}/{len(files)}] {fpath.name}: Dewarping FAILED!")
            continue

        grid_res = grader.extractor.process_rectified_sheet(
            dewarp_res.rectified_norm_gray, fpath.name
        )
        qset = grid_res.metrics.detected_question_set
        
        # Decide answer key
        if qset == 2:
            answer_key = SET_2_ANSWER_KEY
            qset_name = "ชุดที่ 2 (คำสั้น 4-10 อักษร)"
        else:
            answer_key = SET_1_ANSWER_KEY
            qset_name = "ชุดที่ 1 (คำยาว 13-20 อักษร)"

        # Full grading pipeline
        report, res1, grid, state_map, pre_char_map = grader.grade_sheet(
            img_bgr=img_bgr,
            image_name=fpath.name,
            answer_key=answer_key,
            category=f"qset_{qset or 1}",
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Render overlay image
        vis = grader.render_graded_sheet_overlay(
            res1.rectified_bgr,
            grid,
            state_map,
            pre_char_map,
            report,
            header_results=getattr(grader, "_last_header_results", None),
        )
        vis_small = cv2.resize(vis, (1240, 1754), interpolation=cv2.INTER_AREA)
        out_vis_path = output_vis_dir / f"{fpath.stem}_graded.jpg"
        imwrite_unicode(out_vis_path, vis_small)

        # Collect summary
        sheet_summary = {
            "index": idx,
            "filename": fpath.name,
            "stem": fpath.stem,
            "qset_detected": qset,
            "qset_name": qset_name,
            "student_id": report.student_id,
            "subject_id": report.subject_id,
            "school_id": report.school_id,
            "score": report.total_score,
            "max_score": report.max_possible_score,
            "accuracy_pct": report.exact_match_accuracy_pct,
            "cer_pct": report.overall_cer_pct,
            "strikethroughs": report.total_strikethrough_removed,
            "reprojection_rmse": round(dewarp_res.metrics.reprojection_rmse_px, 2),
            "pipeline_ms": round(elapsed_ms, 1),
            "vis_path": str(out_vis_path.resolve()),
            "errors": [],
        }

        for q in report.questions:
            if not q.is_correct:
                sheet_summary["errors"].append({
                    "question_num": q.question_num,
                    "pred": q.final_assembled_word,
                    "gt": q.ground_truth_word,
                    "edit_dist": q.edit_distance,
                    "reason": q.review_reason,
                })

        all_results.append(sheet_summary)

        status_str = f"✅ {report.total_score}/{report.max_possible_score} ({report.exact_match_accuracy_pct:.1f}%)"
        if sheet_summary["errors"]:
            err_details = ", ".join([f"Q{e['question_num']}: '{e['pred']}'!='{e['gt']}'" for e in sheet_summary["errors"]])
            status_str += f" | Missed: {err_details}"
        print(f"[{idx:2d}/{len(files)}] {fpath.name} | QSet {qset} | StuID: {report.student_id or '-'} | Subj: {report.subject_id or '-'} | {status_str} | {elapsed_ms:.0f}ms")

    # Save summary json
    summary_json_path = PROJECT_ROOT / "metrics" / "test_vocab_heic_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Calculate overall aggregate metrics
    total_sheets = len(all_results)
    total_words = sum(r["max_score"] for r in all_results)
    correct_words = sum(r["score"] for r in all_results)
    overall_word_acc = (100.0 * correct_words / total_words) if total_words > 0 else 0.0
    mean_cer = np.mean([r["cer_pct"] for r in all_results]) if all_results else 0.0
    perfect_sheets = sum(1 for r in all_results if r["score"] == r["max_score"])

    # Set 1 stats
    set1_results = [r for r in all_results if r["qset_detected"] == 1]
    set1_words = sum(r["max_score"] for r in set1_results)
    set1_correct = sum(r["score"] for r in set1_results)
    set1_acc = (100.0 * set1_correct / set1_words) if set1_words > 0 else 0.0

    # Set 2 stats
    set2_results = [r for r in all_results if r["qset_detected"] == 2]
    set2_words = sum(r["max_score"] for r in set2_results)
    set2_correct = sum(r["score"] for r in set2_results)
    set2_acc = (100.0 * set2_correct / set2_words) if set2_words > 0 else 0.0

    print(f"\n=======================================================")
    print(f"  OVERALL BATCH TEST SUMMARY")
    print(f"=======================================================")
    print(f"  • Total Sheets Tested     : {total_sheets} sheets")
    print(f"  • Perfect Sheets (20/20)   : {perfect_sheets} / {total_sheets} ({100.0*perfect_sheets/total_sheets:.1f}%)")
    print(f"  • Total Words Graded      : {total_words} words")
    print(f"  • Total Words Correct     : {correct_words} / {total_words}")
    print(f"  • Overall Word Accuracy   : {overall_word_acc:.2f}%")
    print(f"  • Mean Character Error    : {mean_cer:.2f}%")
    print(f"  ---------------------------------------------------")
    print(f"  • Set 1 (Long Words)      : {set1_correct}/{set1_words} ({set1_acc:.2f}%) across {len(set1_results)} sheets")
    print(f"  • Set 2 (Short Words)     : {set2_correct}/{set2_words} ({set2_acc:.2f}%) across {len(set2_results)} sheets")
    print(f"  • Results JSON saved to   : {summary_json_path}")
    print(f"  • Visual overlays saved in: {output_vis_dir}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
