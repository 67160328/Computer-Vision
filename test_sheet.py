# -*- coding: utf-8 -*-
"""
Quick Exam Sheet Test Tool (เครื่องมือทดสอบตรวจกระดาษคำตอบแผ่นเดี่ยว)
ระบบตรวจกระดาษคำตอบอัตโนมัติ Rayong MARCO (Template 2)

วิธีใช้งาน:
    py -3.12 test_sheet.py "data/Test/ตัวอย่างที่ 3.jpg"
    py -3.12 test_sheet.py "path/to/my_new_photo.jpg"
"""

import sys
import os
import argparse
from pathlib import Path
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grading.grader import AutomatedGradingEngine
from src.evaluation.evaluate_phase3 import imread_bgr_unicode, imwrite_unicode
from src.recognition.train_hcr import REAL_EXAM_GROUND_TRUTH
from src.evaluation.evaluate_phase5 import TEST_SPLIT_ANSWER_KEYS


def get_default_answer_key(image_name: str) -> dict:
    """ดึงเฉลยอัตโนมัติหากเป็นภาพที่อยู่ในชุดทดสอบมาตรฐาน"""
    if image_name in REAL_EXAM_GROUND_TRUTH:
        return REAL_EXAM_GROUND_TRUTH[image_name]
    if image_name in TEST_SPLIT_ANSWER_KEYS:
        return TEST_SPLIT_ANSWER_KEYS[image_name]
    
    # หากเป็นภาพทดสอบคำตอบยาว (ทดสอบคำตอบ.jpg)
    if "ทดสอบ" in image_name:
        return {
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
    
    # หากเป็นภาพใหม่ที่ไม่มีเฉลย ให้ใช้ชุดเฉลยคำศัพท์มาตรฐาน
    return {
        1: "SCHEME", 2: "PASSION", 3: "BLEND", 4: "MAINTAIN", 5: "LAYOUT",
        6: "CLARIFY", 7: "SHARP", 8: "MODERATE", 9: "THRIVE", 10: "SINCERE",
        11: "SPLIT", 12: "ADEQUATE", 13: "REWARD", 14: "EMBRACE", 15: "BROAD",
        16: "TREASURE", 17: "FREEZE", 18: "JUSTICE", 19: "GRACE", 20: "VALIDATE",
    }



def main():
    parser = argparse.ArgumentParser(description="ทดสอบตรวจกระดาษคำตอบ 1 แผ่นแบบครบวงจร")
    parser.add_argument(
        "image_path",
        nargs="?",
        default="data/Test/ตัวอย่างที่ 3.jpg",
        help="พาธไฟล์ภาพกระดาษคำตอบ (เช่น data/Test/ตัวอย่างที่ 3.jpg)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="metrics/visuals/single_test_result.jpg",
        help="พาธสำหรับบันทึกภาพผลการตรวจพร้อม Overlay",
    )
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.is_absolute():
        img_path = PROJECT_ROOT / img_path

    if not img_path.exists():
        print(f"[ERROR] ไม่พบไฟล์ภาพ: {img_path}")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"  Rayong MARCO (Template 2) Automated Grading Test")
    print(f"=======================================================")
    print(f"-> กำลังอ่านภาพ: {img_path.name}")

    img_bgr = imread_bgr_unicode(img_path)
    if img_bgr is None:
        print(f"[ERROR] ไม่สามารถอ่านไฟล์ภาพได้ (อาจไม่ใช่ไฟล์รูปภาพที่ถูกต้อง)")
        sys.exit(1)

    # โหลดเอนจิน
    hcr_weights = str(PROJECT_ROOT / "models" / "hcr_resnet28_az.pt")
    strike_weights = str(PROJECT_ROOT / "models" / "strikethrough_cnn.pt")
    digit_weights = str(PROJECT_ROOT / "models" / "hcr_resnet28_digits.pt")

    print("-> กำลังโหลดโมเดล CNN (HCR A-Z, Strikethrough, Digit 0-9)...")
    grader = AutomatedGradingEngine(
        hcr_weights_path=hcr_weights,
        strikethrough_weights_path=strike_weights,
        digit_weights_path=digit_weights,
        device="cpu",
    )

    answer_key = get_default_answer_key(img_path.name)

    print("-> กำลังประมวลผลไปป์ไลน์ Phase 1 ถึง 5...")
    report, res1, grid, state_map, pre_char_map = grader.grade_sheet(
        img_bgr=img_bgr,
        image_name=img_path.name,
        answer_key=answer_key,
        category="single_test",
    )

    # แสดงผลส่วนหัว (Header ID)
    print("\n" + "=" * 60)
    print(" 📌 ผลการสแกนรหัสส่วนหัว (Header ID Strip - 42 ช่องสีม่วง)")
    print("=" * 60)
    print(f"  • รหัสประจำตัว (Student ID) : {report.student_id or '(เว้นว่าง)'}")
    print(f"  • รหัสวิชา (Subject ID)     : {report.subject_id or '(เว้นว่าง)'}")
    print(f"  • รหัสสถานศึกษา (School ID) : {report.school_id or '(เว้นว่าง)'}")
    print(f"  • ชุดข้อสอบที่ตรวจพบ         : {report.detected_question_set or '-'}")
    print(f"  • หน้ากระดาษคำตอบ           : {report.detected_page_number or '-'}")

    # แสดงผลตารางคำตอบ 20 ข้อ
    print("\n" + "=" * 60)
    print(" 📝 ผลการตรวจคำตอบรายข้อ (ข้อ 1 - 20)")
    print("=" * 60)
    print(f"{'ข้อ':^4} | {'คำที่อ่านได้':<18} | {'เฉลย':<18} | {'สถานะ':^8} | {'ขีดฆ่า (=)':^10}")
    print("-" * 68)

    for q in report.questions:
        if q.ground_truth_word:
            status = "✅ PASS" if q.is_correct else "❌ FAIL"
            if q.needs_human_review:
                status += " ⚠️"
        else:
            status = "⚪ NO KEY"

        strike_info = f"{q.strikethrough_count} ช่อง" if q.strikethrough_count > 0 else "-"
        print(f"{q.question_num:^4} | {q.final_assembled_word:<18} | {q.ground_truth_word:<18} | {status:^8} | {strike_info:^10}")

    # สรุปคะแนนรวมและเวลาประมวลผล
    print("-" * 68)
    print(f" 🏆 คะแนนที่ได้: {report.total_score} / {report.max_possible_score} คะแนน")
    print(f" 🎯 ความแม่นยำระดับคำ (Exact Match Word Accuracy): {report.exact_match_accuracy_pct:.2f}%")
    print(f" ⏱️ เวลาประมวลผลรวมทั้งแผ่น: {report.total_pipeline_ms:.1f} ms (~{report.total_pipeline_ms / 1000.0:.2f} วินาที)")

    # บันทึกภาพผลการตรวจ
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis = grader.render_graded_sheet_overlay(
        res1.rectified_bgr,
        grid,
        state_map,
        pre_char_map,
        report,
        header_results=getattr(grader, "_last_header_results", None),
    )
    vis_small = cv2.resize(vis, (1240, 1754), interpolation=cv2.INTER_AREA)
    imwrite_unicode(out_path, vis_small)
    print(f"\n 🖼️ บันทึกภาพรายงานผลการตรวจพร้อม Overlay เรียบร้อยแล้วที่:\n    file:///{out_path.resolve().as_posix()}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
