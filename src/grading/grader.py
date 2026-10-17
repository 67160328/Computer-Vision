# -*- coding: utf-8 -*-
"""
Phase 5: Word Assembly & Automated Grading Engine
สำหรับระบบตรวจข้อสอบอัตโนมัติ Rayong MARCO (Template 2)

คุณลักษณะหลัก:
1. End-to-End Pipeline Integration (Phase 1 -> Phase 2 -> Phase 4 -> Phase 3 -> Phase 5):
   - Phase 1: หมุนภาพตั้งตรงอัตโนมัติและดัดระนาบ 4-Point Projective Homography (2480 x 3508 px)
   - Phase 2: สกัดตาราง 400 ช่อง, ช่องรหัสส่วนหัว 42 ช่อง (กรอบสีม่วง), และวงกลม OMR 7 วง (ชุดข้อสอบ 1-3, หน้า 1-4)
   - Phase 4 (HCR Pre-Scoring): อ่านตัวอักษรภาษาอังกฤษ A-Z พร้อมคำนวณ Top-3 Probability, Margin, และ Shannon Entropy
   - Phase 3 (HCR-Guided Strikethrough Filter): ตรวจจับและตัดช่องที่มีรอยขีดฆ่า 2 เส้น (=) ทิ้งอัตโนมัติ
   - Phase 5 (Word Assembly, Lexicon Correction & Automated Grading):
     * ประกอบตัวอักษรที่ผ่านการคัดกรองในแต่ละแถวเข้าเป็นคำศัพท์ (Raw Assembled Word)
     * รองรับ Top-3 Probability-Weighted Lexicon Matching (ช่วยแก้ไขกรณีตัวอักษรกำกวม เช่น E/F, U/V, I/T หากอยู่ใน Top-3 ของโมเดล)
     * เปรียบเทียบกับชุดเฉลย (Answer Key) คำนวณคะแนน (Binary Score 0/1), Levenshtein Edit Distance, และ Character Error Rate (CER)
     * แจ้งเตือน Human-in-the-loop Review Flag สำหรับข้อที่ความมั่นใจต่ำหรือมีรอยขีดฆ่าซับซ้อน
     * สร้างภาพ Audit Overlay พร้อมป้ายคะแนนรายข้อและสรุปคะแนนรวมบนหัวกระดาษ
"""

import os
import time
import json
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np

from src.preprocessing.dewarp import FiducialDewarper, DewarpOutput
from src.detection.grid_extractor import Template2GridExtractor, GridExtractionOutput
from src.detection.strikethrough_detector import StrikethroughDetectorEngine, CellStatePrediction
from src.recognition.char_classifier import HCRModelEngine, CharPrediction, ALPHABET
from src.recognition.digit_classifier import DigitRecognizerEngine, HeaderFieldResult, DigitCellPrediction


def levenshtein_distance(s1: str, s2: str) -> int:
    """คำนวณ Levenshtein Edit Distance ระหว่างข้อความ 2 สาย"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


@dataclass
class QuestionGradeResult:
    """ผลลัพธ์การตรวจให้คะแนนรายข้อ (ข้อ 1 - 20)"""
    question_num: int
    raw_assembled_word: str              # คำที่ประกอบได้โดยตรงจาก Top-1 HCR หลังตัดขีดฆ่า '='
    final_assembled_word: str            # คำสุทธิหลังผ่าน Top-3 Lexicon-Guided Refinement (ถ้าเปิดใช้งาน)
    ground_truth_word: str               # คำเฉลย (Answer Key)
    is_correct: bool                     # True หากตรงกับเฉลย (ได้ 1 คะแนน)
    score: int                           # 1 หรือ 0
    edit_distance: int                   # Levenshtein distance เทียบกับเฉลย
    cer: float                           # Character Error Rate ของข้อนี้
    valid_char_count: int                # จำนวนช่องตัวอักษรปกติที่นำมาประกอบเป็นคำ
    strikethrough_count: int             # จำนวนช่องที่ถูกขีดฆ่า 2 เส้น ('=') และถูกตัดออก
    empty_count: int                     # จำนวนช่องว่างในแถวนี้
    mean_confidence: float               # ความมั่นใจเฉลี่ยของตัวอักษรในข้อนี้ (0.0 - 1.0)
    min_confidence: float                # ความมั่นใจต่ำสุดของตัวอักษรในข้อนี้ (0.0 - 1.0)
    needs_human_review: bool             # True หากมีตัวอักษรความมั่นใจต่ำ (< 0.55) หรือใกล้เคียงเฉลยมาก (edit_dist == 1)
    review_reason: str                   # เหตุผลที่แนะนำให้ครูตรวจทานซ้ำ


@dataclass
class SheetGradingReport:
    """รายงานผลการตรวจกระดาษคำตอบ 1 แผ่นแบบครบวงจร (Phase 1 - 5 + Header Digits 0-9)"""
    image_name: str
    category: str
    student_id: str                      # รหัสประจำตัวนักเรียนที่สแกนได้จากกรอบสีม่วง 14 ช่องบน
    subject_id: str                      # รหัสวิชาที่สแกนได้จากกรอบสีม่วง 14 ช่องกลาง
    school_id: str                       # รหัสสถานศึกษาที่สแกนได้จากกรอบสีม่วง 14 ช่องล่าง
    header_digits_filled: int            # จำนวนช่องกรอบสีม่วงที่มีการเขียนตัวเลข (จาก 42 ช่อง)
    header_digits_mean_conf: float       # ความมั่นใจเฉลี่ยของการอ่านตัวเลขในกรอบสีม่วง
    detected_question_set: Optional[int]
    detected_page_number: Optional[int]
    header_aligned_cells: int            # จำนวนช่องรหัสสีม่วงที่ตรงพิกัด (เต็ม 42 ช่อง)
    omr_aligned_bubbles: int             # จำนวนวงกลม OMR ที่ตรงพิกัด (เต็ม 7 วง)
    total_questions_evaluated: int
    total_score: int                     # คะแนนที่ได้ (จากจำนวนข้อที่มีเฉลย)
    max_possible_score: int              # คะแนนเต็ม
    exact_match_accuracy_pct: float      # Exact Match Word Accuracy (EMA %)
    raw_exact_match_accuracy_pct: float  # EMA % ก่อนใช้ Top-3 Lexicon Refinement
    overall_cer_pct: float               # Character Error Rate รวมทั้งแผ่น (%)
    total_valid_chars: int
    total_strikethrough_removed: int
    total_empty_cells: int
    flagged_questions_count: int         # จำนวนข้อที่แจ้งเตือน Human-in-the-loop
    phase1_dewarp_ms: float
    phase2_grid_ms: float
    phase3_strikethrough_ms: float
    phase4_hcr_ms: float
    phase4b_digit_ms: float
    phase5_grading_ms: float
    total_pipeline_ms: float
    questions: List[QuestionGradeResult] = field(default_factory=list)


class AutomatedGradingEngine:
    """
    เอนจินหลักสำหรับประกอบคำศัพท์ สแกนรหัสตัวเลข 42 ช่อง ตรวจคำตอบเทียบกับชุดเฉลย และสร้างภาพรายงานผล (Phase 5)
    """
    def __init__(
        self,
        hcr_weights_path: str,
        strikethrough_weights_path: str,
        digit_weights_path: Optional[str] = None,
        device: str = "cpu",
        review_conf_threshold: float = 0.55,
    ):
        self.dewarper = FiducialDewarper()
        self.extractor = Template2GridExtractor()
        self.hcr_engine = HCRModelEngine(weights_path=hcr_weights_path, device=device)
        self.strike_engine = StrikethroughDetectorEngine(
            weights_path=strikethrough_weights_path, device=device
        )
        if digit_weights_path is None:
            default_digit_pt = Path(hcr_weights_path).parent / "hcr_resnet28_digits.pt"
            digit_weights_path = str(default_digit_pt) if default_digit_pt.exists() else None
        self.digit_engine = DigitRecognizerEngine(weights_path=digit_weights_path, device=device)
        self.review_conf_threshold = review_conf_threshold

    @staticmethod
    def _refine_with_top3_lexicon(
        row_char_preds: List[CharPrediction],
        candidate_lexicon: List[str],
    ) -> str:
        """
        Top-3 Probability-Weighted Lexicon Matching:
        เมื่อผู้สอบเขียนตัวอักษรที่มีรูปร่างคล้ายกันมากในภาษาอังกฤษ (เช่น E/F, U/V, I/T, D/O, M/N, S/Z)
        หากคำที่อ่านได้จาก Top-1 มีความคลาดเคลื่อนเพียง 1-2 ตัวอักษรจากคำศัพท์ในคลังคำศัพท์ (Vocabulary Lexicon)
        และตัวอักษรที่ถูกต้องปรากฏอยู่ใน Top-3 predictions ของโมเดล CNN (หรือมี probability ใกล้เคียงกัน)
        จะเลือกคำศัพท์ที่มีความน่าจะเป็นรวม (Log-Likelihood) สูงสุด
        """
        raw_word = "".join(cp.predicted_char for cp in row_char_preds)
        if not row_char_preds or not candidate_lexicon:
            return raw_word

        n_len = len(raw_word)
        if raw_word in candidate_lexicon:
            return raw_word

        # คัดเฉพาะคำในคลังคำศัพท์ที่มีความยาวเท่ากับจำนวนช่องที่นักเรียนเขียนในข้อนั้น
        same_len_words = [w for w in candidate_lexicon if len(w) == n_len]
        if not same_len_words:
            return raw_word

        # คู่ตัวอักษรลายมือเขียนที่มักคล้ายกันตามธรรมชาติ (Handwriting Confusable Pairs)
        confusable_sets = [
            set("EF"), set("UV"), set("ITJ"), set("DOQ"), set("MNW"),
            set("SZ"), set("BPR"), set("CGQ"), set("AHK"), set("CL"),
            set("NO"), set("MY"),
        ]

        def are_confusable(ch1: str, ch2: str) -> bool:
            if ch1 == ch2:
                return True
            return any(ch1 in s and ch2 in s for s in confusable_sets)

        best_word = raw_word
        best_score = -1e9

        for cand in same_len_words:
            # อนุญาตให้แก้ได้ไม่เกิน 2 ตำแหน่งจาก raw_word
            diffs = sum(1 for a, b in zip(raw_word, cand) if a != b)
            if diffs > 2:
                continue

            valid_cand = True
            log_prob_sum = 0.0
            for pos, target_ch in enumerate(cand):
                cp = row_char_preds[pos]
                top3_dict = dict(cp.top3_chars)
                if target_ch == cp.predicted_char:
                    log_prob_sum += float(np.log(max(cp.confidence, 1e-4)))
                elif target_ch in top3_dict and top3_dict[target_ch] >= 0.01:
                    log_prob_sum += float(np.log(top3_dict[target_ch]))
                elif are_confusable(cp.predicted_char, target_ch):
                    log_prob_sum += float(np.log(0.02))
                else:
                    valid_cand = False
                    break

            if valid_cand and log_prob_sum > best_score:
                best_score = log_prob_sum
                best_word = cand

        return best_word

    def grade_sheet(
        self,
        img_bgr: np.ndarray,
        image_name: str,
        answer_key: Dict[int, str],
        category: str = "general",
        vocabulary_lexicon: Optional[List[str]] = None,
        use_top3_lexicon: bool = True,
    ) -> Tuple[SheetGradingReport, DewarpOutput, GridExtractionOutput, Dict[Tuple[int, int], CellStatePrediction], Dict[Tuple[int, int], CharPrediction]]:
        """
        รันไปป์ไลน์ตรวจกระดาษคำตอบแบบครบวงจรตั้งแต่ Phase 1 ถึง Phase 5
        """
        t_total = time.perf_counter()

        # Phase 1: Auto-Orient & 4-Point Homography Dewarping
        t_p1 = time.perf_counter()
        res1 = self.dewarper.process_image(img_bgr, image_name=image_name, return_normalized=True)
        p1_ms = (time.perf_counter() - t_p1) * 1000.0

        # Phase 2: Grid, Header 42 cells & OMR 7 bubbles Extraction
        t_p2 = time.perf_counter()
        grid = self.extractor.process_rectified_sheet(res1.rectified_norm_gray, image_name=image_name)
        p2_ms = (time.perf_counter() - t_p2) * 1000.0

        # Phase 4B (Header Digit Recognition 0-9): สแกนตัวเลขในกรอบสีม่วง 42 ช่อง (Student/Subject/School ID)
        t_p4b = time.perf_counter()
        header_field_results = self.digit_engine.predict_header_fields(grid.header_ids)
        p4b_ms = (time.perf_counter() - t_p4b) * 1000.0

        student_id_str = header_field_results["student_id"].assembled_id
        subject_id_str = header_field_results["subject_id"].assembled_id
        school_id_str = header_field_results["school_id"].assembled_id
        header_filled_total = sum(r.filled_count for r in header_field_results.values())
        header_confs = [
            cp.confidence
            for r in header_field_results.values()
            for cp in r.cell_predictions
            if not cp.is_empty
        ]
        header_mean_conf = float(np.mean(header_confs)) if header_confs else 1.0

        # Phase 4 (Pre-Scoring HCR): อ่านตัวอักษร A-Z บนช่องที่มีหมึกทั้งหมดก่อน
        t_p4 = time.perf_counter()
        occupied_batch = [
            (r_idx, c_idx, grid.answer_grid[r_idx - 1][c_idx - 1].clean_patch)
            for r_idx in range(1, 21)
            for c_idx in range(1, 21)
            if not grid.answer_grid[r_idx - 1][c_idx - 1].is_empty
        ]
        pre_char_preds = self.hcr_engine.predict_cells(occupied_batch) if occupied_batch else []
        pre_char_map = {(cp.row_idx, cp.col_idx): cp for cp in pre_char_preds}
        p4_ms = (time.perf_counter() - t_p4) * 1000.0

        # Phase 3 (HCR-Guided Strikethrough Filter): คัดแยก empty, normal_char, strikethrough
        t_p3 = time.perf_counter()
        all_cells_batch = [
            (r_idx, c_idx, grid.answer_grid[r_idx - 1][c_idx - 1].clean_patch)
            for r_idx in range(1, 21)
            for c_idx in range(1, 21)
        ]
        state_preds = self.strike_engine.predict_cells(all_cells_batch, hcr_predictions=pre_char_map)
        p3_ms = (time.perf_counter() - t_p3) * 1000.0

        state_map = {(sp.row_idx, sp.col_idx): sp for sp in state_preds}
        valid_char_map = {
            (sp.row_idx, sp.col_idx): pre_char_map[(sp.row_idx, sp.col_idx)]
            for sp in state_preds
            if sp.is_valid_char and (sp.row_idx, sp.col_idx) in pre_char_map
        }

        # Phase 5: Word Assembly & Automated Grading
        t_p5 = time.perf_counter()
        lexicon_list = list(vocabulary_lexicon) if vocabulary_lexicon else [w for w in answer_key.values() if w]

        question_results: List[QuestionGradeResult] = []
        total_score = 0
        raw_exact_matches = 0
        max_possible_score = 0
        total_edit_dist = 0
        total_gt_chars = 0
        flagged_count = 0

        for r_idx in range(1, 21):
            row_valid_preds: List[CharPrediction] = []
            row_strikes = 0
            row_empties = 0

            for c_idx in range(1, 21):
                sp = state_map[(r_idx, c_idx)]
                if sp.is_empty:
                    row_empties += 1
                elif sp.is_strikethrough:
                    row_strikes += 1
                else:
                    cp = valid_char_map.get((r_idx, c_idx))
                    if cp is not None:
                        row_valid_preds.append(cp)

            raw_word = "".join(cp.predicted_char for cp in row_valid_preds).strip()
            if use_top3_lexicon and len(row_valid_preds) >= 3:
                final_word = self._refine_with_top3_lexicon(row_valid_preds, lexicon_list)
            else:
                final_word = raw_word

            gt_word = answer_key.get(r_idx, "").strip()
            has_gt = len(gt_word) > 0

            if row_valid_preds:
                confs = [cp.confidence for cp in row_valid_preds]
                mean_conf = float(np.mean(confs))
                min_conf = float(np.min(confs))
            else:
                mean_conf = 1.0
                min_conf = 1.0

            if has_gt:
                max_possible_score += 1
                is_correct = (final_word == gt_word)
                if raw_word == gt_word:
                    raw_exact_matches += 1
                score = 1 if is_correct else 0
                total_score += score
                ed = levenshtein_distance(final_word, gt_word)
                cer = float(ed) / float(max(1, len(gt_word)))
                total_edit_dist += ed
                total_gt_chars += len(gt_word)
            else:
                # กรณีข้อนี้ไม่มีในเฉลย (เช่น กระดาษเปล่า หรือข้อที่เว้นว่าง)
                is_correct = (len(final_word) == 0)
                score = 0
                ed = len(final_word)
                cer = 0.0 if len(final_word) == 0 else 1.0

            # ตรวจสอบเงื่อนไข Human-in-the-Loop Review Flag
            reasons = []
            if row_valid_preds and min_conf < self.review_conf_threshold:
                reasons.append(f"Low HCR conf ({min_conf:.2f})")
            if has_gt and not is_correct and ed == 1:
                reasons.append("Near match (EditDist=1)")
            if row_strikes >= 4:
                reasons.append(f"Heavy strikethrough ({row_strikes} cells)")

            needs_review = len(reasons) > 0
            if needs_review:
                flagged_count += 1

            question_results.append(
                QuestionGradeResult(
                    question_num=r_idx,
                    raw_assembled_word=raw_word,
                    final_assembled_word=final_word,
                    ground_truth_word=gt_word,
                    is_correct=is_correct,
                    score=score,
                    edit_distance=ed,
                    cer=round(cer, 4),
                    valid_char_count=len(row_valid_preds),
                    strikethrough_count=row_strikes,
                    empty_count=row_empties,
                    mean_confidence=round(mean_conf, 4),
                    min_confidence=round(min_conf, 4),
                    needs_human_review=needs_review,
                    review_reason=", ".join(reasons) if reasons else "OK",
                )
            )

        p5_ms = (time.perf_counter() - t_p5) * 1000.0
        total_ms = (time.perf_counter() - t_total) * 1000.0

        ema_pct = (100.0 * total_score / max_possible_score) if max_possible_score > 0 else 100.0
        raw_ema_pct = (100.0 * raw_exact_matches / max_possible_score) if max_possible_score > 0 else 100.0
        overall_cer_pct = (100.0 * total_edit_dist / total_gt_chars) if total_gt_chars > 0 else 0.0

        header_aligned = sum(
            1
            for clist in grid.header_cells_info.values()
            for hc in clist
            if hc.is_aligned
        )
        omr_aligned = sum(
            1
            for blist in grid.omr_bubbles_info.values()
            for b in blist
            if b.is_aligned
        )

        report = SheetGradingReport(
            image_name=image_name,
            category=category,
            student_id=student_id_str,
            subject_id=subject_id_str,
            school_id=school_id_str,
            header_digits_filled=header_filled_total,
            header_digits_mean_conf=round(header_mean_conf, 4),
            detected_question_set=grid.metrics.detected_question_set,
            detected_page_number=grid.metrics.detected_page_number,
            header_aligned_cells=header_aligned,
            omr_aligned_bubbles=omr_aligned,
            total_questions_evaluated=max_possible_score,
            total_score=total_score,
            max_possible_score=max_possible_score,
            exact_match_accuracy_pct=round(ema_pct, 2),
            raw_exact_match_accuracy_pct=round(raw_ema_pct, 2),
            overall_cer_pct=round(overall_cer_pct, 2),
            total_valid_chars=sum(q.valid_char_count for q in question_results),
            total_strikethrough_removed=sum(q.strikethrough_count for q in question_results),
            total_empty_cells=sum(q.empty_count for q in question_results),
            flagged_questions_count=flagged_count,
            phase1_dewarp_ms=round(p1_ms, 2),
            phase2_grid_ms=round(p2_ms, 2),
            phase3_strikethrough_ms=round(p3_ms, 2),
            phase4_hcr_ms=round(p4_ms, 2),
            phase4b_digit_ms=round(p4b_ms, 2),
            phase5_grading_ms=round(p5_ms, 2),
            total_pipeline_ms=round(total_ms, 2),
            questions=question_results,
        )

        self._last_header_results = header_field_results
        return report, res1, grid, state_map, valid_char_map

    @staticmethod
    def render_graded_sheet_overlay(
        rectified_bgr: np.ndarray,
        grid: GridExtractionOutput,
        state_map: Dict[Tuple[int, int], CellStatePrediction],
        char_map: Dict[Tuple[int, int], CharPrediction],
        report: SheetGradingReport,
        header_results: Optional[Dict[str, HeaderFieldResult]] = None,
    ) -> np.ndarray:
        """
        สร้างภาพใบตรวจข้อสอบสมบูรณ์แบบ (Phase 5 Audit Overlay + Header Digit Recognition 0-9):
        - แถบ Banner ด้านบนแสดงคะแนนรวม (Score X/Y), รหัสประจำตัว/รหัสวิชา/รหัสสถานศึกษา, ชุดข้อสอบ, หน้ากระดาษ
        - กรอบสีม่วงบนช่องรหัส 42 ช่อง พร้อมแสดงตัวเลข 0-9 ที่สแกนได้เหนือแต่ละช่องและป้ายสรุปรหัสด้านขวา
        - วงกลม OMR สีเขียว/ฟ้า
        - ช่องตัวอักษรปกติสีเขียว + ตัวอักษรที่อ่านได้สีน้ำเงิน
        - ช่องขีดฆ่า 2 เส้น (`=`) สีแดงสดพร้อมกากบาททับ
        - ป้ายผลการตรวจด้านขวาของแต่ละข้อ (`[1/1] WORD` สีเขียว หรือ `[0/1] PRED != KEY` สีส้ม/แดง)
        """
        vis = rectified_bgr.copy()

        # 1. วาด Header IDs (42 ช่องสีม่วง) พร้อมแสดงตัวเลข 0-9 ที่สแกนได้
        for field_name, cell_list in grid.header_cells_info.items():
            f_res = header_results.get(field_name) if header_results else None
            for idx, hcell in enumerate(cell_list):
                x1, y1, x2, y2 = hcell.bbox
                color = (180, 0, 180) if hcell.is_aligned else (0, 165, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                if f_res and idx < len(f_res.cell_predictions):
                    dp = f_res.cell_predictions[idx]
                    if not dp.is_empty and dp.predicted_digit != "":
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (140, 0, 210), 3)
                        cv2.putText(
                            vis,
                            dp.predicted_digit,
                            (x1 + 5, y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.72,
                            (140, 0, 180),
                            2,
                            cv2.LINE_AA,
                        )

            # วาดป้ายสรุปรหัสที่อ่านได้ทางด้านขวาของแถวกรอบสีม่วง (ถ้ามีตัวเลขเขียน)
            if f_res and f_res.filled_count > 0 and len(cell_list) > 0:
                last_bx2 = cell_list[-1].bbox[2]
                by1, by2 = cell_list[0].bbox[1], cell_list[0].bbox[3]
                cv2.rectangle(
                    vis,
                    (last_bx2 + 8, by1 + 6),
                    (min(vis.shape[1] - 4, last_bx2 + 255), by2 - 6),
                    (248, 232, 255),
                    -1,
                )
                cv2.rectangle(
                    vis,
                    (last_bx2 + 8, by1 + 6),
                    (min(vis.shape[1] - 4, last_bx2 + 255), by2 - 6),
                    (160, 0, 160),
                    2,
                )
                cv2.putText(
                    vis,
                    f"{f_res.assembled_id} ({f_res.mean_confidence*100:.0f}%)",
                    (last_bx2 + 14, by1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (120, 0, 140),
                    2,
                    cv2.LINE_AA,
                )

        for _, b_list in grid.omr_bubbles_info.items():
            for b in b_list:
                cx, cy = int(round(b.cx)), int(round(b.cy))
                color = (0, 180, 0) if b.is_filled else (220, 180, 0)
                cv2.circle(vis, (cx, cy), 22, color, 2)

        # 2. วาด Banner สรุปผลคะแนนและรหัสประจำตัวด้านบนกระดาษ
        banner_x1, banner_y1, banner_x2, banner_y2 = 100, 30, 2120, 135
        cv2.rectangle(vis, (banner_x1, banner_y1), (banner_x2, banner_y2), (242, 248, 255), -1)
        cv2.rectangle(vis, (banner_x1, banner_y1), (banner_x2, banner_y2), (160, 90, 20), 3)

        qset_str = str(report.detected_question_set) if report.detected_question_set else "-"
        page_str = str(report.detected_page_number) if report.detected_page_number else "1"
        score_txt = (
            f"SCORE: {report.total_score}/{report.max_possible_score} "
            f"({report.exact_match_accuracy_pct:.1f}%) | "
            f"CER: {report.overall_cer_pct:.2f}% | "
            f"Strike Removed: {report.total_strikethrough_removed} | "
            f"Set: {qset_str} Page: {page_str}"
        )
        cv2.putText(
            vis,
            score_txt,
            (banner_x1 + 22, banner_y1 + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.86,
            (130, 50, 0),
            3,
            cv2.LINE_AA,
        )

        stu_disp = report.student_id if report.student_id else "-"
        sub_disp = report.subject_id if report.subject_id else "-"
        sch_disp = report.school_id if report.school_id else "-"
        header_txt = f"Student ID: {stu_disp} | Subject ID: {sub_disp} | School ID: {sch_disp}"
        cv2.putText(
            vis,
            header_txt,
            (banner_x1 + 22, banner_y1 + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.76,
            (130, 0, 140),
            2,
            cv2.LINE_AA,
        )

        # 3. วาดตารางคำตอบ 20 ข้อ x 20 ช่อง และป้ายคะแนนรายข้อด้านขวา
        for q_res in report.questions:
            r_idx = q_res.question_num
            row_cells = grid.answer_grid[r_idx - 1]

            for c_idx in range(1, 21):
                cell = row_cells[c_idx - 1]
                x1, y1, x2, y2 = cell.bbox
                st = state_map.get((r_idx, c_idx))

                if st is None or st.is_empty:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (175, 225, 175), 1)
                    continue

                if st.is_strikethrough:
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
                    cp = char_map.get((r_idx, c_idx))
                    ch_str = cp.predicted_char if cp is not None else "?"
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

            # วาดป้ายสรุปคำตอบและคะแนนของข้อนี้ทางด้านขวาของแถว
            if q_res.ground_truth_word or q_res.final_assembled_word or q_res.strikethrough_count > 0:
                last_x2 = row_cells[-1].bbox[2]
                ry1 = row_cells[0].bbox[1]
                ry2 = row_cells[0].bbox[3]

                strike_tag = f" (-{q_res.strikethrough_count}=)" if q_res.strikethrough_count > 0 else ""
                if q_res.ground_truth_word:
                    if q_res.is_correct:
                        badge_bg = (225, 252, 225)
                        badge_fg = (0, 125, 0)
                        badge_txt = f"[{q_res.score}/1] {q_res.final_assembled_word}{strike_tag}"
                    else:
                        badge_bg = (225, 235, 255)
                        badge_fg = (0, 0, 210)
                        badge_txt = f"[0/1] {q_res.final_assembled_word} (Key:{q_res.ground_truth_word}){strike_tag}"
                else:
                    badge_bg = (240, 245, 240)
                    badge_fg = (90, 90, 90)
                    badge_txt = f"R{r_idx:02d}: {q_res.final_assembled_word}{strike_tag}"

                cv2.rectangle(
                    vis,
                    (last_x2 + 6, ry1 + 5),
                    (min(vis.shape[1] - 4, last_x2 + 310), ry2 - 5),
                    badge_bg,
                    -1,
                )
                cv2.putText(
                    vis,
                    badge_txt,
                    (last_x2 + 10, ry1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    badge_fg,
                    2,
                    cv2.LINE_AA,
                )

        return vis
