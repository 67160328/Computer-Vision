# -*- coding: utf-8 -*-
"""
Phase 4B: Handwritten Digit Recognition (HDR CNN Engine 0-9)
สำหรับสแกนตัวเลข (0-9) ในกรอบสีม่วงด้านบนทั้ง 42 ช่องของกระดาษคำตอบ Rayong MARCO (Template 2):
- รหัสประจำตัว (student_id): 14 ช่อง
- รหัสวิชา (subject_id): 14 ช่อง
- รหัสสถานศึกษา (school_id): 14 ช่อง

คุณลักษณะหลัก:
1. HeaderCellEmptyGate & Border Suppressor:
   กรองช่องว่างในกรอบสีม่วงออกอย่างแม่นยำด้วย Core-Zone Ink Density + Connected Component Analysis
   เพื่อไม่ให้เศษเส้นขอบตารางถูกอ่านเป็นเลข 1 หรือ 7
2. Canonical Digit Normalizer (28x28):
   แปลงภาพตัวเลขจากช่องรหัส (หรือจากชุดข้อมูล Number Handwriter 140x90 px)
   ให้เป็นภาพมาตรฐาน 28x28 พิกเซล (ตัวเลขขาวบนพื้นดำ, จัดกึ่งกลางด้วย Center-of-Mass)
3. HCRResNet28 (num_classes=10):
   สถาปัตยกรรม Residual CNN ประสิทธิภาพสูงสำหรับจำแนกตัวเลข 0-9 พร้อมคำนวณ Top-3 Probability
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from src.recognition.char_classifier import HCRResNet28, CanonicalGlyphNormalizer

DIGITS = "0123456789"


@dataclass
class DigitCellPrediction:
    """ผลลัพธ์การทำนายตัวเลขต่อ 1 ช่องในกรอบสีม่วง (Header ID)"""
    field_name: str                         # 'student_id', 'subject_id', 'school_id'
    col_idx: int                            # 1 ถึง 14
    is_empty: bool                          # True หากช่องว่างเปล่า
    predicted_digit: str                    # '0' - '9' หรือ '' (ถ้าว่าง)
    confidence: float                       # 0.0 - 1.0
    margin: float                           # P(top1) - P(top2)
    top3_digits: List[Tuple[str, float]]    # [('5', 0.98), ('6', 0.01), ('3', 0.01)]
    ink_ratio: float                        # สัดส่วนหมึกในโซนกลางช่อง
    is_low_confidence: bool                 # True หากไม่ว่างแต่ความมั่นใจต่ำ (< 0.55)
    glyph_28x28: np.ndarray                 # ภาพ 28x28 ที่แปลงแล้ว


@dataclass
class HeaderFieldResult:
    """ผลลัพธ์การอ่านรหัสประจำตัว / รหัสวิชา / รหัสสถานศึกษา รายแถว (14 ช่อง)"""
    field_name: str
    assembled_id: str                       # รหัสที่ประกอบได้ เช่น "67160328" (ข้ามช่องว่าง)
    formatted_14_slots: str                 # รูปแบบ 14 ช่อง เช่น "67160328------"
    filled_count: int                       # จำนวนช่องที่มีตัวเลขเขียน
    mean_confidence: float                  # ความมั่นใจเฉลี่ยของช่องที่เขียน
    min_confidence: float                   # ความมั่นใจต่ำสุด
    needs_human_review: bool                # True หากมีความมั่นใจต่ำ
    cell_predictions: List[DigitCellPrediction]


class HeaderDigitNormalizer:
    """
    ตัวแปลงภาพช่องตัวเลข (ทั้งจากกรอบสีม่วงบนกระดาษสอบ และจากชุดข้อมูล Number Handwriter)
    ให้เป็นภาพมาตรฐาน 28x28 พิกเซล พร้อมระบบคัดกรองช่องว่าง (Empty-Cell Gate)
    """
    @staticmethod
    def clean_border_artifacts(gray_img: np.ndarray) -> np.ndarray:
        """
        ลบเส้นขอบตารางที่อาจติดมาบริเวณขอบซ้าย/ขวา/บน/ล่าง (เช่น เส้นตั้งใน Number Handwriter หรือกรอบสีม่วง)
        """
        if gray_img is None or gray_img.size == 0:
            return np.full((64, 64), 255, dtype=np.uint8)

        img = gray_img.copy()
        h, w = img.shape[:2]
        bg_val = int(np.percentile(img, 90))

        # ตรวจสอบแถบขอบซ้าย-ขวา-บน-ล่าง (กว้าง ~7% ของภาพ) หากมีเส้นยาวต่อเนื่องเกิน 65% ของความสูง/กว้าง ให้ลบออก
        margin_x = max(3, int(w * 0.08))
        margin_y = max(3, int(h * 0.08))
        dark_thr = max(40, bg_val - 35)

        # ขอบซ้ายและขวา
        for x_col in list(range(0, margin_x)) + list(range(w - margin_x, w)):
            if np.mean(img[:, x_col] < dark_thr) > 0.55:
                img[:, x_col] = bg_val

        # ขอบบนและล่าง
        for y_row in list(range(0, margin_y)) + list(range(h - margin_y, h)):
            if np.mean(img[y_row, :] < dark_thr) > 0.55:
                img[y_row, :] = bg_val

        return img

    @staticmethod
    def to_canonical_digit_28x28(cleaned: np.ndarray) -> np.ndarray:
        """
        แปลงภาพตัวเลขจากกรอบสีม่วงให้เป็น 28x28 โดยไม่ทำ Dilation หนาเกินไปจนรูของเลข 8, 6, 9, 0 ตัน
        """
        if cleaned is None or cleaned.size == 0:
            return np.zeros((28, 28), dtype=np.uint8)

        h, w = cleaned.shape[:2]
        pad_y = max(2, int(h * 0.05))
        pad_x = max(2, int(w * 0.05))
        roi = cleaned[pad_y : h - pad_y, pad_x : w - pad_x]
        if roi.size == 0:
            roi = cleaned

        bg_level = float(np.percentile(roi, 92))
        fg = np.clip((bg_level - roi.astype(np.float32)), 0.0, 255.0)
        max_fg = float(np.max(fg))
        if max_fg < 25.0:
            return np.zeros((28, 28), dtype=np.uint8)

        binary = (fg > max(22.0, 0.22 * max_fg)).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        rh, rw = roi.shape
        valid_mask = np.zeros_like(binary)
        for lbl in range(1, num_labels):
            x, y, bw, bh, area = stats[lbl]
            if area < 10:
                continue
            if (y == 0 or y + bh == rh) and bh <= 4 and bw > 12:
                continue
            if (x == 0 or x + bw == rw) and bw <= 4 and bh > 12:
                continue
            valid_mask[labels == lbl] = 255

        if np.sum(valid_mask) == 0:
            valid_mask = binary

        fg_clean = np.where(valid_mask > 0, fg, 0.0)
        p95 = float(np.percentile(fg_clean[valid_mask > 0], 95)) if np.any(valid_mask > 0) else max_fg
        fg_norm = np.clip(fg_clean * (255.0 / max(p95, 30.0)), 0.0, 255.0).astype(np.uint8)

        ys, xs = np.where(valid_mask > 0)
        if len(ys) == 0:
            return np.zeros((28, 28), dtype=np.uint8)

        y1, y2 = int(np.min(ys)), int(np.max(ys)) + 1
        x1, x2 = int(np.min(xs)), int(np.max(xs)) + 1
        crop = fg_norm[y1:y2, x1:x2]
        ch, cw = crop.shape

        # ทำ Dilation(2,2) เฉพาะกรณีเส้นบางมากๆ (ink_density < 0.14) เพื่อรักษาช่องว่างของเลข 8, 6, 9
        ink_density = float(np.mean(crop > 80))
        if max(ch, cw) >= 26 and ink_density < 0.14:
            crop = cv2.dilate(crop, np.ones((2, 2), np.uint8), iterations=1)

        scale = 20.0 / float(max(ch, cw))
        new_w = max(1, min(20, int(round(cw * scale))))
        new_h = max(1, min(20, int(round(ch * scale))))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        off_y = (28 - new_h) // 2
        off_x = (28 - new_w) // 2
        canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized

        M = cv2.moments(canvas.astype(np.float32))
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            shift_x = float(np.clip(13.5 - cx, -3.5, 3.5))
            shift_y = float(np.clip(13.5 - cy, -3.5, 3.5))
            trans_mat = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
            canvas = cv2.warpAffine(
                canvas, trans_mat, (28, 28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

        if np.max(canvas) > 0:
            canvas = np.clip(canvas.astype(np.float32) * (255.0 / float(np.max(canvas))), 0, 255).astype(np.uint8)
        return canvas

    @classmethod
    def check_is_empty_and_normalize(
        cls, clean_patch: np.ndarray, empty_ink_threshold: float = 0.012
    ) -> Tuple[bool, float, np.ndarray]:
        """
        ตรวจสอบว่าช่องรหัสสีม่วงว่างหรือไม่ และแปลงเป็นภาพ 28x28
        คืนค่า: (is_empty, ink_ratio, glyph_28x28)
        """
        if clean_patch is None or clean_patch.size == 0:
            return True, 0.0, np.zeros((28, 28), dtype=np.uint8)

        cleaned = cls.clean_border_artifacts(clean_patch)
        h, w = cleaned.shape[:2]

        # วัดหมึกเฉพาะโซนกลางช่อง (Core Zone 76%) เพื่อป้องกันขอบตารางหลอก
        my = max(3, int(round(h * 0.12)))
        mx = max(3, int(round(w * 0.12)))
        core = cleaned[my : h - my, mx : w - mx]
        if core.size == 0:
            return True, 0.0, np.zeros((28, 28), dtype=np.uint8)

        bg_level = float(np.percentile(core, 92))
        fg_core = np.clip(bg_level - core.astype(np.float32), 0.0, 255.0)
        max_fg = float(np.max(fg_core))

        # กรณีพื้นหลังขาวล้วน ไม่มีรอยหมึกที่ชัดเจน
        if max_fg < 32.0:
            return True, 0.0, np.zeros((28, 28), dtype=np.uint8)

        ink_mask = (fg_core > max(28.0, 0.28 * max_fg)).astype(np.uint8)
        ink_ratio = float(np.mean(ink_mask > 0))
        if ink_ratio < empty_ink_threshold:
            return True, ink_ratio, np.zeros((28, 28), dtype=np.uint8)

        glyph = cls.to_canonical_digit_28x28(cleaned)
        if np.sum(glyph > 50) < 14:
            return True, ink_ratio, np.zeros((28, 28), dtype=np.uint8)

        return False, ink_ratio, glyph


class DigitRecognizerEngine:
    """
    เอนจินสำหรับโหลดโมเดล HCRResNet28 (10 คลาส 0-9) และสแกนตัวเลขในกรอบสีม่วง 42 ช่อง
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        temperature: float = 1.10,
        low_conf_threshold: float = 0.55,
        empty_ink_threshold: float = 0.012,
    ):
        self.device = torch.device(device)
        self.temperature = temperature
        self.low_conf_threshold = low_conf_threshold
        self.empty_ink_threshold = empty_ink_threshold
        self.normalizer = HeaderDigitNormalizer()

        self.model = HCRResNet28(num_classes=10, dropout=0.0).to(self.device)
        self.mean_norm = 0.1307
        self.std_norm = 0.3081

        if weights_path is None:
            default_weights = os.path.join(os.path.dirname(__file__), "..", "..", "models", "hcr_resnet28_digits.pt")
            if os.path.exists(default_weights):
                weights_path = default_weights

        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
                self.mean_norm = float(state.get("mean_norm", 0.1307))
                self.std_norm = float(state.get("std_norm", 0.3081))
            else:
                self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict_header_fields(
        self, header_ids: Dict[str, List[np.ndarray]]
    ) -> Dict[str, HeaderFieldResult]:
        """
        รับ dict ของภาพช่องรหัสทั้ง 3 แถว:
          - 'student_id': 14 ช่อง
          - 'subject_id': 14 ช่อง
          - 'school_id': 14 ช่อง
        คืนค่าผลลัพธ์การสแกนตัวเลขครบทั้ง 42 ช่องพร้อมรหัสที่ประกอบเสร็จสมบูรณ์
        """
        results: Dict[str, HeaderFieldResult] = {}

        # รวบรวมช่องที่ไม่ว่างทั้งหมดเพื่อรัน CNN แบบ Batch ครั้งเดียว
        non_empty_meta: List[Tuple[str, int, float, np.ndarray]] = []
        cell_states: Dict[Tuple[str, int], Tuple[bool, float, np.ndarray]] = {}

        for field_name in ["student_id", "subject_id", "school_id"]:
            patches = header_ids.get(field_name, [])
            for idx, patch in enumerate(patches):
                col_idx = idx + 1
                is_empty, ink_r, glyph = self.normalizer.check_is_empty_and_normalize(
                    patch, empty_ink_threshold=self.empty_ink_threshold
                )
                cell_states[(field_name, col_idx)] = (is_empty, ink_r, glyph)
                if not is_empty:
                    non_empty_meta.append((field_name, col_idx, ink_r, glyph))

        # รัน Batch Inference เฉพาะช่องที่มีตัวเลขเขียน
        pred_lookup: Dict[Tuple[str, int], Tuple[str, float, float, List[Tuple[str, float]]]] = {}
        if non_empty_meta:
            glyphs_batch = np.stack([item[3] for item in non_empty_meta], axis=0).astype(np.float32) / 255.0
            tensor_batch = torch.from_numpy(
                (glyphs_batch[:, None, :, :] - self.mean_norm) / self.std_norm
            ).to(self.device)
            logits = self.model(tensor_batch) / self.temperature
            probs = F.softmax(logits, dim=-1).cpu().numpy()

            for i, (f_name, c_idx, _, glyph_img) in enumerate(non_empty_meta):
                p = probs[i]
                top3_idx = np.argsort(p)[::-1][:3]
                t1, t2 = int(top3_idx[0]), int(top3_idx[1])
                conf = float(p[t1])
                margin = float(p[t1] - p[t2])
                top3_list = [(DIGITS[int(k)], round(float(p[int(k)]), 4)) for k in top3_idx]
                pred_digit = DIGITS[t1]

                # Topological Disambiguation for 7 vs 9:
                # เลข 9 ตามสัณฐานวิทยาต้องมีวงปิด (Closed Loop / Euler Hole) บริเวณส่วนบน
                # หากทำนายเป็น 9 แต่ไม่มีวงปิดด้านใน (No Enclosed Cavity) และมี '7' ใน Top-3
                if pred_digit == "9":
                    bin_g = (glyph_img > 60).astype(np.uint8) * 255
                    cnts, hier = cv2.findContours(bin_g, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    has_hole = False
                    if hier is not None:
                        for k, h in enumerate(hier[0]):
                            if h[3] != -1 and cv2.contourArea(cnts[k]) >= 4.0:
                                has_hole = True
                    if not has_hole and conf < 0.85:
                        p_7 = float(p[7])
                        top3_dict = dict(top3_list)
                        if p_7 >= 0.025 or ("7" in top3_dict and top3_dict["7"] >= 0.02):
                            pred_digit = "7"
                            conf = max(conf, p_7)

                pred_lookup[(f_name, c_idx)] = (pred_digit, conf, margin, top3_list)

        # ประกอบผลลัพธ์รายแถว (14 ช่องต่อแถว)
        for field_name in ["student_id", "subject_id", "school_id"]:
            patches = header_ids.get(field_name, [])
            cell_preds: List[DigitCellPrediction] = []
            assembled_chars: List[str] = []
            slot_chars: List[str] = []
            confs: List[float] = []

            for idx in range(len(patches)):
                col_idx = idx + 1
                is_empty, ink_r, glyph = cell_states[(field_name, col_idx)]
                if is_empty:
                    cell_preds.append(
                        DigitCellPrediction(
                            field_name=field_name,
                            col_idx=col_idx,
                            is_empty=True,
                            predicted_digit="",
                            confidence=1.0,
                            margin=1.0,
                            top3_digits=[],
                            ink_ratio=round(ink_r, 4),
                            is_low_confidence=False,
                            glyph_28x28=glyph,
                        )
                    )
                    slot_chars.append("-")
                else:
                    d_char, conf, margin, top3_list = pred_lookup[(field_name, col_idx)]
                    is_low = conf < self.low_conf_threshold
                    cell_preds.append(
                        DigitCellPrediction(
                            field_name=field_name,
                            col_idx=col_idx,
                            is_empty=False,
                            predicted_digit=d_char,
                            confidence=round(conf, 4),
                            margin=round(margin, 4),
                            top3_digits=top3_list,
                            ink_ratio=round(ink_r, 4),
                            is_low_confidence=is_low,
                            glyph_28x28=glyph,
                        )
                    )
                    assembled_chars.append(d_char)
                    slot_chars.append(d_char)
                    confs.append(conf)

            mean_c = float(np.mean(confs)) if confs else 1.0
            min_c = float(np.min(confs)) if confs else 1.0
            needs_rev = any(cp.is_low_confidence for cp in cell_preds)

            results[field_name] = HeaderFieldResult(
                field_name=field_name,
                assembled_id="".join(assembled_chars),
                formatted_14_slots="".join(slot_chars),
                filled_count=len(confs),
                mean_confidence=round(mean_c, 4),
                min_confidence=round(min_c, 4),
                needs_human_review=needs_rev,
                cell_predictions=cell_preds,
            )

        return results

    @torch.no_grad()
    def predict_glyphs_batch(self, glyphs_28x28: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        รับอาเรย์ (N, 28, 28) uint8 มาทำนายแบบ Batch สำหรับการประเมินผลชุดข้อมูลขนาดใหญ่
        คืนค่า: (pred_labels, top1_confs, all_probs)
        """
        if len(glyphs_28x28) == 0:
            return np.array([]), np.array([]), np.zeros((0, 10))

        batch_np = glyphs_28x28.astype(np.float32) / 255.0
        tensor_batch = torch.from_numpy(
            (batch_np[:, None, :, :] - self.mean_norm) / self.std_norm
        ).to(self.device)

        probs_list = []
        bs = 512
        for i in range(0, len(tensor_batch), bs):
            logits = self.model(tensor_batch[i : i + bs]) / self.temperature
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            probs_list.append(probs)

        all_probs = np.concatenate(probs_list, axis=0)
        preds = np.argmax(all_probs, axis=1)
        confs = np.max(all_probs, axis=1)
        return preds, confs, all_probs
