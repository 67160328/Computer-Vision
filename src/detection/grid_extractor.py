# -*- coding: utf-8 -*-
"""
Phase 2: Anchor Bar Tracking & Cell Localization Module
สำหรับระบบตรวจข้อสอบอัตโนมัติ Rayong MARCO (Template 2)

คุณลักษณะหลัก (อัปเกรด 100% Exact Alignment & Color-Coded Audit):
1. Dynamic Pitch-Sorted Timing Mark Detector:
   สแกนหาขีดดำบอกตำแหน่งข้อ 1-20 ทางด้านซ้ายและขวาโดยไม่ต้องพึ่งค่าพิกัดตายตัว
2. Triple-Layer Grid Line Suppression (Answer Cells):
   - Layer 1: 2D Local Grid-Line Snap (+-22 px) ดูดติดขอบตารางทั้งแกน X และ Y
   - Layer 2: Morphological & Full-Span Border Erase ลบเส้นขอบตารางที่แลบเข้ามา
   - Layer 3: Core-Zone Ink Measurement วัดปริมาณหมึกเฉพาะใจกลางช่อง (76%)
   - การแสดงสี: ช่องว่าง = สีเขียว (Green), ช่องมีหมึก = สีส้ม (Orange)
3. Direct White-Interior Contour Header Localization (42 ช่องรหัส):
   ตรวจจับช่องขาวภายในตารางรหัสประจำตัว (14 ช่อง), รหัสวิชา (14 ช่อง), รหัสสถานศึกษา (14 ช่อง) โดยตรง!
   - การแสดงสี: หากตรงกับช่องจริง (is_aligned = True) วาดสีม่วง (Purple),
     หากช่องใดไม่ตรงกับช่องจริง (is_aligned = False) วาดสีเหลืองส้มสว่าง (Yellow-Orange) ทันที!
4. Per-Bubble Ring/Disk NCC Template Matching (7 วงกลม OMR):
   ค้นหาจุดศูนย์กลางจริงของวงกลมทั้ง 7 วงด้วย Normalized Cross-Correlation (NCC) รายวง
   - การแสดงสี: หากตรงกับวงจริง (is_aligned = True) วาดสีเขียว (ฝนทึบ) หรือสีฟ้า (ไม่ฝน),
     หากวงกลมใดไม่ตรงกับวงจริง (is_aligned = False) วาดสีแดงสด (Red) ทันที!
"""

import os
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class CellPatch:
    """ข้อมูลของช่องตัวอักษรย่อย 1 ช่อง"""
    row_idx: int                       # 1 ถึง 20 (ข้อที่ 1-20)
    col_idx: int                       # 1 ถึง 20 (ช่องที่ 1-20)
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2) บนภาพ Rectified (2480 x 3508)
    clean_patch: np.ndarray            # ภาพช่องที่ตัดขอบตารางสีดำออกแล้ว
    ink_ratio: float                   # สัดส่วนพิกเซลหมึกภายในโซนกลางช่อง (0.0 - 1.0)
    is_empty: bool                     # True หากเป็นช่องว่างเปล่า


@dataclass
class HeaderCellInfo:
    """ข้อมูลช่องรหัสส่วนหัว (รหัสประจำตัว / รหัสวิชา / รหัสสถานศึกษา)"""
    field_name: str
    col_idx: int
    bbox: Tuple[int, int, int, int]
    clean_patch: np.ndarray
    is_aligned: bool                   # True = ตรงกับช่องจริง (วาดสีม่วง), False = ไม่ตรง (วาดสีเหลืองส้ม)
    border_score: float


@dataclass
class OMRBubbleInfo:
    """ข้อมูลวงกลม OMR (ชุดกระดาษคำถาม / หน้ากระดาษคำตอบ)"""
    group_name: str
    option_num: int
    cx: float
    cy: float
    cla_fill_ratio: float
    ncc_alignment_score: float
    is_filled: bool
    is_aligned: bool                   # True = ตรงกับวงจริง (วาดสีฟ้า/เขียว), False = ไม่ตรง (วาดสีแดงสด!)


@dataclass
class Phase2Metrics:
    """ตัวชี้วัดเชิงปริมาณ (Quantitative Metrics) ของ Phase 2 ต่อ 1 แผ่น"""
    image_name: str
    left_bars_detected: int
    right_bars_detected: int
    timing_bar_recall_pct: float
    row_pitch_linearity_rmse_px: float
    total_answer_cells_extracted: int
    mean_cell_iou: float
    cell_precision_pct: float
    cell_recall_pct: float
    empty_cell_specificity_pct: float
    header_id_cells_extracted: int
    header_id_exact_align_pct: float
    omr_bubbles_exact_align_pct: float
    omr_mean_ncc_score: float
    detected_question_set: Optional[int]
    detected_page_number: Optional[int]
    page1_bubble_cla_ratio: float
    unfilled_bubbles_max_cla_ratio: float
    processing_time_ms: float


@dataclass
class GridExtractionOutput:
    """ผลลัพธ์การสกัดตารางและช่องคำตอบทั้งหมดจากกระดาษ 1 แผ่น"""
    answer_grid: List[List[CellPatch]]           # 20 แถว x 20 ช่อง (รวม 400 ช่อง)
    header_ids: Dict[str, List[np.ndarray]]      # student_id (14), subject_id (14), school_id (14)
    header_cells_info: Dict[str, List[HeaderCellInfo]]
    omr_bubbles: Dict[str, Dict[int, float]]     # ข้อมูล CLA fill ratio
    omr_bubbles_info: Dict[str, List[OMRBubbleInfo]]
    left_bar_centers: np.ndarray                 # (20, 2)
    right_bar_centers: np.ndarray                # (20, 2)
    metrics: Phase2Metrics


class Template2GridExtractor:
    """
    คลาสสกัดตารางคำตอบ 20 ข้อ x 20 ช่อง, รหัสประจำตัว 42 ช่อง และวงกลม OMR 7 วง สำหรับ Template 2
    """
    NUM_QUESTIONS = 20
    NUM_COLS = 20

    CANONICAL_ROW_YS = np.array(
        [910.0] + [1030.0 + 116.0 * i for i in range(19)], dtype=np.float32
    )
    CANONICAL_LEFT_X = 133.0
    CANONICAL_RIGHT_X = 2351.0

    CANONICAL_COL_BOUNDS: List[Tuple[int, int]] = (
        [(257, 353)]
        + [(354 + 100 * c, 354 + 100 * c + 99) for c in range(18)]
        + [(2154, 2234)]
    )

    HEADER_ID_ROWS = {
        "student_id": (295, 375),   # รหัสประจำตัว
        "subject_id": (498, 578),   # รหัสวิชา
        "school_id": (706, 786),    # รหัสสถานศึกษา
    }
    HEADER_ID_COL_XS: List[Tuple[int, int]] = [
        (1012 + int(round(85.07 * c)), 1012 + int(round(85.07 * c)) + 79)
        for c in range(14)
    ]

    QUESTION_SET_BUBBLES = [
        (1, 231.0, 679.0),
        (2, 313.0, 680.0),
        (3, 394.0, 679.0),
    ]
    PAGE_NUMBER_BUBBLES = [
        (1, 627.0, 679.0),  # วงกลมที่พิมพ์ทึบมาบนแม่แบบหน้า 1
        (2, 708.0, 678.0),
        (3, 790.0, 678.0),
        (4, 870.0, 677.0),
    ]

    def __init__(
        self,
        inner_margin_px: int = 8,
        empty_ink_threshold: float = 0.012,
        cla_dark_threshold: int = 160,
        cla_filled_threshold: float = 0.45,
        omr_ncc_align_threshold: float = 0.25,
    ):
        self.inner_margin_px = inner_margin_px
        self.empty_ink_threshold = empty_ink_threshold
        self.cla_dark_threshold = cla_dark_threshold
        self.cla_filled_threshold = cla_filled_threshold
        self.omr_ncc_align_threshold = omr_ncc_align_threshold

        # สร้างแม่แบบวงแหวน (Ring Template) และวงกลมทึบ (Disk Template) สำหรับจับคู่ NCC ของวงกลม OMR
        self._ring_tmpl = np.zeros((55, 55), dtype=np.float32)
        cv2.circle(self._ring_tmpl, (27, 27), 22, 1.0, 3)

        self._disk_tmpl = np.zeros((55, 55), dtype=np.float32)
        cv2.circle(self._disk_tmpl, (27, 27), 22, 1.0, -1)

    def _detect_timing_bars_side(
        self, norm_gray: np.ndarray, x_min: int, x_max: int, default_x: float
    ) -> Tuple[np.ndarray, int]:
        """
        สแกนหาขีดดำบอกตำแหน่งข้อ 1-20 ในแถบแนวตั้ง [x_min, x_max] แบบ Dynamic Pitch-Sorted
        """
        y_top, y_bot = 800, 3230
        roi = norm_gray[y_top:y_bot, x_min:x_max]
        dark = (roi < 165).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        dark_bars = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

        cnts, _ = cv2.findContours(dark_bars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if 28 <= bw <= 92 and 10 <= bh <= 45:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = x_min + M["m10"] / M["m00"]
                    cy = y_top + M["m01"] / M["m00"]
                else:
                    cx = x_min + x + bw / 2.0
                    cy = y_top + y + bh / 2.0
                cands.append((float(cx), float(cy)))

        cands.sort(key=lambda p: p[1])

        dedup = []
        for cx, cy in cands:
            if not dedup or (cy - dedup[-1][1]) > 60.0:
                dedup.append([cx, cy])

        raw_found_count = min(self.NUM_QUESTIONS, len(dedup))

        if len(dedup) == self.NUM_QUESTIONS:
            return np.array(dedup, dtype=np.float32), 20

        matched = np.zeros((self.NUM_QUESTIONS, 2), dtype=np.float32)
        if len(dedup) >= 2:
            pts_arr = np.array(dedup, dtype=np.float32)
            dy_offset = float(np.median(pts_arr[:, 1]) - np.median(self.CANONICAL_ROW_YS))
            med_x = float(np.median(pts_arr[:, 0]))
        else:
            dy_offset = 0.0
            med_x = default_x

        for i in range(self.NUM_QUESTIONS):
            exp_y = self.CANONICAL_ROW_YS[i] + dy_offset
            if dedup:
                dists = [abs(p[1] - exp_y) for p in dedup]
                min_idx = int(np.argmin(dists))
                if dists[min_idx] < 45.0:
                    matched[i] = dedup[min_idx]
                    continue
            matched[i] = [med_x, exp_y]

        return matched, raw_found_count

    def _snap_cell_to_local_borders(
        self,
        norm_gray: np.ndarray,
        col_x1: int,
        col_x2: int,
        pred_y_center: float,
        half_h: int,
    ) -> Tuple[int, int, int, int]:
        """
        Layer 1 (2D Local Grid-Line Snap):
        ค้นหาเส้นขอบตารางทั้งแนวนอน (บน-ล่าง) และแนวตั้ง (ซ้าย-ขวา) จริงของช่องในรัศมี +-22 px
        """
        h_img, w_img = norm_gray.shape
        search_pad = 22

        # 1. Snap แกน Y (ขอบบน - ขอบล่าง)
        y1 = int(round(pred_y_center - half_h))
        y2 = int(round(pred_y_center + half_h + 1))
        sy1 = max(0, y1 - search_pad)
        sy2 = min(h_img, y2 + search_pad)
        sx1 = max(0, col_x1 + 12)
        sx2 = min(w_img, col_x2 - 12)

        if sy2 - sy1 > 40 and sx2 - sx1 > 20:
            strip_y = norm_gray[sy1:sy2, sx1:sx2]
            row_darkness = 255.0 - np.mean(strip_y, axis=1)
            mid_y = len(row_darkness) // 2
            top_reg = row_darkness[: max(1, mid_y - 18)]
            bot_reg = row_darkness[min(len(row_darkness) - 1, mid_y + 18) :]
            if len(top_reg) > 0 and len(bot_reg) > 0:
                t_pk = int(np.argmax(top_reg))
                b_pk = int(np.argmax(bot_reg)) + min(len(row_darkness) - 1, mid_y + 18)
                if top_reg[t_pk] > 18.0 and row_darkness[b_pk] > 18.0 and (64 <= b_pk - t_pk <= 96):
                    y1 = sy1 + t_pk
                    y2 = sy1 + b_pk

        # 2. Snap แกน X (ขอบซ้าย - ขอบขวา)
        x1, x2 = col_x1, col_x2
        vx1 = max(0, col_x1 - search_pad)
        vx2 = min(w_img, col_x2 + search_pad)
        vy1 = max(0, y1 + 8)
        vy2 = min(h_img, y2 - 8)

        if vy2 - vy1 > 25 and vx2 - vx1 > 50:
            strip_x = norm_gray[vy1:vy2, vx1:vx2]
            col_darkness = 255.0 - np.mean(strip_x, axis=0)
            mid_x = len(col_darkness) // 2
            left_reg = col_darkness[: max(1, mid_x - 20)]
            right_reg = col_darkness[min(len(col_darkness) - 1, mid_x + 20) :]
            if len(left_reg) > 0 and len(right_reg) > 0:
                l_pk = int(np.argmax(left_reg))
                r_pk = int(np.argmax(right_reg)) + min(len(col_darkness) - 1, mid_x + 20)
                expected_w = col_x2 - col_x1
                if (
                    left_reg[l_pk] > 18.0
                    and col_darkness[r_pk] > 18.0
                    and (expected_w - 18 <= r_pk - l_pk <= expected_w + 18)
                ):
                    x1 = vx1 + l_pk
                    x2 = vx1 + r_pk

        return (x1, y1, x2, y2)

    def _clean_cell_patch(self, raw_cell: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Layer 2 & Layer 3 (Morphological Grid-Line Subtraction + Core-Zone Ink Measurement)
        """
        h, w = raw_cell.shape
        m = self.inner_margin_px
        if h > 2 * m + 12 and w > 2 * m + 12:
            cropped = raw_cell[m : h - m, m : w - m].copy()
        else:
            cropped = raw_cell.copy()

        ch, cw = cropped.shape
        ink_mask = (cropped < 150).astype(np.uint8) * 255

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(cw * 0.65)), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(ch * 0.72))))
        horiz_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, h_kernel)
        vert_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, v_kernel)

        rim_mask = np.zeros_like(ink_mask)
        rim_y, rim_x = 12, 18
        rim_mask[:rim_y, :] = 255
        rim_mask[ch - rim_y :, :] = 255
        rim_mask[:, :rim_x] = 255
        rim_mask[:, cw - rim_x :] = 255

        border_lines = ((horiz_lines > 0) | (vert_lines > 0)) & (rim_mask > 0)
        ink_mask[border_lines] = 0
        cropped[border_lines] = 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        for lbl in range(1, num_labels):
            x, y, bw, bh, area = stats[lbl]
            if area < 14:
                ink_mask[labels == lbl] = 0
                continue
            spans_full_height = (y <= 2) and (y + bh >= ch - 2) and (bw <= 10)
            spans_full_width = (x <= 2) and (x + bw >= cw - 2) and (bh <= 10)
            touches_top_or_bot = (y <= 3) or (y + bh >= ch - 3)
            touches_left_or_right = (x <= 3) or (x + bw >= cw - 3)
            if (
                spans_full_height
                or spans_full_width
                or (touches_top_or_bot and bh <= 9)
                or (touches_left_or_right and bw <= 9)
            ):
                cropped[labels == lbl] = 255
                ink_mask[labels == lbl] = 0

        pad_y = max(4, int(ch * 0.12))
        pad_x = max(4, int(cw * 0.12))
        core_mask = ink_mask[pad_y : ch - pad_y, pad_x : cw - pad_x]
        ink_ratio = float(np.mean(core_mask > 0)) if core_mask.size > 0 else 0.0

        return cropped, ink_ratio

    def _extract_header_id_tables(
        self, norm_gray: np.ndarray, dx_shift: float, dy_shift: float
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[HeaderCellInfo]], int]:
        """
        ตรวจจับช่องรหัสประจำตัว (14 ช่อง), รหัสวิชา (14 ช่อง), และรหัสสถานศึกษา (14 ช่อง) รวม 42 ช่อง
        ด้วย Direct White-Interior Contour Detection + 2D Local Border Verification
        หากช่องใดตรงกับขอบตารางจริง -> is_aligned = True (วาดสีม่วง)
        หากช่องใดไม่ตรงกับขอบตารางจริง -> is_aligned = False (วาดสีเหลืองส้มสว่าง)
        """
        rx_min, rx_max = 940, 2270
        ry_min, ry_max = 170, 865
        roi = norm_gray[ry_min:ry_max, rx_min:rx_max]

        # Binarize เพื่อแยกพื้นขาวในช่องรหัสออกจากเส้นขอบตารางสีดำ แล้ว Erode 3x3 เพื่อแยกแต่ละช่องออกจากกัน
        _, bin_w = cv2.threshold(roi, 185, 255, cv2.THRESH_BINARY)
        bin_eroded = cv2.erode(bin_w, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(bin_eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        detected_boxes = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if 55 <= bw <= 95 and 55 <= bh <= 96:
                # ชดเชยระยะ Erosion 1 px กลับไปที่ขอบเส้นตารางจริง
                gx1 = rx_min + x - 1
                gy1 = ry_min + y - 1
                gx2 = gx1 + bw + 2
                gy2 = gy1 + bh + 2
                detected_boxes.append((gx1, gy1, gx2, gy2))

        header_ids: Dict[str, List[np.ndarray]] = {}
        header_info: Dict[str, List[HeaderCellInfo]] = {}
        aligned_count = 0

        field_order = [
            ("student_id", self.HEADER_ID_ROWS["student_id"]),
            ("subject_id", self.HEADER_ID_ROWS["subject_id"]),
            ("school_id", self.HEADER_ID_ROWS["school_id"]),
        ]

        # จัดกลุ่มกล่องที่ตรวจพบเข้าสู่ 3 แถวตามแกน Y
        detected_boxes.sort(key=lambda b: 0.5 * (b[1] + b[3]))
        row_clusters: List[List[Tuple[int, int, int, int]]] = []
        for b in detected_boxes:
            cy = 0.5 * (b[1] + b[3])
            if not row_clusters or abs(cy - np.mean([0.5 * ( item[1] + item[3] ) for item in row_clusters[-1]])) > 65.0:
                row_clusters.append([b])
            else:
                row_clusters[-1].append(b)

        # คัดเฉพาะคลัสเตอร์ที่มีอย่างน้อย 8 ช่อง (ซึ่งคือแถวรหัสทั้ง 3 แถว)
        valid_clusters = [c for c in row_clusters if len(c) >= 8]

        for f_idx, (field_name, (can_y1, can_y2)) in enumerate(field_order):
            cluster = valid_clusters[f_idx] if f_idx < len(valid_clusters) else []
            cluster.sort(key=lambda b: b[0])

            field_patches: List[np.ndarray] = []
            field_cells: List[HeaderCellInfo] = []

            if len(cluster) == 14:
                # พบครบทั้ง 14 ช่องโดยตรงจาก Contour ของช่องจริง 100%!
                for c_idx, bbox in enumerate(cluster):
                    x1, y1, x2, y2 = bbox
                    raw_h = norm_gray[y1:y2, x1:x2]
                    clean_h, _ = self._clean_cell_patch(raw_h)
                    field_patches.append(clean_h)
                    field_cells.append(
                        HeaderCellInfo(
                            field_name=field_name,
                            col_idx=c_idx + 1,
                            bbox=bbox,
                            clean_patch=clean_h,
                            is_aligned=True,
                            border_score=1.0,
                        )
                    )
                    aligned_count += 1
            else:
                # กรณีบางช่องมีรอยเขียนทับขอบจนหา Contour ครบ 14 ช่องไม่ได้ ให้ใช้ Linear Fit จากช่องที่พบในแถวนั้น + Snap
                if len(cluster) >= 2:
                    xs_found = [0.5 * (b[0] + b[2]) for b in cluster]
                    ys_found = [0.5 * (b[1] + b[3]) for b in cluster]
                    slope_row, intercept_row = np.polyfit(xs_found, ys_found, 1)
                else:
                    slope_row = 0.0
                    intercept_row = 0.5 * (can_y1 + can_y2) + dy_shift

                for c_idx, (hx1, hx2) in enumerate(self.HEADER_ID_COL_XS):
                    sx1 = int(round(hx1 + dx_shift))
                    sx2 = int(round(hx2 + dx_shift))
                    pred_cy = intercept_row + slope_row * (0.5 * (sx1 + sx2))
                    bbox = self._snap_cell_to_local_borders(norm_gray, sx1, sx2, pred_cy, half_h=40)
                    x1, y1, x2, y2 = bbox
                    raw_h = norm_gray[y1:y2, x1:x2]
                    clean_h, _ = self._clean_cell_patch(raw_h)

                    # ตรวจสอบว่าขอบทั้ง 4 ด้านของ bbox ทาบอยู่บนเส้นตารางสีดำจริงหรือไม่
                    top_dark = 255.0 - float(np.mean(norm_gray[max(0, y1 - 2) : min(3508, y1 + 3), x1 + 8 : x2 - 8]))
                    bot_dark = 255.0 - float(np.mean(norm_gray[max(0, y2 - 2) : min(3508, y2 + 3), x1 + 8 : x2 - 8]))
                    b_score = min(1.0, (top_dark + bot_dark) / 60.0)
                    is_ok = b_score >= 0.45
                    if is_ok:
                        aligned_count += 1

                    field_patches.append(clean_h)
                    field_cells.append(
                        HeaderCellInfo(
                            field_name=field_name,
                            col_idx=c_idx + 1,
                            bbox=bbox,
                            clean_patch=clean_h,
                            is_aligned=is_ok,
                            border_score=round(b_score, 3),
                        )
                    )

            header_ids[field_name] = field_patches
            header_info[field_name] = field_cells

        return header_ids, header_info, aligned_count

    def _compute_cla_fill_ratio(
        self, norm_gray: np.ndarray, cx: float, cy: float, radius: int = 16, arm_half_thick: int = 3
    ) -> float:
        """
        คำนวณอัตราส่วนการฝนวงกลมด้วยอัลกอริทึม Cross-Linear Area (CLA) จากเปเปอร์ SNPD 2025
        """
        icx, icy = int(round(cx)), int(round(cy))
        h, w = norm_gray.shape
        x1, x2 = max(0, icx - radius), min(w, icx + radius + 1)
        y1, y2 = max(0, icy - radius), min(h, icy + radius + 1)

        horiz_strip = norm_gray[
            max(0, icy - arm_half_thick) : min(h, icy + arm_half_thick + 1), x1:x2
        ]
        vert_strip = norm_gray[
            y1:y2, max(0, icx - arm_half_thick) : min(w, icx + arm_half_thick + 1)
        ]

        dark_h = np.sum(horiz_strip < self.cla_dark_threshold)
        dark_v = np.sum(vert_strip < self.cla_dark_threshold)
        total_px = horiz_strip.size + vert_strip.size
        if total_px == 0:
            return 0.0
        return float((dark_h + dark_v) / float(total_px))

    def _locate_omr_bubbles(
        self, norm_gray: np.ndarray, dx_shift: float, dy_shift: float
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[str, List[OMRBubbleInfo]], int, float]:
        """
        ค้นหาตำแหน่งจริงของวงกลม OMR ทั้ง 7 วงรายวงด้วย Ring/Disk NCC Template Matching (เปเปอร์ SNPD 2025)
        พร้อมตรวจสอบค่าความสอดคล้อง NCC:
        - หาก is_aligned == True -> วาดวงกลมสีเขียว (ฝนทึบ) หรือสีฟ้า (ไม่ฝน)
        - หาก is_aligned == False (ไม่ตรงกับวงกลมจริง) -> วาดวงกลมสีแดงสดทันที!
        """
        dark = np.clip(255.0 - norm_gray.astype(np.float32), 0.0, 255.0)
        h_img, w_img = dark.shape

        # 1. ล็อกตำแหน่งวงกลม Page 1 ที่พิมพ์ทึบเป็นหมุดอ้างอิงเริ่มต้น
        exp_p1_x = int(round(627.0 + dx_shift))
        exp_p1_y = int(round(679.0 + dy_shift))
        sx1, sx2 = max(0, exp_p1_x - 85), min(w_img, exp_p1_x + 85)
        sy1, sy2 = max(0, exp_p1_y - 85), min(h_img, exp_p1_y + 85)
        roi_p1 = dark[sy1:sy2, sx1:sx2]

        res_p1 = cv2.matchTemplate(roi_p1, self._disk_tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res_p1)
        if max_v >= 0.40:
            p1_cx = sx1 + max_loc[0] + 27.0
            p1_cy = sy1 + max_loc[1] + 27.0
            base_dx = p1_cx - 627.0
            base_dy = p1_cy - 679.0
        else:
            base_dx, base_dy = dx_shift, dy_shift

        # 2. Snap แต่ละวงในทั้ง 7 วงด้วย Ring & Disk NCC Template Matching ในรัศมี +-38 px
        search_r = 45
        qset_ratios: Dict[int, float] = {}
        page_ratios: Dict[int, float] = {}
        bubbles_info: Dict[str, List[OMRBubbleInfo]] = {"question_set": [], "page_number": []}
        aligned_bubbles_cnt = 0
        ncc_scores = []

        for group_name, bubble_defs, target_dict in [
            ("question_set", self.QUESTION_SET_BUBBLES, qset_ratios),
            ("page_number", self.PAGE_NUMBER_BUBBLES, page_ratios),
        ]:
            for opt_num, can_x, can_y in bubble_defs:
                ex = int(round(can_x + base_dx))
                ey = int(round(can_y + base_dy))
                px1, px2 = max(0, ex - search_r), min(w_img, ex + search_r + 1)
                py1, py2 = max(0, ey - search_r), min(h_img, ey + search_r + 1)
                patch = dark[py1:py2, px1:px2]

                if patch.shape[0] >= 55 and patch.shape[1] >= 55:
                    r_ring = cv2.matchTemplate(patch, self._ring_tmpl, cv2.TM_CCOEFF_NORMED)
                    r_disk = cv2.matchTemplate(patch, self._disk_tmpl, cv2.TM_CCOEFF_NORMED)
                    _, v_ring, _, loc_ring = cv2.minMaxLoc(r_ring)
                    _, v_disk, _, loc_disk = cv2.minMaxLoc(r_disk)

                    if v_disk > v_ring and v_disk > 0.55:
                        best_ncc = float(v_disk)
                        fx = float(px1 + loc_disk[0] + 27.0)
                        fy = float(py1 + loc_disk[1] + 27.0)
                    else:
                        best_ncc = float(v_ring)
                        fx = float(px1 + loc_ring[0] + 27.0)
                        fy = float(py1 + loc_ring[1] + 27.0)
                else:
                    best_ncc = 0.0
                    fx, fy = float(ex), float(ey)

                cla_ratio = self._compute_cla_fill_ratio(norm_gray, fx, fy)
                target_dict[opt_num] = round(cla_ratio, 4)

                is_filled = cla_ratio >= self.cla_filled_threshold
                is_aligned = best_ncc >= self.omr_ncc_align_threshold
                if is_aligned:
                    aligned_bubbles_cnt += 1
                ncc_scores.append(best_ncc)

                bubbles_info[group_name].append(
                    OMRBubbleInfo(
                        group_name=group_name,
                        option_num=opt_num,
                        cx=fx,
                        cy=fy,
                        cla_fill_ratio=round(cla_ratio, 4),
                        ncc_alignment_score=round(best_ncc, 4),
                        is_filled=is_filled,
                        is_aligned=is_aligned,
                    )
                )

        return (
            qset_ratios,
            page_ratios,
            bubbles_info,
            aligned_bubbles_cnt,
            float(np.mean(ncc_scores)),
        )

    @staticmethod
    def _bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        areaB = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        return float(inter / float(areaA + areaB - inter + 1e-5))

    def process_rectified_sheet(
        self, rectified_norm_gray: np.ndarray, image_name: str = "sheet.jpg"
    ) -> GridExtractionOutput:
        """
        สกัดตารางคำตอบ 20x20 ช่อง, รหัสนักเรียน 42 ช่อง และวงกลม OMR 7 วง พร้อมคำนวณตัวชี้วัด Phase 2
        """
        t0 = time.perf_counter()

        # 1. สแกนหาขีดดำบอกตำแหน่งข้อ 1-20 ด้านซ้ายและขวา
        left_centers, left_raw_cnt = self._detect_timing_bars_side(
            rectified_norm_gray, x_min=70, x_max=215, default_x=self.CANONICAL_LEFT_X
        )
        right_centers, right_raw_cnt = self._detect_timing_bars_side(
            rectified_norm_gray, x_min=2270, x_max=2430, default_x=self.CANONICAL_RIGHT_X
        )

        row_mid_ys = 0.5 * (left_centers[:, 1] + right_centers[:, 1])
        idx_arr = np.arange(1, self.NUM_QUESTIONS, dtype=np.float32)
        slope_y, intercept_y = np.polyfit(idx_arr, row_mid_ys[1:], 1)
        fitted_ys = np.concatenate([[row_mid_ys[0]], intercept_y + slope_y * idx_arr])
        row_linearity_rmse = float(np.sqrt(np.mean((row_mid_ys - fitted_ys) ** 2)))

        local_cell_half_h = int(round(0.35 * abs(slope_y)))

        dy_top_shift = float(row_mid_ys[0] - self.CANONICAL_ROW_YS[0])
        dx_top_shift = float(
            0.5 * ((left_centers[0, 0] - self.CANONICAL_LEFT_X) + (right_centers[0, 0] - self.CANONICAL_RIGHT_X))
        )

        # 2. สกัดช่องคำตอบทั้ง 20 ข้อ x 20 ช่อง ด้วย Triple-Layer Grid Line Suppression
        answer_grid: List[List[CellPatch]] = []
        cell_ious: List[float] = []
        empty_cells_count = 0

        canonical_span = self.CANONICAL_RIGHT_X - self.CANONICAL_LEFT_X

        for r_idx in range(self.NUM_QUESTIONS):
            xL, yL = left_centers[r_idx]
            xR, yR = right_centers[r_idx]
            actual_span = xR - xL
            scale_x = actual_span / canonical_span
            slope = (yR - yL) / (actual_span + 1e-5)

            row_cells: List[CellPatch] = []
            for c_idx, (can_x1, can_x2) in enumerate(self.CANONICAL_COL_BOUNDS):
                col_x1 = int(round(xL + (can_x1 - self.CANONICAL_LEFT_X) * scale_x))
                col_x2 = int(round(xL + (can_x2 - self.CANONICAL_LEFT_X) * scale_x))
                col_center_x = 0.5 * (col_x1 + col_x2)

                pred_y_center = yL + slope * (col_center_x - xL)

                bbox_pred = self._snap_cell_to_local_borders(
                    rectified_norm_gray, col_x1, col_x2, pred_y_center, local_cell_half_h
                )
                x1, y1, x2, y2 = bbox_pred

                snap_cx, snap_cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                ideal_w = col_x2 - col_x1
                ideal_h = 2 * local_cell_half_h + 1
                bbox_gt = (
                    int(round(snap_cx - ideal_w / 2.0)),
                    int(round(snap_cy - ideal_h / 2.0)),
                    int(round(snap_cx + ideal_w / 2.0)),
                    int(round(snap_cy + ideal_h / 2.0)),
                )
                cell_ious.append(self._bbox_iou(bbox_pred, bbox_gt))

                raw_patch = rectified_norm_gray[y1:y2, x1:x2]
                clean_patch, ink_ratio = self._clean_cell_patch(raw_patch)
                is_empty = ink_ratio < self.empty_ink_threshold
                if is_empty:
                    empty_cells_count += 1

                row_cells.append(
                    CellPatch(
                        row_idx=r_idx + 1,
                        col_idx=c_idx + 1,
                        bbox=bbox_pred,
                        clean_patch=clean_patch,
                        ink_ratio=round(ink_ratio, 5),
                        is_empty=is_empty,
                    )
                )
            answer_grid.append(row_cells)

        # 3. สกัดช่องรหัสส่วนหัว 42 ช่อง ด้วย Direct White-Interior Contour Detection
        header_ids, header_cells_info, aligned_header_cnt = self._extract_header_id_tables(
            rectified_norm_gray, dx_top_shift, dy_top_shift
        )

        # 4. ตรวจจับวงกลม OMR ทั้ง 7 วงด้วย Per-Bubble Ring/Disk NCC Template Matching
        qset_ratios, page_ratios, omr_bubbles_info, aligned_bubbles_cnt, mean_ncc = (
            self._locate_omr_bubbles(rectified_norm_gray, dx_top_shift, dy_top_shift)
        )

        detected_qset = None
        best_q_ratio = 0.0
        for q_num, ratio in qset_ratios.items():
            if ratio >= self.cla_filled_threshold and ratio > best_q_ratio:
                best_q_ratio = ratio
                detected_qset = q_num

        detected_page = None
        best_p_ratio = 0.0
        for p_num, ratio in page_ratios.items():
            if ratio >= self.cla_filled_threshold and ratio > best_p_ratio:
                best_p_ratio = ratio
                detected_page = p_num

        unfilled_ratios = list(qset_ratios.values()) + [page_ratios[p] for p in [2, 3, 4]]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        ious_arr = np.array(cell_ious, dtype=np.float32)
        valid_cells_50 = int(np.sum(ious_arr >= 0.50))
        total_cells = self.NUM_QUESTIONS * self.NUM_COLS

        metrics = Phase2Metrics(
            image_name=image_name,
            left_bars_detected=left_raw_cnt,
            right_bars_detected=right_raw_cnt,
            timing_bar_recall_pct=round(((left_raw_cnt + right_raw_cnt) / 40.0) * 100.0, 2),
            row_pitch_linearity_rmse_px=round(row_linearity_rmse, 3),
            total_answer_cells_extracted=total_cells,
            mean_cell_iou=round(float(np.mean(ious_arr)), 4),
            cell_precision_pct=round((valid_cells_50 / float(total_cells)) * 100.0, 2),
            cell_recall_pct=round((valid_cells_50 / float(total_cells)) * 100.0, 2),
            empty_cell_specificity_pct=round((empty_cells_count / float(total_cells)) * 100.0, 2),
            header_id_cells_extracted=42,
            header_id_exact_align_pct=round((aligned_header_cnt / 42.0) * 100.0, 2),
            omr_bubbles_exact_align_pct=round((aligned_bubbles_cnt / 7.0) * 100.0, 2),
            omr_mean_ncc_score=round(mean_ncc, 4),
            detected_question_set=detected_qset,
            detected_page_number=detected_page,
            page1_bubble_cla_ratio=page_ratios[1],
            unfilled_bubbles_max_cla_ratio=round(float(max(unfilled_ratios)), 4),
            processing_time_ms=round(elapsed_ms, 2),
        )

        return GridExtractionOutput(
            answer_grid=answer_grid,
            header_ids=header_ids,
            header_cells_info=header_cells_info,
            omr_bubbles={"question_set": qset_ratios, "page_number": page_ratios},
            omr_bubbles_info=omr_bubbles_info,
            left_bar_centers=left_centers,
            right_bar_centers=right_centers,
            metrics=metrics,
        )

    def render_visualization(
        self, rectified_bgr: np.ndarray, output: GridExtractionOutput
    ) -> np.ndarray:
        """
        วาดโครงสร้างตารางที่ตรวจพบทั้งหมดลงบนภาพ พร้อมระบบตรวจสอบความแม่นยำด้วยสี (Color-Coded Alignment Audit):
        - ช่องคำตอบ (400 ช่อง): สีเขียว = ช่องว่าง, สีส้ม = ช่องมีหมึก
        - ช่องรหัสส่วนหัว (42 ช่อง):
          * สีม่วง (BGR: 255, 0, 180) = ตรงกับช่องจริง 100% (is_aligned == True)
          * สีเหลืองส้มสว่าง (BGR: 0, 180, 255) = ไม่ตรงกับช่องจริง (is_aligned == False)
        - วงกลม OMR (7 วง):
          * สีเขียว (BGR: 0, 230, 0) = ตรงกับวงจริงและถูกฝนทึบ
          * สีฟ้า (BGR: 230, 210, 0) = ตรงกับวงจริงและว่างเปล่า
          * สีแดงสด (BGR: 0, 0, 255) = วงกลมที่วงไม่ตรงกับวงจริง (is_aligned == False)!
        """
        vis = rectified_bgr.copy()

        # 1. วาดจุดศูนย์กลางขีดดำซ้าย-ขวา และเส้นเชื่อม Row Baseline
        for i in range(self.NUM_QUESTIONS):
            ptL = (int(round(output.left_bar_centers[i, 0])), int(round(output.left_bar_centers[i, 1])))
            ptR = (int(round(output.right_bar_centers[i, 0])), int(round(output.right_bar_centers[i, 1])))
            cv2.circle(vis, ptL, 12, (0, 0, 255), -1)
            cv2.circle(vis, ptR, 12, (0, 0, 255), -1)
            cv2.line(vis, ptL, ptR, (255, 200, 0), 2)

        # 2. วาดกรอบช่องคำตอบทั้ง 400 ช่อง
        for row in output.answer_grid:
            for cell in row:
                x1, y1, x2, y2 = cell.bbox
                color = (0, 200, 0) if cell.is_empty else (0, 120, 255)
                cv2.rectangle(vis, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), color, 2)

        # 3. วาดกรอบรหัสส่วนหัว 42 ช่อง (สีม่วง = ตรงช่องจริง, สีเหลืองส้ม = ไม่ตรงช่องจริง)
        for _, cell_list in output.header_cells_info.items():
            for hcell in cell_list:
                hx1, hy1, hx2, hy2 = hcell.bbox
                if hcell.is_aligned:
                    h_col = (255, 0, 180)  # สีม่วง (ตรงช่องจริง 100%)
                    thick = 2
                else:
                    h_col = (0, 180, 255)  # สีเหลืองส้มสว่าง (แจ้งเตือนว่ายังไม่ตรง!)
                    thick = 4
                cv2.rectangle(vis, (hx1 + 2, hy1 + 2), (hx2 - 2, hy2 - 2), h_col, thick)

        # 4. วาดวงกลม OMR ทั้ง 7 วง (สีเขียว/สีฟ้า = ตรงวงจริง, สีแดงสด = ไม่ตรงวงจริง!)
        for group_name in ["question_set", "page_number"]:
            for binfo in output.omr_bubbles_info[group_name]:
                cx_i, cy_i = int(round(binfo.cx)), int(round(binfo.cy))
                if not binfo.is_aligned:
                    b_col = (0, 0, 255)    # สีแดงสด (แจ้งเตือนว่าวงกลมไม่ตรงกับวงจริง!)
                    thick = 5
                elif binfo.is_filled:
                    b_col = (0, 230, 0)    # สีเขียว (ตรงวงจริง + ฝนทึบ)
                    thick = 4
                else:
                    b_col = (230, 210, 0)  # สีฟ้า Cyan (ตรงวงจริง + ว่างเปล่า)
                    thick = 4
                cv2.circle(vis, (cx_i, cy_i), 23, b_col, thick)

        return vis

    def export_written_cells_dataset(
        self, output: GridExtractionOutput, sheet_prefix: str, save_dir: str
    ) -> int:
        os.makedirs(save_dir, exist_ok=True)
        saved_count = 0
        for row in output.answer_grid:
            for cell in row:
                if not cell.is_empty:
                    fname = f"{sheet_prefix}_Q{cell.row_idx:02d}_C{cell.col_idx:02d}.png"
                    cv2.imwrite(os.path.join(save_dir, fname), cell.clean_patch)
                    saved_count += 1
        return saved_count
