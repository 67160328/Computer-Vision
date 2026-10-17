# -*- coding: utf-8 -*-
"""
Phase 3: Cell State & Double Strikethrough (=) Detection Engine (Upgraded Multi-Angle + Wing Overhang)
สำหรับระบบตรวจข้อสอบอัตโนมัติ Rayong MARCO (Template 2)

จุดเด่นที่อัปเกรดเพื่อแก้ปัญหาตัว F, Z ถูกลบผิด และแก้ปัญหาขีดฆ่าเอียง (เช่น ตัว S) ไม่ยอมลบ:
1. Multi-Angle Tilted Line Kernels (-12° ถึง +12°):
   สกัดเส้นขีดฆ่าทั้งแบบแนวนอนตรงเป๊ะและแบบลากเฉียงขึ้น/เฉียงลง (-12°, -8°, -4°, 0°, +4°, +8°, +12°)
   ทำให้ไม่พลาดรอยขีดฆ่าที่ตวัดเฉียงอีกต่อไป
2. Interior Body Penetration (`rel_y_ink`):
   วัดตำแหน่งแนวตั้งของเส้นขีดเทียบกับความสูงจริงของหมึกตัวอักษร (`ink_y_min` ถึง `ink_y_max`)
   - ตัว `Z` หรือ `I` ปกติจะมีขีดอยู่ที่ขอบบนสุด (`rel_y < 0.14`) และขอบล่างสุด (`rel_y > 0.86`) เท่านั้น
   - รอยขีดฆ่า `=` จะต้องมีเส้นที่ผ่าเข้าสู่โซนลำตัวของตัวอักษร (`0.14 <= rel_y <= 0.85`)
3. Two-Sided Wing Overhang (`left_wing`, `right_wing`):
   แยกก้านแนวตั้ง/แนวทแยง (Stem) ออกจากเส้นแนวนอน แล้ววัดระยะที่เส้นขีดแลบออกไปทางซ้ายและทางขวาของก้านตัวอักษร
   - ตัว `F`, `E`, `B` ปกติจะมีก้านตั้งอยู่ซ้ายสุด ทำให้ไม่มีปลายเส้นแลบออกทางซ้าย (`left_wing ≈ 0`)
   - รอยขีดฆ่า `=` จะตัดทะลุก้านตัวอักษรออกไปทั้งฝั่งซ้ายและฝั่งขวา
4. 8D Geometric Feature Vector + Dual-Channel 64x64 ResNet:
   ผสานข้อมูลภาพ 2 ช่องสัญญาณเข้ากับเวกเตอร์เรขาคณิต 8 มิติเพื่อความแม่นยำสูงสุด
"""

import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


CELL_CLASSES = ("empty", "normal_char", "strikethrough")
CLASS_TO_IDX = {c: i for i, c in enumerate(CELL_CLASSES)}


@dataclass
class CellStatePrediction:
    """ผลลัพธ์การทำนายสถานะช่องคำตอบและรอยขีดฆ่า 2 เส้น (Phase 3)"""
    row_idx: int
    col_idx: int
    predicted_state: str               # 'empty', 'normal_char', 'strikethrough'
    is_empty: bool
    is_strikethrough: bool
    is_valid_char: bool                # True เมื่อเป็น normal_char (นำไปอ่านค่าตัวอักษรใน Phase 4)
    strikethrough_prob: float          # ความน่าจะเป็นของคลาส strikethrough (0.0 - 1.0)
    normal_char_prob: float            # ความน่าจะเป็นของคลาส normal_char (0.0 - 1.0)
    empty_prob: float                  # ความน่าจะเป็นของคลาส empty (0.0 - 1.0)
    confidence: float                  # ความมั่นใจของคลาสที่ชนะ
    num_horiz_bands: int               # จำนวนเส้นขีดแนวนอน/กึ่งแนวนอนที่ตรวจพบในช่อง
    pair_spacing_px: float             # ระยะห่างแนวตั้งระหว่างเส้นขีดคู่ขนาน (พิกเซล)
    horiz_span_ratio: float            # สัดส่วนความยาวเส้นขีดฆ่าเทียบกับความกว้างช่อง


_KERNEL_CACHE: Dict[int, List[np.ndarray]] = {}


def get_tilted_line_kernels(
    length: int,
    angles_deg: Tuple[int, ...] = (-12, -8, -4, 0, 4, 8, 12),
    thickness: int = 1,
) -> List[np.ndarray]:
    """สร้างและแคช Structuring Elements รูปเส้นตรงหลายมุมเอียง (-12° ถึง +12°)"""
    w = max(5, int(length))
    if w in _KERNEL_CACHE:
        return _KERNEL_CACHE[w]
    kernels = []
    for ang in angles_deg:
        rad = np.deg2rad(ang)
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
    _KERNEL_CACHE[w] = kernels
    return kernels


def extract_dual_channel_and_geom(clean_patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    แปลงภาพช่องคำตอบ (Clean Cell Patch ~70x80 px, หมึกดำบนพื้นขาว) ให้เป็น:
    1. dual_img: เทนเซอร์ภาพ 2 ช่องสัญญาณ ขนาด (2, 64, 64) ค่า [0.0, 1.0]
       - Channel 0: Inverted Ink Map (เส้นหมึกทั้งหมด)
       - Channel 1: Multi-Angle Horizontal Stroke Map (เฉพาะเส้นขีดแนวนอนและกึ่งแนวนอน -12°..+12°)
    2. geom_feat: เวกเตอร์คุณลักษณะเชิงเรขาคณิต 8 มิติ (float32):
       [ink_scaled, num_bands_norm, best_dy_norm, mean_bw_r, max_bw_r, body_penetration_score, left_wing_norm, right_wing_norm]
    """
    if clean_patch is None or clean_patch.size == 0:
        return np.zeros((2, 64, 64), dtype=np.float32), np.zeros(8, dtype=np.float32)

    if len(clean_patch.shape) == 3:
        gray = cv2.cvtColor(clean_patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = clean_patch

    ch, cw = gray.shape[:2]
    m = min(5, ch // 8, cw // 8)
    inner = gray[m : ch - m, m : cw - m]
    ih, iw = inner.shape[:2]

    # Adaptive Binary Inversion (ลด C ลงเหลือ 8 เพื่อให้จับเส้นขีดฆ่าดินสอที่ตวัดเร็วๆ ได้ไม่ขาดช่วง)
    inv = cv2.adaptiveThreshold(
        inner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )
    inv_clean = cv2.morphologyEx(
        inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )

    # วัดสัดส่วนหมึกบริเวณใจกลางช่อง
    cy1, cy2 = int(ih * 0.12), int(ih * 0.88)
    cx1, cx2 = int(iw * 0.12), int(iw * 0.88)
    core = inv_clean[cy1:cy2, cx1:cx2]
    ink_ratio = float(np.count_nonzero(core)) / float(max(1, core.size))

    iy, ix = np.where(inv_clean > 0)
    if len(ix) >= 12:
        ink_y_min, ink_y_max = float(np.percentile(iy, 3)), float(np.percentile(iy, 97))
        ink_x_min, ink_x_max = float(np.percentile(ix, 3)), float(np.percentile(ix, 97))
    else:
        ink_y_min, ink_y_max = 0.0, float(ih)
        ink_x_min, ink_x_max = 0.0, float(iw)
    ink_h = max(12.0, ink_y_max - ink_y_min)

    # สกัดเส้นแนวนอนและกึ่งแนวนอน (-12° ถึง +12°)
    kw = max(16, int(round(iw * 0.42)))
    kernels = get_tilted_line_kernels(kw, angles_deg=(-12, -8, -4, 0, 4, 8, 12), thickness=1)
    horiz_union = np.zeros_like(inv)
    for k in kernels:
        opened = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k)
        horiz_union = cv2.bitwise_or(horiz_union, opened)

    horiz_d = cv2.dilate(
        horiz_union, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2)), iterations=1
    )

    # แยกก้านแนวตั้ง/แนวทแยง (Character Stem) เพื่อวัดการแลบซ้าย-ขวา (Left/Right Wing Overhang)
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
                    "area": int(area),
                    "bw_ratio": float(bw) / float(iw),
                    "rel_y_ink": rel_y_ink,
                    "left_wing": left_wing,
                    "right_wing": right_wing,
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

    best_dy_norm = 0.0
    mean_bw_r = 0.0
    max_bw_r = max([b["bw_ratio"] for b in merged], default=0.0)
    body_pen_score = 0.0
    left_wing_norm = 0.0
    right_wing_norm = 0.0

    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            b1, b2 = merged[i], merged[j]
            dy = abs(b2["cy"] - b1["cy"])
            if 6.5 <= dy <= 35.0:
                in_body1 = 0.14 <= b1["rel_y_ink"] <= 0.85
                in_body2 = 0.14 <= b2["rel_y_ink"] <= 0.85
                if not (in_body1 or in_body2):
                    continue
                cand_mean_bw = 0.5 * (b1["bw_ratio"] + b2["bw_ratio"])
                cand_l_wing = max(b1["left_wing"], b2["left_wing"]) / float(max(1, iw))
                cand_r_wing = max(b1["right_wing"], b2["right_wing"]) / float(max(1, iw))
                cand_body = 1.0 if (in_body1 and in_body2) else 0.5
                if cand_mean_bw + 0.5 * cand_body > mean_bw_r + 0.5 * body_pen_score:
                    mean_bw_r = cand_mean_bw
                    best_dy_norm = dy / float(max(1, ih))
                    body_pen_score = cand_body
                    left_wing_norm = min(1.0, cand_l_wing * 2.5)
                    right_wing_norm = min(1.0, cand_r_wing * 2.5)

    # ปรับขนาดภาพทั้ง 2 ช่องสัญญาณสู่ 64x64
    ch0 = cv2.resize(inv_clean, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    ch1 = cv2.resize(horiz_d, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    dual_img = np.stack([ch0, ch1], axis=0)

    geom_feat = np.array(
        [
            min(1.0, ink_ratio * 5.0),
            min(1.0, len(merged) / 4.0),
            best_dy_norm,
            mean_bw_r,
            max_bw_r,
            body_pen_score,
            left_wing_norm,
            right_wing_norm,
        ],
        dtype=np.float32,
    )
    return dual_img, geom_feat


class ConvResBlock(nn.Module):
    """Residual Block สำหรับโมเดลตรวจจับรอยขีดฆ่า 2 เส้น"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.gelu(out)


class StrikethroughHybridNet(nn.Module):
    """
    Hybrid Dual-Channel CNN + 8D Geometric Feature Fusion Network
    สำหรับจำแนก 3 คลาส: empty (0), normal_char (1), strikethrough (2)
    """
    def __init__(self, num_classes: int = 3, geom_dim: int = 8, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.stage1 = ConvResBlock(24, 48, stride=2)   # 32x32
        self.stage2 = ConvResBlock(48, 96, stride=2)   # 16x16
        self.stage3 = ConvResBlock(96, 128, stride=2)  # 8x8
        self.stage4 = ConvResBlock(128, 160, stride=2) # 4x4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(160 + 32, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, dual_img: torch.Tensor, geom_feat: torch.Tensor) -> torch.Tensor:
        x = self.stem(dual_img)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        vis_emb = self.pool(x).flatten(1)
        geom_emb = self.geom_mlp(geom_feat)
        fused = torch.cat([vis_emb, geom_emb], dim=1)
        return self.classifier(fused)


class StrikethroughDetectorEngine:
    """
    เอนจินสำหรับโหลดโมเดล StrikethroughHybridNet และทำนายสถานะช่องคำตอบทั้ง 400 ช่องแบบ Batch
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = StrikethroughHybridNet(num_classes=3, geom_dim=8, dropout=0.0).to(self.device)
        self.has_trained_weights = False
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            self.has_trained_weights = True
        self.model.eval()

    @torch.no_grad()
    def predict_cells(
        self,
        cells_with_coords: List[Tuple[int, int, np.ndarray]],
        hcr_predictions: Optional[Dict[Tuple[int, int], object]] = None,
    ) -> List[CellStatePrediction]:
        if not cells_with_coords:
            return []

        dual_list = []
        geom_list = []
        for _, _, patch in cells_with_coords:
            d_img, g_feat = extract_dual_channel_and_geom(patch)
            dual_list.append(d_img)
            geom_list.append(g_feat)

        dual_tensor = torch.from_numpy(np.stack(dual_list, axis=0)).to(self.device)
        geom_tensor = torch.from_numpy(np.stack(geom_list, axis=0)).to(self.device)

        logits = self.model(dual_tensor, geom_tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        results: List[CellStatePrediction] = []
        for idx, (r_idx, c_idx, patch) in enumerate(cells_with_coords):
            p_empty, p_normal, p_strike = float(probs[idx, 0]), float(probs[idx, 1]), float(probs[idx, 2])
            g = geom_list[idx]
            ink_scaled = float(g[0])
            n_bands = int(round(float(g[1]) * 4.0))
            dy_px = round(float(g[2]) * max(1, patch.shape[0] - 10), 1)
            mean_bw_r = round(float(g[3]), 4)
            body_pen = float(g[5])
            l_wing = float(g[6])
            r_wing = float(g[7])

            hcr_pred = hcr_predictions.get((r_idx, c_idx)) if hcr_predictions else None
            hcr_conf = float(getattr(hcr_pred, "confidence", 0.5)) if hcr_pred is not None else 0.5
            hcr_char = str(getattr(hcr_pred, "predicted_char", "")) if hcr_pred is not None else ""

            # 1. กฎความปลอดภัยสำหรับช่องว่างสนิท (Zero-Ink Safeguard)
            if ink_scaled < 0.035:
                pred_cls = "empty"
                p_empty = max(p_empty, 0.99)
                p_normal = min(p_normal, 0.01)
                p_strike = 0.0
            # 2. HCR-Guided Protection (เมื่อ Phase 4 รันก่อนแล้วมั่นใจสูงว่าเป็นตัวอักษรปกติ เช่น F, Z, E, H, I, T, B)
            elif body_pen == 0.0 or (l_wing < 0.12 and mean_bw_r < 0.74):
                pred_cls = "normal_char"
                p_normal = max(p_normal, p_strike + 0.20)
            elif hcr_pred is not None and hcr_conf >= 0.82 and mean_bw_r < 0.76 and min(l_wing, r_wing) < 0.35:
                # หากโมเดล Phase 4 อ่านตัวอักษรได้ด้วยความมั่นใจ >= 82% และไม่มีเส้นขีดฆ่าที่แลบทะลุทั้งซ้าย-ขวาอย่างชัดเจน
                pred_cls = "normal_char"
                p_normal = max(p_normal, p_strike + 0.20)
            elif hcr_pred is not None and hcr_conf < 0.45 and n_bands >= 2 and body_pen >= 0.5 and mean_bw_r >= 0.52:
                # หากโมเดล Phase 4 ฟ้องว่าความมั่นใจตัวอักษรต่ำ (< 45%) และมีเส้นขนาน 2 เส้นผ่ากลางลำตัว -> ฟันธงว่าเป็นขีดฆ่า `=` ทันที!
                pred_cls = "strikethrough"
                p_strike = max(p_strike, p_normal + 0.25)
            else:
                cls_idx = int(np.argmax(probs[idx]))
                pred_cls = CELL_CLASSES[cls_idx]

            conf = max(p_empty, p_normal, p_strike)
            results.append(
                CellStatePrediction(
                    row_idx=r_idx,
                    col_idx=c_idx,
                    predicted_state=pred_cls,
                    is_empty=(pred_cls == "empty"),
                    is_strikethrough=(pred_cls == "strikethrough"),
                    is_valid_char=(pred_cls == "normal_char"),
                    strikethrough_prob=round(p_strike, 4),
                    normal_char_prob=round(p_normal, 4),
                    empty_prob=round(p_empty, 4),
                    confidence=round(conf, 4),
                    num_horiz_bands=n_bands,
                    pair_spacing_px=dy_px,
                    horiz_span_ratio=mean_bw_r,
                )
            )

        return results
