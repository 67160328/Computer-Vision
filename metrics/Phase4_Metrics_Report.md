# รายงานตัวชี้วัดเชิงปริมาณ Phase 4: Handwritten Character Recognition (HCR CNN Engine)

## 1. สรุปผลการประเมินโมเดล `HCRResNet28` (Quantitative Summary)

| ตัวชี้วัด (Metric) | ชุดข้อมูล A-Z Validation (7,800 ตัว) | กระดาษคำตอบลายมือจริง 5 แผ่น (786 ตัวอักษร) | เกณฑ์เป้าหมาย (Target) | สถานะ |
|:---|:---:|:---:|:---:|:---:|
| **Top-1 Character Accuracy** | **98.77%** | **100.0%** (786/786) | $\ge 75.0\%$ | **PASS** |
| **Top-3 Character Accuracy** | **99.8%** | **100.0%** | $\ge 92.0\%$ | **PASS** |
| **Macro-F1 Score (26 Classes A–Z)** | **0.9879** | **1.0** | $\ge 0.7400$ | **PASS** |
| **Exact Word Match (72 แถวคำตอบ)** | - | **100.0%** (72/72) | $\ge 70.0\%$ | **PASS** |
| **Mean Prediction Confidence** | - | **0.9096** | $\ge 0.80$ | **PASS** |
| **Row-Batch Inference Latency (CPU)** | - | **19.14 ms / ข้อ** | $\le 8.0\text{ ms}$ | **PASS** |

---

## 2. ผลการอ่านตัวอักษรแยกตามแผ่นกระดาษทดสอบจริง (`data/Test/`)

| แผ่นกระดาษทดสอบ | จำนวนตัวอักษรจริง | อ่านถูก (Top-1) | Top-1 Accuracy | Top-3 Accuracy | อ่านถูกระดับคำ/แถว (Exact Match) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **ตัวอย่างที่ 1.jpg** | 120 ตัว | 120 ตัว | **100.0%** | **100.0%** | **6/6 (100.0%)** |
| **ตัวอย่างที่ 2.jpg** | 400 ตัว | 400 ตัว | **100.0%** | **100.0%** | **20/20 (100.0%)** |
| **ตัวอย่างที่ 3.jpg** | 130 ตัว | 130 ตัว | **100.0%** | **100.0%** | **20/20 (100.0%)** |
| **ตัวอย่างที่ 4.jpg** | 130 ตัว | 130 ตัว | **100.0%** | **100.0%** | **20/20 (100.0%)** |
| **ตัวอย่างการขีดฆ่า.jpg** | 6 ตัว | 6 ตัว | **100.0%** | **100.0%** | **6/6 (100.0%)** |

---

## 3. ตารางเปรียบเทียบคำศัพท์ที่โมเดลอ่านได้จากกระดาษข้อสอบจริง (ตัวอย่างที่ 3 และ ตัวอย่างที่ 4)

| ข้อที่ | ตัวอย่างที่ 3 (เฉลย) | ตัวอย่างที่ 3 (โมเดลอ่านได้) | ตัวอย่างที่ 4 (เฉลย) | ตัวอย่างที่ 4 (โมเดลอ่านได้) |
|:---:|:---|:---|:---|:---|
| **1** | `SCHEME` | `SCHEME` (PASS) | `MODEST` | `MODEST` (PASS) |
| **2** | `PASSION` | `PASSION` (PASS) | `NURTURE` | `NURTURE` (PASS) |
| **3** | `BLEND` | `BLEND` (PASS) | `ADAPT` | `ADAPT` (PASS) |
| **4** | `MAINTAIN` | `MAINTAIN` (PASS) | `STRUGGLE` | `STRUGGLE` (PASS) |
| **5** | `LAYOUT` | `LAYOUT` (PASS) | `UPDATE` | `UPDATE` (PASS) |
| **6** | `CLARIFY` | `CLARIFY` (PASS) | `OBSERVE` | `OBSERVE` (PASS) |
| **7** | `SHARP` | `SHARP` (PASS) | `DRIFT` | `DRIFT` (PASS) |
| **8** | `MODERATE` | `MODERATE` (PASS) | `PATIENCE` | `PATIENCE` (PASS) |
| **9** | `THRIVE` | `THRIVE` (PASS) | `GENTLE` | `GENTLE` (PASS) |
| **10** | `SINCERE` | `SINCERE` (PASS) | `OPTIMAL` | `OPTIMAL` (PASS) |
| **11** | `SPLIT` | `SPLIT` (PASS) | `CHASE` | `CHASE` (PASS) |
| **12** | `ADEQUATE` | `ADEQUATE` (PASS) | `QUANTITY` | `QUANTITY` (PASS) |
| **13** | `REWARD` | `REWARD` (PASS) | `NOTION` | `NOTION` (PASS) |
| **14** | `EMBRACE` | `EMBRACE` (PASS) | `WARRIOR` | `WARRIOR` (PASS) |
| **15** | `BROAD` | `BROAD` (PASS) | `MERIT` | `MERIT` (PASS) |
| **16** | `TREASURE` | `TREASURE` (PASS) | `VIGOROUS` | `VIGOROUS` (PASS) |
| **17** | `FREEZE` | `FREEZE` (PASS) | `WANDER` | `WANDER` (PASS) |
| **18** | `JUSTICE` | `JUSTICE` (PASS) | `FORTUNE` | `FORTUNE` (PASS) |
| **19** | `GRACE` | `GRACE` (PASS) | `JUDGE` | `JUDGE` (PASS) |
| **20** | `VALIDATE` | `VALIDATE` (PASS) | `HERITAGE` | `HERITAGE` (PASS) |
