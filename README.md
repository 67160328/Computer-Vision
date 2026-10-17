# Computer Vision: ระบบตรวจกระดาษคำตอบอัตโนมัติ Rayong MARCO (Template 2)

ระบบตรวจกระดาษคำตอบข้อสอบอัตโนมัติรองรับทั้งการตรวจ **OMR (ฝนตัวเลือก)** และ **ลายมือเขียน (Handwritten Character Recognition - HCR)** บนกระดาษคำตอบรูปแบบ Rayong MARCO (Template 2) ผ่านกระบวนการประมวลผลทาง Computer Vision 5 ขั้นตอน (5-Phase Pipeline) ที่ออกแบบตามมาตรฐานงานวิจัยระดับสากล

---

## 📌 สถาปัตยกรรมระบบ (HCR-First Guided 5-Phase Pipeline)

ในการทำงานจริง ระบบได้ปรับปรุงลำดับการทำงานเชิงสถาปัตยกรรมเป็น **HCR-First Confidence Guided Pipeline** โดยนำผลการทำนายและความเชื่อมั่นของโมเดล HCR (Phase 4) มาร่วมเป็นฟีเจอร์นำทางให้ตัวตรวจจับรอยขีดฆ่า (Phase 3) เพื่อป้องกันการจำแนกตัวอักษรที่มีเส้นขนานในตัว เช่น `F`, `Z`, `E` ผิดพลาดเป็นรอยขีดฆ่าได้อย่างสมบูรณ์ (`FPR = 0.0%`):

```
[ ภาพถ่ายกระดาษคำตอบจากมือถือ (สภาพแสง/มุมเอียงต่างกัน) ]
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Phase 1: Preprocessing & Fiducial Dewarping             │
 │ - ตรวจจับ ArUco/Fiducial Markers 3 มุม                  │
 │ - คำนวณ Homography Matrix และปรับมุมระนาบ 4 มุมมาตรฐาน     │
 │ - ตรวจสอบความถูกต้องของทิศทางกระดาษ (Auto-Orientation)       │
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Phase 2: Anchor Bar Tracking & Cell Localization        │
 │ - ตรวจจับแถบ Timing Bars ซ้าย-ขวา 20 แถว                 │
 │ - Triple-Layer Grid Line Suppression (กรอบคำตอบ 400 ช่อง) │
 │ - Direct Contour Snapping สำหรับ Header ID 42 ช่อง      │
 │ - NCC Template Matching สำหรับวงกลม OMR 7 วง             │
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Phase 4: Handwritten Character Recognition (HCR Pre-Run)│
 │ - สกัดภาพอักขระเดี่ยว ปรับขนาดและจัดกึ่งกลาง 32x32           │
 │ - โมเดล ResNet-28 ทำนายความน่าจะเป็น A-Z (Softmax Conf)  │
 └─────────────────────────┬───────────────────────────────┘
                           │ (ส่ง HCR Confidences & Predictions)
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Phase 3: HCR-Guided Double Strikethrough Filter         │
 │ - Multi-Angle Tilted Kernels (-12° ถึง +12°) ดักจับเส้นเอียง│
 │ - ผสาน HCR Guidance + Morphological + Body Penetration  │
 │ - กรองช่องขีดฆ่า (Strikethrough `=`) และป้องกัน `F`,`Z` หลอก │
 └─────────────────────────┬───────────────────────────────┘
                           │ (กรองช่องว่าง & ช่องขีดฆ่าออก)
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Phase 5: Word Assembly, Lexicon Refinement & Scoring    │
 │ - ประกอบตัวอักษรที่เหลือเป็นคำศัพท์ในแต่ละแถว             │
 │ - Top-3 Softmax Weighted Beam Search ปรับคำศัพท์ตามเฉลย   │
 │ - ตรวจและบันทึกคะแนน พร้อมส่งออก JSON, CSV และภาพรายงาน   │
 └─────────────────────────────────────────────────────────┘
```

---

## 📊 ตัวชี้วัดเชิงปริมาณที่ผ่านการทดสอบ (Current Benchmark Metrics)

สามารถตรวจสอบรายงานตัวชี้วัดฉบับเต็มได้ที่ [`metrics/README.md`](./metrics/README.md)

### Phase 1: Preprocessing & Dewarping
- **3-Marker Detection Recall:** **`100.0%`** (51/51 แผ่น)
- **Auto-Orientation Accuracy:** **`100.0%`** (80/80 เคสทดสอบ)
- **Reprojection RMSE:** **`0.114 ± 0.048 px`**
- **Paper Boundary IoU:** **`0.9873`**

### Phase 2: Grid Localization & Alignment
- **Timing Bar Detection Recall:** **`100.0%`** (2,040/2,040 ขีด)
- **Answer Cell Recall:** **`100.0%`** (20,400/20,400 ช่อง)
- **Blank Cell Specificity (กรอบสีเขียว):** **`100.0%`** (20,400/20,400 ช่อง - ไม่มี False Positive รอยตาราง)
- **Header ID Exact Alignment (กรอบสีม่วง):** **`100.0%`** (2,142/2,142 ช่อง)
- **OMR Bubble Exact Alignment:** **`100.0%`** (357/357 วง | Mean NCC = `0.4653`)
- **OMR Bubble CLA Accuracy:** **`100.0%`** (`100.0%`)

### Phase 3: Double Strikethrough (`=`) Detection & Filtering
- **Multi-Angle Slanted Kernels ($-12^\circ$ to $+12^\circ$):** รองรับรอยขีดฆ่าเอียง/เบี้ยวได้สมบูรณ์
- **Strikethrough Precision / Recall:** **`100.0% / 100.0%` (F1 = 1.000)** บนชุดทดสอบขีดฆ่าจริง
- **Normal Character Invariance:** ตัวอักษรปกติที่มีเส้นขนานเช่น `F` และ `Z` ไม่ถูกตรวจจับผิดพลาดเลยแม้แต่ช่องเดียว (**`FPR = 0.0%`**)

### Phase 4: Handwritten Character Recognition (A–Z HCR CNN Engine)
- **A–Z Validation Accuracy (7,800 ตัวอย่าง):** **`Top-1 = 98.77%`**, **`Top-3 = 99.80%`**, **`Macro-F1 = 0.9879`**
- **Real Exam Sheets Character Accuracy (786 ตัวอักษร):** **`Top-1 = 100.0% (786/786)`**, **`Top-3 = 100.0%`**, **`Macro-F1 = 1.0000`**
- **Real Exam Word / Row Match (72 แถวคำตอบจริง):** **`100.0% (72/72 แถว)`**
- **Row-Batch Latency (CPU):** **`19.14 ms / แถว`** ($\approx 0.96\text{ ms/ช่อง}$)

### Phase 4B: Purple Box Header Digit Recognition (0–9 Digit CNN Engine — 42 ช่อง)
- **Dataset (`data/Number Handwriter` 21,555 ภาพ, Split 80:10:10):**
  - **Validation Accuracy (2,156 ภาพ):** **`Top-1 = 99.77%`**, **`Top-3 = 100.00%`**
  - **Unseen Test Accuracy (2,154 ภาพ):** **`Top-1 = 99.58%`**, **`Top-3 = 99.95%`**, **`Macro-F1 = 0.9958`**
- **Blank Purple Box Specificity (798 ช่องว่างบนกระดาษจริง 19 แผ่น):** **`100.0% (798/798 ช่อง)`**
- **Purple Box End-to-End Digit Accuracy (รหัสประจำตัว / รหัสวิชา / รหัสสถานศึกษา):** **`100.0% (96/96 ตัวเลข, 9/9 แถวรหัส)`**

### Phase 5: Word Assembly, Lexicon Refinement & Automated Scoring
- **Exact Match Word Accuracy (EMA):** **`99.21% (250 / 252 ข้อ)`** บน 19 กระดาษคำตอบชุดทดสอบ
- **Character Error Rate (CER):** **`0.10%`** (จาก 2,921 ตัวอักษรที่มีเฉลย)
- **Score Pearson Correlation ($r$):** **`0.9994`** สอดคล้องกับคะแนนจริงเกือบสมบูรณ์แบบ
- **End-to-End Latency:** **`~2,478.6 ms / แผ่น`** (ประมวลผลตั้งแต่ภาพถ่ายดิบจนออกผลคะแนนบน CPU)

---

## 📁 โครงสร้างโปรเจกต์ (Project Directory Structure)

```text
Lab9/
├── plan/                                # แผนการดำเนินงานและข้อกำหนดทางเทคนิคทั้ง 5 Phase
│   ├── README.md
│   ├── Phase1_Preprocessing_and_Dewarping.md
│   ├── Phase2_Grid_and_Cell_Localization.md
│   ├── Phase3_Double_Strikethrough_Detection.md
│   ├── Phase4_Character_Recognition_HCR.md
│   └── Phase5_Word_Assembly_and_Grading.md
├── src/                                 # ซอร์สโค้ดการประมวลผล
│   ├── preprocessing/                   # Phase 1: Dewarping & Dataset Split Tools
│   │   ├── dewarp.py
│   │   ├── prepare_new_dataset.py
│   │   └── upgrade_dataset_labels.py
│   ├── detection/                       # Phase 2 & 3: Grid, OMR, Strikethrough
│   │   ├── grid_extractor.py
│   │   └── strikethrough_detector.py
│   ├── recognition/                     # Phase 4 & 4B: A-Z HCR & 0-9 Header Digit CNN
│   │   ├── char_classifier.py
│   │   ├── digit_classifier.py
│   │   ├── train_hcr.py
│   │   └── train_digit_hcr.py
│   ├── grading/                         # Phase 5: Word Assembly & Automated Scoring
│   │   └── grader.py
│   └── evaluation/                      # สคริปต์รันประเมินผลเชิงปริมาณ
│       ├── evaluate_phase1.py
│       ├── evaluate_phase2.py
│       ├── evaluate_phase3.py
│       ├── evaluate_phase4.py
│       ├── evaluate_digit_header.py
│       ├── evaluate_phase5.py
│       └── test_full_pipeline_on_test_split.py
├── models/                              # โมเดล Deep Learning ที่เทรนเสร็จสมบูรณ์
│   ├── hcr_resnet28_az.pt               # Lightweight ResNet-28 สำหรับจำแนกอักษร A-Z (26 คลาส)
│   ├── hcr_resnet28_digits.pt           # Lightweight ResNet-28 สำหรับจำแนกตัวเลข 0-9 (10 คลาส)
│   └── strikethrough_cnn.pt             # Hybrid CNN สำหรับจำแนกช่องว่าง/ขีดฆ่า/อักษรปกติ
├── metrics/                             # รายงานตัวชี้วัดเชิงปริมาณและภาพตัวอย่าง
│   ├── README.md
│   ├── Phase1_Metrics_Report.md
│   ├── Phase2_Metrics_Report.md
│   ├── Phase3_Metrics_Report.md
│   ├── Phase4_Metrics_Report.md
│   ├── Phase4B_Digit_Metrics_Report.md
│   ├── Phase5_Metrics_Report.md
│   ├── digit_recognition_metrics.json
│   ├── phase5_evaluation_metrics.json
│   ├── phase5_grading_results.csv
│   └── visuals/                         # ภาพผลลัพธ์การตรวจและให้คะแนนรายแผ่น
├── data/                                # ข้อมูลรูปภาพกระดาษคำตอบ
│   ├── raw/                             # ภาพถ่ายต้นฉบับจากสมาร์ทโฟน
│   ├── Test/                            # ภาพกระดาษคำตอบชุด Benchmark ดั้งเดิม
│   └── dataset_splits/                  # ชุดข้อมูล 80:10:10 (Manifest & Summaries)
├── requirements.txt                     # รายการไลบรารีที่จำเป็น
└── README.md                            # คำอธิบายโปรเจกต์
```

---

## 🚀 การติดตั้งและการใช้งาน (Getting Started)

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. รันการประเมินผลแยกตามแต่ละ Phase
```bash
# Phase 1: Preprocessing & Dewarping
python src/evaluation/evaluate_phase1.py

# Phase 2: Grid & Cell Extraction
python src/evaluation/evaluate_phase2.py

# Phase 3: Double Strikethrough Detection
python src/evaluation/evaluate_phase3.py

# Phase 4: Handwritten Character Recognition (A-Z)
python src/evaluation/evaluate_phase4.py

# Phase 4B: Purple Box Header Digit Recognition (0-9)
python src/evaluation/evaluate_digit_header.py

# Phase 5: End-to-End Automated Grading & Scoring
python src/evaluation/evaluate_phase5.py
```

---

## 👥 ผู้จัดทำ
- **รหัสนิสิต:** 67160328
- **มหาวิทยาลัย:** มหาวิทยาลัยบูรพา (Burapha University)
- **วิชา:** Computer Vision
