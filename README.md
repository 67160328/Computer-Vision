# Computer Vision: ระบบตรวจกระดาษคำตอบอัตโนมัติ Rayong MARCO (Template 2)

ระบบตรวจกระดาษคำตอบข้อสอบอัตโนมัติรองรับทั้งการตรวจ **OMR (ฝนตัวเลือก)** และ **ลายมือเขียน (Handwritten Character Recognition - HCR)** บนกระดาษคำตอบรูปแบบ Rayong MARCO (Template 2) ผ่านกระบวนการประมวลผลทาง Computer Vision 5 ขั้นตอน (5-Phase Pipeline) ที่ออกแบบตามมาตรฐานงานวิจัยระดับสากล

---

## 📌 สถาปัตยกรรมระบบ 5 ขั้นตอน (5-Phase Pipeline)

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
 │ Phase 3: Double Strikethrough Detection                 │
 │ - Hybrid Morphological Profile กรองรอยขีดฆ่า 2 เส้นขนาน  │
 │ - แยกแยะข้อที่ถูกยกเลิก (Strikethrough `=`) ออกจากการตอบจริง│
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Phase 4: Handwritten Character Recognition (HCR)        │
 │ - ปรับขนาดภาพอักขระเดี่ยว (Isolated Character 32x32)     │
 │ - สถาปัตยกรรม CNN จำแนกตัวอักษร A-Z                     │
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Phase 5: Word Assembly, Scoring & Export                │
 │ - รวมคำตอบรายข้อ เปรียบเทียบกับ Answer Key              │
 │ - ส่งออกผลการตรวจเป็น JSON, CSV และ Excel รายบุคคล        │
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

### Phase 4: Handwritten Character Recognition (HCR CNN Engine)
- **A–Z Validation Accuracy (7,800 ตัวอย่าง):** **`Top-1 = 98.77%`**, **`Top-3 = 99.80%`**, **`Macro-F1 = 0.9879`**
- **Real Exam Sheets Character Accuracy (786 ตัวอักษร):** **`Top-1 = 100.0% (786/786)`**, **`Top-3 = 100.0%`**, **`Macro-F1 = 1.0000`**
- **Real Exam Word / Row Match (72 แถวคำตอบจริง):** **`100.0% (72/72 แถว)`**
- **Row-Batch Latency (CPU):** **`19.14 ms / แถว`** ($\approx 0.96\text{ ms/ช่อง}$)

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
│   ├── recognition/                     # Phase 4: HCR Normalizer, ResNet28 & Trainer
│   │   ├── char_classifier.py
│   │   └── train_hcr.py
│   ├── grading/                         # Phase 5: Word Assembly & Automated Scoring
│   │   └── grader.py
│   └── evaluation/                      # สคริปต์รันประเมินผลเชิงปริมาณ
│       ├── evaluate_phase1.py
│       ├── evaluate_phase2.py
│       ├── evaluate_phase3.py
│       ├── evaluate_phase4.py
│       ├── evaluate_phase5.py
│       └── test_full_pipeline_on_test_split.py
├── models/                              # โมเดล Deep Learning ที่เทรนเสร็จสมบูรณ์
│   ├── hcr_resnet28_az.pt               # Lightweight ResNet-28 สำหรับจำแนกอักษร A-Z
│   └── strikethrough_cnn.pt             # Hybrid CNN สำหรับจำแนกช่องว่าง/ขีดฆ่า/อักษรปกติ
├── metrics/                             # รายงานตัวชี้วัดเชิงปริมาณและภาพตัวอย่าง
│   ├── README.md
│   ├── Phase1_Metrics_Report.md
│   ├── Phase2_Metrics_Report.md
│   ├── Phase3_Metrics_Report.md
│   ├── Phase4_Metrics_Report.md
│   ├── Phase5_Metrics_Report.md
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

# Phase 4: Handwritten Character Recognition
python src/evaluation/evaluate_phase4.py

# Phase 5: End-to-End Automated Grading & Scoring
python src/evaluation/evaluate_phase5.py
```

---

## 👥 ผู้จัดทำ
- **รหัสนิสิต:** 67160328
- **มหาวิทยาลัย:** มหาวิทยาลัยบูรพา (Burapha University)
- **วิชา:** Computer Vision
