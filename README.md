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
│   ├── preprocessing/                   # Phase 1: Dewarping & Marker Detection
│   │   └── dewarp.py
│   ├── detection/                       # Phase 2: Timing Bars, Grid & OMR Extraction
│   │   └── grid_extractor.py
│   └── evaluation/                      # สคริปต์รันประเมินผลเชิงปริมาณ
│       ├── evaluate_phase1.py
│       └── evaluate_phase2.py
├── metrics/                             # รายงานตัวชี้วัดเชิงปริมาณและภาพตัวอย่าง
│   ├── README.md
│   ├── Phase1_Metrics_Report.md
│   ├── Phase2_Metrics_Report.md
│   ├── phase1_evaluation_metrics.json
│   ├── phase2_evaluation_metrics.json
│   └── visuals/                         # ภาพผลลัพธ์การจัดตำแหน่งและดัดระนาบ
├── data/                                # ข้อมูลรูปภาพกระดาษคำตอบ
│   ├── raw/                             # ภาพถ่ายต้นฉบับจากสมาร์ทโฟน
│   └── processed/                       # ภาพผลลัพธ์การดัดระนาบและผลวิเคราะห์
├── requirements.txt                     # รายการไลบรารีที่จำเป็น
└── README.md                            # คำอธิบายโปรเจกต์
```

---

## 🚀 การติดตั้งและการใช้งาน (Getting Started)

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. รันการประเมินผล Phase 1 (Preprocessing & Dewarping)
```bash
python src/evaluation/evaluate_phase1.py
```

### 3. รันการประเมินผล Phase 2 (Grid & Cell Extraction)
```bash
python src/evaluation/evaluate_phase2.py
```

---

## 👥 ผู้จัดทำ
- **รหัสนิสิต:** 67160328
- **มหาวิทยาลัย:** มหาวิทยาลัยบูรพา (Burapha University)
- **วิชา:** Computer Vision
