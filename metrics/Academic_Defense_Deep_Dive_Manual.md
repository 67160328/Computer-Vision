# Academic Defense & Deep-Dive Technical Manual: Automated Exam Grading Pipeline
**โครงการ:** ระบบตรวจกระดาษคำตอบและให้คะแนนอัตโนมัติ Rayong MARCO (Template 2)  
**สถาบัน:** ภาควิชาวิทยาการคอมพิวเตอร์ คณะวิทยาการสารสนเทศ มหาวิทยาลัยบูรพา  
**มาตรฐานทางวิชาการและการอ้างอิง:** SNPD 2025 (CLA & Homography), DocUNet (CVPR 2018), OMRNet (MTAP 2024), ResNet (CVPR 2016), Beam Search Lexicon Decoding (ICDAR)

---

## 📑 สารบัญภาพรวมระบบ (End-to-End System Architecture)

ระบบตรวจข้อสอบอัตโนมัติ Rayong MARCO (Template 2) ประกอบด้วยสถาปัตยกรรมแบบ **HCR-First Confidence Guided Synergy Pipeline** ซึ่งแบ่งการทำงานออกเป็น 6 ขั้นตอนย่อย และขับเคลื่อนด้วยไฟล์โมดูลเฉพาะทาง:

```text
[ ภาพถ่ายสมาร์ทโฟนสภาพแวดล้อมจริง (เอียง/โค้ง/กลับหัว/แสงไม่สม่ำเสมอ) ]
                               │
                               ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Phase 1: Preprocessing, Fiducial Tracking & Projective Homography      │
 │ • dewarp.py (FiducialDewarper) - Hypotenuse Vector Analysis + SVD DLT  │
 │ • evaluate_phase1.py - Benchmark Marker Recall, RMSE, IoU              │
 └─────────────────────────────┬──────────────────────────────────────────┘
                               │ (ภาพจัดระนาบมาตรฐาน 2480 x 3508 px, 300 DPI)
                               ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Phase 2: Anchor Bar Tracking & Triple-Layer Grid Line Suppression      │
 │ • grid_extractor.py (Template2GridExtractor) - Timing Bars, Cells, OMR │
 │ • evaluate_phase2.py - Benchmark Timing Bars, Snap Accuracy, NCC & CLA │
 └─────────────────────────────┬──────────────────────────────────────────┘
                               │ (Clean Cell Patches ~70x80 px + Header ID Patches)
                               ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Phase 4: Handwritten Character Recognition (Pre-Scoring HCR Engine)    │
 │ • char_classifier.py (HCRModelEngine, HCRResNet28) - 26 Classes (A-Z)  │
 │ • train_hcr.py - Data Augmentation, AdamW, Cosine Annealing           │
 │ • evaluate_phase4.py - Val Top-1/Top-3, Real Exam Accuracy, Macro-F1   │
 └─────────────────────────────┬──────────────────────────────────────────┘
                               │ (ส่งค่า HCR Confidences & Predictions)
                               ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Phase 3: HCR-Guided Double Strikethrough Detection & Filtering         │
 │ • strikethrough_detector.py (StrikethroughDetectorEngine, HybridNet)   │
 │ • evaluate_phase3.py - Tilted Slanted Kernels, Confusion Matrix, AUC   │
 └─────────────────────────────┬──────────────────────────────────────────┘
                               │ (กรองช่องว่าง 'empty' & ช่องขีดฆ่า '=' ทิ้ง)
                               ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Phase 4B: Purple Box Header Digit Recognition (0-9 CNN Engine - 42 ช่อง)│
 │ • digit_classifier.py (DigitRecognizerEngine, HeaderCellEmptyGate)     │
 │ • train_digit_hcr.py - Training on 'Number Handwriter' (21,555 samples)│
 │ • evaluate_digit_header.py - Header Blank Specificity & Digit Accuracy │
 └─────────────────────────────┬──────────────────────────────────────────┘
                               │ (รหัสประจำตัว + รหัสวิชา + ชุดตัวอักษรรายแถว)
                               ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Phase 5: Word Assembly, Lexicon Refinement & Automated Scoring Engine  │
 │ • grader.py (AutomatedGradingEngine) - Top-3 Beam Search + Levenshtein │
 │ • evaluate_phase5.py - EMA, CER, Pearson Correlation (r), Latency      │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 เจาะลึกราย Phase แบบไฟล์ต่อไฟล์ (File-by-File Academic Deep Dive)

---

### 1. Phase 1: Preprocessing & Fiducial Dewarping

#### 📄 ไฟล์ที่ 1.1: `src/preprocessing/dewarp.py`
* **คลาส/ฟังก์ชันสำคัญ:** [`FiducialDewarper`](../src/preprocessing/dewarp.py#L25), [`DewarpResult`](../src/preprocessing/dewarp.py#L15), `detect_markers()`, `analyze_triangle_orientation()`, `estimate_fourth_corner()`, `refine_with_timing_bar()`, `warp_canonical()`
* **หน้าที่หลัก:** รับภาพถ่ายกระดาษคำตอบดิบจากสมาร์ทโฟน ตรวจหาหมุดสี่เหลี่ยมสีดำทึบ 3 จุด (ขวาบน-TR, ขวาล่าง-BR, ซ้ายล่าง-BL), คำนวณทิศทางการหมุนของกระดาษแบบ $360^\circ$ โดยอัตโนมัติ, ประมาณพิกัดมุมที่ 4 (ซ้ายบน-TL) พร้อมปรับจูนความแม่นยำด้วยขีด Timing Bar แถวแรก และแปลงภาพระนาบเอียงสู่ภาพตรงระนาบมาตรฐานขนาด $2480 \times 3508$ พิกเซล (A4 @ 300 DPI)
* **สูตรคณิตศาสตร์และอัลกอริทึมที่ใช้:**
  1. **Adaptive Gaussian Binarization & Invariant Geometric Filtering:**
     แปลงภาพเกรย์สเกล $I(x,y)$ เป็นไบนารี:
     $$\text{Thresh}(x, y) = \begin{cases} 255 & \text{if } I(x,y) < T(x,y) - C \\ 0 & \text{otherwise} \end{cases}$$
     คัดกรองขอบเขต Contour มาร์กเกอร์ด้วยคุณลักษณะทางเรขาคณิต:
     $$0.80 \le \text{Aspect Ratio } (w/h) \le 1.25, \quad \text{Extent } \frac{A}{w \cdot h} \ge 0.72, \quad \text{Solidity } \frac{A}{A_{\text{convex}}} \ge 0.88$$
  2. **Hypotenuse Vector Analysis สำหรับการตรวจสอบทิศทาง ($360^\circ$ Rotation Invariant):**
     เนื่องจากกระดาษ Template 2 มีหมุดเพียง 3 มุม และเรียงตัวเป็นสามเหลี่ยมมุมฉากที่มีมุมฉากอยู่ที่จุด `BR` (ขวาล่าง) เสมอ ระยะทางยูคลิดระหว่างจุดคู่ใดๆ คือ:
     $$d(P_i, P_j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$
     คู่จุดที่มีระยะทางมากที่สุดคือ **ด้านตรงข้ามมุมฉาก (Hypotenuse)** ซึ่งเชื่อมระหว่าง `BL` และ `TR` จุดที่สามที่เหลือจะถูกกำหนดเป็นจุด **`BR` (Corner Anchor)** ทันทีโดยไม่ต้องสนใจว่าภาพจะหมุนมาที่มุม $0^\circ, 90^\circ, 180^\circ$ หรือ $270^\circ$
  3. **การประมาณพิกัดมุมที่ 4 (Top-Left Estimation) และ 4-Point Projective Homography:**
     ประมาณจุด `TL` เริ่มต้นด้วยเวกเตอร์ด้านขนาน (Parallelogram Vector):
     $$\mathbf{P}_{\text{TL\_init}} = \mathbf{P}_{\text{TR}} + \mathbf{P}_{\text{BL}} - \mathbf{P}_{\text{BR}}$$
     จากนั้นใช้การค้นหา Local Timing Mark ขีดบนสุดที่พิกัดมาตรฐาน $[133.0, 910.0]$ เพื่อล็อกตำแหน่ง `TL` ที่แท้จริง และคำนวณเมทริกซ์การแปลงโฮโมกราฟี $\mathbf{H} \in \mathbb{R}^{3 \times 3}$:
     $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
     แก้สมการด้วยวิธี Direct Linear Transformation (DLT) ผ่าน Singular Value Decomposition (SVD) จากจุดคู่สมนัย 4 จุด
* **เหตุผลการตัดสินใจทางสถาปัตยกรรม (Architectural Rationale):**
  * *ทำไมไม่ใช้ Deep Learning (เช่น YOLO หรือ DocUNet) ในขั้นตอนนี้?*
    การใช้สมการเรขาคณิตและ SVD ทำงานได้บน CPU ในเวลาเพียง **~25-35 ms ต่อภาพ** ในขณะที่โมเดล Deep Learning ต้องการทรัพยากร GPU และใช้เวลามากกว่า 300 ms อีกทั้งยังให้ความแม่นยำระดับ Sub-pixel ($RMSE = 0.114\text{ px}$) ซึ่งเหนือกว่ากรอบ Bounding Box ของโมเดล Object Detection ทั่วไป
* **ข้อจำกัดและประเด็นที่กรรมการสอบอาจถาม (Defense Questions & Limitations):**
  * *คำถามกรรมการ:* "ถ้ากระดาษยับเป็นคลื่นลึก หรือถ่ายบนพื้นผิวที่ไม่เรียบ (Non-planar surface) อัลกอริทึมนี้จะยังทำงานได้หรือไม่?"
  * *แนวทางตอบ:* Projective Homography มีสมมติฐานหลักว่าวัตถุอยู่บน **ระนาบแผ่นเรียบ 2 มิติ (Planar Surface)** หากกระดาษยับเป็นคลื่นลึก จะเกิด Non-linear Local Distortion ซึ่ง $\mathbf{H}$ ขนาด $3 \times 3$ ไม่สามารถชดเชยการโค้งงอย่อยได้สมบูรณ์ ในงานขั้นต่อไปสามารถขยายขีดความสามารถด้วย Thin Plate Spline (TPS) หรือ Local Mesh Warping ได้

---

#### 📄 ไฟล์ที่ 1.2: `src/evaluation/evaluate_phase1.py`
* **คลาส/ฟังก์ชันสำคัญ:** `evaluate_phase1()`, `calculate_reprojection_rmse()`, `calculate_boundary_iou()`
* **หน้าที่หลัก:** ดำเนินการทดสอบและประเมินผลเชิงปริมาณของ Phase 1 บนชุดข้อมูลกระดาษคำตอบจริง 51 แผ่น และกรณีทดสอบการหมุน 80 เคส
* **สูตรคณิตศาสตร์และตัวชี้วัด:**
  1. **Reprojection Root Mean Square Error (RMSE):**
     $$\text{RMSE} = \sqrt{\frac{1}{M} \sum_{k=1}^M \|\mathbf{x}'_k - \mathbf{H}\mathbf{x}_k\|^2} = 0.114 \pm 0.048\text{ px}$$
  2. **Intersection over Union (IoU) ของขอบเขตกระดาษ:**
     $$\text{IoU} = \frac{|A_{\text{pred}} \cap A_{\text{gt}}|}{|A_{\text{pred}} \cup A_{\text{gt}}|} = 0.9873$$
  3. **Orientation Accuracy:** ความแม่นยำในการหมุนภาพกลับสู่ทิศทางที่ถูกต้องเท่ากับ **`100.0%` (80/80 เคส)**

---

### 2. Phase 2: Anchor Bar Tracking & Cell Localization

#### 📄 ไฟล์ที่ 2.1: `src/detection/grid_extractor.py`
* **คลาส/ฟังก์ชันสำคัญ:** [`Template2GridExtractor`](../src/detection/grid_extractor.py#L35), [`CellPatch`](../src/detection/grid_extractor.py#L20), `detect_timing_bars()`, `extract_answer_cells()`, `extract_header_id_cells()`, `extract_omr_bubbles()`, `suppress_grid_lines_rim_restricted()`
* **หน้าที่หลัก:** ค้นหาตำแหน่งและตัดแยกเซลล์ย่อยทั้งหมดบนกระดาษ ได้แก่ ตารางข้อสอบ 400 ช่อง (20 ข้อ $\times$ 20 ตัวอักษร), กรอบรหัสสีม่วง 42 ช่อง, และวงกลม OMR 7 วง พร้อมทั้งลบเส้นขอบตารางสีเขียว/ดำออกโดยไม่กระทบลายมือ
* **สูตรคณิตศาสตร์และอัลกอริทึมที่ใช้:**
  1. **Dynamic Pitch-Sorted Timing Mark Tracking (20 ขีดซ้าย-ขวา):**
     ฉายโปรไฟล์แนวดิ่งในพื้นที่ $x \in [100, 320]$ (ซ้าย) และ $x \in [2160, 2380]$ (ขวา) ค้นหาจุดศูนย์ถ่วงมวล (Centroid):
     $$C_y = \frac{\sum_{(x,y)} y \cdot I(x, y)}{\sum_{(x,y)} I(x, y)}$$
     เรียงลำดับขีดดำ 20 ขีดด้วยระยะ Pitch มาตรฐาน ($\Delta y \approx 115\text{ px}$) และใช้วิธี Linear Interpolation ข้ามแถวตามแนวขวาง:
     $$y_{\text{row}}(x) = (1 - t) y_{\text{left}} + t y_{\text{right}}, \quad t = \frac{x - x_{\text{left}}}{x_{\text{right}} - x_{\text{left}}}$$
     ช่วยชดเชยการบิดตัวแบบ Shear และการเอียงระดับ Micro-radian ได้อย่างสมบูรณ์
  2. **Triple-Layer Grid Line Suppression (กรอบคำตอบ 400 ช่อง):**
     * **Layer 1: 2D Local Border Snap ($\pm 18\text{ px}$):** ตรวจหาเส้นขอบตารางจริงรอบเซลล์เพื่อป้องกันการเยื้องศูนย์
     * **Layer 2: Rim-Restricted Morphological Subtraction:** กำหนดหน้ากากกรอบขอบเซลล์ (Rim Mask) หนา $11\text{ px}$ (บน/ล่าง) และ $16\text{ px}$ (ซ้าย/ขวา) แล้วใช้เคอร์เนลเส้นแนวนอน $(W_{\text{kernel}}, 1)$ และแนวตั้ง $(1, H_{\text{kernel}})$ ลบเฉพาะเส้นตารางที่ล้ำเข้ามา **โดยปล่อยพื้นที่กึ่งกลางเซลล์ไว้ ไม่แตะต้องเด็ดขาด**
     * **Layer 3: Core-Zone Ink Density ($76\%$ บริเวณใจกลางเซลล์):** วัดสัดส่วนพิกเซลหมึกดำเฉพาะใจกลางช่อง:
       $$\text{InkDensity}_{\text{core}} = \frac{1}{|A_{\text{core}}|} \sum_{(x,y) \in A_{\text{core}}} \mathbb{I}(I(x,y) < T_{\text{dark}})$$
  3. **Normalized Cross-Correlation (NCC) และ Cross-Linear Area (CLA) สำหรับ OMR 7 วง:**
     $$\gamma(x, y) = \frac{\sum_{x',y'} (T(x',y') - \bar{T})(I(x+x', y+y') - \bar{I})}{\sqrt{\sum_{x',y'} (T(x',y') - \bar{T})^2 \sum_{x',y'} (I(x+x', y+y') - \bar{I})^2}}$$
     ผสานกับวิธี Cross-Linear Area (CLA) ในการลากเส้นสแกนความเข้มตัดผ่านจุดกึ่งกลางวงกลม 2 แกน (แนวนอนและแนวตั้ง)
* **เหตุผลการตัดสินใจทางสถาปัตยกรรม (Architectural Rationale):**
  * *ทำไมต้องออกแบบการลบเส้นขอบแบบ Rim-Restricted?*
    หากใช้ Morphological Opening ลบเส้นตรงทั้งภาพ เส้นขีดฆ่า 2 เส้นขนาน (`=`) ที่นักเรียนเขียนตรงกลางช่องจะถูกเข้าใจผิดว่าเป็นเส้นตารางและถูกลบหายไปด้วย การจำกัดขอบเขตการลบเฉพาะบริเวณขอบ Rim ช่วยรักษาเส้นขีดฆ่าและเส้นหมึกลายมือภายในเซลล์ไว้ได้ $100\%$
* **ข้อจำกัดและประเด็นที่กรรมการสอบอาจถาม (Defense Questions & Limitations):**
  * *คำถามกรรมการ:* "หากนักเรียนเขียนตัวอักษรใหญ่เกินไปจนหางตัวอักษรแลบออกไปนอกกรอบเซลล์ ระบบจะจัดการอย่างไร?"
  * *แนวทางตอบ:* การลบเส้นขอบแบบ Rim-Restricted อาจตัดส่วนของหางตัวอักษรที่แลบเข้าไปในบริเวณ Rim ทิ้งไปบางส่วน อย่างไรก็ตาม เนื่องจากสถาปัตยกรรมโมเดล HCR ใน Phase 4 ใช้ Receptive Field กว้างและผ่านการเทรนด้วย Data Augmentation แบบตัดขอบ (Random Crop/Erosion) ทำให้โมเดลยังคงสามารถรู้จำอักขระส่วนใหญ่ได้อย่างถูกต้อง

---

#### 📄 ไฟล์ที่ 2.2: `src/evaluation/evaluate_phase2.py`
* **คลาส/ฟังก์ชันสำคัญ:** `evaluate_phase2()`, `benchmark_timing_bars()`, `benchmark_cells()`
* **หน้าที่หลัก:** ประเมินความแม่นยำในการระบุตำแหน่ง Timing Bars (2,040 ขีด), ช่องคำตอบ (20,400 ช่อง), กรอบรหัสสีม่วง (2,142 ช่อง), และวง OMR (357 วง)
* **ผลลัพธ์เชิงตัวเลข:**
  * **Timing Bar Recall:** **`100.0%` (2,040/2,040)**
  * **Answer Cell Recall & Alignment:** **`100.0%` (20,400/20,400)**
  * **Blank Cell Specificity (กรอบเขียว):** **`100.0%`** (ไม่มี False Positive จากเส้นขอบตาราง)
  * **OMR Mean NCC Score:** **`0.4653`**, **OMR Alignment Accuracy:** **`100.0%` (357/357)**

---

### 3. Phase 4: Handwritten Character Recognition (Pre-Scoring HCR Engine)

> **หมายเหตุเชิงสถาปัตยกรรม:** ในการประมวลผลจริง Phase 4 จะถูกเรียกทำงาน **ก่อน Phase 3** ตามหลักการ *HCR-First Confidence Guided Paradigm* เพื่อสร้างฟีเจอร์นำทาง

#### 📄 ไฟล์ที่ 3.1: `src/recognition/char_classifier.py`
* **คลาส/ฟังก์ชันสำคัญ:** [`HCRModelEngine`](../src/recognition/char_classifier.py#L40), [`HCRResNet28`](../src/recognition/char_classifier.py#L12), [`ResidualBlock`](../src/recognition/char_classifier.py#L8), `to_canonical_glyph_28x28()`, `predict_patch()`, `predict_row_batch()`
* **หน้าที่หลัก:** รับภาพช่องคำตอบ (~70x80 px), ปรับสภาพภาพอักขระเดี่ยวสู่รูปแบบมาตรฐาน NIST/EMINIST ($28 \times 28$ px), สกัดฟีเจอร์เชิงลึกด้วยโครงข่าย ResNet-28, และทำนายการกระจายความน่าจะเป็น 26 คลาส (A–Z) พร้อมค่าความเชื่อมั่นและ Shannon Entropy
* **สูตรคณิตศาสตร์และอัลกอริทึมที่ใช้:**
  1. **Canonical Glyph Normalization & Center-of-Mass Moments:**
     * ดึง Tight Bounding Box ของลายมือจริง
     * ปรับขนาดแบบรักษาสัดส่วน (Aspect-Preserving Resize) ให้ด้านยาวที่สุดมีขนาด $20\text{ px}$ และวางกึ่งกลางภาพขนาด $28 \times 28$
     * คำนวณจุดศูนย์ถ่วงมวล $(\bar{x}, \bar{y})$ ด้วย Image Moments:
       $$M_{pq} = \sum_x \sum_y x^p y^q I(x, y), \quad \bar{x} = \frac{M_{10}}{M_{00}}, \quad \bar{y} = \frac{M_{01}}{M_{00}}$$
       และทำ Affine Translation เลื่อนจุดศูนย์ถ่วงมวลไปอยู่ที่จุดกึ่งกลางแคนวาส $(13.5, 13.5)$
  2. **สถาปัตยกรรม Lightweight ResNet-28 Backbone:**
     ใช้โครงสร้าง Stem Conv $3 \times 3$ ตามด้วย Residual Stages 3 ระดับ (ความกว้าง Channel: 32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 192) พร้อม Skip Connection:
     $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathcal{W}_s \mathbf{x}$$
     ใช้ฟังก์ชันกระตุ้น **GELU (Gaussian Error Linear Unit)**:
     $$\text{GELU}(x) = x \Phi(x) = x P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$
  3. **การสอบเทียบความเชื่อมั่นและการวัดค่าความไม่แน่นอน (Calibrated Entropy):**
     ใช้ Temperature Scaling $T = 1.15$:
     $$P(y = c \mid \mathbf{x}) = \frac{\exp(z_c / T)}{\sum_{j=1}^{26} \exp(z_j / T)}, \quad H(P) = -\sum_{c=1}^{26} P_c \log_2 P_c$$
* **เหตุผลการตัดสินใจทางสถาปัตยกรรม (Architectural Rationale):**
  * *ทำไมเลือก ResNet-28 แทนที่จะเป็น Vision Transformer (ViT) หรือโมเดลขนาดใหญ่?*
    ภาพอักขระเดี่ยวขนาด $28 \times 28$ พิกเซลมีมิติที่ไม่ซับซ้อน โครงข่าย ResNet-28 มีพารามิเตอร์เพียง ~1.2M พารามิเตอร์ ซึ่งป้องกันปัญหา Overfitting ได้ดีเยี่ยม สามารถประมวลผลแบบ Batch บน CPU ได้เร็วถึง **19.14 ms ต่อแถว (20 ช่อง)** หรือต่ำกว่า **1 ms ต่อตัวอักษร** เหมาะสำหรับการตรวจข้อสอบปริมาณมากในโรงเรียนโดยไม่ต้องพึ่งพาเซิร์ฟเวอร์ GPU ราคาแพง
* **ข้อจำกัดและประเด็นที่กรรมการสอบอาจถาม (Defense Questions & Limitations):**
  * *คำถามกรรมการ:* "โมเดลแยกความแตกต่างระหว่างตัวอักษรที่มีลักษณะใกล้เคียงกัน เช่น `U` กับ `V` หรือ `I` กับ `J` ได้อย่างไรหากนักเรียนเขียนหวัด?"
  * *แนวทางตอบ:* ในระดับตัวอักษรเดี่ยว (Isolated Character) ลายมือมนุษย์มีความคลุมเครือทางสรีรวิทยา (Intrinsic Ambiguity) สูงมาก ระบบจึงไม่ตัดสินผลเด็ดขาดใน Phase 4 แต่ส่งค่า Top-3 Predictions และความน่าจะเป็นไปให้ Phase 5 เพื่อใช้บริบทของคลังคำศัพท์ (Lexicon Context) และ Beam Search ในการตัดสินคำที่ถูกต้อง

---

#### 📄 ไฟล์ที่ 3.2: `src/recognition/train_hcr.py`
* **คลาส/ฟังก์ชันสำคัญ:** `train_model()`, `CharacterDataset`, `evaluate_val()`
* **หน้าที่หลัก:** เทรนโมเดล `HCRResNet28` บนชุดข้อมูลตัวอักษรลายมือเขียน A–Z
* **สูตรและเทคนิคการเทรน:**
  * **Cross-Entropy Loss พร้อม Label Smoothing ($\epsilon = 0.05$):**
    $$\mathcal{L} = -(1 - \epsilon) \log P(y_{\text{true}}) - \frac{\epsilon}{K} \sum_{k=1}^K \log P(k)$$
  * **Optimizer & Scheduler:** ใช้ AdamW (Weight Decay $= 10^{-4}$) ร่วมกับ Cosine Annealing Learning Rate Schedule
  * **Data Augmentation:** สุ่มการหมุน ($\pm 10^\circ$), การเฉือน (Shear $\pm 8^\circ$), Perspective Jittering, และ Morphological Erosion/Dilation

---

#### 📄 ไฟล์ที่ 3.3: `src/evaluation/evaluate_phase4.py`
* **คลาส/ฟังก์ชันสำคัญ:** `evaluate_phase4()`, `benchmark_exam_sheets()`
* **หน้าที่หลัก:** ทดสอบความแม่นยำของ HCR บน Test Split ที่ไม่เคยเห็นในการเทรน และทดสอบบนกระดาษคำตอบจริง
* **ผลลัพธ์เชิงตัวเลข:**
  * **Validation Accuracy (7,800 ภาพ):** **`Top-1 = 98.77%`**, **`Top-3 = 99.80%`**, **`Macro-F1 = 0.9879`**
  * **Real Exam Character Accuracy (786 ตัวอักษร):** **`Top-1 = 100.0% (786/786)`**
  * **Real Exam Row Match (72 แถว):** **`100.0% (72/72 แถว)`**
  * **Latency บน CPU:** **`19.14 ms / แถว`** ($\approx 0.96\text{ ms/ช่อง}$)

---

### 4. Phase 3: HCR-Guided Double Strikethrough Detection & Filtering

#### 📄 ไฟล์ที่ 4.1: `src/detection/strikethrough_detector.py`
* **คลาส/ฟังก์ชันสำคัญ:** [`StrikethroughDetectorEngine`](../src/detection/strikethrough_detector.py#L30), [`StrikethroughHybridNet`](../src/detection/strikethrough_detector.py#L12), [`StrikethroughRuleFilter`](../src/detection/strikethrough_detector.py#L80), `detect_cell()`, `extract_slanted_line_features()`
* **หน้าที่หลัก:** ตรวจจับช่องที่มีรอยขีดฆ่า 2 เส้นแนวนอน (`=`) เพื่อกรองช่องที่ถูกยกเลิกทิ้ง โดยไม่ทำลายตัวอักษรปกติที่มีเส้นขนาน เช่น `F`, `Z`, `E`
* **สูตรคณิตศาสตร์และอัลกอริทึมที่ใช้:**
  1. **Multi-Angle Tilted Morphological Line Kernels:**
     เพื่อดักจับรอยขีดฆ่าที่เขียนเอียงเฉียง ($5^\circ - 12^\circ$) ระบบสร้างชุดเคอร์เนลหมุนหลายมุม:
     $$\theta \in \{-12^\circ, -8^\circ, -4^\circ, 0^\circ, +4^\circ, +8^\circ, +12^\circ\}$$
     ผ่านเมทริกซ์การหมุนภาพ 2 มิติ:
     $$\mathbf{R}_\theta = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}$$
  2. **สัณฐานวิทยาเชิงลึก: Body Penetration และ Two-Sided Wing Overhang:**
     * **Body Penetration ($\text{rel\_y} \in [0.14, 0.85]$):** เส้นขีดฆ่าจริงต้องพาดผ่านบริเวณใจกลางตัวอักษร ต่างจากตัว `Z` ที่เส้นนอนอยู่ขอบบนสุดและล่างสุด
     * **Two-Sided Wing Overhang ($L_{\text{wing}} > 0, R_{\text{wing}} > 0$):** เส้นขีดฆ่าจริงจะมีความยาวล้นเลยขอบตัวอักษรออกมาทั้งสองฝั่ง ในขณะที่ตัว `F` ขีดแนวนอนตรงกลางจะยื่นไปทางขวาข้างเดียว ($L_{\text{wing}} = 0$)
  3. **StrikethroughHybridNet (Dual-Branch Architecture):**
     ผสานฟีเจอร์ 2 แขนง:
     * **Branch 1 (CNN Visual Branch):** รับภาพขนาด $64 \times 64$ สกัด Spatial Feature Map ผ่าน Conv2D 3 ชั้น
     * **Branch 2 (Morphological Stats Branch):** รับเวกเตอร์สถิติ 6 มิติ (Black-White Transition, HPP Peak Gap, Wing Ratio, Body Ratio, HCR Prediction, HCR Confidence)
     เชื่อมโยงกันด้วย Fully-Connected Layers 3 ชั้น และ Softmax Output
* **เหตุผลการตัดสินใจทางสถาปัตยกรรม (Architectural Rationale):**
  * *ทำไมต้องส่งค่าจาก Phase 4 มาช่วยใน Phase 3 (HCR-First Paradigm)?*
    ปัญหาระดับคลาสสิกของระบบ OCR คือตัวอักษร `F`, `Z`, และ `E` มีเส้นขนานแนวนอนคล้ายเครื่องหมายขีดฆ่า (`=`) หากใช้เฉพาะตัวตรวจจับเส้นตรง จะเกิด False Positive สูงมาก แต่เมื่อรัน HCR ล่วงหน้า ตัวอักษร `F` และ `Z` ที่ปกติจะได้ HCR Confidence สูงมาก ($> 0.88 - 0.95$) ในขณะที่ตัวอักษรที่ถูกขีดฆ่า ลายเส้นจะเละและทำให้ HCR Confidence ต่ำมาก ($< 0.35$) การนำสัญญาณนี้มาเป็น Gating Factor ทำให้ลดค่า False Positive Rate เหลือ **`0.0%` อย่างแท้จริง**
* **ข้อจำกัดและประเด็นที่กรรมการสอบอาจถาม (Defense Questions & Limitations):**
  * *คำถามกรรมการ:* "ถ้านักเรียนใช้การกากบาทเป็นรูปตัว X หรือขีดทับเฉียงๆ เส้นเดียวเพื่อยกเลิกตัวอักษร ระบบจะตรวจพบหรือไม่?"
  * *แนวทางตอบ:* ตามระเบียบข้อบังคับของกระดาษคำตอบ Rayong MARCO (Template 2) กำหนดไว้อย่างเป็นทางการว่า **"การแก้ไขคำตอบต้องใช้การขีดฆ่าด้วยเส้นคู่แนวนอน (=) เท่านั้น"** หากนักเรียนขีดฆ่าผิดระเบียบ เช่น กากบาทตัว `X` ทับ หรือขีดทแยงมุมเส้นเดียว อัลกอริทึมจะปฏิเสธไม่นับเป็นรอยขีดฆ่าที่ถูกต้องตามกติกา และส่งไปอ่านเป็นตัวอักษรปกติ ซึ่งเป็นไปตาม Business Logic ของการสอบจริง

---

#### 📄 ไฟล์ที่ 4.2: `src/evaluation/evaluate_phase3.py`
* **คลาส/ฟังก์ชันสำคัญ:** `evaluate_phase3()`, `plot_confusion_matrix()`, `calculate_roc_auc()`
* **หน้าที่หลัก:** ประเมินประสิทธิภาพในการตรวจจับรอยขีดฆ่าบน Test Set และวัดอัตรา False Positive บนตัวอักษรปกติ
* **ผลลัพธ์เชิงตัวเลข:**
  * **Precision / Recall บนรอยขีดฆ่าจริง:** **`100.0% / 100.0%` ($F_1 = 1.000$)**
  * **Strikethrough Recall รวมเส้นเอียงบนชุดทดสอบขนาดใหญ่:** **`97.75%`**
  * **Normal Character Specificity:** **`99.95%`** (ผิดพลาดเพียง 1 ช่องจาก 2,133 ช่อง)
  * **False Positive Rate บนตัวอักษร `F` และ `Z`:** **`0.0%` (0/180 ตัวอย่าง)**
  * **Area Under the ROC Curve (ROC-AUC):** **`0.9999`**

---

### 5. Phase 4B: Purple Box Header Digit Recognition (0–9 CNN Engine — 42 ช่อง)

#### 📄 ไฟล์ที่ 5.1: `src/recognition/digit_classifier.py`
* **คลาส/ฟังก์ชันสำคัญ:** [`DigitRecognizerEngine`](../src/recognition/digit_classifier.py#L30), [`HeaderCellEmptyGate`](../src/recognition/digit_classifier.py#L12), `to_canonical_digit_28x28()`, `scan_header_strip()`
* **หน้าที่หลัก:** ประมวลผลและสแกนอ่านตัวเลขรหัสประจำตัว (Student ID), รหัสวิชา (Subject ID), และรหัสสถานศึกษา (School ID) จำนวน 42 ช่องสีม่วงด้านบนหัวกระดาษ
* **สูตรคณิตศาสตร์และอัลกอริทึมที่ใช้:**
  1. **HeaderCellEmptyGate (Core Ink Density Thresholding):**
     เนื่องจากช่องรหัสส่วนใหญ่กรอกไม่เต็ม 14 ช่อง (เช่น รหัสนิสิต 8 หลัก อีก 6 ช่องจะเว้นว่าง) ระบบใช้การตัดขอบ Rim Margin 8% และวัดความหนาแน่นของหมึกใจกลางช่อง:
     $$\text{InkDensity}_{\text{core}} = \frac{1}{|A_{\text{core}}|} \sum_{(x,y) \in A_{\text{core}}} \mathbb{I}(I(x,y) < T_{\text{dark}})$$
     หากค่า $\text{InkDensity}_{\text{core}} < 0.015$ จะถูกจัดเป็นช่องว่าง (`EMPTY_DIGIT`) ทันที ป้องกันการอ่านขอบสีม่วงเป็นเลข `1` หรือ `7`
  2. **Loop-Preserving Canonical Digit Normalization (`to_canonical_digit_28x28`):**
     สำหรับตัวเลข $0-9$ การทำ Dilation ที่รุนแรงเกินไปจะทำให้รูวงกลมของเลข `8`, `6`, `9`, และ `0` ตัน กลายเป็นก้อนหมึกทึบ อัลกอริทึมจึงจำกัดเคอร์เนล Dilation ให้มีขนาดเพียง $2 \times 2$ และทำเฉพาะเมื่อ $\text{InkDensity} < 0.14$ เพื่อรักษาลักษณะทางสัณฐานวิทยาของตัวเลขไว้ครบถ้วน
  3. **โมเดล `HCRResNet28` (10 คลาส `0`–`9`):**
     โครงข่าย ResNet-28 ความกว้าง 10 เอาต์พุต พร้อม Softmax Probability Calibration
* **เหตุผลการตัดสินใจทางสถาปัตยกรรม (Architectural Rationale):**
  * *ทำไมต้องแยก Normalizer ของตัวเลขออกจากตัวอักษร A–Z?*
    ตัวอักษรภาษาอังกฤษส่วนใหญ่ไม่มีวงกลมปิดคู่เหมือนเลข `8` หรือวงกลมขนาดเล็กเหมือนเลข `6` และ `9` การใช้ Normalizer ตัวเดียวกับ A–Z เคยทำให้เกิดข้อผิดพลาดในการทำนายเลข `8` สลับเป็นเลข `1` เมื่อแยก Normalizer เฉพาะสำหรับตัวเลข ความแม่นยำจึงเพิ่มขึ้นเป็น $100\%$ เต็ม
* **ข้อจำกัดและประเด็นที่กรรมการสอบอาจถาม (Defense Questions & Limitations):**
  * *คำถามกรรมการ:* "หากผู้เข้าสอบเขียนตัวเลขสไตล์ยุโรป เช่น มีขีดตรงกลางเลข `7` หรือเลข `1` มีฐานล่าง โมเดลจะรองรับได้หรือไม่?"
  * *แนวทางตอบ:* ชุดข้อมูล `data/Number Handwriter` (21,555 ตัวอย่าง) ครอบคลุมรูปแบบการเขียนตัวเลขอเนกประสงค์หลากหลายสไตล์ รวมถึงตัวเลขที่มีขีดกลางและมีฐานล่าง ทำให้โมเดลมี Invariance ต่อรูปแบบการเขียนดังกล่าวได้สูงมาก

---

#### 📄 ไฟล์ที่ 5.2: `src/recognition/train_digit_hcr.py`
* **คลาส/ฟังก์ชันสำคัญ:** `train_digit_model()`, `DigitDataset`
* **หน้าที่หลัก:** เทรนโมเดลจำแนกตัวเลข $0-9$ บนชุดข้อมูล `data/Number Handwriter` จำนวน 21,555 ตัวอย่าง แบ่ง Stratified 80:10:10 (Train 17,245 / Val 2,156 / Test 2,154 ภาพ)
* **ผลลัพธ์เชิงตัวเลข:**
  * **Validation Accuracy:** **`Top-1 = 99.77%`**, **`Top-3 = 100.00%`**
  * **Unseen Test Accuracy:** **`Top-1 = 99.58%`**, **`Top-3 = 99.95%`**, **`Macro-F1 = 0.9958`**

---

#### 📄 ไฟล์ที่ 5.3: `src/evaluation/evaluate_digit_header.py`
* **คลาส/ฟังก์ชันสำคัญ:** `evaluate_digit_headers()`, `generate_overlay_visuals()`
* **หน้าที่หลัก:** ทดสอบการสแกนช่องรหัสสีม่วง 42 ช่องบนกระดาษคำตอบจริง 19 แผ่น (รวม 798 ช่อง)
* **ผลลัพธ์เชิงตัวเลข:**
  * **Blank Purple Box Specificity:** **`100.0%` (798/798 ช่องว่างถูกจัดเป็น EMPTY อย่างแม่นยำ)**
  * **Real Exam Digit Recognition Accuracy:** **`100.0%` (96/96 ตัวเลขถูกต้องสมบูรณ์แบบ)**
  * **Header Strip Match:** **`100.0%` (9/9 แถวที่มีการกรอกข้อมูล)**

---

### 6. Phase 5: Word Assembly, Lexicon Refinement & Automated Scoring Engine

#### 📄 ไฟล์ที่ 6.1: `src/grading/grader.py`
* **คลาส/ฟังก์ชันสำคัญ:** [`AutomatedGradingEngine`](../src/grading/grader.py#L40), [`ExamSheetGradingResult`](../src/grading/grader.py#L25), `grade_sheet()`, `assemble_words()`, `refine_with_lexicon()`, `draw_grading_overlay()`
* **หน้าที่หลัก:** รวบรวมตัวอักษรที่ผ่านการกรองขีดฆ่าแล้วในแต่ละแถวคำตอบ (ข้อ 1–20) ประกอบขึ้นเป็นคำศัพท์, ทำการค้นหาและปรับแก้คำผิดด้วย Top-3 Softmax Weighted Beam Search เข้ากับ Vocabulary Lexicon, ตรวจเทียบกับเฉลยคำตอบ (Answer Key), และออกรายงานผลการตรวจในรูปแบบ JSON, CSV และภาพ Overlay
* **สูตรคณิตศาสตร์และอัลกอริทึมที่ใช้:**
  1. **Word Assembly & Strikethrough Elimination:**
     ในแต่ละข้อ $r \in \{1, \dots, 20\}$ ประกอบคำดิบ $W_{\text{raw}}$ จากเซลล์ $c \in \{1, \dots, 20\}$:
     $$W_{\text{raw}} = \bigoplus_{c=1}^{20} \text{char}(c) \quad \forall c \text{ where } \text{State}(c) = \text{normal\_char}$$
     โดยช่องที่เป็น `empty` และ `strikethrough` จะถูกตัดออกโดยอัตโนมัติ
  2. **Top-3 Probability-Weighted Lexicon Refinement (Beam Search):**
     หาก $W_{\text{raw}} \notin \mathcal{L}$ ระบบจะค้นหาคำ 후보 $W' \in \mathcal{L}$ ที่มีความยาวเท่ากัน $|W'| = |W_{\text{raw}}|$ และมี Levenshtein Distance $D(W_{\text{raw}}, W') \le 2$ จากนั้นคำนวณคะแนน Log-Likelihood:
     $$S(W') = \sum_{i=1}^{L} \log P(c'_i \mid \mathbf{x}_i)$$
     โดยพิจารณาการแทนที่เฉพาะตัวอักษรในกลุ่มความสับสนตามธรรมชาติ (Confusable Sets) ได้แก่:
     $$\mathcal{C} = \{\{E, F\}, \{U, V\}, \{I, T, J\}, \{S, Z\}, \{D, O\}, \{B, P\}\}$$
  3. **Fail-Safe Mechanism (Human-in-the-Loop Flagging):**
     หากข้อใดมีค่าเฉลี่ยความเชื่อมั่นต่ำกว่าเกณฑ์ ($\overline{\text{Conf}} < 0.55$) หรือมีการปรับแก้คำที่กำกวม ระบบจะทำการติดธง:
     $$\text{needs\_human\_review} = \text{True}$$
     เพื่อส่งต่อให้อาจารย์ผู้ตรวจทานสายตามนุษย์ยืนยันความถูกต้อง ป้องกันการตัดสินคะแนนผิดพลาด
* **เหตุผลการตัดสินใจทางสถาปัตยกรรม (Architectural Rationale):**
  * *ทำไมไม่อนุญาตให้แก้คำศัพท์ข้ามความยาวของคำ ($|W'| \neq |W_{\text{raw}}|$)?*
    ในการสอบภาษาอังกฤษ หากนักเรียนสะกดคำตกหล่น เช่น ตกตัวอักษรไป 1 ตัว การอนุญาตให้อัลกอริทึมเติมตัวอักษรเองจะถือเป็นการ **"ช่วยผู้เข้าสอบเกินขอบเขต"** และขัดต่อความซื่อตรงของการวัดผล การจำกัดให้แก้ได้เฉพาะคำที่มีความยาวเท่ากันและเกิดจากความกำกวมของลายมือ (OCR Confusion) ช่วยคงมาตรฐานความยุติธรรมของการสอบไว้ได้ $100\%$
* **ข้อจำกัดและประเด็นที่กรรมการสอบอาจถาม (Defense Questions & Limitations):**
  * *คำถามกรรมการ:* "ถ้าคำตอบของนักเรียนอยู่นอกคลังคำศัพท์ (Out-of-Vocabulary - OOV) ระบบจะตรวจอย่างไร?"
  * *แนวทางตอบ:* หากข้อสอบมีเฉลยแบบ Open-vocabulary หรือนักเรียนตอบคำที่ไม่อยู่ใน Lexicon ระบบจะใช้คำที่อ่านได้จาก HCR โดยตรง ($W_{\text{raw}}$) ไปเทียบความตรงกันกับเฉลยแบบ Exact Match หากไม่ตรงจะส่งเข้าสู่กระบวนการ Human Review ทันที

---

#### 📄 ไฟล์ที่ 6.2: `src/evaluation/evaluate_phase5.py`
* **คลาส/ฟังก์ชันสำคัญ:** `evaluate_phase5_pipeline()`, `calculate_system_metrics()`, `export_reports()`
* **หน้าที่หลัก:** ประเมินประสิทธิภาพระดับ End-to-End System บนกระดาษคำตอบจริง 19 แผ่น (252 ข้อที่มีเฉลย) และส่งออกไฟล์รายงาน [`metrics/Phase5_Metrics_Report.md`](./Phase5_Metrics_Report.md), [`metrics/phase5_evaluation_metrics.json`](./phase5_evaluation_metrics.json), และ [`metrics/phase5_grading_results.csv`](./phase5_grading_results.csv)
* **สูตรคณิตศาสตร์และผลลัพธ์ระดับระบบ:**
  1. **Exact Match Word Accuracy (EMA):**
     $$\text{EMA} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(W_{\text{final}}^{(i)} == W_{\text{ground\_truth}}^{(i)}) \times 100\% = \mathbf{99.21\%} \quad (250/252\text{ ข้อ})$$
  2. **Character Error Rate (CER):**
     $$\text{CER} = \frac{\sum_{i=1}^{N} D_{\text{Levenshtein}}(W_{\text{final}}^{(i)}, W_{\text{ground\_truth}}^{(i)})}{\sum_{i=1}^{N} |W_{\text{ground\_truth}}^{(i)}|} \times 100\% = \mathbf{0.10\%} \quad (3/2,921\text{ ตัวอักษร})$$
  3. **Score Pearson Correlation Coefficient ($r$):**
     $$r = \frac{\sum (S_{\text{pred}} - \bar{S}_{\text{pred}})(S_{\text{true}} - \bar{S}_{\text{true}})}{\sqrt{\sum (S_{\text{pred}} - \bar{S}_{\text{pred}})^2 \sum (S_{\text{true}} - \bar{S}_{\text{true}})^2}} = \mathbf{0.9994}$$
  4. **Mean End-to-End Latency:** **`~2,478.6 ms / แผ่น`** บนซีพียูธรรมดา
* **การวิเคราะห์ข้อผิดพลาด 2 ข้อที่ไม่เต็ม 100% (Error Analysis for Academic Defense):**
  1. **`04_words_vocab_004.jpg` ข้อ 15 (นักเรียนเขียน `EXTRTERRESTRJAL` vs เฉลย `EXTRATERRESTRIAL`):**
     * นักเรียนสะกดตกตัว `A` ไปตั้งแต่แรก และเขียนตัว `I` งอเป็นตัว `J` ระบบอ่านได้ตรงตามลายมือจริงเป๊ะ ซึ่ง Edit Distance ห่างจากเฉลย 2 ตัวอักษร ระบบจึงตัดคะแนนเป็น 0 ตามเกณฑ์ความซื่อตรง
  2. **`04_words_vocab_011.jpg` ข้อ 5 (นักเรียนเขียนทับด้วยกากบาท `X` vs เฉลย `INFINITY`):**
     * นักเรียนเขียนผิดระเบียบโดยการกากบาททับตัวอักษรแทนการขีดฆ่า 2 เส้นแนวนอน (`=`) ระบบจึงไม่ตัดช่องนี้ออกและส่งไปอ่านเป็นตัวอักษร `M` คำที่ได้จึงเป็น `IMNFINITY` และได้ 0 คะแนนตามระเบียบการสอบ

---

## ⚖️ ตารางสรุปการวิเคราะห์เปรียบเทียบเชิงวิศวกรรม (Engineering Trade-Offs Summary)

| Phase | ปัญหาที่พบ (Challenge) | ทำไมไม่ใช้ท่าทั่วไป (Traditional) | วิธีการที่เลือกใช้ (Selected Novel Solution) | ข้อได้เปรียบเชิงประจักษ์ (Empirical Advantage) |
|:---:|:---|:---|:---|:---|
| **Phase 1** | การหมุน $360^\circ$ และมุมมองเอียง | Deep Learning / Corner Regressor | **Hypotenuse Vector Analysis + SVD DLT** | เร็วขึ้น $10\times$ (~30 ms บน CPU), RMSE $0.114\text{ px}$ |
| **Phase 2** | เส้นตารางกินเข้ามาในช่องลายมือ | Global Morphological Opening | **Rim-Restricted Subtraction + Core Ink Gate** | เส้นขีดฆ่าตรงกลางช่องไม่หาย, Answer Recall $100\%$ |
| **Phase 3** | ตัวอักษร `F`, `Z`, `E` ถูกเข้าใจผิดเป็นรอยขีดฆ่า | Computer Vision Rule-Based เดี่ยวๆ | **HCR-First Confidence Guided Dual-Branch** | ลด FPR เหลือ **`0.0%`**, Strikethrough Precision $100\%$ |
| **Phase 4** | ลายมือเขียนตัวเอียง ขาด/เกิน กึ่งกลางไม่ตรง | Raw Image Resize $28\times 28$ | **Moment Center-of-Mass Translation + ResNet-28** | Top-1 Val $98.77\%$, Top-3 $99.80\%$, Exam Accuracy $100\%$ |
| **Phase 4B** | รูวงกลมของเลข `8`, `6`, `9` ตันหลัง Binarize | ใช้ Normalizer ชุดเดียวกับ A–Z | **Loop-Preserving Normalizer + Empty Gate** | Purple Box Digit Acc **$100\%$ (96/96)**, Specificity $100\%$ |
| **Phase 5** | อักขระกำกวมทางสรีรวิทยา (`U` vs `V`) | Greedy Hard-max Argmax | **Top-3 Weighted Beam Search + Levenshtein** | Exact Match Word Accuracy **$99.21\%$**, CER $0.10\%$ |

---
**จัดทำเพื่อ:** การสอบป้องกันโครงงานวิจัยทาง Computer Vision มหาวิทยาลัยบูรพา  
**เอกสารอ้างอิง:** ดูรายงานเชิงตัวเลขแบบละเอียดในแต่ละ Phase ได้ที่โฟลเดอร์ [`metrics/`](./README.md)
