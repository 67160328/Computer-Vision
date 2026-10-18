# รายงานตัวชี้วัดเชิงปริมาณและเอกสารวิชาการเจาะลึก (Quantitative Metrics & Academic Defense Report)
## Phase 3: HCR-Guided Double Strikethrough Detection & Filtering

**โครงการ:** ระบบตรวจกระดาษคำตอบและให้คะแนนอัตโนมัติ Rayong MARCO (Template 2)  
**ไฟล์โมดูลที่ประเมิน:** [`src/detection/strikethrough_detector.py`](../src/detection/strikethrough_detector.py) (`StrikethroughDetectorEngine`, `StrikethroughHybridNet`)  
**สคริปต์ประเมินผล:** [`src/evaluation/evaluate_phase3.py`](../src/evaluation/evaluate_phase3.py)  
**ชุดข้อมูลทดสอบ:** สกัดจากกระดาษคำตอบต้นฉบับ 145 แผ่น รวม **25,802 ช่อง** แบ่งแบบ Sheet-Level Stratified 80:10:10 (Train 20,820 / Val 2,440 / Test 2,542 ช่อง)  
**ไฟล์ข้อมูลดิบ JSON:** [`metrics/phase3_evaluation_metrics.json`](./phase3_evaluation_metrics.json)  
**โมเดลที่บันทึก:** `models/strikethrough_cnn.pt`  
**มาตรฐานทางวิชาการที่อ้างอิง:** ICDAR Strikethrough Detection, Dual-Branch Hybrid CNN-Morphology, HCR Confidence Gated Inference

---

## 1. สรุปตัวชี้วัดหลักเทียบกับเกณฑ์เป้าหมาย (Executive SLA Summary)

| ตัวชี้วัดเชิงปริมาณ (Quantitative Metric) | สูตร/วิธีการวัดผล | เกณฑ์เป้าหมาย (Target SLA) | ผลลัพธ์ที่ได้จริง (Actual Result) | สถานะ |
|:---|:---|:---:|:---:|:---:|
| **1. Strikethrough Precision** | อัตราความแม่นยำในการระบุช่องขีดฆ่าจริง ($TP / (TP + FP)$) | $\ge 95.0\%$ | **$98.86\%$ (Test) / $100.0\%$ (Benchmark)** | **PASS** |
| **2. Strikethrough Recall** | อัตราการดักจับช่องขีดฆ่าได้ครบ ($TP / (TP + FN)$) | $\ge 95.0\%$ | **$97.75\%$ (Test) / $100.0\%$ (Benchmark)** | **PASS** |
| **3. Strikethrough Macro-F1** | ค่าฮาร์มอนิกเฉลี่ยของ Precision และ Recall | $\ge 0.9500$ | **$0.9831$ (Test) / $1.000$ (Benchmark)** | **PASS** |
| **4. Normal Character Specificity** | อัตราการไม่ตัดตัวอักษรปกติทิ้ง ($TN / (TN + FP)$) | $\ge 99.50\%$ | **$99.95\%$ (2,132 / 2,133 ช่อง)** | **PASS** |
| **5. False Positive Rate on `F` & `Z`** | อัตราการตรวจจับผิดพลาดบนตัวอักษรที่มีเส้นขนานในตัว | $\le 0.50\%$ | **$0.00\%$ (0 / 180 ตัวอย่าง)** | **PASS** |
| **6. Area Under ROC Curve (ROC-AUC)** | ความสามารถในการแยกแยะคลาสขีดฆ่า | $\ge 0.9900$ | **$0.9999$** | **PASS** |

---

## 2. ทฤษฎีและที่มาเชิงคณิตศาสตร์อย่างละเอียด (Mathematical Formulations & Algorithmic Rigor)

ตามกติกาของกระดาษคำตอบ Rayong MARCO ผู้เข้าสอบแก้ไขคำตอบโดยการ **"ขีดฆ่าด้วยเส้นคู่แนวนอน (=)"** ปัญหาทาง Computer Vision ที่ท้าทายคือ:
1. เส้นขีดฆ่ามักเขียนเอียงเฉียง ($5^\circ - 12^\circ$)
2. ตัวอักษรปกติ เช่น `F` (มี 2 ขีดบน) และ `Z` (มีขีดบนและล่าง) มีลักษณะทางเรขาคณิตคล้ายเครื่องหมาย `=` มาก

ระบบแก้ปัญหานี้ด้วย **3 เสาหลักทางคณิตศาสตร์และสถาปัตยกรรม**:

### 2.1 Multi-Angle Tilted Line Morphological Kernels
เคอร์เนลเส้นแนวนอนตรงแบบเดิม $(W_k, 1)$ ล้มเหลวเมื่อผู้สอบขีดเส้นเอียง ระบบจึงสร้างชุดเคอร์เนลหมุนหลายมุม:
$$\theta \in \{-12^\circ, -8^\circ, -4^\circ, 0^\circ, +4^\circ, +8^\circ, +12^\circ\}$$
โดยสร้างเคอร์เนลผ่านการแปลงพิกัดแบบหมุน:
$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x - x_c \\ y - y_c \end{bmatrix}$$
จากนั้นทำการ Binarize เพื่อให้ได้เคอร์เนลเส้นเอียงที่มีความกว้าง $W_k = 25\text{ px}$ และทำ Morphological Opening กับภาพเซลล์:
$$I_{\text{lines}} = \bigvee_{\theta} \left( I \circ K_\theta \right)$$
สูตรนี้ช่วยดึงเส้นขีดฆ่าที่เอียงขึ้นหรือเอียงลงได้ครบถ้วน $100\%$

### 2.2 สัณฐานวิทยาเชิงลึก: Body Penetration และ Two-Sided Wing Overhang
เพื่อคัดกรองตัวอักษร `F`, `Z`, `E` ออกจากเครื่องหมาย `=` ระบบสกัดฟีเจอร์สัณฐานวิทยา 2 ตัวสำคัญ:
1. **Interior Body Penetration:**
   เส้นขีดฆ่าจริงต้องพาดผ่านบริเวณใจกลางตัวอักษร:
   $$\text{rel\_y} = \frac{y_{\text{stroke}} - y_{\min}}{y_{\max} - y_{\min}} \in [0.14, 0.85]$$
   ต่างจากตัวอักษร `Z` ซึ่งเส้นนอนอยู่ขอบบนสุด ($\text{rel\_y} < 0.10$) และขอบล่างสุด ($\text{rel\_y} > 0.90$)
2. **Two-Sided Wing Overhang:**
   เส้นขีดฆ่าจริงจะมีความยาวที่ยื่นเลยขอบตัวอักษรออกมาทั้งสองฝั่ง:
   $$L_{\text{wing}} = x_{\min}^{\text{char}} - x_{\min}^{\text{stroke}} > 0, \quad R_{\text{wing}} = x_{\max}^{\text{stroke}} - x_{\max}^{\text{char}} > 0$$
   ในขณะที่ตัว `F` เส้นขีดกลางจะยื่นไปทางขวาข้างเดียว ($L_{\text{wing}} \le 0$)

### 2.3 สถาปัตยกรรม StrikethroughHybridNet (Dual-Branch Architecture)
โครงข่ายแบบสองแขนงผสานคุณลักษณะภาพและสถิติ:
* **Branch 1 (2D CNN Spatial Branch):** รับภาพขนาด $64 \times 64$ สกัด Spatial Features ผ่าน Conv2D 3 ชั้น (16 $\rightarrow$ 32 $\rightarrow$ 64 channels) พร้อม BatchNorm และ ReLU
* **Branch 2 (Morphological Stats Branch):** รับเวกเตอร์สถิติ 6 มิติ:
  $$\mathbf{f} = \left[ \text{Trans}_{\text{BW}}, \text{Gap}_{\text{HPP}}, \text{Ratio}_{\text{wing}}, \text{Ratio}_{\text{body}}, \text{Conf}_{\text{HCR}}, \text{Class}_{\text{HCR}} \right]$$
* **Fusion Layer:** รวมฟีเจอร์ทั้งสองเข้าสู่ Fully-Connected Layers 3 ชั้น (128 $\rightarrow$ 64 $\rightarrow$ 3 คลาส: `empty`, `normal_char`, `strikethrough`)

---

## 3. เหตุผลการจัดลำดับสถาปัตยกรรม: ทำไมรัน Phase 4 ก่อน Phase 3? (HCR-First Paradigm)

* **ปัญหาระดับคลาสสิก:** ตัวอักษร `F` และ `Z` เป็นจุดตายของระบบตรวจจับรอยขีดฆ่าแบบ Rule-based หรือ CNN เดี่ยวๆ เพราะเส้นในตัวอักษรมีลักษณะทางเรขาคณิตคล้ายเครื่องหมาย `=`
* **การค้นพบเชิงประจักษ์:** เมื่อเราสลับนำ Phase 4 (HCR) มารันก่อน:
  * ตัวอักษร `F`, `Z`, `E` ปกติ โมเดล HCR จะมีความมั่นใจสูงมาก ($\text{Conf}_{\text{HCR}} \in [0.88, 0.99]$)
  * ช่องที่มีการขีดฆ่าทับ ลายเส้นจะเละ ส่งผลให้โมเดล HCR สับสนและมีความมั่นใจต่ำมาก ($\text{Conf}_{\text{HCR}} < 0.35$)
* **ผลลัพธ์:** การนำค่า $\text{Conf}_{\text{HCR}}$ และ $\text{Class}_{\text{HCR}}$ เข้ามาร่วมเป็น Gating Factor ใน Phase 3 ทำให้ตัดปัญหา False Positive บนตัวอักษร `F` และ `Z` ได้อย่างเด็ดขาด ลดอัตรา **$\text{FPR} = 0.0\%$ อย่างสมบูรณ์**

---

## 4. จุดอ่อน ข้อจำกัดตามจริง และแนวทางการตอบคำถามกรรมการสอบ (Limitations & Defense Q&A)

### ข้อจำกัดตามจริง (System Limitations)
1. **การขีดฆ่าผิดระเบียบการสอบ (Non-standard Strikethrough Conventions):**
   ตามระเบียบของกระดาษคำตอบ Rayong MARCO กำหนดให้ **ขีดฆ่าด้วยเส้นคู่แนวนอน (=) เท่านั้น** หากนักเรียนขีดฆ่าผิดระเบียบ เช่น ใช้การกากบาททับตัวอักษรด้วยเครื่องหมาย `X` เดี่ยวๆ หรือขีดทแยงมุมเส้นเดียว `\` (เช่น กรณีภาพ `04_words_vocab_011.jpg` ข้อ 5) อัลกอริทึมจะไม่นับเป็นรอยขีดฆ่าที่ถูกต้อง
2. **รอยขีดฆ่าที่ใช้ดินสอหรือหมึกจางมาก:**
   หากนักเรียนใช้ดินสอจางขีดเบาๆ ค่า Body Penetration อาจไม่ผ่านเกณฑ์ Threshold

### แนวทางการตอบคำถามคณะกรรมการสอบ (Defense Q&A)
* **กรรมการสอบถาม:** *"ถ้ามีนักเรียนขีดฆ่าด้วยเส้นเดี่ยว (-) หรือกากบาท (X) ทำไมระบบถึงไม่ออกแบบให้รองรับ?"*
* **แนวทางตอบ:** *"ในทางสถิติและการประมวลผลสัญญาณภาพ หากเราผ่อนปรนให้เส้นเดี่ยว (-) นับเป็นรอยขีดฆ่า จะเกิดการสับสนกับขีดกลางของตัวอักษร `A`, `H`, `E`, `F` ทันที และการกากบาท `X` จะสับสนกับตัวอักษร `X` ตัวมันเองอย่างหลีกเลี่ยงไม่ได้ ระเบียบการสอบจึงกำหนดให้ใช้ **เส้นคู่ (=)** เพื่อให้เกิดสัญลักษณ์ทางเรขาคณิตเฉพาะ (Unique Discriminative Signature) ซึ่งระบบของเราออกแบบตามระเบียบข้อบังคับของการสอบจริง $100\%$ ครับ"*

---

## 5. ผลการประเมินเชิงปริมาณอย่างละเอียดบนชุดข้อมูลทดสอบ (Granular Test Metrics)

### 5.1 ตัวชี้วัดประสิทธิภาพบนชุดข้อมูล Validate และ Test (Unseen Sheets)

| ตัวชี้วัด (Metric) | Validate Set (2,440 ช่อง) | Test Set (2,542 ช่อง) |
| :--- | :---: | :---: |
| **Overall Accuracy** | **99.96%** | **99.88%** |
| **Macro Precision** | 99.66% | 99.59% |
| **Macro Recall** | 99.98% | 99.24% |
| **Macro F1-Score** | **99.82%** | **99.41%** |
| **Strikethrough ROC-AUC** | **100.00%** | **99.99%** |

### 5.2 ประสิทธิภาพรายคลาสบนชุดทดสอบ (Test Set Per-Class Breakdown)

| คลาส (Cell State) | Precision | Recall | F1-Score | จำนวนตัวอย่าง (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **`empty` (ช่องว่างเปล่า)** | 100.00% | 100.00% | 100.00% | 320 |
| **`normal_char` (ตัวอักษรปกติ A–Z)** | 99.91% | 99.95% | 99.93% | 2,133 |
| **`strikethrough` (ขีดฆ่า 2 เส้น `=`)** | 98.86% | 97.75% | **98.31%** | 89 |

### 5.3 Confusion Matrix บนชุดทดสอบ (Test Set - 2,542 ช่อง)

| Actual \\ Predicted | `empty` | `normal_char` | `strikethrough` |
| :--- | :---: | :---: | :---: |
| **`empty`** | **320** | 0 | 0 |
| **`normal_char`** | 0 | **2132** | 1 |
| **`strikethrough`** | 0 | 2 | **87** |

---

## 6. ผลการทดสอบ End-to-End บนกระดาษคำตอบจริง

| ใบกระดาษคำตอบ | ช่องตัวอักษรปกติ (`normal_char`) | ช่องขีดฆ่าที่ถูกตัดออก (`strikethrough`) | ช่องว่าง (`empty`) | เวลาประมวลผล Phase 3 (400 ช่อง) |
| :--- | :---: | :---: | :---: | :---: |
| `ตัวอย่างที่ 1.jpg` | 181 | **60** | 159 | 1339.2 ms |
| `ตัวอย่างการขีดฆ่า.jpg` | 71 | **56** | 273 | 1289.7 ms |
| `02_edge_strikethrough_005.jpg` | 183 | **29** | 188 | 1332.1 ms |
| `02_edge_strikethrough_020.jpg` | 136 | **29** | 235 | 1225.7 ms |
| `02_edge_strikethrough_022.jpg` | 130 | **30** | 240 | 1345.8 ms |