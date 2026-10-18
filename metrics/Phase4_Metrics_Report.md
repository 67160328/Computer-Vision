# รายงานตัวชี้วัดเชิงปริมาณและเอกสารวิชาการเจาะลึก (Quantitative Metrics & Academic Defense Report)
## Phase 4: Handwritten Character Recognition (HCR CNN Engine)

**โครงการ:** ระบบตรวจกระดาษคำตอบและให้คะแนนอัตโนมัติ Rayong MARCO (Template 2)  
**ไฟล์โมดูลที่ประเมิน:** [`src/recognition/char_classifier.py`](../src/recognition/char_classifier.py) (`HCRModelEngine`, [`HCRResNet28`](../src/recognition/char_classifier.py#L12))  
**สคริปต์การเทรน:** [`src/recognition/train_hcr.py`](../src/recognition/train_hcr.py)  
**สคริปต์ประเมินผล:** [`src/evaluation/evaluate_phase4.py`](../src/evaluation/evaluate_phase4.py)  
**ชุดข้อมูลที่ใช้ประเมิน:** ชุดข้อมูลตัวอักษร A–Z Validation (7,800 ภาพ) + กระดาษคำตอบลายมือจริง 5 แผ่น (786 ตัวอักษร, 72 แถวคำตอบ)  
**ไฟล์ข้อมูลดิบ JSON:** [`metrics/phase4_evaluation_metrics.json`](./phase4_evaluation_metrics.json)  
**โมเดลที่บันทึก:** `models/hcr_resnet28_best.pt`  
**มาตรฐานทางวิชาการที่อ้างอิง:** Deep Residual Learning (CVPR 2016), NIST / EMNIST Standardization, Calibrated Softmax Uncertainty (ICML 2017)

---

## 1. สรุปตัวชี้วัดหลักเทียบกับเกณฑ์เป้าหมาย (Executive SLA Summary)

| ตัวชี้วัดเชิงปริมาณ (Quantitative Metric) | สูตร/วิธีการวัดผล | เกณฑ์เป้าหมาย (Target SLA) | ผลลัพธ์ที่ได้จริง (Actual Result) | สถานะ |
|:---|:---|:---:|:---:|:---:|
| **1. Validation Top-1 Accuracy** | ความแม่นยำในการทำนายคลาสตัวอักษรเดี่ยวอันดับ 1 (7,800 ภาพ) | $\ge 95.0\%$ | **$98.77\%$** | **PASS** |
| **2. Validation Top-3 Accuracy** | ความน่าจะเป็นที่คลาสเฉลยติดอยู่ใน 3 อันดับแรก | $\ge 99.0\%$ | **$99.80\%$** | **PASS** |
| **3. Validation Macro-F1 Score** | ค่า F1-Score เฉลี่ยแบบไม่ถ่วงน้ำหนักครบทั้ง 26 คลาส A–Z | $\ge 0.9500$ | **$0.9879$** | **PASS** |
| **4. Real Exam Character Accuracy** | ความแม่นยำบนตัวอักษรลายมือจริงจากกระดาษสอบ 5 แผ่น (786 ตัว) | $\ge 98.0\%$ | **$100.00\%$ (786 / 786 ตัว)** | **PASS** |
| **5. Real Exam Exact Word Match** | อัตราความถูกต้องระดับแถว/คำศัพท์ (72 แถวคำตอบจริง) | $\ge 95.0\%$ | **$100.00\%$ (72 / 72 แถว)** | **PASS** |
| **6. Mean Prediction Confidence** | ค่าเฉลี่ยความเชื่อมั่น Softmax บนตัวอักษรจริง | $\ge 0.8500$ | **$0.9096$** | **PASS** |
| **7. Row-Batch Inference Latency (CPU)** | เวลาประมวลผลต่อแถวคำตอบ (20 ช่อง) บนซีพียูธรรมดา | $\le 50.0\text{ ms}$ | **$19.14\text{ ms / แถว}$** ($\approx 0.96\text{ ms / ตัว}$) | **PASS** |

---

## 2. ทฤษฎีและที่มาเชิงคณิตศาสตร์อย่างละเอียด (Mathematical Formulations & Algorithmic Rigor)

### 2.1 Canonical Glyph Normalization ด้วย Image Moments (การปรับมาตรฐานอักขระ)
ภาพช่องคำตอบที่ถูกตัดแยกใน Phase 2 มีขนาดประมาณ $70 \times 80\text{ px}$ ซึ่งตำแหน่งลายมือมักไม่อยู่กึ่งกลาง และมีความหนาของเส้นหมึกแตกต่างกัน ระบบแปลงภาพสู่รูปแบบ Canonical NIST $28 \times 28\text{ px}$ ผ่านขั้นตอน:
1. **Foreground Bounding Box:** สกัดพื้นที่ลายมือจริง (Tight Bounding Box ขนาด $w \times h$)
2. **Aspect-Preserving Resize:** ขยาย/ย่อให้ด้านที่ยาวที่สุดมีขนาดพอดี $20\text{ px}$ และวางบนแคนวาสขนาด $28 \times 28\text{ px}$
3. **การคำนวณจุดศูนย์ถ่วงมวล (Center of Mass) ด้วย Image Moments:**
   $$M_{pq} = \sum_{x=0}^{27} \sum_{y=0}^{27} x^p y^q I(x, y)$$
   $$\bar{x} = \frac{M_{10}}{M_{00}}, \quad \bar{y} = \frac{M_{01}}{M_{00}}$$
   จากนั้นทำ Affine Translation เลื่อนจุดศูนย์ถ่วงมวล $(\bar{x}, \bar{y})$ ไปยังกึ่งกลางแคนวาส $(13.5, 13.5)$ อย่างแม่นยำ ขจัดปัญหาตัวอักษรชิดซ้ายหรือชิดขวาของช่อง

### 2.2 สถาปัตยกรรม Lightweight ResNet-28 Backbone
โครงข่ายถูกออกแบบให้มีขนาดกระทัดรัด (เพียง ~1.2M Parameters) เพื่อให้ทำงานได้รวดเร็วบน CPU:
* **Stem Layer:** Conv2D $3 \times 3$, Stride 1, Padding 1 ขยายช่องสัญญาณเป็น 32 Channels ตามด้วย BatchNorm และฟังก์ชันกระตุ้น **GELU (Gaussian Error Linear Unit)**:
  $$\text{GELU}(x) = x \Phi(x) = x P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$
* **Residual Stages (3 ระดับ):**
  - Stage 1: 2 Residual Blocks (32 $\rightarrow$ 64 Channels)
  - Stage 2: 2 Residual Blocks (64 $\rightarrow$ 128 Channels)
  - Stage 3: 2 Residual Blocks (128 $\rightarrow$ 192 Channels)
  โดยในแต่ละบล็อกมี Skip Connection เพื่อป้องกันปัญหา Vanishing Gradient:
  $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathcal{W}_s \mathbf{x}$$
* **Head Layer:** Adaptive Average Pooling $1 \times 1$ ตามด้วย Dropout ($p = 0.20$) และ Linear Classifier 26 คลาส (A–Z)

### 2.3 การสอบเทียบความน่าจะเป็นและการวัดค่าความไม่แน่นอน (Calibrated Softmax & Shannon Entropy)
เพื่อไม่ให้โมเดล Overconfident จนเกินไป ระบบใช้ Temperature Scaling $T = 1.15$:
$$P(y = c \mid \mathbf{x}) = \frac{\exp(z_c / T)}{\sum_{j=1}^{26} \exp(z_j / T)}$$
และคำนวณ Shannon Entropy $H(P)$ เพื่อใช้เป็นตัวชี้วัดความไม่แน่นอนของลายมือ:
$$H(P) = -\sum_{c=1}^{26} P_c \log_2 P_c$$
หากค่า $H(P) > 1.8$ หรือความเชื่อมั่น Top-1 $< 0.55$ ระบบจะส่งค่าไปเป็นฟีเจอร์นำทางใน Phase 3 และ Phase 5 ต่อไป

### 2.4 กลยุทธ์การฝึกสอนโมเดล (Training Recipe)
* **Loss Function:** Cross-Entropy พร้อม Label Smoothing ($\epsilon = 0.05$):
  $$\mathcal{L} = -(1 - \epsilon) \log P(y_{\text{true}}) - \frac{\epsilon}{K} \sum_{k=1}^K \log P(k)$$
* **Optimizer:** AdamW (Learning Rate $= 10^{-3}$, Weight Decay $= 10^{-4}$) ร่วมกับ Cosine Annealing LR Schedule
* **Data Augmentation:** สุ่มการหมุน ($\pm 10^\circ$), การเฉือน (Shear $\pm 8^\circ$), Perspective Jittering, และ Morphological Erosion/Dilation จำลองความหนาบางของหัวปากกา

---

## 3. เหตุผลการตัดสินใจทางสถาปัตยกรรม (Architectural Rationales & Trade-Offs)

* **ทำไมเลือก ResNet-28 แทนที่จะเป็น Vision Transformer (ViT) หรือโมเดลขนาดใหญ่?**
  1. **ขนาดของมิติข้อมูล:** ภาพตัวอักษรเดี่ยว $28 \times 28$ พิกเซล มีมิติขนาดเล็ก โครงข่าย Convolution มี Inductive Bias ที่เหมาะสมกับสัญญาณภาพขนาดเล็กมากกว่า Attention Mechanism ของ ViT
  2. **ประสิทธิภาพในการตรวจจริง (Inference Latency):** การประมวลผลข้อสอบทั้งแผ่น (400 ช่อง) ด้วย ResNet-28 ใช้เวลาเพียง **~380 ms บน CPU** ขณะที่ ViT ต้องการเวลามากกว่า 2.5 วินาที ทำให้ระบบสามารถทำงานได้บนคอมพิวเตอร์ทั่วไปของโรงเรียนโดยไม่จำเป็นต้องมีการ์ดจอ GPU

---

## 4. จุดอ่อน ข้อจำกัดตามจริง และแนวทางการตอบคำถามกรรมการสอบ (Limitations & Defense Q&A)

### ข้อจำกัดตามจริง (System Limitations)
1. **ความคลุมเครือทางสรีรวิทยาของลายมือเดี่ยว (Intrinsic Ambiguity):**
   ลายมือเขียนของมนุษย์บางตัวอักษรมีลักษณะใกล้เคียงกันมาก เช่น `U` กับ `V` เมื่อเขียนโค้งตรงก้น, `I` กับ `J` เมื่อเขียนไม่มีหางงอ, หรือ `D` กับ `O` เมื่อเขียนปิดไม่สนิท หากตัดภาพมาดูเดี่ยวๆ โดยไม่มีบริบทของคำศัพท์ แม้แต่มนุษย์เองก็อาจแยกไม่ออก
2. **การเขียนทับซ้ำๆ (Overwriting):**
   หากผู้สอบเขียนตัวอักษรหนึ่งแล้วเปลี่ยนใจใช้ปากกาเขียนทับตัวเดิมโดยไม่ขีดฆ่า เส้นหมึกที่ซ้อนกันจะทำให้โมเดลได้ค่าความเชื่อมั่นต่ำมาก

### แนวทางการตอบคำถามคณะกรรมการสอบ (Defense Q&A)
* **กรรมการสอบถาม:** *"ถ้าตัวอักษรโดดๆ อ่านยาก เช่น แยกไม่ออกว่าเป็น U หรือ V โมเดลแก้ปัญหานี้อย่างไร?"*
* **แนวทางตอบ:** *"ระบบของเราไม่ได้ตัดสินผลคะแนนเด็ดขาดใน Phase 4 ครับ แต่ Phase 4 ทำหน้าที่เป็น **Soft Decision Engine** ที่ส่งออกค่าความน่าจะเป็น Top-3 (เช่น ให้ U = 52%, V = 46%) ส่งต่อไปยัง Phase 5 ซึ่งมี **Lexicon Beam Search** ตรวจสอบกับคลังคำศัพท์ภาษาอังกฤษ หากคำศัพท์ในบริบทนั้นคือ 'UPDATE' ระบบจะเลือกตัว U ให้โดยอัตโนมัติด้วยคะแนน Log-Likelihood รวมของคำศัพท์ครับ"*

---

## 5. ผลการอ่านตัวอักษรแยกตามแผ่นกระดาษทดสอบจริง (`data/Test/`)

| แผ่นกระดาษทดสอบ | จำนวนตัวอักษรจริง | อ่านถูก (Top-1) | Top-1 Accuracy | Top-3 Accuracy | อ่านถูกระดับคำ/แถว (Exact Match) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **ตัวอย่างที่ 1.jpg** | 120 ตัว | 120 ตัว | **100.0%** | **100.0%** | **6/6 (100.0%)** |
| **ตัวอย่างที่ 2.jpg** | 400 ตัว | 400 ตัว | **100.0%** | **100.0%** | **20/20 (100.0%)** |
| **ตัวอย่างที่ 3.jpg** | 130 ตัว | 130 ตัว | **100.0%** | **100.0%** | **20/20 (100.0%)** |
| **ตัวอย่างที่ 4.jpg** | 130 ตัว | 130 ตัว | **100.0%** | **100.0%** | **20/20 (100.0%)** |
| **ตัวอย่างการขีดฆ่า.jpg** | 6 ตัว | 6 ตัว | **100.0%** | **100.0%** | **6/6 (100.0%)** |

---

## 6. ตารางเปรียบเทียบคำศัพท์ที่โมเดลอ่านได้จากกระดาษข้อสอบจริง (ตัวอย่างที่ 3 และ ตัวอย่างที่ 4)

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
