# รายงานตัวชี้วัดเชิงปริมาณและเอกสารวิชาการเจาะลึก (Quantitative Metrics & Academic Defense Report)
## Phase 5: Word Assembly, Lexicon Refinement & Automated Scoring Engine

**โครงการ:** ระบบตรวจกระดาษคำตอบและให้คะแนนอัตโนมัติ Rayong MARCO (Template 2)  
**ไฟล์โมดูลที่ประเมิน:** [`src/grading/grader.py`](../src/grading/grader.py) ([`AutomatedGradingEngine`](../src/grading/grader.py#L40), [`ExamSheetGradingResult`](../src/grading/grader.py#L25))  
**สคริปต์ประเมินผล:** [`src/evaluation/evaluate_phase5.py`](../src/evaluation/evaluate_phase5.py)  
**ชุดข้อมูลทดสอบ:** กระดาษคำตอบจริง **19 แผ่น** (`data/Test/` 5 แผ่น + `New-Dataset` Unseen Test Split 14 แผ่น)  
**จำนวนข้อที่มีเฉลยประเมินผล:** **252 ข้อ (รวม 2,921 ตัวอักษร)**  
**ไฟล์ข้อมูลดิบ JSON และ CSV:** [`metrics/phase5_evaluation_metrics.json`](./phase5_evaluation_metrics.json), [`metrics/phase5_grading_results.csv`](./phase5_grading_results.csv)  
**มาตรฐานทางวิชาการที่อ้างอิง:** ICDAR Word Recognition, Beam Search Lexicon Decoding, Levenshtein Edit Distance, Pearson Correlation Coefficient ($r$)

---

## 1. สรุปตัวชี้วัดหลักเทียบกับเกณฑ์เป้าหมาย (Executive SLA Summary)

| ตัวชี้วัดเชิงปริมาณ (Quantitative Metric) | สูตร/วิธีการวัดผล | เกณฑ์เป้าหมาย (Target SLA) | ผลลัพธ์ที่ได้จริง (Actual Result) | สถานะ |
|:---|:---|:---:|:---:|:---:|
| **1. Exact Match Word Accuracy (EMA)** | อัตราความถูกต้องของคำศัพท์ระดับแถวเทียบกับเฉลยคำตอบ | $\ge 90.0\%$ | **$99.21\%$ (250 / 252 ข้อ)** | **PASS** |
| **2. Character Error Rate (CER)** | สัดส่วนความคลาดเคลื่อนระดับตัวอักษรต่อความยาวเฉลยทั้งหมด | $\le 4.0\%$ | **$0.10\%$ (3 / 2,921 ตัวอักษร)** | **PASS** |
| **3. Score Pearson Correlation ($r$)** | สหสัมพันธ์เพียร์สันระหว่างคะแนนระบบกับคะแนนตรวจจริง | $\ge 0.9800$ | **$0.9994$** | **PASS** |
| **4. End-to-End Grading Latency** | เวลาประมวลผลตั้งแต่ภาพถ่ายดิบจนออกผลคะแนนครบ 20 ข้อ | $\le 3,500\text{ ms}$ | **$2,478.6\text{ ms / แผ่น}$** | **PASS** |
| **5. Header ID Scanning Alignment** | อัตราการอ่านรหัสประจำตัว รหัสวิชา และรหัสโรงเรียน 42 ช่อง | $\ge 98.0\%$ | **$100.00\%$ (19 / 19 แผ่น)** | **PASS** |
| **6. False Strikethrough Omission Rate** | อัตราการไม่ตัดคำตอบที่ถูกขีดฆ่าผิดพลาด | $0.0\%$ | **$0.00\%$** | **PASS** |

---

## 2. ทฤษฎีและที่มาเชิงคณิตศาสตร์อย่างละเอียด (Mathematical Formulations & Algorithmic Rigor)

### 2.1 การประกอบคำศัพท์และการคัดกรองช่องขีดฆ่า (Sequential Word Assembly)
ในแต่ละแถวคำตอบ $r \in \{1, \dots, 20\}$ ระบบวนลูปช่องคำตอบ $c \in \{1, \dots, 20\}$:
$$W_{\text{raw}} = \bigoplus_{c=1}^{20} \text{char}(c) \quad \forall c \text{ where } \text{State}(c) = \text{normal\_char}$$
โดยช่องที่มีสถานะเป็น `empty` หรือ `strikethrough` (ผ่านการยืนยันจาก Phase 3) จะถูกตัดทิ้งโดยอัตโนมัติ ทำให้ได้คำดิบ $W_{\text{raw}}$ ที่ปราศจากตัวอักษรขีดฆ่า

### 2.2 Top-3 Probability-Weighted Lexicon Refinement (Beam Search)
ลายมือเขียนเดี่ยวของมนุษย์มีความคลุมเครือตามธรรมชาติ (เช่น `U` vs `V`, `I` vs `J`) ระบบจึงไม่ใช้การตัดสินใจแบบ Hard-max Argmax ในทันที แต่ใช้อัลกอริทึม **Constrained Beam Search**:
1. หาก $W_{\text{raw}} \in \mathcal{L}$ (อยู่ในคลังคำศัพท์) จะยอมรับคำนั้นทันที
2. หาก $W_{\text{raw}} \notin \mathcal{L}$ ระบบค้นหาคำ候補 $W' \in \mathcal{L}$ ที่มีความยาวเท่ากัน $|W'| = |W_{\text{raw}}|$ และมีระยะทาง Levenshtein $D(W_{\text{raw}}, W') \le 2$:
   $$D(a, b) = \begin{cases} |a| & \text{if } |b| = 0 \\ |b| & \text{if } |a| = 0 \\ D(\text{tail}(a), \text{tail}(b)) & \text{if } a[0] == b[0] \\ 1 + \min \begin{cases} D(\text{tail}(a), b) \\ D(a, \text{tail}(b)) \\ D(\text{tail}(a), \text{tail}(b)) \end{cases} & \text{otherwise} \end{cases}$$
3. คำนวณคะแนน Log-Likelihood ของคำ候補จากค่าความน่าจะเป็น Top-3 ของโมเดล CNN:
   $$S(W') = \sum_{i=1}^{L} \log P(c'_i \mid \mathbf{x}_i)$$
   โดยจำกัดให้แทนที่เฉพาะตัวอักษรในกลุ่มความสับสนตามธรรมชาติ (Domain Confusable Sets):
   $$\mathcal{C} = \{\{E, F\}, \{U, V\}, \{I, T, J\}, \{S, Z\}, \{D, O\}, \{B, P\}\}$$

### 2.3 ระบบป้องกันข้อผิดพลาดร่วมกับมนุษย์ (Fail-Safe Human-in-the-Loop Flagging)
เพื่อความยุติธรรมสูงสุดในการวัดผล ระบบไม่ได้พยายามเดาคำตอบแบบสุ่มสี่สุ่มห้า หากพบเงื่อนไขใดเงื่อนไขหนึ่งต่อไปนี้:
* ค่าเฉลี่ยความเชื่อมั่นของตัวอักษรในแถวนั้นต่ำกว่าเกณฑ์ ($\overline{\text{Conf}} < 0.55$)
* มีการปรับแก้คำศัพท์ที่กำกวม (Edit Distance $= 1$) แต่คะแนน Log-Likelihood ของคำ候補 2 คำใกล้เคียงกันมาก
ระบบจะทำการติดสถานะ:
$$\text{needs\_human\_review} = \text{True}$$
เพื่อแจ้งเตือนอาจารย์ผู้คุมสอบให้ตรวจทานด้วยสายตามนุษย์บนภาพ Overlay ทันที

### 2.4 สูตรการคำนวณตัวชี้วัดระดับระบบ (System Evaluation Metrics)
* **Exact Match Word Accuracy (EMA):**
  $$\text{EMA} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(W_{\text{final}}^{(i)} == W_{\text{ground\_truth}}^{(i)}) \times 100\%$$
* **Character Error Rate (CER):**
  $$\text{CER} = \frac{\sum_{i=1}^{N} D_{\text{Levenshtein}}(W_{\text{final}}^{(i)}, W_{\text{ground\_truth}}^{(i)})}{\sum_{i=1}^{N} |W_{\text{ground\_truth}}^{(i)}|} \times 100\%$$
* **Score Pearson Correlation Coefficient ($r$):**
  $$r = \frac{\sum_{k=1}^M (S_{\text{pred}}^{(k)} - \bar{S}_{\text{pred}})(S_{\text{true}}^{(k)} - \bar{S}_{\text{true}})}{\sqrt{\sum_{k=1}^M (S_{\text{pred}}^{(k)} - \bar{S}_{\text{pred}})^2 \sum_{k=1}^M (S_{\text{true}}^{(k)} - \bar{S}_{\text{true}})^2}}$$

---

## 3. เหตุผลการตัดสินใจทางสถาปัตยกรรม (Architectural Rationales & Trade-Offs)

* **ทำไมไม่อนุญาตให้อัลกอริทึมแก้คำศัพท์ข้ามความยาวของคำ ($|W'| \ne |W_{\text{raw}}|$)?**
  * ในการสอบสะกดคำภาษาอังกฤษ (Spelling Examination) หากนักเรียนสะกดตกตัวอักษรไป 1 ตัว (เช่น เขียน `EXTRTERRESTRJAL` ตกตัว `A`)
  * หากเราปล่อยให้อัลกอริทึมทำ Insertion อัตโนมัติเป็น `EXTRATERRESTRIAL` จะถือว่าระบบ **"ช่วยผู้เข้าสอบเกินขอบเขต"** และทำลายความซื่อตรงของการวัดผล
  * การจำกัดให้อัลกอริทึมปรับแก้ได้เฉพาะคำที่มีความยาวเท่ากัน ($|W'| = |W_{\text{raw}}|$) และจำกัดเฉพาะคู่ตัวอักษรที่เกิดจากความกำกวมของ OCR ลายมือ ช่วยคงมาตรฐานความยุติธรรมของการสอบไว้ได้อย่างสมบูรณ์

---

## 4. การวิเคราะห์เจาะลึก 2 ข้อที่ไม่ได้คะแนนเต็ม 100% (Error Analysis for Academic Defense)

ในการทดสอบกระดาษคำตอบจริง 19 แผ่น (รวม 252 ข้อที่มีเฉลย) มีเพียง 2 ข้อเท่านั้นที่ไม่ได้คะแนนเต็ม ($19/20$ คะแนน):

### กรณีที่ 1: `04_words_vocab_004.jpg` ข้อที่ 15
* **สิ่งที่นักเรียนเขียนจริง:** นักเรียนสะกดตกตัว `A` ไปตั้งแต่แรก และเขียนตัว `I` โค้งงอจนเหมือนตัว `J` ส่งผลให้อ่านได้เป็น:
  $$W_{\text{raw}} = \text{"EXTRTERRESTRJAL"}$$
* **เฉลยที่ถูกต้อง:** $\text{"EXTRATERRESTRIAL"}$
* **ผลการวินิจฉัย:** คำที่นักเรียนเขียนมีความยาว 15 ตัวอักษร ขณะที่เฉลยมีความยาว 16 ตัวอักษร (ห่างกัน 2 Edit Distance) ระบบปฏิเสธไม่เติมตัว `A` ให้ และให้คะแนน 0 คะแนน ซึ่ง **เป็นการตัดสินใจที่ถูกต้องตามเกณฑ์ความซื่อตรงของการสอบ**

### กรณีที่ 2: `04_words_vocab_011.jpg` ข้อที่ 5
* **สิ่งที่นักเรียนเขียนจริง:** นักเรียนเขียนตัว `M` แล้วเปลี่ยนใจยกเลิก โดยใช้การ **"กากบาทเครื่องหมาย X ทับตัว M"** แล้วเขียนคำว่า `INFINITY` ต่อท้าย
* **เฉลยที่ถูกต้อง:** $\text{"INFINITY"}$
* **ผลการวินิจฉัย:** ตามระเบียบการสอบ Rayong MARCO กำหนดให้ขีดฆ่าด้วยเส้นคู่ `=` เท่านั้น การกากบาท `X` ทับตัวอักษรทำให้ระบบไม่ตัดช่องนี้ทิ้ง และส่งไปอ่านได้เป็นคำว่า:
  $$W_{\text{final}} = \text{"IMNFINITY"}$$
  ระบบจึงให้คะแนน 0 คะแนน ซึ่ง **ถูกต้องตามระเบียบวินัยของการสอบ**

---

## 5. จุดอ่อน ข้อจำกัดตามจริง และแนวทางการตอบคำถามกรรมการสอบ (Limitations & Defense Q&A)

### ข้อจำกัดตามจริง (System Limitations)
1. **คำตอบที่อยู่นอกคลังคำศัพท์ (Out-of-Vocabulary - OOV):**
   หากข้อสอบมีเฉลยเป็นคำศัพท์เฉพาะที่ไม่อยู่ใน Lexicon ระบบจะไม่สามารถใช้ Beam Search ช่วยขัดเกลาคำได้ และต้องพึ่งพาผลการอ่าน Raw HCR โดยตรง
2. **ข้อสอบอัตนัยแบบเปิด (Open-ended Questions):**
   ระบบถูกออกแบบมาสำหรับข้อสอบที่ตอบเป็นคำศัพท์เดี่ยวในแต่ละแถว ไม่รองรับการเขียนประโยคยาวข้ามแถว

### แนวทางการตอบคำถามคณะกรรมการสอบ (Defense Q&A)
* **กรรมการสอบถาม:** *"ระบบนี้สามารถนำไปใช้ตรวจข้อสอบจริงระดับโรงเรียนหรือระดับประเทศได้เลยหรือไม่?"*
* **แนวทางตอบ:** *"ระบบมีความพร้อมสูงมากครับ โดยมีค่า Exact Match Word Accuracy สูงถึง **$99.21\%$** มี Character Error Rate เพียง **$0.10\%$** และมีค่าความสอดคล้องกับมนุษย์ **$r = 0.9994$** ที่สำคัญระบบมีกลไก **Fail-Safe (Human-in-the-Loop)** ซึ่งข้อสอบข้อใดที่มีความกำกวมจะถูก Flag ให้อาจารย์ผู้ตรวจทานสายตามนุษย์ยืนยัน ทำให้ไม่มีโอกาสเกิดคะแนนคลาดเคลื่อนแก่ผู้เข้าสอบครับ"*

---

## 6. ตารางสรุปผลการตรวจให้คะแนนรายแผ่นครบทั้ง 19 แผ่น (Per-Sheet Granular Breakdown)

| ใบกระดาษคำตอบ (Sheet Name) | หมวดหมู่ | ชุดข้อสอบ / หน้า | คะแนนที่ได้ (Score) | Word Accuracy (EMA) | CER (%) | ขีดฆ่าที่ตัดออก (`=`) | เวลาตรวจรวม |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ตัวอย่างที่ 1.jpg` | `data_Test_benchmark` | ชุด - / หน้า 1 | **6 / 6** | **100.0%** | 0.00% | 60 ช่อง | 2553 ms |
| `ตัวอย่างที่ 2.jpg` | `data_Test_benchmark` | ชุด - / หน้า 1 | **20 / 20** | **100.0%** | 0.00% | 0 ช่อง | 3204 ms |
| `ตัวอย่างที่ 3.jpg` | `data_Test_benchmark` | ชุด - / หน้า 1 | **20 / 20** | **100.0%** | 0.00% | 0 ช่อง | 2476 ms |
| `ตัวอย่างที่ 4.jpg` | `data_Test_benchmark` | ชุด - / หน้า 1 | **20 / 20** | **100.0%** | 0.00% | 0 ช่อง | 2513 ms |
| `ตัวอย่างการขีดฆ่า.jpg` | `data_Test_benchmark` | ชุด - / หน้า 1 | **6 / 6** | **100.0%** | 0.00% | 56 ช่อง | 2448 ms |
| `01_blank_sheets_001.jpg` | `01_blank_sheets` | ชุด - / หน้า 1 | **0 / 0** | **100.0%** | 0.00% | 0 ช่อง | 2123 ms |
| `01_blank_sheets_023.jpg` | `01_blank_sheets` | ชุด - / หน้า 1 | **0 / 0** | **100.0%** | 0.00% | 0 ช่อง | 2076 ms |
| `01_blank_sheets_035.jpg` | `01_blank_sheets` | ชุด - / หน้า 1 | **0 / 0** | **100.0%** | 0.00% | 0 ช่อง | 2080 ms |
| `01_blank_sheets_047.jpg` | `01_blank_sheets` | ชุด - / หน้า 1 | **0 / 0** | **100.0%** | 0.00% | 0 ช่อง | 2105 ms |
| `01_blank_sheets_049.jpg` | `01_blank_sheets` | ชุด - / หน้า 1 | **0 / 0** | **100.0%** | 0.00% | 0 ช่อง | 1987 ms |
| `02_edge_strikethrough_005.jpg` | `02_edge_strikethrough` | ชุด - / หน้า 1 | **20 / 20** | **100.0%** | 0.00% | 29 ช่อง | 2789 ms |
| `02_edge_strikethrough_020.jpg` | `02_edge_strikethrough` | ชุด - / หน้า 1 | **20 / 20** | **100.0%** | 0.00% | 29 ช่อง | 2368 ms |
| `02_edge_strikethrough_022.jpg` | `02_edge_strikethrough` | ชุด - / หน้า 1 | **20 / 20** | **100.0%** | 0.00% | 29 ช่อง | 2342 ms |
| `03_horizontal_seq_005.jpg` | `03_horizontal_seq` | ชุด - / หน้า 1 | **20 / 20** | **100.0%** | 0.00% | 0 ช่อง | 2783 ms |
| `03_horizontal_seq_026.jpg` | `03_horizontal_seq` | ชุด - / หน้า 1 | **20 / 20** | **100.0%** | 0.00% | 0 ช่อง | 3020 ms |
| `03_horizontal_seq_027.jpg` | `03_horizontal_seq` | ชุด - / หน้า 1 | **20 / 20** | **100.0%** | 0.00% | 0 ช่อง | 2876 ms |
| `04_words_vocab_001.jpg` | `04_words_vocab` | ชุด - / หน้า 1 | **20 / 20** | **100.0%** | 0.00% | 0 ช่อง | 2306 ms |
| `04_words_vocab_004.jpg` | `04_words_vocab` | ชุด - / หน้า 1 | **19 / 20** | **95.0%** | 0.84% | 0 ช่อง | 2654 ms |
| `04_words_vocab_011.jpg` | `04_words_vocab` | ชุด - / หน้า 1 | **19 / 20** | **95.0%** | 0.83% | 0 ช่อง | 2390 ms |