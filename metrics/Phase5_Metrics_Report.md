# รายงานผลการประเมินเชิงปริมาณ Phase 5: การประกอบคำศัพท์และตรวจให้คะแนนข้อสอบอัตโนมัติ (End-to-End Automated Grading)

## 1. สรุปตัวชี้วัดประสิทธิภาพระดับระบบ (Overall End-to-End Grading Metrics)
- **ไปป์ไลน์การทำงาน**: `Phase 1 (4-Point Dewarping)` $\rightarrow$ `Phase 2 (Grid, Header 42 ช่อง & OMR 7 วง)` $\rightarrow$ `Phase 4 (HCR Pre-Scoring)` $\rightarrow$ `Phase 3 (HCR-Guided Double Strikethrough '=' Filter)` $\rightarrow$ `Phase 5 (Top-3 Lexicon Word Assembly & Automated Scoring)`
- **จำนวนกระดาษคำตอบที่ตรวจทั้งหมด**: **19 แผ่น** (`data/Test/` 5 แผ่น + `New-Dataset` Unseen Test Split 14 แผ่น)
- **จำนวนข้อที่มีคำตอบพร้อมเฉลยรวม**: **252 ข้อ (2,921 ตัวอักษร)**

| ตัวชี้วัดเชิงปริมาณ (Metric) | เกณฑ์เป้าหมาย (Target) | ผลลัพธ์ที่ได้จริง (Actual) | สถานะ |
| :--- | :---: | :---: | :---: |
| **Exact Match Word Accuracy (EMA)** | $\ge 90.0\%$ | **99.21% (250/252 ข้อ)** | ผ่านเกณฑ์ดีเยี่ยม |
| **Character Error Rate (CER)** | $\le 4.0\%$ | **0.10%** | ผ่านเกณฑ์ดีเยี่ยม |
| **Score Correlation (Pearson $r$)** | $\ge 0.9800$ | **0.9994** | ผ่านเกณฑ์ดีเยี่ยม |
| **End-to-End Grading Latency** | $\le 3,500\text{ ms}$ | **2478.6 ms / แผ่น** | ผ่านเกณฑ์ดีเยี่ยม |

## 2. ตารางสรุปผลการตรวจให้คะแนนรายแผ่น (Per-Sheet Grading Breakdown)

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