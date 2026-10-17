# ศูนย์รวมรายงานตัวชี้วัดเชิงปริมาณ (Quantitative Metrics Hub)
**โครงการ:** ระบบตรวจกระดาษคำตอบอัตโนมัติ Rayong MARCO (Template 2)

โฟลเดอร์ `metrics/` นี้ใช้สำหรับเก็บรวบรวม **ตัวชี้วัดที่เป็นตัวเลข (Quantitative Metrics)** ของการพัฒนาและทดสอบโมเดลในแต่ละ Phase เพื่อใช้อ้างอิงประสิทธิภาพและความแม่นยำอย่างเป็นระบบ

---

## สารบัญรายงานตัวชี้วัดแยกตาม Phase

| Phase | ชื่อรายงาน (Markdown Report) | ไฟล์ข้อมูลดิบ (JSON) | สถานะการประเมินผล | สรุปผลลัพธ์หลัก |
|:---:|:---|:---|:---:|:---|
| **Phase 1** | **[Phase1_Metrics_Report.md](./Phase1_Metrics_Report.md)** | [`phase1_evaluation_metrics.json`](./phase1_evaluation_metrics.json) | **เสร็จสมบูรณ์ (PASS)** | Detection **100%**, Orientation Acc **100%**, RMSE **0.114 px**, IoU **0.9873** |
| **Phase 2** | **[Phase2_Metrics_Report.md](./Phase2_Metrics_Report.md)** | [`phase2_evaluation_metrics.json`](./phase2_evaluation_metrics.json) | **เสร็จสมบูรณ์ (PASS)** | Timing Bar Recall **100%**, Answer Green **100% (20,400/20,400)**, Header Purple Align **100% (2,142/2,142)**, OMR Align **100% (357/357)** |
| **Phase 3** | **[Phase3_Metrics_Report.md](./Phase3_Metrics_Report.md)** | [`phase3_evaluation_metrics.json`](./phase3_evaluation_metrics.json) | **เสร็จสมบูรณ์ (PASS)** | Precision **100.0%**, Recall **100.0%** (F1 = **1.000**), FPR บนตัวอักษรปกติ `F`/`Z` **0.0%** |
| **Phase 4** | **[Phase4_Metrics_Report.md](./Phase4_Metrics_Report.md)** | [`phase4_evaluation_metrics.json`](./phase4_evaluation_metrics.json) | **เสร็จสมบูรณ์ (PASS)** | A–Z Val Top-1 **98.77%** (Top-3 **99.80%**), Real Exam Char Top-1 **100.0% (786/786)**, Word Match **100.0% (72/72)** |
| **Phase 4B** | **[Phase4B_Digit_Metrics_Report.md](./Phase4B_Digit_Metrics_Report.md)** | [`digit_recognition_metrics.json`](./digit_recognition_metrics.json) | **เสร็จสมบูรณ์ (PASS)** | Digit 0–9 Val Top-1 **99.77%**, Unseen Test Top-1 **99.58%** (Top-3 **99.95%**), Blank Purple Box Specificity **100.0% (798/798)**, Purple Box Digit Acc **100.0% (96/96)** |
| **Phase 5** | **[Phase5_Metrics_Report.md](./Phase5_Metrics_Report.md)** | [`phase5_evaluation_metrics.json`](./phase5_evaluation_metrics.json) | **เสร็จสมบูรณ์ (PASS)** | Exact Match Word Accuracy **99.21% (250/252 ข้อ)**, CER **0.10%**, Score Pearson $r$ = **0.9994**, Latency **~2.48 s/แผ่น** |
