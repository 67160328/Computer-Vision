# ศูนย์รวมรายงานตัวชี้วัดเชิงปริมาณ (Quantitative Metrics Hub)
**โครงการ:** ระบบตรวจกระดาษคำตอบอัตโนมัติ Rayong MARCO (Template 2)

โฟลเดอร์ `metrics/` นี้ใช้สำหรับเก็บรวบรวม **ตัวชี้วัดที่เป็นตัวเลข (Quantitative Metrics)** ของการพัฒนาและทดสอบโมเดลในแต่ละ Phase เพื่อใช้อ้างอิงประสิทธิภาพและความแม่นยำอย่างเป็นระบบ

---

## สารบัญรายงานตัวชี้วัดแยกตาม Phase

| Phase | ชื่อรายงาน (Markdown Report) | ไฟล์ข้อมูลดิบ (JSON) | สถานะการประเมินผล | สรุปผลลัพธ์หลัก |
|:---:|:---|:---|:---:|:---|
| **Phase 1** | **[Phase1_Metrics_Report.md](./Phase1_Metrics_Report.md)** | [`phase1_evaluation_metrics.json`](./phase1_evaluation_metrics.json) | **เสร็จสมบูรณ์ (PASS)** | Detection **100%**, Orientation Acc **100%**, RMSE **0.114 px**, IoU **0.9873** |
| **Phase 2** | **[Phase2_Metrics_Report.md](./Phase2_Metrics_Report.md)** | [`phase2_evaluation_metrics.json`](./phase2_evaluation_metrics.json) | **เสร็จสมบูรณ์ (PASS)** | Timing Bar Recall **100%**, Answer Green **100% (20,400/20,400)**, Header Purple Align **100% (2,142/2,142)**, OMR Align **100% (357/357)** |
| **Phase 3** | `Phase3_Metrics_Report.md` | `phase3_evaluation_metrics.json` | *รอคำยืนยันเริ่มดำเนินการ* | Double Strikethrough (`=`) Accuracy, F1-Score, False Positive Rate (FPR) |
| **Phase 4** | **[Phase4_Metrics_Report.md](./Phase4_Metrics_Report.md)** | [`phase4_evaluation_metrics.json`](./phase4_evaluation_metrics.json) | **เสร็จสมบูรณ์ (PASS)** | A–Z Val Top-1 **98.77%** (Top-3 **99.80%**), Real Exam Char Top-1 **100.0% (786/786)**, Word Match **100.0% (72/72)** |
| **Phase 5** | `Phase5_Metrics_Report.md` | `phase5_evaluation_metrics.json` | *รอคำยืนยันเริ่มดำเนินการ* | End-to-End Exact Word Match (EMA), Character Error Rate (CER), Pearson $r$ |
