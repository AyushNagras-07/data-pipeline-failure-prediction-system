# Project Documentation (`docs/`)

This directory contains the formal academic documentation, technical reports, architecture guides, and publication-grade artifacts for the **Data Pipeline Failure Prediction System**.

---

## 1. Documentation Index

| File / Directory | Purpose & Description |
|:---|:---|
| **[`Data_Pipeline_Failure_Prediction_System_Technical_Report.pdf`](Data_Pipeline_Failure_Prediction_System_Technical_Report.pdf)** | **20-Page Comprehensive Academic Technical Report** in monochrome B&W print-ready format. Matches official ANNA / MITAOE institutional criteria across Chapters 1 to 14. |
| **[`generate_full_technical_report.py`](generate_full_technical_report.py)** | Standalone Python ReportLab script that programmatically compiles the 20-page PDF report with Calibri typography, official MITAOE logo, and rendered math equations. |
| **[`AIRFLOW_INTEGRATION_AND_MID_REVIEW_GUIDE.md`](AIRFLOW_INTEGRATION_AND_MID_REVIEW_GUIDE.md)** | Deep-dive guide explaining the 4-stage Apache Airflow pipeline grounding, failure injection mechanisms, and mid-review defense points. |
| **[`figures/`](figures/)** | High-resolution publication figures, application screenshots, and rendered mathematical equation images embedded in the technical report. |

---

## 2. Academic Project Information

* **Course:** Advanced Neural Network Architectures (Course Code: `2307411T` — Track 2: Sequential Models)
* **Institution:** Department of Electronics and Telecommunication Engineering, MIT Academy of Engineering, Pune (MITAOE)
* **Guide:** Dr. Smita Kulkarni
* **Academic Year:** 2026–2027
* **Group 12 Members:**
  1. **Ayush Nagras** (PRN: 202301070001)
  2. **Siddhesh Bhivare** (PRN: 202301070005)
  3. **Om Nikam** (PRN: 202301070006)
  4. **Vaibhav Tajane** (PRN: 202301070008)

---

## 3. How to Recompile the Technical Report PDF

To regenerate the 20-page technical report PDF from scratch:

```bash
# Ensure reportlab, matplotlib, and pillow are installed
python docs/generate_full_technical_report.py
```

The output file is written to:
`docs/Data_Pipeline_Failure_Prediction_System_Technical_Report.pdf`
