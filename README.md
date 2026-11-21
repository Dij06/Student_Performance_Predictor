# Student Performance Prediction Dashboard

An AI-powered academic decision-support platform built using **Streamlit** and a **Random Forest Classifier** trained on the UCI Student Performance dataset.

The system predicts student outcomes (Pass vs. Fail), visualizes multi-cohort data across Mathematics and Portuguese courses, and simulates how study habits and attendance influence passing chances.

---

## Clean Modular Structure

```text
Student_Performance_Predictor/
│
├── .streamlit/
│   └── config.toml                   # Streamlit UI theme configuration
│
├── data/
│   ├── student-mat.csv               # Mathematics course dataset (395 students)
│   └── student-por.csv               # Portuguese language dataset (649 students)
│
├── modules/
│   ├── member1_data_analysis.py      # [MEMBER 1] Multi-dataset loading, EDA & cohort filtering
│   ├── member2_ml_models.py          # [MEMBER 2] ML prediction, probability calibration & evaluation
│   └── member3_ui_views.py           # [MEMBER 3] Clean Streamlit UI, styling & What-If simulator
│
├── app.py                            # Main Streamlit application entry point
├── requirements.txt                  # Python package dependencies
├── .python-version                   # Python runtime pinning (3.11)
├── README.md                         # Project documentation
└── .gitignore                        # Git ignore rules
```

---

## Features

1. **Student Pass Predictor:** Predicts pass/fail results with calibrated probability, risk level, and key contributing factors across school year stages (Start of Year, Midterm, End of Term).
2. **Data Analysis & Graphs:** Interactive exploratory charts for Mathematics, Portuguese, or Combined cohorts, with grade progression trends and CSV export.
3. **Model Evaluation & Results:** Real-world metrics evaluated on held-out test students with confusion matrix and benchmarks.
4. **What-If Habit Simulator:** Tests how improving weekly study hours or reducing missed days impacts passing chances.

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.
