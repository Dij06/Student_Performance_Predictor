# Student Performance Prediction & Risk Dashboard

An AI-powered academic decision-support platform built using **Streamlit** and a **Random Forest Classifier** trained on the UCI Student Performance benchmark dataset.

The system predicts final exam outcomes (Pass vs. Fail), visualizes multi-cohort exploratory data (1,044 students across Mathematics and Portuguese), benchmarks machine learning algorithms, explains individual predictions via SHAP, and simulates how study habits and attendance influence passing chances.

---

## Clean Modular Architecture

```text
Student_Performance_Streamlit/
│
├── .streamlit/
│   └── config.toml                   # Streamlit custom theme & server configuration
│
├── data/
│   ├── student-mat.csv               # Mathematics course dataset (395 students)
│   ├── student-por.csv               # Portuguese language dataset (649 students)
│   └── student.txt                   # Attribute documentation dictionary
│
├── modules/
│   ├── member1_data_analysis.py      # [MEMBER 1] Multi-dataset loading, EDA & cohort filtering
│   ├── member2_ml_models.py          # [MEMBER 2] 3-Stage ML prediction, SHAP XAI & evaluation
│   └── member3_ui_views.py           # [MEMBER 3] Clean Streamlit UI, styling & What-If simulator
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py              # Automated unit tests for data, ML, and simulations
│
├── app.py                            # Main Streamlit application entry point
├── requirements.txt                  # Python package dependencies
├── README.md                         # Documentation
└── .gitignore                        # Git ignore rules
```

---

## Team Roles & Ownership

### Member 1: Dataset Analysis & Exploratory Data Analysis (EDA)
* **File:** `modules/member1_data_analysis.py`
* **Responsibilities:**
  * Multi-course dataset loader supporting **Mathematics (395)**, **Portuguese (649)**, or **Combined (1,044)** records.
  * Cohort KPI metrics (pass rate, average marks, absences, at-risk count).
  * Grade progression analysis ($G_1 \rightarrow G_2 \rightarrow G_3$).
  * Study hours, absences, and past failure distributions.
  * Family background, parental education, and college aspiration impacts.
  * Interactive cohort filter with 1-click CSV export for educators.
* **Dashboard Tab:** **Data Analysis & Graphs**

### Member 2: Machine Learning Models, Evaluation & Explainability
* **File:** `modules/member2_ml_models.py`
* **Responsibilities:**
  * **3-Stage Academic Early Warning System:**
    * **Stage 1 (Start of Year):** Predicts risk on Day 1 using habits and background before any exams occur.
    * **Stage 2 (Midterm):** Incorporates Exam 1 ($G_1$) for mid-semester intervention.
    * **Stage 3 (End of Term):** Integrates Exam 1, Exam 2 ($G_2$), and score momentum ($G_2 - G_1$).
  * **Class Imbalance Handling:** Random Forest trained with `class_weight='balanced'`.
  * **Probability Calibration:** Platt Scaling via `CalibratedClassifierCV` ensures output probabilities reflect true posterior chances.
  * **Zero Data Leakage:** Honest evaluation strictly on a held-out 20% stratified test set.
  * **Explainable AI (XAI):** Integrated **SHAP (TreeExplainer)** providing local, per-student feature contribution breakdowns.
  * **Cached Benchmarking:** 5-Fold Stratified Cross-Validation benchmark against Decision Tree and Logistic Regression.
* **Dashboard Tab:** **Model Evaluation & Results**

### Member 3: Dashboard Design, User Experience & Integration
* **File:** `modules/member3_ui_views.py` and `app.py`
* **Responsibilities:**
  * Clean UI layout with simple, plain-English wording and zero clutter.
  * Timing stage selector (Start of School Year, Midterm, End of Term).
  * Student preset profiles for quick testing.
  * Live student assessment form, calibrated pass chance gauges, and strength/risk breakdowns.
  * Downloadable student evaluation report (.txt).
  * What-If Habit and Attendance Sensitivity Simulator.
* **Dashboard Tabs:** **Student Pass Predictor** & **What-If Habit Simulator**

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` (or the port displayed in your terminal).

### 3. Run Automated Unit Tests
```bash
python -m unittest tests.test_pipeline
```
