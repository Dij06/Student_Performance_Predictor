"""
MEMBER 3: STREAMLIT DASHBOARD DESIGN, UI & INTEGRATION
Clean, Simple UI Views:
- Multi-Stage Predictions (Start of Year, Midterm, End of Term)
- Plain English Insights & Feature Impacts
- Clean Charts without unnecessary emojis or complex jargon
- Student Reports
"""

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from modules.member2_ml_models import predict_student, simulate_study_time, simulate_absences

STUDY_TIME_MAP = {
    1: "Under 2 hours a week",
    2: "2 to 5 hours a week",
    3: "5 to 10 hours a week",
    4: "More than 10 hours a week"
}

EDU_MAP = {
    0: "None",
    1: "Primary School (4th Grade)",
    2: "Middle School (5th to 9th Grade)",
    3: "High School",
    4: "College or Higher Degree"
}


def apply_styles():
    """Injects clean, professional CSS styling."""
    st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        .hero-container {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
            border-radius: 12px;
            padding: 22px 26px;
            color: #ffffff;
            margin-bottom: 20px;
        }
        .hero-title {
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            font-size: 0.95rem;
            color: #cbd5e1;
            margin-top: 6px;
            line-height: 1.5;
        }
        .simple-badge {
            display: inline-block;
            background-color: #0284c7;
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renders the sidebar navigation, timing stages, and student presets."""
    with st.sidebar:
        st.markdown("## **Student Predictor**")
        st.caption("Academic Early Warning System")
        st.divider()

        st.markdown("### 1. Timing in School Year")
        st.caption("When are you making this prediction?")
        stage_choice = st.selectbox(
            "School Year Stage",
            [
                "End of Term (Both Exams Done)",
                "Midterm (First Exam Done)",
                "Start of School Year (No Exams Yet)"
            ],
            index=0
        )
        stage_num = 3 if "End of Term" in stage_choice else (2 if "Midterm" in stage_choice else 1)

        st.divider()
        st.markdown("### 2. Example Students")
        st.caption("Load a student profile:")
        persona = st.selectbox(
            "Select Student Profile",
            [
                "Custom Input",
                "High Performing Student",
                "Average Student (Borderline)",
                "Struggling Student (Needs Help)",
                "Improving Student (Comeback)",
                "Attendance Risk Student"
            ],
            index=0
        )

        # Persona defaults
        if persona == "High Performing Student":
            default_study = 3; default_fail = 0; default_abs = 2; default_g1 = 16; default_g2 = 17
            default_medu = 4; default_fedu = 4; default_schoolsup = 0; default_famsup = 1
            default_higher = 1; default_internet = 1; default_goout = 2; default_health = 5; default_alcohol = 1.0
            default_name = "Maya Lin"
        elif persona == "Average Student (Borderline)":
            default_study = 2; default_fail = 1; default_abs = 8; default_g1 = 10; default_g2 = 10
            default_medu = 2; default_fedu = 2; default_schoolsup = 0; default_famsup = 1
            default_higher = 1; default_internet = 1; default_goout = 3; default_health = 3; default_alcohol = 2.0
            default_name = "Jordan Blake"
        elif persona == "Struggling Student (Needs Help)":
            default_study = 1; default_fail = 2; default_abs = 18; default_g1 = 6; default_g2 = 7
            default_medu = 1; default_fedu = 1; default_schoolsup = 1; default_famsup = 0
            default_higher = 0; default_internet = 0; default_goout = 4; default_health = 3; default_alcohol = 3.5
            default_name = "Sam Taylor"
        elif persona == "Improving Student (Comeback)":
            default_study = 3; default_fail = 1; default_abs = 3; default_g1 = 8; default_g2 = 14
            default_medu = 3; default_fedu = 2; default_schoolsup = 0; default_famsup = 1
            default_higher = 1; default_internet = 1; default_goout = 2; default_health = 4; default_alcohol = 1.5
            default_name = "David Miller"
        elif persona == "Attendance Risk Student":
            default_study = 2; default_fail = 0; default_abs = 20; default_g1 = 13; default_g2 = 11
            default_medu = 3; default_fedu = 3; default_schoolsup = 0; default_famsup = 0
            default_higher = 1; default_internet = 1; default_goout = 4; default_health = 2; default_alcohol = 2.5
            default_name = "Chris Evans"
        else:
            default_study = 2; default_fail = 0; default_abs = 4; default_g1 = 11; default_g2 = 12
            default_medu = 2; default_fedu = 3; default_schoolsup = 0; default_famsup = 1
            default_higher = 1; default_internet = 1; default_goout = 3; default_health = 4; default_alcohol = 1.5
            default_name = "Alex Morgan"

        st.divider()
        st.markdown("### 3. Student Details")
        student_name = st.text_input("Student Name", value=default_name)
        student_id = st.text_input("Student ID", value="STU-2026-089")
        course_name = st.selectbox("Subject", ["Mathematics", "Language", "Applied Sciences"])

        st.divider()
        st.caption("Module Ownership: Data Analysis (Member 1) · Machine Learning (Member 2) · User Interface (Member 3)")

    return {
        "stage_num": stage_num,
        "stage_name": stage_choice,
        "student_name": student_name,
        "student_id": student_id,
        "course_name": course_name,
        "default_g1": default_g1,
        "default_g2": default_g2,
        "default_study": default_study,
        "default_fail": default_fail,
        "default_abs": default_abs,
        "default_medu": default_medu,
        "default_fedu": default_fedu,
        "default_schoolsup": default_schoolsup,
        "default_famsup": default_famsup,
        "default_higher": default_higher,
        "default_internet": default_internet,
        "default_goout": default_goout,
        "default_health": default_health,
        "default_alcohol": default_alcohol
    }


def render_prediction_tab(sidebar_state):
    """Renders Tab 1: Live Student Predictor with Simple Insights."""
    stage_num = sidebar_state["stage_num"]

    st.markdown(f"### Student Evaluation: **{sidebar_state['stage_name']}**")
    if stage_num == 1:
        st.info("Start of School Year Mode: Predicts risk using study habits and background before any exams take place. Helps teachers identify students who need support on Day 1.")
    elif stage_num == 2:
        st.info("Midterm Mode: Uses First Exam (G1) marks and study habits to check progress midway through the term.")
    else:
        st.info("End of Term Mode: Uses both exams (G1 and G2), score progress, and habits for a final result prediction.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### Exam Scores and History")
        if stage_num == 3:
            g1 = st.slider("First Exam Score (G1)", 0, 20, sidebar_state["default_g1"], help="Passing score is 10 out of 20.")
            st.caption(f"Score: **{g1} / 20** ({'Pass' if g1 >= 10 else 'Needs Help'})")

            g2 = st.slider("Second Exam Score (G2)", 0, 20, sidebar_state["default_g2"], help="Passing score is 10 out of 20.")
            st.caption(f"Score: **{g2} / 20** ({'Pass' if g2 >= 10 else 'Needs Help'})")
        elif stage_num == 2:
            g1 = st.slider("First Exam Score (G1)", 0, 20, sidebar_state["default_g1"], help="Passing score is 10 out of 20.")
            st.caption(f"Score: **{g1} / 20** ({'Pass' if g1 >= 10 else 'Needs Help'})")
            g2 = g1
            st.caption("Second Exam has not happened yet.")
        else:
            g1 = 10
            g2 = 10
            st.caption("No exams have taken place yet.")

        failures = st.selectbox(
            "Past Failed Classes",
            [0, 1, 2, 3, 4],
            index=sidebar_state["default_fail"],
            help="Number of subjects this student failed in previous years."
        )

        studytime_key = st.selectbox(
            "Weekly Study Time",
            [1, 2, 3, 4],
            format_func=lambda x: STUDY_TIME_MAP[x],
            index=sidebar_state["default_study"] - 1
        )

    with col2:
        st.markdown("#### Attendance and Lifestyle")
        absences = st.number_input("Days Absent from School", 0, 93, sidebar_state["default_abs"], step=1)
        if absences > 10:
            st.warning(f"High absences ({absences} days missed). Missing school significantly reduces passing chances.")
        else:
            st.caption(f"Attendance: {absences} days absent (Normal).")

        goout = st.slider("Time Spent Going Out with Friends", 1, 5, sidebar_state["default_goout"], help="1 = Very Little, 5 = Very Frequently")
        health = st.slider("Health Status", 1, 5, sidebar_state["default_health"], help="1 = Very Poor, 5 = Very Good")

    # Optional Family Background Accordion
    with st.expander("Additional Background and Support (Optional)", expanded=(stage_num == 1)):
        b1, b2 = st.columns(2)
        with b1:
            medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4], format_func=lambda x: EDU_MAP[x], index=sidebar_state["default_medu"])
            fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4], format_func=lambda x: EDU_MAP[x], index=sidebar_state["default_fedu"])
            schoolsup = 1 if st.checkbox("Gets Extra School Tutoring", value=(sidebar_state["default_schoolsup"] == 1)) else 0
        with b2:
            famsup = 1 if st.checkbox("Gets Help from Family at Home", value=(sidebar_state["default_famsup"] == 1)) else 0
            higher = 1 if st.checkbox("Wants to Go to College / Higher Education", value=(sidebar_state["default_higher"] == 1)) else 0
            internet = 1 if st.checkbox("Has Internet at Home", value=(sidebar_state["default_internet"] == 1)) else 0
            alcohol = float(sidebar_state["default_alcohol"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Calculate Prediction", type="primary")

    # Construct student record
    student_data = {
        "studytime": studytime_key,
        "failures": failures,
        "absences": absences,
        "G1": g1,
        "G2": g2,
        "Medu": medu,
        "Fedu": fedu,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "higher": higher,
        "internet": internet,
        "goout": goout,
        "health": health,
        "Dalc": alcohol,
        "Walc": alcohol,
        "stage": stage_num
    }

    # Execute Prediction
    res = predict_student(student_data, stage=stage_num)

    st.divider()
    st.markdown(f"### Prediction for **{sidebar_state['student_name']}** ({sidebar_state['student_id']})")

    # Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Predicted Result",
            res["status"],
            delta="On Track" if res["prediction"] == 1 else "Needs Attention",
            delta_color="normal" if res["prediction"] == 1 else "inverse"
        )
    with m2:
        st.metric("Pass Chance", f"{res['pass_prob']}%", delta=f"{res['pass_prob'] - 50:+.1f}% vs 50% Pass Line")
    with m3:
        st.metric("Risk Category", res["risk_badge"])
    with m4:
        if stage_num == 3:
            st.metric("Exam Progress", res["momentum_str"])
        elif stage_num == 2:
            st.metric("Exam 1 Score", f"{g1} / 20")
        else:
            st.metric("Evaluation Basis", "Habits and History")

    st.markdown("<br>", unsafe_allow_html=True)

    # Simple Horizontal Bar for Pass Chance
    chart_df = pd.DataFrame({
        "Outcome": ["Chance of Passing", "Risk of Failing"],
        "Percent": [res["pass_prob"], res["fail_prob"]],
        "Color": ["#10b981", "#ef4444"]
    })
    bar_chart = alt.Chart(chart_df).mark_bar(cornerRadius=6, size=22).encode(
        x=alt.X("Percent:Q", scale=alt.Scale(domain=[0, 100]), title="Percentage (%)"),
        y=alt.Y("Outcome:N", title="", sort="-x"),
        color=alt.Color("Color:N", scale=None),
        tooltip=["Outcome", alt.Tooltip("Percent:Q", format=".1f")]
    ).properties(height=120)
    st.altair_chart(bar_chart)

    # ---------------------------------------------------------
    # Key Factors Behind This Prediction (Simple SHAP Explanation)
    # ---------------------------------------------------------
    st.divider()
    st.markdown("### Key Factors Behind This Prediction")
    st.caption("Here is what influenced this student's result the most:")

    shap_df = res["shap_df"].head(7)

    c_chart, c_text = st.columns([3, 2], gap="large")

    with c_chart:
        chart_data = shap_df.copy()
        chart_data["Direction"] = chart_data["Impact"].apply(lambda x: "Helps Student Pass" if x >= 0 else "Risk Factor")

        impact_chart = alt.Chart(chart_data).mark_bar(cornerRadius=4).encode(
            x=alt.X("Impact:Q", title="Impact on Pass Chance (%)"),
            y=alt.Y("Feature:N", sort=alt.SortField(field="Impact", order="descending"), title=""),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["Helps Student Pass", "Risk Factor"], range=["#10b981", "#ef4444"]),
                title=""
            ),
            tooltip=["Feature", "Impact", "Direction"]
        ).properties(height=240)
        st.altair_chart(impact_chart)

    with c_text:
        positives = shap_df[shap_df["Impact"] > 0]
        negatives = shap_df[shap_df["Impact"] < 0]

        if not positives.empty:
            st.markdown("**Top Strengths (Helping Pass):**")
            for _, row in positives.head(3).iterrows():
                st.markdown(f"* **{row['Feature']}:** adds **+{row['Impact']}%**")

        if not negatives.empty:
            st.markdown("<br>**Main Risk Factors:**", unsafe_allow_html=True)
            for _, row in negatives.head(3).iterrows():
                st.markdown(f"* **{row['Feature']}:** subtracts **{row['Impact']}%**")

        if negatives.empty and not positives.empty:
            st.success("No major risk factors detected for this student.")

    # Download Report
    st.divider()
    report_text = f"""==================================================
STUDENT ACADEMIC PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Evaluation Stage: {res['stage_used']}
==================================================

STUDENT INFORMATION:
- Name: {sidebar_state['student_name']}
- ID: {sidebar_state['student_id']}
- Course: {sidebar_state['course_name']}

SCORES AND HABITS:
- Exam 1: {g1 if stage_num >= 2 else 'Not taken yet'}
- Exam 2: {g2 if stage_num == 3 else 'Not taken yet'}
- Past Failed Classes: {failures}
- Weekly Study Hours: {STUDY_TIME_MAP[studytime_key]}
- Days Missed: {absences}

PREDICTION:
- Predicted Outcome: {res['status']}
- Chance of Passing: {res['pass_prob']}%
- Risk of Failing: {res['fail_prob']}%
- Risk Level: {res['risk_level']}

MAIN INFLUENCING FACTORS:
{shap_df[['Feature', 'Impact', 'Type']].to_string(index=False)}
==================================================
"""
    st.download_button(
        label="Download Report (.txt)",
        data=report_text,
        file_name=f"Report_{sidebar_state['student_id']}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

    return student_data


def render_simulation_tab(current_student):
    """Renders Tab 4: Simple What-If Simulator."""
    st.markdown("### What-If Simulator")
    st.write("See how increasing study time or reducing missed days improves this student's chances of passing.")

    stage_num = current_student.get("stage", 3)

    st.markdown("#### 1. What happens if weekly study time increases?")
    st.caption("Keeping other grades and attendance the same:")

    df_study = simulate_study_time(current_student, stage=stage_num)
    study_chart = alt.Chart(df_study).mark_bar(color="#0284c7", cornerRadius=6).encode(
        x=alt.X("Pass_Chance:Q", title="Chance of Passing (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Study_Time:N", sort=None, title=""),
        tooltip=["Study_Time", "Pass_Chance"]
    ).properties(height=180)
    st.altair_chart(study_chart)

    st.markdown("#### 2. How many missed days can this student afford?")
    st.caption("Shows how passing chances drop as missed days increase:")

    df_abs = simulate_absences(current_student, stage=stage_num)
    line_chart = alt.Chart(df_abs).mark_line(point=True, color="#8b5cf6", strokeWidth=3).encode(
        x=alt.X("Absences:Q", title="Days Missed"),
        y=alt.Y("Pass_Chance:Q", scale=alt.Scale(domain=[0, 100]), title="Chance of Passing (%)"),
        tooltip=["Absences", "Pass_Chance"]
    ).properties(height=240)
    st.altair_chart(line_chart)
    st.caption("Missing more than 8 to 10 days quickly puts borderline students at risk of failing.")
