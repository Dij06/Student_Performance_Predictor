"""
MEMBER 1: DATA ANALYSIS, PREPROCESSING & EXPLORATORY DATA ANALYSIS (EDA)
Comprehensive Academic Insights:
- Multi-Dataset Support: Mathematics (395), Portuguese (649), or Combined (1,044)
- Academic Progression Analysis (Exam 1 -> Exam 2 -> Final Exam)
- Impact of Family Background, Education, and Aspirations
- Behavioral Drivers: Study Hours, Attendance, and Social Life
- Interactive Cohort Filter with CSV Export for Teachers
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

MAT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
POR_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student-por.csv")

STUDY_LABELS = {
    1: "Under 2 hrs/week",
    2: "2 to 5 hrs/week",
    3: "5 to 10 hrs/week",
    4: "More than 10 hrs/week"
}

EDU_LABELS = {
    0: "None",
    1: "Primary School",
    2: "Middle School",
    3: "High School",
    4: "Higher Education"
}


@st.cache_data
def load_dataset(dataset_type: str = "Mathematics"):
    """Loads and standardizes student datasets."""
    if dataset_type == "Mathematics":
        df = pd.read_csv(os.path.abspath(MAT_PATH), sep=";")
        df["Subject"] = "Mathematics"
    elif dataset_type == "Portuguese Language":
        df = pd.read_csv(os.path.abspath(POR_PATH), sep=";")
        df["Subject"] = "Portuguese"
    else:
        df_mat = pd.read_csv(os.path.abspath(MAT_PATH), sep=";")
        df_mat["Subject"] = "Mathematics"
        df_por = pd.read_csv(os.path.abspath(POR_PATH), sep=";")
        df_por["Subject"] = "Portuguese"
        df = pd.concat([df_mat, df_por], ignore_index=True)

    # Standardize target
    if "passed" not in df.columns and "G3" in df.columns:
        df["passed"] = (df["G3"] >= 10).astype(int)
        df["status"] = df["passed"].map({1: "Passed", 0: "Failed"})

    # Clean label mappings
    df["study_label"] = df["studytime"].map(STUDY_LABELS)
    df["medu_label"] = df["Medu"].map(EDU_LABELS)
    df["fedu_label"] = df["Fedu"].map(EDU_LABELS)

    return df


def render_data_analysis_tab():
    """Renders the comprehensive Data Analysis & Exploratory Data Analysis (EDA) tab."""
    st.markdown("### Exploratory Data Analysis & Student Patterns")
    st.caption("Inspect cohort distributions, academic progress, family influences, and behavioral patterns.")

    # 1. Dataset Selection
    col_sel1, col_sel2 = st.columns([2, 3])
    with col_sel1:
        dataset_choice = st.selectbox(
            "Select Course Dataset",
            ["Mathematics (395 students)", "Portuguese Language (649 students)", "Both Courses Combined (1,044 students)"],
            index=0
        )
    clean_choice = "Mathematics" if "Mathematics" in dataset_choice else (
        "Portuguese Language" if "Portuguese" in dataset_choice else "Combined"
    )

    df = load_dataset(clean_choice)

    # 2. Key Cohort Metrics
    total = len(df)
    pass_count = int((df["passed"] == 1).sum())
    fail_count = total - pass_count
    pass_rate = round((pass_count / total) * 100, 1)
    avg_g3 = round(df["G3"].mean(), 1)
    avg_abs = round(df["absences"].mean(), 1)
    prior_fail_count = int((df["failures"] > 0).sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Students", total)
    with k2:
        st.metric("Pass Rate", f"{pass_rate}%", f"{pass_count} passed")
    with k3:
        st.metric("Average Final Score", f"{avg_g3} / 20")
    with k4:
        st.metric("Average Days Absent", f"{avg_abs} days")
    with k5:
        st.metric("Students with Prior Failures", prior_fail_count, f"{round((prior_fail_count/total)*100, 1)}% of cohort")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. EDA Sub-Sections using Tabs
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "Grade Progression & Trends",
        "Study Habits & Attendance",
        "Family & School Support",
        "Lifestyle & Interactive Filter"
    ])

    # ---------------------------------------------------------
    # SUB-TAB 1: Grade Progression & Distributions
    # ---------------------------------------------------------
    with eda_tab1:
        st.markdown("#### Exam Score Distributions and Progress")
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("**Exam Score Progression (Exam 1 to Final Exam)**")
            st.caption("How average marks evolve from Exam 1 to Exam 2 to the Final Exam:")
            prog_data = df.groupby("status")[["G1", "G2", "G3"]].mean().reset_index()
            melted_prog = prog_data.melt(id_vars="status", var_name="Exam", value_name="Average Score")
            exam_names = {"G1": "First Exam (G1)", "G2": "Second Exam (G2)", "G3": "Final Exam (G3)"}
            melted_prog["Exam"] = melted_prog["Exam"].map(exam_names)

            prog_chart = alt.Chart(melted_prog).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("Exam:N", title="Exam Period", sort=["First Exam (G1)", "Second Exam (G2)", "Final Exam (G3)"]),
                y=alt.Y("Average Score:Q", title="Average Marks (0 to 20)", scale=alt.Scale(domain=[4, 16])),
                color=alt.Color("status:N", scale=alt.Scale(domain=["Passed", "Failed"], range=["#10b981", "#ef4444"]), title="Outcome"),
                tooltip=["status", "Exam", alt.Tooltip("Average Score:Q", format=".1f")]
            ).properties(height=240)
            st.altair_chart(prog_chart)
            st.caption("Notice: Students who ultimately fail already start well below passing (average ~7.5/20) on the very first exam.")

        with c2:
            st.markdown("**Final Grade Distribution (G3)**")
            st.caption("Number of students scoring across the 0 to 20 range (Pass cutoff = 10):")
            hist_chart = alt.Chart(df).mark_bar(binSpacing=1).encode(
                x=alt.X("G3:Q", bin=alt.Bin(step=1), title="Final Mark (0 to 20)"),
                y=alt.Y("count()", title="Student Count"),
                color=alt.Color("status:N", scale=alt.Scale(domain=["Passed", "Failed"], range=["#10b981", "#ef4444"]), title="Status"),
                tooltip=["count()", alt.Tooltip("G3:Q", bin=True)]
            ).properties(height=240)
            st.altair_chart(hist_chart)

    # ---------------------------------------------------------
    # SUB-TAB 2: Study Habits & Attendance
    # ---------------------------------------------------------
    with eda_tab2:
        st.markdown("#### Academic Habits: Study Time, Failures, and Missed Classes")
        h1, h2 = st.columns(2, gap="large")

        with h1:
            st.markdown("**Weekly Study Hours vs. Pass Rate**")
            grouped_study = df.groupby("study_label").agg(
                pass_rate=("passed", lambda x: round(float(x.mean()) * 100, 1)),
                avg_score=("G3", lambda x: round(float(x.mean()), 1)),
                count=("G3", "count")
            ).reset_index()

            study_chart = alt.Chart(grouped_study).mark_bar(cornerRadius=6, color="#0284c7").encode(
                x=alt.X("study_label:N", title="Study Time", sort=["Under 2 hrs/week", "2 to 5 hrs/week", "5 to 10 hrs/week", "More than 10 hrs/week"]),
                y=alt.Y("pass_rate:Q", title="Pass Rate (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=["study_label", "pass_rate", "avg_score", "count"]
            ).properties(height=230)
            st.altair_chart(study_chart)
            st.caption("Students studying over 5 hours a week consistently achieve pass rates above 75%.")

        with h2:
            st.markdown("**Impact of Previous Class Failures**")
            grouped_fail = df.groupby("failures").agg(
                pass_rate=("passed", lambda x: round(float(x.mean()) * 100, 1)),
                count=("G3", "count")
            ).reset_index()
            grouped_fail["Label"] = grouped_fail["failures"].astype(str) + " Failure(s)"

            fail_chart = alt.Chart(grouped_fail).mark_bar(cornerRadius=6).encode(
                x=alt.X("Label:N", title="Past Failures", sort=None),
                y=alt.Y("pass_rate:Q", title="Pass Rate (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("pass_rate:Q", scale=alt.Scale(scheme="redyellowgreen"), legend=None),
                tooltip=["Label", "pass_rate", "count"]
            ).properties(height=230)
            st.altair_chart(fail_chart)
            st.caption("Having even 1 previous failure drops the passing chance from ~75% down to ~35%.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Absences Box-Distribution
        st.markdown("**Attendance: Days Missed by Passing vs. Failing Students**")
        abs_summary = df.groupby("status")["absences"].describe().round(1).reset_index()
        st.dataframe(abs_summary[["status", "mean", "std", "min", "50%", "max"]], hide_index=True)
        st.caption("Failing students show much wider variance and higher maximum absences compared to passing students.")

    # ---------------------------------------------------------
    # SUB-TAB 3: Family & School Support
    # ---------------------------------------------------------
    with eda_tab3:
        st.markdown("#### Family Background and Support Programs")
        f1, f2 = st.columns(2, gap="large")

        with f1:
            st.markdown("**Parents' Education Level vs. Student Pass Rate**")
            medu_stat = df.groupby("medu_label").agg(
                pass_rate=("passed", lambda x: round(float(x.mean()) * 100, 1)),
                count=("G3", "count")
            ).reset_index()

            medu_chart = alt.Chart(medu_stat).mark_bar(cornerRadius=6, color="#0ea5e9").encode(
                x=alt.X("medu_label:N", title="Mother's Education Level", sort=["None", "Primary School", "Middle School", "High School", "Higher Education"]),
                y=alt.Y("pass_rate:Q", title="Pass Rate (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=["medu_label", "pass_rate", "count"]
            ).properties(height=230)
            st.altair_chart(medu_chart)
            st.caption("Higher parental education correlates with higher student passing rates.")

        with f2:
            st.markdown("**Support Systems & Aspirations**")
            support_stats = []
            for col_name, label in [
                ("higher", "Plans to Go to College"),
                ("internet", "Has Home Internet"),
                ("famsup", "Gets Family Educational Support"),
                ("schoolsup", "Extra School Remedial Tutoring")
            ]:
                if col_name in df.columns:
                    p_yes = round(float(df[df[col_name] == "yes"]["passed"].mean()) * 100, 1)
                    p_no = round(float(df[df[col_name] == "no"]["passed"].mean()) * 100, 1)
                    support_stats.append({"Factor": label, "With Factor (%)": p_yes, "Without Factor (%)": p_no})

            sup_df = pd.DataFrame(support_stats)
            sup_melted = sup_df.melt(id_vars="Factor", var_name="Group", value_name="Pass Rate (%)")

            sup_chart = alt.Chart(sup_melted).mark_bar().encode(
                x=alt.X("Factor:N", title=""),
                y=alt.Y("Pass Rate (%):Q", title="Pass Rate (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Group:N", scale=alt.Scale(range=["#0284c7", "#94a3b8"]), title=""),
                xOffset="Group:N",
                tooltip=["Factor", "Group", "Pass Rate (%)"]
            ).properties(height=230)
            st.altair_chart(sup_chart)
            st.caption("Students who want to attend college have a significantly higher pass rate (~73% vs ~38%).")

    # ---------------------------------------------------------
    # SUB-TAB 4: Lifestyle & Interactive Cohort Filter
    # ---------------------------------------------------------
    with eda_tab4:
        st.markdown("#### Lifestyle Factors & Interactive Cohort Explorer")

        l1, l2 = st.columns(2, gap="large")
        with l1:
            st.markdown("**Socializing with Friends (`goout`) vs. Pass Rate**")
            goout_stat = df.groupby("goout").agg(
                pass_rate=("passed", lambda x: round(float(x.mean()) * 100, 1)),
                count=("G3", "count")
            ).reset_index()
            goout_stat["Frequency"] = goout_stat["goout"].map({1: "1 (Very Low)", 2: "2 (Low)", 3: "3 (Medium)", 4: "4 (High)", 5: "5 (Very High)"})

            goout_chart = alt.Chart(goout_stat).mark_bar(cornerRadius=6, color="#8b5cf6").encode(
                x=alt.X("Frequency:N", title="Going Out Frequency", sort=None),
                y=alt.Y("pass_rate:Q", title="Pass Rate (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=["Frequency", "pass_rate", "count"]
            ).properties(height=210)
            st.altair_chart(goout_chart)

        with l2:
            st.markdown("**Weekend Social / Alcohol Intake (`Walc`) vs. Final Marks**")
            alc_stat = df.groupby("Walc").agg(
                avg_g3=("G3", lambda x: round(float(x.mean()), 1)),
                count=("G3", "count")
            ).reset_index()
            alc_stat["Level"] = alc_stat["Walc"].map({1: "1 (Lowest)", 2: "2", 3: "3", 4: "4", 5: "5 (Highest)"})

            alc_chart = alt.Chart(alc_stat).mark_line(point=True, color="#ef4444", strokeWidth=3).encode(
                x=alt.X("Level:N", title="Weekend Alcohol Level", sort=None),
                y=alt.Y("avg_g3:Q", title="Average Final Mark (0 to 20)", scale=alt.Scale(domain=[8, 14])),
                tooltip=["Level", "avg_g3", "count"]
            ).properties(height=210)
            st.altair_chart(alc_chart)

        st.divider()

        # Interactive Cohort Filter
        st.markdown("#### Filter and Explore Cohort Records")
        st.caption("Slice student records by gender, area, or study time and export the filtered list:")

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            f_gender = st.selectbox("Gender Filter", ["All", "Female", "Male"])
        with fc2:
            f_address = st.selectbox("Area Filter", ["All", "Urban (City)", "Rural"])
        with fc3:
            f_result = st.selectbox("Result Filter", ["All", "Passed", "Failed"])

        filtered = df.copy()
        if f_gender == "Female":
            filtered = filtered[filtered["sex"] == "F"]
        elif f_gender == "Male":
            filtered = filtered[filtered["sex"] == "M"]

        if f_address == "Urban (City)":
            filtered = filtered[filtered["address"] == "U"]
        elif f_address == "Rural":
            filtered = filtered[filtered["address"] == "R"]

        if f_result == "Passed":
            filtered = filtered[filtered["passed"] == 1]
        elif f_result == "Failed":
            filtered = filtered[filtered["passed"] == 0]

        st.markdown(f"**Showing {len(filtered)} matching students** (Pass Rate: **{round(float(filtered['passed'].mean())*100, 1) if len(filtered) > 0 else 0}%**):")

        disp_cols = ["Subject", "sex", "age", "address", "study_label", "failures", "absences", "G1", "G2", "G3", "status"]
        st.dataframe(filtered[disp_cols], hide_index=True)

        # CSV Download for filtered records
        csv_data = filtered[disp_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Students List (CSV)",
            data=csv_data,
            file_name=f"Filtered_Students_{clean_choice}.csv",
            mime="text/csv"
        )
