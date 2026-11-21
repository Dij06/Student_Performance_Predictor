import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# Import the 3 modules corresponding to the 3 team members
from modules.member1_data_analysis import render_data_analysis_tab
from modules.member2_ml_models import render_model_evaluation_tab
from modules.member3_ui_views import (
    apply_styles,
    render_sidebar,
    render_prediction_tab,
    render_simulation_tab
)

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Clean Styling & Sidebar (Member 3)
apply_styles()
sidebar_state = render_sidebar()

# ---------------------------------------------------------
# Dashboard Title Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        Student Performance Predictor
    </div>
    <div class="hero-subtitle">
        Predict student pass or fail chances, spot risks early in the semester, and see how better study habits help.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4 Main Feature Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Student Pass Predictor",
    "Data Analysis & Graphs",
    "Model Evaluation & Results",
    "What-If Habit Simulator"
])

# Tab 1: Live Student Predictor (Member 3)
with tab1:
    current_student = render_prediction_tab(sidebar_state)

# Tab 2: Dataset Analysis & Visualizations (Member 1)
with tab2:
    render_data_analysis_tab()

# Tab 3: ML Model Evaluation & Benchmarks (Member 2)
with tab3:
    render_model_evaluation_tab()

# Tab 4: What-If Habit Simulator (Member 3)
with tab4:
    render_simulation_tab(current_student)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    "Student Performance Predictor · Built with Streamlit and Random Forest."
)
