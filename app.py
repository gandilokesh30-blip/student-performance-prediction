"""
Smart Student Performance Predictor - Interactive Web Application
Built with Streamlit, scikit-learn, and seaborn.
"""

import os
import io
import time
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import (
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    FEATURE_CONSTRAINTS,
    DISCLAIMER_TEXT,
    categorize_performance,
    generate_personalized_insights,
    validate_student_inputs,
)
from src.predict import StudentPerformancePredictor
from src.data_preprocessing import load_dataset, inspect_dataset, detect_outliers_iqr

# Configure page settings
st.set_page_config(
    page_title="Smart Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    /* Global Styles */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #FFFFFF;
        padding: 1.25rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-radius: 8px;
        border: 1px solid #BFDBFE;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1D4ED8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.9rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #92400E;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

DATASET_PATH = "data/student_study_performance_prediction_dataset.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


@st.cache_data
def get_cached_dataset() -> pd.DataFrame:
    """Loads and caches the student dataset."""
    if os.path.exists(DATASET_PATH):
        return load_dataset(DATASET_PATH)
    return pd.DataFrame()


@st.cache_data
def get_model_results() -> pd.DataFrame:
    """Loads saved model benchmark results."""
    results_path = os.path.join(REPORTS_DIR, "model_results.csv")
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return pd.DataFrame()


@st.cache_resource
def get_predictor() -> StudentPerformancePredictor:
    """Instantiates cached prediction engine."""
    return StudentPerformancePredictor(models_dir=MODELS_DIR)


# Sidebar Header & Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/education.png", width=64)
    st.title("🎓 Navigation")
    navigation_option = st.radio(
        "Go to Section:",
        [
            "🏠 Home",
            "🔮 Prediction Engine",
            "📊 Data Analysis (EDA)",
            "🏆 Model Performance",
            "📁 Dataset Explorer",
            "📖 About Project",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Quick Profile Parameters")
    st.caption("Adjust student metrics for on-the-fly predictions across views:")
    
    # Input Sliders
    study_hours = st.slider("Study Hours / Day", 0.0, 24.0, 5.5, 0.1, help="Average daily study time in hours")
    attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 82.0, 0.5, help="Percentage of total academic lectures attended")
    previous_score = st.slider("Previous Exam Score", 0.0, 100.0, 72.0, 0.5, help="Prior semester/unit test score out of 100")
    assignment_score = st.slider("Assignment Score", 0.0, 100.0, 75.0, 0.5, help="Continuous assignment assessment score")
    internal_score = st.slider("Internal Exam Score", 0.0, 100.0, 70.0, 0.5, help="Mid-term internal exam score")
    practice_score = st.slider("Practice Test Score", 0.0, 100.0, 68.0, 0.5, help="Score achieved in mock practice papers")
    sleep_hours = st.slider("Sleep Hours / Day", 0.0, 24.0, 7.0, 0.1, help="Average daily sleep duration in hours")

    quick_inputs = {
        "Study_Hours_Per_Day": study_hours,
        "Attendance_Percentage": attendance,
        "Previous_Exam_Score": previous_score,
        "Assignment_Score": assignment_score,
        "Internal_Exam_Score": internal_score,
        "Practice_Test_Score": practice_score,
        "Sleep_Hours_Per_Day": sleep_hours,
    }


# ==============================================================================
# SECTION 1: HOME
# ==============================================================================
if navigation_option == "🏠 Home":
    st.markdown('<div class="main-header">Smart Student Performance Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning Based Academic Performance Prediction & Decision Support</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">1,000+</div>
            <div class="metric-label">Dataset Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">7</div>
            <div class="metric-label">Input Predictors</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">5+</div>
            <div class="metric-label">ML Models Evaluated</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">5-Fold</div>
            <div class="metric-label">Cross Validation</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    col_left, col_right = st.columns([1.2, 0.8])
    with col_left:
        st.markdown("### 📌 Executive Summary")
        st.markdown("""
        Welcome to the **Smart Student Performance Predictor**, a comprehensive Machine Learning and Predictive Analytics system engineered to estimate student final examination scores using academic history and lifestyle metrics.

        **Key Project Workflow:**
        1. **Data Ingestion & Quality Control:** Median imputation for missing values, IQR outlier diagnostics, and domain boundary validations.
        2. **Exploratory Data Analysis:** Full statistical distribution and correlation mapping across study hours, attendance, sleep, and test scores.
        3. **Leak-Free Engineering:** Isolated 80/20 train/test split preventing data leakage while generating composite indicators like weekly study volume and academic preparation indices.
        4. **Multi-Model Regression Benchmark:** Evaluation across Linear Regression, Ridge, Decision Trees, Random Forest, and Gradient Boosting.
        5. **Interactive Prediction & Recommendations:** Real-time inference paired with personalized, actionable study recommendations.
        """)
        
        st.info("💡 **Tip:** Navigate to the **Prediction Engine** in the sidebar to test custom student profiles, or inspect the **Model Performance** tab for benchmark graphs.")

    with col_right:
        st.markdown("### ⚡ Quick Score Estimation")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Estimate using current sidebar settings:")
        if st.button("🚀 Calculate Estimated Score", use_container_width=True, type="primary"):
            try:
                predictor = get_predictor()
                result = predictor.predict(quick_inputs)
                score = result["predicted_score"]
                cat = result["category"]
                color = result["category_color"]

                st.markdown(f"""
                <div style="text-align:center; padding: 1rem 0;">
                    <div style="font-size: 2.2rem; font-weight: bold; color: {color};">
                        {score:.2f} / 100
                    </div>
                    <div style="font-size: 1rem; color: #4B5563; margin-top: 0.3rem;">
                        Project Level: <b style="color: {color};">{cat}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction could not be computed: {str(e)}")

        st.markdown(f'<div class="disclaimer-box">⚠️ {DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: PREDICTION ENGINE
# ==============================================================================
elif navigation_option == "🔮 Prediction Engine":
    st.markdown('<div class="main-header">Student Performance Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive academic estimation with personalized feedback</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### 📋 Student Evaluation Profile")
        with st.form("detailed_prediction_form"):
            st.markdown("##### 📚 Academic Indicators")
            f_prev = st.number_input("Previous Examination Score (0 - 100)", min_value=0.0, max_value=100.0, value=quick_inputs["Previous_Exam_Score"], step=0.5)
            f_internal = st.number_input("Internal Examination Score (0 - 100)", min_value=0.0, max_value=100.0, value=quick_inputs["Internal_Exam_Score"], step=0.5)
            f_assign = st.number_input("Assignment Assessment Score (0 - 100)", min_value=0.0, max_value=100.0, value=quick_inputs["Assignment_Score"], step=0.5)
            f_practice = st.number_input("Practice / Mock Test Score (0 - 100)", min_value=0.0, max_value=100.0, value=quick_inputs["Practice_Test_Score"], step=0.5)

            st.markdown("##### ⏱️ Study Routine & Habits")
            f_study = st.number_input("Study Hours Per Day (0 - 24)", min_value=0.0, max_value=24.0, value=quick_inputs["Study_Hours_Per_Day"], step=0.25)
            f_attend = st.number_input("Attendance Percentage (0 - 100 %)", min_value=0.0, max_value=100.0, value=quick_inputs["Attendance_Percentage"], step=0.5)
            f_sleep = st.number_input("Sleep Hours Per Day (0 - 24)", min_value=0.0, max_value=24.0, value=quick_inputs["Sleep_Hours_Per_Day"], step=0.25)

            submitted = st.form_submit_button("Predict Final Score", type="primary", use_container_width=True)

    with col_result:
        st.markdown("### 🎯 Prediction Results & Insights")
        if submitted:
            inputs = {
                "Study_Hours_Per_Day": f_study,
                "Attendance_Percentage": f_attend,
                "Previous_Exam_Score": f_prev,
                "Assignment_Score": f_assign,
                "Internal_Exam_Score": f_internal,
                "Practice_Test_Score": f_practice,
                "Sleep_Hours_Per_Day": f_sleep,
            }
            try:
                predictor = get_predictor()
                res = predictor.predict(inputs)
                score = res["predicted_score"]
                cat = res["category"]
                color = res["category_color"]
                desc = res["category_description"]
                insights = res["personalized_insights"]

                st.markdown(f"""
                <div class="card" style="border-top: 5px solid {color}; text-align: center;">
                    <div style="font-size: 0.95rem; color: #6B7280; font-weight: 500;">ESTIMATED FINAL EXAM SCORE</div>
                    <div style="font-size: 3rem; font-weight: 800; color: {color}; margin: 0.5rem 0;">
                        {score:.2f} <span style="font-size: 1.5rem; color: #9CA3AF;">/ 100</span>
                    </div>
                    <div>
                        <span class="badge" style="background-color: {color}20; color: {color}; border: 1px solid {color};">
                            {cat}
                        </span>
                    </div>
                    <p style="margin-top: 0.8rem; font-size: 0.9rem; color: #4B5563;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 💡 Personalized Recommendations")
                for insight in insights:
                    st.markdown(f"- {insight}")

                st.markdown("#### 📋 Score Interpretation Guide")
                st.caption("Standard project-defined classification thresholds:")
                st.markdown("""
                * **90 – 100:** Excellent
                * **75 – 89:** Good
                * **60 – 74:** Average
                * **Below 60:** Needs Improvement
                """)

                # Download Prediction Report CSV
                report_df = pd.DataFrame([{
                    **inputs,
                    "Predicted_Final_Exam_Score": score,
                    "Performance_Category": cat,
                }])
                csv_buffer = report_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Prediction Report (CSV)",
                    data=csv_buffer,
                    file_name="student_prediction_result.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Error executing prediction pipeline: {str(e)}")
        else:
            st.info("👈 Enter or adjust student details and click **Predict Final Score** to view model estimations and personalized insights.")

        st.markdown(f'<div class="disclaimer-box">⚠️ {DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 3: DATA ANALYSIS (EDA)
# ==============================================================================
elif navigation_option == "📊 Data Analysis (EDA)":
    st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive statistical analysis of academic and behavioral features</div>', unsafe_allow_html=True)

    df = get_cached_dataset()
    if df.empty:
        st.error("Dataset not found at data/student_study_performance_prediction_dataset.csv.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Feature Distributions",
            "🔗 Feature vs Target Relationships",
            "🔥 Correlation Matrix",
            "⚠️ Outlier & Quality Diagnostics",
        ])

        with tab1:
            st.markdown("### Distributions of Academic & Behavioral Predictors")
            col_sel = st.selectbox(
                "Select Feature to Inspect Distribution:",
                RAW_FEATURE_COLUMNS + [TARGET_COLUMN],
                index=0,
            )
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.histplot(df[col_sel], kde=True, color="#2563EB", edgecolor="#1E3A8A", alpha=0.55, ax=ax)
            mean_val = df[col_sel].mean()
            med_val = df[col_sel].median()
            ax.axvline(mean_val, color="#DC2626", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
            ax.axvline(med_val, color="#10B981", linestyle=":", linewidth=2, label=f"Median: {med_val:.2f}")
            ax.set_title(f"Distribution of {col_sel.replace('_', ' ')}", fontweight="bold")
            ax.set_xlabel(col_sel.replace('_', ' '))
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(f"Descriptive Statistics for **{col_sel}**:")
            st.dataframe(df[[col_sel]].describe().T, use_container_width=True)

        with tab2:
            st.markdown("### Bivariate Relationships with Final Exam Score")
            feat_sel = st.selectbox(
                "Select Predictor for Scatter & Trend Analysis:",
                RAW_FEATURE_COLUMNS,
                index=0,
            )

            fig, ax = plt.subplots(figsize=(8, 4.8))
            sns.regplot(
                data=df,
                x=feat_sel,
                y=TARGET_COLUMN,
                scatter_kws={"alpha": 0.45, "color": "#0284C7", "s": 35},
                line_kws={"color": "#DC2626", "linewidth": 2.2, "label": "Linear Fit"},
                ax=ax,
            )
            corr_val = df[feat_sel].corr(df[TARGET_COLUMN])
            ax.set_title(f"{feat_sel.replace('_', ' ')} vs Final Exam Score (r = {corr_val:.2f})", fontweight="bold")
            ax.set_xlabel(feat_sel.replace('_', ' '))
            ax.set_ylabel("Final Exam Score")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            st.write(f"**Interpretation:** The correlation between `{feat_sel}` and `{TARGET_COLUMN}` is **{corr_val:.3f}**.")

        with tab3:
            st.markdown("### Pearson Correlation Heatmap")
            numeric_cols = [c for c in RAW_FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
            corr = df[numeric_cols].corr()

            fig, ax = plt.subplots(figsize=(9, 7))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                linewidths=0.6,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                ax=ax,
            )
            ax.set_title("Correlation Heatmap: Student Metrics", fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
            plt.close(fig)

        with tab4:
            st.markdown("### Outlier Detection (IQR Method)")
            st.markdown("Summary of outliers detected using the 1.5 × IQR boundary rule:")
            outliers = detect_outliers_iqr(df)
            outlier_rows = []
            for col, data in outliers.items():
                outlier_rows.append({
                    "Feature": col,
                    "Q1 (25%)": round(data["q1"], 2),
                    "Q3 (75%)": round(data["q3"], 2),
                    "IQR": round(data["iqr"], 2),
                    "Lower Bound": round(data["lower_bound"], 2),
                    "Upper Bound": round(data["upper_bound"], 2),
                    "Outlier Count": data["outlier_count"],
                })
            outliers_df = pd.DataFrame(outlier_rows)
            st.dataframe(outliers_df, use_container_width=True)
            st.info("ℹ️ Outliers were assessed and verified to reflect genuine high/low achieving students within realistic boundaries rather than measurement errors, so valid extreme records are retained.")


# ==============================================================================
# SECTION 4: MODEL PERFORMANCE
# ==============================================================================
elif navigation_option == "🏆 Model Performance":
    st.markdown('<div class="main-header">Model Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous benchmark comparison across multiple regression algorithms</div>', unsafe_allow_html=True)

    results_df = get_model_results()
    if results_df.empty:
        st.warning("Model benchmark results not found. Please run training script first (`python -m src.train_models`).")
    else:
        st.markdown("### 📊 Benchmark Comparison Table")
        st.dataframe(
            results_df.style.highlight_min(subset=["RMSE", "MAE", "MSE"], color="#DCFCE7")
                            .highlight_max(subset=["R2"], color="#DCFCE7"),
            use_container_width=True,
        )

        # Download Model Comparison Results
        csv_results = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Benchmark Results (CSV)",
            data=csv_results,
            file_name="model_comparison_results.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("### 📉 Visual Evaluation Diagnostics")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            comp_fig_path = os.path.join(FIGURES_DIR, "model_comparison.png")
            if os.path.exists(comp_fig_path):
                st.image(comp_fig_path, caption="Algorithm Comparison (RMSE, MAE, R²)")
            else:
                st.info("Comparison chart will display here once trained.")

        with col_m2:
            feat_imp_path = os.path.join(FIGURES_DIR, "feature_importance.png")
            if os.path.exists(feat_imp_path):
                st.image(feat_imp_path, caption="Feature Importance / Predictor Weights")
            else:
                st.info("Feature importance will display here once trained.")

        col_m3, col_m4 = st.columns(2)
        with col_m3:
            act_pred_path = os.path.join(FIGURES_DIR, "actual_vs_predicted.png")
            if os.path.exists(act_pred_path):
                st.image(act_pred_path, caption="Actual vs. Predicted Final Exam Score")

        with col_m4:
            res_path = os.path.join(FIGURES_DIR, "residual_plot.png")
            if os.path.exists(res_path):
                st.image(res_path, caption="Residual Analysis (Errors vs Fitted)")


# ==============================================================================
# SECTION 5: DATASET EXPLORER
# ==============================================================================
elif navigation_option == "📁 Dataset Explorer":
    st.markdown('<div class="main-header">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Raw data inspection, schema metadata, and download options</div>', unsafe_allow_html=True)

    df = get_cached_dataset()
    if df.empty:
        st.error("Dataset missing at data/student_study_performance_prediction_dataset.csv.")
    else:
        st.markdown(f"**Total Records:** {df.shape[0]} rows | **Total Features:** {df.shape[1]} columns")

        # Preview Data
        st.markdown("### 🔍 Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("### 📑 Column Details & Missing Values")
        col_info = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values,
        })
        st.dataframe(col_info, use_container_width=True)

        # Download Cleaned Dataset
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Dataset (CSV)",
            data=csv_data,
            file_name="student_performance_dataset.csv",
            mime="text/csv",
        )


# ==============================================================================
# SECTION 6: ABOUT PROJECT
# ==============================================================================
elif navigation_option == "📖 About Project":
    st.markdown('<div class="main-header">About the Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Architectural foundations, methodology, college viva guide, and disclaimers</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Problem Statement & Objectives
    Early detection of students needing academic assistance is vital in educational institutions. 
    By leveraging predictive analytics on continuous assessments (assignments, mock tests, internal exams) 
    and lifestyle indicators (daily study habits and sleep), educators can proactively intervene before final examinations.

    ### 🧱 System Architecture
    ```
    Student Input Metrics
           │
           ▼
    Validation & Domain Bounds Check
           │
           ▼
    Leak-Free Preprocessing (Medians & Scaler)
           │
           ▼
    Feature Engineering (Study Habits & Prep Index)
           │
           ▼
    Trained Regression Champion Model
           │
           ▼
    Score Estimation + Interpretation Category + Actionable Insights
    ```

    ### 🤖 Machine Learning Algorithms Implemented
    1. **Linear Regression:** Baseline parametric model estimating linear relationships.
    2. **Ridge Regression (L2 Regularization):** Mitigates multicollinearity among continuous tests.
    3. **Decision Tree Regressor:** Captures non-linear decision thresholds in study habits.
    4. **Random Forest Regressor:** Bagging ensemble minimizing prediction variance.
    5. **Gradient Boosting Regressor:** Sequential boosting optimizing residual errors.

    ### ⚠️ Ethical Disclaimer & Data Notice
    - **Synthetic Dataset:** This project uses a synthetic dataset designed specifically for academic exploration and demonstration.
    - **Non-Causality:** Feature importances represent associations within the model and do not prove causal relationships.
    - **Decision-Support Only:** Estimations are mathematical guidelines and should not replace pedagogical judgment.
    """)
