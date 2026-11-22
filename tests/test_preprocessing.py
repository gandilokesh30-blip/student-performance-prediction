"""
Unit tests for data loading, data cleaning, validation, outlier detection,
feature engineering, and train/test split functionality.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_preprocessing import (
    load_dataset,
    inspect_dataset,
    detect_outliers_iqr,
    clean_data,
    add_engineered_features,
    prepare_train_test_split,
)
from src.utils import RAW_FEATURE_COLUMNS, TARGET_COLUMN

DATASET_PATH = "data/student_study_performance_prediction_dataset.csv"


def test_load_dataset():
    """Verify that dataset loads properly and has expected columns."""
    assert os.path.exists(DATASET_PATH), f"Dataset file missing at {DATASET_PATH}"
    df = load_dataset(DATASET_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 100
    for col in RAW_FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature: {col}"
    assert TARGET_COLUMN in df.columns, f"Missing target: {TARGET_COLUMN}"


def test_inspect_dataset():
    """Verify inspection metadata outputs."""
    df = load_dataset(DATASET_PATH)
    inspection = inspect_dataset(df)
    assert "shape" in inspection
    assert "columns" in inspection
    assert "missing_values" in inspection
    assert "summary_statistics" in inspection
    assert inspection["shape"][0] == df.shape[0]


def test_clean_data_handles_missing_and_duplicates():
    """Verify median imputation and duplicate row removal."""
    sample_data = pd.DataFrame({
        "Study_Hours_Per_Day": [4.0, np.nan, 4.0],
        "Attendance_Percentage": [85.0, 90.0, 85.0],
        "Previous_Exam_Score": [70.0, 75.0, 70.0],
        "Assignment_Score": [65.0, 70.0, 65.0],
        "Internal_Exam_Score": [68.0, 72.0, 68.0],
        "Practice_Test_Score": [64.0, 68.0, 64.0],
        "Sleep_Hours_Per_Day": [7.0, 8.0, 7.0],
        "Final_Exam_Score": [72.0, 78.0, 72.0],
    })
    cleaned_df, medians = clean_data(sample_data)
    # Check duplicate row removed (3 -> 2 rows)
    assert len(cleaned_df) == 2
    # Check NaN was filled with median
    assert not cleaned_df["Study_Hours_Per_Day"].isnull().any()
    assert cleaned_df["Study_Hours_Per_Day"].iloc[1] == 4.0


def test_clean_data_bounds_clipping():
    """Verify physical bounds validation and clipping."""
    invalid_data = pd.DataFrame({
        "Study_Hours_Per_Day": [-5.0, 35.0],
        "Attendance_Percentage": [-10.0, 150.0],
        "Previous_Exam_Score": [-5.0, 105.0],
        "Assignment_Score": [50.0, 50.0],
        "Internal_Exam_Score": [50.0, 50.0],
        "Practice_Test_Score": [50.0, 50.0],
        "Sleep_Hours_Per_Day": [-1.0, 30.0],
        "Final_Exam_Score": [-10.0, 120.0],
    })
    cleaned_df, _ = clean_data(invalid_data)
    assert cleaned_df["Study_Hours_Per_Day"].min() >= 0.0
    assert cleaned_df["Study_Hours_Per_Day"].max() <= 24.0
    assert cleaned_df["Attendance_Percentage"].min() >= 0.0
    assert cleaned_df["Attendance_Percentage"].max() <= 100.0
    assert cleaned_df["Final_Exam_Score"].min() >= 0.0
    assert cleaned_df["Final_Exam_Score"].max() <= 100.0


def test_detect_outliers_iqr():
    """Verify IQR calculation reports bounds and outlier counts."""
    df = load_dataset(DATASET_PATH)
    outliers = detect_outliers_iqr(df)
    assert "Study_Hours_Per_Day" in outliers
    for col, stats in outliers.items():
        assert "q1" in stats
        assert "q3" in stats
        assert "lower_bound" in stats
        assert "upper_bound" in stats
        assert stats["lower_bound"] <= stats["upper_bound"]


def test_add_engineered_features():
    """Verify calculated features logic and target leakage absence."""
    df = pd.DataFrame([{
        "Study_Hours_Per_Day": 5.0,
        "Attendance_Percentage": 80.0,
        "Previous_Exam_Score": 70.0,
        "Assignment_Score": 80.0,
        "Internal_Exam_Score": 90.0,
        "Practice_Test_Score": 85.0,
        "Sleep_Hours_Per_Day": 7.0,
    }])
    feat_df = add_engineered_features(df)
    assert "Study_Hours_Per_Week" in feat_df.columns
    assert feat_df["Study_Hours_Per_Week"].iloc[0] == 35.0  # 5 * 7

    assert "Average_Internal_Performance" in feat_df.columns
    # (80 + 90 + 85) / 3 = 85.0
    assert feat_df["Average_Internal_Performance"].iloc[0] == 85.0

    assert "Academic_Preparation_Score" in feat_df.columns
    # 70 * 0.40 + 85 * 0.35 + 90 * 0.25 = 28 + 29.75 + 22.5 = 80.25
    assert feat_df["Academic_Preparation_Score"].iloc[0] == 80.25


def test_prepare_train_test_split():
    """Verify 80/20 train/test split and strict isolation."""
    df = load_dataset(DATASET_PATH)
    X_train, X_test, y_train, y_test, medians, scaler, feature_cols = prepare_train_test_split(df)

    total_rows = len(df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert round(len(X_test) / total_rows, 1) == 0.2
    assert "Student_ID" not in feature_cols
    assert TARGET_COLUMN not in feature_cols
