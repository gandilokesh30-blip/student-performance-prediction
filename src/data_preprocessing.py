"""
Data loading, cleaning, validation, outlier detection, feature engineering,
and train/test splitting pipeline for student performance prediction.
"""

import os
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import (
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN,
    FEATURE_CONSTRAINTS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads student performance CSV dataset into a pandas DataFrame.

    Args:
        filepath: Relative or absolute path to dataset CSV.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataset from '{filepath}' with shape {df.shape}.")
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive dataset inspection.

    Args:
        df: Input DataFrame.

    Returns:
        Dict containing dataset metadata and quality indicators.
    """
    inspection = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "summary_statistics": df.describe().to_dict(),
    }
    return inspection


def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
    """
    Detects outliers across numeric columns using the Interquartile Range (IQR) method.
    Outliers are values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Args:
        df: Input DataFrame.
        columns: List of columns to evaluate. Defaults to RAW_FEATURE_COLUMNS + [TARGET_COLUMN].

    Returns:
        Dictionary mapping column name to outlier statistics and row indices.
    """
    if columns is None:
        columns = [c for c in RAW_FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]

    outlier_report = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_report[col] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(len(outliers)),
            "outlier_indices": outliers.index.tolist(),
        }
    return outlier_report


def clean_data(
    df: pd.DataFrame,
    feature_medians: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Cleans dataset by:
    1. Removing duplicate rows.
    2. Validating domain value ranges (capping invalid physical ranges if present).
    3. Performing median imputation for missing numerical values.

    Args:
        df: Input DataFrame.
        feature_medians: Optional pre-calculated medians (from train set) to prevent data leakage.

    Returns:
        Tuple of (Cleaned DataFrame, Dictionary of feature medians used).
    """
    df_clean = df.copy()

    # 1. Remove duplicate records
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        logger.info(f"Removing {dup_count} duplicate rows.")
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 2. Impute missing values with medians
    calculated_medians = {}
    cols_to_impute = [c for c in RAW_FEATURE_COLUMNS if c in df_clean.columns]
    
    for col in cols_to_impute:
        if feature_medians and col in feature_medians:
            med_val = feature_medians[col]
        else:
            med_val = df_clean[col].median()
        
        calculated_medians[col] = float(med_val)
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(med_val)

    # Target column imputation if target is present and has missing values
    if TARGET_COLUMN in df_clean.columns and df_clean[TARGET_COLUMN].isnull().sum() > 0:
        target_med = df_clean[TARGET_COLUMN].median()
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].fillna(target_med)

    # 3. Validate and enforce domain constraints
    for col, (min_val, max_val) in FEATURE_CONSTRAINTS.items():
        if col in df_clean.columns:
            # Clip domain boundaries gracefully without dropping valid extreme performers
            df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)

    if TARGET_COLUMN in df_clean.columns:
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].clip(lower=0.0, upper=100.0)

    return df_clean, calculated_medians


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logically derives domain features:
    1. Study_Hours_Per_Week: (Study_Hours_Per_Day * 7.0)
    2. Average_Internal_Performance: Mean of Assignment, Internal Exam, and Practice Test.
    3. Academic_Preparation_Score: Composite index emphasizing practice and internal exams.

    Note: Target variable Final_Exam_Score is NEVER used here (Strict Target Leakage Prevention).

    Args:
        df: Input DataFrame containing raw features.

    Returns:
        DataFrame with additional engineered columns.
    """
    df_feat = df.copy()

    # Feature 1: Study hours scaled to weekly habit
    if "Study_Hours_Per_Day" in df_feat.columns:
        df_feat["Study_Hours_Per_Week"] = (df_feat["Study_Hours_Per_Day"] * 7.0).round(2)

    # Feature 2: Average internal continuous assessment performance
    internal_components = ["Assignment_Score", "Internal_Exam_Score", "Practice_Test_Score"]
    if all(c in df_feat.columns for c in internal_components):
        df_feat["Average_Internal_Performance"] = df_feat[internal_components].mean(axis=1).round(2)

    # Feature 3: Academic preparation composite index (weighted)
    prep_components = ["Previous_Exam_Score", "Practice_Test_Score", "Internal_Exam_Score"]
    if all(c in df_feat.columns for c in prep_components):
        df_feat["Academic_Preparation_Score"] = (
            df_feat["Previous_Exam_Score"] * 0.40
            + df_feat["Practice_Test_Score"] * 0.35
            + df_feat["Internal_Exam_Score"] * 0.25
        ).round(2)

    return df_feat


def prepare_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, float], StandardScaler, list]:
    """
    Executes leak-free preprocessing and train-test split:
    1. Splits raw data into train and test sets (80/20) with random_state=42.
    2. Computes median imputation parameters strictly on training set.
    3. Applies cleaning and feature engineering to train and test sets.
    4. Drops non-feature identifiers (Student_ID).
    5. Fits StandardScaler on training features and transforms test features.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, train_medians, scaler, feature_names)
    """
    # Exclude ID column from modeling
    cols_to_use = [c for c in df.columns if c != ID_COLUMN]
    df_model = df[cols_to_use].copy()

    # Initial train-test split before learning any parameters
    train_df, test_df = train_test_split(
        df_model, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train split shape: {train_df.shape}, Test split shape: {test_df.shape}")

    # Compute medians on training data ONLY
    train_medians = {
        col: float(train_df[col].median())
        for col in RAW_FEATURE_COLUMNS
        if col in train_df.columns
    }

    # Clean train and test sets using train_medians
    train_clean, _ = clean_data(train_df, feature_medians=train_medians)
    test_clean, _ = clean_data(test_df, feature_medians=train_medians)

    # Add engineered features
    train_feat = add_engineered_features(train_clean)
    test_feat = add_engineered_features(test_clean)

    # Separate X and y
    feature_cols = [c for c in train_feat.columns if c != TARGET_COLUMN]
    X_train = train_feat[feature_cols].copy()
    y_train = train_feat[TARGET_COLUMN].copy()
    X_test = test_feat[feature_cols].copy()
    y_test = test_feat[TARGET_COLUMN].copy()

    # Fit scaler on X_train only
    scaler = StandardScaler()
    scaler.fit(X_train)

    return X_train, X_test, y_train, y_test, train_medians, scaler, feature_cols
