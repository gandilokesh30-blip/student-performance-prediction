"""
Inference module providing robust, standalone prediction functionality,
feature transformation matching training steps, domain validation,
and category/insight generation.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional
import joblib
import pandas as pd
import numpy as np

from src.utils import (
    RAW_FEATURE_COLUMNS,
    validate_student_inputs,
    categorize_performance,
    generate_personalized_insights,
    DISCLAIMER_TEXT,
)
from src.data_preprocessing import add_engineered_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StudentPerformancePredictor:
    """
    Encapsulates artifact loading, input preparation, pipeline execution,
    and post-processing for student score predictions.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model = None
        self.feature_columns = None
        self.feature_medians = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model and feature metadata."""
        model_path = os.path.join(self.models_dir, "best_model.pkl")
        cols_path = os.path.join(self.models_dir, "feature_columns.pkl")
        medians_path = os.path.join(self.models_dir, "feature_medians.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model artifact not found at {model_path}. Please train models first."
            )
        
        self.model = joblib.load(model_path)
        
        if os.path.exists(cols_path):
            self.feature_columns = joblib.load(cols_path)
        if os.path.exists(medians_path):
            self.feature_medians = joblib.load(medians_path)

        logger.info(f"Loaded model successfully from {model_path}.")

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end prediction:
        1. Validates input values.
        2. Imputes missing features with training medians if necessary.
        3. Generates domain engineered features.
        4. Predicts score, clamps between [0.0, 100.0], rounds to 2 decimals.
        5. Computes performance category and personalized recommendations.

        Args:
            input_data: Dictionary containing the 7 raw student attributes.

        Returns:
            Dictionary containing prediction score, category, insights, and disclaimer.
        """
        is_valid, err_msg = validate_student_inputs(input_data)
        if not is_valid:
            raise ValueError(f"Input validation error: {err_msg}")

        # Construct single-row DataFrame
        df_input = pd.DataFrame([{col: float(input_data[col]) for col in RAW_FEATURE_COLUMNS}])

        # Impute if any column missing (graceful handling)
        if self.feature_medians:
            for col, med in self.feature_medians.items():
                if col in df_input.columns and df_input[col].isnull().any():
                    df_input[col] = df_input[col].fillna(med)

        # Apply feature engineering
        df_featured = add_engineered_features(df_input)

        # Align with training feature columns
        if self.feature_columns:
            # Ensure exact ordering
            df_featured = df_featured[self.feature_columns]

        # Model inference
        raw_pred = float(self.model.predict(df_featured)[0])

        # Clamp between 0.0 and 100.0 and round to 2 decimals
        clamped_score = max(0.0, min(100.0, raw_pred))
        rounded_score = round(clamped_score, 2)

        category_name, category_color, category_desc = categorize_performance(rounded_score)
        insights = generate_personalized_insights(input_data)

        explanation = (
            f"Based on the academic history and study habits provided, the trained model estimates "
            f"a Final Exam Score of {rounded_score:.2f} out of 100."
        )

        return {
            "predicted_score": rounded_score,
            "category": category_name,
            "category_color": category_color,
            "category_description": category_desc,
            "explanation": explanation,
            "personalized_insights": insights,
            "disclaimer": DISCLAIMER_TEXT,
            "raw_inputs": input_data,
        }


def predict_student_score(
    study_hours_per_day: float,
    attendance_percentage: float,
    previous_exam_score: float,
    assignment_score: float,
    internal_exam_score: float,
    practice_test_score: float,
    sleep_hours_per_day: float,
    models_dir: str = "models",
) -> float:
    """
    Standard function interface accepting 7 parameters and returning rounded float score.

    Args:
        study_hours_per_day: Daily study hours (0-24)
        attendance_percentage: Percentage attendance (0-100)
        previous_exam_score: Prior exam score (0-100)
        assignment_score: Continuous assignment score (0-100)
        internal_exam_score: Internal test score (0-100)
        practice_test_score: Mock test score (0-100)
        sleep_hours_per_day: Daily sleep hours (0-24)
        models_dir: Directory where model artifacts reside

    Returns:
        Predicted Final Exam Score rounded to two decimal places.
    """
    inputs = {
        "Study_Hours_Per_Day": study_hours_per_day,
        "Attendance_Percentage": attendance_percentage,
        "Previous_Exam_Score": previous_exam_score,
        "Assignment_Score": assignment_score,
        "Internal_Exam_Score": internal_exam_score,
        "Practice_Test_Score": practice_test_score,
        "Sleep_Hours_Per_Day": sleep_hours_per_day,
    }
    predictor = StudentPerformancePredictor(models_dir=models_dir)
    result = predictor.predict(inputs)
    return result["predicted_score"]
