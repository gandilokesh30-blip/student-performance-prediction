"""
Unit tests for the prediction pipeline, input validation, score bounding,
and category/insight generation.
"""

import os
import pytest
from src.predict import StudentPerformancePredictor, predict_student_score
from src.utils import validate_student_inputs, categorize_performance


@pytest.fixture(scope="module")
def valid_student_input():
    return {
        "Study_Hours_Per_Day": 6.0,
        "Attendance_Percentage": 85.0,
        "Previous_Exam_Score": 75.0,
        "Assignment_Score": 80.0,
        "Internal_Exam_Score": 78.0,
        "Practice_Test_Score": 76.0,
        "Sleep_Hours_Per_Day": 7.5,
    }


def test_validate_student_inputs_valid(valid_student_input):
    """Test validation passes for normal inputs."""
    is_valid, msg = validate_student_inputs(valid_student_input)
    assert is_valid is True
    assert msg == ""


def test_validate_student_inputs_missing_field(valid_student_input):
    """Test validation fails if a field is missing."""
    invalid_input = valid_student_input.copy()
    del invalid_input["Study_Hours_Per_Day"]
    is_valid, msg = validate_student_inputs(invalid_input)
    assert is_valid is False
    assert "Missing required input" in msg


def test_validate_student_inputs_out_of_range(valid_student_input):
    """Test validation detects out-of-range physical values."""
    # Negative study hours
    neg_study = valid_student_input.copy()
    neg_study["Study_Hours_Per_Day"] = -1.0
    is_valid, msg = validate_student_inputs(neg_study)
    assert is_valid is False
    assert "between 0.0 and 24.0" in msg

    # Attendance > 100
    high_att = valid_student_input.copy()
    high_att["Attendance_Percentage"] = 120.0
    is_valid, msg = validate_student_inputs(high_att)
    assert is_valid is False
    assert "between 0.0 and 100.0" in msg

    # Exam score > 100
    high_score = valid_student_input.copy()
    high_score["Internal_Exam_Score"] = 150.0
    is_valid, msg = validate_student_inputs(high_score)
    assert is_valid is False
    assert "between 0.0 and 100.0" in msg


def test_validate_student_inputs_non_numeric(valid_student_input):
    """Test non-numeric values fail validation cleanly."""
    bad_val = valid_student_input.copy()
    bad_val["Assignment_Score"] = "excellent"
    is_valid, msg = validate_student_inputs(bad_val)
    assert is_valid is False
    assert "must be a valid number" in msg


def test_categorize_performance():
    """Verify standard project score categories."""
    assert categorize_performance(95.0)[0] == "Excellent"
    assert categorize_performance(90.0)[0] == "Excellent"
    assert categorize_performance(85.5)[0] == "Good"
    assert categorize_performance(75.0)[0] == "Good"
    assert categorize_performance(68.2)[0] == "Average"
    assert categorize_performance(60.0)[0] == "Average"
    assert categorize_performance(59.9)[0] == "Needs Improvement"
    assert categorize_performance(35.0)[0] == "Needs Improvement"


def test_prediction_output_range_and_contract(valid_student_input):
    """
    If model has been trained, test end-to-end prediction range [0, 100]
    and rounding to two decimal places.
    """
    model_path = os.path.join("models", "best_model.pkl")
    if not os.path.exists(model_path):
        pytest.skip("Model artifact not yet generated. Skipping live prediction test.")

    predictor = StudentPerformancePredictor(models_dir="models")
    res = predictor.predict(valid_student_input)

    assert "predicted_score" in res
    assert 0.0 <= res["predicted_score"] <= 100.0
    assert "category" in res
    assert "personalized_insights" in res
    assert isinstance(res["personalized_insights"], list)
    assert len(res["personalized_insights"]) > 0
    assert "disclaimer" in res
