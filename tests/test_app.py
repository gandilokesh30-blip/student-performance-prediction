"""
Unit tests for general application components, insights generation,
edge case inputs, and report format compliance.
"""

import os
import pytest
import pandas as pd
from src.utils import generate_personalized_insights, DISCLAIMER_TEXT
from src.predict import StudentPerformancePredictor


def test_personalized_insights_low_study_and_attendance():
    """Verify appropriate insights triggered for at-risk habits."""
    inputs = {
        "Study_Hours_Per_Day": 1.5,
        "Attendance_Percentage": 60.0,
        "Previous_Exam_Score": 50.0,
        "Assignment_Score": 55.0,
        "Internal_Exam_Score": 52.0,
        "Practice_Test_Score": 48.0,
        "Sleep_Hours_Per_Day": 5.0,
    }
    insights = generate_personalized_insights(inputs)
    assert any("study time" in s.lower() for s in insights)
    assert any("attendance" in s.lower() for s in insights)
    assert any("practice tests" in s.lower() or "mock" in s.lower() for s in insights)
    assert any("sleep" in s.lower() for s in insights)


def test_personalized_insights_balanced_student():
    """Verify well-performing student receives positive feedback."""
    inputs = {
        "Study_Hours_Per_Day": 6.5,
        "Attendance_Percentage": 92.0,
        "Previous_Exam_Score": 88.0,
        "Assignment_Score": 90.0,
        "Internal_Exam_Score": 89.0,
        "Practice_Test_Score": 87.0,
        "Sleep_Hours_Per_Day": 7.5,
    }
    insights = generate_personalized_insights(inputs)
    assert len(insights) >= 1
    assert any("balanced" in s.lower() or "strong" in s.lower() for s in insights)


def test_disclaimer_text_contains_educational_caveat():
    """Verify mandatory disclaimer notice format."""
    assert "synthetic data" in DISCLAIMER_TEXT.lower()
    assert "not be treated as a guaranteed result" in DISCLAIMER_TEXT.lower()


def test_edge_case_zero_inputs():
    """Verify prediction does not fail with edge-case extreme minimums."""
    model_path = os.path.join("models", "best_model.pkl")
    if not os.path.exists(model_path):
        pytest.skip("Model not yet trained.")

    predictor = StudentPerformancePredictor(models_dir="models")
    min_inputs = {
        "Study_Hours_Per_Day": 0.0,
        "Attendance_Percentage": 0.0,
        "Previous_Exam_Score": 0.0,
        "Assignment_Score": 0.0,
        "Internal_Exam_Score": 0.0,
        "Practice_Test_Score": 0.0,
        "Sleep_Hours_Per_Day": 0.0,
    }
    res = predictor.predict(min_inputs)
    assert 0.0 <= res["predicted_score"] <= 100.0


def test_edge_case_max_inputs():
    """Verify prediction does not fail with edge-case maximums."""
    model_path = os.path.join("models", "best_model.pkl")
    if not os.path.exists(model_path):
        pytest.skip("Model not yet trained.")

    predictor = StudentPerformancePredictor(models_dir="models")
    max_inputs = {
        "Study_Hours_Per_Day": 16.0,
        "Attendance_Percentage": 100.0,
        "Previous_Exam_Score": 100.0,
        "Assignment_Score": 100.0,
        "Internal_Exam_Score": 100.0,
        "Practice_Test_Score": 100.0,
        "Sleep_Hours_Per_Day": 12.0,
    }
    res = predictor.predict(max_inputs)
    assert 0.0 <= res["predicted_score"] <= 100.0
