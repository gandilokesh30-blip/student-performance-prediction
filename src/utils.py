"""
Utility functions, constants, validation routines, and insight generation
for the Student Performance Prediction system.
"""

from typing import Dict, Any, Tuple, List

# Feature Definitions
RAW_FEATURE_COLUMNS: List[str] = [
    "Study_Hours_Per_Day",
    "Attendance_Percentage",
    "Previous_Exam_Score",
    "Assignment_Score",
    "Internal_Exam_Score",
    "Practice_Test_Score",
    "Sleep_Hours_Per_Day",
]

TARGET_COLUMN: str = "Final_Exam_Score"
ID_COLUMN: str = "Student_ID"

# Feature boundaries for domain validation
FEATURE_CONSTRAINTS: Dict[str, Tuple[float, float]] = {
    "Study_Hours_Per_Day": (0.0, 24.0),
    "Attendance_Percentage": (0.0, 100.0),
    "Previous_Exam_Score": (0.0, 100.0),
    "Assignment_Score": (0.0, 100.0),
    "Internal_Exam_Score": (0.0, 100.0),
    "Practice_Test_Score": (0.0, 100.0),
    "Sleep_Hours_Per_Day": (0.0, 24.0),
}

DISCLAIMER_TEXT: str = (
    "Prediction is an estimate produced by a machine-learning model trained on synthetic data "
    "and should not be treated as a guaranteed result. This tool is designed for educational "
    "and academic decision-support purposes."
)


def validate_student_inputs(inputs: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates user input values against expected numeric types and boundary constraints.

    Args:
        inputs: Dictionary of feature name to numeric value.

    Returns:
        Tuple of (is_valid, error_message).
    """
    for col, (min_val, max_val) in FEATURE_CONSTRAINTS.items():
        if col not in inputs:
            return False, f"Missing required input parameter: '{col}'."
        
        val = inputs[col]
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            return False, f"Value for '{col}' must be a valid number, got {val}."
        
        if val_float < min_val or val_float > max_val:
            return (
                False,
                f"'{col}' must be between {min_val} and {max_val}. Received: {val_float}.",
            )
            
    return True, ""


def categorize_performance(score: float) -> Tuple[str, str, str]:
    """
    Categorizes predicted score into a project-defined interpretation level.

    Args:
        score: Predicted final exam score (0 - 100).

    Returns:
        Tuple of (Category Name, Color Hex Code, Badge Description)
    """
    if score >= 90.0:
        return "Excellent", "#10B981", "Outstanding academic mastery demonstrated across indicators."
    elif score >= 75.0:
        return "Good", "#3B82F6", "Strong, consistent academic performance with solid foundation."
    elif score >= 60.0:
        return "Average", "#F59E0B", "Moderate performance with clear scope for targeted improvement."
    else:
        return "Needs Improvement", "#EF4444", "Critical focus required on key study habits and foundational revision."


def generate_personalized_insights(inputs: Dict[str, float]) -> List[str]:
    """
    Generates actionable, non-prescriptive, data-driven academic suggestions
    based on the student's study habits and interim test scores.

    Args:
        inputs: Dictionary of input values.

    Returns:
        List of actionable suggestion strings.
    """
    insights: List[str] = []

    study_hours = float(inputs.get("Study_Hours_Per_Day", 0))
    attendance = float(inputs.get("Attendance_Percentage", 0))
    practice = float(inputs.get("Practice_Test_Score", 0))
    sleep = float(inputs.get("Sleep_Hours_Per_Day", 0))
    previous = float(inputs.get("Previous_Exam_Score", 0))
    assignment = float(inputs.get("Assignment_Score", 0))
    internal = float(inputs.get("Internal_Exam_Score", 0))

    if study_hours < 4.0:
        insights.append(
            "Consider increasing consistent daily study time to reinforce retention and problem-solving readiness."
        )
    elif study_hours > 10.0:
        insights.append(
            "High study hours detected. Balancing high study volume with adequate rest helps avoid cognitive fatigue."
        )

    if attendance < 75.0:
        insights.append(
            "Maintaining regular attendance (75%+ recommended) supports academic consistency and live concept discussions."
        )

    if practice < 65.0:
        insights.append(
            "Taking additional mock and practice tests under timed conditions may help identify syllabus gaps."
        )

    if sleep < 6.5:
        insights.append(
            "Adequate restorative sleep (7–8 hours) supports cognitive recall and concentration during examinations."
        )
    elif sleep > 10.0:
        insights.append(
            "Optimizing daily schedule routines may free up dedicated focus blocks for study and practice."
        )

    if internal < 65.0 or assignment < 65.0:
        insights.append(
            "Reviewing internal exam feedback and assignment rubrics can clarify specific areas needing revision."
        )

    if previous < 60.0:
        insights.append(
            "Strengthening core conceptual foundations from previous semesters can create positive momentum for the final exam."
        )

    if not insights:
        insights.append(
            "Student exhibits balanced study routines and strong continuous evaluation scores across all metrics."
        )

    return insights
