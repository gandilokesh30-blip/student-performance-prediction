"""
Generates the System Architecture diagram image for the project documentation.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def generate_architecture_diagram(output_path: str = "reports/figures/system_architecture.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis("off")

    stages = [
        ("User / Student / Educator", "Inputs academic scores, study routine & sleep habits", "#DBEAFE", "#1D4ED8"),
        ("Streamlit Web Dashboard", "Interactive UI with sliders, reactive widgets & charts", "#E0E7FF", "#4338CA"),
        ("Input Validation Layer", "Domain boundary checks (0-24 hrs, 0-100 scores)", "#FEF3C7", "#D97706"),
        ("Data Cleaning & Preprocessing", "Median imputation, domain clipping, StandardScaler", "#DCFCE7", "#15803D"),
        ("Feature Engineering", "Weekly study hours, average internal score, prep index", "#F3E8FF", "#7E22CE"),
        ("Trained ML Model (Champion)", "Supervised Regression (Random Forest / Gradient Boosting)", "#FCE7F3", "#BE185D"),
        ("Prediction & Bounding", "Continuous score estimation clamped strictly to [0, 100]", "#E0F2FE", "#0284C7"),
        ("Performance Interpretation", "Project categorization (Excellent / Good / Average / Needs Imp)", "#FEF9C3", "#A16207"),
        ("Dashboard Output & Analytics", "Visual metric cards, personalized recommendations & CSV download", "#CCFBF1", "#0F766E"),
    ]

    y_pos = 0.92
    box_height = 0.065
    box_width = 0.84
    x_left = 0.08

    for i, (title, subtitle, bg_color, border_color) in enumerate(stages):
        # Draw box
        rect = patches.FancyBboxPatch(
            (x_left, y_pos - box_height),
            box_width,
            box_height,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=2.0,
        )
        ax.add_patch(rect)

        # Add text
        ax.text(
            0.5,
            y_pos - box_height * 0.35,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#1F2937",
        )
        ax.text(
            0.5,
            y_pos - box_height * 0.75,
            subtitle,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#4B5563",
        )

        # Draw connecting arrow
        if i < len(stages) - 1:
            arrow_start = y_pos - box_height - 0.005
            arrow_end = y_pos - box_height - 0.035
            ax.annotate(
                "",
                xy=(0.5, arrow_end),
                xytext=(0.5, arrow_start),
                arrowprops=dict(facecolor="#4B5563", edgecolor="#4B5563", width=2.0, headwidth=8),
            )

        y_pos -= 0.105

    plt.title("System Architecture: Smart Student Performance Prediction", fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Architecture diagram generated at: {output_path}")


if __name__ == "__main__":
    generate_architecture_diagram()
