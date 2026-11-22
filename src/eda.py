"""
Exploratory Data Analysis (EDA) module for generating descriptive visualizations,
distribution plots, feature-target relationships, and correlation matrices.
All plots are rendered with professional styling and saved to reports/figures/.
"""

import os
import logging
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import RAW_FEATURE_COLUMNS, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Modern theme configuration
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
})


def ensure_dir(directory: str):
    """Ensures that the target directory exists."""
    os.makedirs(directory, exist_ok=True)


def plot_distributions(df: pd.DataFrame, output_dir: str = "reports/figures") -> List[str]:
    """
    Generates and saves distribution plots (histogram + KDE) for all 7 raw features
    plus the target Final_Exam_Score (Items 1 to 8 in EDA checklist).
    """
    ensure_dir(output_dir)
    generated_files = []
    
    all_numeric = [c for c in RAW_FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]

    for col in all_numeric:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(
            data=df,
            x=col,
            kde=True,
            color="#2563EB",
            edgecolor="#1E3A8A",
            alpha=0.6,
            ax=ax,
        )
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="#10B981", linestyle=":", linewidth=2.0, label=f"Median: {median_val:.2f}")

        formatted_name = col.replace("_", " ")
        ax.set_title(f"Distribution of {formatted_name}", fontweight="bold", pad=12)
        ax.set_xlabel(formatted_name)
        ax.set_ylabel("Student Count / Frequency")
        ax.legend(frameon=True, facecolor="white", loc="best")
        plt.tight_layout()

        filename = f"dist_{col.lower()}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        generated_files.append(filepath)

    logger.info(f"Generated {len(generated_files)} distribution plots in {output_dir}.")
    return generated_files


def plot_feature_vs_target(df: pd.DataFrame, output_dir: str = "reports/figures") -> List[str]:
    """
    Generates and saves bivariate scatter/regression plots for all 7 input features
    against the target Final_Exam_Score (Items 9 to 15 in EDA checklist).
    """
    ensure_dir(output_dir)
    generated_files = []

    for col in RAW_FEATURE_COLUMNS:
        if col not in df.columns or TARGET_COLUMN not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.8))
        sns.regplot(
            data=df,
            x=col,
            y=TARGET_COLUMN,
            scatter_kws={"alpha": 0.45, "color": "#0284C7", "s": 35},
            line_kws={"color": "#DC2626", "linewidth": 2.2, "label": "Trend Line"},
            ax=ax,
        )
        corr_val = df[col].corr(df[TARGET_COLUMN])
        col_name = col.replace("_", " ")
        
        ax.set_title(f"{col_name} vs Final Exam Score (r = {corr_val:.2f})", fontweight="bold", pad=12)
        ax.set_xlabel(col_name)
        ax.set_ylabel("Final Exam Score")
        ax.legend(loc="upper left", frameon=True)
        plt.tight_layout()

        filename = f"rel_{col.lower()}_vs_final.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        generated_files.append(filepath)

    logger.info(f"Generated {len(generated_files)} feature vs target plots in {output_dir}.")
    return generated_files


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """
    Generates and saves Pearson correlation heatmap for numeric features and target
    (Item 16 in EDA checklist).
    """
    ensure_dir(output_dir)
    numeric_cols = [c for c in RAW_FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Student Academic & Behavioral Factors", fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    logger.info(f"Saved correlation heatmap to {filepath}.")
    return filepath


def generate_all_eda_reports(df: pd.DataFrame, output_dir: str = "reports/figures") -> List[str]:
    """Runs all 16 EDA plot routines and returns all file paths."""
    all_files = []
    all_files.extend(plot_distributions(df, output_dir=output_dir))
    all_files.extend(plot_feature_vs_target(df, output_dir=output_dir))
    all_files.append(plot_correlation_heatmap(df, output_dir=output_dir))
    return all_files
