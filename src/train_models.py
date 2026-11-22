"""
Model training, 5-fold cross validation, hyperparameter tuning, model comparison,
evaluation metric generation, and model artifact serialization.
"""

import os
import logging
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import load_dataset, prepare_train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_candidate_pipelines() -> Dict[str, Pipeline]:
    """
    Constructs candidate regression pipelines.
    Scales features for Linear & Ridge models while keeping tree-based models natural.
    """
    pipelines = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("regressor", DecisionTreeRegressor(max_depth=6, min_samples_split=5, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=4, random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("regressor", GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)),
        ]),
    }
    return pipelines


def tune_best_tree_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    """
    Performs grid search tuning on the strongest ensemble models
    (Random Forest and Gradient Boosting) using 5-fold cross-validation.
    """
    logger.info("Executing hyperparameter tuning for ensemble models...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Random Forest Tuning Grid
    rf_param_grid = {
        "regressor__n_estimators": [60, 120],
        "regressor__max_depth": [5, 8, 12],
        "regressor__min_samples_split": [2, 5],
    }
    rf_grid = GridSearchCV(
        Pipeline([("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))]),
        rf_param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    logger.info(f"Tuned Random Forest best params: {rf_grid.best_params_}")

    # Gradient Boosting Tuning Grid
    gb_param_grid = {
        "regressor__n_estimators": [60, 100],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [3, 5],
    }
    gb_grid = GridSearchCV(
        Pipeline([("regressor", GradientBoostingRegressor(random_state=42))]),
        gb_param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    gb_grid.fit(X_train, y_train)
    logger.info(f"Tuned Gradient Boosting best params: {gb_grid.best_params_}")

    tuned_models = {
        "Random Forest (Tuned)": rf_grid.best_estimator_,
        "Gradient Boosting (Tuned)": gb_grid.best_estimator_,
    }
    return tuned_models


def evaluate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Trains each candidate model, runs 5-fold cross-validation on the training set,
    evaluates on the unseen test set, and compiles MAE, MSE, RMSE, R2, and CV RMSE.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    test_predictions = {}

    for name, pipeline in models.items():
        # 1. 5-Fold Cross Validation on Training Data ONLY
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        cv_rmse = float(-np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # 2. Fit on full training set
        pipeline.fit(X_train, y_train)

        # 3. Evaluate on unseen Test Set
        y_pred = pipeline.predict(X_test)
        test_predictions[name] = y_pred

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "CV_RMSE_Mean": round(cv_rmse, 4),
            "CV_RMSE_Std": round(cv_std, 4),
        })
        logger.info(f"{name} -> Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, CV RMSE: {cv_rmse:.4f}")

    results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    return results_df, test_predictions


def plot_model_comparison(results_df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Generates comparison bar charts for MAE, RMSE, and R2 across models."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.barplot(data=results_df, x="Model", y="RMSE", palette="Blues_r", ax=axes[0])
    axes[0].set_title("Test RMSE (Lower is Better)", fontweight="bold")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=30)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")

    sns.barplot(data=results_df, x="Model", y="MAE", palette="Greens_r", ax=axes[1])
    axes[1].set_title("Test MAE (Lower is Better)", fontweight="bold")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=30)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")

    sns.barplot(data=results_df, x="Model", y="R2", palette="Purples_r", ax=axes[2])
    axes[2].set_title("Test R² Score (Higher is Better)", fontweight="bold")
    axes[2].set_ylabel("R²")
    axes[2].tick_params(axis="x", rotation=30)
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_actual_vs_predicted(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str = "reports/figures",
) -> str:
    """Generates Actual vs. Predicted scatter plot with 45-degree reference line."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_test, y_pred, alpha=0.55, color="#2563EB", edgecolors="none", s=40, label="Predictions")
    min_val = min(float(y_test.min()), float(y_pred.min()))
    max_val = max(float(y_test.max()), float(y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], color="#DC2626", linestyle="--", linewidth=2.0, label="Ideal Prediction (y = x)")

    ax.set_title(f"Actual vs Predicted Final Exam Score\n({model_name})", fontweight="bold", pad=12)
    ax.set_xlabel("Actual Final Exam Score")
    ax.set_ylabel("Predicted Final Exam Score")
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "actual_vs_predicted.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_residuals(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str = "reports/figures",
) -> str:
    """Generates Residual Plot (Residuals vs Predicted) to check error variance."""
    os.makedirs(output_dir, exist_ok=True)
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.scatter(y_pred, residuals, alpha=0.55, color="#7C3AED", edgecolors="none", s=40)
    ax.axhline(0, color="#DC2626", linestyle="--", linewidth=2.0, label="Zero Error Line")

    ax.set_title(f"Residual Plot: Prediction Errors vs Fitted Values\n({model_name})", fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Final Exam Score")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "residual_plot.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_feature_importance_or_coefficients(
    pipeline: Pipeline,
    feature_names: list,
    model_name: str,
    output_dir: str = "reports/figures",
) -> str:
    """
    Visualizes feature importance (for tree models) or standardized coefficients
    (for linear models) as a horizontal bar chart with non-causal phrasing.
    """
    os.makedirs(output_dir, exist_ok=True)
    regressor = pipeline.named_steps["regressor"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
        title = f"Relative Feature Importance ({model_name})\n(Associated with model predictions; does not prove causation)"
        xlabel = "Gini / Impurity Reduction Importance"
        data = pd.DataFrame({"Feature": [f.replace("_", " ") for f in feature_names], "Weight": importances})
    elif hasattr(regressor, "coef_"):
        coefs = regressor.coef_
        title = f"Standardized Coefficients ({model_name})\n(Direction and magnitude associated with predictions)"
        xlabel = "Standardized Regression Coefficient"
        data = pd.DataFrame({"Feature": [f.replace("_", " ") for f in feature_names], "Weight": coefs})
    else:
        logger.warning(f"Model {model_name} does not expose feature importances or coefficients.")
        return ""

    data = data.sort_values(by="Weight", ascending=True)
    sns.barplot(data=data, x="Weight", y="Feature", palette="viridis", ax=ax)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Features")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def train_and_select_best_pipeline(
    dataset_path: str = "data/student_study_performance_prediction_dataset.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Main orchestration routine:
    1. Loads dataset and executes train-test split.
    2. Builds base models and tunes ensemble candidates.
    3. Evaluates all models with 5-fold CV and test set metrics.
    4. Selects the champion model based on lowest test RMSE.
    5. Serializes best_model.pkl, scaler.pkl, feature_columns.pkl, feature_medians.pkl.
    6. Saves reports/model_results.csv and diagnostic plots.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Prepare data
    df = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test, train_medians, scaler, feature_cols = prepare_train_test_split(df)

    # 2. Candidate models + tuning
    candidate_models = build_candidate_pipelines()
    tuned_ensembles = tune_best_tree_models(X_train, y_train)
    all_models = {**candidate_models, **tuned_ensembles}

    # 3. Evaluate models
    results_df, test_predictions = evaluate_models(all_models, X_train, y_train, X_test, y_test)
    
    # Save model results CSV
    results_csv_path = os.path.join(reports_dir, "model_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Saved model results to {results_csv_path}")

    # 4. Select champion model (lowest Test RMSE)
    best_row = results_df.iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline = all_models[best_model_name]
    logger.info(f"Champion Model Selected: '{best_model_name}' (RMSE: {best_row['RMSE']}, R2: {best_row['R2']})")

    # 5. Serialize artifacts
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    feat_cols_path = os.path.join(models_dir, "feature_columns.pkl")
    medians_path = os.path.join(models_dir, "feature_medians.pkl")

    joblib.dump(best_pipeline, best_model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, feat_cols_path)
    joblib.dump(train_medians, medians_path)
    logger.info("Saved serialized model artifacts to models/ directory.")

    # 6. Generate evaluation plots
    plot_model_comparison(results_df, output_dir=figures_dir)
    plot_actual_vs_predicted(y_test, test_predictions[best_model_name], best_model_name, output_dir=figures_dir)
    plot_residuals(y_test, test_predictions[best_model_name], best_model_name, output_dir=figures_dir)
    plot_feature_importance_or_coefficients(best_pipeline, feature_cols, best_model_name, output_dir=figures_dir)

    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_metrics": best_row.to_dict(),
        "feature_columns": feature_cols,
    }


if __name__ == "__main__":
    train_and_select_best_pipeline()
