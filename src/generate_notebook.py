"""
Script to generate the student_performance_analysis.ipynb Jupyter Notebook
programmatically using nbformat.
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# ── Cell 0: Title Markdown
cells.append(nbf.v4.new_markdown_cell("""# 🎓 Smart Student Performance Prediction
## Complete ML & Predictive Analytics Notebook

**Target Variable:** `Final_Exam_Score` (Regression)  
**Dataset:** Synthetic student study performance dataset (1,000 records)  
**Author:** B.Tech Final Year Project  

---
"""))

# ── Cell 1: Import Libraries
cells.append(nbf.v4.new_markdown_cell("## 1. Import Libraries"))
cells.append(nbf.v4.new_code_cell("""# Standard Data Science Stack
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams.update({'figure.dpi': 100, 'axes.titlesize': 14})

print("✅ All libraries imported successfully!")
print(f"   pandas: {pd.__version__} | numpy: {np.__version__} | sklearn: imported")
"""))

# ── Cell 2: Load Dataset
cells.append(nbf.v4.new_markdown_cell("## 2. Load Dataset"))
cells.append(nbf.v4.new_code_cell("""DATASET_PATH = 'data/student_study_performance_prediction_dataset.csv'
df_raw = pd.read_csv(DATASET_PATH)

print(f"Dataset Shape: {df_raw.shape}")
print(f"Rows: {df_raw.shape[0]:,}  |  Columns: {df_raw.shape[1]}")
df_raw.head(5)
"""))

# ── Cell 3: Inspect Dataset
cells.append(nbf.v4.new_markdown_cell("## 3. Dataset Inspection"))
cells.append(nbf.v4.new_code_cell("""# Column names and data types
print("Column Names:")
print(df_raw.columns.tolist())
print()
print("Data Types:")
print(df_raw.dtypes)
"""))

cells.append(nbf.v4.new_code_cell("""# Descriptive statistics
df_raw.describe().round(3)
"""))

# ── Cell 4: Data Quality Analysis
cells.append(nbf.v4.new_markdown_cell("## 4. Data Quality Analysis"))
cells.append(nbf.v4.new_code_cell("""# Missing values
print("=== Missing Values ===")
missing = df_raw.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values detected ✅")

# Duplicates
dup_count = df_raw.duplicated().sum()
print(f"\\n=== Duplicate Rows ===")
print(f"Found: {dup_count} duplicate rows")
"""))

cells.append(nbf.v4.new_code_cell("""# Outlier detection using IQR
FEATURE_COLS = [
    'Study_Hours_Per_Day', 'Attendance_Percentage', 'Previous_Exam_Score',
    'Assignment_Score', 'Internal_Exam_Score', 'Practice_Test_Score',
    'Sleep_Hours_Per_Day'
]
TARGET_COL = 'Final_Exam_Score'

print("=== IQR Outlier Report ===")
outlier_summary = []
for col in FEATURE_COLS + [TARGET_COL]:
    q1 = df_raw[col].quantile(0.25)
    q3 = df_raw[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_out = ((df_raw[col] < lo) | (df_raw[col] > hi)).sum()
    outlier_summary.append({'Feature': col, 'Q1': round(q1,2), 'Q3': round(q3,2), 'IQR': round(iqr,2), 'Lower': round(lo,2), 'Upper': round(hi,2), 'Outliers': n_out})

pd.DataFrame(outlier_summary)
"""))

# ── Cell 5: Data Cleaning
cells.append(nbf.v4.new_markdown_cell("## 5. Data Cleaning"))
cells.append(nbf.v4.new_code_cell("""df = df_raw.copy()

# Remove duplicates
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows. Remaining: {len(df)}")

# Median imputation (compute from full dataset for notebook demo; in production: train-only)
for col in FEATURE_COLS:
    if df[col].isnull().any():
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"  Imputed '{col}' with median={med:.2f}")

# Domain clipping
clip_bounds = {
    'Study_Hours_Per_Day': (0, 24), 'Attendance_Percentage': (0, 100),
    'Previous_Exam_Score': (0, 100), 'Assignment_Score': (0, 100),
    'Internal_Exam_Score': (0, 100), 'Practice_Test_Score': (0, 100),
    'Sleep_Hours_Per_Day': (0, 24),
}
for col, (lo, hi) in clip_bounds.items():
    df[col] = df[col].clip(lower=lo, upper=hi)

print(f"\\nFinal cleaned dataset shape: {df.shape}")
print("Data cleaning complete ✅")
"""))

# ── Cell 6: EDA — Distributions
cells.append(nbf.v4.new_markdown_cell("## 6. Exploratory Data Analysis\n### 6.1 Feature Distributions"))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

all_cols = FEATURE_COLS + [TARGET_COL]
for i, col in enumerate(all_cols):
    sns.histplot(df[col], kde=True, color='#2563EB', alpha=0.6, ax=axes[i], edgecolor='#1E3A8A')
    axes[i].axvline(df[col].mean(), color='#DC2626', linestyle='--', lw=2, label=f'Mean: {df[col].mean():.1f}')
    axes[i].axvline(df[col].median(), color='#10B981', linestyle=':', lw=2, label=f'Median: {df[col].median():.1f}')
    axes[i].set_title(col.replace('_', ' '), fontweight='bold')
    axes[i].legend(fontsize=8)
    axes[i].set_xlabel('')

plt.suptitle('Distribution Profiles — Student Academic & Lifestyle Metrics', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('reports/figures/notebook_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("\\nDistribution plots saved ✅")
"""))

# ── Cell 7: EDA — Feature vs Target
cells.append(nbf.v4.new_markdown_cell("### 6.2 Feature vs Final Exam Score Relationships"))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, col in enumerate(FEATURE_COLS):
    sns.regplot(
        data=df, x=col, y=TARGET_COL,
        scatter_kws={'alpha': 0.4, 'color': '#0284C7', 's': 30},
        line_kws={'color': '#DC2626', 'linewidth': 2},
        ax=axes[i]
    )
    corr = df[col].corr(df[TARGET_COL])
    axes[i].set_title(f"{col.replace('_', ' ')}\\n(r = {corr:.3f})", fontweight='bold', fontsize=11)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Final Exam Score' if i % 4 == 0 else '')

axes[-1].set_visible(False)
plt.suptitle('Bivariate Relationships: Predictors vs Final Exam Score', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/notebook_feature_vs_target.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ── Cell 8: Correlation Heatmap
cells.append(nbf.v4.new_markdown_cell("### 6.3 Correlation Heatmap"))
cells.append(nbf.v4.new_code_cell("""numeric_df = df[FEATURE_COLS + [TARGET_COL]]
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', vmin=-1, vmax=1, center=0,
    square=True, linewidths=0.8,
    cbar_kws={'shrink': 0.8}, ax=ax
)
ax.set_title('Pearson Correlation Heatmap — Student Performance Dataset', fontweight='bold', pad=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('reports/figures/notebook_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nTop correlations with Final_Exam_Score:")
print(corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False).round(3))
"""))

# ── Cell 9: Feature Engineering
cells.append(nbf.v4.new_markdown_cell("## 7. Feature Engineering\n> **Anti-Leakage Rule:** `Final_Exam_Score` is NEVER used to create any input feature."))
cells.append(nbf.v4.new_code_cell("""df_feat = df.copy()

# Feature 1: Weekly study time
df_feat['Study_Hours_Per_Week'] = (df_feat['Study_Hours_Per_Day'] * 7).round(2)

# Feature 2: Average continuous assessment score
df_feat['Average_Internal_Performance'] = df_feat[
    ['Assignment_Score', 'Internal_Exam_Score', 'Practice_Test_Score']
].mean(axis=1).round(2)

# Feature 3: Weighted academic preparation composite
df_feat['Academic_Preparation_Score'] = (
    df_feat['Previous_Exam_Score'] * 0.40
    + df_feat['Practice_Test_Score'] * 0.35
    + df_feat['Internal_Exam_Score'] * 0.25
).round(2)

print("Engineered Features:")
print(df_feat[['Study_Hours_Per_Week', 'Average_Internal_Performance', 'Academic_Preparation_Score']].describe().round(2))
"""))

# ── Cell 10: Train/Test Split
cells.append(nbf.v4.new_markdown_cell("## 8. Train / Test Split (80/20, random_state=42)"))
cells.append(nbf.v4.new_code_cell("""# Exclude Student_ID from features
feature_cols = [c for c in df_feat.columns if c not in ['Student_ID', TARGET_COL]]
X = df_feat[feature_cols]
y = df_feat[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Test set:     {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"Features used: {len(feature_cols)}")
print(f"\\nFeature list: {feature_cols}")
"""))

# ── Cell 11: Model Training
cells.append(nbf.v4.new_markdown_cell("## 9. Model Training & 5-Fold Cross Validation"))
cells.append(nbf.v4.new_code_cell("""cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Linear Regression': Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())]),
    'Ridge Regression':  Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))]),
    'Decision Tree':     Pipeline([('reg', DecisionTreeRegressor(max_depth=6, random_state=42))]),
    'Random Forest':     Pipeline([('reg', RandomForestRegressor(n_estimators=100, random_state=42))]),
    'Gradient Boosting': Pipeline([('reg', GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42))]),
}

results = []
for name, pipeline in models.items():
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
    cv_rmse = -cv_scores.mean()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    results.append({'Model': name, 'MAE': round(mae,4), 'MSE': round(mse,4), 'RMSE': round(rmse,4), 'R2': round(r2,4), 'CV_RMSE': round(cv_rmse,4)})
    print(f"  {name:<25} RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  CV-RMSE={cv_rmse:.4f}")

results_df = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
"""))

# ── Cell 12: Hyperparameter Tuning
cells.append(nbf.v4.new_markdown_cell("## 10. Hyperparameter Tuning (GridSearchCV)"))
cells.append(nbf.v4.new_code_cell("""print("Tuning Gradient Boosting...")
gb_grid = GridSearchCV(
    Pipeline([('reg', GradientBoostingRegressor(random_state=42))]),
    {'reg__n_estimators': [80, 120], 'reg__learning_rate': [0.05, 0.1], 'reg__max_depth': [3, 5]},
    cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1
)
gb_grid.fit(X_train, y_train)
print(f"Best GB params: {gb_grid.best_params_}")

best_model = gb_grid.best_estimator_
y_pred_best = best_model.predict(X_test)

print(f"\\nTuned Gradient Boosting Test Results:")
print(f"  MAE:  {mean_absolute_error(y_test, y_pred_best):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}")
print(f"  R²:   {r2_score(y_test, y_pred_best):.4f}")
"""))

# ── Cell 13: Model Comparison Table
cells.append(nbf.v4.new_markdown_cell("## 11. Model Comparison"))
cells.append(nbf.v4.new_code_cell("""results_df.style.highlight_min(subset=['RMSE', 'MAE'], color='#DCFCE7').highlight_max(subset=['R2'], color='#DCFCE7')
"""))

# ── Cell 14: Model Comparison Chart
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, metric, palette in zip(axes, ['RMSE', 'MAE', 'R2'], ['Blues_r', 'Greens_r', 'Purples_r']):
    bars = ax.bar(results_df['Model'], results_df[metric], color=sns.color_palette(palette, len(results_df)))
    ax.set_title(f"{'Lower is Better ↓' if metric != 'R2' else 'Higher is Better ↑'}\\n{metric}", fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.3f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Algorithm Performance Benchmark — Student Final Exam Score Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/notebook_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ── Cell 15: Actual vs Predicted
cells.append(nbf.v4.new_markdown_cell("## 12. Actual vs Predicted Analysis"))
cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs Predicted
ax1.scatter(y_test, y_pred_best, alpha=0.5, color='#2563EB', s=40, edgecolors='none')
lo, hi = y_test.min(), y_test.max()
ax1.plot([lo, hi], [lo, hi], color='#DC2626', linestyle='--', lw=2, label='Ideal (y=x)')
ax1.set_title('Actual vs Predicted Final Exam Score\\n(Gradient Boosting Tuned)', fontweight='bold')
ax1.set_xlabel('Actual Final Exam Score')
ax1.set_ylabel('Predicted Final Exam Score')
ax1.legend()

# Residuals
residuals = y_test - y_pred_best
ax2.scatter(y_pred_best, residuals, alpha=0.5, color='#7C3AED', s=40, edgecolors='none')
ax2.axhline(0, color='#DC2626', linestyle='--', lw=2)
ax2.set_title('Residual Plot\\n(Errors vs Fitted Values)', fontweight='bold')
ax2.set_xlabel('Predicted Value')
ax2.set_ylabel('Residual (Actual − Predicted)')

plt.tight_layout()
plt.savefig('reports/figures/notebook_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Mean Residual: {residuals.mean():.4f} (should be near 0)")
print(f"Std Residual:  {residuals.std():.4f}")
"""))

# ── Cell 16: Feature Importance
cells.append(nbf.v4.new_markdown_cell("## 13. Feature Importance\n> Feature importances indicate **statistical association** with model predictions — not causation."))
cells.append(nbf.v4.new_code_cell("""regressor = best_model.named_steps['reg']
importances = regressor.feature_importances_

feat_imp_df = pd.DataFrame({'Feature': [f.replace('_', ' ') for f in X_train.columns], 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color=sns.color_palette('viridis', len(feat_imp_df)))
ax.set_title('Feature Importance — Gradient Boosting (Tuned)\\nAssociated with model predictions; does not prove causation', fontweight='bold', fontsize=12)
ax.set_xlabel('Gini Impurity Reduction Importance')
plt.tight_layout()
plt.savefig('reports/figures/notebook_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ── Cell 17: Save Model
cells.append(nbf.v4.new_markdown_cell("## 14. Model Serialization"))
cells.append(nbf.v4.new_code_cell("""os.makedirs('models', exist_ok=True)

joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')
print("✅ Model and feature columns saved to models/")
print(f"   Best Model: Gradient Boosting (Tuned)")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}")
print(f"   R²:   {r2_score(y_test, y_pred_best):.4f}")
"""))

# ── Cell 18: Prediction Demo
cells.append(nbf.v4.new_markdown_cell("## 15. Live Prediction Demo"))
cells.append(nbf.v4.new_code_cell("""import sys
sys.path.insert(0, '.')
from src.predict import StudentPerformancePredictor

predictor = StudentPerformancePredictor(models_dir='models')

sample_student = {
    'Study_Hours_Per_Day': 6.5,
    'Attendance_Percentage': 88.0,
    'Previous_Exam_Score': 78.0,
    'Assignment_Score': 82.0,
    'Internal_Exam_Score': 79.0,
    'Practice_Test_Score': 76.0,
    'Sleep_Hours_Per_Day': 7.5,
}

result = predictor.predict(sample_student)

print("=" * 55)
print(f"  PREDICTED FINAL EXAM SCORE: {result['predicted_score']:.2f} / 100")
print(f"  PERFORMANCE CATEGORY:       {result['category']}")
print("=" * 55)
print("\\nPersonalized Recommendations:")
for tip in result['personalized_insights']:
    print(f"  • {tip}")
print(f"\\n⚠️ {result['disclaimer']}")
"""))

# ── Cell 19: Conclusion
cells.append(nbf.v4.new_markdown_cell("""## 16. Conclusion

This notebook demonstrates a complete end-to-end machine learning pipeline:

| Step | Status |
|---|---|
| Data Loading & Inspection | ✅ Complete |
| Missing Value Treatment (Median Imputation) | ✅ Complete |
| Duplicate Detection | ✅ Complete |
| Outlier Analysis (IQR Method) | ✅ Complete |
| EDA: Distributions, Scatter Plots, Heatmap | ✅ Complete |
| Feature Engineering (No Target Leakage) | ✅ Complete |
| 80/20 Train/Test Split (random_state=42) | ✅ Complete |
| 5 Regression Models Trained | ✅ Complete |
| 5-Fold Cross Validation | ✅ Complete |
| Hyperparameter Tuning (GridSearchCV) | ✅ Complete |
| Model Evaluation (MAE, RMSE, R²) | ✅ Complete |
| Actual vs Predicted & Residual Analysis | ✅ Complete |
| Feature Importance Visualization | ✅ Complete |
| Model Serialization (joblib) | ✅ Complete |
| Live Prediction Demo | ✅ Complete |

**Champion Model:** Gradient Boosting (Tuned)  
**Note:** Results are derived from a synthetic dataset and are for educational demonstration purposes only.
"""))

nb.cells = cells

output_path = 'notebooks/student_performance_analysis.ipynb'
os.makedirs('notebooks', exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook generated at: {output_path}")
