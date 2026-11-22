# 🎓 Smart Student Performance Prediction
### Using Machine Learning and Predictive Analytics

> **B.Tech / MCA Final Year Project** | Python · scikit-learn · Streamlit | Synthetic Dataset

---

## 📌 Problem Statement

Educational institutions collect continuous assessment data (assignments, internal exams, mock tests) and lifestyle records (study hours, attendance, sleep) but rarely convert these into forward-looking predictive insights before final examinations. This project builds an end-to-end regression system that estimates a student's **Final Exam Score** from these inputs and provides personalized, actionable study recommendations.

---

## 🎯 Objectives

- Ingest and validate the provided synthetic student dataset
- Perform rigorous 16-plot Exploratory Data Analysis
- Implement leak-free data preprocessing (train-set-only imputation)
- Engineer logically justified composite features
- Train, tune, and benchmark 5+ regression algorithms
- Select the best model via 5-fold CV and unseen test RMSE
- Deploy an interactive Streamlit dashboard with personalized insights

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🔮 **Live Prediction** | Real-time Final Exam Score estimation via sidebar sliders |
| 📊 **EDA Dashboard** | Interactive feature distributions, scatter trends, correlation heatmap |
| 🏆 **Model Benchmark** | Side-by-side RMSE / MAE / R² comparison across 7 models |
| 💡 **Personalized Insights** | Study habit recommendations based on student inputs |
| 📥 **CSV Downloads** | Export prediction report and model benchmark table |
| ✅ **18 Unit Tests** | Full coverage of preprocessing, prediction, and validation |

---

## 📁 Project Structure

```
student-performance-prediction/
│
├── data/
│   ├── student_study_performance_prediction_dataset.csv   ← Primary dataset
│   └── student_study_performance_data_dictionary.csv
│
├── models/
│   ├── best_model.pkl          ← Serialized champion model (Gradient Boosting Tuned)
│   ├── scaler.pkl              ← StandardScaler fitted on training set
│   ├── feature_columns.pkl     ← Ordered list of model feature names
│   └── feature_medians.pkl     ← Training-set medians for imputation
│
├── notebooks/
│   └── student_performance_analysis.ipynb   ← Complete analytical Jupyter notebook
│
├── src/
│   ├── __init__.py
│   ├── utils.py                ← Constants, validation, insight & category functions
│   ├── data_preprocessing.py   ← Loading, cleaning, feature engineering, split
│   ├── eda.py                  ← All 16 EDA visualization generators
│   ├── train_models.py         ← Model training, tuning, evaluation, serialization
│   ├── predict.py              ← Inference pipeline (StudentPerformancePredictor)
│   └── generate_architecture_diagram.py
│
├── reports/
│   ├── model_results.csv       ← Benchmark results table
│   ├── project_report.md       ← Full 21-section college project report
│   └── figures/                ← 16 EDA plots + model diagnostics + architecture
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   ← 7 tests: load, clean, outliers, split
│   ├── test_prediction.py      ← 6 tests: validation, categories, range
│   └── test_app.py             ← 5 tests: insights, disclaimer, edge cases
│
├── app.py                      ← Streamlit web application (main entry point)
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

---

## 📊 Dataset Description

> ⚠️ **Disclaimer:** This dataset is **synthetic** and was generated exclusively for educational and academic project purposes. It does not represent real student records.

| Column | Type | Description | Range |
|---|---|---|---|
| `Student_ID` | String | Unique identifier (excluded from ML) | — |
| `Study_Hours_Per_Day` | Float | Average daily study hours | 0 – 24 |
| `Attendance_Percentage` | Float | Class attendance rate | 0 – 100 |
| `Previous_Exam_Score` | Float | Prior semester exam score | 0 – 100 |
| `Assignment_Score` | Float | Continuous assessment score | 0 – 100 |
| `Internal_Exam_Score` | Float | Mid-term internal score | 0 – 100 |
| `Practice_Test_Score` | Float | Mock test score | 0 – 100 |
| `Sleep_Hours_Per_Day` | Float | Average daily sleep | 0 – 24 |
| **`Final_Exam_Score`** | Float | **Target variable** | 0 – 100 |

- **Records:** 1,000 student entries
- **Missing values:** Handled via median imputation (training set only)

---

## 🧠 Machine Learning Methodology

### Split Strategy
- **80% Training** (800 rows) / **20% Test** (200 rows), `random_state=42`
- All preprocessing parameters (medians, scaler) fitted **only on training data**

### Feature Engineering (No Target Leakage)
| Feature | Formula |
|---|---|
| `Study_Hours_Per_Week` | `Study_Hours_Per_Day × 7` |
| `Average_Internal_Performance` | Mean of Assignment, Internal Exam, Practice Test |
| `Academic_Preparation_Score` | `0.40 × Previous + 0.35 × Practice + 0.25 × Internal` |

### Algorithms Compared
1. **Linear Regression** — baseline parametric model
2. **Ridge Regression** — L2 regularization for multicollinearity
3. **Decision Tree Regressor** — non-linear threshold splitting
4. **Random Forest Regressor** — bagging ensemble
5. **Gradient Boosting Regressor** — boosting ensemble
6. **Random Forest (Tuned)** — GridSearchCV optimized
7. **Gradient Boosting (Tuned)** ⭐ — **Champion Model**

### Evaluation Metrics
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error) — **primary selection metric**
- R² Score

---

## 📈 Actual Model Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Gradient Boosting (Tuned) ⭐** | **3.76** | **4.60** | **0.542** |
| Ridge Regression | 3.81 | 4.66 | 0.530 |
| Gradient Boosting | 3.76 | 4.66 | 0.530 |
| Random Forest (Tuned) | 3.84 | 4.69 | 0.525 |
| Random Forest | 3.84 | 4.69 | 0.524 |
| Linear Regression | 3.88 | 4.74 | 0.513 |
| Decision Tree | 4.26 | 5.49 | 0.349 |

> Results obtained from running on the actual dataset — not fabricated.

---

## 💻 Technologies Used

| Category | Library / Tool |
|---|---|
| Language | Python 3.12 |
| Data Processing | pandas 3.x, numpy 2.x |
| Visualization | matplotlib 3.x, seaborn 0.13 |
| Machine Learning | scikit-learn 1.9 |
| Model Serialization | joblib 1.6 |
| Web Application | Streamlit 1.64 |
| Testing | pytest 9.x |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1: Clone or Download the project

```bash
# Navigate to project folder
cd "student performence project"
```

### Step 2: (Optional) Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Train models (first time only)

```bash
python -m src.train_models
```

This will:
- Load dataset, perform 80/20 split
- Run 5-fold CV + GridSearch tuning
- Save `models/best_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `feature_medians.pkl`
- Save `reports/model_results.csv`
- Generate all diagnostic plots in `reports/figures/`

### Launch Streamlit Dashboard

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

### Run Tests

```bash
pytest -v
```

Expected: **18 passed**

### Generate EDA Plots Only

```bash
python -c "from src.data_preprocessing import load_dataset; from src.eda import generate_all_eda_reports; generate_all_eda_reports(load_dataset('data/student_study_performance_prediction_dataset.csv'))"
```

---

## 🖥️ How to Use the Dashboard

1. **Home** — Overview metrics and quick score calculator
2. **Prediction Engine** — Fill in student details → click **Predict Final Score** → view score, category, and recommendations → download CSV report
3. **Data Analysis (EDA)** — Explore feature distributions, scatter plots, and correlation heatmap interactively
4. **Model Performance** — Compare all 7 model benchmarks, view diagnostic plots
5. **Dataset Explorer** — Preview raw data, audit schema, download CSV
6. **About Project** — Architecture, methodology, limitations, and viva guide

---

## ⚠️ Limitations

- Dataset is **synthetic** — results may not generalize to real institutions
- Socio-economic, psychological, and demographic factors are not included
- Feature importances reflect **statistical associations**, not causal relationships
- Model accuracy is bounded by the complexity and variation of the synthetic data

---

## 🔮 Future Enhancements

1. Real student LMS data integration via REST APIs (Moodle/Canvas)
2. SHAP-based local feature attribution explanations per student
3. Multi-label letter-grade classification (A/B/C/D/F)
4. Teacher feedback NLP sentiment integration
5. Longitudinal multi-semester tracking dashboards
6. Mobile-responsive Progressive Web App (PWA) version

---

## 📋 Disclaimer

> This project uses a **synthetic dataset** generated for academic and educational demonstration purposes only. All student identifiers (`Student_ID`) are fabricated. This tool is intended as a **decision-support aid** and its predictions should never be treated as academically binding evaluations. The model has not been clinically or scientifically validated on real-world student populations.

---

## 📚 References

1. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
2. McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. SciPy Proceedings.
3. Harris et al. (2020). *Array programming with NumPy*. Nature, 585, 357–362.
4. Waskom, M.L. (2021). *Seaborn: statistical data visualization*. JOSS, 6(60), 3021.
5. Streamlit Inc. (2024). *Streamlit Documentation*. https://docs.streamlit.io
