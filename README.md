# 🫀 Heart Disease Prediction Using Machine Learning

> **Minor Project 1 — Supervised Machine Learning**  
> Course: Artificial Intelligence | Deadline: 27 June 2026

---

## 📌 Problem Statement

Cardiovascular disease is one of the leading causes of death globally. Early and accurate prediction of heart disease can help physicians intervene in time and potentially save lives.

**Goal:** Build a supervised machine learning model that predicts whether a patient is likely to have heart disease based on 13 clinical features, and compare the performance of multiple ML algorithms.

---

## 📂 Dataset

| Detail | Info |
|--------|------|
| **Name** | Heart Disease UCI Dataset |
| **Source** | [Kaggle — johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Original Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) |
| **Rows** | 302 (after removing 723 duplicates from 1025 raw rows) |
| **Features** | 13 input features + 1 target column |
| **Task Type** | Binary Classification |

### Feature Description

| Feature | Description |
|---------|-------------|
| `age` | Age of the patient (years) |
| `sex` | Sex (1 = Male, 0 = Female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| `restecg` | Resting ECG results (0, 1, 2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = Yes, 0 = No) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (0, 1, 2) |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect) |
| `target` | **Heart disease present (1 = Yes, 0 = No)** ← Target variable |

---

## 🧹 Data Preprocessing

1. **Missing Values:** No missing values found in the dataset
2. **Duplicates:** Removed 723 duplicate rows (1025 → 302 clean rows)
3. **Feature Encoding:** Not required — all features already numerically encoded
4. **Feature Scaling:** Applied `StandardScaler` on features used by Logistic Regression to normalize values
5. **Train-Test Split:** 80% training / 20% testing with `stratify=y` to preserve class balance

---

## 📊 Exploratory Data Analysis (EDA)

EDA plots are saved in the `results/` folder:

| Plot | File |
|------|------|
| Target class distribution | `01_target_distribution.png` |
| Age distribution by disease | `02_age_distribution.png` |
| Categorical features vs target | `03_categorical_features.png` |
| Numerical features boxplots | `04_boxplots.png` |
| Feature correlation heatmap | `05_correlation_heatmap.png` |

### Key EDA Insights

- The dataset is nearly balanced (~52% disease, ~48% no disease)
- **Chest pain type (`cp`)** is the strongest predictor of heart disease
- Patients **with** heart disease tend to have **higher max heart rate (`thalach`)** — counterintuitive but medically explained by different stress response
- **ST depression (`oldpeak`)** is significantly lower in disease patients
- `ca` (vessels colored) and `thal` show strong correlation with the target

---

## 🤖 Model Development

Three supervised ML algorithms were trained and compared:

| Model | Notes |
|-------|-------|
| **Logistic Regression** | Baseline linear classifier; trained on scaled features |
| **Decision Tree** | Non-linear; `max_depth=5` to prevent overfitting |
| **Random Forest** | Ensemble of 200 trees; primary model |

---

## 📈 Evaluation Metrics & Results

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | **0.8033** | 0.8000 | **0.8485** | **0.8235** | — |
| Decision Tree | 0.7705 | **0.8065** | 0.7576 | 0.7812 | — |
| Random Forest | 0.7541 | 0.7647 | 0.7879 | 0.7761 | — |

> Full evaluation metrics saved in `results/evaluation_metrics.csv`

### Result Plots

| Plot | File |
|------|------|
| Confusion matrices (all 3 models) | `06_confusion_matrices.png` |
| ROC-AUC curves | `07_roc_curves.png` |
| Feature importance (Random Forest) | `08_feature_importance.png` |
| Model comparison bar chart | `09_model_comparison.png` |

### Top 5 Most Important Features (Random Forest)
1. `cp` — Chest pain type
2. `thalach` — Maximum heart rate
3. `ca` — Number of major vessels
4. `oldpeak` — ST depression
5. `thal` — Thalassemia type

---

## ✅ Conclusion

- **Logistic Regression** achieved the best overall performance with **80.33% accuracy** and **0.8235 F1-Score**, outperforming both tree-based models on this clean, small dataset
- Logistic Regression's higher Recall (84.85%) is especially important in medical contexts — it minimizes false negatives (missing actual disease cases)
- The most predictive features were `cp`, `thalach`, `ca`, `oldpeak`, and `thal`
- All models showed reasonable performance; an ensemble approach or hyperparameter tuning (GridSearchCV) could push accuracy further

### Limitations
- Dataset has only ~302 unique records after deduplication — a larger real-world dataset would improve reliability
- No external validation set used
- Class imbalance handling (SMOTE) was not required here but may help on imbalanced datasets

---




---

## 📚 References

1. Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
2. UCI Repository: https://archive.ics.uci.edu/ml/datasets/heart+disease
3. Scikit-learn Documentation: https://scikit-learn.org/stable/
4. Detrano, R. et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64(5), 304–310.
