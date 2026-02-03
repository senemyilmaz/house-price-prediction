# House Price Prediction with Machine Learning

## Project Overview
This project implements an end-to-end machine learning pipeline to predict house prices using structured tabular data.
The workflow covers the full data science lifecycle, including data preprocessing, model comparison, feature selection, error diagnostics, learning curve analysis, and experimental target transformation.
The primary objective of the project is not only achieving high predictive accuracy, but also ensuring model interpretability, robustness, and generalization.

---

## Dataset
The dataset is loaded dynamically from an external public source:

**Source:**
https://d32aokrjazspmn.cloudfront.net/materials/ML_Houses_dataset.csv

The dataset contains residential housing information such as:

-Structural features (e.g. living area, basement size, garage size)
-Quality indicators (e.g. overall quality, kitchen quality)
-Categorical attributes (e.g. neighborhood, zoning)
-Target variable: SalePrice

The dataset is not stored in the repository and is accessed directly within the notebook.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Missing value analysis
- Correlation analysis between numerical features and target
- Identification of numerical and categorical variables

### 2. Data Preprocessing
- Median imputation for numerical variables
- Mode imputation for categorical variables
- Standard scaling for numerical features
- One-hot encoding for categorical features

All preprocessing steps are implemented using **scikit-learn pipelines** to avoid data leakage.

### 3. Baseline Model Comparison
Three baseline models were evaluated using 5-fold cross-validation:
-Ridge Regression
-Random Forest Regressor
-Gradient Boosting Regressor

Metrics used:
- R²
- RMSE
- MAE

### 4. Final Model Selection
The **Gradient Boosting Regressor (GBR)** achieved the best overall cross-validation performance and was selected as the final model.

### 5. Permutation Feature Importance
Permutation importance was applied to interpret model behavior and identify the most influential predictors.

Key influential features included:
-Overall quality
-Living area
-Basement size
-Floor area

### 6. Feature Selection
Low-impact features were removed using a threshold on permutation importance scores.
The model was retrained using the reduced feature set.

Result:
-Feature count significantly reduced
-Performance remained almost unchanged
-Model became more interpretable and efficient

### 7. Error Analysis and Residual Diagnostics
Residual analysis showed:
- Errors centered around zero (no systematic bias)
- Slight heteroskedasticity for high-priced houses
- No strong nonlinear patterns in residuals

### 8. Learning Curve Analysis
Learning curves indicated:
-Mild overfitting
-Strong generalization
-Stable bias–variance trade-off
-Additional data would provide only marginal improvement

### 9. Log-Transformation Experiment
An experimental log-transformation of the target variable was tested.
Findings:
-Did not significantly improve prediction performance
-Residual structure remained similar
-Original target scale retained as final choice

---

## Final Model Performance

| Metric | Value |
|--------|-------|
| R²     | 0.94 |
| RMSE  | ~20,700 |
| MAE   | ~13,000 |

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Key Contributions of the Project
-Fully leakage-safe preprocessing pipeline
-Systematic model comparison
-Explainable feature importance
-Feature selection with performance validation
-Residual diagnostics and learning curves
-Experimental validation with target transformation
This project demonstrates a complete applied machine learning workflow rather than a single predictive model.

---

## How to Run
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Open the notebook:
```bash
jupyter notebook House_Price_Prediction.ipynb
```

---

## Results

The final Gradient Boosting model achieved strong predictive performance on the unseen test set.

### Performance Comparison

| Model | R² | RMSE | MAE |
|------|----|------|-----|
| Ridge Regression | 0.86 | ~30,000 | ~18,000 |
| Random Forest | 0.87 | ~29,000 | ~16,000 |
| **Gradient Boosting (Final)** | **0.94** | **~20,700** | **~13,000** |

---

### Feature Selection Impact

| Model Version | #Features | R² | RMSE | MAE |
|--------------|----------|----|------|-----|
| Full feature set | 80 | 0.938 | ~20,752 | ~13,025 |
| Reduced feature set | 57 | 0.937 | ~20,885 | ~13,220 |

Feature selection reduced dimensionality by nearly **30%** while maintaining almost identical performance.

---

### Diagnostic Findings

- Residuals are centered around zero, indicating no systematic bias.  
- Slight heteroskedasticity observed for high-priced houses.  
- Learning curves show stable convergence and good generalization.  
- Log-transformation experiment did not yield performance improvement.

---

### Final Interpretation

The model explains approximately **94% of the variance** in house prices and predicts prices with an average error of about **13,000 units**, demonstrating strong real-world applicability.
