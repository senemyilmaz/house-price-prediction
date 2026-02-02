# House Price Prediction with Machine Learning

## Project Overview
This project aims to build a machine learning model to predict house prices using structured tabular data.  
The workflow follows a complete end-to-end data science pipeline including data preprocessing, model comparison, feature selection, error analysis, and learning curve evaluation.  

The main focus of the project is not only achieving high predictive performance, but also understanding model behavior and improving generalization.

---

## Dataset
The dataset is loaded directly from an external public source:  

**Source:**  
https://d32aokrjazspmn.cloudfront.net/materials/ML_Houses_dataset.csv  

The dataset contains information about residential houses, including:

- Structural features (e.g. living area, basement size, garage size)  
- Quality indicators (e.g. overall quality, kitchen quality)  
- Categorical attributes (e.g. neighborhood, zoning)  
- Target variable: **SalePrice**  

The dataset is **not stored in the repository** and is accessed dynamically inside the notebook.

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

All preprocessing steps are implemented using **scikit-learn pipelines**.

### 3. Baseline Model Comparison
Two baseline models were evaluated using 5-fold cross-validation:
- Ridge Regression  
- Random Forest Regressor  

Metrics used:
- R²  
- RMSE  
- MAE  

### 4. Final Model Selection
Random Forest was selected as the final model based on superior cross-validation performance.

### 5. Permutation Feature Importance
Permutation importance was applied to interpret the model and identify the most influential features.

### 6. Feature Selection
Low-importance features were removed using a threshold on permutation importance scores.  
The model was retrained using the reduced feature set.

### 7. Error Analysis and Residual Diagnostics
Residual analysis showed:
- Errors increase for high-priced houses  
- No strong systematic bias  
- Slight heteroskedasticity  

### 8. Learning Curve Analysis
Learning curves indicated:
- Mild overfitting  
- Strong generalization  
- Additional data would provide only marginal improvement  

### 9. Log-Transformation Experiment
As an experimental improvement, log transformation of the target variable was tested.  
However, it did **not improve overall performance**, and the original target scale was retained.

---

## Final Model Performance

| Metric | Value |
|--------|-------|
| R²     | ~0.90 |
| RMSE  | ~28,000 |
| MAE   | ~17,500 |

---

## Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---

## How to Run
1. Clone the repository  
2. Install dependencies:
```bash
pip install -r requirements.txt

  
