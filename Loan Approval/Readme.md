# Loan Approval Predictor

## Project Overview
This project develops a machine learning model to predict loan approval status based on various financial and personal attributes of applicants. The goal is to assist financial institutions in making informed decisions, minimizing risk, and streamlining the loan application process.

## Dataset
The dataset used in this project contains information about loan applicants, including:
- `no_of_dependents`: Number of dependents the applicant has.
- `education`: Applicant's educational background (e.g., Graduate, Not Graduate).
- `self_employed`: Indicates if the applicant is self-employed.
- `income_annum`: Annual income of the applicant.
- `loan_amount`: The amount of loan requested.
- `loan_term`: The duration of the loan in years.
- `cibil_score`: Credit Information Bureau Limited score, a measure of creditworthiness.
- `residential_assets_value`: Value of residential assets owned by the applicant.
- `commercial_assets_value`: Value of commercial assets owned by the applicant.
- `luxury_assets_value`: Value of luxury assets owned by the applicant.
- `bank_asset_value`: Value of bank assets owned by the applicant.
- `loan_status`: The target variable, indicating whether the loan was 'Approved' or 'Rejected'.

### Data Cleaning and Preprocessing
1.  **Column Renaming**: Column names were standardized (stripped spaces and converted to lowercase).
2.  **ID Removal**: The `loan_id` column was dropped as it's a non-predictive identifier.
3.  **Target Encoding**: The `loan_status` column was encoded to numerical values: `Approved` as 1 and `Rejected` as 0.
4.  **Feature Separation**: Features (X) and target (y) were separated.
5.  **Data Splitting**: The dataset was split into training and testing sets (75% train, 25% test) using `stratify=y` to maintain the class distribution.

## Methodology

### Preprocessing Pipeline
A `ColumnTransformer` was used to apply different preprocessing steps to numerical and categorical features:
-   **Numerical Features**: Missing values were imputed using the median, and features were scaled using `StandardScaler`.
-   **Categorical Features**: Missing values were imputed using the most frequent strategy, and features were one-hot encoded using `OneHotEncoder`.

### Model Training
Two classification models were trained:
1.  **Logistic Regression**: Initialized with `class_weight='balanced'` to handle potential class imbalance.
2.  **Random Forest Classifier**: Initialized with `class_weight='balanced'` and `n_estimators=300`.

### Handling Class Imbalance with SMOTE
To further address class imbalance, both models were re-trained within an `imblearn.pipeline.Pipeline` that included SMOTE (Synthetic Minority Over-sampling Technique) as an oversampling step before the classifier.

## Results

### Model Performance (Before SMOTE)
| Model               | Precision | Recall   | F1-score |
|---------------------|-----------|----------|----------|
| Logistic Regression | 0.956     | 0.919    | 0.937    |
| Random Forest       | 0.978     | 0.989    | 0.984    |

### Model Performance (After SMOTE)
| Model          | Precision | Recall   | F1-score |
|----------------|-----------|----------|----------|
| LogReg + SMOTE | 0.956     | 0.922    | 0.939    |
| RF + SMOTE     | 0.983     | 0.983    | 0.983    |

The Random Forest model consistently showed superior performance both before and after applying SMOTE. SMOTE further improved the F1-score slightly for Random Forest.

### Feature Importance
The top financial risk drivers identified by the Random Forest model (after SMOTE) are:

| Feature                  | Importance |
|--------------------------|------------|
| `cibil_score`            | 0.810      |
| `loan_term`              | 0.059      |
| `loan_amount`            | 0.026      |
| `income_annum`           | 0.018      |
| `luxury_assets_value`    | 0.017      |
| `residential_assets_value`| 0.017      |
| `commercial_assets_value`| 0.016      |
| `bank_asset_value`       | 0.015      |
| `no_of_dependents`       | 0.014      |

`CIBIL Score` is overwhelmingly the most important feature in predicting loan approval, followed by `loan_term` and `loan_amount`.

### SHAP Explainability
SHAP (SHapley Additive exPlanations) values were used to provide local interpretability for the Random Forest model, showing how each feature influences individual predictions.

## Usage

### Prerequisites
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, `shap`

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imblearn shap
```

### Running the Script
1.  **Clone the repository (if applicable)**: If this script is part of a larger project, clone the repository.
2.  **Update `ds_file_path`**: Ensure the `ds_file_path` variable in the script points to your dataset file (e.g., `'/content/drive/Dataset_Path/dataset_File.csv'`).
3.  **Execute the notebook/script**: Run all cells in the Jupyter notebook or execute the Python script.

### Example (within a Python environment)
```python
# Import required libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import shap

sns.set(style="whitegrid")

# Load data (replace with your actual path)
ds_file_path = '/content/drive/Dataset_Path/dataset_File.csv'
df = pd.read_csv(ds_file_path)

# ... (rest of the code for preprocessing, model training, and evaluation)

# To get predictions from the best model (RF + SMOTE):
y_pred_final = rf_smote.predict(X_test)
print(classification_report(y_test, y_pred_final))
```

## Future Improvements
-   Hyperparameter tuning for Random Forest and Logistic Regression.
-   Exploring other advanced models like Gradient Boosting or Neural Networks.
-   More extensive feature engineering.
-   Deployment of the model as a web service.
```
