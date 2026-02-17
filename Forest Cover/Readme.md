# Forest Cover Type Prediction

## Project Overview
This project aims to classify forest cover types using various environmental features. We employ machine learning models, specifically Random Forest and XGBoost, to predict seven different forest cover types. The notebook covers data loading, exploratory data analysis, feature engineering, model training, hyperparameter tuning, model evaluation, explainability using SHAP, and basic fairness checks.

## Dataset
The dataset used is the Forest Cover Type dataset, loaded using `sklearn.datasets.fetch_covtype`. It contains 581,012 entries and 55 features, including 10 continuous features and 44 binary (one-hot encoded) wilderness area and soil type features. The target variable, `Cover_Type`, has 7 classes.

**Key Observations from Data Inspection:**
*   No missing values.
*   Significant class imbalance in the `Cover_Type` distribution.
*   Features include a mix of continuous and one-hot encoded categorical variables.

## Exploratory Data Analysis (EDA)

**Target Distribution:**
*   A count plot revealed the severe class imbalance, with `Cover_Type` 1 and 2 being dominant, while other classes (especially 4) are rare.

**Feature Correlations:**
*   A correlation heatmap of the first 10 continuous features showed that `Elevation` is a highly influential feature. Some multicollinearity was observed, which tree-based models can generally handle well.

**Feature Distributions:**
*   Histograms of the first 6 continuous features indicated various distributions, with some features exhibiting skewness (e.g., horizontal and vertical distances to hydrology).

**EDA Insights:**
*   Elevation is strongly correlated with `Cover_Type`.
*   Horizontal and vertical distances show skewness.
*   Multicollinearity exists, but Random Forest and XGBoost are robust to it.

## Data Preprocessing and Feature Engineering

**Data Cleaning:**
*   The dataset was found to have no missing values, eliminating the need for imputation.
*   Wilderness and Soil types were already one-hot encoded, so no further categorical encoding was required.
*   Outlier handling was deemed unnecessary due to the robustness of tree-based models.

**Feature Engineering:**
Two new features were engineered:
1.  **`Distance_To_Hydrology`**: Calculated as the Euclidean distance from horizontal and vertical distances to hydrology: `sqrt(Horizontal_Distance_To_Hydrology^2 + Vertical_Distance_To_Hydrology^2)`.
2.  **`Elevation_Hydrology_Diff`**: The difference between `Elevation` and `Vertical_Distance_To_Hydrology`.

## Data Splitting
The data was split into training, validation, and test sets with a 70/15/15 ratio, ensuring stratification by the target variable (`y`) to maintain class distribution across splits.
*   `X_train`, `y_train` (70%)
*   `X_val`, `y_val` (15%)
*   `X_test`, `y_test` (15%)

For XGBoost, the target variable `y` was label-encoded (from 1-7 to 0-6) before splitting, as XGBoost expects class labels to start from zero for multi-class classification.

## Model Training and Evaluation

### Random Forest Classifier
*   **Baseline Model**: Trained with `n_estimators=200`, `max_depth=None`, `random_state=77`.
    *   **Accuracy:** 0.960
    *   `RandomForestClassifier` performed very well on the test set.
*   **Hyperparameter Tuning (RandomizedSearchCV - Ultra-Fast Configuration)**:
    *   Used a limited search space for `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.
    *   Best parameters found: `{'n_estimators': 80, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 15}`
    *   **Tuned Model Accuracy:** 0.838
    *   **Observation:** The tuned Random Forest model surprisingly showed a significant drop in accuracy compared to the baseline model. This suggests that the 'Ultra-Fast' tuning parameters might have been too constrained or that the default parameters were already performing optimally or overfitting the training data.

### XGBoost Classifier
*   **Baseline Model**: Trained with `objective='multi:softprob'`, `num_class=7`, `n_estimators=300`, `learning_rate=0.1`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric='mlogloss'`, `random_state=77`.
    *   **Accuracy:** 0.880
*   **Hyperparameter Tuning (GridSearchCV)**:
    *   Used a limited search space for `max_depth` and `learning_rate`.
    *   Best parameters found: `{'learning_rate': 0.1, 'max_depth': 6}`
    *   **Tuned Model Accuracy:** 0.813
    *   **Observation:** Similar to Random Forest, the tuned XGBoost model also resulted in a decrease in accuracy compared to its baseline, indicating potential issues with the tuning strategy or parameter range.

## Feature Importance
*   **Random Forest (Baseline & Tuned)**: `Elevation`, `Horizontal_Distance_To_Fire_Points`, `Horizontal_Distance_To_Roadways`, and `Distance_To_Hydrology` consistently appeared as the most important features. `Elevation` was by far the most dominant feature.

## Model Explainability (SHAP)
*   SHAP (SHapley Additive exPlanations) was used with the XGBoost model to interpret feature contributions to predictions. A `shap.TreeExplainer` was initialized, and a summary plot was generated to visualize feature importance and impact on model output.

## Fairness, Ethics & Bias Checks
*   **Fairness Considerations**: Given the ecological nature of the data, the project did not involve human or demographic attributes, and no sensitive features (like race, gender, income) were present. This suggests a low ethical risk concerning bias related to protected characteristics.
*   **Bias Checks**: A confusion matrix analysis (`pd.crosstab`) was performed on the `y_test` vs `y_pred_rf` to check for systematic misclassifications across different cover types. While the overall accuracy was high for the baseline RF, there were variations in recall for smaller classes (e.g., class 4 had lower recall).

## Model Comparison
A comparison table and bar plots were generated to summarize the accuracy of all models:

| Model                      | Accuracy |
| :------------------------- | :------- |
| Random Forest (Baseline)   | 0.960    |
| Random Forest (Tuned)      | 0.838    |
| XGBoost (Baseline)         | 0.880    |
| XGBoost (Tuned)            | 0.813    |

**Key Takeaway:** Contrary to typical expectations, both hyperparameter-tuned models performed worse than their baseline counterparts with default or manually set parameters. This suggests that the chosen tuning ranges or the number of iterations (`n_iter`, `cv`) in RandomizedSearchCV/GridSearchCV were likely too restrictive or insufficient, potentially leading to sub-optimal parameter selection. The baseline Random Forest Classifier achieved the highest accuracy.

## Conclusion
The Random Forest Classifier (baseline) achieved the best performance with an accuracy of 96%. This indicates its strong capability in distinguishing between the different forest cover types based on the provided environmental features. The unexpected drop in performance for tuned models highlights the importance of carefully selecting tuning strategies and parameter ranges. Further investigation into broader search spaces and more extensive cross-validation might yield better-tuned models.

## Requirements
To run this notebook, the following Python libraries are required:
*   `numpy`
*   `pandas`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `xgboost`
*   `shap`

