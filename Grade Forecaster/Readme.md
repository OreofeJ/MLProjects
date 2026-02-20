# Grade Prediction Project including Fairness Analysis: using T Test for Gender and Family Income Bias

## Overview
This script develops and evaluates predictive models for student exam scores based on various academic and socio-economic factors. It explores both linear and polynomial regression models, performs data preprocessing, and conducts a fairness analysis to identify potential biases in the predictions.

## Dataset
The analysis uses a dataset which contains information on student performance and various influencing factors. Key columns include:
- `Hours_Studied`: Number of hours a student studied.
- `Attendance`: Student attendance percentage.
- `Parental_Involvement`: Level of parental involvement (e.g., Low, Medium, High).
- `Access_to_Resources`: Availability of learning resources.
- `Sleep_Hours`: Number of hours of sleep.
- `Previous_Scores`: Scores from previous exams.
- `Motivation_Level`: Student's motivation level.
- `Internet_Access`: Access to the internet.
- `Tutoring_Sessions`: Number of tutoring sessions attended.
- `Family_Income`: Family income level (e.g., Low, Medium, High).
- `Teacher_Quality`: Perceived quality of teachers.
- `School_Type`: Type of school (Public/Private).
- `Peer_Influence`: Influence of peers.
- `Physical_Activity`: Level of physical activity.
- `Learning_Disabilities`: Presence of learning disabilities.
- `Parental_Education_Level`: Education level of parents.
- `Distance_from_Home`: Distance from home to school.
- `Gender`: Student's gender.
- `Exam_Score`: The target variable, representing the student's exam score.

## Data Preprocessing
1.  **Missing Value Handling**: Missing values in numerical columns were imputed using the median, while categorical columns were imputed using the mode.
2.  **Outlier Handling**: Outliers in the `Hours_Studied` column were handled using the Interquartile Range (IQR) method to ensure that extreme values do not disproportionately influence the models.

## Models Used
Two regression models were implemented and compared:
1.  **Linear Regression**: A simple linear model to predict `Exam_Score` based on `Hours_Studied`.
2.  **Polynomial Regression (Degree 2)**: A polynomial model (degree 2) to capture potential non-linear relationships between `Hours_Studied` and `Exam_Score`.

## Model Evaluation
Both models were evaluated using the following metrics:
-   **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction.
-   **Root Mean Squared Error (RMSE)**: Measures the square root of the average of the squared errors, giving more weight to larger errors.
-   **R² Score**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

### Evaluation Results
| Model                 | MAE      | RMSE     | R2 Score |
| :-------------------- | :------- | :------- | :------- |
| Linear Regression     | 2.490517 | 3.483801 | 0.201646 |
| Polynomial Regression | 2.490401 | 3.483394 | 0.201832 |

**Interpretation:**
-   Both models show similar performance, with MAE around 2.49-2.50 marks and RMSE around 3.48-3.55 marks, indicating that predictions are typically off by a few points.
-   The R² scores for both models are relatively low (~0.20), suggesting that `Hours_Studied` alone explains only about 20% of the variance in `Exam_Score`. This implies that other factors not included in these  models are significant in determining exam scores.

## Fairness Analysis
To assess potential biases, a fairness analysis was conducted by examining model residuals across different demographic groups:

1.  **Residuals by Gender**:
    -   MAE by Gender: Female (2.5395), Male (2.4725)
    -   T-test (Male vs Female residuals): t-stat = -0.0424, p-value = 0.9662
    -   **Conclusion**: The p-value (0.9662) is high, indicating no statistically significant difference in prediction errors between male and female students. The model appears to be fair with respect to gender.

2.  **Residuals by Family Income**:
    -   MAE by Family Income: High (2.5565), Low (2.4722), Medium (2.5029)
    -   T-tests:
        -   Low vs Medium: t-stat = -4.7893, p-value = 0.0000
        -   Low vs High: t-stat = -8.0854, p-value = 0.0000
        -   Medium vs High: t-stat = -4.5192, p-value = 0.0000
    -   **Conclusion**: The extremely low p-values for all income group comparisons suggest statistically significant differences in prediction errors across family income levels. This indicates a potential bias where the model's predictions are systematically more accurate or inaccurate for certain income groups. Further investigation is needed to understand the nature of this bias and mitigate it.

## Key Findings
-   `Hours_Studied` is a weak predictor of `Exam_Score` when used alone, as indicated by low R² values.
-   The model shows no significant bias with respect to `Gender`.
-   There is a statistically significant bias in prediction errors across different `Family_Income` levels, indicating a potential fairness concern that needs to be addressed.

## Usage
To run this script:
1.  Ensure you have the required libraries installed (e.g., `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`).
2.  Mount your Google Drive to access the `my_grades.csv` file.
3.  Update the `grades_file_path` variable to the correct location of your dataset.
4.  Execute the cells sequentially.

## Dependencies
-   `pandas`
-   `numpy`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`
-   `scipy`

