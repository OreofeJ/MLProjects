# Weekly Sales Forecaster

## Project Overview
This project aims to forecast weekly sales for Walmart stores using historical sales data, promotional information, economic indicators, and store characteristics. The primary goal is to leverage machine learning models, specifically Random Forest and XGBoost, to build accurate predictive models that can assist in inventory management, staffing optimization, and strategic planning.

## Dataset Description
The dataset used in this project consists of several interconnected files:
- `train.csv`: Contains historical sales data for various stores and departments, along with holiday information.
- `test.csv`: Similar to `train.csv` but without the `Weekly_Sales` column, used for generating future predictions.
- `features.csv`: Provides additional details such as temperature, fuel price, CPI, unemployment, and markdown information for specific dates.
- `stores.csv`: Contains static information about each store, including its type and size.

### Key Features:
- **Store**: Unique identifier for each store.
- **Dept**: Unique identifier for each department within a store.
- **Date**: The week of sales activity.
- **Weekly_Sales**: Sales for the given week (target variable).
- **IsHoliday**: Boolean indicating if the week is a special holiday week.
- **Temperature**: Average temperature in the region.
- **Fuel_Price**: Cost of fuel in the region.
- **MarkDown1-5**: Anonymized data related to promotional markdowns.
- **CPI**: Consumer Price Index.
- **Unemployment**: Unemployment rate.
- **Type**: Type of the store (A, B, or C).
- **Size**: Store size.

## Data Preprocessing & Feature Engineering

### Merging Data
All datasets (`train`, `features`, `stores`, `test`) are merged based on common identifiers (`Store`, `Date`, `IsHoliday`) to create a comprehensive dataset for modeling.

### Handling Missing Values
- Markdown features (`MarkDown1` to `MarkDown5`) are imputed with 0, assuming missing values indicate no markdown activity.
- `CPI` and `Unemployment` are forward-filled (`ffill`) to handle missing entries, assuming recent past values are good approximations.

### Outlier Treatment
- `Weekly_Sales` are winsorized at the 1st and 99th percentiles to mitigate the impact of extreme outliers.

### Feature Engineering (Time-Based)
New time-based features are created from the `Date` column:
- `Year`, `Month`, `Week`, `Day` are extracted.
- **Lag Features**: `Lag_1` and `Lag_2` represent `Weekly_Sales` from the previous one and two weeks, respectively, grouped by `Store` and `Dept`.
- **Rolling Average**: `Rolling_4` calculates the 4-week rolling average of `Weekly_Sales` for each `Store` and `Dept`.

### Categorical Feature Encoding
- The `Type` column (Store Type A, B, C) is one-hot encoded to convert it into a numerical format suitable for machine learning models. To ensure consistency between training and forecasting, train and test sets are combined for encoding, then split back.

## Exploratory Data Analysis (EDA)

- **Sales Distribution**: A histogram of `Weekly_Sales` is plotted to understand its distribution.
- **Sales Over Time**: Total `Weekly_Sales` summed by `Date` are plotted to visualize temporal trends and seasonality.
- **Correlation Heatmap**: A heatmap is generated to show correlations between numerical features.
- **Seasonal Decomposition**: `Weekly_Sales` are decomposed into trend, seasonal, and residual components to identify underlying patterns.

## Modeling
Two tree-based ensemble models are used for forecasting:

1.  **Random Forest Regressor**
2.  **XGBoost Regressor**

### Train/Test Split
The data is split in a time-aware manner to prevent temporal leakage. Data before '2012-01-01' is used for training, and data from '2012-01-01' onwards is used for testing.

### Model Training
- **Random Forest**: Trained with `n_estimators=200`, `max_depth=20`, and `random_state=77`.
- **XGBoost**: Trained with `n_estimators=500`, `learning_rate=0.05`, `max_depth=8`, `subsample=0.8`, `colsample_bytree=0.8`, `objective="reg:squarederror"`, and `random_state=77`.

### Evaluation Metrics
The models are evaluated using:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared (R²)**

### Visualization of Predictions
Plots comparing actual vs. predicted weekly sales are generated for a subset of the test data to visually assess model performance.

## Forecasting Future Periods
The trained models are used to forecast sales on the `test.csv` dataset. The `test_data` undergoes similar preprocessing and feature engineering steps as the training data, ensuring consistency. Lag and rolling features for the test data are derived from the last known values of the training data.

## Feature Importance
Feature importance plots are generated for both Random Forest and XGBoost models to identify the most influential factors driving weekly sales. These insights can help in understanding model decisions and business strategies.

## Fairness, Ethics & Explainability

### Bias Checks
- The project acknowledges potential biases related to store sizes dominating loss functions. Winsorization and regularization techniques (inherent in tree models) help mitigate this.
- No demographic attributes are used in the model, reducing risks of demographic bias.

### Explainability
- Feature importance plots provide transparency into which features contribute most to the predictions.

### Ethical Considerations
- **Avoiding Over-forecasting**: The project aims to prevent models from consistently over-forecasting for smaller stores, which could lead to inventory issues.
- **Promotional Fairness**: Awareness is maintained to ensure that markdown features do not inadvertently reinforce historical inequalities or unfair practices.
- **Transparency**: Feature importance is explicitly provided to ensure transparency in model decision-making.

## Setup and Usage

### Prerequisites
- Python 3.x
- Jupyter Notebook or Google Colab

### Libraries
Install the required libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost statsmodels
```

### Data
Place the `features.csv`, `stores.csv`, `train.csv`, and `test.csv` files in a directory accessible by the notebook, as specified in the `Dataset_Path` variable (e.g., `/Dataset_Path/`).

### Running the Notebook
1.  Open the `.ipynb` file in Jupyter Notebook or Google Colab.
2.  Run all cells sequentially.

## Future Enhancements
- **Hyperparameter Tuning**: More extensive hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- **Ensemble Modeling**: Explore stacking or blending different models for improved performance.
- **Deep Learning Models**: Investigate recurrent neural networks (RNNs) or Transformers for time series forecasting.
- **External Data**: Incorporate additional external data sources (e.g., local events, competitor promotions).
- **Interactive Visualizations**: Create interactive dashboards for exploring forecasts and feature importances.

