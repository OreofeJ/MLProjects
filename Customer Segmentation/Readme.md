# Customer Segmentation Analysis Project

Objective: Perform customer segmentation using two clustering algorithms: K-Means and DBSCAN. The goal is to identify distinct customer groups based on their demographic and behavioral attributes (specifically Annual Income and Spending Score) to help businesses tailor marketing strategies and improve customer engagement.

## Table of Contents
1.  [Setup and Data Loading](#setup-and-data-loading)
2.  [Initial Data Exploration](#initial-data-exploration)
3.  [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
4.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5.  [Feature Scaling](#feature-scaling)
6.  [K-Means Clustering](#k-means-clustering)
7.  [DBSCAN Clustering](#dbscan-clustering)
8.  [Model Comparison](#model-comparison)
9.  [Fairness & Bias Analysis](#fairness--bias-analysis)

## 1. Setup and Data Loading

This section imports necessary libraries for data manipulation, visualization, scaling, and clustering. It also mounts Google Drive to access the dataset and loads the `Dataset_Name.csv` file into a Pandas DataFrame.

### Libraries Used:
*   **Data Manipulation**: `pandas`, `numpy`
*   **Visualization**: `matplotlib.pyplot`, `seaborn`
*   **Scaling and Clustering**: `sklearn.preprocessing.StandardScaler`, `sklearn.cluster.KMeans`, `sklearn.metrics.silhouette_score`, `sklearn.neighbors.NearestNeighbors`, `sklearn.cluster.DBSCAN`

### Outcomes:
*   The dataset is loaded, and the first 5 rows are displayed, showing columns like `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`.

## 2. Initial Data Exploration

This step provides a quick overview of the dataset's structure and summary statistics.

### Methods:
*   `df.info()`: To check data types, non-null counts, and memory usage.
*   `df.describe()`: To get descriptive statistics (count, mean, std, min, max, quartiles) for numerical columns.

### Outcomes:
*   Confirmed entries with no missing values in the initial check.
*   Identified data types: `int64` for numerical columns and `object` for `Gender`.
*   Observed ranges for Age (18-70), Annual Income (15-137 k$), and Spending Score (1-99).

## 3. Data Cleaning & Preprocessing

This section prepares the data for clustering by handling missing values, duplicates, irrelevant columns, and converting categorical features.

### Methods:
*   **Missing Values**: Checked using `df.isnull().sum()`. (None found initially, but code includes imputation logic).
*   **Duplicates**: Checked using `df.duplicated().sum()`. (None found).
*   **Column Dropping**: `CustomerID` is dropped as it's a unique identifier and not useful for clustering.
*   **Categorical Encoding**: `Gender` is converted from 'Male'/'Female' to 0/1 (binary encoding).
*   **Column Renaming**: Columns `Annual Income (k$)` and `Spending Score (1-100)` are renamed to `Income` and `SpendingScore` for easier access.
*   **Outlier Detection**: Outliers in `Income` and `SpendingScore` are identified and removed using the Interquartile Range (IQR) method. Data points outside `Q1 - 1.5*IQR` and `Q3 + 1.5*IQR` are filtered out.

### Outcomes:
*   Cleaned DataFrame `df_clean` without `CustomerID` and with encoded `Gender`.
*   Potential outliers removed, leading to a more robust clustering process.

## 4. Exploratory Data Analysis (EDA)

EDA helps understand data distributions and relationships between variables through visualizations.

### Methods:
*   **Distribution Plots**: Histograms with KDE for `Age`, `Income`, and `SpendingScore` to visualize their distributions.
*   **Correlation Heatmap**: Displays the correlation matrix of numerical features to identify linear relationships.

### Outcomes:
*   **Age**: Appears somewhat uniformly distributed, with a slight peak in younger adults and a spread across older groups.
*   **Income**: Shows a relatively normal distribution, centered around the mean.
*   **Spending Score**: Also appears normally distributed.
*   **Correlation Matrix**: Revealed a low correlation between `Income` and `SpendingScore`, suggesting they are independent features and useful for segmentation.

## 5. Feature Scaling

Scaling ensures that all features contribute equally to the clustering process, preventing features with larger numerical ranges from dominating.

### Methods:
*   `StandardScaler`: Features `Income` and `SpendingScore` are standardized (mean=0, variance=1).

### Outcomes:
*   Transformed data `X` ready for clustering, where `Income` and `SpendingScore` have comparable scales.

## 6. K-Means Clustering

K-Means is an unsupervised learning algorithm that partitions data into *k* clusters, where each data point belongs to the cluster with the nearest mean.

### Methods:
*   **Optimal K Determination (Elbow Method)**: Plots the Within-Cluster Sum of Squares (WCSS) against the number of clusters (K) to find the 'elbow point' where the rate of decrease in WCSS slows down, suggesting an optimal K.
*   **Optimal K Determination (Silhouette Method)**: Calculates the silhouette score for different numbers of clusters. A higher silhouette score indicates better-defined clusters.
*   **K-Means Application**: K-Means is applied with `n_clusters=6` (based on prior analysis from the optimal K methods).
*   **Cluster Visualization**: Scatter plot of `Income` vs `SpendingScore` colored by K-Means cluster assignments.
*   **Cluster Analysis**: Grouping `df_clean` by `Cluster_KMeans` to calculate the mean `Income` and `SpendingScore` for each cluster.
*   **Descriptive Labels**: Assigning business-friendly labels to each cluster based on their average `Income` and `SpendingScore` profiles.
*   **Enhanced Visualization**: A more detailed scatter plot with custom markers, centroids, and descriptive labels.

### Outcomes:
*   **Optimal K**: Both Elbow and Silhouette methods suggest `K=6` as a suitable number of clusters.
*   **K-Means Clusters**: The algorithm successfully segmented customers into 6 distinct groups.
*   **Cluster Statistics**: Identified clear profiles for each cluster:
    *   **Premium Big Spenders (Cluster 0)**: High income, high spending.
    *   **Average Customers (Cluster 1)**: Medium income, medium spending.
    *   **Young/Impulsive Spenders (Cluster 2)**: Low income, high spending.
    *   **Wealthy but Frugal (Cluster 3)**: High income, low spending.
    *   **High Net Worth Low Engagement (Cluster 4)**: Very high income, low spending.
    *   **Low-Value Customers (Cluster 5)**: Low income, low spending.
*   **Visualizations**: Clear separation of customer segments in scatter plots, aiding in interpretation.

## 7. DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions.

### Methods:
*   **DBSCAN Application**: DBSCAN is applied with optimized parameters (`eps=0.39`, `min_samples=6`) based on prior hyperparameter tuning.
*   **Cluster Visualization**: Scatter plot of `Income` vs `SpendingScore` colored by DBSCAN cluster assignments, including noise points.
*   **Cluster Count**: Prints the number of clusters detected and the number of noise points.

### Outcomes:
*   **DBSCAN Clusters**: Identified 6 distinct clusters and a significant number of noise points (21) which are effectively outliers.
*   **Visualization**: Showed density-based clusters, with noise points clearly marked.

## 8. Model Comparison

This section compares the performance of K-Means and DBSCAN using the Silhouette Score.

### Methods:
*   **Silhouette Score Calculation**: Computed for both K-Means and DBSCAN. For DBSCAN, noise points (-1 label) are excluded from the score calculation.

### Outcomes:
*   **KMeans Silhouette Score**: 0.540
*   **DBSCAN Silhouette Score**: 0.537
*   **Conclusion**: Both models performed comparably well on the dataset, with K-Means showing a slightly higher Silhouette Score. The choice between them might depend on whether identifying noise points is critical.

## 9. Fairness & Bias Analysis

This step examines the distribution of gender across the identified K-Means clusters to check for potential biases.

### Methods:
*   **Gender Distribution per Cluster**: Calculated the normalized value counts of `Gender` within each K-Means cluster.
*   **Heatmap Visualization**: A heatmap is used to visually represent the gender distribution across clusters.

### Outcomes:
*   The analysis of gender distribution per cluster shows that clusters are not disproportionately dominated by one gender. For example, Cluster 0 (Premium Big Spenders) has 44.7% Male and 55.3% Female, while Cluster 4 (High Net Worth Low Engagement) has 22.2% Male and 77.8% Female. This suggests that the clustering is not heavily biased towards gender for the most part, though some clusters do show a leaning towards one gender which might warrant further investigation to ensure impartial marketing strategies.
