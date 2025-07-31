# 🌳 AI-Powered Urban Green Space Management System (UGSMS)

## Project Overview

The **Urban Green Space Management System (UGSMS)** is an end-to-end data pipeline designed to analyze environmental, usage, and sentiment data from parks in German cities. Leveraging machine learning models, the system generates **actionable recommendations** for proactive park management, aiming to improve urban green spaces for both the environment and visitors.

---

## ✨ Key Features

* **Multi-source Data Integration:** Seamlessly combines air quality, visitor footfall, public sentiment (from tweet analysis), and detailed park information.
* **Real-time Data Processing:** Utilizes **Apache Spark** for scalable and efficient data ingestion and processing, handling large datasets with ease.
* **Machine Learning Pipeline:** Implements multiple supervised ML algorithms (Logistic Regression, Random Forest, Gradient Boosting) to predict intervention needs.
* **Delta Lake Storage:** Employs **Delta Lake** for robust data storage, providing ACID transactions, schema enforcement, and data versioning.
* **MLflow Integration:** Ensures comprehensive model tracking, experiment management, and reproducibility for all machine learning workflows.
* **Automated Recommendations:** Generates specific, actionable recommendations tailored to individual park needs based on predictive analytics.
* **Proactive Management:** Enables park managers to anticipate issues and intervene before they escalate, optimizing resource allocation.
* **Data-Driven Decisions:** Empowers management with insights derived from comprehensive data analytics, fostering more effective strategies.

---

## 💡 Project Value

The UGSMS delivers significant value by transforming traditional park management into a proactive, data-driven discipline:

* **Proactive Park Management:** Predict intervention needs (e.g., maintenance, air quality improvements, public engagement campaigns) before issues escalate.
* **Resource Optimization:** Allocate maintenance, personnel, and financial resources more efficiently by targeting areas with the greatest need.
* **Data-Driven Decisions:** Base management strategies on comprehensive analytics, moving beyond guesswork to informed, impactful actions.
* **Environmental & Social Impact:** Improve air quality, enhance visitor experiences, and foster more sustainable and enjoyable urban green spaces.

---

## 🏛️ Architecture Overview

The system follows a robust data architecture, ensuring data quality, accessibility, and utility for machine learning.


## 🚀 Getting Started

This section outlines the main steps involved in the UGSMS pipeline. For detailed execution, please refer to the project notebooks or scripts.

### Step 1: Data Ingestion and Cleaning

The first step involves collecting and preparing data from various sources. For this project, **synthetic data** simulates real-world environmental, usage, and sentiment patterns, stored in this repository for direct ingestion.

**Data Sources:**

* `national_parks_air_quality.csv`: Contains Air Quality Index (AQI), NO2, PM2.5, and O3 levels.
* `national_parks_footfall.csv`: Tracks visitor counts and event day indicators.
* `national_parks_sentiment.csv`: Features tweet text analysis with sentiment scores (-1 to 1) and labels.
* `german_national_parks.csv`: Provides geographic coordinates and area measurements for parks.

**Process:**

Data is read into Spark DataFrames using a `read_data` function and then undergoes cleaning and preprocessing via `clean_data`. This includes removing nulls, converting timestamps (`M/d/yyyy H:mm`), and casting data types (e.g., `visitor_count` to integer, `event_day` to boolean).

```python
# Example Snippet (Conceptual)
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp

# Assuming 'spark' is an initialized SparkSession
# def read_data(url):
#     return spark.createDataFrame(pd.read_csv(url))

# def clean_data(df):
#     df = df.dropna(how='any', subset=[c for c in df.columns if c != ''])
#     if 'timestamp' in df.columns:
#         df = df.withColumn("timestamp", to_timestamp(col("timestamp"), "M/d/yyyy H:mm"))
#     if 'visitor_count' in df.columns:
#         df = df.withColumn("visitor_count", col("visitor_count").cast("integer"))
#     if 'event_day' in df.columns:
#         df = df.withColumn("event_day", col("event_day").cast("boolean"))
#     return df
Step 2: Data Storage and Medallion Architecture
Data is stored using a Medallion Architecture to ensure data quality and organization:

Bronze Layer: Raw, unprocessed data directly from ingestion sources.

Silver Layer: Cleaned and integrated data, ready for feature engineering.

Gold Layer: Aggregated, business-ready data, including engineered features and model predictions.

This architecture ensures data traceability, reliability, and optimized performance for downstream tasks.

Step 3: Feature Engineering
Park-level aggregated features are created from time-series data using the create_aggregated_features function. This involves:

Grouping: Data is grouped by park identifiers (park_id, name, city, area_sqm, latitude, longitude).

Aggregation: Features like max_aqi, avg_visitors, min_sentiment, and event_days_count are aggregated.

Derived Features: New features like aqi_range, sentiment_range, event_frequency, and park_density are computed to enrich the dataset for machine learning.

Python

# Example Snippet (Conceptual)
# from pyspark.sql import functions as F

# def create_aggregated_features(df_spark):
#     park_features = df_spark.groupBy("park_id", "name", "city", "area_sqm", "latitude", "longitude") \
#         .agg(
#             F.max(F.col("aqi")).alias("max_aqi"),
#             F.avg(F.col("visitor_count")).alias("avg_visitors"),
#             F.min(F.col("sentiment_score")).alias("min_sentiment"),
#             F.sum(F.when(F.col("event_day") == True, 1).otherwise(0)).alias("event_days_count")
#         )
#     park_features_enhanced = park_features \
#         .withColumn("aqi_range", F.col("max_aqi") - F.col("min_aqi")) \
#         .withColumn("park_density", F.col("total_visitors") / F.col("area_sqm")) \
#         .fillna(0)
#     return park_features_enhanced
Step 4: Target Variable Creation
A critical step is defining the intervention_required target variable based on specific business logic. This variable indicates whether a park needs attention.

Conditions for Intervention:

max_aqi > 100 (Poor air quality)

max_aqi > 75 AND min_sentiment < 0 (Moderate air quality with negative sentiment)

min_sentiment < -0.5 (Significantly negative sentiment)

max_visitors < 50 AND area_sqm > 10000 (Low visitor count in a large park, possibly indicating neglect or lack of appeal)

The distribution of this target variable is also checked to understand class balance.

Python

# Example Snippet (Conceptual)
# from pyspark.sql.functions import when, col

# def create_target_variable(df_spark):
#     df_with_target = df_spark.withColumn(
#         "intervention_required",
#         when(
#             (col("max_aqi") > 100) |
#             ((col("max_aqi") > 75) & (col("min_sentiment") < 0)) |
#             (col("min_sentiment") < -0.5) |
#             ((col("max_visitors") < 50) & (col("area_sqm") > 10000)),
#             1
#         ).otherwise(0)
#     )
#     return df_with_target
Step 5: Machine Learning Pipeline
The core of the UGSMS involves a robust machine learning pipeline for predicting intervention requirements.

Components:

Models: Logistic Regression, Random Forest, and Gradient Boosting Classifiers are employed.

Hyperparameter Tuning: GridSearchCV is used to find optimal parameters for each model, ensuring peak performance.

Evaluation: Models are rigorously evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, and AUC-ROC, along with visual aids like Confusion Matrices, ROC Curves, and Precision-Recall Curves.

Python

# Example Snippet (Conceptual)
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.model_selection import GridSearchCV

# def create_ml_pipelines():
#     pipelines = {
#         'logistic_regression': Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=42))]),
#         'random_forest': Pipeline([('scaler', RobustScaler()), ('classifier', RandomForestClassifier(random_state=42))]),
#         'gradient_boosting': Pipeline([('scaler', StandardScaler()), ('classifier', GradientBoostingClassifier(random_state=42))])
#     }
#     return pipelines

# param_grids = { # Example param_grids
#     'logistic_regression': {'classifier__C': [0.1, 1.0, 10.0]},
#     'random_forest': {'classifier__n_estimators': [50, 100]},
#     'gradient_boosting': {'classifier__learning_rate': [0.1, 0.2]}
# }
Step 6: Model Comparison and MLflow Integration
The performance of each trained model is compared to identify the best-suited algorithm for the task.

Visualization: A custom function create_model_comparison_visualization generates insightful plots, comparing key metrics (Accuracy, Precision, Recall, F1-Score) and illustrating the relationship between cross-validation scores and test performance.

MLflow Tracking: All experiments, including parameters, metrics, and models, are meticulously tracked using MLflow. This ensures:

Reproducibility: Easily recreate past experiments.

Efficient Model Management: Centralized logging of model artifacts.

Collaboration: Share and compare results seamlessly.

Python

# MLflow Logging (Conceptual)
# import mlflow
# mlflow.sklearn.log_model(best_pipeline, f"{model_name}_pipeline")
# mlflow.log_params(grid_search.best_params_)
# mlflow.log_metrics(metrics)
Step 7: Feature Importance Analysis
For tree-based models, feature importance is analyzed to understand which park characteristics most significantly influence the prediction of intervention needs.

Python

# Example Snippet (Conceptual)
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# def analyze_feature_importance(model_results, feature_names, best_model_name):
#     best_model = model_results[best_model_name]['model']
#     if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
#         importances = best_model.named_steps['classifier'].feature_importances_
#         feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
#         # ... visualization logic ...
#         return feature_importance_df
Step 8: Generating & Visualizing Recommendations
The system culminates in generating actionable recommendations based on the predicted intervention probabilities.

Recommendation Logic: A custom generate_recommendation function translates probabilities into specific advice, sometimes tailored by city (e.g., Hamburg, Berlin).

prob > 0.9: High urgency (e.g., increase tree canopy, public awareness, waste management).

prob > 0.7: Moderate concern (e.g., promote eco-tourism, park ranger programs).

prob > 0.4: Low risk (e.g., monitor visitor impact, seasonal clean-up drives).

else: No immediate action needed (regular observation).

These recommendations are then visualized to provide park managers with an intuitive and clear understanding of where and how to intervene. Predictions are also saved back to Spark tables for persistence and further analysis.

Python

# Example Snippet (Conceptual)
# from datetime import datetime

# def generate_recommendation(prob, city):
#     if prob > 0.9: return "High urgency: ..."
#     elif prob > 0.7: return "Moderate concern: ..."
#     # ... other conditions ...
#     else: return "No immediate action needed: ..."

# def save_predictions_to_spark(...):
#     # ... make predictions ...
#     predictions_df = pd.DataFrame({
#         'park_id': df_pandas['park_id'],
#         'intervention_pred': predictions,
#         'intervention_probability': prediction_proba,
#         'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     })
#     predictions_spark = spark.createDataFrame(predictions_df)
#     predictions_spark.write.mode("overwrite").saveAsTable("...")
🛠️ Technologies Used
Apache Spark: For scalable data ingestion, processing, and transformation.

Delta Lake: For reliable and performant data storage.

MLflow: For machine learning lifecycle management (experiment tracking, model management).

Scikit-learn: For machine learning model implementation.

Pandas: For data manipulation (primarily for initial data loading from synthetic CSVs).

Matplotlib & Seaborn: For data visualization.

Python: The primary programming language.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

📄 License
[Specify your project's license here, e.g., MIT, Apache 2.0, etc.]

📞 Contact
For any inquiries or further information, please contact [Your Name/Email/LinkedIn Profile].