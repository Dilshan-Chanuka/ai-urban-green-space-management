
# 🌳 AI-Powered Urban Green Space Management System (UGSMS)

## 📚 Table of Contents

- [🚀 Project Overview](#-project-overview)
- [🔑 Key Features](#-key-features)
- [🌍 Project Value](#-project-value)
- [📁 Project Structure](#-project-structure)
- [📦 Architecture](#-architecture)
  - [Step 1: Data Ingestion and Cleaning](#step-1-data-ingestion-and-cleaning)
  - [Step 2: Data Storage and Medallion Architecture](#step-2-data-storage-and-medallion-architecture)
  - [Step 3: Feature Engineering](#step-3-feature-engineering)
  - [Step 4: Target Variable Creation](#step-4-target-variable-creation)
  - [Step 5: Machine Learning Pipeline](#step-5-machine-learning-pipeline)
  - [Step 6: Model Evaluation & Comparison](#step-6-model-evaluation--comparison)
  - [Step 7: MLflow Integration](#step-7-mlflow-integration)
  - [Step 8: Feature Importance](#step-8-feature-importance)
  - [Step 9: Saving & Visualizing Predictions](#step-9-saving--visualizing-predictions)
  - [Step 10: Recommendation Engine](#-step-10-recommendation-engine)
- [🛠️ Installation](#️-installation)
- [📄 License](#-license)
- [📬 Contact](#-contact)


## 🚀 Project Overview

The **AI-Powered Urban Green Space Management System (UGSMS)** is an end-to-end machine learning pipeline developed to optimize the management of urban parks and green spaces. By leveraging advanced data analytics, machine learning, and real-time data processing, the UGSMS aims to improve the overall quality, sustainability, and visitor experience of urban green spaces.
---

## 🔑 Key Features

- ✅ **Multi-source Data Integration**: Ingests air quality, footfall, sentiment, and park data.
- ⚡ **Real-time Data Processing**: Uses Apache Spark for scalable and efficient processing.
- 🧠 **ML Pipeline**: Logistic Regression, Random Forest, and Gradient Boosting for prediction.
- 🏛️ **Delta Lake Architecture**: Bronze → Silver → Gold medallion model.
- 📈 **MLflow Integration**: Tracks experiments, models, and metrics.
- 🎯 **Automated Recommendations**: Actionable outputs for park improvement planning.

---

## 🌍 Project Value

- 🌱 **Proactive Park Management**
- 🔧 **Efficient Resource Allocation**
- 📊 **Data-Driven Decision Making**
- 🍃 **Enhanced Environmental Impact**

---
## 📁 Project Structure

```bash
📦 ugsms/
├── notebooks/
│   ├── 01_bronze_notebook.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_gold_notebook.ipynb
│   └── 04_model_traning.ipynb    
├── data_sample/
│   ├── *german_national_parks.csv (synthetic data)
│   ├── *national_parks_air_quality.csv (synthetic data)
│   ├── *national_parks_air_quality.csv (synthetic data)
│   └── *national_parks_air_quality.csv (synthetic data)
├── README.md
└── requirements.txt
```
---

## 📦 Architecture

The system architecture is designed to find a balance between **scalability**, **maintainability**, and **performance**.

![UGSMS Architecture](./docs/Architecture.png)

### Step 1: Data Ingestion and Cleaning
The first step in developing the UGSMS is to ingest and clean the data from various sources.

**Note on Data:** For the purpose of this project, synthetic data was generated to simulate real-world environmental, usage, and sentiment patterns. These data files are stored in a GitHub repository and are used directly for ingestion into the system.

* **Air Quality Data** (`national_parks_air_quality.csv`)
    * Air Quality Index (AQI)
    * NO2, PM2.5, O3 levels
* **Footfall Data** (`national_parks_footfall.csv`)
    * Visitor counts
    * Event day indicators
* **Sentiment Data** (`national_parks_sentiment.csv`)
    * Tweet text analysis
    * Sentiment scores (-1 to 1)
    * Sentiment labels
* **Parks Information** (`german_national_parks.csv`)
    * Geographic coordinates
    * Area measurements

The data ingestion process involves reading the CSV files into Spark DataFrames using the `read_data` function.
The `clean_data` function performs several cleaning operations:

* **Removing null values:** The function uses `df.dropna` to remove rows with null values in any column except for empty strings.
* **Converting timestamp columns:** If the DataFrame contains a timestamp column, the function converts it to a timestamp data type using `to_timestamp`. The format `M/d/yyyy H:mm` is used to parse the timestamp.
* **Casting data types:** The function casts the `visitor_count` column to an integer data type using `col("visitor_count").cast("integer")`. Similarly, it casts the `event_day` column to a boolean data type using `col("event_day").cast("boolean")`.

By cleaning and preprocessing the data, the system ensures that it is in a suitable format for further analysis and modeling.

Functions:
```ipynbthon
def read_data(url): ...
def clean_data(df): ...
```

---

### Step 2: Data Storage and Medallion Architecture
The system implements a medallion architecture with three layers:
- **Bronze**: Raw, unprocessed data from sources
- **Silver**: Cleaned and integrated data
- **Gold**: Aggregated, business-ready data and predictions

---

### Step 3: Feature Engineering
The **UGSMS** uses feature engineering to create **park-level aggregated features** from time-series data.

The `create_aggregated_features` function performs the following operations:  

- **Grouping data**: The function groups the data by `park_id`, `name`, `city`, `area_sqm`, `latitude`, and `longitude` using `groupBy`.  
- **Aggregating features**: The function aggregates various features, including air quality, visitor counts, sentiment scores, and event days, using `agg`.  
- **Creating derived features**: The function creates derived features, such as `aqi_range`, `sentiment_range`, `event_frequency`, and `park_density`, using `withColumn`.

The aggregated features are then used as input to machine learning models to predict intervention requirements.

```ipynbthon
def create_aggregated_features(df_spark): ...
```

---

### Step 4: Target Variable Creation
The target variable `intervention_required` is created based on business logic.

The `create_target_variable` function creates the `intervention_required` target variable based on the following conditions:  

- **Air quality**: If the maximum AQI is greater than 100, intervention is required.  
- **Sentiment and air quality**: If the maximum AQI is greater than 75 and the minimum sentiment score is less than 0, intervention is required.  
- **Sentiment**: If the minimum sentiment score is less than -0.5, intervention is required.  
- **Visitor count and park area**: If the maximum visitor count is less than 50 and the park area is greater than 10,000 square meters, intervention is required.

The target variable is used to train machine learning models to predict intervention requirements.
```ipynbthon
def create_target_variable(df_spark): ...
```

---

### Step 5: Machine Learning Pipeline

ML Algorithms Used:
- Logistic Regression
- Random Forest
- Gradient Boosting

Each model is built using:
```ipynbthon
def create_ml_pipelines(): ...
```

Hyperparameter tuning via `GridSearchCV` with custom `param_grids`.

---

### Step 6: Model Evaluation & Comparison

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

Visual tools include:
- Confusion Matrix
- ROC Curve & Precision-Recall Curve
- Model Performance Comparison & CV Score vs Test F1 Score

**Figure: Evaluating logistic_regression_performance.**
![UGSMS Architecture](./docs/Evaluating_logistic_regression_performance.png)
![UGSMS Architecture](./docs/Evaluating_random_forest_performance.png)
![UGSMS Architecture](./docs/Evaluating_gradient_boosting_performance.png)
![UGSMS Architecture](./docs/Performance_Comparison_&_CV_Score_vs_Test_F1_Score.png)

```ipynbthon
def evaluate_model_performance(...): ...
def create_model_comparison_visualization(...): ...
```

---

### Step 7: MLflow Integration
All models and experiments are logged using MLflow.
The MLflow integration enables comprehensive model management and reproducibility.
```ipynbthon
# Log model to MLflow
with mlflow.start_run(run_name=f"{model_name}_experiment"):
    # Log parameters
    mlflow.log_params(grid_search.best_params_)
    
    # Log metrics
    mlflow.log_metrics(metrics)
    mlflow.log_metric("cv_score", grid_search.best_score_)
    
    # Log model
    mlflow.sklearn.log_model(best_pipeline, f"{model_name}_pipeline")
```

---

### Step 8: Feature Importance
The feature importance is analyzed for the best-performing model:

![UGSMS Architecture](./docs/Analyzing_Feature_Importance_for_random_forest.png)

```ipynbthon
def analyze_feature_importance(...): ...
```

---

### Step 9: Saving & Visualizing Predictions

Model outputs are saved to Spark tables.
To help park managers understand the recommendations, the system generates visualizations that show the intervention probabilities and corresponding recommendations for each park.

![UGSMS Architecture](./docs/Visualizing_Predictions.png)

```ipynbthon
def save_predictions_to_spark(...): ...
```

---

### 🔮 Step 10: Recommendation Engine
The system uses a custom-built function to generate recommendations based on predicted intervention probability and park data.
The `generate_recommendation` function takes into account the predicted intervention probability and the city to provide a tailored recommendation.

![UGSMS Architecture](./docs/Recommendation.png)

```ipynbthon
def generate_recommendation(prob, city): ...
```

---

## 🛠️ Installation

This project is built for **Databricks Community Edition**.

1. Upload notebooks to Databricks
2. Run notebooks sequentially

---

## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

Made by **Dilshan Chanuka**  
🔗 [LinkedIn](https://www.linkedin.com/in/dilshan-chanuka/)  
