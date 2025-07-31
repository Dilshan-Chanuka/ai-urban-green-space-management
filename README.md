
# 🌳 AI-Powered Urban Green Space Management System (UGSMS)

An end-to-end machine learning pipeline built on Databricks to optimize park-level interventions in German cities using environmental, usage, and sentiment data.

---

## 🚀 Project Overview

The **Urban Green Space Management System (UGSMS)** is designed to help city authorities and park managers make **data-driven decisions** by integrating multi-source data (air quality, footfall, sentiment, and geographic info) and applying ML to recommend timely interventions.

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

## 📦 Architecture

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

Functions:
```python
def read_data(url): ...
def clean_data(df): ...
```

---

### 2. Data Storage: Medallion Architecture
- **Bronze**: Raw source data
- **Silver**: Cleaned and joined datasets
- **Gold**: Aggregated features and ML predictions

---

### 3. Feature Engineering
Aggregates metrics such as AQI, sentiment, visitor counts, and events per park using:
```python
def create_aggregated_features(df_spark): ...
```

---

### 4. Target Variable
Defines `intervention_required` based on business rules:
```python
def create_target_variable(df_spark): ...
```

---

### 5. Machine Learning Pipeline

ML Algorithms Used:
- Logistic Regression
- Random Forest
- Gradient Boosting

Each model is built using:
```python
def create_ml_pipelines(): ...
```

Hyperparameter tuning via `GridSearchCV` with custom `param_grids`.

---

### 6. Model Evaluation & Comparison

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

Visual tools include:
- Confusion Matrix
- ROC & PR Curves
- Comparison Bar & Scatter Plots

```python
def evaluate_model_performance(...): ...
def create_model_comparison_visualization(...): ...
```

---

### 7. MLflow Integration
All models and experiments are logged using:
```python
with mlflow.start_run(...): ...
```

---

### 8. Feature Importance
Visualizes which features influence predictions most:
```python
def analyze_feature_importance(...): ...
```

---

### 9. Saving & Visualizing Predictions

Model outputs are saved to Spark tables:
```python
def save_predictions_to_spark(...): ...
```

---

### 🔮 Recommendation Engine
Recommends actions based on prediction probabilities:
```python
def generate_recommendation(prob, city): ...
```

---

## 📊 Demo & Visualizations

Coming soon:  
✅ Model performance charts  
✅ Feature importance plots  
✅ Intervention recommendation maps

---

## 📁 Project Structure

```bash
📦 ugsms/
├── notebooks/
│   ├── 1_data_ingestion.py
│   ├── 2_data_cleaning.py
│   ├── 3_feature_engineering.py
│   ├── 4_ml_pipeline.py
│   ├── 5_model_evaluation.py
│   └── 6_recommendations.py
├── data/
│   └── *.csv (synthetic data)
├── README.md
└── requirements.txt
```

---

## 🛠️ Installation

This project is built for **Databricks Community Edition**.

1. Clone the repo:
```bash
git clone https://github.com/yourusername/UGSMS.git
```
2. Upload notebooks to Databricks
3. Create a cluster with:
   - Runtime: 12.x LTS (Scala 2.12, Spark 3.x)
4. Run notebooks sequentially

---

## 🤝 Contributing

Pull requests are welcome! Please fork the repo and open a PR.

---

## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

Made with ❤️ by **Dilshan Chanuka**  
🔗 [LinkedIn](https://www.linkedin.com/in/dilshan-chanuka/)  
