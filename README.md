# Software Project Risk Prediction
This repository contains code for predicting software project risks using machine learning techniques.

# Requirements
To run the code, you need to have the following Python packages installed:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pyspark
- xgboost
- shap
- streamlit

# Dataset
The dataset used in this project is available at [Apache Jira Issues](https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues).

Please download the dataset and place the CSV files in the `src/apache` directory.

# Usage

Run the notebooks in order:

1. `issue-prep.ipynb`: Prepares the issue data.
2. `changelog-prep.ipynb`: Prepares the changelog data.
3. `jira-merge.ipynb`: Merges the issue and changelog data.
4. `jira-label.ipynb`: Labels the merged data.
5. `jira-eda.ipynb`: Performs exploratory data analysis on the labeled data, and create new features.
6. `jira-feature.ipynb`: Extracts best features from the labeled data.
7. `mls_*.ipynb`: Runs machine learning models for schedule risk prediction.
8. `mlq_*.ipynb`: Runs machine learning models for quality risk prediction.
9. `model_comparison.ipynb`: Compares the performance of different machine learning models.

You can also run the `risk_prediction_demo.py` script to see a demo of the risk prediction process by running the command `streamlit run risk_prediction_demo.py` (make sure to run all the notebooks first).
