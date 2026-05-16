 md
  CT-583: Tools and Techniques for Data Science

 End-of-Term Technical Challenge — Task 1

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)

 Project Overview

This project was completed as part of the CT-583 end-of-term technical challenge. The objective of the assignment is to build a machine learning model that predicts whether a person earns more than $50K per year using the UCI Adult Census Income dataset.

The notebook includes the complete workflow starting from data loading and preprocessing to model training, evaluation, fairness analysis, and feature importance interpretation.

 Open in Google Colab

You can run the notebook directly in Google Colab using the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/rahimpg4300201-ship-it/CT583-Data-Science-Assignment/blob/main/Assignment_2.ipynb)

 

 Repository Structure

text
CT583-Data-Science-Assignment/

├── Assignment_2.ipynb       Main notebook containing the full implementation
├── task1_report.md          Written report and findings
└── README.md                Project documentation
 

 Dataset

The project uses the UCI Adult Census Income Dataset available through OpenML.

* Approximately 48,000 records
* 14 input features
* Binary classification target:

  * > 50K income
  * <=50K income

The dataset is loaded directly inside the notebook using Scikit-learn.

 Main Tasks Covered

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Logistic Regression model
* Random Forest model
* XGBoost model
* Hyperparameter tuning
* Model evaluation using multiple metrics
* Fairness and bias analysis
* Failure case analysis
* Feature importance visualization

 Feature Engineering

Additional features were created from the original dataset, including:

* Age groups
* Net capital gain/loss
* Working hour categories
* Education-age interaction feature

 Fairness Analysis

The model performance was compared across different demographic groups including gender and race to identify possible bias in predictions.

 How to Run

 Option 1 — Google Colab

1. Open the notebook using the Colab badge
2. Click Runtime → Run All
3. Wait for all cells to execute

 Option 2 — Local Jupyter Notebook

bash
git clone https://github.com/rahimpg4300201-ship-it/CT583-Data-Science-Assignment.git

cd CT583-Data-Science-Assignment

pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter

jupyter notebook


 Author

| Detail     | Information                                  |
|  - |             -- |
| Name       | Insha Rahim Shahwani                         |
| Student ID | GD-001/2025                                  |
| Course     | CT-583 Tools and Techniques for Data Science |

 




