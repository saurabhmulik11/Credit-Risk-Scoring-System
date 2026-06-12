# Credit-Risk-Scoring-System
# Credit Risk Scoring System

## Overview

The Credit Risk Scoring System is an end-to-end Machine Learning project designed to predict the probability of loan default and assist financial institutions in making informed lending decisions.

Traditional loan approval processes rely heavily on manual assessment and fixed rule-based systems. This project leverages Machine Learning to evaluate borrower risk, generate a credit risk score, and automate loan approval recommendations.

The system predicts the likelihood of default, converts it into a risk score ranging from 0 to 100, and categorizes applicants into approval, review, or rejection groups.

---

## Problem Statement

Financial institutions face significant losses when borrowers fail to repay loans. Accurately identifying high-risk applicants is critical for minimizing defaults while maintaining profitable lending operations.

The objective of this project is to build a predictive credit risk scoring system that:

* Predicts loan default probability
* Generates a credit risk score
* Supports automated lending decisions
* Improves risk management and portfolio quality

---

## Dataset

**Dataset Used:** Give Me Some Credit Dataset

The dataset contains customer demographic, financial, and credit-related information.

### Features

| Feature                              | Description                    |
| ------------------------------------ | ------------------------------ |
| RevolvingUtilizationOfUnsecuredLines | Credit utilization ratio       |
| age                                  | Applicant age                  |
| DebtRatio                            | Monthly debt obligations ratio |
| MonthlyIncome                        | Applicant monthly income       |
| NumberOfOpenCreditLinesAndLoans      | Total open credit accounts     |
| NumberOfTime30-59DaysPastDueNotWorse | Delinquency history            |
| NumberOfDependents                   | Number of dependents           |

### Target Variable

| Value | Meaning     |
| ----- | ----------- |
| 0     | Non-Default |
| 1     | Default     |

---

## Project Workflow

### 1. Data Preprocessing

* Missing value handling
* Data quality checks
* Feature scaling
* Train-test split
* Class imbalance handling using SMOTE

---

### 2. Feature Engineering

Created additional financial indicators:

#### Debt-to-Income Ratio

```python
DebtRatio / MonthlyIncome
```

#### Income per Dependent

```python
MonthlyIncome / (NumberOfDependents + 1)
```

#### Credit Utilization Features

Derived borrower financial behavior metrics.

---

### 3. Model Development

Two models were trained and compared.

#### Baseline Model

* Logistic Regression

#### Final Model

* XGBoost Classifier

---

### 4. Risk Scoring System

The model predicts the probability of default.

Risk score is calculated as:

```text
Risk Score = (1 - Probability of Default) × 100
```

### Example

| Probability of Default | Risk Score |
| ---------------------- | ---------- |
| 0.10                   | 90         |
| 0.25                   | 75         |
| 0.50                   | 50         |
| 0.80                   | 20         |

---

### 5. Decision Engine

Loan decisions are generated based on risk score thresholds.

| Risk Score | Decision      |
| ---------- | ------------- |
| > 70       | Approve       |
| 40 – 70    | Manual Review |
| < 40       | Reject        |

---

## Model Evaluation

The models were evaluated using:

### Classification Metrics

* ROC-AUC
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Business Focus

Special emphasis was placed on Recall because failing to identify high-risk borrowers can result in significant financial losses.

---

## Tech Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-Learn (SMOTE)

### Visualization

* Matplotlib
* Seaborn

### Model Explainability

* SHAP

### Deployment

* Streamlit
* FastAPI / Flask

### Version Control

* Git
* GitHub

---

## Project Structure

```text
credit-risk-scoring-system/
│
├── data/
│   └── credit_data.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│
├── models/
│   ├── xgboost_model.pkl
│   └── scaler.pkl
│
├── app/
│   ├── api.py
│   └── streamlit_app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Results

* Successfully built an end-to-end credit risk prediction pipeline.
* Improved borrower risk assessment through financial feature engineering.
* XGBoost outperformed Logistic Regression in identifying high-risk borrowers.
* Generated interpretable risk scores for lending decisions.
* Designed a deployable solution suitable for real-world financial applications.

---

## Business Impact

### Risk Reduction

Improves identification of potentially risky borrowers before loan approval.

### Faster Decision Making

Supports automated and scalable credit evaluation.

### Better Portfolio Quality

Helps financial institutions balance profitability with risk management.

### Explainable AI

Provides transparency into model predictions using SHAP values.

---

## Future Enhancements

* Real-time API deployment using FastAPI
* Credit score dashboard using Streamlit
* Model monitoring and drift detection
* Integration with cloud services (AWS EC2, S3)
* Automated retraining pipeline using MLOps practices

---

## Key Skills Demonstrated

* Machine Learning
* Credit Risk Analytics
* Feature Engineering
* Imbalanced Data Handling
* XGBoost
* Model Evaluation
* Explainable AI (SHAP)
* Financial Data Analysis
* Streamlit
* FastAPI
* MLOps Fundamentals

---

## Author

**Saurabh Mulik**

Machine Learning | FinTech Analytics | GenAI | MLOps Enthusiast

This project demonstrates the application of Machine Learning for financial risk assessment and decision-making in lending systems.
