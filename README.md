# Supervised Learning: End-to-End Fraud Detection Pipeline & EDA

## 📌 Project Overview
This repository contains my submission for **Project 2** of the DecodeLabs Industrial Data Science Training Track (Batch 2026). The project implements a robust machine learning infrastructure to detect fraudulent e-commerce transactions using a highly imbalanced dataset of 1,200 order records. It moves from exploratory analysis to cross-validated supervised learning pipelines.

---

## 📊 Core Data Forensics & Strategic Insights (EDA)
Before model training, a thorough **Exploratory Data Analysis (EDA)** was performed on the transactional ledger. Key financial health metrics discovered include:
* **Total Portfolio Revenue:** \$1,264,761.96
* **Total Transactions Analysed:** 1,200 orders
* **Mean Ticket Size:** \$1,053.97
* **Top Revenue Drivers:** *Chairs* (\$195,620.11) and *Printers* (\$195,612.61) dominate market capture.
* **Acquisition Engine:** Social channels like *Instagram* (\$275,285.45) yield the highest ROI compared to organic search channels.

---

## 🛠️ Machine Learning Pipeline Architecture
Fraud detection inherently deals with highly imbalanced classes. To prevent data leakage and guarantee objective scoring, the backend architecture handles data through a strictly structured pipeline:

1. **Pre-processing & Encoding:** Automated categorical encoding (Label Encoding) and numerical feature scaling using `StandardScaler`.
2. **Handling Class Imbalance:** Integrated **SMOTE (Synthetic Minority Over-sampling Technique)** strictly inside the training split loop to prevent target leakage into the validation folds.
3. **Hyperparameter Optimization:** Utilized **5-Fold GridSearchCV** optimizing specifically for **Recall Score** (ensuring false negatives are minimized, as missing a fraud transaction is highly costly for business operations).
4. **Model Comparison:** Evaluated and cross-examined a parametric model (**Logistic Regression**) against a non-parametric ensemble (**Random Forest Classifier**).

---

## 📈 Evaluation & Model Performance
The final pipeline evaluated the models on an un-resampled, realistic imbalanced test distribution. The performance matrix benchmarks are detailed below:

| Machine Learning Model | Recall (Target Metric) | ROC-AUC Score | Status |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | *0.8000+* | *0.8500+* | Baseline Standard |
| **Random Forest Classifier** | **0.9100+** | **0.9400+** | 🏆 **Winner Model** |

### Key Takeaways:
* **Random Forest** successfully navigated non-linear transactional anomalies and achieved superior class separation.
* Optimizing for Recall ensured that the pipeline flags over 91% of actual fraudulent attempts, safeguarding revenue channels.

---

## 📁 Repository Structure
* `fraud_detection_pipeline.py`: Production-ready python script containing complete data processing, SMOTE integration, grid-search tuning, and visualization outputs.
* `Exploratory_Data_Analysis_Project_Report.pdf`: Deep-dive data forensics report outlining statistical distribution summaries, outliers mapping, and portfolio suggestions.
* `Dataset_for_Data_Analytics.csv`: The underlying e-commerce transactional ledger.

---

## 🚀 How to Run the Pipeline
1. Clone this repository to your local system.
2. Install required statistical and machine learning dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
