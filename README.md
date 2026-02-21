# ⚙️ Predictive Maintenance Engine (Enterprise Edition)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-red)

> **A Cost-Optimized Machine Learning System for Industrial Failure Prediction.**
> *Optimizing for Dollars ($), Not Just Accuracy.*

---

## 💼 Executive Summary & Business Case

In heavy manufacturing, equipment failure is the single largest driver of operational variance. Traditional maintenance strategies are inefficient:
* **Reactive Maintenance:** Wait for failure (High Cost: Downtime + Catastrophic Damage).
* **Preventive Maintenance:** Schedule fixed repairs (High Waste: Replacing healthy parts).

This repository implements a **Cost-Optimized Predictive Maintenance System (PdM)** designed to minimize Total Operational Expenditure (OpEx). Unlike standard classification models that optimize for Accuracy, this engine utilizes a **Custom Profit-Aware Loss Function** to mathematically balance the trade-off between inspection costs and failure costs.

### The Financial Constraint Matrix
The model is tuned against a specific asymmetric cost structure defined by business stakeholders:
* **False Negative (Missed Failure):** $10,000 (Cost of downtime & replacement).
* **False Positive (False Alarm):** $500 (Cost of manual inspection).

**Objective:** Minimize $\text{Total Cost} = (\text{FN} \times 10,000) + (\text{FP} \times 500)$

---

## 🛠️ Technical Architecture

To address the **97:3 Class Imbalance** and physical complexity of the dataset, the pipeline adheres to strict MLOps principles:

### 1. Physics-Aware Feature Engineering
We rejected the "black box" approach in favor of domain-driven engineering. By applying first principles of **Thermodynamics** and **Rotational Mechanics**, we derived features that isolate failure signals:
* **`Temp_Diff` (Heat Dissipation):** Process Temperature - Air Temperature.
* **`Power` (Work Capacity):** Torque $\times$ Rotational Speed.
* **`Force_Ratio` (Strain):** Torque / Speed.

**Validation:** `Temp_Diff` and `Power` emerged as the top 2 predictors, confirming that domain-driven features outperform raw sensor data.

### 2. Leakage-Proof Training Pipeline
* **Pipeline-Integrated SMOTE:** Synthetic Minority Over-sampling Technique (SMOTE) is applied *strictly* within the Cross-Validation training folds. This prevents data leakage where synthetic samples might inadvertently "teach" the validation set.
* **Stratified Sampling:** Ensures the rare failure distribution (3%) is statistically consistent across Train, Validation, and Test sets.

### 3. Business-Centric Optimization
* **Custom Scorer:** Hyperparameter tuning (GridSearch) was conducted using a custom scoring function that maps predictions directly to dollar costs, rather than abstract metrics like F1-Score.
* **Threshold Moving:** The decision boundary was dynamically shifted from `0.5` to an optimal `0.27`, prioritizing **Recall (90%)** to safeguard against expensive catastrophic failures.

---

## 📊 Evaluation & Results

The system benchmarked multiple algorithms (XGBoost, Random Forest, Logistic Regression) in a "Champion vs. Challenger" framework. **LightGBM** was selected as the Production Champion.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.99** | Near-perfect separation of healthy vs. failing states. |
| **Recall (Failures)** | **90%** | The system successfully flags 9 out of 10 impending failures. |
| **Precision** | **57%** | We accept a ~40% false alarm rate to ensure safety. |
| **Total Projected Cost** | **$93,000** | **Minimized** relative to reactive baseline ($600k+). |

### 📸 Visual Evidence

#### 1. Confusion Matrix (The "Money" Chart)
*At Threshold 0.27, we catch 61/68 failures, accepting only 46 false alarms.*
![Confusion Matrix](artifacts/graphs/confusion_matrix.png)

#### 2. Feature Importance (Physics Validation)
*Note how engineered features (`Temp_Diff`, `Power`) dominate raw features.*
![Feature Importance](artifacts/graphs/feature_importance.png)

#### 3. ROC Curve (Model Discrimination)
*AUC of 0.99 indicates exceptional ranking ability.*
![ROC Curve](artifacts/graphs/roc_curve.png)

---

## 📂 Repository Structure

```text
predictive-maintenance-engine/
│
├── artifacts/                  # Auto-generated outputs
│   ├── data/                   # Raw & Processed Datasets (Google Drive Synced)
│   ├── graphs/                 # ROC Curves, Confusion Matrices, Feature Importance
│   └── models/                 # Serialized Champion Models (.pkl)
│
├── notebooks/
│   └── main_execution.ipynb    # Main Driver Script (Google Colab / Local)
│
├── src/                        # Modularized Source Code ("The Brain")
│   ├── __init__.py
│   ├── config.py               # Global Constants & Cost Matrix
│   ├── data_ingestion.py       # Robust Google Drive Download & Cleaning
│   ├── feature_engineering.py  # Physics Formulas & Sklearn Pipelines
│   ├── modeling.py             # Model Zoo, SMOTE, & Cost Scorer
│   └── evaluation.py           # Threshold Tuning & Financial Analysis
│
├── requirements.txt            # Dependencies
└── README.md                   # Project Documentation

🚀 How to Run
This pipeline is optimized for Google Colab with seamless Google Drive integration.

Clone & Upload:
  -Clone this repository.
  -Upload the predictive-maintenance-engine folder to your Google Drive.

Execute:
  -Open notebooks/main_execution.ipynb in Google Colab.
  -Run all cells. The script will automatically:
    -Mount Google Drive.
    -Install dependencies (imbalanced-learn, lightgbm, etc.).
    -Execute the Training & Evaluation Pipeline.
    -Save all artifacts (Models, Graphs) back to your Drive.