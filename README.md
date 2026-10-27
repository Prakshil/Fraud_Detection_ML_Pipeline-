# 🛡️ Credit Card Fraud Detection Production ML Pipeline
### Big Data Analytics (BDA) Course Project

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade machine learning pipeline for detecting fraudulent credit card transactions on highly skewed datasets. Engineered for a Big Data Analytics course, this project combines domain-driven feature engineering, cost-sensitive learning, multi-model benchmarking, and decision threshold optimization—all within a **leak-free cross-validation framework**.

---

## 📌 Project Overview & Problem Statement

* **Dataset:** 284,807 credit card transactions made by European cardholders in September 2013.
* **Class Imbalance:**
  * **Genuine Transactions (Class 0):** 284,315 (99.827%)
  * **Fraudulent Transactions (Class 1):** 492 (0.173%)
  * **Imbalance Ratio:** ~1 fraud per 578 legitimate transactions.
* **The Core Challenge:** Traditional accuracy is deceptive (a naive zero-classifier gets 99.83% accuracy while catching 0 frauds). Models must be evaluated on **Precision-Recall Area Under Curve (PR-AUC)**, **F1-Score**, and **Financial Cost-Utility**.

---

## ⚡ The Hybrid Architecture: Overcoming Public Kernel Pitfalls

Most public fraud detection kernels suffer from **severe data leakage**: they apply synthetic resampling (SMOTE) or feature scaling *before* train-test splitting, resulting in artificial test distributions (e.g., 34% or 50% fraud) that collapse when exposed to real-world data.

Our production pipeline resolves this through a **Hybrid Architecture**:

1. **Strict Leak-Free Partitioning:** Holds out an untouched **20% test partition (56,962 transactions)** with the natural **0.17% fraud prevalence**. Training data is split into 70% Train and 10% Validation (for threshold tuning).
2. **Domain-Informed Feature Engineering:**
   * **`RobustScaler`:** Robust scaling on skewed transaction `Amount` and `Time` using median and IQR (immune to extreme transaction outliers).
   * **Log Transformation:** `log1p(Amount)` to normalize high positive skew ($0 to $25k+).
   * **Cyclical Time Encoding:** 24-hour periodic transformation ($\sin$ and $\cos$) capturing diurnal spending patterns.
   * **Non-Linear PCA Interactions:** Cross-multiplication terms among top discriminating PCA components (`V14_x_V12`, `V14_x_V17`, `V10_x_V11`, `V4_x_V11`).
   * **Anomaly Tail Indicators:** Binary flags identifying extreme tails on high-variance fraud features.
3. **Threshold Optimization:** Dynamically tunes decision thresholds using Precision-Recall trade-offs rather than assuming an arbitrary 0.5 cutoff.

---

## 📊 Benchmark Results (Holdout Test Set: 56,962 Transactions)

All models were evaluated on the **pristine, untouched 20% holdout test partition**:

| Algorithm | Train Time | PR-AUC | ROC-AUC | Tuned Threshold | Tuned F1-Score | Frauds Caught (Recall) | False Alarms (FP) | Default (0.5) False Alarms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Balanced Subsample)** 🏆 | **15.85s** | **0.8402** | **0.9773** | **0.4754** | **0.8229** | **79 / 98 (80.6%)** | **15** | **15** |
| **Neural Network (MLP Classifier)** | 30.26s | **0.7991** | **0.9758** | 0.1163 | 0.7568 | **84 / 98 (85.7%)** | 40 | 17 |
| **HistGradientBoosting (Balanced)** | **2.37s** | 0.7365 | 0.9661 | 0.9185 | 0.7477 | 80 / 98 (81.6%) | 36 | 150 |
| **Logistic Regression (Cost-Sensitive)** | **4.19s** | 0.7071 | 0.9747 | 0.9999 | 0.7547 | 80 / 98 (81.6%) | 34 | 1,522 |

### Key Findings:
* **Top Performer:** **Random Forest** achieved the highest PR-AUC (**0.8402**) and F1-score (**0.8229**), stopping **80.6% of frauds with only 15 false alarms** across 56,864 genuine transactions (Precision = **84.0%**).
* **Highest Catch Rate:** The **MLP Neural Network** caught **85.7% of all frauds** (84/98).
* **Threshold Tuning Impact:** On Cost-Sensitive Logistic Regression, default thresholding produced **1,522 false alarms** (F1 = 0.1030). Tuning the threshold reduced false alarms to **34** and boosted F1 to **0.7547**.

Visual performance artifacts can be found in [`reports/`](reports/):
* Precision-Recall & ROC Curves: `reports/precision_recall_and_roc_curves.png`
* Confusion Matrix Grid: `reports/confusion_matrix_grid.png`
* Metrics Summary: `reports/benchmark_summary.csv`

---

## 💼 Business Cost-Utility Analysis

In financial operations, false negatives and false positives have asymmetric costs:
* **Cost of False Negative ($FN$):** Average fraud transaction loss = **$122.21**.
* **Cost of False Positive ($FP$):** Cardholder friction & verification cost = **$10.00**.

$$\text{Total Cost} = (FN \times \$122.21) + (FP \times \$10.00)$$

* **Unmitigated Fraud Loss (No Model):** $98 \times \$122.21 = \mathbf{\$11,976.58}$
* **Random Forest Pipeline Cost:** $(19 \times \$122.21) + (15 \times \$10.00) = \mathbf{\$2,471.99}$
* **Net Financial Savings:** **$9,504.59 (79.4% fraud loss reduction)** on a single test partition.

---

## 📁 Repository Structure

```
├── credit_card_fraud_pipeline.ipynb    # Comprehensive course notebook (EDA, plots, story)
├── run_pipeline.py                      # Standalone production runner & benchmark
├── src/
│   ├── __init__.py                      # Package exports
│   ├── feature_engineering.py           # FraudFeatureEngineer transformer (scikit-learn)
│   ├── evaluation.py                   # PR-AUC, ROC, threshold tuning, and plotting routines
│   └── pipeline.py                      # FraudDetectionPipeline class (save/load/tune)
├── models/
│   └── best_fraud_pipeline.joblib       # Serialized winning pipeline artifact (1.98 MB)
├── reports/
│   ├── benchmark_summary.csv            # Summary metrics table
│   ├── precision_recall_and_roc_curves.png
│   └── confusion_matrix_grid.png
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Prakshil/Fraud_Detection_ML_Pipeline-.git
cd Fraud_Detection_ML_Pipeline-
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn joblib
```

### 2. Dataset Setup
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root directory.

### 3. Run the Production Pipeline
To train, benchmark all candidate models, export high-res charts, and save the best pipeline:
```bash
python run_pipeline.py
```

### 4. Direct Inference with the Saved Model
```python
import pandas as pd
from src.pipeline import FraudDetectionPipeline

# Load serialized pipeline artifact
pipeline = FraudDetectionPipeline.load("models/best_fraud_pipeline.joblib")

# Pass raw transaction data (accepts unscaled Time, Amount, V1-V28)
df_new = pd.read_csv("creditcard.csv").drop(columns=['Class']).iloc[:5]

probabilities = pipeline.predict_proba(df_new)
predictions = pipeline.predict(df_new)

print("Fraud Probabilities:", probabilities)
print("Binary Predictions (0=Normal, 1=Fraud):", predictions)
```

---

## 📝 License
This project is licensed under the MIT License.
