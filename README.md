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

| Algorithm | Train Time | PR-AUC | ROC-AUC | Tuned Threshold | Tuned F1-Score | Frauds Caught (Recall) | False Alarms (FP) | Total Financial Cost | Default (0.5) False Alarms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Balanced Subsample)** 🏆 | **12.62s** | **0.8344** | 0.9648 | **0.5847** | **0.8065** | **75 / 98 (76.5%)** | **13** | **$2,940.83** | **15** |
| **Neural Network (MLP Classifier)** | 17.38s | **0.8307** | **0.9810** | 0.5919 | **0.8079** | **82 / 98 (83.7%)** | 23 | **$2,185.36** | 28 |
| **HistGradientBoosting (Balanced)** | **1.58s** | 0.7364 | 0.9572 | 0.9315 | 0.7407 | 80 / 98 (81.6%) | 38 | $2,579.78 | 139 |
| **Logistic Regression (Cost-Sensitive)** | **2.23s** | 0.7266 | 0.9783 | 0.9999 | 0.7547 | 80 / 98 (81.6%) | 34 | $2,539.78 | 1,459 |

### Key Findings:
* **Top Performer:** **Random Forest** achieved the highest PR-AUC (**0.8344**), stopping **76.5% of frauds with only 13 false alarms** across 56,864 genuine transactions (Precision = **85.2%**, F1 = **0.8065**).
* **Highest Catch Rate:** The **MLP Neural Network** caught **83.7% of all frauds** (82/98) with only 23 false alarms, achieving the lowest net business cost (**$2,185.36**).
* **Threshold Tuning Impact:** On Cost-Sensitive Logistic Regression, default thresholding produced **1,459 false alarms** (F1 = 0.1070). Tuning the threshold reduced false alarms to **34** and boosted F1 to **0.7547**.

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
* **Random Forest Pipeline Cost:** $(23 \times \$122.21) + (13 \times \$10.00) = \mathbf{\$2,940.83}$
* **Net Financial Savings:** **$9,035.75 (75.4% fraud loss reduction)** on a single test partition.
* **MLP Neural Network Cost:** $(16 \times \$122.21) + (23 \times \$10.00) = \mathbf{\$2,185.36}$ (**$9,791.22 / 81.8% savings**).

---

## 📁 Repository Structure

```
├── app.py                               # Interactive Streamlit Web Platform & XAI Dashboard
├── credit_card_fraud_pipeline.ipynb    # Comprehensive course notebook (executed with all outputs)
├── run_pipeline.py                      # Standalone production runner & benchmark
├── pyproject.toml                       # Build & project metadata specification
├── requirements.txt                     # Pinned runtime and development dependencies
├── src/
│   ├── __init__.py                      # Package exports
│   ├── feature_engineering.py           # FraudFeatureEngineer transformer (scikit-learn compatible)
│   ├── evaluation.py                   # PR-AUC, ROC, vectorized cost/F1 threshold tuning
│   ├── explainer.py                    # FraudExplainer engine (TreeSHAP & automated reasons)
│   └── pipeline.py                      # FraudDetectionPipeline class (save/load/tune/infer)
├── tests/                               # Automated unit test suite (pytest)
│   ├── test_feature_engineering.py
│   ├── test_evaluation.py
│   ├── test_explainer.py
│   └── test_pipeline.py
├── models/
│   ├── best_fraud_pipeline.joblib       # Serialized winning pipeline artifact
│   ├── rf_pipeline.joblib               # Random Forest pipeline
│   ├── mlp_pipeline.joblib              # Multi-Layer Perceptron neural network
│   ├── hgb_pipeline.joblib              # HistGradientBoosting pipeline
│   └── logreg_pipeline.joblib           # Cost-Sensitive Logistic Regression
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
pip install -r requirements.txt
```

### 2. Dataset Setup
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root directory.

### 3. Launch the Interactive Web Platform (Streamlit)
To launch the real-time fraud scoring dashboard with SHAP explainability and algorithm switching:
```bash
streamlit run app.py
```
*Optional custom port or headless execution:*
```bash
streamlit run app.py --server.port 8501 --server.headless true
```

#### Streamlit UI Configuration (`.streamlit/config.toml`)
The platform includes pre-configured visual and server settings:
```toml
[theme]
primaryColor = "#1e3d59"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafc"
textColor = "#1e293b"
font = "sans serif"

[server]
headless = true
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false
```

#### Dashboard Capabilities:
* **🔍 Live Transaction Inspector:** Real-time scoring, risk probability gauge, interactive decision threshold slider, and automated plain-English reasons for fraud alerts.
* **💡 SHAP Explainability Engine:** Waterfall and bar charts breaking down the exact mathematical contribution of each PCA feature and interaction term.
* **🤖 Multi-Algorithm Selection:** Switch between Random Forest, Neural Network (MLP), HistGradientBoosting, and Logistic Regression on the fly.
* **📁 Batch Processing:** Drag-and-drop CSV upload, batch scoring, flag rate metrics, and downloadable audited CSV.
* **💼 Financial ROI Simulator:** Live sliders for Cost of False Negative ($FN) and Cost of False Positive ($FP) modeling business cost impact.

---

## 🧪 Synthetic Test Cases for Manual / Live Verification

To test the live dashboard without needing historical dataset samples, use these **handcrafted, domain-synthesized test scenarios** (preloaded into the dashboard dropdown or enterable manually into the parameter controls):

| Scenario Profile | Time (s) / Time-of-Day | Amount ($) | V14 | V12 | V10 | V17 | V4 | V11 | Remaining V1-V28 | Expected Model Verdict | Predicted Fraud Probability | Primary Algorithmic Reason |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **🟢 Scenario 1: Everyday Legitimate Purchase** | `50,400` *(14:00 Midday)* | **$24.50** | `+0.25` | `-0.15` | `+0.10` | `+0.05` | `-0.30` | `-0.20` | `0.00` | **✅ APPROVED (NORMAL)** | **~0.05%** | Nominal midday purchase; V14 and V4 within normal cardholder variance. |
| **🔴 Scenario 2: Severe Account Takeover** | `10,800` *(03:00 Midnight)* | **$980.00** | **`-7.20`** | **`-4.80`** | **`-3.90`** | **`-4.10`** | **`+3.85`** | **`+3.10`** | `0.00` | **🚨 FLAGGED FRAUD (DECLINE)** | **~71.95%** | Critical V14/V12 negative anomaly; high V4 authorization variance; off-peak timing. |
| **⚠️ Scenario 3: Borderline Suspicious E-Commerce** | `77,400` *(21:30 Evening)* | **$349.00** | **`-2.60`** | **`-1.95`** | **`-1.50`** | **`-1.20`** | **`+1.80`** | **`+1.40`** | `0.00` | **⚠️ BORDERLINE (EVALUATE)** | **~25.52%** | Moderate negative drift on V14/V12; flags if decision threshold tuned below 0.25. |
| **🤖 Scenario 4: Automated Card-Testing Bot** | `16,200` *(04:30 Early Morning)* | **$1.00** | **`-4.50`** | **`-3.20`** | **`-2.80`** | **`-2.90`** | **`+2.60`** | **`+2.10`** | `0.00` | **🚨 FLAGGED FRAUD (DECLINE)** | **~71.95%** | Severe component anomalies despite nominal $1.00 transaction volume (classic bot probing). |

---

### 4. Run Automated Unit Tests
Verify pipeline components, schema compatibility, math operations, and explainer:
```bash
pytest -v
```

### 5. Run the Production Pipeline & Benchmark
To train, benchmark all candidate models, export high-res charts, and save candidate pipelines:
```bash
python run_pipeline.py
```

### 6. Direct Inference with Python SDK
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
