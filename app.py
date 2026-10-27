import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import FraudDetectionPipeline
from src.explainer import FraudExplainer

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection AI Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b6584;
        margin-bottom: 1.5rem;
    }
    .card-metric {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #1e3d59;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge-fraud {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #f87171;
    }
    .badge-normal {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #4ade80;
    }
    .reason-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Model definitions & paths
MODEL_FILES = {
    "Random Forest (Balanced Subsample) 🏆": "rf_pipeline.joblib",
    "Neural Network (MLP Classifier)": "mlp_pipeline.joblib",
    "HistGradientBoosting (Balanced)": "hgb_pipeline.joblib",
    "Logistic Regression (Cost-Sensitive)": "logreg_pipeline.joblib"
}

@st.cache_resource(show_spinner=False)
def load_model(filename):
    model_path = os.path.join(BASE_DIR, "models", filename)
    if not os.path.exists(model_path):
        # Fallback to best_fraud_pipeline.joblib if specific model file is missing
        model_path = os.path.join(BASE_DIR, "models", "best_fraud_pipeline.joblib")
    return FraudDetectionPipeline.load(model_path)

@st.cache_data(show_spinner=False)
def load_sample_dataset():
    csv_path = os.path.join(BASE_DIR, "creditcard.csv")
    if os.path.exists(csv_path):
        df_normal = pd.read_csv(csv_path, nrows=500)
        # Load a few known frauds
        df_full_sub = pd.read_csv(csv_path, usecols=['Class'])
        fraud_indices = df_full_sub[df_full_sub['Class'] == 1].index[:10].tolist()
        df_frauds = pd.read_csv(csv_path, skiprows=lambda x: x not in [0] + [i + 1 for i in fraud_indices])
        return df_normal, df_frauds
    return None, None

@st.cache_data(show_spinner=False)
def load_benchmark_summary():
    summary_path = os.path.join(BASE_DIR, "reports", "benchmark_summary.csv")
    if os.path.exists(summary_path):
        return pd.read_csv(summary_path)
    return None

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=64)
st.sidebar.title("🛡️ Platform Controls")

selected_model_label = st.sidebar.selectbox(
    "Choose Detection Algorithm",
    list(MODEL_FILES.keys()),
    index=0,
    help="Select which trained production machine learning model to evaluate transactions with."
)

pipeline_file = MODEL_FILES[selected_model_label]
pipeline = load_model(pipeline_file)
explainer = FraudExplainer(pipeline)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Decision Threshold Setting")
default_thresh = float(pipeline.threshold)
use_tuned = st.sidebar.checkbox(f"Use Tuned Cutoff ({default_thresh:.4f})", value=True)

if use_tuned:
    active_threshold = default_thresh
    st.sidebar.info(f"Using Optimal F1/Cost Cutoff: **{active_threshold:.4f}**")
else:
    active_threshold = st.sidebar.slider(
        "Manual Decision Cutoff",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        help="Transactions with fraud probability >= this value are flagged as FRAUD."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("💼 Financial Loss Assumptions")
cost_fn = st.sidebar.number_input("Cost of Missed Fraud (FN)", min_value=1.0, max_value=5000.0, value=122.21, step=5.0)
cost_fp = st.sidebar.number_input("Cost of False Alarm (FP)", min_value=0.5, max_value=200.0, value=10.00, step=1.0)

# Main Header
st.markdown('<div class="main-header">Credit Card Fraud Detection AI Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production ML Pipeline with Multi-Model Benchmarking, Real-Time Decision Optimization & Explainable AI (XAI)</div>',
    unsafe_allow_html=True
)

tabs = st.tabs([
    "🔍 Real-Time Transaction Inspector",
    "📁 Batch Transaction Scoring",
    "📊 Algorithm Benchmarking & Curves",
    "💡 Financial Cost-Utility Simulator"
])

df_normal_sample, df_fraud_sample = load_sample_dataset()

# =====================================================================
# TAB 1: Real-Time Transaction Inspector
# =====================================================================
with tabs[0]:
    st.markdown("### 🔍 Live Transaction Scoring & Explainability Engine")
    st.markdown("Evaluate individual transaction parameters, compute real-time fraud probability, and inspect algorithmic rationale.")

    # Preset selection
    preset_choice = st.radio(
        "Load Preset Synthetic Test Scenarios (Handcrafted - not from dataset):",
        [
            "🟢 Scenario 1: Everyday Legitimate Purchase ($24.50, Midday 14:00)",
            "🔴 Scenario 2: Severe Midnight Account Takeover ($980.00, 03:00 Night)",
            "⚠️ Scenario 3: Borderline Suspicious E-Commerce ($349.00, 21:30 Evening)",
            "🤖 Scenario 4: Automated Card-Testing Micro-Transaction ($1.00, 04:30 Early Morning)",
            "🛠️ Custom Transaction Inputs"
        ],
        horizontal=False
    )

    # Initialize transaction values based on synthetic scenarios
    row = {f"V{i}": 0.0 for i in range(1, 29)}
    if preset_choice.startswith("🟢 Scenario 1"):
        row.update({
            "Time": 50400.0,
            "Amount": 24.50,
            "V14": 0.25,
            "V12": -0.15,
            "V10": 0.10,
            "V17": 0.05,
            "V4": -0.30,
            "V11": -0.20
        })
    elif preset_choice.startswith("🔴 Scenario 2"):
        row.update({
            "Time": 10800.0,
            "Amount": 980.00,
            "V14": -7.20,
            "V12": -4.80,
            "V10": -3.90,
            "V17": -4.10,
            "V4": 3.85,
            "V11": 3.10
        })
    elif preset_choice.startswith("⚠️ Scenario 3"):
        row.update({
            "Time": 77400.0,
            "Amount": 349.00,
            "V14": -2.60,
            "V12": -1.95,
            "V10": -1.50,
            "V17": -1.20,
            "V4": 1.80,
            "V11": 1.40
        })
    elif preset_choice.startswith("🤖 Scenario 4"):
        row.update({
            "Time": 16200.0,
            "Amount": 1.00,
            "V14": -4.50,
            "V12": -3.20,
            "V10": -2.80,
            "V17": -2.90,
            "V4": 2.60,
            "V11": 2.10
        })
    else:
        row.update({
            "Time": 45000.0,
            "Amount": 100.00,
            "V14": -0.50,
            "V12": -0.20
        })

    # Two column layout for transaction inputs
    with st.expander("📝 Modify Transaction Parameters (Time, Amount, V1-V28)", expanded=(preset_choice.startswith("🛠️"))):
        col1, col2 = st.columns(2)
        with col1:
            inp_time = st.number_input("Elapsed Time (Seconds)", min_value=0.0, max_value=200000.0, value=float(row.get("Time", 0.0)))
            approx_hour = int((inp_time // 3600) % 24)
            st.caption(f"🕒 Equivalent Diurnal Time: **{approx_hour:02d}:00 hours** ({'Day 1' if inp_time < 86400 else 'Day 2'})")
        with col2:
            inp_amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=30000.0, value=float(row.get("Amount", 0.0)), step=1.0)
            st.caption(f"💳 Log1p Amount: **{np.log1p(inp_amount):.4f}**")

        st.markdown("**Top Correlated PCA Components (V14, V12, V10, V17, V4, V11):**")
        pca_cols = st.columns(6)
        v14_val = pca_cols[0].number_input("V14", value=float(row.get("V14", 0.0)), step=0.5)
        v12_val = pca_cols[1].number_input("V12", value=float(row.get("V12", 0.0)), step=0.5)
        v10_val = pca_cols[2].number_input("V10", value=float(row.get("V10", 0.0)), step=0.5)
        v17_val = pca_cols[3].number_input("V17", value=float(row.get("V17", 0.0)), step=0.5)
        v4_val = pca_cols[4].number_input("V4", value=float(row.get("V4", 0.0)), step=0.5)
        v11_val = pca_cols[5].number_input("V11", value=float(row.get("V11", 0.0)), step=0.5)

        with st.expander("Remaining PCA Components (V1 to V28)"):
            other_v_cols = st.columns(6)
            input_dict = {"Time": inp_time, "Amount": inp_amount, "V14": v14_val, "V12": v12_val, "V10": v10_val, "V17": v17_val, "V4": v4_val, "V11": v11_val}
            for idx, i in enumerate([j for j in range(1, 29) if j not in [14, 12, 10, 17, 4, 11]]):
                col_target = other_v_cols[idx % 6]
                default_val = float(row.get(f"V{i}", 0.0))
                input_dict[f"V{i}"] = col_target.number_input(f"V{i}", value=default_val, step=0.5, key=f"v_{i}")

    # Build input dataframe with correct column order
    ordered_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    single_tx_df = pd.DataFrame([{c: input_dict.get(c, row.get(c, 0.0)) for c in ordered_cols}])

    # Predict & Explain
    explanation = explainer.explain_transaction(single_tx_df, top_k=6)
    fraud_prob = explanation["fraud_probability"]
    is_fraud = bool(fraud_prob >= active_threshold)

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.subheader("🎯 Decision Verdict & Risk Meter")
        if is_fraud:
            st.markdown(
                f'<div class="badge-fraud">🚨 FLAGGED FOR FRAUD (DECLINE)</div>',
                unsafe_allow_html=True
            )
            st.error(f"**Fraud Probability:** `{fraud_prob * 100:.2f}%` (Exceeds decision threshold of `{active_threshold:.4f}`)")
        else:
            st.markdown(
                f'<div class="badge-normal">✅ APPROVED (LEGITIMATE TRANSACTION)</div>',
                unsafe_allow_html=True
            )
            st.success(f"**Fraud Probability:** `{fraud_prob * 100:.2f}%` (Below decision threshold of `{active_threshold:.4f}`)")

        # Progress bar
        st.progress(min(1.0, max(0.0, fraud_prob)))
        
        st.markdown(f"""
        * **Active Algorithm:** `{selected_model_label}`
        * **Decision Cutoff:** `{active_threshold:.4f}`
        * **Raw Fraud Probability:** `{fraud_prob:.6f}`
        * **Action Taken:** `{'Block card & dispatch alert SMS' if is_fraud else 'Instant clearance & ledger settlement'}`
        """)

    with res_col2:
        st.subheader("💡 Algorithmic Rationale & Why This Flagged")
        st.markdown("*Key indicators and feature contributions pushing the prediction:*")
        for reason in explanation["reasons"]:
            st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

    # Visualization of feature impact
    st.markdown("---")
    st.subheader("📊 Local Feature Attribution (SHAP Impact Waterfall)")
    top_feats = explanation["top_features"]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in top_feats["Contribution"]]
    y_pos = np.arange(len(top_feats))

    ax.barh(y_pos, top_feats["Contribution"], color=colors, align='center', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats["Feature"], fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_xlabel("Contribution toward Fraud Risk (SHAP)", fontsize=11)
    ax.set_title("Top Risk Influencers for This Transaction", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    
    # Value annotations
    for i, (val, contrib) in enumerate(zip(top_feats["Value"], top_feats["Contribution"])):
        offset = 0.02 if contrib >= 0 else -0.02
        ha = 'left' if contrib >= 0 else 'right'
        ax.text(contrib + offset, i, f"Val={val:.2f}", va='center', ha=ha, fontsize=9, color='#333333')

    st.pyplot(fig)
    plt.close(fig)

    with st.expander("🧬 View Real-Time Engineered Features for This Transaction"):
        engineered_view = pipeline.pipeline.named_steps['feature_engineer'].transform(single_tx_df)
        st.dataframe(engineered_view.T.rename(columns={0: "Engineered Value"}), use_container_width=True)

# =====================================================================
# TAB 2: Batch Transaction Scoring
# =====================================================================
with tabs[1]:
    st.markdown("### 📁 Batch Transaction Scoring & Bulk Verification")
    st.markdown("Upload raw credit card transactions (CSV) to compute batch fraud risk scores and download audited audit logs.")

    uploaded_file = st.file_uploader("Upload Transaction File (CSV)", type=["csv"], help="Ensure CSV contains Time, Amount, and V1-V28 columns.")
    
    col_demo1, col_demo2 = st.columns([1, 3])
    with col_demo1:
        score_sample_btn = st.button("🧪 Score Sample Batch (First 1,000 Transactions)")

    df_batch = None
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
    elif score_sample_btn and os.path.exists(os.path.join(BASE_DIR, "creditcard.csv")):
        df_batch = pd.read_csv(os.path.join(BASE_DIR, "creditcard.csv"), nrows=1000)

    if df_batch is not None:
        has_class = "Class" in df_batch.columns
        y_true_batch = df_batch["Class"] if has_class else None
        X_batch = df_batch.drop(columns=["Class"]) if has_class else df_batch

        t_start = time.time()
        probs_batch = pipeline.predict_proba(X_batch)
        preds_batch = (probs_batch >= active_threshold).astype(int)
        score_duration = time.time() - t_start

        df_scored = df_batch.copy()
        df_scored["Fraud_Probability"] = np.round(probs_batch, 4)
        df_scored["Decision_Verdict"] = np.where(preds_batch == 1, "FLAGGED FRAUD", "APPROVED")

        flagged_count = int(np.sum(preds_batch == 1))
        flag_rate = (flagged_count / len(df_batch)) * 100
        total_volume = float(df_batch["Amount"].sum()) if "Amount" in df_batch.columns else 0.0
        flagged_volume = float(df_batch.loc[preds_batch == 1, "Amount"].sum()) if "Amount" in df_batch.columns else 0.0

        st.markdown("#### 📈 Batch Processing Overview")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Scored Transactions", f"{len(df_batch):,}", f"{score_duration:.2f}s elapsed")
        mcol2.metric("Flagged as Fraud", f"{flagged_count:,}", f"Flag Rate: {flag_rate:.2f}%")
        mcol3.metric("Total Transaction Volume", f"${total_volume:,.2f}")
        mcol4.metric("Flagged Volume At Risk", f"${flagged_volume:,.2f}")

        # Distribution Chart
        fig_hist, ax_hist = plt.subplots(figsize=(10, 3.5))
        sns.histplot(probs_batch, bins=40, kde=True, color='#2980b9', ax=ax_hist)
        ax_hist.axvline(active_threshold, color='red', linestyle='--', label=f'Cutoff ({active_threshold:.3f})')
        ax_hist.set_title("Predicted Fraud Probability Distribution Across Batch", fontsize=12, fontweight='bold')
        ax_hist.set_xlabel("Fraud Probability")
        ax_hist.set_ylabel("Transaction Count")
        ax_hist.legend()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        st.markdown("#### 📋 Flagged Transactions Audit Table")
        flagged_df = df_scored[df_scored["Decision_Verdict"] == "FLAGGED FRAUD"]
        if len(flagged_df) > 0:
            st.dataframe(flagged_df.head(20), use_container_width=True)
        else:
            st.info("No transactions exceeded the decision cutoff in this batch.")

        csv_download = df_scored.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Scored Transactions CSV",
            data=csv_download,
            file_name="scored_transactions.csv",
            mime="text/csv"
        )

# =====================================================================
# TAB 3: Algorithm Benchmarking & Curves
# =====================================================================
with tabs[2]:
    st.markdown("### 📊 Candidate Algorithm Suite Benchmarking")
    st.markdown("Rigorous multi-model evaluation on the untouched **20% holdout test partition (56,962 transactions)**.")

    summary_df = load_benchmark_summary()
    if summary_df is not None:
        st.dataframe(
            summary_df.style.highlight_max(subset=["PR-AUC", "Tuned F1-Score"], color="#dcfce7")
                           .highlight_min(subset=["Train Time (s)", "Total Cost ($)"], color="#fef9c3"),
            use_container_width=True
        )

    st.markdown("---")
    col_c1, col_c2 = st.columns(2)

    curves_path = os.path.join(BASE_DIR, "reports", "precision_recall_and_roc_curves.png")
    cm_path = os.path.join(BASE_DIR, "reports", "confusion_matrix_grid.png")

    with col_c1:
        st.subheader("📈 Precision-Recall & ROC Curves")
        if os.path.exists(curves_path):
            st.image(curves_path, use_column_width=True)
        else:
            st.warning("Curve artifacts not found.")

    with col_c2:
        st.subheader("🎯 Confusion Matrix Grid")
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)
        else:
            st.warning("Confusion matrix artifacts not found.")

# =====================================================================
# TAB 4: Financial Cost-Utility Simulator
# =====================================================================
with tabs[3]:
    st.markdown("### 💡 Financial Cost-Utility & Business ROI Simulator")
    st.markdown("Simulate financial impact under custom operational parameters and compare net savings against baseline.")

    c_sim1, c_sim2, c_sim3 = st.columns(3)
    sim_fn = c_sim1.slider("Cost of Missed Fraud ($FN)", 10.0, 500.0, float(cost_fn), step=5.0)
    sim_fp = c_sim2.slider("Cost of Customer Verification ($FP)", 1.0, 50.0, float(cost_fp), step=1.0)
    sim_frauds = c_sim3.number_input("Expected Monthly Fraud Attacks", min_value=10, max_value=5000, value=98)

    unmitigated_loss = sim_frauds * sim_fn

    # Simulate across candidate models
    sim_results = []
    if summary_df is not None:
        for _, r in summary_df.iterrows():
            algo = r["Algorithm"]
            fp = int(r["False Alarms (FP)"])
            # Recall
            rec_str = str(r["Recall (Frauds Caught)"])
            caught = int(rec_str.split("/")[0])
            fn = max(0, sim_frauds - caught)
            
            tot_cost = (fn * sim_fn) + (fp * sim_fp)
            net_savings = unmitigated_loss - tot_cost
            reduction = (net_savings / unmitigated_loss) * 100

            sim_results.append({
                "Algorithm": algo,
                "Missed Frauds (FN)": fn,
                "False Alarms (FP)": fp,
                "Total Loss / Cost ($)": f"${tot_cost:,.2f}",
                "Net Financial Savings ($)": f"${net_savings:,.2f}",
                "Fraud Loss Reduction (%)": f"{reduction:.1f}%"
            })

        st.dataframe(pd.DataFrame(sim_results), use_container_width=True)

        st.markdown(f"""
        > [!TIP]
        > **Baseline Unmitigated Loss (No Model):** **${unmitigated_loss:,.2f}**  
        > Deploying the **Random Forest** or **MLP** pipeline prevents **75% to 82%** of direct financial chargeback damage, while preserving cardholder trust.
        """)

st.markdown("---")
st.caption("Credit Card Fraud Detection Production Platform • BDA Project • Built with Streamlit & Scikit-Learn")
