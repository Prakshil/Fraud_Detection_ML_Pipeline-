import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline import FraudDetectionPipeline
from src.evaluation import evaluate_model, find_optimal_threshold, plot_curves, plot_confusion_matrix_grid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "creditcard.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("=" * 75)
    print("      PRODUCTION CREDIT CARD FRAUD DETECTION PIPELINE RUNNER")
    print("=" * 75)
    
    # 1. Load Data
    print(f"\n[1/6] Ingesting dataset from {CSV_PATH}...")
    t0 = time.time()
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df):,} transactions in {time.time()-t0:.2f}s.")
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # 2. Stratified Data Split (70% Train, 10% Validation, 20% Holdout Test)
    print("\n[2/6] Splitting data into Stratified Train (70%), Val (10%), and Test (20%)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    # Further split train_val into 70% train and 10% validation (1/8th of train_val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, stratify=y_train_val, random_state=42
    )
    
    print(f"  - Train Set:      {len(X_train):,} samples | Frauds: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    print(f"  - Validation Set: {len(X_val):,} samples | Frauds: {y_val.sum()} ({y_val.mean()*100:.3f}%)")
    print(f"  - Test Set:       {len(X_test):,} samples | Frauds: {y_test.sum()} ({y_test.mean()*100:.3f}%)")
    print("  * Note: The test set is completely held out and untouched for unbiased assessment.")
    
    # 3. Model Suite Benchmarking
    print("\n[3/6] Training & Benchmarking Candidate Production Pipelines...")
    model_types = ["logreg", "rf", "hgb", "mlp"]
    model_labels = {
        "logreg": "Logistic Regression (Cost-Sensitive)",
        "rf": "Random Forest (Balanced Subsample)",
        "hgb": "HistGradientBoosting (Balanced)",
        "mlp": "Neural Network (MLP Classifier)"
    }
    
    trained_pipelines = {}
    test_probs = {}
    benchmark_rows = []
    evaluation_results = []
    
    for m_type in model_types:
        label = model_labels[m_type]
        print(f"\n  ---> Training {label}...")
        t_start = time.time()
        
        # Build & Fit Pipeline
        pipe = FraudDetectionPipeline(model_type=m_type)
        pipe.fit(X_train, y_train, tune_thresh=True, X_val=X_val, y_val=y_val)
        train_time = time.time() - t_start
        
        # Predict on Holdout Test Set
        probs = pipe.predict_proba(X_test)
        test_probs[label] = probs
        trained_pipelines[m_type] = pipe
        
        # Evaluate with Tuned Threshold
        eval_res = evaluate_model(y_test, probs, threshold=pipe.threshold, model_name=label)
        evaluation_results.append(eval_res)
        
        # Evaluate with Default Threshold (0.5) for comparison
        eval_default = evaluate_model(y_test, probs, threshold=0.5, model_name=label)
        
        # Compute Business Financial Cost: (FN * $122.21) + (FP * $10.00)
        cost_tuned = (eval_res["false_negatives"] * 122.21) + (eval_res["false_positives"] * 10.00)
        
        benchmark_rows.append({
            "Algorithm": label,
            "Train Time (s)": round(train_time, 2),
            "PR-AUC": round(eval_res["pr_auc"], 4),
            "ROC-AUC": round(eval_res["roc_auc"], 4),
            "Tuned Threshold": round(pipe.threshold, 4),
            "Tuned F1-Score": round(eval_res["f1_score"], 4),
            "Recall (Frauds Caught)": f"{eval_res['true_positives']}/98 ({eval_res['recall']*100:.1f}%)",
            "False Alarms (FP)": eval_res["false_positives"],
            "Total Cost ($)": round(cost_tuned, 2),
            "Default Threshold (0.5) F1": round(eval_default["f1_score"], 4),
            "Default (0.5) False Alarms": eval_default["false_positives"]
        })
        
        print(f"       Trained in {train_time:.2f}s | PR-AUC: {eval_res['pr_auc']:.4f} | Tuned F1: {eval_res['f1_score']:.4f}")
        print(f"       Recall: {eval_res['true_positives']}/98 | False Alarms: {eval_res['false_positives']} | Cost: ${cost_tuned:,.2f}")
        
    summary_df = pd.DataFrame(benchmark_rows)
    print("\n" + "=" * 75)
    print("                     BENCHMARK SUMMARY (HOLDOUT TEST SET)")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    
    # Save Benchmark CSV
    csv_out = os.path.join(REPORTS_DIR, "benchmark_summary.csv")
    summary_df.to_csv(csv_out, index=False)
    print(f"\n[4/6] Saved benchmark summary to {csv_out}")
    
    # 4. Generate & Save High-Resolution Visualization Plots
    print("\n[5/6] Generating evaluation visual artifacts...")
    curves_path = os.path.join(REPORTS_DIR, "precision_recall_and_roc_curves.png")
    plot_curves(test_probs, y_test, save_path=curves_path)
    print(f"  - Saved PR & ROC Curves: {curves_path}")
    
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix_grid.png")
    plot_confusion_matrix_grid(evaluation_results, save_path=cm_path)
    print(f"  - Saved Confusion Matrix Grid: {cm_path}")
    
    # 5. Select Winning Model and Serialize to Disk
    best_row = summary_df.sort_values(by="PR-AUC", ascending=False).iloc[0]
    best_algo_label = best_row["Algorithm"]
    best_type = [k for k, v in model_labels.items() if v == best_algo_label][0]
    best_pipeline = trained_pipelines[best_type]
    
    model_save_path = os.path.join(MODELS_DIR, "best_fraud_pipeline.joblib")
    best_pipeline.save(model_save_path)
    print(f"\n[6/6] Winning Model selected: {best_algo_label} (PR-AUC: {best_row['PR-AUC']:.4f})")
    print(f"  - Production pipeline persisted to: {model_save_path}")
    
    # 6. Verification: Reload model and test inference on test sample
    print("\n--- Verifying Serialized Artifact with Reloaded Inference ---")
    reloaded_pipe = FraudDetectionPipeline.load(model_save_path)
    sample_batch = X_test.iloc[:5]
    sample_preds = reloaded_pipe.predict(sample_batch)
    sample_probs = reloaded_pipe.predict_proba(sample_batch)
    print(f"Reload verification successful!")
    print(f"Sample predicted probabilities: {np.round(sample_probs, 4)}")
    print(f"Sample binary predictions:      {sample_preds}")
    print("\n" + "=" * 75)
    print("      PIPELINE EXECUTION AND SERIALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 75)

if __name__ == "__main__":
    main()
