import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score,
    recall_score,
    precision_score
)

def find_optimal_threshold(y_true, y_probs, metric='f1', cost_fn=122.0, cost_fp=10.0):
    """
    Finds the optimal classification decision threshold based on either:
    1. Maximum F1-score ('f1')
    2. Minimum Business Cost ('cost'): Cost = (FN * cost_fn) + (FP * cost_fp)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    if metric == 'f1':
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_metric = f1_scores[best_idx]
        return float(best_thresh), float(best_metric), "f1"
    
    elif metric == 'cost':
        costs = []
        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            total_cost = (fn * cost_fn) + (fp * cost_fp)
            costs.append(total_cost)
            
        best_idx = np.argmin(costs)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_cost = costs[best_idx]
        return float(best_thresh), float(best_cost), "cost"
    else:
        raise ValueError(f"Unknown metric: {metric}")

def evaluate_model(y_true, y_probs, threshold=0.5, model_name="Model"):
    """
    Computes a comprehensive performance dictionary for a model given its predicted probabilities.
    """
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    pr_auc = average_precision_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "model_name": model_name,
        "threshold": threshold,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0)
    }

def plot_curves(models_probs, y_true, save_path=None):
    """
    Plots side-by-side PR Curves and ROC Curves for multiple models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (name, probs) in enumerate(models_probs.items()):
        c = colors[idx % len(colors)]
        # PR Curve
        prec, rec, _ = precision_recall_curve(y_true, probs)
        pr_score = average_precision_score(y_true, probs)
        ax1.plot(rec, prec, color=c, lw=2, label=f"{name} (PR-AUC = {pr_score:.4f})")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_score = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=c, lw=2, label=f"{name} (ROC-AUC = {roc_score:.4f})")
        
    ax1.set_title("Precision-Recall Curves (Critical for Imbalanced Data)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Recall", fontsize=12)
    ax1.set_ylabel("Precision", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left', fontsize=10)
    
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax2.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight='bold')
    ax2.set_xlabel("False Positive Rate", fontsize=12)
    ax2.set_ylabel("True Positive Rate", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig

def plot_confusion_matrix_grid(models_results, save_path=None):
    """
    Generates a grid of confusion matrices for compared models.
    """
    n_models = len(models_results)
    cols = min(4, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = np.array(axes).ravel()
        
    for idx, res in enumerate(models_results):
        ax = axes[idx]
        cm = res["confusion_matrix"]
        name = res["model_name"]
        thresh = res["threshold"]
        f1 = res["f1_score"]
        
        annot = [
            [f"TN\n{cm[0,0]:,}", f"FP\n{cm[0,1]:,}"],
            [f"FN\n{cm[1,0]:,}", f"TP\n{cm[1,1]:,}"]
        ]
        
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred Normal", "Pred Fraud"],
                    yticklabels=["True Normal", "True Fraud"], annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(f"{name}\n(Threshold={thresh:.3f} | F1={f1:.3f})", fontsize=12, fontweight='bold')
        
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig
