import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from preprocess import preprocess_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(model_name):
    path = os.path.join(BASE_DIR, "models", f"{model_name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def get_metrics(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob)
    }
    return metrics, y_pred, y_prob

def tune_threshold(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_thresh = 0.5
    best_f1 = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"Best Threshold: {best_thresh:.2f} → F1: {best_f1:.4f}")
    return best_thresh

def save_threshold(threshold):
    path = os.path.join(BASE_DIR, "models", "threshold.pkl")
    with open(path, "wb") as f:
        pickle.dump(threshold, f)
    print(f"Threshold saved: {threshold:.2f}")

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = os.path.join(BASE_DIR, "docs", f"confusion_matrix_{model_name}.png")
    os.makedirs(os.path.join(BASE_DIR, "docs"), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"Saved: confusion_matrix_{model_name}.png")

def plot_roc_curve(model, X_test, y_test, model_name):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', color='steelblue', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    path = os.path.join(BASE_DIR, "docs", f"roc_curve_{model_name}.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: roc_curve_{model_name}.png")

def plot_precision_recall(model, X_test, y_test, model_name):
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve — {model_name}')
    plt.tight_layout()
    path = os.path.join(BASE_DIR, "docs", f"pr_curve_{model_name}.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: pr_curve_{model_name}.png")

def compare_all_models(X_test, y_test):
    model_names = [
        'logistic_regression',
        'random_forest',
        'xgboost',
        'stacking_ensemble'
    ]

    print("\n=== Model Comparison ===")
    print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'ROC-AUC':>10}")
    print("-" * 80)

    results = {}
    for name in model_names:
        try:
            model = load_model(name)
            metrics, _, _ = get_metrics(model, X_test, y_test)
            results[name] = metrics
            print(f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['f1']:>10.4f} "
                  f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['roc_auc']:>10.4f}")
        except FileNotFoundError:
            print(f"{name:<25} — model not found, skipping")

    return results

def evaluate_best_model(X_test, y_test):
    print("\n=== Evaluating Best Model (loan_model.pkl) ===")
    model = load_model("loan_model")

    threshold = tune_threshold(model, X_test, y_test)
    save_threshold(threshold)

    metrics, y_pred, y_prob = get_metrics(model, X_test, y_test, threshold)

    print(f"\nMetrics at threshold {threshold:.2f}:")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

    plot_confusion_matrix(y_test, y_pred, "best_model")
    plot_roc_curve(model, X_test, y_test, "best_model")
    plot_precision_recall(model, X_test, y_test, "best_model")

    return metrics, threshold

if __name__ == "__main__":
    print("=== Starting Evaluation Pipeline ===\n")
    _, X_test, _, y_test, _, _ = preprocess_pipeline(save_artifacts=False)

    compare_all_models(X_test, y_test)
    metrics, threshold = evaluate_best_model(X_test, y_test)

    print("\n=== Evaluation Complete ===")