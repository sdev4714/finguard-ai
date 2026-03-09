import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from preprocess import preprocess_pipeline, get_feature_names

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(model_name="loan_model"):
    path = os.path.join(BASE_DIR, "models", f"{model_name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_threshold():
    path = os.path.join(BASE_DIR, "models", "threshold.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return 0.5

def get_predictions(model, X, threshold):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob

def demographic_parity(y_pred, sensitive_col, df_test):
    """
    Demographic Parity: approval rate should be equal across groups.
    Difference < 0.1 is generally acceptable.
    """
    results = {}
    groups = df_test[sensitive_col].unique()

    for group in groups:
        mask = df_test[sensitive_col] == group
        group_preds = y_pred[mask]
        approval_rate = group_preds.mean()
        results[str(group)] = round(float(approval_rate), 4)

    rates = list(results.values())
    disparity = round(max(rates) - min(rates), 4)
    results['_disparity'] = disparity
    results['_fair'] = disparity < 0.1

    return results

def equalized_odds(y_pred, y_true, sensitive_col, df_test):
    """
    Equalized Odds: TPR and FPR should be equal across groups.
    """
    results = {}
    groups = df_test[sensitive_col].unique()

    for group in groups:
        mask = df_test[sensitive_col] == group
        g_pred = y_pred[mask]
        g_true = np.array(y_true)[mask]

        if len(np.unique(g_true)) < 2:
            continue

        cm = confusion_matrix(g_true, g_pred)
        tn, fp, fn, tp = cm.ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        results[str(group)] = {
            'tpr': round(float(tpr), 4),
            'fpr': round(float(fpr), 4)
        }

    tpr_vals = [v['tpr'] for v in results.values()]
    fpr_vals = [v['fpr'] for v in results.values()]
    results['_tpr_disparity'] = round(max(tpr_vals) - min(tpr_vals), 4)
    results['_fpr_disparity'] = round(max(fpr_vals) - min(fpr_vals), 4)
    results['_fair'] = results['_tpr_disparity'] < 0.1 and results['_fpr_disparity'] < 0.1

    return results

def predictive_parity(y_pred, y_true, sensitive_col, df_test):
    """
    Predictive Parity: precision should be equal across groups.
    """
    results = {}
    groups = df_test[sensitive_col].unique()

    for group in groups:
        mask = df_test[sensitive_col] == group
        g_pred = y_pred[mask]
        g_true = np.array(y_true)[mask]

        f1 = f1_score(g_true, g_pred, zero_division=0)
        acc = accuracy_score(g_true, g_pred)

        results[str(group)] = {
            'f1': round(float(f1), 4),
            'accuracy': round(float(acc), 4)
        }

    return results

def run_fairness_audit(X_test, y_test, df_original, model, threshold):
    sensitive_cols = ['gender', 'education', 'property_area']
    y_pred, y_prob = get_predictions(model, X_test, threshold)
    y_test_arr = np.array(y_test)

    audit_results = {}

    for col in sensitive_cols:
        if col not in df_original.columns:
            continue

        df_test_subset = df_original.iloc[y_test.index].reset_index(drop=True)

        dp = demographic_parity(y_pred, col, df_test_subset)
        eo = equalized_odds(y_pred, y_test_arr, col, df_test_subset)
        pp = predictive_parity(y_pred, y_test_arr, col, df_test_subset)

        audit_results[col] = {
            'demographic_parity': dp,
            'equalized_odds': eo,
            'predictive_parity': pp
        }

        print(f"\n--- Fairness Audit: {col.upper()} ---")
        print(f"  Demographic Parity (approval rates): {dp}")
        print(f"  Equalized Odds disparity → TPR: {eo['_tpr_disparity']} | FPR: {eo['_fpr_disparity']}")
        print(f"  Fair: {'✅' if dp['_fair'] and eo['_fair'] else '⚠️  Bias detected'}")

    return audit_results

def plot_fairness_report(audit_results):
    docs_dir = os.path.join(BASE_DIR, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    for col, results in audit_results.items():
        dp = results['demographic_parity']
        groups = [k for k in dp.keys() if not k.startswith('_')]
        rates = [dp[g] for g in groups]

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['#2ecc71' if r >= 0.4 else '#e74c3c' for r in rates]
        bars = ax.bar(groups, rates, color=colors, edgecolor='black', linewidth=0.7)
        ax.axhline(y=np.mean(rates), color='steelblue', linestyle='--',
                   linewidth=1.5, label=f'Mean: {np.mean(rates):.2f}')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Approval Rate')
        ax.set_title(f'Demographic Parity — {col.capitalize()}')
        ax.legend()

        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{rate:.2%}', ha='center', fontsize=10)

        plt.tight_layout()
        path = os.path.join(docs_dir, f"fairness_{col}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: fairness_{col}.png")

def save_audit(audit_results):
    path = os.path.join(BASE_DIR, "models", "fairness_audit.pkl")
    with open(path, "wb") as f:
        pickle.dump(audit_results, f)
    print("\nSaved: fairness_audit.pkl")

def load_audit():
    path = os.path.join(BASE_DIR, "models", "fairness_audit.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    print("=== Running Fairness Audit ===\n")

    _, X_test, _, y_test, _, _ = preprocess_pipeline(save_artifacts=False)

    df_original = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "loan_data.csv"))

    model = load_model("loan_model")
    threshold = load_threshold()
    print(f"Using threshold: {threshold:.2f}\n")

    audit_results = run_fairness_audit(X_test, y_test, df_original, model, threshold)
    plot_fairness_report(audit_results)
    save_audit(audit_results)

    print("\n=== Fairness Audit Complete ===")