import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from preprocess import preprocess_pipeline, get_feature_names

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(model_name="loan_model"):
    path = os.path.join(BASE_DIR, "models", f"{model_name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def get_explainer(model, X_background):
    model_type = type(model).__name__

    if model_type == 'StackingClassifier':
        # Use the XGBoost from stacking estimators for SHAP
        for name, estimator in model.estimators_:
            if 'xgboost' in name:
                print("Using XGBoost sub-model for SHAP explainer")
                explainer = shap.TreeExplainer(estimator)
                return explainer, 'tree'

    elif model_type == 'XGBClassifier':
        explainer = shap.TreeExplainer(model)
        return explainer, 'tree'

    elif model_type == 'RandomForestClassifier':
        explainer = shap.TreeExplainer(model)
        return explainer, 'tree'

    elif model_type == 'LogisticRegression':
        explainer = shap.LinearExplainer(model, X_background)
        return explainer, 'linear'

    else:
        print("Falling back to KernelExplainer (slower)")
        background = shap.sample(X_background, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        return explainer, 'kernel'

def get_shap_values_single(explainer, X_single, explainer_type):
    shap_vals = explainer.shap_values(X_single)

    # For tree explainers returning list (binary classification)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # class 1 = approved

    # Flatten to 1D
    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]

    return shap_vals

def explain_single(X_single, explainer, explainer_type, feature_names=None):
    if feature_names is None:
        feature_names = get_feature_names()

    shap_vals = get_shap_values_single(explainer, X_single, explainer_type)

    explanation = []
    for fname, fval, sval in zip(feature_names, X_single[0], shap_vals):
        explanation.append({
            'feature': fname,
            'value': round(float(fval), 4),
            'shap_value': round(float(sval), 4),
            'impact': 'positive' if sval > 0 else 'negative'
        })

    explanation.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    return explanation

def shap_to_dict(explanation, top_n=10):
    top = explanation[:top_n]
    return {
        'features': [e['feature'] for e in top],
        'shap_values': [e['shap_value'] for e in top],
        'raw_values': [e['value'] for e in top],
        'impacts': [e['impact'] for e in top]
    }

def plot_shap_bar(explanation, top_n=10, title="SHAP Feature Impact", save_path=None):
    top = explanation[:top_n]
    features = [e['feature'] for e in top]
    values = [e['shap_value'] for e in top]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(features[::-1], values[::-1], color=colors[::-1])
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('SHAP Value (impact on approval probability)')
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP bar chart: {save_path}")
    else:
        plt.show()

    return fig

def plot_shap_summary(explainer, X_test, explainer_type, feature_names=None):
    if feature_names is None:
        feature_names = get_feature_names()

    shap_vals = explainer.shap_values(X_test[:200])

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_vals,
        X_test[:200],
        feature_names=feature_names,
        show=False,
        plot_type='bar'
    )
    plt.tight_layout()
    path = os.path.join(BASE_DIR, "docs", "shap_summary.png")
    os.makedirs(os.path.join(BASE_DIR, "docs"), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary plot: {path}")

def save_explainer(explainer, explainer_type):
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "explainer.pkl"), "wb") as f:
        pickle.dump((explainer, explainer_type), f)
    print("Saved: explainer.pkl")

def load_explainer():
    path = os.path.join(BASE_DIR, "models", "explainer.pkl")
    with open(path, "rb") as f:
        explainer, explainer_type = pickle.load(f)
    return explainer, explainer_type

if __name__ == "__main__":
    print("=== Building SHAP Explainer ===\n")
    X_train, X_test, y_train, y_test, _, _ = preprocess_pipeline(save_artifacts=False)

    model = load_model("loan_model")
    explainer, explainer_type = get_explainer(model, X_train)

    print("\nGenerating SHAP summary plot...")
    plot_shap_summary(explainer, X_test, explainer_type)

    print("\nTesting single explanation...")
    sample = X_test[0:1]
    explanation = explain_single(sample, explainer, explainer_type)

    print("\nTop 5 factors for sample applicant:")
    for e in explanation[:5]:
        direction = "↑ Increases" if e['impact'] == 'positive' else "↓ Decreases"
        print(f"  {direction} approval — {e['feature']}: {e['value']} (SHAP: {e['shap_value']})")

    save_path = os.path.join(BASE_DIR, "docs", "shap_bar_sample.png")
    plot_shap_bar(explanation, save_path=save_path)

    save_explainer(explainer, explainer_type)
    print("\n=== Explainer Ready ===")

# **What this does:**

# - `get_explainer` — auto-detects model type and picks the right SHAP explainer (TreeExplainer for RF/XGB, LinearExplainer for LR, falls back to KernelExplainer for stacking)
# - `explain_single` — returns a ranked list of features with their SHAP values and impact direction for a single applicant — this is what the API will call
# - `shap_to_dict` — converts explanation to a clean dict for JSON response
# - `plot_shap_bar` — generates a green/red horizontal bar chart per applicant — rendered in the UI
# - `plot_shap_summary` — global feature importance across all test samples, saved to `docs/`
# - Saves `explainer.pkl` so the API loads it once at startup

# **Run:**
# ```
# python src/explain.py