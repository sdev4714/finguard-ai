import os
import sys
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from fastapi import HTTPException

# ── Shared artifact store (populated by main.py lifespan) ────────────────────

_artifacts = {}

def set_artifacts(data: dict):
    """Called once by main.py lifespan on startup."""
    _artifacts.update(data)

def get_artifacts() -> dict:
    if not _artifacts or "model" not in _artifacts:
        raise HTTPException(
            status_code=503,
            detail="Model is still warming up. Please check /api/health and try again in a few minutes."
        )
    return _artifacts

# ── Individual dependency getters ─────────────────────────────────────────────

def get_model():
    artifacts = get_artifacts()
    if "model" not in artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return artifacts["model"]

def get_scaler():
    artifacts = get_artifacts()
    if "scaler" not in artifacts:
        raise HTTPException(status_code=503, detail="Scaler not loaded.")
    return artifacts["scaler"]

def get_encoders():
    artifacts = get_artifacts()
    if "encoders" not in artifacts:
        raise HTTPException(status_code=503, detail="Label encoders not loaded.")
    return artifacts["encoders"]

def get_explainer():
    artifacts = get_artifacts()
    if "explainer" not in artifacts:
        raise HTTPException(status_code=503, detail="Explainer not loaded.")
    return artifacts["explainer"], artifacts["explainer_type"]

def get_threshold() -> float:
    artifacts = get_artifacts()
    return artifacts.get("threshold", 0.5)

def get_fairness() -> dict:
    artifacts = get_artifacts()
    return artifacts.get("fairness", {})

# ── Prediction helper (used by both predict & simulate routers) ───────────────

def run_prediction(input_dict: dict) -> dict:
    from preprocess import preprocess_single
    from explain import explain_single, shap_to_dict

    model     = get_model()
    scaler    = get_scaler()
    encoders  = get_encoders()
    threshold = get_threshold()
    explainer, explainer_type = get_explainer()

    # Preprocess
    X = preprocess_single(input_dict, scaler, encoders)

    # Predict
    prob       = float(model.predict_proba(X)[:, 1][0])
    approved   = bool(prob >= threshold)              # ← native bool
    risk_score = round((1 - prob) * 100, 2)

    # SHAP explanation
    explanation = explain_single(X, explainer, explainer_type)

    # Convert all SHAP values to native Python types
    clean_explanation = [
        {
            'feature':    str(e['feature']),
            'value':      float(e['value']),
            'shap_value': float(e['shap_value']),
            'impact':     str(e['impact'])
        }
        for e in explanation
    ]

    shap_data = shap_to_dict(clean_explanation, top_n=10)

    # Ensure shap_data values are all native types
    clean_shap = {
        'features':    [str(f)   for f in shap_data['features']],
        'shap_values': [float(v) for v in shap_data['shap_values']],
        'raw_values':  [float(v) for v in shap_data['raw_values']],
        'impacts':     [str(i)   for i in shap_data['impacts']],
    }

    return {
        "approved":       bool(approved),
        "risk_score":     float(risk_score),
        "confidence":     float(round(prob, 4)),
        "decision":       "Approved ✅" if approved else "Rejected ❌",
        "threshold_used": float(round(threshold, 2)),
        "top_factors":    clean_explanation[:10],
        "shap_data":      clean_shap,
    }