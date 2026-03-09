import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "app"))

from fastapi import APIRouter, HTTPException
from dependencies import get_artifacts, get_model, get_scaler, get_encoders, get_threshold, get_fairness
from preprocess import preprocess_pipeline, get_feature_names

router = APIRouter()

def get_test_predictions(model, threshold):
    try:
        _, X_test, _, y_test, _, _ = preprocess_pipeline(save_artifacts=False)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        return X_test, y_test, y_pred, y_prob
    except Exception as e:
        raise RuntimeError(f"Could not load test data: {str(e)}")

def compute_model_metrics(y_test, y_pred, y_prob):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    y_test_arr = np.array(y_test)
    return {
        "accuracy":  round(float(accuracy_score(y_test_arr, y_pred)), 4),
        "f1":        round(float(f1_score(y_test_arr, y_pred)), 4),
        "precision": round(float(precision_score(y_test_arr, y_pred)), 4),
        "recall":    round(float(recall_score(y_test_arr, y_pred)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test_arr, y_prob)), 4),
    }

def compute_risk_distribution(y_prob):
    risk_scores = ((1 - y_prob) * 100).tolist()
    buckets = {
        "very_low_risk (0-20)":   int(np.sum((np.array(risk_scores) >= 0)  & (np.array(risk_scores) < 20))),
        "low_risk (20-40)":       int(np.sum((np.array(risk_scores) >= 20) & (np.array(risk_scores) < 40))),
        "medium_risk (40-60)":    int(np.sum((np.array(risk_scores) >= 40) & (np.array(risk_scores) < 60))),
        "high_risk (60-80)":      int(np.sum((np.array(risk_scores) >= 60) & (np.array(risk_scores) < 80))),
        "very_high_risk (80-100)":int(np.sum((np.array(risk_scores) >= 80) & (np.array(risk_scores) <= 100))),
    }
    return {
        "distribution": buckets,
        "avg_risk":     round(float(np.mean(risk_scores)), 2),
        "median_risk":  round(float(np.median(risk_scores)), 2),
    }

def format_fairness(fairness_audit):
    summary = []
    for attribute, results in fairness_audit.items():
        dp = results.get("demographic_parity", {})
        eo = results.get("equalized_odds", {})

        groups = {k: v for k, v in dp.items() if not k.startswith("_")}
        is_fair = dp.get("_fair", True) and eo.get("_fair", True)

        summary.append({
            "attribute":          attribute,
            "approval_rates":     groups,
            "disparity":          dp.get("_disparity", 0),
            "tpr_disparity":      eo.get("_tpr_disparity", 0),
            "fpr_disparity":      eo.get("_fpr_disparity", 0),
            "is_fair":            is_fair,
            "verdict":            "✅ Fair" if is_fair else "⚠️ Bias Detected"
        })

    return summary

@router.get("/dashboard", response_model=None)
async def get_dashboard():
    try:
        model     = get_model()
        threshold = get_threshold()
        fairness  = get_fairness()
        artifacts = get_artifacts()

        X_test, y_test, y_pred, y_prob = get_test_predictions(model, threshold)

        total          = len(y_pred)
        total_approved = int(np.sum(y_pred))
        total_rejected = total - total_approved
        approval_rate  = round(float(total_approved / total), 4)

        metrics          = compute_model_metrics(y_test, y_pred, y_prob)
        risk_distribution = compute_risk_distribution(y_prob)
        fairness_summary  = format_fairness(fairness)

        return {
            "overview": {
                "total_applications": total,
                "total_approved":     total_approved,
                "total_rejected":     total_rejected,
                "approval_rate":      approval_rate,
                "avg_risk_score":     risk_distribution["avg_risk"],
                "median_risk_score":  risk_distribution["median_risk"],
            },
            "model_info": {
                "model_type":  type(model).__name__,
                "threshold":   round(threshold, 2),
                "version":     "1.0.0",
                "features":    get_feature_names(),
                "n_features":  len(get_feature_names()),
            },
            "metrics":            metrics,
            "risk_distribution":  risk_distribution,
            "fairness_summary":   fairness_summary,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard failed: {str(e)}")


@router.get("/dashboard/fairness", response_model=None)
async def get_fairness_report():
    try:
        fairness = get_fairness()
        if not fairness:
            return {"message": "No fairness audit found. Run src/fairness.py first.", "data": {}}
        return {"data": format_fairness(fairness)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fairness report failed: {str(e)}")


@router.get("/dashboard/metrics", response_model=None)
async def get_metrics():
    try:
        model     = get_model()
        threshold = get_threshold()
        _, y_test, y_pred, y_prob = get_test_predictions(model, threshold)
        metrics = compute_model_metrics(y_test, y_pred, y_prob)
        return {"metrics": metrics, "threshold": round(threshold, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics failed: {str(e)}")


@router.get("/dashboard/risk", response_model=None)
async def get_risk_distribution():
    try:
        model     = get_model()
        threshold = get_threshold()
        _, _, _, y_prob = get_test_predictions(model, threshold)
        return compute_risk_distribution(y_prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk distribution failed: {str(e)}")