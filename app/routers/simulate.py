import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "app"))

from fastapi import APIRouter, HTTPException
from schemas import SimulationRequest
from dependencies import run_prediction

router = APIRouter()

@router.post("/simulate", response_model=None)
async def simulate(request: SimulationRequest):
    try:
        # ── Original application ──────────────────────────────────────────────
        base_dict = {
            "age":                 request.base_application.age,
            "gender":              request.base_application.gender.value,
            "education":           request.base_application.education.value,
            "employment_type":     request.base_application.employment_type.value,
            "income":              request.base_application.income,
            "co_applicant_income": request.base_application.co_applicant_income,
            "loan_amount":         request.base_application.loan_amount,
            "loan_term":           request.base_application.loan_term,
            "credit_score":        request.base_application.credit_score,
            "existing_loans":      request.base_application.existing_loans,
            "property_area":       request.base_application.property_area.value,
            "dependents":          request.base_application.dependents,
        }

        # ── Modified application (apply overrides) ────────────────────────────
        modified_dict = base_dict.copy()
        valid_fields = base_dict.keys()

        for key, value in request.overrides.items():
            if key not in valid_fields:
                raise ValueError(f"Invalid override field: '{key}'. Valid fields: {list(valid_fields)}")
            modified_dict[key] = value

        # ── Run both predictions ──────────────────────────────────────────────
        original_result = run_prediction(base_dict)
        modified_result = run_prediction(modified_dict)

        # ── Compute deltas ────────────────────────────────────────────────────
        risk_delta       = round(modified_result["risk_score"] - original_result["risk_score"], 2)
        confidence_delta = round(modified_result["confidence"] - original_result["confidence"], 4)
        approval_changed = original_result["approved"] != modified_result["approved"]

        return {
            "original": {
                "label":      "Original Application",
                "approved":   original_result["approved"],
                "risk_score": original_result["risk_score"],
                "confidence": original_result["confidence"],
                "decision":   original_result["decision"],
                "shap_data":  original_result["shap_data"],
            },
            "modified": {
                "label":      "Modified Application",
                "approved":   modified_result["approved"],
                "risk_score": modified_result["risk_score"],
                "confidence": modified_result["confidence"],
                "decision":   modified_result["decision"],
                "shap_data":  modified_result["shap_data"],
            },
            "delta": {
                "risk_score":       risk_delta,
                "confidence":       confidence_delta,
                "approval_changed": approval_changed,
                "risk_direction":   "improved" if risk_delta < 0 else "worsened",
                "summary":          _delta_summary(risk_delta, approval_changed, request.overrides)
            },
            "overrides": request.overrides
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


def _delta_summary(risk_delta: float, approval_changed: bool, overrides: dict) -> str:
    changed_fields = ", ".join(overrides.keys())

    if approval_changed and risk_delta < 0:
        return f"Changing {changed_fields} improved risk score by {abs(risk_delta)} points and flipped decision to Approved ✅"
    elif approval_changed and risk_delta > 0:
        return f"Changing {changed_fields} worsened risk score by {abs(risk_delta)} points and flipped decision to Rejected ❌"
    elif risk_delta < 0:
        return f"Changing {changed_fields} improved risk score by {abs(risk_delta)} points. Decision unchanged."
    elif risk_delta > 0:
        return f"Changing {changed_fields} worsened risk score by {abs(risk_delta)} points. Decision unchanged."
    else:
        return f"Changing {changed_fields} had no significant impact on the decision."