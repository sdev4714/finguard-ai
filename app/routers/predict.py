import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "app"))

from fastapi import APIRouter, HTTPException
from schemas import LoanApplication, PredictionResponse
from dependencies import run_prediction

router = APIRouter()

@router.post("/predict", response_model=None)
async def predict(application: LoanApplication):
    try:
        input_dict = {
            "age":                 application.age,
            "gender":              application.gender.value,
            "education":           application.education.value,
            "employment_type":     application.employment_type.value,
            "income":              application.income,
            "co_applicant_income": application.co_applicant_income,
            "loan_amount":         application.loan_amount,
            "loan_term":           application.loan_term,
            "credit_score":        application.credit_score,
            "existing_loans":      application.existing_loans,
            "property_area":       application.property_area.value,
            "dependents":          application.dependents,
        }

        result = run_prediction(input_dict)

        return {
            "approved":       result["approved"],
            "risk_score":     result["risk_score"],
            "confidence":     result["confidence"],
            "decision":       result["decision"],
            "threshold_used": result["threshold_used"],
            "top_factors":    result["top_factors"],
            "shap_data":      result["shap_data"],
            "input_summary": {
                "age":            input_dict["age"],
                "income":         input_dict["income"],
                "loan_amount":    input_dict["loan_amount"],
                "credit_score":   input_dict["credit_score"],
                "employment":     input_dict["employment_type"],
                "property_area":  input_dict["property_area"],
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")