import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "app"))        # ← this was missing

from fastapi import APIRouter
from dependencies import get_artifacts, get_threshold

router = APIRouter()

@router.get("/health", response_model=None)
async def health_check():
    try:
        artifacts = get_artifacts()
        model     = artifacts.get("model")
        threshold = get_threshold()

        return {
            "status":     "healthy",
            "model_type": type(model).__name__ if model else "unknown",
            "threshold":  round(threshold, 2),
            "version":    "1.0.0",
            "artifacts":  {
                "model":     "model"     in artifacts,
                "scaler":    "scaler"    in artifacts,
                "encoders":  "encoders"  in artifacts,
                "explainer": "explainer" in artifacts,
                "threshold": "threshold" in artifacts,
                "fairness":  "fairness"  in artifacts,
            }
        }

    except Exception as e:
        return {
            "status":  "unhealthy",
            "error":   str(e),
            "version": "1.0.0"
        }