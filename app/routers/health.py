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
        ready     = model is not None
        threshold = get_threshold() if ready else 0.5

        return {
            "status":     "ready" if ready else "warming_up",
            "message":    "Server is ready" if ready else "ML pipeline is running in background, check back in 3-5 minutes",
            "model_type": type(model).__name__ if model else "loading...",
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
            "status":  "warming_up",
            "message": "Pipeline is running, please wait 3-5 minutes",
            "version": "1.0.0"
        }

 