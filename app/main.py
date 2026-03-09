import os
import sys
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "app"))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dependencies import set_artifacts
from routers import health, predict, simulate, dashboard        # only import what exists for now

artifacts = {}

def load_artifacts():
    models_dir = os.path.join(BASE_DIR, "models")

    with open(os.path.join(models_dir, "loan_model.pkl"), "rb") as f:
        artifacts["model"] = pickle.load(f)
    with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
        artifacts["scaler"] = pickle.load(f)
    with open(os.path.join(models_dir, "label_encoders.pkl"), "rb") as f:
        artifacts["encoders"] = pickle.load(f)
    with open(os.path.join(models_dir, "explainer.pkl"), "rb") as f:
        artifacts["explainer"], artifacts["explainer_type"] = pickle.load(f)

    threshold_path = os.path.join(models_dir, "threshold.pkl")
    if os.path.exists(threshold_path):
        with open(threshold_path, "rb") as f:
            artifacts["threshold"] = pickle.load(f)
    else:
        artifacts["threshold"] = 0.5

    fairness_path = os.path.join(models_dir, "fairness_audit.pkl")
    if os.path.exists(fairness_path):
        with open(fairness_path, "rb") as f:
            artifacts["fairness"] = pickle.load(f)
    else:
        artifacts["fairness"] = {}

    print("✅ All artifacts loaded successfully")
    print(f"   Model    : {type(artifacts['model']).__name__}")
    print(f"   Threshold: {artifacts['threshold']:.2f}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    set_artifacts(artifacts)
    yield
    artifacts.clear()
    print("Artifacts cleared on shutdown")

app = FastAPI(
    title="Loan Approval Prediction System",
    description="ML-powered loan approval with SHAP explainability, what-if simulation, and fairness auditing.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir   = os.path.join(BASE_DIR, "app", "static")
template_dir = os.path.join(BASE_DIR, "app", "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=template_dir)

app.include_router(health.router,  prefix="/api", tags=["Health"])
app.include_router(predict.router, prefix="/api", tags=["Predict"])
app.include_router(simulate.router, prefix="/api", tags=["Simulate"])
app.include_router(dashboard.router, prefix="/api", tags=["Dashboard"])

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/result")
async def result(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})

@app.get("/simulator")
async def simulator(request: Request):
    return templates.TemplateResponse("simulator.html", {"request": request})

@app.get("/dashboard")
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
