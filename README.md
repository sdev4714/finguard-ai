# FinGuard AI — Loan Approval Prediction System

> ML-powered loan approval engine with SHAP explainability, What-If simulation, fairness auditing, and a real-time dashboard. Deployed on a custom domain via Render + Cloudflare.

🌐 **Live Demo:** [loan.devrishi.tech](https://loan.devrishi.tech)  
📦 **API Docs:** [loan.devrishi.tech/docs](https://loan.devrishi.tech/docs)

---

## Overview

FinGuard AI is a production-grade machine learning system that predicts loan approval decisions in real time. Unlike typical ML projects that stop at model accuracy, this system is built end-to-end — from data generation and model training to a FastAPI backend, interactive frontend, and live deployment.

Key differentiators:
- Every prediction is **explained** using SHAP values — showing exactly which factors drove the decision
- A **What-If Simulator** lets users tweak application parameters and see how changes impact the outcome
- A **Fairness Audit** checks for demographic bias across gender, education, and property area
- A **Live Dashboard** shows model performance metrics, risk distribution, and fairness verdicts in real time

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | Scikit-learn, XGBoost, SHAP, imbalanced-learn |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | HTML/CSS/JS, Chart.js, Jinja2 |
| Deployment | Render, Cloudflare, GitHub Actions |
| Language | Python 3.11 |

---

## Architecture

```
Data Generation → Preprocessing (SMOTE) → Model Training (4 Models)
       ↓                                          ↓
  loan_data.csv                          loan_model.pkl
                                               ↓
User Form (index.html) → POST /api/predict → FastAPI → preprocess_single()
                                                      → model.predict_proba()
                                                      → SHAP explainer
                                                      → JSON Response
                                               ↓
                                        result.html (decision + SHAP bars + gauge)
```

---

## ML Pipeline

### Models Trained
| Model | Description |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosted trees |
| Stacking Ensemble | Meta-learner on top of all 3 base models |

The best model by CV F1 score is automatically selected and saved as `loan_model.pkl`.

### Features (18 total)
**Raw:** age, gender, education, employment type, income, co-applicant income, loan amount, loan term, credit score, existing loans, property area, dependents

**Engineered:** debt-to-income ratio, total income, income per dependent, loan-to-income ratio, EMI, EMI-to-income ratio

### Key Techniques
- **SMOTE** — handles class imbalance by oversampling minority class
- **Threshold tuning** — optimizes decision boundary for F1 instead of defaulting to 0.5
- **SHAP TreeExplainer** — generates per-prediction feature importance
- **Fairness audit** — demographic parity, equalized odds, predictive parity across sensitive attributes

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Predict loan approval with SHAP explanation |
| `POST` | `/api/simulate` | What-If simulation with side-by-side comparison |
| `GET` | `/api/dashboard` | Full dashboard data — metrics, risk distribution, fairness |
| `GET` | `/api/dashboard/fairness` | Fairness audit results only |
| `GET` | `/api/dashboard/metrics` | Model performance metrics only |
| `GET` | `/api/health` | Health check + artifact status |

Full interactive API documentation available at `/docs` (Swagger UI).

### Example Request

```bash
curl -X POST https://loan.devrishi.tech/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 32,
    "gender": "Male",
    "education": "Graduate",
    "employment_type": "Salaried",
    "income": 75000,
    "co_applicant_income": 25000,
    "loan_amount": 500000,
    "loan_term": 120,
    "credit_score": 720,
    "existing_loans": 1,
    "property_area": "Urban",
    "dependents": 2
  }'
```

### Example Response

```json
{
  "approved": true,
  "risk_score": 18.4,
  "confidence": 0.816,
  "decision": "Approved ✅",
  "threshold_used": 0.44,
  "top_factors": [
    { "feature": "credit_score", "shap_value": 0.412, "impact": "positive" },
    { "feature": "debt_to_income", "shap_value": -0.183, "impact": "negative" }
  ]
}
```

---

## Project Structure

```
finguard-ai/
│
├── data/
│   ├── raw/                    # Generated dataset
│   ├── processed/              # Cleaned + feature-engineered data
│   └── generate_data.py        # Synthetic dataset generator (5000 records)
│
├── src/
│   ├── preprocess.py           # Cleaning, encoding, SMOTE, feature engineering
│   ├── train.py                # Train 4 models, cross-validation, auto-select best
│   ├── evaluate.py             # Metrics, threshold tuning, ROC/PR curves
│   ├── explain.py              # SHAP explainer — global + per-prediction
│   └── fairness.py             # Demographic parity, equalized odds, predictive parity
│
├── models/                     # Serialized artifacts (.pkl files)
│
├── app/
│   ├── main.py                 # FastAPI app, lifespan, route registration
│   ├── schemas.py              # Pydantic request/response models
│   ├── dependencies.py         # Artifact store, prediction helper
│   ├── routers/
│   │   ├── predict.py          # POST /api/predict
│   │   ├── simulate.py         # POST /api/simulate
│   │   ├── dashboard.py        # GET /api/dashboard
│   │   └── health.py           # GET /api/health
│   ├── templates/              # Jinja2 HTML templates
│   └── static/                 # CSS, JS, assets
│
├── docs/                       # Generated plots (ROC, SHAP, fairness)
├── build.sh                    # Render build script
├── start.sh                    # Render start script
├── render.yaml                 # Render deployment config
├── requirements.txt
└── config.py                   # Centralized path and config constants
```

---

## Running Locally

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Clone the repo
git clone https://github.com/sdev4714/finguard-ai.git
cd finguard-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run ML Pipeline

```bash
python data/generate_data.py    # Generate synthetic dataset
python src/preprocess.py        # Clean, encode, SMOTE
python src/train.py             # Train all 4 models
python src/evaluate.py          # Evaluate + tune threshold
python src/explain.py           # Build SHAP explainer
python src/fairness.py          # Run fairness audit
```

### Start Server

```bash
uvicorn app.main:app --reload --port 8000
```

Open:
- `http://localhost:8000` — Loan application form
- `http://localhost:8000/simulator` — What-If simulator
- `http://localhost:8000/dashboard` — Live dashboard
- `http://localhost:8000/docs` — Swagger API docs

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 86.4% |
| F1 Score | 88.7% |
| Precision | 87.9% |
| Recall | 89.4% |
| ROC-AUC | 94.3% |

---

## Fairness Audit Results

| Attribute | Disparity | TPR Δ | Verdict |
|---|---|---|---|
| Gender | 0.26% | 3.09% | ✅ Fair |
| Education | 8.89% | 1.53% | ✅ Fair |
| Property Area | 1.32% | 5.80% | ✅ Fair |

Disparity threshold: < 10% considered fair (industry standard).

---

## Deployment

Deployed on **Render** (free tier) with a custom domain via **Cloudflare DNS**.

```
git push origin main
       ↓
GitHub Actions triggers
       ↓
Render deploy hook fires
       ↓
Render pulls latest code + rebuilds
       ↓
Server live at loan.devrishi.tech
```

---

## Resume Highlights

- Built an end-to-end ML system with **stacking ensemble** (LR + RF + XGBoost) achieving **94.3% ROC-AUC**
- Implemented **SHAP explainability** on every prediction — regulatory-grade transparency
- Engineered **6 derived financial features** (debt-to-income, EMI ratio etc.) improving model signal
- Applied **SMOTE** for class imbalance and **threshold tuning** for business-cost-aware decisions
- Conducted **fairness audit** across 3 sensitive attributes — demographic parity + equalized odds
- Built a **FastAPI REST API** with Pydantic validation, dependency injection, and auto Swagger docs
- Deployed on a **custom domain** (`loan.devrishi.tech`) with CI/CD via GitHub Actions + Render

---

## Author

**Rishi Dev** — B.Tech CSE, 3rd Year  
GitHub: [@sdev4714](https://github.com/sdev4714)  
Domain: [devrishi.tech](https://devrishi.tech)
