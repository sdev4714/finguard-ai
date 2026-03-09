import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR         = os.path.join(BASE_DIR, "models")
DOCS_DIR           = os.path.join(BASE_DIR, "docs")

RAW_DATA_PATH      = os.path.join(DATA_RAW_DIR,       "loan_data.csv")
PROCESSED_DATA_PATH= os.path.join(DATA_PROCESSED_DIR, "cleaned_data.csv")
MODEL_PATH         = os.path.join(MODELS_DIR,          "loan_model.pkl")
SCALER_PATH        = os.path.join(MODELS_DIR,          "scaler.pkl")
ENCODERS_PATH      = os.path.join(MODELS_DIR,          "label_encoders.pkl")
EXPLAINER_PATH     = os.path.join(MODELS_DIR,          "explainer.pkl")
THRESHOLD_PATH     = os.path.join(MODELS_DIR,          "threshold.pkl")
FAIRNESS_PATH      = os.path.join(MODELS_DIR,          "fairness_audit.pkl")

# ── Model settings ────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD  = 0.5
TEST_SIZE          = 0.2
RANDOM_STATE       = 42
CV_FOLDS           = 5
SMOTE_ENABLED      = True

# ── Feature config ────────────────────────────────────────────────────────────
CATEGORICAL_COLS   = ['gender', 'education', 'employment_type', 'property_area']
TARGET_COL         = 'approved'
SENSITIVE_COLS     = ['gender', 'education', 'property_area']

FEATURE_COLS = [
    'age', 'gender', 'education', 'employment_type',
    'income', 'co_applicant_income', 'loan_amount',
    'loan_term', 'credit_score', 'existing_loans',
    'property_area', 'dependents',
    'debt_to_income', 'total_income', 'income_per_dependent',
    'loan_to_income_ratio', 'emi', 'emi_to_income'
]

# ── App settings ──────────────────────────────────────────────────────────────
APP_HOST           = "0.0.0.0"
APP_PORT           = 8000
APP_VERSION        = "1.0.0"
APP_TITLE          = "FinGuard AI — Loan Approval System"
DEBUG              = os.getenv("DEBUG", "false").lower() == "true"



