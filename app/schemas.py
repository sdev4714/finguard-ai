from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum

# ── Enums ─────────────────────────────────────────────────────────────────────

class GenderEnum(str, Enum):
    male   = "Male"
    female = "Female"

class EducationEnum(str, Enum):
    graduate     = "Graduate"
    not_graduate = "Not Graduate"

class EmploymentEnum(str, Enum):
    salaried      = "Salaried"
    self_employed = "Self-Employed"
    business      = "Business"

class PropertyAreaEnum(str, Enum):
    urban     = "Urban"
    semiurban = "Semiurban"
    rural     = "Rural"

# ── Request Schemas ───────────────────────────────────────────────────────────

class LoanApplication(BaseModel):
    age:                 int   = Field(..., ge=18, le=75,        description="Applicant age")
    gender:              GenderEnum                              
    education:           EducationEnum                          
    employment_type:     EmploymentEnum                         
    income:              float = Field(..., ge=0,               description="Monthly income in INR")
    co_applicant_income: float = Field(0.0, ge=0,              description="Co-applicant monthly income")
    loan_amount:         float = Field(..., ge=1000,            description="Requested loan amount in INR")
    loan_term:           int   = Field(..., ge=6, le=360,       description="Loan term in months")
    credit_score:        int   = Field(..., ge=300, le=900,     description="Credit score")
    existing_loans:      int   = Field(0, ge=0, le=10,         description="Number of existing loans")
    property_area:       PropertyAreaEnum                       
    dependents:          int   = Field(0, ge=0, le=10,         description="Number of dependents")

    @field_validator('income', 'co_applicant_income', 'loan_amount')
    @classmethod
    def must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Value must be non-negative")
        return round(v, 2)

    @field_validator('credit_score')
    @classmethod
    def valid_credit_score(cls, v):
        if not (300 <= v <= 900):
            raise ValueError("Credit score must be between 300 and 900")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
            }
        }
    }

class SimulationRequest(BaseModel):
    base_application: LoanApplication
    overrides:        dict = Field(
        ...,
        description="Fields to override for simulation. e.g. {'income': 100000, 'credit_score': 750}"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "base_application": {
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
                },
                "overrides": {
                    "income": 120000,
                    "credit_score": 780
                }
            }
        }
    }

# ── Response Schemas ──────────────────────────────────────────────────────────

class SHAPEntry(BaseModel):
    feature:    str
    value:      float
    shap_value: float
    impact:     str

class PredictionResponse(BaseModel):
    approved:        bool
    risk_score:      float = Field(..., description="Risk score 0-100, lower is safer")
    confidence:      float = Field(..., description="Model confidence 0-1")
    decision:        str   = Field(..., description="Approved / Rejected")
    threshold_used:  float
    top_factors:     list[SHAPEntry]
    shap_data:       dict

class SimulationEntry(BaseModel):
    label:      str
    approved:   bool
    risk_score: float
    confidence: float

class SimulationResponse(BaseModel):
    original:   SimulationEntry
    modified:   SimulationEntry
    delta:      dict = Field(..., description="Changes in risk score and confidence")
    overrides:  dict

class HealthResponse(BaseModel):
    status:     str
    model_type: str
    threshold:  float
    version:    str

class FairnessMetric(BaseModel):
    attribute:            str
    demographic_parity:   dict
    equalized_odds:       dict
    predictive_parity:    dict

class DashboardResponse(BaseModel):
    total_applications: int
    approval_rate:      float
    avg_risk_score:     float
    fairness_summary:   list[FairnessMetric]
    model_info:         dict