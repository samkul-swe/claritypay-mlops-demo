from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, List
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="ClarityPay Credit Scoring API",
    description="Real-time credit risk assessment with explainability",
    version="1.0.0"
)

# Load model at startup
model = None
feature_names = [
    'age', 'annual_income', 'debt_to_income_ratio',
    'num_credit_lines', 'num_late_payments', 'credit_utilization',
    'months_since_last_delinquency', 'num_credit_inquiries', 'purchase_amount'
]

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load('models/credit_model.pkl')
    print("✅ Model loaded successfully")

# Request model
class CreditApplication(BaseModel):
    applicant_id: str = Field(..., description="Unique applicant identifier")
    age: int = Field(..., ge=18, le=100, description="Applicant age")
    annual_income: float = Field(..., ge=0, description="Annual income in USD")
    debt_to_income_ratio: float = Field(..., ge=0, le=5, description="Debt-to-income ratio")
    num_credit_lines: int = Field(..., ge=0, description="Number of open credit lines")
    num_late_payments: int = Field(..., ge=0, description="Number of late payments")
    credit_utilization: float = Field(..., ge=0, le=2, description="Credit utilization ratio")
    months_since_last_delinquency: int = Field(..., ge=0, description="Months since last delinquency")
    num_credit_inquiries: int = Field(..., ge=0, description="Number of credit inquiries")
    purchase_amount: float = Field(..., ge=0, description="Purchase amount")

    class Config:
        json_schema_extra = {
            "example": {
                "applicant_id": "APP_000001",
                "age": 35,
                "annual_income": 65000,
                "debt_to_income_ratio": 0.35,
                "num_credit_lines": 5,
                "num_late_payments": 1,
                "credit_utilization": 0.45,
                "months_since_last_delinquency": 24,
                "num_credit_inquiries": 2,
                "purchase_amount": 3500
            }
        }

# Response model
class CreditDecision(BaseModel):
    applicant_id: str
    credit_score: int
    default_probability: float
    approval_recommendation: str
    recommended_terms: Dict[str, any]
    explanation: List[Dict[str, any]]
    confidence: str

def calculate_loan_terms(credit_score: int, purchase_amount: float) -> Dict:
    """Calculate loan terms based on credit score"""
    if credit_score >= 750:
        return {
            "approved": True,
            "term_months": 12,
            "apr": 8.99,
            "monthly_payment": round(purchase_amount * 1.0899 / 12, 2)
        }
    elif credit_score >= 650:
        return {
            "approved": True,
            "term_months": 6,
            "apr": 14.99,
            "monthly_payment": round(purchase_amount * 1.1499 / 6, 2)
        }
    elif credit_score >= 550:
        return {
            "approved": True,
            "term_months": 4,
            "apr": 22.99,
            "monthly_payment": round(purchase_amount * 1.2299 / 4, 2)
        }
    else:
        return {
            "approved": False,
            "reason": "Credit score below minimum threshold"
        }

def get_explanation(features: np.ndarray) -> List[Dict]:
    """Generate SHAP-based explanation"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    # Get top 3 influential features
    feature_impact = []
    for i, feature_name in enumerate(feature_names):
        impact = shap_values[0][i]
        feature_impact.append({
            "feature": feature_name,
            "value": float(features[0][i]),
            "impact": float(impact),
            "direction": "increases" if impact > 0 else "decreases"
        })
    
    # Sort by absolute impact
    feature_impact.sort(key=lambda x: abs(x["impact"]), reverse=True)
    
    return feature_impact[:3]

@app.get("/")
def root():
    return {
        "message": "ClarityPay Credit Scoring API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=CreditDecision)
def predict_credit_risk(application: CreditApplication):
    """
    Predict credit risk and provide loan terms recommendation
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Prepare features
    features = np.array([[
        application.age,
        application.annual_income,
        application.debt_to_income_ratio,
        application.num_credit_lines,
        application.num_late_payments,
        application.credit_utilization,
        application.months_since_last_delinquency,
        application.num_credit_inquiries,
        application.purchase_amount
    ]])
    
    # Make prediction
    default_probability = float(model.predict_proba(features)[0][1])
    credit_score = int((1 - default_probability) * 850)  # Convert to FICO-like score
    
    # Calculate terms
    terms = calculate_loan_terms(credit_score, application.purchase_amount)
    
    # Get explanation
    explanation = get_explanation(features)
    
    # Determine approval
    if terms["approved"]:
        recommendation = "APPROVED"
        confidence = "HIGH" if default_probability < 0.2 else "MEDIUM"
    else:
        recommendation = "DECLINED"
        confidence = "HIGH"
    
    return CreditDecision(
        applicant_id=application.applicant_id,
        credit_score=credit_score,
        default_probability=round(default_probability, 4),
        approval_recommendation=recommendation,
        recommended_terms=terms,
        explanation=explanation,
        confidence=confidence
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)