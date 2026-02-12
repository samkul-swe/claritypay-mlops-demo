from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import uvicorn

# MongoDB integration
from database import PredictionLogger

# Initialize FastAPI
app = FastAPI(
    title="ClarityPay Credit Scoring API",
    description="Production-ready MLOps platform for point-of-sale lending",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB logger
logger = PredictionLogger()

# Global model variable
model = None
feature_names = [
    'age', 'annual_income', 'debt_to_income_ratio',
    'num_credit_lines', 'num_late_payments', 'credit_utilization',
    'months_since_last_delinquency', 'num_credit_inquiries', 'purchase_amount'
]

@app.on_event("startup")
def load_model():
    global model
    try:
        model_path = os.path.join('models', 'credit_model.pkl')
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# Request model
class CreditApplication(BaseModel):
    applicant_id: str = Field(..., description="Unique applicant identifier")
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., ge=0)
    debt_to_income_ratio: float = Field(..., ge=0, le=5)
    num_credit_lines: int = Field(..., ge=0)
    num_late_payments: int = Field(..., ge=0)
    credit_utilization: float = Field(..., ge=0, le=2)
    months_since_last_delinquency: int = Field(..., ge=0)
    num_credit_inquiries: int = Field(..., ge=0)
    purchase_amount: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "applicant_id": "APP_DEMO_001",
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

def calculate_loan_terms(credit_score: int, purchase_amount: float):
    """Calculate loan terms based on credit score"""
    if credit_score >= 750:
        return {
            "approved": True,
            "term_months": 12,
            "apr": 8.99,
            "monthly_payment": round(purchase_amount * 1.0899 / 12, 2),
            "risk_tier": "Prime"
        }
    elif credit_score >= 650:
        return {
            "approved": True,
            "term_months": 6,
            "apr": 14.99,
            "monthly_payment": round(purchase_amount * 1.1499 / 6, 2),
            "risk_tier": "Near-Prime"
        }
    elif credit_score >= 550:
        return {
            "approved": True,
            "term_months": 4,
            "apr": 22.99,
            "monthly_payment": round(purchase_amount * 1.2299 / 4, 2),
            "risk_tier": "Subprime"
        }
    else:
        return {
            "approved": False,
            "reason": "Credit score below minimum threshold (550)",
            "risk_tier": "High Risk"
        }

def get_simple_explanation(features: np.ndarray, prediction_proba: float):
    """Generate simple explanation"""
    feature_dict = {name: float(val) for name, val in zip(feature_names, features[0])}
    
    explanations = []
    
    if feature_dict['debt_to_income_ratio'] > 0.5:
        explanations.append({
            "factor": "High debt-to-income ratio",
            "value": feature_dict['debt_to_income_ratio'],
            "impact": "negative"
        })
    elif feature_dict['debt_to_income_ratio'] < 0.3:
        explanations.append({
            "factor": "Low debt-to-income ratio",
            "value": feature_dict['debt_to_income_ratio'],
            "impact": "positive"
        })
    
    if feature_dict['num_late_payments'] > 2:
        explanations.append({
            "factor": "Multiple late payments",
            "value": feature_dict['num_late_payments'],
            "impact": "negative"
        })
    elif feature_dict['num_late_payments'] == 0:
        explanations.append({
            "factor": "No late payments",
            "value": feature_dict['num_late_payments'],
            "impact": "positive"
        })
    
    if feature_dict['credit_utilization'] > 0.7:
        explanations.append({
            "factor": "High credit utilization",
            "value": feature_dict['credit_utilization'],
            "impact": "negative"
        })
    elif feature_dict['credit_utilization'] < 0.3:
        explanations.append({
            "factor": "Low credit utilization",
            "value": feature_dict['credit_utilization'],
            "impact": "positive"
        })
    
    return explanations[:3]

@app.get("/")
def root():
    return {
        "message": "ClarityPay Credit Scoring API - MLOps Demo",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "stats": "/stats",
            "recent": "/recent",
            "docs": "/docs"
        },
        "mongodb_connected": logger.db is not None
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mongodb_connected": logger.db is not None,
        "version": "1.0.0"
    }

@app.post("/predict")
def predict_credit_risk(application: CreditApplication):
    """Predict credit risk and log to MongoDB"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
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
        credit_score = int((1 - default_probability) * 850)
        
        # Calculate terms
        terms = calculate_loan_terms(credit_score, application.purchase_amount)
        
        # Get explanation
        explanation = get_simple_explanation(features, default_probability)
        
        # Determine confidence
        if abs(default_probability - 0.5) > 0.3:
            confidence = "HIGH"
        elif abs(default_probability - 0.5) > 0.15:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Build response
        result = {
            "applicant_id": application.applicant_id,
            "credit_score": credit_score,
            "default_probability": round(default_probability, 4),
            "approval_recommendation": "APPROVED" if terms["approved"] else "DECLINED",
            "recommended_terms": terms,
            "explanation": explanation,
            "confidence": confidence,
            "model_version": "1.0.0"
        }
        
        # Log to MongoDB (if connected)
        doc_id = logger.log_prediction(application.dict(), result)
        if doc_id:
            result["logged_to_mongodb"] = True
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/stats")
def get_prediction_stats():
    """Get prediction statistics from MongoDB"""
    return logger.get_stats()

@app.get("/recent")
def get_recent_predictions(limit: int = 10):
    """Get recent predictions from MongoDB"""
    predictions = logger.get_recent_predictions(limit)
    return {"count": len(predictions), "predictions": predictions}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
