# 📊 ClarityPay MLOps Demo: Credit Risk Assessment Platform

> Production-ready MLOps system for point-of-sale lending with explainable AI

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.8-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 🎯 Project Overview

This project demonstrates end-to-end MLOps capabilities for fintech credit scoring:

- ✅ Real-time credit risk prediction API (<200ms latency)
- ✅ Explainable AI with SHAP values for regulatory compliance
- ✅ Model monitoring and drift detection
- ✅ MLflow experiment tracking and model registry
- ✅ Production-ready Docker deployment
- ✅ Interactive monitoring dashboard

**Built for:** ClarityPay MLOps Engineer position  
**Author:** Sampada Kulkarni  
**Tech Stack:** Python, XGBoost, FastAPI, MLflow, Docker, Evidently AI

---

## 🏗️ Architecture
┌─────────────────┐
│   Credit Data   │
└────────┬────────┘
│
┌────────▼────────┐
│ Feature Eng     │
│ + Training      │
│  (XGBoost)      │
└────────┬────────┘
│
┌────────▼────────┐
│  MLflow         │
│  Tracking       │
└────────┬────────┘
│
┌────────▼────────┐
│  FastAPI        │
│  Inference      │
│  + SHAP         │
└────────┬────────┘
│
┌────────▼────────┐
│  Monitoring     │
│  Dashboard      │
└─────────────────┘
---

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone repo
git clone <your-repo-url>
cd claritypay-mlops-demo

# Start all services
docker-compose up

# Access services:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - MLflow: http://localhost:5000
# - Dashboard: http://localhost:8501
```

### Option 2: Local Development
```bash
# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train model
jupyter notebook notebooks/02_model_training.ipynb

# Run API
python src/api/main.py

# Run monitoring
streamlit run src/monitoring/dashboard.py
```

---

## 📋 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Credit Decision Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Response
```json
{
  "applicant_id": "APP_000001",
  "credit_score": 720,
  "default_probability": 0.1234,
  "approval_recommendation": "APPROVED",
  "recommended_terms": {
    "approved": true,
    "term_months": 12,
    "apr": 8.99,
    "monthly_payment": 318.73
  },
  "explanation": [
    {
      "feature": "debt_to_income_ratio",
      "value": 0.35,
      "impact": -0.15,
      "direction": "decreases"
    },
    {
      "feature": "num_late_payments",
      "value": 1,
      "impact": 0.08,
      "direction": "increases"
    }
  ],
  "confidence": "HIGH"
}
```

---

## 🎯 Key Features

### 1. Model Training with MLflow
- Automated experiment tracking
- Hyperparameter logging
- Model versioning and registry
- Feature importance analysis

### 2. Real-Time Inference API
- FastAPI with automatic documentation
- Input validation with Pydantic
- <200ms response time
- Explainable predictions (SHAP)

### 3. Explainable AI
- SHAP values for each prediction
- Top 3 influential features highlighted
- Regulatory compliance ready (ECOA, FCRA)
- Adverse action reason generation

### 4. Model Monitoring
- Data drift detection (Evidently AI)
- Performance tracking
- Business metrics dashboard
- Automated alerting (ready to integrate)

### 5. Production-Ready
- Docker containerization
- Docker Compose orchestration
- Health check endpoints
- Comprehensive logging

---

## 📊 Model Performance

- **AUC-ROC:** 0.8542
- **Accuracy:** 85.4%
- **Inference Time:** ~145ms (p95)
- **Training Samples:** 10,000
- **Features:** 9

---

## 🔍 Monitoring & Observability

### Drift Detection
```bash
python src/monitoring/drift_detection.py
```
Generates HTML report in `monitoring/drift_report.html`

### Dashboard
```bash
streamlit run src/monitoring/dashboard.py
```
Access at http://localhost:8501

**Dashboard Features:**
- Model performance metrics
- Drift detection status
- Recent predictions
- System health

---

## 📁 Project Structure
claritypay-mlops-demo/
├── data/
│   ├── credit_applications.csv    # Generated dataset
│   ├── train_reference.csv         # Training data
│   └── test_data.csv               # Test data
├── models/
│   ├── credit_model.pkl            # Trained model
│   ├── feature_importance.png      # Feature viz
│   └── MODEL_CARD.md               # Model documentation
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI application
│   └── monitoring/
│       ├── drift_detection.py      # Drift monitoring
│       └── dashboard.py            # Streamlit dashboard
├── monitoring/
│   ├── drift_report.html           # Latest drift report
│   └── drift_summary.json          # Drift metrics
├── docker-compose.yml              # Multi-service setup
├── Dockerfile                      # API container
├── requirements.txt                # Dependencies
└── README.md                       # This file
---

## 🎓 MLOps Capabilities Demonstrated

✅ **Data Engineering:** Synthetic data generation, feature engineering  
✅ **Model Training:** XGBoost with proper validation  
✅ **Experiment Tracking:** MLflow integration  
✅ **Model Serving:** FastAPI with low latency  
✅ **Explainability:** SHAP for regulatory compliance  
✅ **Monitoring:** Drift detection and performance tracking  
✅ **Containerization:** Docker and Docker Compose  
✅ **Documentation:** Model cards, API docs, README  

---

## 🔄 Future Enhancements

- [ ] A/B testing framework
- [ ] Automated retraining pipeline
- [ ] Integration with AWS SageMaker
- [ ] Fairness metrics and bias detection
- [ ] Multi-model ensemble
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline with GitHub Actions

---

## 💼 Why This Matters for ClarityPay

**Point-of-Sale Lending Alignment:**
- Fast credit decisions (<500ms target)
- Explainable AI for compliance (ECOA, FCRA)
- Risk-based pricing (different terms by score)
- Production monitoring from day one

**MLOps Best Practices:**
- Full ML lifecycle coverage
- Reproducible experiments
- Model governance
- Operational monitoring

---

## 🤝 About

**Author:** Sampada Kulkarni  
**LinkedIn:** [linkedin.com/in/samkul-swe](https://linkedin.com/in/samkul-swe/)  
**Portfolio:** [samkul-swe/portfolio](https://github.com/samkul-swe/portfolio)  
**Email:** kulkarni.samp@northeastern.edu

Built to demonstrate MLOps capabilities for the ClarityPay MLOps Engineer position.

**Background:**
- 3+ years at IBM building monitoring & AIOps platforms
- Expertise in data integration, containerization, CI/CD
- MS Computer Science, Northeastern University

---

## 📝 License

MIT License - Feel free to use this for learning or demonstration purposes.