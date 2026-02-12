# Save model card as markdown
model_card = f"""
# Credit Scoring Model Card

## Model Details
- **Model Type:** XGBoost Classifier
- **Purpose:** Predict default risk for point-of-sale credit applications
- **Version:** 1.0
- **Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Performance Metrics
- **AUC-ROC:** {auc_score:.4f}
- **Accuracy:** {(y_pred == y_test).mean():.4f}

## Training Data
- **Samples:** {len(X_train)}
- **Features:** {len(features)}
- **Default Rate:** {y_train.mean():.2%}

## Features Used
{chr(10).join([f'- {f}' for f in features])}

## Top 3 Important Features
{chr(10).join([f'{i+1}. {row["feature"]}: {row["importance"]:.4f}' 
               for i, row in feature_importance.head(3).iterrows()])}

## Intended Use
This model is designed for point-of-sale credit decisions for purchases between $500-$10,000.
The model provides a risk score (0-1) indicating probability of default.

## Limitations
- Model trained on synthetic data for demonstration
- Should not be used for actual credit decisions without proper validation
- Requires monitoring for data drift in production

## Ethical Considerations
- Model should be monitored for fairness across demographic groups
- Adverse action reasons must be provided for declined applications
- Regular bias testing is required

## Contact
Built by Sampada Kulkarni for ClarityPay MLOps demonstration
"""

with open('models/MODEL_CARD.md', 'w') as f:
    f.write(model_card)

mlflow.log_artifact('models/MODEL_CARD.md')
print("✅ Model card created!")