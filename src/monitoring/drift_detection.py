import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently import ColumnMapping
import json

def detect_drift():
    """Check for data drift between reference and current data"""
    
    # Load reference data (training set)
    reference_data = pd.read_csv('data/train_reference.csv')
    
    # Load current data (test set - simulating production)
    current_data = pd.read_csv('data/test_data.csv')
    
    # Define column mapping
    column_mapping = ColumnMapping(
        target='default_risk',
        numerical_features=[
            'age', 'annual_income', 'debt_to_income_ratio',
            'num_credit_lines', 'num_late_payments', 'credit_utilization',
            'months_since_last_delinquency', 'num_credit_inquiries', 'purchase_amount'
        ]
    )
    
    # Generate drift report
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])
    
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    # Save report
    report.save_html('monitoring/drift_report.html')
    
    # Extract drift scores
    report_dict = report.as_dict()
    drift_detected = report_dict['metrics'][0]['result']['dataset_drift']
    
    print(f"🔍 Drift Detection Results:")
    print(f"Dataset Drift Detected: {drift_detected}")
    
    # Save summary
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "drift_detected": drift_detected,
        "reference_size": len(reference_data),
        "current_size": len(current_data)
    }
    
    with open('monitoring/drift_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return drift_detected

if __name__ == "__main__":
    drift = detect_drift()
    print("✅ Drift detection complete! Check monitoring/drift_report.html")