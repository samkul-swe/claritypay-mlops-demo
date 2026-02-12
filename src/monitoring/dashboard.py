import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

st.set_page_config(page_title="ClarityPay MLOps Dashboard", layout="wide")

st.title("🎯 ClarityPay Credit Scoring - MLOps Dashboard")
st.markdown("Real-time monitoring of credit scoring model performance")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔍 Data Drift", "📋 Recent Predictions"])

with tab1:
    st.header("Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model AUC-ROC", "0.8542", "+0.02")
    with col2:
        st.metric("Approval Rate", "68%", "-2%")
    with col3:
        st.metric("Avg Response Time", "145ms", "-15ms")
    with col4:
        st.metric("Daily Predictions", "1,234", "+156")
    
    # Mock performance over time
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    performance_data = pd.DataFrame({
        'date': dates,
        'auc': 0.85 + pd.Series(range(len(dates))).apply(lambda x: 0.01 * (x % 3 - 1)),
        'predictions': 1000 + pd.Series(range(len(dates))) * 10
    })
    
    fig = px.line(performance_data, x='date', y='auc', 
                  title='Model AUC-ROC Over Time')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Data Drift Detection")
    
    # Load drift summary if exists
    try:
        with open('monitoring/drift_summary.json', 'r') as f:
            drift_summary = json.load(f)
        
        drift_status = "🔴 Detected" if drift_summary['drift_detected'] else "🟢 No Drift"
        st.metric("Drift Status", drift_status)
        
        st.markdown(f"**Last Check:** {drift_summary['timestamp']}")
        st.markdown(f"**Reference Size:** {drift_summary['reference_size']} samples")
        st.markdown(f"**Current Size:** {drift_summary['current_size']} samples")
        
        if st.button("View Full Drift Report"):
            st.markdown("[Open Report](../../monitoring/drift_report.html)")
    except:
        st.warning("Run drift detection first: python src/monitoring/drift_detection.py")

with tab3:
    st.header("Recent Credit Decisions")
    
    # Mock recent predictions
    recent_predictions = pd.DataFrame({
        'Applicant ID': [f'APP_{i:06d}' for i in range(10)],
        'Credit Score': [720, 650, 580, 750, 690, 630, 770, 600, 710, 680],
        'Decision': ['APPROVED', 'APPROVED', 'APPROVED', 'APPROVED', 'APPROVED', 
                     'DECLINED', 'APPROVED', 'DECLINED', 'APPROVED', 'APPROVED'],
        'Amount': [3500, 2000, 1500, 5000, 2800, 1200, 4500, 900, 3200, 2500],
        'Timestamp': pd.date_range(start='2024-01-30 10:00', periods=10, freq='15min')
    })
    
    st.dataframe(recent_predictions, use_container_width=True)

st.sidebar.header("Model Information")
st.sidebar.markdown("""
**Model Version:** 1.0.0  
**Last Updated:** 2024-01-30  
**Framework:** XGBoost  
**Features:** 9  
**Training Samples:** 8,000  

**Status:** 🟢 Healthy
""")

if __name__ == "__main__":
    # Run: streamlit run src/monitoring/dashboard.py
    pass