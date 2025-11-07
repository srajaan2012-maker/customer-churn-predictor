import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# Cache the model loading for performance
@st.cache_resource
def load_model():
    try:
        # Load model assets
        assets = joblib.load('churn_prediction_model.pkl')
        st.success("✅ Model loaded successfully!")
        return assets
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        return None

def main():
    st.title("🏦 Customer Churn Prediction Dashboard")
    st.markdown("Predict which customers are likely to churn and take proactive action!")
    
    # Display model info
    st.sidebar.header("📊 Model Information")
    st.sidebar.metric("Accuracy", "86.5%")
    st.sidebar.metric("Recall", "46.4%")
    st.sidebar.metric("AUC Score", "85.0%")
    
    st.sidebar.header("🎯 Top Risk Factors")
    st.sidebar.write("1. **Age** - Older customers")
    st.sidebar.write("2. **Geography** - German customers")  
    st.sidebar.write("3. **Activity** - Inactive members")
    st.sidebar.write("4. **Balance** - High balance")
    st.sidebar.write("5. **Gender** - Female customers")
    
    # Load model
    assets = load_model()
    if assets is None:
        st.error("Failed to load the prediction model. Please check the deployment.")
        return
    
    model = assets['model']
    scaler = assets['scaler']
    feature_names = assets['feature_names']
    
    # Main input form
    st.header("📋 Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        age = st.slider('Age', 18, 80, 40)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        geography = st.selectbox('Country', ['France', 'Germany', 'Spain'])
        
        st.subheader("Financial Information")
        credit_score = st.slider('Credit Score', 350, 850, 650)
        balance = st.number_input('Account Balance ($)', 0.0, 500000.0, 50000.0)
        estimated_salary = st.number_input('Estimated Salary ($)', 0.0, 200000.0, 50000.0)
    
    with col2:
        st.subheader("Banking Relationship")
        tenure = st.slider('Tenure (Years)', 0, 10, 5)
        num_products = st.slider('Number of Products', 1, 4, 2)
        has