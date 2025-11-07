"Replace your app.py with this version that includes better error handling:

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys

# Set page configuration
st.set_page_config(
    page_title=\"Customer Churn Predictor\",
    page_icon=\"🏦\",
    layout=\"wide\"
)

# Display system info for debugging
st.sidebar.write(f\"Python version: {sys.version}\")
st.sidebar.write(f\"Pandas version: {pd.__version__}\")

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        st.info(\"🔄 Loading prediction model...\")
        assets = joblib.load('churn_prediction_model.pkl')
        st.success(\"✅ Model loaded successfully!\")
        return assets
    except Exception as e:
        st.error(f\"❌ Error loading model: {e}\")
        st.info(\"💡 Please ensure 'churn_prediction_model.pkl' is in the root directory\")
        return None

def main():
    st.title(\"🏦 Customer Churn Prediction Dashboard\")
    st.markdown(\"Predict which customers are likely to churn and take proactive action!\")
    
    # Load model
    assets = load_model()
    if assets is None:
        st.stop()
    
    model = assets['model']
    scaler = assets['scaler']
    feature_names = assets['feature_names']
    
    st.sidebar.header(\"📊 Model Information\")
    st.sidebar.metric(\"Accuracy\", \"86.5%\")
    st.sidebar.metric(\"Recall\", \"46.4%\")
    st.sidebar.metric(\"AUC Score\", \"85.0%\")
    
    # Main input form
    st.header(\"📋 Enter Customer Information\")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(\"Demographics\")
        age = st.slider('Age', 18, 80, 40)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        geography = st.selectbox('Country', ['France', 'Germany', 'Spain'])
        
        st.subheader(\"Financial Information\")
        credit_score = st.slider('Credit Score', 350, 850, 650)
        balance = st.number_input('Account Balance ($)', 0.0, 500000.0, 50000.0)
        estimated_salary = st.number_input('Estimated Salary ($)', 0.0, 200000.0, 50000.0)
    
    with col2:
        st.subheader(\"Banking Relationship\")
        tenure = st.slider('Tenure (Years)', 0, 10, 5)
        num_products = st.slider('Number of Products', 1, 4, 2)
        has_credit_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
        is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])
    
    # Convert inputs
    gender_encoded = 1 if gender == 'Female' else 0
    has_credit_card_encoded = 1 if has_credit_card == 'Yes' else 0
    is_active_member_encoded = 1 if is_active_member == 'Yes' else 0
    
    # Create feature vector
    feature_vector = []
    for feature in feature_names:
        if feature.startswith('Geo_'):
            geo_feature = f\"Geo_{geography}\"
            feature_vector.append(1 if feature == geo_feature else 0)
        else:
            feature_vector.append(locals()[feature.lower()])
    
    # Prediction
    st.header(\"🎯 Churn Prediction Results\")
    
    if st.button('🔍 Predict Churn Risk', type='primary', use_container_width=True):
        try:
            feature_array = np.array(feature_vector).reshape(1, -1)
            feature_array_scaled = scaler.transform(feature_array)
            
            with st.spinner('Analyzing customer data...'):
                prediction = model.predict(feature_array_scaled)[0]
                probability = model.predict_proba(feature_array_scaled)[0][1]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error(f\"🚨 HIGH CHURN RISK\")
                    st.metric(\"Churn Probability\", f\"{probability:.1%}\")
                else:
                    st.success(f\"✅ LOW CHURN RISK\")
                    st.metric(\"Churn Probability\", f\"{probability:.1%}\")
            
            with col2:
                st.subheader(\"Customer Profile\")
                st.write(f\"**Age:** {age}\")
                st.write(f\"**Gender:** {gender}\")
                st.write(f\"**Country:** {geography}\")
                st.write(f\"**Active:** {is_active_member}\")
                
        except Exception as e:
            st.error(f\"Prediction error: {e}\")

if __name__ == '__main__':
    main()
```"