import streamlit as st
import pandas as pd
import joblib

# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# Now you can load your model and define other elements
@st.cache_resource
def load_pipeline():
    return joblib.load("churn_pipeline_v2.pkl")

model = load_pipeline()

# Now you can use st.title
st.title("📞 Telco Customer Churn Predictor")

# Sidebar for choosing mode
mode = st.sidebar.selectbox("Choose Prediction Mode", ["Single Customer", "Batch (CSV Upload)"])

if mode == "Single Customer":
    st.subheader("Enter Customer Details")
    
    # Organize inputs into 4 columns for better UI
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen (0/1)", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    with c2:
        tenure = st.number_input("Tenure (months)", min_value=0, value=12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with c3:
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with c4:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    # Bottom Row for Payment & Charges
    c5, c6 = st.columns(2)
    with c5:
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    with c6:
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0)
        total = st.number_input("Total Charges ($)", min_value=0.0, value=780.0)

    if st.button("Predict Churn"):
        # Map all inputs exactly as they appear in the dataset header
        data = {
            'gender': [gender], 'SeniorCitizen': [senior], 'Partner': [partner],
            'Dependents': [dependents], 'tenure': [tenure], 'PhoneService': [phone],
            'MultipleLines': [multiple], 'InternetService': [internet],
            'OnlineSecurity': [security], 'OnlineBackup': [backup],
            'DeviceProtection': [protection], 'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv], 'StreamingMovies': [streaming_movies],
            'Contract': [contract], 'PaperlessBilling': [paperless],
            'PaymentMethod': [payment], 'MonthlyCharges': [monthly],
            'TotalCharges': [total]
        }
        input_df = pd.DataFrame(data)
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ **High Risk**: Likelihood of Churn is **{probability:.2%}**")
        else:
            st.success(f"✅ **Low Risk**: Likelihood of Churn is **{probability:.2%}**")

else:
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload customer CSV file", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        # Drop non-feature columns if they exist
        features_df = batch_df.drop(columns=['customerID', 'Churn'], errors='ignore')
        
        predictions = model.predict(features_df)
        probs = model.predict_proba(features_df)[:, 1]
        
        batch_df['Churn_Prediction'] = ["Yes" if p == 1 else "No" for p in predictions]
        batch_df['Churn_Probability'] = probs
        
        st.write(batch_df[['customerID', 'Churn_Prediction', 'Churn_Probability']].head())
        
        # Download button for results
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "churn_predictions.csv", "text/csv")
