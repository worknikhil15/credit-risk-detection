import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("extra_trees_credit_model.pkl")

# Load encoders
encoders = {
    col: joblib.load(f"{col}_encoder.pkl")
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}

# App title
st.title("Credit Risk Prediction App")

st.write(
    "Enter applicant information to predict whether the credit risk is Good or Bad."
)

# User inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=80,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

job = st.number_input(
    "Job (0-3)",
    min_value=0,
    max_value=3,
    value=1
)

housing = st.selectbox(
    "Housing",
    ["own", "rent", "free"]
)

saving_accounts = st.selectbox(
    "Saving Accounts",
    ["little", "moderate", "rich", "quite rich"]
)

checking_account = st.selectbox(
    "Checking Account",
    ["little", "moderate", "rich", "quite rich"]
)

credit_amount = st.number_input(
    "Credit Amount",
    min_value=0,
    value=1000
)

duration = st.number_input(
    "Duration (months)",
    min_value=1,
    value=12
)

# Create dataframe for prediction
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [
        encoders["Saving accounts"].transform([saving_accounts])[0]
    ],
    "Checking account": [
        encoders["Checking account"].transform([checking_account])[0]
    ],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

# Predict button
if st.button("Predict Risk"):

    try:
        # Make prediction
        prediction = model.predict(input_df)[0]

        # Display result
        if prediction == 1:
            st.success("The predicted credit risk is: Good")
        else:
            st.error("The predicted credit risk is: Bad")

        # Optional: show entered data
        st.subheader("Input Data")
        st.write(input_df)

    except Exception as e:
        st.error(f"Prediction error: {e}")