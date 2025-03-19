import streamlit as st
import requests
import json
import os

# Azure ML Endpoint Details
AZURE_ENDPOINT = "https://credit-endpoint-2e0d036d.eastus2.inference.ml.azure.com/score"
API_KEY = "8Ar1mIs6kV4QejpO0OUNYcKjjdHpvGWCZfg99cCF6FbkWdCZpsfAJQQJ99BCAAAAAAAAAAAAINFRAZML1n0R"


# Headers for request
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

# Set page configuration to wide layout
st.set_page_config(layout="wide")

# Streamlit UI
st.title("Azure ML Model Deployment with Streamlit")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Data")
    # Example input form with increased height
    input_data = st.text_area("Enter input data in JSON format:", "{}", height=250)

    # Prediction button
    if st.button("Predict", use_container_width=True):
        try:
            # Convert input to JSON
            payload = json.loads(input_data)

            # Display formatted input JSON
            st.markdown("### Input JSON")
            st.json(payload)

            # Send request to Azure ML Endpoint
            response = requests.post(AZURE_ENDPOINT, headers=HEADERS, json=payload)

            # Display Results in the right column
            with col2:
                st.subheader("Prediction Results")
                if response.status_code == 200:
                    st.success("Prediction Successful!")
                    result = response.json()
                    st.markdown("### Complete Prediction Output")
                    st.json(result)
                else:
                    st.error(f"Request failed with status code {response.status_code}")
                    st.text(response.text)

        except json.JSONDecodeError:
            st.error("Invalid JSON input. Please enter valid JSON data.")
