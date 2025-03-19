import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    st.title("Azure Machine Learning Endpoint Tester")

    # Sidebar for configuration
    st.sidebar.header("API Configuration")
    api_endpoint = st.sidebar.text_input(
        "Azure ML Endpoint URL", value=os.getenv("AZURE_ML_ENDPOINT", "")
    )

    # Main area for user input
    st.subheader("Input Data")

    # Create input data dictionary with the specified features
    input_data = {}

    # Demographic features
    st.markdown("### Demographic Information")
    col1, col2 = st.columns(2)

    with col1:
        input_data["LIMIT_BAL"] = st.number_input(
            "Credit Limit (LIMIT_BAL)", value=200000.0, step=10000.0
        )
        input_data["SEX"] = st.selectbox(
            "Gender (SEX)",
            options=[1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female",
        )
        input_data["AGE"] = st.number_input(
            "Age (AGE)", min_value=18, max_value=100, value=35
        )

    with col2:
        input_data["EDUCATION"] = st.selectbox(
            "Education Level (EDUCATION)",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Graduate",
                2: "University",
                3: "High School",
                4: "Others",
            }.get(x, str(x)),
        )
        input_data["MARRIAGE"] = st.selectbox(
            "Marital Status (MARRIAGE)",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}.get(
                x, str(x)
            ),
        )

    # Payment history features
    st.markdown("### Payment History")
    col1, col2, col3 = st.columns(3)

    with col1:
        input_data["PAY_0"] = st.number_input(
            "PAY_0", min_value=-2, max_value=9, value=0
        )
        input_data["PAY_2"] = st.number_input(
            "PAY_2", min_value=-2, max_value=9, value=0
        )

    with col2:
        input_data["PAY_3"] = st.number_input(
            "PAY_3", min_value=-2, max_value=9, value=0
        )
        input_data["PAY_4"] = st.number_input(
            "PAY_4", min_value=-2, max_value=9, value=0
        )

    with col3:
        input_data["PAY_5"] = st.number_input(
            "PAY_5", min_value=-2, max_value=9, value=0
        )
        input_data["PAY_6"] = st.number_input(
            "PAY_6", min_value=-2, max_value=9, value=0
        )

    # Bill amount features
    st.markdown("### Bill Amounts")
    col1, col2 = st.columns(2)

    with col1:
        input_data["BILL_AMT1"] = st.number_input("BILL_AMT1", value=0.0, step=1000.0)

    with col2:
        input_data["BILL_AMT2"] = st.number_input("BILL_AMT2", value=0.0, step=1000.0)

    # Advanced: JSON input option
    use_json = st.checkbox("Use custom JSON input")
    if use_json:
        json_input = st.text_area(
            "Enter JSON input:", value=json.dumps({"data": [input_data]}, indent=2)
        )
        try:
            input_data = json.loads(json_input)
        except json.JSONDecodeError:
            st.error("Invalid JSON format")

    # Button to trigger the prediction
    if st.button("Get Prediction"):
        if not api_endpoint:
            st.error("Please provide the API endpoint URL.")
        else:
            with st.spinner("Getting prediction..."):
                prediction = call_azure_ml_endpoint(api_endpoint, input_data)

                # Display results
                st.subheader("Prediction Result")
                if prediction:
                    st.json(prediction)


def call_azure_ml_endpoint(endpoint_url, input_data):
    """Call the Azure ML endpoint with the provided input data"""
    try:
        # Format the payload based on whether we're using custom JSON
        payload = input_data if isinstance(input_data, dict) else {"data": [input_data]}

        headers = {
            "Content-Type": "application/json",
        }

        response = requests.post(endpoint_url, json=payload, headers=headers)
        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling the endpoint: {e}")
        return None


if __name__ == "__main__":
    main()
