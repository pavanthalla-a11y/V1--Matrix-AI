import streamlit as st
import requests
import json
import pandas as pd
import time

# Base URL of your FastAPI application
FASTAPI_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Matrix AI - Synthetic Data Generator",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Matrix AI - Synthetic Data Generator")
st.markdown("Use AI to generate, preview, and store synthetic data.")
st.markdown("---")


# --- Session State Management ---
# Use session state to persist data across reruns
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'generated_metadata' not in st.session_state:
    st.session_state.generated_metadata = None
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None


## Generate Synthetic Data Section
st.header("1. Generate Data")
with st.container(border=True):
    data_description = st.text_area(
        "Describe the data you want to generate:",
        "A database for an e-commerce platform with tables for customers, products, and their orders.",
        height=100
    )
    num_records = st.number_input(
        "Number of records to generate:",
        min_value=1,
        value=500
    )

    if st.button("Generate Synthetic Data", use_container_width=True):
        if data_description:
            payload = {
                "data_description": data_description,
                "num_records": num_records
            }
            try:
                # Use a progress bar for a better user experience
                with st.spinner("Calling AI agent and generating data... This may take a few moments."):
                    progress_bar = st.progress(0, text="Generating data...")
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.01)

                    response = requests.post(
                        f"{FASTAPI_URL}/generate",
                        json=payload,
                        timeout=600  # Set a generous timeout for the long-running process
                    )
                    response.raise_for_status()
                    
                    st.session_state.data_generated = True
                    st.session_state.generated_metadata = response.json().get("metadata")
                    st.success("✅ Data generation complete!")
                    st.json(response.json())

            except requests.exceptions.RequestException as e:
                st.session_state.data_generated = False
                st.error(f"Error communicating with the API: {e}")
                st.info("Please ensure your FastAPI server is running on http://localhost:8000.")
        else:
            st.warning("Please provide a data description.")

st.markdown("---")

## View Sample Data Section
st.header("2. View Sample Data")
with st.container(border=True):
    if st.session_state.data_generated and st.session_state.generated_metadata:
        tables = list(st.session_state.generated_metadata['tables'].keys())
        selected_table = st.selectbox(
            "Select a table to view a sample:",
            options=tables
        )
        sample_size = st.number_input(
            "Number of sample records to display:",
            min_value=1,
            value=10
        )

        if st.button("Get Sample Data", use_container_width=True):
            try:
                response = requests.get(
                    f"{FASTAPI_URL}/sample",
                    params={"table_name": selected_table, "sample_size": sample_size}
                )
                response.raise_for_status()
                
                sample_data = response.json().get("sample_data")
                st.session_state.sample_data = sample_data
                
                if sample_data:
                    st.subheader(f"Sample from {selected_table}")
                    st.dataframe(pd.DataFrame(sample_data))
                else:
                    st.warning("No sample data returned.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching sample data: {e}")
    else:
        st.info("Please generate data first to enable this section.")
        
st.markdown("---")

## Store Data Section
st.header("3. Store Data in GCS")
with st.container(border=True):
    if st.session_state.data_generated:
        st.warning("Storing data will clear the current session. Please ensure you have previewed all necessary data.")
        if st.button("Store to GCS", type="primary", use_container_width=True):
            payload = {"confirm_storage": True}
            try:
                with st.spinner("Uploading files to Google Cloud Storage..."):
                    response = requests.post(
                        f"{FASTAPI_URL}/store",
                        json=payload,
                        timeout=600  # Another generous timeout for upload
                    )
                    response.raise_for_status()
                    st.success("🎉 All files uploaded successfully!")
                    st.json(response.json())
                    
                    # Clear session state on successful storage
                    st.session_state.data_generated = False
                    st.session_state.generated_metadata = None
                    st.session_state.sample_data = None
                    st.rerun()

            except requests.exceptions.RequestException as e:
                st.error(f"Error storing data: {e}")
    else:
        st.info("Please generate data first to enable the storage option.")