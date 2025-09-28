import streamlit as st
import requests
import json
import pandas as pd
import time
import re
from typing import Dict, Any, Optional

# --- Configuration & Aesthetics ---
FASTAPI_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Matrix AI - Synthetic Data Generator",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS Injection for Light Cyan Background and Themed Styling
st.markdown("""
    <style>
    /* 1. Main Content Background Color (Light Cyan/Teal) */
    /* Targets the main content area for a unified colored background */
    .st-emotion-cache-z5fcl4 { /* This class targets the primary block container */
        background-color: #E0F7FA; /* Light Cyan/Aqua */
    }
    
    /* 2. Container Background Color (Slightly Darker for Section Contrast) */
    .stContainer {
        border: 1px solid #00BFA5; /* Teal border */
        border-radius: 8px;
        padding: 15px;
        background-color: #F0FFFF; /* White-ish background for content contrast inside sections */
        margin-bottom: 20px;
    }

    /* 3. Primary Button Styling (Deep Purple) */
    div.stButton > button[kind="primary"] {
        background-color: #6A1B9A; /* Deep Purple */
        border-color: #6A1B9A;
        color: white;
        font-weight: bold;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #7B1FA2;
        border-color: #7B1FA2;
    }
    
    /* 4. Secondary Button Styling (Teal/Green) */
    div.stButton > button[kind="secondary"] {
        background-color: #00897B; /* Teal Green */
        border-color: #00897B;
        color: white;
        font-weight: bold;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #00A693;
        border-color: #00A693;
    }
    
    /* 5. Header Visual Hierarchy (The vertical colored bar) */
    /* This targets the <h2> header elements */
    .st-emotion-cache-1dp5aa { 
        border-left: 6px solid #00BFA5; /* Bright Teal line beside the header */
        padding-left: 15px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* 6. Text Input/Text Area Background */
    /* Ensure the input fields themselves are light for readability */
    .stTextArea, .stTextInput {
        background-color: white;
    }

    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Matrix AI - Interactive Data Synthesis")
st.markdown(
    """
    **Use the power of Gemini and SDV** to design, verify, and generate complex, large-scale synthetic datasets.
    """
)
st.markdown("---")
# --- END Configuration & Aesthetics ---


# --- Session State Management (Kept Intact) ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'ai_design_output' not in st.session_state:
    st.session_state.ai_design_output = None
if 'synthetic_data_metadata' not in st.session_state:
    st.session_state.synthetic_data_metadata = None
if 'num_records' not in st.session_state:
    st.session_state.num_records = 10000
if 'email' not in st.session_state:
    st.session_state.email = "pavan.thalla@latentview.com" 
if 'synthesis_status' not in st.session_state:
    st.session_state.synthesis_status = "Not Started" 
if 'data_description' not in st.session_state:
    st.session_state.data_description = "A multi-table subscription database with products, offers, subscriptions, and entitlements."


# --- Helper Function for API Calls (Kept Intact) ---
def call_api(method: str, url: str, payload: Dict[str, Any] = None, params: Dict[str, Any] = None):
    try:
        response = requests.request(
            method,
            url,
            json=payload,
            params=params,
            timeout=600 
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 202:
            st.session_state.synthesis_status = "Processing"
            return {"status": "processing_started"}
        st.error(f"API Error ({e.response.status_code}): {e.response.json().get('detail', e.response.text)}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the API: {e}")
        st.info("Please ensure your FastAPI server is running on http://localhost:8000.")
        return None


# --- Step 1: Design (Call Gemini) ---
st.header(" Design and Refine Schema 📝")
st.caption("Use natural language to describe your required tables and relationships.")

with st.container(border=True):
    col_desc, col_num = st.columns([3, 1])

    with col_desc:
        st.session_state.data_description = st.text_area(
            "Schema Description or Refinement Instructions:",
            st.session_state.data_description,
            height=150,
            key="main_data_description_input"
        )
    
    with col_num:
        st.session_state.num_records = st.number_input(
            "Total Target Records:",
            min_value=1,
            value=st.session_state.num_records
        )
        
        # PRIMARY BUTTON COLOR: Deep Purple (via CSS injection)
        if st.button("Generate / Refine Schema", use_container_width=True, type="primary"):
            if st.session_state.data_description:
                existing_metadata = None
                if st.session_state.ai_design_output:
                    existing_metadata = st.session_state.ai_design_output.get("metadata_dict")
                    st.toast("Refinement requested: Sending context to Gemini.", icon="💡")
                    
                payload = {
                    "data_description": st.session_state.data_description,
                    "num_records": st.session_state.num_records,
                    "existing_metadata": existing_metadata
                }
                
                with st.spinner("Calling Gemini to design/refine schema and generate seed data..."):
                    time.sleep(0.5) 
                    response = call_api("POST", f"{FASTAPI_URL}/design", payload)

                if response and response.get("status") == "review_required":
                    st.session_state.ai_design_output = {
                        "metadata_dict": response.get("metadata_preview"),
                        "seed_tables_dict": response.get("seed_data_preview"),
                    }
                    st.session_state.step = 2
                    st.session_state.synthesis_status = "Not Started"
                    st.success("✅ Schema design complete. Review the updated output below.")
                    st.rerun()

st.markdown("---")

# --- Step 2: Review and Approve ---
if st.session_state.step >= 2 and st.session_state.ai_design_output:
    st.header(" Verify and Approve Design ✅")
    st.caption("Confirm the AI-generated metadata and seed data are accurate before starting the long synthesis process.")
    
    ai_output = st.session_state.ai_design_output
    
    col_meta, col_seed = st.columns([1, 1])

    with col_meta:
        with st.container(border=True):
            st.subheader("Metadata Structure")
            st.json(ai_output["metadata_dict"], expanded=False)

    with col_seed:
        with st.container(border=True):
            st.subheader("Seed Data Sample (20 Rows)")
            st.markdown("This sample trains the SDV model.")
            
            table_names = list(ai_output["seed_tables_dict"].keys())
            tabs = st.tabs(table_names)
            
            for i, table_name in enumerate(table_names):
                with tabs[i]:
                    df_seed = pd.DataFrame.from_records(ai_output["seed_tables_dict"][table_name])
                    st.dataframe(df_seed, use_container_width=True)

    st.subheader("Start Asynchronous Synthesis")
    
    col_email, col_start_btn = st.columns([2, 1])

    with col_email:
        st.session_state.email = st.text_input(
            "Email for background notification (Required):", 
            value=st.session_state.email, 
            key="user_email_input_2"
        )

    with col_start_btn:
        st.markdown("<br>", unsafe_allow_html=True) 
        # PRIMARY BUTTON COLOR: Deep Purple (via CSS injection)
        if st.button("Approve & Start Synthesis", use_container_width=True, type="primary"):
            if not re.match(r"[^@]+@[^@]+\.[^@]+", st.session_state.email):
                st.error("Please enter a valid email address.")
            else:
                st.session_state.step = 3
                st.rerun()

st.markdown("---")


# --- Step 3: Synthesize (Start Background Task) ---
if st.session_state.step == 3:
    st.header(" Run Synthesis Job ⏱️")
    
    if st.session_state.synthesis_status == "Not Started":
        # SECONDARY BUTTON COLOR: Teal/Green (via CSS injection)
        if st.button(f"Begin Synthesis for {st.session_state.num_records:,} Records", use_container_width=True, type="secondary"):
            payload = {
                "num_records": st.session_state.num_records,
                "metadata_dict": st.session_state.ai_design_output["metadata_dict"],
                "seed_tables_dict": st.session_state.ai_design_output["seed_tables_dict"],
                "user_email": st.session_state.email
            }
            
            with st.spinner("Initiating long-running synthesis task..."):
                response = call_api("POST", f"{FASTAPI_URL}/synthesize", payload)

            if response and response.get("status") == "processing_started":
                st.session_state.synthesis_status = "Processing"
                st.success(f"✅ Synthesis task started! Notification will be sent to **{st.session_state.email}**.")
                st.info("The process is running in the background. **Proceed to Step 4 to check status.**")
            
            st.rerun()

    elif st.session_state.synthesis_status == "Processing":
        st.warning(f"Synthesis is actively running in the background. You will receive an email at **{st.session_state.email}** when it finishes.")
        st.progress(50, text="Training SDV Model (Check email for completion)...") 


# --- Step 4: Finalize (Sample and Store) ---
st.header("Final Review and Storage 💾")

# SECONDARY BUTTON COLOR: Teal/Green (via CSS injection)
if st.button("Check if Data is Ready", use_container_width=True, type="secondary"):
    ready_response = call_api("GET", f"{FASTAPI_URL}/sample", params={"sample_size": 20})
    
    if ready_response and ready_response.get("status") == "success":
        st.session_state.synthesis_status = "Complete"
        st.session_state.step = 4
        st.success("🎉 Data is ready! Review samples and store the data.")
        st.rerun()
    elif st.session_state.synthesis_status == "Processing":
        st.info("Data is still generating in the background. Please wait for the email notification.")
    else:
        st.error("Data is not yet ready or the background process failed. Check the server logs.")

if st.session_state.synthesis_status == "Complete":
    # Show Sample Data Section (All samples 20 rows each)
    st.subheader(" Sample All Synthesized Data (20 Rows)")
    
    sample_response = call_api("GET", f"{FASTAPI_URL}/sample", params={"sample_size": 20})
    
    if sample_response and sample_response.get("all_samples"):
        all_samples = sample_response["all_samples"]
        tabs = st.tabs(list(all_samples.keys()))
        for i, table_name in enumerate(all_samples.keys()):
            with tabs[i]:
                st.subheader(f"Sample from {table_name}")
                st.dataframe(pd.DataFrame(all_samples[table_name]), use_container_width=True)
        
        # Store Data Section
        st.subheader(" Store to Google Cloud Storage")
        st.warning("Storing data will clear the current session. Confirm only when satisfied with the data quality.")

        # FINAL ACTION BUTTON COLOR: Deep Purple (via CSS injection)
        if st.button("Store All Data to GCS (Final Step)", type="primary", use_container_width=True):
            payload = {"confirm_storage": True}
            with st.spinner("Uploading files to Google Cloud Storage..."):
                response = call_api("POST", f"{FASTAPI_URL}/store", payload)

            if response and response.get("status") == "storage_complete":
                st.balloons()
                st.success("🚀 All files uploaded successfully! Session cleared. Starting a new workflow.")
                st.session_state.step = 1
                st.session_state.synthesis_status = "Not Started"
                st.rerun()