import streamlit as st
import requests
import json
import pandas as pd
import time
import re
from typing import Dict, Any, Optional

# --- Configuration ---
FASTAPI_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Matrix AI - Synthetic Data Generator",
    page_icon="🤖",
    layout="wide"
)

# Simple Matrix Theme - Clean Black and Green
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap');
    
    /* Main App Background - Pure Black */
    .stApp {
        background-color: #000000;
        color: #00ff00;
        font-family: 'Courier Prime', monospace;
    }
    
    /* Main Content Container */
    .main .block-container {
        background-color: #000000;
        padding: 2rem;
        max-width: 1200px;
    }
    
    /* Headers - Matrix Green */
    h1, h2, h3, h4, h5, h6 {
        color: #00ff00 !important;
        font-family: 'Courier Prime', monospace !important;
        font-weight: 700 !important;
    }
    
    h1 {
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 2rem !important;
        text-shadow: 0 0 10px #00ff00;
    }
    
    /* Text and Labels */
    .stMarkdown, .stText, p, div, span, label {
        color: #00ff00 !important;
        font-family: 'Courier Prime', monospace !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #001100 !important;
        color: #00ff00 !important;
        border: 1px solid #00ff00 !important;
        font-family: 'Courier Prime', monospace !important;
    }
    
    .stTextInput > label,
    .stTextArea > label,
    .stNumberInput > label,
    .stSelectbox > label {
        color: #00ff00 !important;
        font-family: 'Courier Prime', monospace !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #001100;
        color: #00ff00;
        border: 2px solid #00ff00;
        font-family: 'Courier Prime', monospace;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #00ff00;
        color: #000000;
    }
    
    /* Primary Button */
    div.stButton > button[kind="primary"] {
        background-color: #00ff00;
        color: #000000;
        border: 2px solid #00ff00;
    }
    
    div.stButton > button[kind="primary"]:hover {
        background-color: #00cc00;
        border-color: #00cc00;
    }
    
    /* Secondary Button */
    div.stButton > button[kind="secondary"] {
        background-color: #003300;
        color: #00ff00;
        border: 2px solid #00ff00;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background-color: #00ff00;
    }
    
    /* Containers and Borders */
    .stContainer, div[data-testid="stContainer"] {
        border: 1px solid #00ff00;
        background-color: #001100;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #00ff00;
    }
    
    /* Sidebar */
    .css-1d391kg, .stSidebar {
        background-color: #000000 !important;
        border-right: 2px solid #00ff00;
    }
    
    /* Fix sidebar content */
    .stSidebar > div {
        background-color: #000000 !important;
    }
    
    /* Fix top header area */
    .stApp > header {
        background-color: #000000 !important;
    }
    
    /* Fix main container top area */
    .main > div {
        background-color: #000000 !important;
    }
    
    /* Fix any remaining white areas */
    div[data-testid="stSidebar"] {
        background-color: #000000 !important;
    }
    
    div[data-testid="stHeader"] {
        background-color: #000000 !important;
    }
    
    /* Fix selectbox dropdown */
    .stSelectbox > div > div > div {
        background-color: #001100 !important;
        color: #00ff00 !important;
    }
    
    /* Fix slider components */
    .stSlider > div > div > div {
        color: #00ff00 !important;
    }
    
    /* Fix checkbox */
    .stCheckbox > label {
        color: #00ff00 !important;
    }
    
    /* Fix any remaining white backgrounds */
    .element-container, .stMarkdown, .stText {
        background-color: transparent !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: #001100;
        border: 1px solid #00ff00;
        color: #00ff00;
    }
    
    .stError {
        background-color: #110000;
        border: 1px solid #ff0000;
        color: #ff0000;
    }
    
    .stWarning {
        background-color: #111100;
        border: 1px solid #ffff00;
        color: #ffff00;
    }
    
    .stInfo {
        background-color: #001111;
        border: 1px solid #00ffff;
        color: #00ffff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000;
        border-bottom: 2px solid #00ff00;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #00ff00;
        font-family: 'Courier Prime', monospace;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #001100;
        border-bottom: 2px solid #00ff00;
    }
    
    /* JSON Display */
    .stJson {
        background-color: #001100;
        border: 1px solid #00ff00;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #001100;
        border: 1px solid #00ff00;
        padding: 1rem;
    }
    
    [data-testid="metric-container"] > div {
        color: #00ff00;
        font-family: 'Courier Prime', monospace;
    }
    
    /* Remove default Streamlit styling */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp > .main > .block-container {
        padding-top: 1rem;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Simple Matrix Title
st.markdown("""
    <h1 style="text-align: center; color: #00ff00; font-family: 'Courier Prime', monospace; 
               text-shadow: 0 0 10px #00ff00; margin-bottom: 2rem;">
        MATRIX AI - SYNTHETIC DATA GENERATOR
    </h1>
    """, unsafe_allow_html=True)

st.markdown("**High-Performance AI-Powered Data Generation using Gemini and SDV**")
st.markdown("---")

# --- Session State Management ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'ai_design_output' not in st.session_state:
    st.session_state.ai_design_output = None
if 'synthetic_data_metadata' not in st.session_state:
    st.session_state.synthetic_data_metadata = None
if 'num_records' not in st.session_state:
    st.session_state.num_records = 1000
if 'email' not in st.session_state:
    st.session_state.email = "pavan.thalla@latentview.com" 
if 'synthesis_status' not in st.session_state:
    st.session_state.synthesis_status = "Not Started" 
if 'data_description' not in st.session_state:
    st.session_state.data_description = "A multi-table subscription database with products, offers, subscriptions, and entitlements."
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 1000
if 'use_fast_synthesizer' not in st.session_state:
    st.session_state.use_fast_synthesizer = True

# --- Sidebar for Performance Settings ---
with st.sidebar:
    st.markdown("### PERFORMANCE SETTINGS")
    
    st.session_state.batch_size = st.slider(
        "Batch Size", 
        min_value=100, 
        max_value=5000, 
        value=st.session_state.batch_size,
        step=100,
        help="Larger batches = faster processing but more memory usage"
    )
    
    st.session_state.use_fast_synthesizer = st.checkbox(
        "Use Fast Synthesizer", 
        value=st.session_state.use_fast_synthesizer,
        help="Enable optimized algorithms for large datasets"
    )
    
    st.markdown("### SYSTEM STATUS")
    
    if st.button("Refresh Status", use_container_width=True):
        st.rerun()

# --- Helper Function for API Calls ---
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
        st.error(f"API ERROR ({e.response.status_code}): {e.response.json().get('detail', e.response.text)}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"CONNECTION ERROR: {e}")
        st.info("Ensure your FastAPI server is running on http://localhost:8000")
        return None

# --- Progress Monitoring Function ---
def get_progress_info():
    try:
        response = call_api("GET", f"{FASTAPI_URL}/progress")
        if response:
            return response
    except:
        pass
    return {
        "status": "idle",
        "current_step": "No active process",
        "progress_percent": 0,
        "records_generated": 0,
        "error_message": None
    }

# Display current progress in sidebar
progress_info = get_progress_info()
with st.sidebar:
    if progress_info["status"] == "processing":
        st.markdown("### ACTIVE SYNTHESIS")
        st.progress(progress_info["progress_percent"] / 100)
        st.markdown(f"**Step:** {progress_info['current_step']}")
        st.markdown(f"**Progress:** {progress_info['progress_percent']}%")
        if progress_info["records_generated"] > 0:
            st.markdown(f"**Records:** {progress_info['records_generated']:,}")
    elif progress_info["status"] == "complete":
        st.markdown("### SYNTHESIS COMPLETE")
        st.success(f"Generated {progress_info['records_generated']:,} records")
    elif progress_info["status"] == "error":
        st.markdown("### ERROR DETECTED")
        st.error(progress_info["error_message"])
    else:
        st.markdown("### SYSTEM IDLE")
        st.info("Ready for new synthesis task")

st.markdown("---")

# --- Step 1: Design Schema ---
st.header("STEP 1: DESIGN SCHEMA")
st.caption("Use natural language to describe your required tables and relationships.")

with st.container():
    col_desc, col_num = st.columns([3, 1])

    with col_desc:
        st.session_state.data_description = st.text_area(
            "Schema Description:",
            st.session_state.data_description,
            height=150,
            key="main_data_description_input"
        )
    
    with col_num:
        st.session_state.num_records = st.number_input(
            "Target Records:",
            min_value=1,
            max_value=1000000,
            value=st.session_state.num_records,
            step=100
        )
        
        estimated_time = max(1, st.session_state.num_records / 1000 * 0.5)
        st.markdown(f"**Est. Time:** ~{estimated_time:.1f} min")
        
        if st.button("GENERATE SCHEMA", use_container_width=True, type="primary"):
            if st.session_state.data_description:
                existing_metadata = None
                if st.session_state.ai_design_output:
                    existing_metadata = st.session_state.ai_design_output.get("metadata_dict")
                    st.toast("Refinement requested: Sending context to Gemini.")
                    
                payload = {
                    "data_description": st.session_state.data_description,
                    "num_records": st.session_state.num_records,
                    "existing_metadata": existing_metadata
                }
                
                with st.spinner("Calling Gemini to design schema and generate seed data..."):
                    response = call_api("POST", f"{FASTAPI_URL}/design", payload)

                if response and response.get("status") == "review_required":
                    st.session_state.ai_design_output = {
                        "metadata_dict": response.get("metadata_preview"),
                        "seed_tables_dict": response.get("seed_data_preview"),
                    }
                    st.session_state.step = 2
                    st.session_state.synthesis_status = "Not Started"
                    st.success("SCHEMA DESIGN COMPLETE. Review the output below.")
                    st.rerun()

st.markdown("---")

# --- Step 2: Review and Approve ---
if st.session_state.step >= 2 and st.session_state.ai_design_output:
    st.header("STEP 2: REVIEW SCHEMA")
    st.caption("Confirm the AI-generated metadata and seed data before synthesis.")
    
    ai_output = st.session_state.ai_design_output
    
    col_meta, col_seed = st.columns([1, 1])

    with col_meta:
        with st.container():
            st.subheader("METADATA STRUCTURE")
            st.json(ai_output["metadata_dict"], expanded=False)

    with col_seed:
        with st.container():
            st.subheader("SEED DATA SAMPLE")
            st.markdown("This sample trains the SDV model.")
            
            table_names = list(ai_output["seed_tables_dict"].keys())
            tabs = st.tabs(table_names)
            
            for i, table_name in enumerate(table_names):
                with tabs[i]:
                    df_seed = pd.DataFrame.from_records(ai_output["seed_tables_dict"][table_name])
                    st.dataframe(df_seed, use_container_width=True)

    st.subheader("LAUNCH SYNTHESIS")
    
    col_email, col_settings, col_start_btn = st.columns([2, 1, 1])

    with col_email:
        st.session_state.email = st.text_input(
            "Email for notifications:", 
            value=st.session_state.email, 
            key="user_email_input_2"
        )

    with col_settings:
        st.markdown("**Performance Settings:**")
        st.markdown(f"Batch Size: {st.session_state.batch_size:,}")
        st.markdown(f"Fast Mode: {'Yes' if st.session_state.use_fast_synthesizer else 'No'}")

    with col_start_btn:
        st.markdown("<br>", unsafe_allow_html=True) 
        if st.button("APPROVE & START", use_container_width=True, type="primary"):
            if not re.match(r"[^@]+@[^@]+\.[^@]+", st.session_state.email):
                st.error("Please enter a valid email address.")
            else:
                st.session_state.step = 3
                st.rerun()

st.markdown("---")

# --- Step 3: Synthesize ---
if st.session_state.step == 3:
    st.header("STEP 3: SYNTHESIS EXECUTION")
    
    if st.session_state.synthesis_status == "Not Started":
        if st.button(f"BEGIN SYNTHESIS FOR {st.session_state.num_records:,} RECORDS", use_container_width=True, type="secondary"):
            payload = {
                "num_records": st.session_state.num_records,
                "metadata_dict": st.session_state.ai_design_output["metadata_dict"],
                "seed_tables_dict": st.session_state.ai_design_output["seed_tables_dict"],
                "user_email": st.session_state.email,
                "batch_size": st.session_state.batch_size,
                "use_fast_synthesizer": st.session_state.use_fast_synthesizer
            }
            
            with st.spinner("Initiating optimized synthesis task..."):
                response = call_api("POST", f"{FASTAPI_URL}/synthesize", payload)

            if response and response.get("status") == "processing_started":
                st.session_state.synthesis_status = "Processing"
                st.success(f"OPTIMIZED SYNTHESIS STARTED! Notification will be sent to {st.session_state.email}")
                st.info("The process is running in the background. Proceed to Step 4 to check status.")
            
            st.rerun()

    elif st.session_state.synthesis_status == "Processing":
        st.warning(f"Synthesis is running in the background. You will receive an email at {st.session_state.email} when finished.")
        
        current_progress = get_progress_info()
        if current_progress["status"] == "processing":
            st.progress(current_progress["progress_percent"] / 100)
            st.markdown(f"**Current Step:** {current_progress['current_step']}")
            st.markdown(f"**Progress:** {current_progress['progress_percent']}%")
            if current_progress["records_generated"] > 0:
                st.markdown(f"**Records Generated:** {current_progress['records_generated']:,}")

st.markdown("---")

# --- Step 4: Finalize ---
st.header("STEP 4: DATA FINALIZATION")

if st.button("CHECK IF DATA IS READY", use_container_width=True, type="secondary"):
    ready_response = call_api("GET", f"{FASTAPI_URL}/sample", params={"sample_size": 20})
    
    if ready_response and ready_response.get("status") == "success":
        st.session_state.synthesis_status = "Complete"
        st.session_state.step = 4
        st.success("DATA IS READY! Review samples and store the data.")
        st.rerun()
    elif st.session_state.synthesis_status == "Processing":
        st.info("Data is still generating in the background. Please wait for the email notification.")
    else:
        st.error("Data is not yet ready or the background process failed. Check the server logs.")

if st.session_state.synthesis_status == "Complete":
    st.subheader("SYNTHESIS METRICS")
    
    sample_response = call_api("GET", f"{FASTAPI_URL}/sample", params={"sample_size": 20})
    
    if sample_response and sample_response.get("metadata"):
        metadata = sample_response["metadata"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Records", 
                f"{metadata.get('total_records_generated', 0):,}",
                delta="Generated"
            )
        
        with col2:
            generation_time = metadata.get('generation_time_seconds', 0)
            st.metric(
                "Generation Time", 
                f"{generation_time:.1f}s",
                delta=f"{metadata.get('total_records_generated', 0) / max(generation_time, 1):.0f} rec/sec"
            )
        
        with col3:
            st.metric(
                "Tables Created", 
                len(metadata.get('tables', {})),
                delta="Tables"
            )
        
        with col4:
            avg_columns = sum(len(table.get('columns', [])) for table in metadata.get('tables', {}).values()) / max(len(metadata.get('tables', {})), 1)
            st.metric(
                "Avg Columns", 
                f"{avg_columns:.0f}",
                delta="Per Table"
            )
    
    st.subheader("SAMPLE DATA PREVIEW")
    
    if sample_response and sample_response.get("all_samples"):
        all_samples = sample_response["all_samples"]
        tabs = st.tabs(list(all_samples.keys()))
        
        for i, table_name in enumerate(all_samples.keys()):
            with tabs[i]:
                st.subheader(f"Sample from {table_name}")
                df_sample = pd.DataFrame(all_samples[table_name])
                st.dataframe(df_sample, use_container_width=True)
                
                col_stats1, col_stats2 = st.columns(2)
                with col_stats1:
                    st.markdown(f"**Rows:** {len(df_sample):,}")
                    st.markdown(f"**Columns:** {len(df_sample.columns)}")
                with col_stats2:
                    st.markdown(f"**Memory Usage:** {df_sample.memory_usage(deep=True).sum() / 1024:.1f} KB")
                    st.markdown(f"**Data Types:** {len(df_sample.dtypes.unique())} unique")
        
        st.subheader("DATA DOWNLOAD")
        st.info("Download your generated synthetic data as a ZIP package containing all CSV files and metadata.")

        col_download1, col_download2 = st.columns([3, 1])
        
        with col_download1:
            st.markdown("**Download Package Includes:**")
            st.markdown("- All synthetic data tables as CSV files")
            st.markdown("- Metadata schema (JSON format)")
            st.markdown("- Seed data used for training (JSON format)")
            st.markdown("- Generation summary with statistics")
            st.markdown("- Timestamped files for version control")
        
        with col_download2:
            if st.button("DOWNLOAD ALL DATA", type="primary", use_container_width=True):
                with st.spinner("Preparing download package..."):
                    try:
                        # Call the download endpoint
                        download_response = requests.get(f"{FASTAPI_URL}/download", timeout=300)
                        download_response.raise_for_status()
                        
                        # Generate filename with timestamp
                        from datetime import datetime
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"matrix_ai_synthetic_data_{timestamp}.zip"
                        
                        # Create download button
                        st.download_button(
                            label="📥 CLICK TO DOWNLOAD ZIP FILE",
                            data=download_response.content,
                            file_name=filename,
                            mime="application/zip",
                            use_container_width=True
                        )
                        
                        st.balloons()
                        st.success("Download package ready! Click the button above to save your data.")
                        
                    except requests.exceptions.RequestException as e:
                        st.error(f"Download failed: {e}")
                        st.info("Ensure your FastAPI server is running and data synthesis is complete.")
                    except Exception as e:
                        st.error(f"Unexpected error during download: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("**Matrix AI v6.0 - Optimized Synthetic Data Generation**")
st.markdown("Powered by Gemini AI & SDV | Enhanced Performance")
