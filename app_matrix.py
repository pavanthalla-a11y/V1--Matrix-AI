import streamlit as st
import requests
import json
import pandas as pd
import time
import re
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration & Aesthetics ---
FASTAPI_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Matrix AI - Synthetic Data Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Matrix-style CSS with animated background
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    /* Matrix Rain Animation */
    @keyframes matrix-rain {
        0% { transform: translateY(-100vh); opacity: 1; }
        100% { transform: translateY(100vh); opacity: 0; }
    }
    
    @keyframes matrix-glow {
        0%, 100% { text-shadow: 0 0 5px #00ff41, 0 0 10px #00ff41, 0 0 15px #00ff41; }
        50% { text-shadow: 0 0 10px #00ff41, 0 0 20px #00ff41, 0 0 30px #00ff41; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Main Background - Dark Matrix Theme */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #00ff41;
        font-family: 'Orbitron', monospace;
    }
    
    /* Matrix Rain Effect Background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #00ff41, transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(0,255,65,0.3), transparent),
            radial-gradient(1px 1px at 90px 40px, #00ff41, transparent),
            radial-gradient(1px 1px at 130px 80px, rgba(0,255,65,0.3), transparent),
            radial-gradient(2px 2px at 160px 30px, #00ff41, transparent);
        background-repeat: repeat;
        background-size: 200px 100px;
        animation: matrix-rain 20s linear infinite;
        opacity: 0.1;
        z-index: -1;
        pointer-events: none;
    }
    
    /* Main Content Styling */
    .main .block-container {
        background: rgba(13, 17, 23, 0.95);
        border: 1px solid #00ff41;
        border-radius: 10px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.3);
    }
    
    /* Headers with Matrix Glow */
    h1, h2, h3 {
        color: #00ff41 !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
        animation: matrix-glow 2s ease-in-out infinite alternate;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    h1 {
        font-size: 2.5rem !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    
    /* Matrix-style Containers */
    .stContainer, div[data-testid="stContainer"] {
        background: linear-gradient(145deg, rgba(0, 255, 65, 0.05), rgba(0, 255, 65, 0.02));
        border: 1px solid rgba(0, 255, 65, 0.3);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 
            inset 0 1px 0 rgba(0, 255, 65, 0.1),
            0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Matrix Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #003d0f, #00ff41);
        color: #000000 !important;
        border: 2px solid #00ff41;
        border-radius: 8px;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #00ff41, #00cc33);
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.6);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.4);
    }
    
    /* Primary Button (Bright Green) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #00ff41, #00cc33);
        animation: pulse 2s infinite;
    }
    
    /* Secondary Button (Darker Green) */
    div.stButton > button[kind="secondary"] {
        background: linear-gradient(45deg, #003d0f, #006622);
        color: #00ff41 !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background: rgba(0, 0, 0, 0.7) !important;
        color: #00ff41 !important;
        border: 1px solid #00ff41 !important;
        border-radius: 5px;
        font-family: 'Orbitron', monospace;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #00ff41 !important;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.5) !important;
    }
    
    /* Labels */
    .stTextInput > label,
    .stTextArea > label,
    .stNumberInput > label {
        color: #00ff41 !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 600 !important;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00ff41, #00cc33);
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #00ff41;
        border-radius: 5px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 0, 0, 0.3);
        border-bottom: 2px solid #00ff41;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #00ff41;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(0, 255, 65, 0.2);
        border-bottom: 2px solid #00ff41;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(0, 255, 65, 0.1);
        border: 1px solid #00ff41;
        color: #00ff41;
    }
    
    .stError {
        background: rgba(255, 0, 0, 0.1);
        border: 1px solid #ff0000;
        color: #ff4444;
    }
    
    .stWarning {
        background: rgba(255, 255, 0, 0.1);
        border: 1px solid #ffff00;
        color: #ffff44;
    }
    
    .stInfo {
        background: rgba(0, 255, 255, 0.1);
        border: 1px solid #00ffff;
        color: #44ffff;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #0d1117, #161b22);
        border-right: 2px solid #00ff41;
    }
    
    /* JSON Display */
    .stJson {
        background: rgba(0, 0, 0, 0.8);
        border: 1px solid #00ff41;
        border-radius: 5px;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: rgba(0, 255, 65, 0.05);
        border: 1px solid rgba(0, 255, 65, 0.3);
        border-radius: 8px;
        padding: 1rem;
    }
    
    [data-testid="metric-container"] > div {
        color: #00ff41;
        font-family: 'Orbitron', monospace;
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00ff41;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00cc33;
    }
    
    /* Matrix Code Effect for specific elements */
    .matrix-code {
        font-family: 'Courier New', monospace;
        color: #00ff41;
        text-shadow: 0 0 5px #00ff41;
        animation: matrix-glow 1.5s ease-in-out infinite alternate;
    }
    
    /* Loading Animation */
    .matrix-loading {
        display: inline-block;
        animation: pulse 1s infinite;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Matrix-style title with animated effect
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 3rem; color: #00ff41; font-family: 'Orbitron', monospace; 
                   text-shadow: 0 0 20px #00ff41; animation: matrix-glow 2s ease-in-out infinite alternate;">
            🤖 MATRIX AI
        </h1>
        <p style="color: #00ff41; font-family: 'Orbitron', monospace; font-size: 1.2rem; 
                  text-transform: uppercase; letter-spacing: 3px;">
            SYNTHETIC DATA GENERATOR
        </p>
        <div style="height: 2px; background: linear-gradient(90deg, transparent, #00ff41, transparent); 
                    margin: 1rem auto; width: 60%;"></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    """
    **HARNESS THE POWER OF GEMINI AND SDV** to design, verify, and generate complex, large-scale synthetic datasets with **OPTIMIZED PERFORMANCE**.
    """
)

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
    st.markdown("### ⚡ PERFORMANCE SETTINGS")
    
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
    
    st.markdown("### 📊 SYSTEM STATUS")
    
    # Real-time progress monitoring
    if st.button("🔄 Refresh Status", use_container_width=True):
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
        st.error(f"🚨 API ERROR ({e.response.status_code}): {e.response.json().get('detail', e.response.text)}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 CONNECTION ERROR: {e}")
        st.info("💡 Ensure your FastAPI server is running on http://localhost:8000")
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
        st.markdown("### 🔄 ACTIVE SYNTHESIS")
        st.progress(progress_info["progress_percent"] / 100)
        st.markdown(f"**Step:** {progress_info['current_step']}")
        st.markdown(f"**Progress:** {progress_info['progress_percent']}%")
        if progress_info["records_generated"] > 0:
            st.markdown(f"**Records:** {progress_info['records_generated']:,}")
    elif progress_info["status"] == "complete":
        st.markdown("### ✅ SYNTHESIS COMPLETE")
        st.success(f"Generated {progress_info['records_generated']:,} records")
    elif progress_info["status"] == "error":
        st.markdown("### ❌ ERROR DETECTED")
        st.error(progress_info["error_message"])
    else:
        st.markdown("### 💤 SYSTEM IDLE")
        st.info("Ready for new synthesis task")

st.markdown("---")

# --- Step 1: Design (Call Gemini) ---
st.markdown("## 🧠 NEURAL SCHEMA DESIGN")
st.caption("Use natural language to describe your required tables and relationships.")

with st.container():
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
            max_value=1000000,
            value=st.session_state.num_records,
            step=100
        )
        
        # Performance estimate
        estimated_time = max(1, st.session_state.num_records / 1000 * 0.5)  # Rough estimate
        st.markdown(f"**Est. Time:** ~{estimated_time:.1f} min")
        
        if st.button("🚀 GENERATE SCHEMA", use_container_width=True, type="primary"):
            if st.session_state.data_description:
                existing_metadata = None
                if st.session_state.ai_design_output:
                    existing_metadata = st.session_state.ai_design_output.get("metadata_dict")
                    st.toast("🔄 Refinement requested: Sending context to Gemini.", icon="💡")
                    
                payload = {
                    "data_description": st.session_state.data_description,
                    "num_records": st.session_state.num_records,
                    "existing_metadata": existing_metadata
                }
                
                with st.spinner("🤖 Calling Gemini to design/refine schema and generate seed data..."):
                    time.sleep(0.5) 
                    response = call_api("POST", f"{FASTAPI_URL}/design", payload)

                if response and response.get("status") == "review_required":
                    st.session_state.ai_design_output = {
                        "metadata_dict": response.get("metadata_preview"),
                        "seed_tables_dict": response.get("seed_data_preview"),
                    }
                    st.session_state.step = 2
                    st.session_state.synthesis_status = "Not Started"
                    st.success("✅ SCHEMA DESIGN COMPLETE. Review the updated output below.")
                    st.rerun()

st.markdown("---")

# --- Step 2: Review and Approve ---
if st.session_state.step >= 2 and st.session_state.ai_design_output:
    st.markdown("## 🔍 SCHEMA VERIFICATION")
    st.caption("Confirm the AI-generated metadata and seed data are accurate before starting synthesis.")
    
    ai_output = st.session_state.ai_design_output
    
    col_meta, col_seed = st.columns([1, 1])

    with col_meta:
        with st.container():
            st.subheader("📋 METADATA STRUCTURE")
            st.json(ai_output["metadata_dict"], expanded=False)

    with col_seed:
        with st.container():
            st.subheader("🌱 SEED DATA SAMPLE")
            st.markdown("This sample trains the SDV model.")
            
            table_names = list(ai_output["seed_tables_dict"].keys())
            tabs = st.tabs(table_names)
            
            for i, table_name in enumerate(table_names):
                with tabs[i]:
                    df_seed = pd.DataFrame.from_records(ai_output["seed_tables_dict"][table_name])
                    st.dataframe(df_seed, use_container_width=True)

    st.subheader("⚡ LAUNCH SYNTHESIS")
    
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
        st.markdown(f"Fast Mode: {'✅' if st.session_state.use_fast_synthesizer else '❌'}")

    with col_start_btn:
        st.markdown("<br>", unsafe_allow_html=True) 
        if st.button("🚀 APPROVE & START", use_container_width=True, type="primary"):
            if not re.match(r"[^@]+@[^@]+\.[^@]+", st.session_state.email):
                st.error("⚠️ Please enter a valid email address.")
            else:
                st.session_state.step = 3
                st.rerun()

st.markdown("---")

# --- Step 3: Synthesize (Start Background Task) ---
if st.session_state.step == 3:
    st.markdown("## ⚡ SYNTHESIS EXECUTION")
    
    if st.session_state.synthesis_status == "Not Started":
        if st.button(f"🔥 BEGIN SYNTHESIS FOR {st.session_state.num_records:,} RECORDS", use_container_width=True, type="secondary"):
            payload = {
                "num_records": st.session_state.num_records,
                "metadata_dict": st.session_state.ai_design_output["metadata_dict"],
                "seed_tables_dict": st.session_state.ai_design_output["seed_tables_dict"],
                "user_email": st.session_state.email,
                "batch_size": st.session_state.batch_size,
                "use_fast_synthesizer": st.session_state.use_fast_synthesizer
            }
            
            with st.spinner("🚀 Initiating optimized synthesis task..."):
                response = call_api("POST", f"{FASTAPI_URL}/synthesize", payload)

            if response and response.get("status") == "processing_started":
                st.session_state.synthesis_status = "Processing"
                st.success(f"✅ OPTIMIZED SYNTHESIS STARTED! Notification will be sent to **{st.session_state.email}**.")
                st.info("🔄 The process is running in the background. **Proceed to Step 4 to check status.**")
            
            st.rerun()

    elif st.session_state.synthesis_status == "Processing":
        st.warning(f"⚡ Synthesis is actively running in the background. You will receive an email at **{st.session_state.email}** when it finishes.")
        
        # Real-time progress display
        current_progress = get_progress_info()
        if current_progress["status"] == "processing":
            progress_bar = st.progress(current_progress["progress_percent"] / 100)
            st.markdown(f"**Current Step:** {current_progress['current_step']}")
            st.markdown(f"**Progress:** {current_progress['progress_percent']}%")
            if current_progress["records_generated"] > 0:
                st.markdown(f"**Records Generated:** {current_progress['records_generated']:,}")

st.markdown("---")

# --- Step 4: Finalize (Sample and Store) ---
st.markdown("## 💾 DATA FINALIZATION")

if st.button("🔍 CHECK IF DATA IS READY", use_container_width=True, type="secondary"):
    ready_response = call_api("GET", f"{FASTAPI_URL}/sample", params={"sample_size": 20})
    
    if ready_response and ready_response.get("status") == "success":
        st.session_state.synthesis_status = "Complete"
        st.session_state.step = 4
        st.success("🎉 DATA IS READY! Review samples and store the data.")
        st.rerun()
    elif st.session_state.synthesis_status == "Processing":
        st.info("⏳ Data is still generating in the background. Please wait for the email notification.")
    else:
        st.error("❌ Data is not yet ready or the background process failed. Check the server logs.")

if st.session_state.synthesis_status == "Complete":
    # Performance Metrics Display
    st.markdown("### 📊 SYNTHESIS METRICS")
    
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
    
    # Show Sample Data Section
    st.markdown("### 🔬 SAMPLE DATA PREVIEW")
    
    if sample_response and sample_response.get("all_samples"):
        all_samples = sample_response["all_samples"]
        tabs = st.tabs(list(all_samples.keys()))
        
        for i, table_name in enumerate(all_samples.keys()):
            with tabs[i]:
                st.subheader(f"📋 Sample from {table_name}")
                df_sample = pd.DataFrame(all_samples[table_name])
                st.dataframe(df_sample, use_container_width=True)
                
                # Table statistics
                col_stats1, col_stats2 = st.columns(2)
                with col_stats1:
                    st.markdown(f"**Rows:** {len(df_sample):,}")
                    st.markdown(f"**Columns:** {len(df_sample.columns)}")
                with col_stats2:
                    st.markdown(f"**Memory Usage:** {df_sample.memory_usage(deep=True).sum() / 1024:.1f} KB")
                    st.markdown(f"**Data Types:** {len(df_sample.dtypes.unique())} unique")
        
        # Store Data Section
        st.markdown("### ☁️ CLOUD STORAGE")
        st.warning("⚠️ Storing data will clear the current session. Confirm only when satisfied with the data quality.")

        col_store1, col_store2 = st.columns([3, 1])
        
        with col_store1:
            st.markdown("**Storage Details:**")
            st.markdown("- All tables will be uploaded to Google Cloud Storage")
            st.markdown("- Files will be timestamped for version control")
            st.markdown("- CSV format for maximum compatibility")
            st.markdown("- Session will be cleared after successful upload")
        
        with col_store2:
            if st.button("🚀 STORE ALL DATA TO GCS", type="primary", use_container_width=True):
                payload = {"confirm_storage": True}
                with st.spinner("☁️ Uploading files to Google Cloud Storage..."):
                    response = call_api("POST", f"{FASTAPI_URL}/store", payload)

                if response and response.get("status") == "storage_complete":
                    st.balloons()
                    st.success("🎉 ALL FILES UPLOADED SUCCESSFULLY! Session cleared. Starting a new workflow.")
                    
                    # Reset session state
                    for key in ['step', 'ai_design_output', 'synthetic_data_metadata', 'synthesis_status']:
                        if key in st.session_state:
                            if key == 'step':
                                st.session_state[key] = 1
                            elif key == 'synthesis_status':
                                st.session_state[key] = "Not Started"
                            else:
                                st.session_state[key] = None
                    
                    time.sleep(2)
                    st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #00ff41; font-family: 'Orbitron', monospace; 
                opacity: 0.7; font-size: 0.9rem; margin-top: 2rem;">
        <p>🤖 MATRIX AI v6.0 - OPTIMIZED SYNTHETIC DATA GENERATION</p>
        <p>Powered by Gemini AI & SDV | Enhanced Performance & Matrix UI</p>
    </div>
    """, unsafe_allow_html=True)
