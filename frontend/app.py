import streamlit as st
from styles import apply_matrix_style
from components.sidebar import show_sidebar
from components.steps import show_step1, show_step2, show_step3, show_step4

def main():
    st.set_page_config(
        page_title="Matrix AI - Synthetic Data Generator",
        page_icon="🤖",
        layout="wide"
    )
    apply_matrix_style()

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

    st.markdown("""
        <h1 style="text-align: center; color: #00ff00; font-family: 'Courier Prime', monospace; 
                   text-shadow: 0 0 10px #00ff00; margin-bottom: 2rem;">
            MATRIX AI - SYNTHETIC DATA GENERATOR
        </h1>
        """, unsafe_allow_html=True)

    st.markdown("**High-Performance AI-Powered Data Generation using Gemini and SDV**")
    st.markdown("---")

    show_sidebar()

    show_step1()
    st.markdown("---")

    if st.session_state.step >= 2 and st.session_state.ai_design_output:
        show_step2()
        st.markdown("---")

    if st.session_state.step == 3:
        show_step3()
        st.markdown("---")

    show_step4()

    st.markdown("---")
    st.markdown("**Matrix AI v6.0 - Optimized Synthetic Data Generation**")
    st.markdown("Powered by Gemini AI & SDV | Enhanced Performance")

if __name__ == "__main__":
    main()