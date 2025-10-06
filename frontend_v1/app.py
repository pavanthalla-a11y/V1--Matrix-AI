
import streamlit as st
from styles import apply_matrix_style
from components.sidebar import show_sidebar
from components.stepper import show_stepper
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
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 2rem;">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 4H8V8H4V4Z" fill="var(--primary-color)"/>
                <path d="M4 10H8V14H4V10Z" fill="var(--primary-color)"/>
                <path d="M4 16H8V20H4V16Z" fill="var(--primary-color)"/>
                <path d="M10 4H14V8H10V4Z" fill="var(--primary-color)"/>
                <path d="M10 10H14V14H10V10Z" fill="var(--primary-color)"/>
                <path d="M10 16H14V20H10V16Z" fill="var(--primary-color)"/>
                <path d="M16 4H20V8H16V4Z" fill="var(--primary-color)"/>
                <path d="M16 10H20V14H16V10Z" fill="var(--primary-color)"/>
                <path d="M16 16H20V20H16V16Z" fill="var(--primary-color)"/>
            </svg>
            <h1 style="margin-left: 10px; color: var(--primary-color); font-family: var(--font-family); font-weight: 700;">
                MATRIX AI
            </h1>
        </div>
        """, unsafe_allow_html=True)

    show_stepper()

    show_sidebar()

    with st.container():
        st.markdown("<div class='step-container'>", unsafe_allow_html=True)
        if st.session_state.step == 1:
            show_step1()
        elif st.session_state.step == 2:
            show_step2()
        elif st.session_state.step == 3:
            show_step3()
        elif st.session_state.step == 4:
            show_step4()
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
