
import streamlit as st

def show_stepper():
    steps = ["🎨 Design", "⚗️ Synthesize", "📊 Analyze", "📥 Download"]
    current_step = st.session_state.get('step', 1)

    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        with cols[i]:
            if i < current_step - 1:
                st.markdown(f"<div style='text-align: center;'><div style='margin: 0 auto; width: 40px; height: 40px; border-radius: 50%; background-color: var(--primary-color); color: white; display: flex; justify-content: center; align-items: center; font-size: 24px;'>✓</div><p style='font-weight: 500; color: var(--primary-color);'>{step}</p></div>", unsafe_allow_html=True)
            elif i == current_step - 1:
                st.markdown(f"<div style='text-align: center;'><div style='margin: 0 auto; width: 50px; height: 50px; border-radius: 50%; background-color: var(--primary-color); color: white; display: flex; justify-content: center; align-items: center; font-size: 30px; border: 4px solid #f0f4f9; box-shadow: 0 0 0 3px var(--primary-color);'>{step.split(' ')[0]}</div><p style='font-weight: 700; color: var(--primary-color); font-size: 1.1em;'>{step.split(' ')[1]}</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: center;'><div style='margin: 0 auto; width: 40px; height: 40px; border-radius: 50%; background-color: var(--border-color); color: var(--subtle-text-color); display: flex; justify-content: center; align-items: center; font-size: 24px;'>{step.split(' ')[0]}</div><p style='font-weight: 500; color: var(--subtle-text-color);'>{step.split(' ')[1]}</p></div>", unsafe_allow_html=True)
