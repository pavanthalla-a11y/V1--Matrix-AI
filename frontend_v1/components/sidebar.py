import streamlit as st
from api import get_progress_info

def show_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: var(--primary-color);'>Settings</h2>", unsafe_allow_html=True)



        with st.expander("System Status", expanded=True):
            if st.button("Refresh Status", use_container_width=True):
                st.rerun()

            progress_info = get_progress_info()
            if progress_info["status"] == "processing":
                st.markdown("<h5>Active Synthesis</h5>", unsafe_allow_html=True)
                st.progress(progress_info["progress_percent"] / 100)
                st.markdown(f"**Step:** {progress_info['current_step']}")
                st.markdown(f"**Progress:** {progress_info['progress_percent']}%" )
                if progress_info["records_generated"] > 0:
                    st.markdown(f"**Records:** {progress_info['records_generated']:,}")
            elif progress_info["status"] == "complete":
                st.markdown("<h5>Synthesis Complete</h5>", unsafe_allow_html=True)
                st.success(f"Generated {progress_info['records_generated']:,} records")
            elif progress_info["status"] == "error":
                st.markdown("<h5>Error Detected</h5>", unsafe_allow_html=True)
                st.error(progress_info["error_message"])
            else:
                st.markdown("<h5>System Idle</h5>", unsafe_allow_html=True)
                st.info("Ready for new synthesis task")

