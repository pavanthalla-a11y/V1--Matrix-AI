import streamlit as st
from ..api import get_progress_info

def show_sidebar():
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

        progress_info = get_progress_info()
        if progress_info["status"] == "processing":
            st.markdown("### ACTIVE SYNTHESIS")
            st.progress(progress_info["progress_percent"] / 100)
            st.markdown(f"**Step:** {progress_info['current_step']}")
            st.markdown(f"**Progress:** {progress_info['progress_percent']}%" )
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
