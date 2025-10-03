import requests
import json
import time
import streamlit as st
from typing import Dict, Any

FASTAPI_URL = "http://localhost:8000/api/v1"

def call_api(method: str, url: str, payload: Dict[str, Any] = None, params: Dict[str, Any] = None, max_retries: int = 3):
    """
    DEMO-SAFE API calls with retry logic and better error handling.
    This prevents frontend crashes during live presentations.
    """
    for attempt in range(max_retries):
        try:
            # DEMO-SAFE: Use appropriate timeouts for different endpoints
            if "synthesize" in url:
                timeout = 900  # 15 minutes for synthesis
            elif "design" in url:
                timeout = 420  # 7 minutes for Gemini AI calls (design endpoint)
            else:
                timeout = 120   # 2 minutes for other endpoints (increased from 60)
            
            response = requests.request(
                method,
                url,
                json=payload,
                params=params,
                timeout=timeout
            )
            
            # DEMO SUCCESS: Handle successful responses
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 202:
                st.session_state.synthesis_status = "Processing"
                return {"status": "processing_started"}
            else:
                response.raise_for_status()
                
        except requests.exceptions.HTTPError as e:
            # DEMO-SAFE: Graceful handling of HTTP errors
            try:
                error_detail = e.response.json().get('detail', 'Unknown error')
            except (ValueError, json.JSONDecodeError):
                error_detail = e.response.text if e.response.text else 'Unknown error'
            
            if attempt < max_retries - 1:
                st.warning(f"API Error (attempt {attempt + 1}/{max_retries}): {error_detail}")
                time.sleep(2)  # Brief pause before retry
                continue
            else:
                st.error(f"🚨 API ERROR ({e.response.status_code}): {error_detail}")
                if "timeout" in error_detail.lower():
                    st.info("💡 **Demo Tip**: Large data generation is running in background. Check progress or try smaller dataset.")
                return None
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}). Retrying...")
                time.sleep(3)
                continue
            else:
                st.error("🚨 REQUEST TIMEOUT: Server is taking too long to respond")
                st.info("💡 **Demo Solution**: The process may be running in background. Check the progress section.")
                return None
                
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                st.warning(f"Connection failed (attempt {attempt + 1}/{max_retries}). Retrying...")
                time.sleep(2)
                continue
            else:
                st.error("🚨 CONNECTION ERROR: Cannot connect to FastAPI server")
                st.info("💡 **Demo Fix**: Ensure FastAPI server is running: `python main_v2.py`")
                return None
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                st.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(2)
                continue
            else:
                st.error(f"🚨 NETWORK ERROR: {str(e)}")
                st.info("💡 **Demo Troubleshooting**: Check network connection and server status")
                return None
    
    return None  # All retries failed

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
