import threading
from typing import Dict, Any

_cache_lock = threading.Lock()
generated_data_cache: Dict[str, Any] = {
    "design_output": None,
    "synthetic_data": None,
    "num_records_target": 0,
    "progress": {
        "status": "idle",
        "current_step": "",
        "progress_percent": 0,
        "estimated_time_remaining": 0,
        "records_generated": 0,
        "error_message": None
    }
}

def get_cache():
    """Thread-safe cache getter"""
    with _cache_lock:
        return generated_data_cache.copy()

def set_cache(key: str, value: Any):
    """Thread-safe cache setter"""
    with _cache_lock:
        generated_data_cache[key] = value

def update_cache(updates: Dict[str, Any]):
    """Thread-safe cache update"""
    with _cache_lock:
        generated_data_cache.update(updates)

def update_progress(status: str, step: str, percent: int, records: int = 0, error: str = None):
    """Update progress tracking"""
    with _cache_lock:
        generated_data_cache["progress"].update({
            "status": status,
            "current_step": step,
            "progress_percent": percent,
            "records_generated": records,
            "error_message": error
        })
