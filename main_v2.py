import pandas as pd
import numpy as np
import json
import traceback
import os
import google.auth
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from typing import Dict, Any, List, Optional
import time
import uuid
import asyncio
import threading
from starlette.status import HTTP_202_ACCEPTED
import logging
from concurrent.futures import ThreadPoolExecutor
import gc
import re
from datetime import datetime
import zipfile
import io
import tempfile

# --- SDV IMPORTS ---
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.multi_table import HMASynthesizer

# --- GOOGLE CLOUD IMPORTS ---
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Import configuration variables from your config file
from config import GCP_PROJECT_ID, GCS_BUCKET_NAME, GCP_LOCATION, GEMINI_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Matrix AI - Optimized Synthetic Data Generator",
    description="High-performance AI-powered data generation using Gemini and SDV.",
    version="6.1.0"
)

# --- Thread-safe Cache Structure with Progress Tracking ---
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

def _get_cache():
    """Thread-safe cache getter"""
    with _cache_lock:
        return generated_data_cache.copy()

def _set_cache(key: str, value: Any):
    """Thread-safe cache setter"""
    with _cache_lock:
        generated_data_cache[key] = value

def _update_cache(updates: Dict[str, Any]):
    """Thread-safe cache update"""
    with _cache_lock:
        generated_data_cache.update(updates)

def _update_progress(status: str, step: str, percent: int, records: int = 0, error: str = None):
    """Update progress tracking"""
    with _cache_lock:
        generated_data_cache["progress"].update({
            "status": status,
            "current_step": step,
            "progress_percent": percent,
            "records_generated": records,
            "error_message": error
        })

# --- Pydantic Models for API Requests ---
class DesignRequest(BaseModel):
    data_description: str
    num_records: int
    existing_metadata: Optional[Dict[str, Any]] = {} 

class SynthesizeRequest(BaseModel):
    num_records: int
    metadata_dict: Dict[str, Any]
    seed_tables_dict: Dict[str, Any]
    user_email: EmailStr
    batch_size: Optional[int] = 1000  # New: batch processing
    use_fast_synthesizer: Optional[bool] = True  # New: faster algorithms

class StoreRequest(BaseModel):
    confirm_storage: bool

# --- Core Logic Functions ---

def notify_user_by_email(email: str, status: str, details: str):
    """[CRITICAL PLACEHOLDER] - Simulates the email notification service."""
    logger.info(f"EMAIL NOTIFICATION - TO: {email}, STATUS: {status}, DETAILS: {details}")

def setup_google_auth():
    """Setup Google Cloud authentication with proper error handling"""
    try:
        # Try to get default credentials
        credentials, project = google.auth.default()
        
        # Set quota project if not already set
        if hasattr(credentials, 'quota_project_id') and not credentials.quota_project_id:
            credentials = credentials.with_quota_project(GCP_PROJECT_ID)
        
        return credentials, project
    except Exception as e:
        logger.error(f"Authentication setup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {e}")

def validate_and_fix_json_response(response_text: str) -> Dict[str, Any]:
    """Validate and fix common JSON issues in AI responses"""
    try:
        # First, try to parse as-is
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parse failed: {e}. Attempting to fix...")
        
        # Common fixes for AI-generated JSON
        fixed_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if fixed_text.startswith('```json'):
            fixed_text = fixed_text[7:]
        if fixed_text.endswith('```'):
            fixed_text = fixed_text[:-3]
        
        # Remove any leading/trailing whitespace
        fixed_text = fixed_text.strip()
        
        # Fix common issues with trailing commas before closing brackets/braces
        fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)
        
        # Fix missing commas between object properties
        fixed_text = re.sub(r'"\s*\n\s*"', r'",\n"', fixed_text)
        fixed_text = re.sub(r'}\s*\n\s*"', r'},\n"', fixed_text)
        fixed_text = re.sub(r']\s*\n\s*"', r'],\n"', fixed_text)
        
        # Fix missing quotes around unquoted keys
        fixed_text = re.sub(r'(\w+)(\s*:)', r'"\1"\2', fixed_text)
        
        # Fix single quotes to double quotes
        fixed_text = fixed_text.replace("'", '"')
        
        # Fix common datetime format issues in JSON strings
        fixed_text = re.sub(r'"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"', r'"\1"', fixed_text)
        
        # Fix corrupted datetime format strings (common AI issue)
        fixed_text = re.sub(r'"%Y-%m-%d %"H":%"M":%S"', r'"%Y-%m-%d %H:%M:%S"', fixed_text)
        fixed_text = re.sub(r'"%Y-%m-%d %H:%M:%S"', r'"%Y-%m-%d %H:%M:%S"', fixed_text)
        
        # Fix other common format string corruptions
        fixed_text = re.sub(r'"%([YmdHMS])"', r'%\1', fixed_text)
        fixed_text = re.sub(r'"([^"]*%[YmdHMS][^"]*)"([^,}\]])', r'"\1"\2', fixed_text)
        
        # Try multiple parsing attempts with different fixes
        for attempt in range(3):
            try:
                return json.loads(fixed_text)
            except json.JSONDecodeError as e2:
                if attempt == 0:
                    # Attempt 1: Fix missing commas more aggressively
                    fixed_text = re.sub(r'(\d+|"[^"]*"|\]|\})\s*\n\s*("|\{|\[)', r'\1,\n\2', fixed_text)
                elif attempt == 1:
                    # Attempt 2: Try to extract just the JSON object part
                    json_start = fixed_text.find('{')
                    json_end = fixed_text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        fixed_text = fixed_text[json_start:json_end]
                else:
                    # Final attempt failed
                    logger.error(f"All JSON fix attempts failed: {e2}")
                    logger.error(f"Problematic JSON around character {e2.pos}: {fixed_text[max(0, e2.pos-50):e2.pos+50]}")
                    
                    # Try to create a minimal valid response as fallback
                    fallback_response = {
                        "metadata_dict": {
                            "tables": {
                                "data_table": {
                                    "columns": {
                                        "id": {"sdtype": "id"},
                                        "value": {"sdtype": "categorical"},
                                        "created_at": {"sdtype": "datetime", "datetime_format": "%Y-%m-%d %H:%M:%S"}
                                    },
                                    "primary_key": "id"
                                }
                            },
                            "relationships": []
                        },
                        "seed_tables_dict": {
                            "data_table": [
                                {"id": 1, "value": "Sample Data", "created_at": "2023-01-01 12:00:00"},
                                {"id": 2, "value": "Test Data", "created_at": "2023-01-02 12:00:00"}
                            ]
                        }
                    }
                    logger.warning("Using fallback minimal schema due to JSON parsing failure")
                    return fallback_response
        
        # This should never be reached, but just in case
        raise HTTPException(
            status_code=500, 
            detail=f"AI generated invalid JSON that could not be fixed. Parse error: {str(e)}"
        )

def call_ai_agent(data_description: str, existing_metadata_json: str = None) -> Dict[str, Any]:
    """
    Calls the Gemini API to get the metadata schema and seed data.
    """
    try:
        credentials, project = google.auth.default()
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION, credentials=credentials)
        
        model = GenerativeModel(
            GEMINI_MODEL,
            generation_config={"response_mime_type": "application/json"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Vertex AI: {e}")

    prompt = f"""
    You are a professional data architect. Generate a single JSON object with two top-level keys: 'metadata_dict' and 'seed_tables_dict'.

    **Core Request:** "{data_description}"
    
    {'**Refinement/Modification Instruction:** Based on the user\'s new description, modify the schema and seed data. Here is the existing metadata: ' + existing_metadata_json if existing_metadata_json else ''}

    **CRITICAL INSTRUCTIONS (SDV Modern Format):**
    1. The `metadata_dict` must have "tables" and "relationships" keys. 
    **Relationships:** - Use SDV format: "parent_table_name", "child_table_name", "parent_primary_key", "child_foreign_key"
    2. **PRIMARY KEY ENFORCEMENT:** For every table, you MUST define a "primary_key" field.
    3. **DATA VALUE CONSTRAINTS:**
        a. **DO NOT USE PLACEHOLDERS.** All date, time, and numerical fields in the `seed_tables_dict` MUST contain SPECIFIC, VALID DATA.
        b. All columns defined as **"datetime" MUST use the full, consistent timestamp format: %Y-%m-%d %H:%M:%S**. 
        c. Specifically, **NEVER** output the strings '(format)', 'YYYY', or 'HH:MM:SS' inside any data value.
        d. **CRITICAL:** For datetime format strings, use EXACTLY: "datetime_format": "%Y-%m-%d %H:%M:%S" - NO extra quotes around individual format codes.
    4. The `seed_tables_dict` must contain 20 realistic data rows for each table, ensuring all foreign key relationships are valid.
    5. **Output:** Return ONLY the complete JSON object.
    """

    logger.info("Generating/Refining schema with Gemini...")
    try:
        response = model.generate_content(prompt)
        ai_output = json.loads(response.text)
        
        if "metadata_dict" not in ai_output or "seed_tables_dict" not in ai_output:
             raise ValueError("AI output missing required top-level keys.")
        
        return ai_output
    except Exception as e:
        logger.error(f"ERROR during AI call: {e}")
        raise HTTPException(status_code=500, detail=f"AI agent failed to generate a valid schema: {e}")

def clean_seed_data(seed_tables_dict: Dict[str, List[Dict[str, Any]]], metadata_dict: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Cleans seed data by removing AI placeholder text/invalid dates from datetime columns.
    Enhanced with better error handling and validation.
    """
    logger.info("Pre-processing seed data: Robustly removing AI placeholders...")
    cleaned_seed_tables = {}

    REPLACEMENT_DATE = '2020-01-01 00:00:00'
    
    # Get all datetime columns for each table
    datetime_columns_by_table = {}
    
    # Handle different metadata structures
    tables_data = metadata_dict.get('tables', {})
    if isinstance(tables_data, list):
        # If tables is a list, convert to dict format
        tables_dict = {}
        for table_info in tables_data:
            if isinstance(table_info, dict) and 'name' in table_info:
                tables_dict[table_info['name']] = table_info
        tables_data = tables_dict
    
    for table_name, table_meta in tables_data.items():
        datetime_cols = []
        columns_data = table_meta.get('columns', {})
        
        # Handle different column structures
        if isinstance(columns_data, dict):
            for col_name, col_data in columns_data.items():
                if isinstance(col_data, dict) and col_data.get('sdtype') == 'datetime':
                    datetime_cols.append(col_name)
        elif isinstance(columns_data, list):
            for col_info in columns_data:
                if isinstance(col_info, dict) and col_info.get('sdtype') == 'datetime':
                    datetime_cols.append(col_info.get('name', ''))
        
        datetime_columns_by_table[table_name] = datetime_cols

    for table_name, data_records in seed_tables_dict.items():
        if not data_records: 
            cleaned_seed_tables[table_name] = []
            continue
        
        # Convert to DataFrame for easier manipulation
        cleaned_records = []
        datetime_cols = datetime_columns_by_table.get(table_name, [])
        
        for record_idx, record in enumerate(data_records):
            try:
                cleaned_record = record.copy()
                
                # Clean datetime columns
                for col in datetime_cols:
                    if col in cleaned_record:
                        value = str(cleaned_record[col])
                        
                        # Check for common AI placeholders and invalid formats
                        placeholder_patterns = [
                            '(format)', 'YYYY', 'MM', 'DD', 'HH:MM:SS', 'Time', 
                            '(value)', 'null', 'None', 'placeholder', 'example',
                            'sample', 'format', 'timestamp'
                        ]
                        
                        if any(placeholder.lower() in value.lower() for placeholder in placeholder_patterns):
                            cleaned_record[col] = REPLACEMENT_DATE
                            logger.info(f"Replaced placeholder '{value}' with '{REPLACEMENT_DATE}' in table '{table_name}', column '{col}'")
                        else:
                            # Try to parse the date to ensure it's valid
                            try:
                                parsed_date = pd.to_datetime(value)
                                cleaned_record[col] = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                cleaned_record[col] = REPLACEMENT_DATE
                                logger.info(f"Replaced invalid date '{value}' with '{REPLACEMENT_DATE}' in table '{table_name}', column '{col}'")
                
                cleaned_records.append(cleaned_record)
                
            except Exception as e:
                logger.warning(f"Error processing record {record_idx} in table '{table_name}': {e}")
                # Skip problematic records rather than failing entirely
                continue
        
        logger.info(f"Table '{table_name}': Processed {len(cleaned_records)} records, cleaned {len(datetime_cols)} datetime columns")
        cleaned_seed_tables[table_name] = cleaned_records

    return cleaned_seed_tables

def generate_sdv_data_optimized(num_records: int, metadata_dict: Dict[str, Any], seed_tables_dict: Dict[str, Any], 
                               batch_size: int = 1000, use_fast_synthesizer: bool = True) -> Dict[str, pd.DataFrame]:
    """
    OPTIMIZED: Uses batch processing and faster synthesizers for large datasets with enhanced error handling.
    """
    try:
        _update_progress("processing", "Cleaning seed data", 5)
        cleaned_seed_tables_dict = clean_seed_data(seed_tables_dict, metadata_dict)

        # Convert to DataFrames with validation
        seed_tables = {}
        for table_name, data in cleaned_seed_tables_dict.items():
            try:
                if not data:
                    logger.warning(f"No data for table '{table_name}', skipping")
                    continue
                df = pd.DataFrame.from_records(data)
                if df.empty:
                    logger.warning(f"Empty DataFrame for table '{table_name}', skipping")
                    continue
                seed_tables[table_name] = df
                logger.info(f"Table '{table_name}': {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"Error creating DataFrame for table '{table_name}': {e}")
                continue

        if not seed_tables:
            raise ValueError("No valid seed tables found after cleaning")

        # Create and validate metadata
        try:
            metadata = Metadata.load_from_dict(metadata_dict)
            _update_progress("processing", "Validating metadata", 10)
            metadata.validate()
            logger.info("Metadata is valid. Proceeding to optimized synthesis.")
        except Exception as e:
            logger.error(f"Metadata validation failed: {e}")
            raise ValueError(f"Invalid metadata: {e}")

        num_tables = len(seed_tables)
        has_relationships = len(metadata.relationships) > 0 if hasattr(metadata, 'relationships') else False
        all_synthetic_data = {}

        if num_tables == 1:
            logger.info("Detected single table. Using optimized single-table synthesis.")
            table_name = list(seed_tables.keys())[0]
            table_df = list(seed_tables.values())[0]
            
            # Use faster synthesizer for large datasets
            try:
                if use_fast_synthesizer and num_records > 500:
                    synthesizer = GaussianCopulaSynthesizer(metadata)
                else:
                    synthesizer = GaussianCopulaSynthesizer(metadata)
                
                _update_progress("processing", "Training synthesizer", 20)
                synthesizer.fit(table_df)
                
                # Batch generation for large datasets
                if num_records > batch_size:
                    synthetic_data_parts = []
                    batches = (num_records + batch_size - 1) // batch_size
                    
                    for batch_idx in range(batches):
                        current_batch_size = min(batch_size, num_records - batch_idx * batch_size)
                        
                        _update_progress("processing", f"Generating batch {batch_idx + 1}/{batches}", 
                                       20 + int(70 * (batch_idx + 1) / batches))
                        
                        batch_data = synthesizer.sample(num_rows=current_batch_size)
                        synthetic_data_parts.append(batch_data)
                        
                        # Memory management
                        if batch_idx % 5 == 0:
                            gc.collect()
                    
                    synthetic_df = pd.concat(synthetic_data_parts, ignore_index=True)
                else:
                    _update_progress("processing", "Generating synthetic data", 50)
                    synthetic_df = synthesizer.sample(num_rows=num_records)
                
                all_synthetic_data[table_name] = synthetic_df
                
            except Exception as e:
                logger.error(f"Single table synthesis failed: {e}")
                raise ValueError(f"Synthesis failed for table '{table_name}': {e}")

        elif num_tables > 1 and has_relationships:
            logger.info("Detected relational tables. Using optimized HMA synthesis.")
            try:
                synthesizer = HMASynthesizer(metadata)
                
                _update_progress("processing", "Training multi-table synthesizer", 30)
                synthesizer.fit(seed_tables)

                # Calculate optimal scale factor
                total_seed_rows = sum(len(df) for df in seed_tables.values())
                scale_factor = min(num_records / total_seed_rows if total_seed_rows > 0 else 1.0, 10.0)  # Cap scale factor
                
                logger.info(f"Using optimized scale factor: {scale_factor:.2f}")
                
                _update_progress("processing", "Generating relational data", 60)
                synthetic_data = synthesizer.sample(scale=scale_factor)
                all_synthetic_data = synthetic_data
                
            except Exception as e:
                logger.error(f"Multi-table synthesis failed: {e}")
                raise ValueError(f"Multi-table synthesis failed: {e}")

        else:
            logger.info("Detected multiple UNRELATED tables. Using parallel synthesis.")
            
            def synthesize_table(table_info):
                table_name, df = table_info
                try:
                    logger.info(f"Synthesizing independent table: {table_name}")
                    
                    # Create single table metadata
                    single_table_metadata = Metadata.from_dict({
                        "tables": {table_name: metadata.tables[table_name]}
                    })

                    synthesizer = GaussianCopulaSynthesizer(single_table_metadata)
                    synthesizer.fit(df)
                    
                    # Batch processing for individual tables
                    if num_records > batch_size:
                        synthetic_parts = []
                        batches = (num_records + batch_size - 1) // batch_size
                        
                        for batch_idx in range(batches):
                            current_batch_size = min(batch_size, num_records - batch_idx * batch_size)
                            batch_data = synthesizer.sample(num_rows=current_batch_size)
                            synthetic_parts.append(batch_data)
                        
                        return table_name, pd.concat(synthetic_parts, ignore_index=True)
                    else:
                        return table_name, synthesizer.sample(num_rows=num_records)
                        
                except Exception as e:
                    logger.error(f"Failed to synthesize table '{table_name}': {e}")
                    raise e
            
            # Parallel processing for multiple tables
            try:
                with ThreadPoolExecutor(max_workers=min(4, num_tables)) as executor:
                    _update_progress("processing", "Parallel table synthesis", 40)
                    results = list(executor.map(synthesize_table, seed_tables.items()))
                
                for table_name, synthetic_df in results:
                    all_synthetic_data[table_name] = synthetic_df
                    
            except Exception as e:
                logger.error(f"Parallel synthesis failed: {e}")
                raise ValueError(f"Parallel synthesis failed: {e}")

        _update_progress("processing", "Finalizing data", 90)
        total_records = sum(len(df) for df in all_synthetic_data.values())
        logger.info(f"Successfully generated {total_records} total records")
        
        if total_records == 0:
            raise ValueError("No synthetic data was generated")
        
        return all_synthetic_data

    except Exception as e:
        error_msg = f"Error in generate_sdv_data_optimized: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        _update_progress("error", f"Synthesis failed: {str(e)}", 0, error=str(e))
        raise e

def run_synthesis_in_background(request: SynthesizeRequest):
    """
    Executes the optimized long-running synthesis and updates the cache upon completion.
    Enhanced with better error handling and recovery.
    """
    global generated_data_cache
    
    try:
        start_time = time.time()
        _update_progress("processing", "Starting synthesis", 0)
        
        synthetic_data = generate_sdv_data_optimized(
            request.num_records,
            request.metadata_dict,
            request.seed_tables_dict,
            request.batch_size,
            request.use_fast_synthesizer
        )
        
        total_records = sum(len(df) for df in synthetic_data.values())
        end_time = time.time()
        duration = end_time - start_time
        
        _update_cache({
            "synthetic_data": synthetic_data,
            "metadata": {
                "total_records_generated": total_records,
                "generation_time_seconds": duration,
                "tables": {
                    name: {"rows": len(df), "columns": df.columns.tolist()}
                    for name, df in synthetic_data.items()
                }
            }
        })
        
        _update_progress("complete", "Synthesis completed", 100, total_records)
        
        notify_user_by_email(
            request.user_email,
            "Complete",
            f"Optimized synthesis completed in {duration:.1f} seconds! {total_records} records generated."
        )

    except Exception as e:
        error_detail = f"Optimized SDV synthesis failed: {str(e)}"
        logger.error(error_detail)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        _update_progress("error", error_detail, 0, error=error_detail)
        generated_data_cache["synthetic_data"] = None 
        
        notify_user_by_email(
            request.user_email,
            "Failed",
            f"Synthetic data generation failed: {error_detail}"
        )

def upload_to_gcs(bucket_name: str, synthetic_data: Dict[str, pd.DataFrame]):
    """
    Uploads each DataFrame in the synthetic_data dictionary to a GCS bucket.
    Enhanced with better error handling and validation.
    """
    try:
        credentials, _ = setup_google_auth()
        storage_client = storage.Client(project=GCP_PROJECT_ID, credentials=credentials)
        
        # Validate bucket exists
        try:
            bucket = storage_client.bucket(bucket_name)
            bucket.reload()  # This will raise an exception if bucket doesn't exist
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"GCS bucket '{bucket_name}' not accessible: {e}"
            )

        uploaded_files = []
        for table_name, df in synthetic_data.items():
            try:
                if df.empty:
                    logger.warning(f"Skipping empty DataFrame for table '{table_name}'")
                    continue
                    
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                file_name = f"{table_name}_synthetic_{timestamp}.csv"
                
                logger.info(f"Uploading {file_name} to GCS bucket '{bucket_name}'...")
                blob = bucket.blob(file_name)
                
                # Convert DataFrame to CSV string
                csv_data = df.to_csv(index=False)
                blob.upload_from_string(csv_data, 'text/csv')
                
                uploaded_files.append(file_name)
                logger.info(f"Successfully uploaded {file_name} ({len(df)} rows)")
                
            except Exception as e:
                logger.error(f"Failed to upload table '{table_name}': {e}")
                raise e
            
        logger.info(f"All {len(uploaded_files)} files uploaded successfully.")
        return uploaded_files
        
    except Exception as e:
        error_msg = f"GCS upload failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

# --- API Endpoints ---

@app.post("/api/v1/design")
async def design_schema_endpoint(request: DesignRequest):
    """STEP 1: Calls AI to generate metadata and seed data with enhanced error handling."""
    global generated_data_cache
    
    try:
        existing_metadata_json = None
        if request.existing_metadata:
            existing_metadata_json = json.dumps(request.existing_metadata, indent=2)

        ai_output = call_ai_agent(request.data_description, existing_metadata_json)
        
        generated_data_cache["design_output"] = ai_output
        generated_data_cache["num_records_target"] = request.num_records
        
        # Create preview with error handling
        seed_data_preview = {}
        for table, data in ai_output["seed_tables_dict"].items():
            try:
                if data:
                    df = pd.DataFrame.from_records(data)
                    seed_data_preview[table] = df.head(20).to_dict('records')
                else:
                    seed_data_preview[table] = []
            except Exception as e:
                logger.warning(f"Error creating preview for table '{table}': {e}")
                seed_data_preview[table] = []
        
        return {
            "status": "review_required",
            "message": "AI model has generated the schema and seed data. Please review before proceeding to synthesis.",
            "metadata_preview": ai_output["metadata_dict"],
            "seed_data_preview": seed_data_preview,
            "tables_count": len(ai_output["metadata_dict"].get("tables", {})),
            "total_seed_records": sum(len(data) for data in ai_output["seed_tables_dict"].values())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Design endpoint failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/v1/synthesize")
async def synthesize_data_endpoint(request: SynthesizeRequest, background_tasks: BackgroundTasks):
    """STEP 2: Starts the optimized long-running synthesis in a background task."""
    global generated_data_cache
    
    try:
        if not generated_data_cache["design_output"]:
            raise HTTPException(status_code=400, detail="Schema design must be completed (POST /design) before synthesis.")

        # Validate request parameters
        if request.num_records <= 0:
            raise HTTPException(status_code=400, detail="num_records must be positive")
        
        if request.batch_size <= 0:
            raise HTTPException(status_code=400, detail="batch_size must be positive")

        background_tasks.add_task(run_synthesis_in_background, request)
        
        generated_data_cache["synthetic_data"] = "Processing"
        _update_progress("processing", "Initializing synthesis", 0)
        
        return {
            "status": "processing_started",
            "message": "Optimized synthesis started in the background. You will be notified via email when complete.",
            "target_email": request.user_email,
            "batch_size": request.batch_size,
            "use_fast_synthesizer": request.use_fast_synthesizer,
            "estimated_records": request.num_records
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Synthesize endpoint failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/v1/progress")
async def get_progress():
    """NEW: Get real-time progress of synthesis."""
    try:
        progress = _get_cache()["progress"]
        return {
            "status": progress["status"],
            "current_step": progress["current_step"],
            "progress_percent": progress["progress_percent"],
            "records_generated": progress["records_generated"],
            "error_message": progress["error_message"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Progress endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {e}")

@app.get("/api/v1/sample")
async def sample_data_endpoint(table_name: Optional[str] = None, sample_size: int = 20):
    """STEP 3: Returns a sample of the synthesized data for review."""
    global generated_data_cache
    
    try:
        synthetic_data = generated_data_cache.get("synthetic_data")
        
        if synthetic_data == "Processing":
            raise HTTPException(status_code=202, detail="Data synthesis is still processing in the background.")
        
        if not synthetic_data:
            raise HTTPException(status_code=404, detail="Synthesis not complete or failed.")

        if table_name:
            if table_name not in synthetic_data:
                available = list(synthetic_data.keys())
                raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found. Available tables: {available}")
            
            table_df = synthetic_data[table_name]
            sample_df = table_df.head(min(sample_size, len(table_df)))
            
            return {
                "table_name": table_name,
                "sample_data": sample_df.to_dict('records'),
                "total_rows": len(table_df),
                "sample_size": len(sample_df)
            }
        else:
            all_samples = {}
            for name, df in synthetic_data.items():
                all_samples[name] = df.head(min(sample_size, len(df))).to_dict('records')
            
            return {
                "status": "success",
                "message": "Returning samples for all generated tables.",
                "all_samples": all_samples,
                "metadata": generated_data_cache.get("metadata", {}),
                "total_tables": len(synthetic_data)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Sample endpoint failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/v1/store")
async def store_data_endpoint(request: StoreRequest):
    """STEP 4: Stores the final synthesized data into Google Cloud Storage."""
    global generated_data_cache
    
    try:
        synthetic_data = generated_data_cache.get("synthetic_data")
        
        if synthetic_data == "Processing":
            raise HTTPException(status_code=400, detail="Synthesis is still processing. Please wait for completion.")

        if not synthetic_data:
            raise HTTPException(status_code=404, detail="No synthesized data to store.")
        
        if not request.confirm_storage:
            raise HTTPException(status_code=400, detail="Storage not confirmed. Set `confirm_storage` to `true`.")

        uploaded_files = upload_to_gcs_with_fallback(GCS_BUCKET_NAME, synthetic_data)
        
        # Clear cache
        generated_data_cache = {
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
        
        return {
            "status": "storage_complete",
            "message": f"Successfully stored {len(synthetic_data)} tables to GCS.",
            "bucket": GCS_BUCKET_NAME,
            "uploaded_files": uploaded_files,
            "total_files": len(uploaded_files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Store endpoint failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

def create_download_package() -> io.BytesIO:
    """
    Creates a ZIP package containing all CSV files and metadata for download.
    """
    try:
        synthetic_data = generated_data_cache.get("synthetic_data")
        design_output = generated_data_cache.get("design_output")
        
        if synthetic_data == "Processing":
            raise HTTPException(status_code=400, detail="Synthesis is still processing. Please wait for completion.")
        
        if not synthetic_data:
            raise HTTPException(status_code=404, detail="No synthesized data available for download.")
        
        # Create a BytesIO buffer for the ZIP file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add CSV files for each table
            for table_name, df in synthetic_data.items():
                if not df.empty:
                    csv_data = df.to_csv(index=False)
                    zip_file.writestr(f"{table_name}_synthetic_data.csv", csv_data)
                    logger.info(f"Added {table_name}.csv to download package ({len(df)} rows)")
            
            # Add metadata file if available
            if design_output:
                metadata_json = json.dumps(design_output["metadata_dict"], indent=2)
                zip_file.writestr("metadata.json", metadata_json)
                logger.info("Added metadata.json to download package")
                
                # Add seed data file
                seed_data_json = json.dumps(design_output["seed_tables_dict"], indent=2)
                zip_file.writestr("seed_data.json", seed_data_json)
                logger.info("Added seed_data.json to download package")
            
            # Add summary file
            summary_info = {
                "generation_timestamp": datetime.now().isoformat(),
                "total_tables": len(synthetic_data),
                "total_records": sum(len(df) for df in synthetic_data.values()),
                "table_summary": {
                    name: {"rows": len(df), "columns": df.columns.tolist()}
                    for name, df in synthetic_data.items()
                },
                "metadata_info": generated_data_cache.get("metadata", {})
            }
            summary_json = json.dumps(summary_info, indent=2)
            zip_file.writestr("generation_summary.json", summary_json)
            logger.info("Added generation_summary.json to download package")
        
        zip_buffer.seek(0)
        return zip_buffer
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to create download package: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

def upload_to_gcs_with_fallback(bucket_name: str, synthetic_data: Dict[str, pd.DataFrame]):
    """
    Enhanced GCS upload with better error handling and fallback options.
    """
    try:
        # First, try to set up authentication with proper project configuration
        credentials, project = setup_google_auth()
        
        # Initialize storage client with explicit project and credentials
        storage_client = storage.Client(
            project=GCP_PROJECT_ID, 
            credentials=credentials
        )
        
        # Test if we can access the bucket
        try:
            bucket = storage_client.bucket(bucket_name)
            # Try to get bucket metadata to verify access
            bucket.reload()
            logger.info(f"Successfully connected to GCS bucket '{bucket_name}'")
        except Exception as bucket_error:
            # Provide detailed error information for GCS permission issues
            error_details = str(bucket_error)
            
            if "403" in error_details or "serviceusage.services.use" in error_details:
                detailed_error = f"""
GCS Permission Error: {error_details}

Possible solutions:
1. Enable the Cloud Storage API for your project:
   gcloud services enable storage-component.googleapis.com --project={GCP_PROJECT_ID}

2. Grant the necessary IAM roles to your service account:
   - Storage Admin or Storage Object Admin
   - Service Usage Consumer

3. Set up Application Default Credentials:
   gcloud auth application-default login

4. Verify your project ID is correct in config.py: {GCP_PROJECT_ID}

5. Create the bucket if it doesn't exist:
   gsutil mb gs://{bucket_name}

Current authentication: {type(credentials).__name__}
Project ID: {GCP_PROJECT_ID}
                """
                raise HTTPException(status_code=403, detail=detailed_error.strip())
            else:
                raise HTTPException(status_code=500, detail=f"GCS bucket '{bucket_name}' not accessible: {error_details}")

        # If we reach here, bucket access is working
        uploaded_files = []
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        for table_name, df in synthetic_data.items():
            try:
                if df.empty:
                    logger.warning(f"Skipping empty DataFrame for table '{table_name}'")
                    continue
                    
                file_name = f"{table_name}_synthetic_{timestamp}.csv"
                
                logger.info(f"Uploading {file_name} to GCS bucket '{bucket_name}'...")
                blob = bucket.blob(file_name)
                
                # Convert DataFrame to CSV string
                csv_data = df.to_csv(index=False)
                blob.upload_from_string(csv_data, 'text/csv')
                
                uploaded_files.append(file_name)
                logger.info(f"Successfully uploaded {file_name} ({len(df)} rows)")
                
            except Exception as e:
                logger.error(f"Failed to upload table '{table_name}': {e}")
                raise e
        
        # Upload metadata file if available
        design_output = generated_data_cache.get("design_output")
        if design_output:
            try:
                metadata_file_name = f"metadata_{timestamp}.json"
                metadata_blob = bucket.blob(metadata_file_name)
                metadata_json = json.dumps(design_output["metadata_dict"], indent=2)
                metadata_blob.upload_from_string(metadata_json, 'application/json')
                uploaded_files.append(metadata_file_name)
                logger.info(f"Successfully uploaded {metadata_file_name}")
            except Exception as e:
                logger.warning(f"Failed to upload metadata file: {e}")
            
        logger.info(f"All {len(uploaded_files)} files uploaded successfully to GCS.")
        return uploaded_files
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"GCS upload failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

# --- NEW: Download Endpoint ---
@app.get("/api/v1/download")
async def download_data_endpoint():
    """NEW: Downloads all generated CSV files and metadata as a ZIP package."""
    try:
        logger.info("Creating download package...")
        zip_buffer = create_download_package()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"matrix_ai_synthetic_data_{timestamp}.zip"
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Download failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    try:
        # Test Google Cloud authentication
        credentials, project = setup_google_auth()
        
        return {
            "status": "healthy",
            "version": "6.1.0",
            "project_id": GCP_PROJECT_ID,
            "timestamp": datetime.now().isoformat(),
            "cache_status": _get_cache()["progress"]["status"]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

# --- Root Endpoint ---
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Matrix AI - Optimized Synthetic Data Generator",
        "version": "6.1.0",
        "status": "running",
        "endpoints": {
            "design": "/api/v1/design",
            "synthesize": "/api/v1/synthesize", 
            "progress": "/api/v1/progress",
            "sample": "/api/v1/sample",
            "download": "/api/v1/download",
            "store": "/api/v1/store",
            "health": "/health"
        },
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
