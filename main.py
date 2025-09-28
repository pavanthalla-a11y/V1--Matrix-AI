import pandas as pd
import numpy as np
import json
import traceback
import google.auth
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
from typing import Dict, Any, List, Optional
import time
import uuid
import asyncio
import threading
from starlette.status import HTTP_202_ACCEPTED

# --- SDV IMPORTS ---
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from sdv.single_table import GaussianCopulaSynthesizer 

# --- GOOGLE CLOUD IMPORTS ---
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel

# IMPORTANT: You must install 'email-validator' for EmailStr to work:
# pip install 'pydantic[email]'

# Import configuration variables from your config file
from config import GCP_PROJECT_ID, GCS_BUCKET_NAME, GCP_LOCATION, GEMINI_MODEL

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Matrix AI - Synthetic Data Generator",
    description="Interactive AI-powered data generation using Gemini and SDV.",
    version="5.0.0"
)

# --- Thread-safe Cache Structure ---
_cache_lock = threading.Lock()
generated_data_cache: Dict[str, Any] = {
    "design_output": None,
    "synthetic_data": None,
    "num_records_target": 0
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

class StoreRequest(BaseModel):
    confirm_storage: bool


# --- Core Logic Functions ---

def notify_user_by_email(email: str, status: str, details: str):
    """[CRITICAL PLACEHOLDER] - Simulates the email notification service."""
    print(f"\n--- EMAIL NOTIFICATION SIMULATION ---")
    print(f"TO: {email}")
    print(f"STATUS: {status}")
    print(f"DETAILS: {details}")
    print(f"-------------------------------------\n")
    pass

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
    4. The `seed_tables_dict` must contain 20 realistic data rows for each table, ensuring all foreign key relationships are valid.
    5. **Output:** Return ONLY the complete JSON object.
    """

    print("Generating/Refining schema with Gemini...")
    try:
        response = model.generate_content(prompt)
        ai_output = json.loads(response.text)
        
        if "metadata_dict" not in ai_output or "seed_tables_dict" not in ai_output:
             raise ValueError("AI output missing required top-level keys.")
        
        return ai_output
    except Exception as e:
        print(f"ERROR during AI call: {e}")
        raise HTTPException(status_code=500, detail=f"AI agent failed to generate a valid schema: {e}")


def clean_seed_data(seed_tables_dict: Dict[str, List[Dict[str, Any]]], metadata_dict: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Cleans seed data by removing AI placeholder text/invalid dates from datetime columns.
    This function prevents SDV from failing due to AI hallucination.
    """
    print("Pre-processing seed data: Robustly removing AI placeholders...")
    cleaned_seed_tables = {}

    # More specific placeholder pattern to avoid false positives
    PLACEHOLDER_PATTERN = r'\(format\)|YYYY|MM|DD|HH:MM:SS|Time|\(value\)|^\s*null\s*$|^\s*None\s*$'
    REPLACEMENT_DATE = '2020-01-01 00:00:00'
    
    datetime_columns = {
        col_name: table_name 
        for table_name, table_meta in metadata_dict['tables'].items() 
        for col_name, col_data in table_meta['columns'].items() 
        if col_data.get('sdtype') == 'datetime'
    }

    for table_name, data_records in seed_tables_dict.items():
        if not data_records: 
            cleaned_seed_tables[table_name] = []
            continue
        
        df = pd.DataFrame.from_records(data_records)
        
        current_date_cols = [col for col in df.columns if col in datetime_columns and datetime_columns[col] == table_name]

        if not current_date_cols:
            cleaned_seed_tables[table_name] = df.to_dict('records')
            continue

        rows_before = len(df)
        
        for col in current_date_cols:
            df[col] = df[col].astype(str).fillna('')
            df[col] = df[col].replace(to_replace=PLACEHOLDER_PATTERN, value=REPLACEMENT_DATE, regex=True)

        for col in current_date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        cleaned_df = df.dropna(subset=current_date_cols).copy()
        
        for col in current_date_cols:
            cleaned_df[col] = cleaned_df[col].fillna(pd.to_datetime(REPLACEMENT_DATE))
        
        rows_after = len(cleaned_df)
        print(f"  - Table '{table_name}': Dropped {rows_before - rows_after} invalid seed rows.")
        
        cleaned_seed_tables[table_name] = cleaned_df.to_dict('records')

    return cleaned_seed_tables


def generate_sdv_data(num_records: int, metadata_dict: Dict[str, Any], seed_tables_dict: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Uses SDV to generate synthetic data, utilizing the cleaned data.
    Handles all cases: single table, multi-table with relationships, and multi-table without relationships.
    """
    try:
        cleaned_seed_tables_dict = clean_seed_data(seed_tables_dict, metadata_dict)

        seed_tables = {
            table_name: pd.DataFrame.from_records(data)
            for table_name, data in cleaned_seed_tables_dict.items()
        }

        metadata = Metadata.load_from_dict(metadata_dict)

        print("Validating AI-generated metadata...")
        metadata.validate()
        print("Metadata is valid. Proceeding to synthesis.")

        num_tables = len(seed_tables)
        has_relationships = len(metadata.relationships) > 0 if hasattr(metadata, 'relationships') else False
        all_synthetic_data = {}

        if num_tables == 1:
            print("Detected single table. Using GaussianCopulaSynthesizer.")
            synthesizer = GaussianCopulaSynthesizer(metadata)
            synthesizer.fit(seed_tables)
            synthetic_df = synthesizer.sample(num_rows=num_records)
            all_synthetic_data[list(seed_tables.keys())[0]] = synthetic_df

        elif num_tables > 1 and has_relationships:
            print("Detected relational tables. Using HMASynthesizer.")
            synthesizer = HMASynthesizer(metadata)
            synthesizer.fit(seed_tables)

            # Calculate scale factor based on seed data size
            total_seed_rows = sum(len(df) for df in seed_tables.values())
            scale_factor = num_records / total_seed_rows if total_seed_rows > 0 else 1.0

            print(f"Seed data has {total_seed_rows} total rows, using scale factor: {scale_factor:.2f}")
            synthetic_data = synthesizer.sample(scale=scale_factor)
            all_synthetic_data = synthetic_data

        else:
            print("Detected multiple UNRELATED tables. Synthesizing individually.")
            for table_name, df in seed_tables.items():
                print(f"-> Synthesizing independent table: {table_name}")
                single_table_metadata = Metadata.from_dict({
                    "tables": {table_name: metadata.tables[table_name]}
                })

                synthesizer = GaussianCopulaSynthesizer(single_table_metadata)
                synthesizer.fit(df)
                synthetic_df = synthesizer.sample(num_rows=num_records)
                all_synthetic_data[table_name] = synthetic_df

        return all_synthetic_data

    except Exception as e:
        print(f"Error in generate_sdv_data: {str(e)}")
        raise e


def run_synthesis_in_background(request: SynthesizeRequest):
    """
    Executes the long-running synthesis and updates the cache upon completion.
    """
    global generated_data_cache
    
    try:
        synthetic_data = generate_sdv_data(
            request.num_records,
            request.metadata_dict,
            request.seed_tables_dict
        )
        
        total_records = sum(len(df) for df in synthetic_data.values())
        
        generated_data_cache["synthetic_data"] = synthetic_data
        
        generated_data_cache["metadata"] = {
            "total_records_generated": total_records,
            "tables": {
                name: {"rows": len(df), "columns": df.columns.tolist()}
                for name, df in synthetic_data.items()
            }
        }
        
        notify_user_by_email(
            request.user_email,
            "Complete",
            f"Your synthetic data generation is finished! {total_records} records were created. You can now view samples and store the data."
        )

    except Exception as e:
        error_detail = f"SDV synthesis failed: {str(e)}"
        print(error_detail)
        generated_data_cache["synthetic_data"] = None 
        
        notify_user_by_email(
            request.user_email,
            "Failed",
            f"Synthetic data generation failed due to an error: {error_detail}"
        )


def upload_to_gcs(bucket_name: str, synthetic_data: Dict[str, pd.DataFrame]):
    """
    Uploads each DataFrame in the synthetic_data dictionary to a GCS bucket.
    """
    try:
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)

        for table_name, df in synthetic_data.items():
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"{table_name}_synthetic_{timestamp}.csv"
            
            print(f"Uploading {file_name} to GCS bucket '{bucket_name}'...")
            blob = bucket.blob(file_name)
            blob.upload_from_string(df.to_csv(index=False), 'text/csv')
            
        print("All files uploaded successfully.")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"GCS upload failed: {traceback.format_exc()}"
        )


@app.post("/api/v1/design")
async def design_schema_endpoint(request: DesignRequest):
    """
    STEP 1: Calls AI to generate metadata and seed data.
    """
    global generated_data_cache
    
    existing_metadata_json = None
    if request.existing_metadata:
        existing_metadata_json = json.dumps(request.existing_metadata, indent=2)

    ai_output = call_ai_agent(request.data_description, existing_metadata_json)
    
    generated_data_cache["design_output"] = ai_output
    generated_data_cache["num_records_target"] = request.num_records
    
    return {
        "status": "review_required",
        "message": "AI model has generated the schema and seed data. Please review before proceeding to synthesis.",
        "metadata_preview": ai_output["metadata_dict"],
        "seed_data_preview": {
            table: pd.DataFrame.from_records(data).head(20).to_dict('records')
            for table, data in ai_output["seed_tables_dict"].items()
        }
    }


@app.post("/api/v1/synthesize")
async def synthesize_data_endpoint(request: SynthesizeRequest, background_tasks: BackgroundTasks):
    """
    STEP 2: Starts the long-running synthesis in a background task and returns immediately (202 Accepted).
    """
    global generated_data_cache
    
    if not generated_data_cache["design_output"]:
        raise HTTPException(status_code=400, detail="Schema design must be completed (POST /design) before synthesis.")

    background_tasks.add_task(
        run_synthesis_in_background,
        request
    )
    
    generated_data_cache["synthetic_data"] = "Processing"
    
    return {
        "status": "processing_started",
        "message": "Synthesis started in the background. You will be notified via email when the data is ready to view and store.",
        "target_email": request.user_email
    }


@app.get("/api/v1/sample")
async def sample_data_endpoint(table_name: Optional[str] = None, sample_size: int = 20):
    """
    STEP 3: Returns a sample of the synthesized data for review.
    """
    global generated_data_cache
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
            "sample_data": sample_df.to_dict('records')
        }
    else:
        all_samples = {}
        for name, df in synthetic_data.items():
            all_samples[name] = df.head(min(sample_size, len(df))).to_dict('records')
        
        return {
            "status": "success",
            "message": "Returning samples for all generated tables.",
            "all_samples": all_samples
        }


@app.post("/api/v1/store")
async def store_data_endpoint(request: StoreRequest):
    """
    STEP 4: Stores the final synthesized data into Google Cloud Storage.
    """
    global generated_data_cache
    synthetic_data = generated_data_cache.get("synthetic_data")
    
    if synthetic_data == "Processing":
        raise HTTPException(status_code=400, detail="Synthesis is still processing. Please wait for the email notification.")

    if not synthetic_data:
        raise HTTPException(status_code=404, detail="No synthesized data to store.")
    
    if not request.confirm_storage:
        raise HTTPException(status_code=400, detail="Storage not confirmed. Set `confirm_storage` to `true`.")

    upload_to_gcs(GCP_BUCKET_NAME, synthetic_data)
    
    generated_data_cache = {"design_output": None, "synthetic_data": None, "num_records_target": 0}
    
    return {
        "status": "storage_complete",
        "message": f"Successfully stored {len(synthetic_data)} tables to GCS.",
        "bucket": GCS_BUCKET_NAME,
    }
