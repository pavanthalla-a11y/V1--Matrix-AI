import pandas as pd
from sdv.multi_table import HMASynthesizer
from sdv.metadata import MultiTableMetadata
import io
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel

# Import the configuration variables
from config import GCP_PROJECT_ID, GCS_BUCKET_NAME

app = FastAPI(
    title="SDG All-in-One Orchestrator",
    description="A single service to generate data and store it in GCS.",
    version="2.0.0"
)

# --- Pydantic Models for API ---
class DataRequest(BaseModel):
    data_description: str
    num_records: int
    
# --- Core Logic Functions ---

def call_ai_agent(data_description: str) -> Dict[str, Any]:
    """
    Calls the Gemini API to get the metadata schema and seed data.
    Returns a dictionary with 'metadata_dict' and 'seed_tables_dict'.
    """
    vertexai.init(project=GCP_PROJECT_ID, location="us-central1")
    model = GenerativeModel("gemini-1.0-pro")

    prompt_text = f"""
    You are a professional data architect. Generate a single JSON object that contains:
    1. A 'metadata_dict' for a synthetic multi-table dataset that matches the description.
    2. A 'seed_tables_dict' with mock seed data for each table.
    
    The description of the data is: "{data_description}"
    
    The metadata format must follow the SDV MultiTableMetadata specifications with 'sdtype's.
    Ensure all tables have a 'primary_key' and that 'relationships' are correctly defined.
    The seed data for each table should be a list of dictionaries.
    
    Return only the complete JSON object, nothing else.
    """
    
    try:
        response = model.generate_content(prompt_text, generation_config={"response_mime_type": "application/json"})
        response_json = response.text
        return json.loads(response_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Agent failed to generate schema: {e}")

def generate_sdv_data(num_records: int, metadata_dict: Dict[str, Any], seed_tables_dict: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Uses the SDV library to generate synthetic data.
    """
    try:
        metadata = MultiTableMetadata.load_from_dict(metadata_dict)
        seed_tables = {
            table_name: pd.DataFrame.from_dict(table_data)
            for table_name, table_data in seed_tables_dict.items()
        }
        synthesizer = HMASynthesizer(metadata)
        synthesizer.fit(seed_tables)
        synthetic_tables = synthesizer.sample(num_records)
        return synthetic_tables
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SDV generation failed: {e}")

def upload_to_gcs(bucket_name: str, synthetic_data: Dict[str, pd.DataFrame]):
    """
    Uploads each DataFrame to a Google Cloud Storage bucket as a CSV file.
    """
    try:
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)

        for table_name, df in synthetic_data.items():
            file_name = f"{table_name}_synthetic.csv"
            
            csv_in_memory = io.StringIO()
            df.to_csv(csv_in_memory, index=False)
            csv_in_memory.seek(0)
            
            blob = bucket.blob(file_name)
            blob.upload_from_file(csv_in_memory, content_type='text/csv')
            print(f"Uploaded {file_name} to GCS.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GCS upload failed: {e}")

# --- API Endpoints ---

@app.post("/generate-and-store")
async def generate_and_store(request: DataRequest):
    """
    Orchestrates the entire data generation and storage process.
    """
    print("Step 1: Calling AI Agent to generate schema and seed data...")
    ai_output = call_ai_agent(request.data_description)
    metadata_dict = ai_output.get("metadata_dict")
    seed_tables_dict = ai_output.get("seed_tables_dict")
    
    if not metadata_dict or not seed_tables_dict:
        raise HTTPException(status_code=500, detail="AI Agent returned incomplete data.")

    print("Step 2: Generating synthetic data with SDV...")
    synthetic_data = generate_sdv_data(
        request.num_records,
        metadata_dict,
        seed_tables_dict
    )

    print("Step 3: Uploading generated files to GCS...")
    upload_to_gcs(GCS_BUCKET_NAME, synthetic_data)

    return {"message": "Data generation and storage successful!", "status": "completed"}