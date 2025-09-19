import pandas as pd
from sdv.multi_table import HMASynthesizer
from sdv.metadata import MultiTableMetadata
import io
import json
import traceback
import sys
import google.auth
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel

# Import the configuration variables
from config import GCP_PROJECT_ID, GCS_BUCKET_NAME

app = FastAPI(
    title="Matrix AI - Synthetic Data Generator",
    description="Generate, preview, and store synthetic data using AI + SDV.",
    version="3.0.0"
)

# Global storage for generated data (in production, use Redis or similar)
generated_data_cache = {
    "data": None,
    "metadata": None,
    "quality_report": None,
    "timestamp": None
}

# --- Pydantic Models for API ---
class GenerateRequest(BaseModel):
    data_description: str
    num_records: int

class SampleRequest(BaseModel):
    sample_size: int = 100

class StoreRequest(BaseModel):
    confirm_storage: bool = True
    
# --- Core Logic Functions ---



def call_ai_agent(data_description: str) -> Dict[str, Any]:
    """
    Calls the Gemini API to get the metadata schema and seed data.
    Returns a dictionary with 'metadata_dict' and 'seed_tables_dict'.
    """
    print("Starting AI agent call...")
    print(f"Project ID: {GCP_PROJECT_ID}")
    
    try:
        # Initialize Google Cloud
        print("Initializing Google Cloud...")
        credentials, project = google.auth.default()
        
        # Initialize Vertex AI with the credentials
        print("Initializing Vertex AI...")
        vertexai.init(
            project=GCP_PROJECT_ID,
            location="us-central1",
            credentials=credentials
        )
        print("Vertex AI initialized successfully")
        
        # Configure the model with correct name
        print("Creating model instance...")
        # Try with just the base model name without project path
        model = GenerativeModel(
            "gemini-2.5-pro",
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            }
        )
        print("Model instance created successfully")
        
        # Prepare a generic prompt that works for any use case
        prompt_text = f"""
You are a professional data architect. Generate a single JSON object that contains the following two top-level keys:
- 'metadata_dict'
- 'seed_tables_dict'

The description of the data is: "{data_description}"

CRITICAL INSTRUCTIONS:
- The `metadata_dict` MUST have a 'tables' key. Under the 'tables' key, there will be a dictionary where the keys are the table names.
- Foreign key relationships **MUST be defined in the 'relationships' list** at the top level of the metadata_dict. **DO NOT** define foreign keys inside the individual table dictionaries.
- For multi-table datasets, infer and define relationships. The relationships list must contain a dictionary for each relationship with the following keys:
    - 'parent_table_name'
    - 'parent_primary_key'
    - 'child_table_name'
    - 'child_foreign_key'
- If the description implies multiple tables (e.g., 'customers and orders'), create separate tables and infer the primary and foreign key relationships.
- If the description implies a single table (e.g., 'a customer table'), create just one table.
- Define complex relationships, such as many-to-many, if the prompt implies them.
- Adhere to any specific data constraints mentioned in the description (e.g., 'dates between 2020 and 2024', 'IDs must be unique strings').
- For EACH table, generate exactly 15 diverse, unique, and realistic examples.
- Use appropriate SDV `sdtypes` (id, text, categorical, numerical, datetime, etc.).
-**DO NOT include any extra, non-standard keys like 'pii' in the column metadata. The metadata must strictly adhere to SDV's official schema.**

Return ONLY the complete JSON object, with no extra text or explanations.
"""

        print("Preparing prompt...")
        print(f"Prompt text: {prompt_text}")
        
        print("Calling model.generate_content()...")
        response = model.generate_content(prompt_text)
        print(f"Raw response received: {response}")
        print(f"Response type: {type(response)}")
        
        response_text = response.text
        if not response_text:
            print("Empty response from Gemini API")
            raise ValueError("Empty response from Gemini API")
            
        print(f"Response text length: {len(response_text)}")
        print(f"Response text (first 500 chars): {response_text[:500]}")
        print(f"Response text (last 500 chars): {response_text[-500:]}")
        
        def clean_json_response(text):
            """Clean and fix common JSON issues in LLM responses"""
            import re
            import json
            
            # Remove any leading/trailing whitespace
            text = text.strip()
            
            # Remove any markdown code block formatting
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'\s*```', '', text)
            
            # Fix common JSON issues
            try:
                # First attempt - try to parse as is
                json.loads(text)
                return text
            except json.JSONDecodeError:
                pass
            
            # If response is truncated or has issues, try to fix
            if text and not text.endswith('}'):
                # Find the last complete closing brace for the main object
                last_brace = text.rfind('}')
                if last_brace > 0:
                    text = text[:last_brace + 1]
            
            # Fix common JSON syntax issues
            # Fix trailing commas in arrays and objects
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            
            # Fix missing commas between objects in arrays
            text = re.sub(r'}(\s*){', r'},\1{', text)
            
            # Fix quotes issues - ensure all strings are properly quoted
            text = re.sub(r'(\w+):', r'"\1":', text)  # Fix unquoted keys
            
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                # If still failing, try to extract the main structure
                main_match = re.search(r'(\{.*"metadata_dict".*"seed_tables_dict".*\})', text, re.DOTALL)
                if main_match:
                    return main_match.group(1)
                
            return text
        
        # Clean and parse the JSON response
        try:
            cleaned_text = clean_json_response(response_text)
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"JSON parse failed even after cleaning: {str(e)}")
            print(f"Cleaned text (first 1000 chars): {cleaned_text[:1000]}")
            print(f"Cleaned text (last 1000 chars): {cleaned_text[-1000:]}")
            
            
            return cleaned_text
            
    except Exception as e:
        error_type = type(e).__name__
        error_details = traceback.format_exc()
        
        print(f"Error Type: {error_type}")
        print(f"Error Message: {str(e)}")
        print(f"Full Error Details:\n{error_details}")
        print(f"Python version: {sys.version}")
        
        # Check authentication status
        try:
            credentials, project = google.auth.default()
            print(f"Current credentials: {credentials}")
            print(f"Detected project: {project}")
        except Exception as auth_e:
            print(f"Auth check error: {auth_e}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "AI Agent failed to generate schema",
                "error_type": error_type,
                "error_message": str(e),
                "details": error_details
            }
        )






def generate_sdv_data(num_records: int, metadata_dict: Dict[str, Any], seed_tables_dict: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Uses the SDV library to generate synthetic data.
    """
    try:
        print(f"Metadata dict: {metadata_dict}")
        print(f"Seed tables dict: {seed_tables_dict}")
        
        metadata = MultiTableMetadata.load_from_dict(metadata_dict)
        print(f"Loaded metadata successfully")
        
        seed_tables = {
            table_name: pd.DataFrame.from_dict(table_data)
            for table_name, table_data in seed_tables_dict.items()
        }
        print(f"Created seed tables: {list(seed_tables.keys())}")
        
        # Check if we have single table vs multiple tables
        if len(seed_tables) == 1:
            table_name = list(seed_tables.keys())[0]
            table_data = seed_tables[table_name]
            
            print(f"Using single table synthesizer for table: {table_name}")
            
            single_metadata = SingleTableMetadata()
            single_metadata.detect_from_dataframe(table_data)
            
            synthesizer = CTGANSynthesizer(single_metadata)
            synthesizer.fit(table_data)
            synthetic_table = synthesizer.sample(num_records)
            
            return {table_name: synthetic_table}
        else:
            print("Using multi-table synthesizer for multiple tables")
            print(f"Tables to synthesize: {list(seed_tables.keys())}")
            
            synthesizer = HMASynthesizer(metadata)
            synthesizer.fit(seed_tables)
            synthetic_tables = synthesizer.sample(num_records)
            
            return synthetic_tables
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"SDV Error: {str(e)}")
        print(f"SDV Error Details: {error_details}")
        raise HTTPException(status_code=500, detail=f"SDV generation failed: {str(e)} | Details: {error_details}")

    
def upload_to_gcs(bucket_name: str, synthetic_data: Dict[str, pd.DataFrame]):
    """
    Uploads each DataFrame to a Google Cloud Storage bucket as a CSV file.
    """
    try:
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)

        for table_name, df in synthetic_data.items():
            file_name = f"{table_name}_synthetic_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            print(f"Preparing to upload {file_name} with {len(df)} rows...")
            
            # Convert DataFrame to CSV string
            csv_string = df.to_csv(index=False)
            csv_bytes = csv_string.encode('utf-8')
            
            print(f"CSV size: {len(csv_bytes)} bytes")
            
            # Create blob and upload with proper content type
            blob = bucket.blob(file_name)
            
            # Use upload_from_string for better handling of large content
            blob.upload_from_string(
                csv_string,
                content_type='text/csv'
            )
            
            print(f"Successfully uploaded {file_name} to GCS bucket '{bucket_name}'")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"GCS Upload Error: {str(e)}")
        print(f"Error Details: {error_details}")
        raise HTTPException(status_code=500, detail=f"GCS upload failed: {str(e)}")

# --- API Endpoints ---

@app.post("/api/v1/generate")
async def generate_synthetic_data(request: GenerateRequest):
    """
    Step 1: Generate synthetic data based on description and store in memory.
    """
    global generated_data_cache
    
    try:
        print(f"🚀 Starting data generation for: {request.data_description}")
        print(f"📊 Requested records: {request.num_records}")
        
        # Step 1: Get AI schema and seed data
        ai_output = call_ai_agent(request.data_description)
        metadata_dict = ai_output.get("metadata_dict")
        seed_tables_dict = ai_output.get("seed_tables_dict")
        
        if not metadata_dict or not seed_tables_dict:
            raise HTTPException(status_code=500, detail="AI Agent returned incomplete data.")

        # Step 2: Generate synthetic data
        synthetic_data = generate_sdv_data(request.num_records, metadata_dict, seed_tables_dict)
        
        # Step 3: Store in cache with metadata
        table_name = list(synthetic_data.keys())[0]
        main_data = synthetic_data[table_name]
        
        generated_data_cache = {
            "data": synthetic_data,
            "metadata": {
                "description": request.data_description,
                "num_records": len(main_data),
                "columns": main_data.columns.tolist(),
                "table_name": table_name
            },
            "quality_report": "Available in logs",
            "timestamp": pd.Timestamp.now()
        }
        
        return {
            "status": "success",
            "message": f"Generated {len(main_data)} records successfully",
            "metadata": generated_data_cache["metadata"],
            "next_step": "Use /api/v1/sample to preview data before storing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")


@app.get("/api/v1/sample")
async def get_sample_data(sample_size: int = 100):
    """
    Step 2: Get a sample of generated data for preview.
    """
    global generated_data_cache
    
    if generated_data_cache["data"] is None:
        raise HTTPException(status_code=404, detail="No data generated yet. Use /api/v1/generate first.")
    
    try:
        table_name = list(synthetic_data.keys())[0]
        main_data = synthetic_data[table_name]
        generated_data_cache = {
            "data": synthetic_data,
            "metadata": {
                "description": request.data_description,
                "num_records": len(main_data),
                "columns": main_data.columns.tolist(),
                "table_name": table_name
            },
            "quality_report": "Available in logs",
            "timestamp": pd.Timestamp.now()
        }
        
        return {
            "status": "success",
            "message": f"Generated {len(main_data)} records successfully",
            "metadata": generated_data_cache["metadata"],
            "next_step": "Use /api/v1/sample to preview data before storing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sample: {str(e)}")


@app.post("/api/v1/store")
async def store_data_to_gcs(request: StoreRequest):
    """
    Step 3: Store the generated data to Google Cloud Storage.
    """
    global generated_data_cache
    
    if generated_data_cache["data"] is None:
        raise HTTPException(status_code=404, detail="No data to store. Use /api/v1/generate first.")
    
    if not request.confirm_storage:
        raise HTTPException(status_code=400, detail="Storage not confirmed. Set confirm_storage=true")
    
    try:
        # Upload to GCS
        upload_to_gcs(GCS_BUCKET_NAME, generated_data_cache["data"])
        
        # Get summary
        table_name = generated_data_cache["metadata"]["table_name"]
        record_count = generated_data_cache["metadata"]["num_records"]
        
        # Clear cache after successful storage
        generated_data_cache = {"data": None, "metadata": None, "quality_report": None, "timestamp": None}
        
        return {
            "status": "success",
            "message": f"Successfully stored {record_count} records to GCS",
            "bucket": GCS_BUCKET_NAME,
            "table": table_name,
            "timestamp": pd.Timestamp.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(e)}")


