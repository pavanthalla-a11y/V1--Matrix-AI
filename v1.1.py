import pandas as pd
import json
import traceback
import google.auth
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from sdv.multi_table import HMASynthesizer
from sdv.metadata import MultiTableMetadata, SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition

# Google Cloud integrations
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel

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

# Import configuration variables from your config file
from config import GCP_PROJECT_ID, GCS_BUCKET_NAME, GCP_LOCATION, GEMINI_MODEL

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Matrix AI - Synthetic Data Generator",
    description="Generate, preview, and store synthetic data using AI + SDV.",
    version="4.0.0"
)

# --- In-memory Cache ---
generated_data_cache: Dict[str, Any] = {"data": None, "metadata": None}


# --- Pydantic Models for API Requests ---
class GenerateRequest(BaseModel):
    data_description: str
    num_records: int

class StoreRequest(BaseModel):
    confirm_storage: bool


def optimize_relationships(metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimizes the relationships in the metadata to improve synthesis performance.
    """
    if 'relationships' not in metadata_dict:
        return metadata_dict

    # Create a graph representation of tables and their relationships
    graph = {}
    for rel in metadata_dict['relationships']:
        parent = rel['parent_table_name']
        child = rel['child_table_name']
        if parent not in graph:
            graph[parent] = {'children': [], 'parents': []}
        if child not in graph:
            graph[child] = {'children': [], 'parents': []}
        graph[parent]['children'].append(child)
        graph[child]['parents'].append(parent)

    # Identify and break cyclic relationships
    def find_cycles(node: str, visited: set, path: set) -> List[str]:
        visited.add(node)
        path.add(node)
        cycles = []
        
        for child in graph[node]['children']:
            if child in path:
                # Found a cycle
                cycle_start = list(path)
                cycle_start = cycle_start[cycle_start.index(child):]
                cycles.append(cycle_start + [child])
            elif child not in visited:
                cycles.extend(find_cycles(child, visited, path))
        
        path.remove(node)
        return cycles

    # Find all cycles in the graph
    cycles = []
    visited = set()
    for node in graph:
        if node not in visited:
            cycles.extend(find_cycles(node, visited, set()))

    # Break cycles by removing the least impactful relationships
    optimized_relationships = []
    for rel in metadata_dict['relationships']:
        parent = rel['parent_table_name']
        child = rel['child_table_name']
        
        # Check if this relationship is part of a cycle
        is_cyclic = False
        for cycle in cycles:
            if parent in cycle and child in cycle:
                # Only keep the relationship if it's not creating a direct cycle
                if cycle.index(parent) + 1 != cycle.index(child):
                    optimized_relationships.append(rel)
                is_cyclic = True
                break
        
        if not is_cyclic:
            optimized_relationships.append(rel)

    metadata_dict['relationships'] = optimized_relationships
    return metadata_dict


def preprocess_tables(seed_tables: Dict[str, pd.DataFrame], metadata: MultiTableMetadata) -> Dict[str, pd.DataFrame]:
    """
    Preprocesses tables to improve synthesis quality and performance.
    """
    processed_tables = {}
    
    for table_name, df in seed_tables.items():
        # Convert data types to appropriate formats
        processed_df = df.copy()
        
        # Handle datetime columns
        # datetime_columns = [col for col in df.columns if any(
        #     time_suffix in col.lower() 
        #     for time_suffix in ['_at', '_date', '_time']
        # )]
        # for col in datetime_columns:
        #     processed_df[col] = pd.to_datetime(df[col])
        
        # Handle boolean columns
        bool_columns = [col for col in df.columns if col.startswith('is_')]
        for col in bool_columns:
            processed_df[col] = processed_df[col].astype(bool)
        
        # Handle categorical columns
        categorical_columns = [
            col for col in df.columns 
            if col in ['status', 'currency', 'duration_unit'] or 
            col.endswith('_type') or col.endswith('_status')
        ]
        for col in categorical_columns:
            processed_df[col] = processed_df[col].astype('category')
            
        processed_tables[table_name] = processed_df
    
    # Ensure referential integrity
    for relation in metadata.relationships:
        parent_table = relation['parent_table_name']
        child_table = relation['child_table_name']
        parent_key = relation['parent_primary_key']
        child_key = relation['child_foreign_key']
        
        if parent_table in processed_tables and child_table in processed_tables:
            parent_df = processed_tables[parent_table]
            child_df = processed_tables[child_table]
            
            # Ensure foreign key values exist in parent table
            valid_keys = set(parent_df[parent_key])
            child_df = child_df[child_df[child_key].isin(valid_keys)]
            processed_tables[child_table] = child_df
    
    return processed_tables


# --- Core Logic Functions ---

def call_ai_agent(data_description: str) -> Dict[str, Any]:
    """
    Calls the Gemini API to get the metadata schema and seed data.
    """
    print("Initializing Vertex AI...")
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
    You are a professional data architect. Based on the following description, generate a single JSON object with two top-level keys: 'metadata_dict' and 'seed_tables_dict'.

    Description: "{data_description}"

    CRITICAL INSTRUCTIONS:
    1.  **Metadata:** The `metadata_dict` must follow the older SDV format.
    2.  **Relationships:** Define relationships in a top-level 'relationships' list. Each relationship MUST use these four keys:
    - **`parent_table_name`**: The name of the parent table.
    - **`child_table_name`**: The name of the child table.
    - **`parent_primary_key`**: The primary key of the parent.
    - **`child_foreign_key`**: The foreign key of the child.
    3.  **CRITICAL NAMING:** All table names used in the `metadata_dict` and as keys in the `seed_tables_dict` **MUST be identical and in all uppercase** (e.g., 'CUSTOMERS', 'PRODUCTS').
    4.  **Seed Data:** Provide 15 realistic seed records for each table, ensuring referential integrity.
    5.  **Output:** Return ONLY the complete JSON object.
"""

    print("Generating schema with Gemini...")
    try:
        response = model.generate_content(prompt)
        print(response)
        return json.loads(response.text)
    except json.JSONDecodeError:
        print(f"ERROR: Gemini returned malformed JSON.\nResponse Text: {response.text}")
        raise HTTPException(status_code=500, detail="AI agent returned invalid JSON. Please try a different prompt.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during AI call: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while communicating with the AI agent: {e}")

def generate_sdv_data(num_records: int, metadata_dict: Dict[str, Any], seed_tables_dict: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Uses SDV to generate synthetic data using the stable legacy API.
    """
    print(metadata_dict['tables'])

    try:
        # --- Single-Table Logic ---
        if len(metadata_dict['tables']) == 1:
            table_name = list(metadata_dict['tables'].keys())[0]
            print(f"Using single-table synthesizer for '{table_name}'...")
            
            seed_table_df = pd.DataFrame.from_records(seed_tables_dict[table_name])
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(seed_table_df)
            
            synthesizer = CTGANSynthesizer(metadata)
            synthesizer.fit(seed_table_df)
            synthetic_table = synthesizer.sample(num_rows=num_records)
            return {table_name: synthetic_table}

        # --- Multi-Table Logic using legacy classes ---
        else:
            print("Multi-table detected. Using HMASynthesizer with MultiTableMetadata.")
            
            # Clean metadata and optimize relationships
            metadata_dict = optimize_relationships(metadata_dict)
            
            # Remove unsupported keys
            for table_meta in metadata_dict.get('tables', {}).values():
                for column_meta in table_meta.get('columns', {}).values():
                    if 'subtype' in column_meta:
                        del column_meta['subtype']
                        print(f"Cleaned unsupported 'subtype' key.")

            # Initialize metadata
            metadata = MultiTableMetadata.load_from_dict(metadata_dict)
            print(metadata)

            # Create and preprocess seed tables
            seed_tables = {}
            metadata_name_map = {name.lower(): name for name in metadata.tables}

            for seed_name, seed_data in seed_tables_dict.items():
                lower_seed_name = seed_name.lower()
                if lower_seed_name in metadata_name_map:
                    correct_name = metadata_name_map[lower_seed_name]
                    seed_tables[correct_name] = pd.DataFrame.from_records(seed_data)
                    print(f"Matched seed table '{seed_name}' to metadata table '{correct_name}'.")

            # Preprocess tables for better synthesis
            seed_tables = preprocess_tables(seed_tables, metadata)
            
            # Initialize and fit synthesizer with optimized parameters
            synthesizer = HMASynthesizer(
                metadata,
                locales=['en_US'] # Specify locale for consistent data generation
              )
            
            print("Fitting synthesizer...")
            synthesizer.fit(seed_tables)
            
            print(f"Sampling {num_records} records...")
            try:
                # First attempt: Try generating all data at once
                return synthesizer.sample(num_records)
            except Exception as e:
                print(f"Full sampling failed, trying batched approach: {e}")
                # Second attempt: Generate data in smaller batches
                synthetic_data = {}
                batch_size = max(1, num_records // 10)
                remaining = num_records
                
                while remaining > 0:
                    current_batch = min(batch_size, remaining)
                    batch_data = synthesizer.sample(current_batch)
                    
                    # Merge batch data with existing synthetic data
                    for table_name, df in batch_data.items():
                        if table_name not in synthetic_data:
                            synthetic_data[table_name] = df
                        else:
                            synthetic_data[table_name] = pd.concat([synthetic_data[table_name], df])
                    
                    remaining -= current_batch
                
                return synthetic_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SDV generation failed: {traceback.format_exc()}")


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


# --- API Endpoints ---

@app.post("/api/v1/generate")
async def generate_data_endpoint(request: GenerateRequest):
    """
    Generates synthetic data based on a description and caches it.
    """
    global generated_data_cache
    
    ai_output = call_ai_agent(request.data_description)
    synthetic_data = generate_sdv_data(
        request.num_records,
        ai_output["metadata_dict"],
        ai_output["seed_tables_dict"]
    )
    
    total_records = sum(len(df) for df in synthetic_data.values())
    
    generated_data_cache["data"] = synthetic_data
    generated_data_cache["metadata"] = {
        "description": request.data_description,
        "total_records_generated": total_records,
        "tables": {
            name: {"rows": len(df), "columns": df.columns.tolist()}
            for name, df in synthetic_data.items()
        }
    }
    
    return {
        "status": "success",
        "message": f"Generated {total_records} records across {len(synthetic_data)} tables.",
        "metadata": generated_data_cache["metadata"],
        "next_step": "Use GET /api/v1/sample?table_name=<table_name> to preview data."
    }


@app.get("/api/v1/sample")
async def sample_data_endpoint(table_name: str, sample_size: int = 100):
    """
    Returns a sample of the generated data for a specific table.
    """
    global generated_data_cache
    if not generated_data_cache.get("data"):
        raise HTTPException(status_code=404, detail="No data generated. Use POST /api/v1/generate first.")
    
    if table_name not in generated_data_cache["data"]:
        available = list(generated_data_cache["data"].keys())
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found. Available tables: {available}")

    table_df = generated_data_cache["data"][table_name]
    sample_df = table_df.head(min(sample_size, len(table_df)))
    
    return {
        "table_name": table_name,
        "total_records_in_table": len(table_df),
        "sample_size": len(sample_df),
        "sample_data": sample_df.to_dict('records')
    }


@app.post("/api/v1/store")
async def store_data_endpoint(request: StoreRequest):
    """
    Stores all tables from the cache into Google Cloud Storage.
    """
    global generated_data_cache
    if not generated_data_cache.get("data"):
        raise HTTPException(status_code=404, detail="No data to store. Use POST /api/v1/generate first.")
    
    if not request.confirm_storage:
        raise HTTPException(status_code=400, detail="Storage not confirmed. Set `confirm_storage` to `true`.")

    upload_to_gcs(GCS_BUCKET_NAME, generated_data_cache["data"])
    
    stored_tables_metadata = generated_data_cache["metadata"]["tables"]
    
    # Clear the cache after successful storage
    generated_data_cache = {"data": None, "metadata": None}
    
    return {
        "status": "success",
        "message": f"Successfully stored {len(stored_tables_metadata)} tables to GCS.",
        "bucket": GCS_BUCKET_NAME,
        "stored_tables": stored_tables_metadata
    }
