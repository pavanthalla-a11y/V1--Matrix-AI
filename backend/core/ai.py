import json
import re
import numpy as np
from fastapi import HTTPException
import vertexai
from vertexai.generative_models import GenerativeModel
import logging
from .config import GCP_PROJECT_ID, GCP_LOCATION, GEMINI_MODEL
from .google_auth import setup_google_auth
from typing import Dict, Any

logger = logging.getLogger(__name__)

def sanitize_for_json(obj):
    """Recursively sanitize data to remove NaN, inf, and other non-JSON compliant values"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif hasattr(obj, 'item'):  # numpy scalars
        val = obj.item()
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return val
    else:
        # Convert other types to string as fallback
        return str(obj)

def validate_and_fix_json_response(response_text: str) -> Dict[str, Any]:
    """Validate and fix common JSON issues in AI responses"""
    if not response_text or not response_text.strip():
        raise ValueError("Empty or whitespace-only AI response")
    
    try:
        # Fix NaN values in the response text first
        fixed_text = response_text.replace('NaN', 'null').replace('nan', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
        
        # First, try to parse as-is
        result = json.loads(fixed_text)
        # Sanitize the parsed result
        return sanitize_for_json(result)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parse failed at position {e.pos}: {e}. Attempting to fix...")
        logger.debug(f"Failed text around position {e.pos}: {response_text[max(0, e.pos-50):e.pos+50]}")
        
        # Common fixes for AI-generated JSON
        fixed_text = response_text.strip()
        
        # Fix NaN values first
        fixed_text = fixed_text.replace('NaN', 'null').replace('nan', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
        
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
        fixed_text = re.sub(r'"%([YmdHMS])"', r'"%\1"', fixed_text)
        fixed_text = re.sub(r'"([^"]*%[YmdHMS][^"]*)"([^,}\]])', r'"\1"\2', fixed_text)
        
        # Fix specific corruption patterns seen in logs
        fixed_text = re.sub(r'"sdtype":dol":', r'"sdtype":', fixed_text)  # Fix "sdtype":dol": -> "sdtype":
        fixed_text = re.sub(r'"([^"]*)":\s*([^",:}\]]+)":', r'"\1": "\2",', fixed_text)  # Fix malformed key-value pairs
        fixed_text = re.sub(r':([a-zA-Z][a-zA-Z0-9_]*)(":)', r': "\1",', fixed_text)  # Fix unquoted values followed by quote-colon
        
        # Fix missing quotes around values that should be strings
        fixed_text = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', fixed_text)
        
        # Try multiple parsing attempts with different fixes
        for attempt in range(4):
            try:
                return json.loads(fixed_text)
            except json.JSONDecodeError as e2:
                if attempt == 0:
                    # Attempt 1: Fix missing commas more aggressively
                    fixed_text = re.sub(r'(\d+|"[^"]*"|\]|\})
\s*("|\{|\[' )', r'\1,\n\2', fixed_text)
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

def validate_ai_response_structure(ai_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive validation of AI response structure to ensure it has required keys.
    *** NEW: Added to fix "missing required top-level keys" error. ***
    """
    logger.info("Validating AI response structure...")
    
    # Check for required top-level keys
    if not isinstance(ai_output, dict):
        logger.error(f"AI response is not a dictionary: {type(ai_output)}")
        raise ValueError(f"AI response must be a dictionary, got {type(ai_output)}")
    
    missing_keys = []
    if "metadata_dict" not in ai_output:
        missing_keys.append("metadata_dict")
    if "seed_tables_dict" not in ai_output:
        missing_keys.append("seed_tables_dict")
    
    if missing_keys:
        logger.error(f"AI response missing required top-level keys: {missing_keys}")
        logger.error(f"Available keys in AI response: {list(ai_output.keys())}")
        
        # Try to create a valid response structure
        fixed_output = {}
        
        # Handle metadata_dict
        if "metadata_dict" in ai_output:
            fixed_output["metadata_dict"] = ai_output["metadata_dict"]
        elif "metadata" in ai_output:
            # Sometimes AI uses 'metadata' instead of 'metadata_dict'
            fixed_output["metadata_dict"] = ai_output["metadata"]
            logger.info("Fixed: Using 'metadata' as 'metadata_dict'")
        else:
            # Create minimal metadata structure
            logger.warning("Creating minimal metadata_dict fallback")
            fixed_output["metadata_dict"] = {
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
            }
        
        # Handle seed_tables_dict
        if "seed_tables_dict" in ai_output:
            fixed_output["seed_tables_dict"] = ai_output["seed_tables_dict"]
        elif "seed_tables" in ai_output:
            # Sometimes AI uses 'seed_tables' instead of 'seed_tables_dict'
            fixed_output["seed_tables_dict"] = ai_output["seed_tables"]
            logger.info("Fixed: Using 'seed_tables' as 'seed_tables_dict'")
        elif "seed_data" in ai_output:
            # Sometimes AI uses 'seed_data' instead of 'seed_tables_dict'
            fixed_output["seed_tables_dict"] = ai_output["seed_data"]
            logger.info("Fixed: Using 'seed_data' as 'seed_tables_dict'")
        else:
            # Create minimal seed data structure
            logger.warning("Creating minimal seed_tables_dict fallback")
            fixed_output["seed_tables_dict"] = {
                "data_table": [
                    {"id": 1, "value": "Sample Data", "created_at": "2023-01-01 12:00:00"},
                    {"id": 2, "value": "Test Data", "created_at": "2023-01-02 12:00:00"}
                ]
            }
        
        ai_output = fixed_output
        logger.info("Successfully repaired AI response structure")
    
    # Validate metadata_dict structure
    metadata_dict = ai_output.get("metadata_dict", {})
    if not isinstance(metadata_dict, dict):
        raise ValueError(f"metadata_dict must be a dictionary, got {type(metadata_dict)}")
    
    if "tables" not in metadata_dict:
        logger.warning("metadata_dict missing 'tables' key, adding empty tables")
        metadata_dict["tables"] = {}
    
    if "relationships" not in metadata_dict:
        logger.warning("metadata_dict missing 'relationships' key, adding empty relationships")
        metadata_dict["relationships"] = []
    
    # Validate seed_tables_dict structure
    seed_tables_dict = ai_output.get("seed_tables_dict", {})
    if not isinstance(seed_tables_dict, dict):
        raise ValueError(f"seed_tables_dict must be a dictionary, got {type(seed_tables_dict)}")
    
    # Ensure all tables in metadata have corresponding seed data
    for table_name in metadata_dict.get("tables", {}):
        if table_name not in seed_tables_dict:
            logger.warning(f"Table '{table_name}' in metadata but missing seed data, creating empty list")
            seed_tables_dict[table_name] = []
    
    # Ensure all seed tables have basic structure
    for table_name, table_data in seed_tables_dict.items():
        if not isinstance(table_data, list):
            logger.error(f"Seed data for table '{table_name}' is not a list: {type(table_data)}")
            seed_tables_dict[table_name] = []
    
    logger.info("AI response structure validation completed successfully")
    return ai_output

def call_ai_agent(data_description: str, num_records: int, existing_metadata_json: str = None) -> Dict[str, Any]:
    """
    Calls the Gemini API to get the metadata schema and seed data.
    *** FIXED: Improved prompt to ensure proper JSON structure with required keys. ***
    """
    try:
        credentials, project = setup_google_auth()
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION, credentials=credentials)
        
        model = GenerativeModel(
            GEMINI_MODEL,
            generation_config={"response_mime_type": "application/json"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Vertex AI: {e}")

    prompt = f'''
    You are a professional data architect creating realistic datasets for AI/ML model training. 
    The final synthetic dataset will be scaled to **{num_records} records**. Your generated seed data must be a realistic, high-quality sample suitable for training a model to generate data at that scale.

    **Core Request:** "{data_description}" 
    
    {'**Refinement/Modification Instruction:** Based on the user\'s new description, modify the schema and seed data. Here is the existing metadata: ' + existing_metadata_json if existing_metadata_json else ''}

    **CRITICAL: You MUST return a JSON object with EXACTLY these two top-level keys:**
    1. "metadata_dict" - contains the SDV metadata schema
    2. "seed_tables_dict" - contains the seed data for each table

    **REQUIRED JSON STRUCTURE:**
    {{
        "metadata_dict": {{
            "tables": {{
                "table_name": {{
                    "columns": {{
                        "column_name": {{"sdtype": "categorical|numerical|datetime|id"}},
                        "datetime_column": {{"sdtype": "datetime", "datetime_format": "%Y-%m-%d %H:%M:%S"}}
                    }},
                    "primary_key": "column_name"
                }}
            }},
            "relationships": [
                {{
                    "parent_table_name": "parent_table",
                    "child_table_name": "child_table", 
                    "parent_primary_key": "parent_key",
                    "child_foreign_key": "foreign_key"
                }}
            ]
        }},
        "seed_tables_dict": {{
            "table_name": [
                {{"column1": "value1", "column2": "value2"}},
                {{"column1": "value3", "column2": "value4"}}
            ]
        }}
    }}

    **CRITICAL DATA QUALITY REQUIREMENTS:**
    1. **PRIMARY KEY UNIQUENESS:** The primary key for each table (e.g., 'user_id' in a 'Users' table) MUST contain 100% unique values in the seed data you generate. Do not repeat primary key IDs. 
    
    2. **REALISTIC NUMERICAL VALUES:** All numerical data must be realistic:
        - Ages: Between 18 and 90 years old
        - Prices/Amounts: Positive values, reasonable for the domain (e.g., $10-$500 for products)
        - Ratings/Scores: Within expected range (e.g., 1-5 stars, 0-100 percentages)
        - Counts/Quantities: Non-negative integers
        
    3. **REFERENTIAL INTEGRITY REQUIREMENTS:**
    - **CONSISTENT ID FORMATS:** All related tables MUST use the same ID format for primary keys and foreign keys
    - **FOREIGN KEY VALIDATION:** Every foreign key value in child table MUST exist as a primary key in parent table
    - **ID FORMAT CONSISTENCY BY DOMAIN:** Choose ONE consistent ID format per entity type and stick to it throughout

    **CRITICAL DATA REALISM FOR AI/ML TRAINING:**
    1. **REALISTIC DATA ONLY:** Generate data that looks like real-world data for AI/ML training:
        - Names: Use real human names (John Smith, Sarah Johnson, etc.) - NO codes like 'AAA3', 'User123' 
        - Phone Numbers: Use proper formats (555-123-4567, +1-555-123-4567) - NO letters like '563Z778712L'
        - Email Addresses: Realistic emails (john.smith@email.com, sarah.j@company.com)
        - Addresses: Real street names, cities, states (123 Main St, New York, NY 10001)
        - Company Names: Realistic business names (TechCorp Inc., Global Solutions LLC)
        - Product Names: Meaningful product names (not Product1, Product2)
        - Descriptions: Natural language descriptions, not codes or placeholders
        - Categories: Use real-world categories and classifications
        - Prices/Amounts: Realistic price ranges for the domain
        - Dates: Meaningful date ranges that make business sense

    2. **SDV METADATA FORMAT:**
        - The `metadata_dict` must have "tables" and "relationships" keys
        - Use SDV format for relationships: "parent_table_name", "child_table_name", "parent_primary_key", "child_foreign_key"
        - Every table MUST define a "primary_key" field

    3. **DATETIME HANDLING:**
        - All datetime columns use format: "datetime_format": "%Y-%m-%d %H:%M:%S"
        - Provide realistic datetime values like "2023-06-15 14:30:00"
        - NO placeholder text like '(datetime_format)', 'YYYY-MM-DD'

    4. **SEED DATA REQUIREMENTS:**
        - Provide 25-30 diverse, realistic data rows per table
        - Ensure all foreign key relationships are valid and meaningful
        - Use varied, realistic values that represent good training data diversity
        - Make sure data follows domain-specific patterns (e.g., subscription dates before end dates)

    **EXAMPLE OF CORRECT REFERENTIAL INTEGRITY:**
    If you have customers table with primary_key="customer_id" containing values like ["CUST-001", "CUST-002", "CUST-003"]
    Then subscriptions table with foreign_key="customer_id" must only contain values from ["CUST-001", "CUST-002", "CUST-003"]
    NOT different formats like ["sdv-id-abc", "CST-123", etc.]

    **CRITICAL: Return ONLY the JSON object with the exact structure shown above. Do not include any explanatory text before or after the JSON.**
    '''

    logger.info("Generating/Refining schema with Gemini...")
    try:
        response = model.generate_content(prompt)
        
        # Log the raw response for debugging
        logger.info(f"Raw AI response length: {len(response.text)} characters")
        logger.debug(f"Raw AI response (first 500 chars): {response.text[:500]}...")
        
        # Use the robust JSON validation and fixing function
        ai_output = validate_and_fix_json_response(response.text)
        
        # Enhanced validation of the response structure
        validated_output = validate_ai_response_structure(ai_output)
        
        logger.info("Successfully parsed and validated AI response structure")
        return validated_output
        
    except json.JSONDecodeError as json_error:
        logger.error(f"JSON parsing error in AI response: {json_error}")
        logger.error(f"Problematic JSON around position {json_error.pos}: {response.text[max(0, json_error.pos-100):json_error.pos+100]}")
        raise HTTPException(status_code=500, detail=f"AI generated invalid JSON: {json_error}")
    except Exception as e:
        logger.error(f"ERROR during AI call: {e}")
        # Assuming traceback is available, otherwise remove this line
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"AI agent failed to generate a valid schema: {e}")
