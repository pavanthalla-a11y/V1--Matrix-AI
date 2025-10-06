import asyncio
import json
import logging
import time
import traceback
from typing import Dict, Any
import io
import zipfile

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

from .core.ai import call_ai_agent
from .core.analytics import analyze_data_distribution, collect_synthesis_metrics
from .core.cache import get_cache, update_cache, update_progress
from .core.config import GCP_PROJECT_ID, GCS_BUCKET_NAME
from .core.notifications import notify_user_by_email
from .core.schemas import DesignRequest, SynthesizeRequest, StoreRequest
from .core.sdv import generate_sdv_data_optimized

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Matrix AI - Optimized Synthetic Data Generator",
    description="High-performance AI-powered data generation using Gemini and SDV.",
    version="6.1.0",
)


@app.post("/api/v1/design")
async def design_schema_endpoint(request: DesignRequest):
    """STEP 1: Calls AI to generate metadata and seed data with enhanced error handling."""
    try:
        existing_metadata_json = None
        if request.existing_metadata:
            existing_metadata_json = json.dumps(request.existing_metadata, indent=2)

        ai_output = await asyncio.to_thread(
            call_ai_agent, request.data_description, request.num_records, existing_metadata_json
        )

        update_cache({"design_output": ai_output, "num_records_target": request.num_records})

        return {
            "status": "review_required",
            "message": "AI model has generated the schema and seed data. Please review before proceeding to synthesis.",
            "metadata_preview": ai_output["metadata_dict"],
            "seed_data_preview": {
                table: data[:20] for table, data in ai_output["seed_tables_dict"].items()
            },
        }
    except Exception as e:
        logger.error(f"Error in design endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/synthesize")
async def synthesize_data_endpoint(request: SynthesizeRequest, background_tasks: BackgroundTasks):
    """STEP 2: Starts the long-running synthesis in a background task and returns immediately (202 Accepted)."""
    design_output = get_cache().get("design_output")
    if not design_output:
        raise HTTPException(
            status_code=400, detail="Schema design must be completed (POST /design) before synthesis."
        )

    background_tasks.add_task(run_synthesis_in_background, request)

    update_cache({"synthetic_data": "Processing"})

    return {
        "status": "processing_started",
        "message": "Synthesis started in the background. You will be notified via email when the data is ready to view and store.",
        "target_email": request.user_email,
    }


@app.get("/api/v1/sample")
async def sample_data_endpoint(table_name: str = None, sample_size: int = 20):
    """STEP 3: Returns a sample of the synthesized data for review."""
    cache = get_cache()
    synthetic_data = cache.get("synthetic_data")

    if synthetic_data == "Processing":
        raise HTTPException(status_code=202, detail="Data synthesis is still processing in the background.")

    if not synthetic_data:
        raise HTTPException(status_code=404, detail="Synthesis not complete or failed.")

    if table_name:
        if table_name not in synthetic_data:
            available = list(synthetic_data.keys())
            raise HTTPException(
                status_code=404, detail=f"Table '{table_name}' not found. Available tables: {available}"
            )

        table_df = synthetic_data[table_name]
        sample_df = table_df.head(min(sample_size, len(table_df)))

        return {"table_name": table_name, "sample_data": sample_df.to_dict("records")}
    else:
        all_samples = {}
        for name, df in synthetic_data.items():
            all_samples[name] = df.head(min(sample_size, len(df))).to_dict("records")

        return {
            "status": "success",
            "message": "Returning samples for all generated tables.",
            "all_samples": all_samples,
            "metadata": cache.get("metadata"),
        }


@app.get("/api/v1/reports")
async def reports_endpoint():
    """Returns comprehensive analysis and metrics of the synthesized data."""
    cache = get_cache()
    if not cache.get("synthesis_metrics") or not cache.get("distribution_analysis"):
        raise HTTPException(status_code=404, detail="Reports are not available yet.")

    # Convert numpy types to native Python types for JSON serialization
    synthesis_metrics = convert_numpy_types(cache["synthesis_metrics"])
    distribution_analysis = convert_numpy_types(cache["distribution_analysis"])

    return {
        "synthesis_metrics": synthesis_metrics,
        "distribution_analysis": distribution_analysis,
    }


@app.post("/api/v1/store")
async def store_data_endpoint(request: StoreRequest):
    """STEP 4: Clears the cache after user has downloaded the data."""
    cache = get_cache()
    synthetic_data = cache.get("synthetic_data")

    if synthetic_data == "Processing":
        raise HTTPException(status_code=400, detail="Synthesis is still processing. Please wait.")

    if not synthetic_data:
        raise HTTPException(status_code=404, detail="No synthesized data to store.")

    if not request.confirm_storage:
        raise HTTPException(status_code=400, detail="Storage not confirmed. Set `confirm_storage` to `true`.")

    # Clear cache after user confirms they've downloaded the data
    update_cache({"design_output": None, "synthetic_data": None, "num_records_target": 0})

    return {
        "status": "storage_complete",
        "message": f"Data session cleared. {len(synthetic_data)} tables are ready for download.",
    }


@app.get("/api/v1/progress")
async def progress_endpoint():
    """Returns the current progress of the synthesis task."""
    return get_cache().get("progress", {})


async def run_synthesis_in_background(request: SynthesizeRequest):
    """Runs the synthesis in a separate thread to prevent blocking."""
    try:
        start_time = time.time()
        update_progress("processing", "Starting synthesis", 0)

        synthetic_data = await asyncio.to_thread(
            generate_sdv_data_optimized,
            request.num_records,
            request.metadata_dict,
            request.seed_tables_dict,
            request.batch_size,
            request.use_fast_synthesizer,
        )

        total_records = sum(len(df) for df in synthetic_data.values())
        end_time = time.time()
        duration = end_time - start_time

        update_progress("processing", "Generating analysis report", 95)

        synthesis_params = {
            "batch_size": request.batch_size,
            "use_fast_synthesizer": request.use_fast_synthesizer,
            "num_records": request.num_records,
        }

        synthesis_metrics = collect_synthesis_metrics(
            start_time, end_time, synthetic_data, request.metadata_dict, synthesis_params
        )

        distribution_analysis = analyze_data_distribution(
            synthetic_data, request.seed_tables_dict, request.metadata_dict
        )

        update_cache(
            {
                "synthetic_data": synthetic_data,
                "synthesis_metrics": synthesis_metrics,
                "distribution_analysis": distribution_analysis,
                "metadata": {
                    "total_records_generated": total_records,
                    "generation_time_seconds": duration,
                    "tables": {
                        name: {"rows": len(df), "columns": df.columns.tolist()}
                        for name, df in synthetic_data.items()
                    },
                },
            }
        )

        update_progress("complete", "Synthesis completed", 100, total_records)

        notify_user_by_email(
            request.user_email,
            "Complete",
            f"Optimized synthesis completed in {duration:.1f} seconds! {total_records} records generated.",
        )

    except Exception as e:
        error_detail = f"Optimized SDV synthesis failed: {str(e)}"
        logger.error(error_detail)
        logger.error(traceback.format_exc())
        update_progress("error", error_detail, 0, error=error_detail)
        update_cache({"synthetic_data": None})

        notify_user_by_email(
            request.user_email, "Failed", f"Synthetic data generation failed: {error_detail}"
        )


@app.get("/api/v1/download")
async def download_data_endpoint():
    """Packages and returns all generated data and reports in a single ZIP file."""
    cache = get_cache()
    synthetic_data = cache.get("synthetic_data")
    if not synthetic_data or synthetic_data == "Processing":
        raise HTTPException(status_code=404, detail="No data available to download.")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add synthetic data CSVs
        for table_name, df in synthetic_data.items():
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            zip_file.writestr(f"synthetic_data/{table_name}.csv", csv_buffer.getvalue())

        # Add metadata and reports
        reports = {
            "metadata.json": cache.get("design_output", {}).get("metadata_dict"),
            "seed_data.json": cache.get("design_output", {}).get("seed_tables_dict"),
            "synthesis_metrics.json": convert_numpy_types(cache.get("synthesis_metrics")),
            "distribution_analysis.json": convert_numpy_types(cache.get("distribution_analysis")),
        }
        for filename, content in reports.items():
            if content:
                zip_file.writestr(f"reports/{filename}", json.dumps(content, indent=2))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=matrix_ai_synthetic_data.zip"},
    )
