"""
Analysis endpoints for Vizora Web API.
"""

import io
import pandas as pd
from uuid import uuid4
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Body, Header
from fastapi.responses import JSONResponse, Response

from vizora.web.models.responses import (
    JobStatus,
    JobCreatedResponse,
    ProgressInfo,
    AnalysisResult
)
from vizora.web.services.file_manager import file_manager
from vizora.web.services.analysis import analysis_service
from vizora.web.services.inference import (
    load_artifacts,
    load_metadata,
    list_models,
    delete_artifacts,
    validate_and_prepare_features,
    predict_with_optional_proba
)
from vizora.web.services.pdf_generator import pdf_generator
from vizora.web.auth.supabase_client import supabase_client
from vizora.web.billing.service import billing_service


router = APIRouter(prefix="/api", tags=["analysis"])


# In-memory job store
# For production, replace with Redis or similar
job_store: dict[str, JobStatus] = {}

# Store job metadata (user_id, goal, mode) for PDF export
job_metadata: dict[str, dict] = {}


def get_user_id_from_token(authorization: Optional[str]) -> Optional[str]:
    """Extract user ID from authorization token if valid."""
    if not authorization or not supabase_client.is_configured():
        return None

    try:
        token = authorization.replace("Bearer ", "")
        user_response = supabase_client.auth.get_user(token)
        if user_response and user_response.user:
            return user_response.user.id
    except Exception:
        pass

    return None


def run_analysis_task(
    job_id: str,
    mode: str,
    goal: str,
    target_column: Optional[str],
    forecast_horizon: Optional[int] = None,
    forecast_frequency: Optional[str] = None,
    date_column: Optional[str] = None
) -> None:
    """
    Background task that runs the full analysis pipeline.

    Args:
        job_id: Unique job identifier.
        mode: Analysis mode (eda, predictive, hybrid, forecast).
        goal: User's analysis goal.
        target_column: Target column name (optional).
        forecast_horizon: Number of periods to forecast (forecast mode).
        forecast_frequency: Forecast frequency (daily, weekly, monthly).
        date_column: Date column name for time series.
    """
    def update_progress(step: str, percentage: int):
        job_store[job_id].progress = ProgressInfo(
            current_step=step,
            percentage=percentage
        )

    try:
        # Update status to running
        job_store[job_id].status = "running"
        update_progress("Loading dataset...", 0)

        # Load CSV
        filepath = file_manager.get_file_path(job_id)
        if not filepath:
            raise FileNotFoundError(f"CSV file not found for job {job_id}")

        df = pd.read_csv(filepath)

        # Run analysis
        result = analysis_service.run_analysis(
            df=df,
            mode=mode,
            goal=goal,
            target_column=target_column,
            progress_callback=update_progress,
            run_id=job_id,
            forecast_horizon=forecast_horizon,
            forecast_frequency=forecast_frequency,
            date_column=date_column
        )

        # Store result
        job_store[job_id].status = "completed"
        job_store[job_id].result = result

    except Exception as e:
        job_store[job_id].status = "failed"
        job_store[job_id].error_message = str(e)

    finally:
        # Cleanup temp file
        file_manager.delete_file(job_id)


@router.post("/analyze", response_model=JobCreatedResponse)
async def start_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV dataset file"),
    mode: str = Form(..., description="Analysis mode: eda, predictive, hybrid, or forecast"),
    goal: str = Form(..., description="User's analysis goal"),
    target_column: Optional[str] = Form(None, description="Target column name (optional)"),
    forecast_horizon: Optional[int] = Form(None, description="Number of periods to forecast"),
    forecast_frequency: Optional[str] = Form(None, description="Forecast frequency: daily, weekly, monthly"),
    date_column: Optional[str] = Form(None, description="Date column name for time series"),
    authorization: Optional[str] = Header(None, description="Bearer token")
) -> JobCreatedResponse:
    """
    Start a new analysis job.

    Uploads the CSV file and queues a background analysis task.
    Returns immediately with a job_id for status polling.
    """
    # Get user ID from token (optional - for usage tracking)
    user_id = get_user_id_from_token(authorization)

    # Check usage limits if user is authenticated
    if user_id:
        can_analyze, reason = billing_service.can_run_analysis(user_id)
        if not can_analyze:
            raise HTTPException(
                status_code=402,  # Payment Required
                detail=reason
            )

    # Validate mode
    if mode not in ("eda", "predictive", "hybrid", "forecast"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be eda, predictive, hybrid, or forecast."
        )

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    # Validate goal length
    if len(goal) < 10:
        raise HTTPException(
            status_code=400,
            detail="Goal must be at least 10 characters."
        )

    if mode in ("predictive", "hybrid") and not target_column:
        raise HTTPException(
            status_code=400,
            detail="Target column is required for predictive or hybrid analysis."
        )

    # Generate job ID
    job_id = str(uuid4())

    # Save uploaded file
    content = await file.read()
    file_manager.save_upload(content, job_id)

    # Initialize job status
    job_store[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        progress=ProgressInfo(current_step="Queued", percentage=0)
    )

    # Store job metadata for PDF export
    job_metadata[job_id] = {
        "user_id": user_id,
        "goal": goal,
        "mode": mode,
        "target_column": target_column,
        "forecast_horizon": forecast_horizon,
        "forecast_frequency": forecast_frequency,
        "date_column": date_column,
    }

    # Log usage if user is authenticated
    if user_id:
        billing_service.log_usage(user_id, job_id, mode)

    # Queue background task
    background_tasks.add_task(
        run_analysis_task,
        job_id=job_id,
        mode=mode,
        goal=goal,
        target_column=target_column,
        forecast_horizon=forecast_horizon,
        forecast_frequency=forecast_frequency,
        date_column=date_column
    )

    return JobCreatedResponse(job_id=job_id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """
    Get the status of an analysis job.

    Use this endpoint to poll for job completion.
    When status is "completed", the result field will contain the analysis results.
    """
    if job_id not in job_store:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found."
        )

    return job_store[job_id]


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict:
    """
    Cancel/delete a job.

    Removes the job from the store and cleans up any temp files.
    Note: Cannot cancel a running job, only queued or completed ones.
    """
    if job_id not in job_store:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found."
        )

    job = job_store[job_id]

    if job.status == "running":
        raise HTTPException(
            status_code=400,
            detail="Cannot cancel a running job."
        )

    # Cleanup
    file_manager.delete_file(job_id)
    del job_store[job_id]

    return {"message": f"Job {job_id} deleted."}


@router.post("/runs/{run_id}/predict_csv")
async def predict_csv(
    run_id: str,
    file: UploadFile = File(..., description="CSV dataset for scoring")
) -> Response:
    """
    Run batch predictions against a CSV and return a downloadable CSV.
    """
    try:
        model, meta = load_artifacts(run_id)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "run_not_found"})
    except Exception:
        return JSONResponse(status_code=500, content={"error": "model_load_failed"})

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid_csv"})

    X, missing = validate_and_prepare_features(df, meta)
    if missing:
        return JSONResponse(status_code=400, content={"error": "missing_columns", "missing": missing})

    try:
        preds, probs = predict_with_optional_proba(model, X, meta)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "invalid_features", "detail": str(e)})

    output_df = df.copy()
    output_df["prediction"] = preds
    if probs is not None:
        output_df["probability"] = probs

    csv_content = output_df.to_csv(index=False)
    headers = {"Content-Disposition": f"attachment; filename=predictions_{run_id}.csv"}
    return Response(content=csv_content, media_type="text/csv", headers=headers)


@router.post("/runs/{run_id}/predict")
async def predict_json(
    run_id: str,
    payload: dict | list[dict] = Body(...)
) -> dict:
    """
    Run predictions on JSON rows and return prediction lists.
    """
    try:
        model, meta = load_artifacts(run_id)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "run_not_found"})
    except Exception:
        return JSONResponse(status_code=500, content={"error": "model_load_failed"})

    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        return JSONResponse(status_code=400, content={"error": "invalid_payload"})

    df = pd.DataFrame(rows)
    X, missing = validate_and_prepare_features(df, meta)
    if missing:
        return JSONResponse(status_code=400, content={"error": "missing_columns", "missing": missing})

    try:
        preds, probs = predict_with_optional_proba(model, X, meta)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "invalid_features", "detail": str(e)})

    response = {"run_id": run_id, "predictions": preds}
    if probs is not None:
        response["probabilities"] = probs
    return response


@router.get("/models/{model_id}")
async def get_model(model_id: str) -> dict:
    try:
        meta = load_metadata(model_id)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "model_not_found"})
    except Exception:
        return JSONResponse(status_code=500, content={"error": "model_load_failed"})
    return meta


@router.get("/models")
async def get_models() -> list[dict]:
    return list_models()


@router.delete("/models/{model_id}")
async def delete_model(model_id: str) -> dict:
    deleted = delete_artifacts(model_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": "model_not_found"})
    return {"status": "deleted", "model_id": model_id}


@router.get("/jobs/{job_id}/export-pdf")
async def export_pdf(
    job_id: str,
    authorization: Optional[str] = Header(None, description="Bearer token")
) -> Response:
    """
    Export the analysis results as a PDF report.

    Requires the job to be completed.
    Returns a downloadable PDF file.
    """
    if job_id not in job_store:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found."
        )

    job = job_store[job_id]

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}"
        )

    if not job.result:
        raise HTTPException(
            status_code=400,
            detail="No results available for this job."
        )

    user_id = get_user_id_from_token(authorization)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required to export PDF."
        )

    if not billing_service.can_export_pdf(user_id):
        raise HTTPException(
            status_code=402,
            detail="Upgrade to Pro to export PDF reports."
        )

    # Extract data from result
    result = job.result
    figures = [
        {
            'type': fig.type,
            'name': fig.name,
            'base64_png': fig.base64_png
        }
        for fig in result.figures
    ] if result.figures else []

    # Build metadata from job_metadata if available
    stored_meta = job_metadata.get(job_id, {})
    metadata = {
        'goal': stored_meta.get('goal', 'Data Analysis'),
        'mode': stored_meta.get('mode', 'Analysis').title(),
        'dataset_info': {
            'rows': 'N/A',
            'columns': 'N/A',
            'target': stored_meta.get('target_column'),
        }
    }

    # Try to extract mode from plan
    if result.plan:
        if 'modeling' in result.plan or 'preprocessing' in result.plan:
            if 'eda' in result.plan:
                metadata['mode'] = 'Hybrid'
            else:
                metadata['mode'] = 'Predictive'
        elif 'analysis' in result.plan or 'eda' in result.plan:
            metadata['mode'] = 'EDA'

    # Generate PDF
    try:
        pdf_bytes = pdf_generator.generate(
            figures=figures,
            metrics=result.metrics,
            summary_markdown=result.summary_markdown or '',
            plan=result.plan or {},
            metadata=metadata
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF: {str(e)}"
        )

    # Return PDF as downloadable file
    headers = {
        "Content-Disposition": f"attachment; filename=vizora_report_{job_id[:8]}.pdf"
    }
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers=headers
    )
