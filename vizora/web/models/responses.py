"""
Response models for Vizora Web API.
"""

from typing import Optional, Literal, Any
from pydantic import BaseModel, Field


class ProgressInfo(BaseModel):
    """Progress information for a running job."""

    current_step: str = Field(..., description="Current step being executed")
    percentage: int = Field(..., ge=0, le=100, description="Completion percentage")


class FigureData(BaseModel):
    """A single visualization figure."""

    id: str = Field(..., description="Unique figure identifier")
    type: str = Field(..., description="Figure type (histogram, boxplot, etc.)")
    name: str = Field(..., description="Figure name/title")
    base64_png: str = Field(..., description="Base64-encoded PNG image data")


class AnalysisResult(BaseModel):
    """Complete analysis results."""

    figures: list[FigureData] = Field(default_factory=list, description="Generated visualizations")
    metrics: Optional[dict[str, Any]] = Field(None, description="Model performance metrics")
    summary_markdown: str = Field(..., description="AI-generated summary in markdown")
    plan: dict[str, Any] = Field(..., description="Execution plan from orchestrator")
    errors: list[str] = Field(default_factory=list, description="Any warnings/errors during execution")
    preprocessing: dict[str, Any] = Field(default_factory=dict, description="Preprocessing artifacts")


class JobStatus(BaseModel):
    """Status of an analysis job."""

    job_id: str = Field(..., description="Unique job identifier")
    status: Literal["queued", "running", "completed", "failed"] = Field(
        ...,
        description="Current job status"
    )
    progress: Optional[ProgressInfo] = Field(None, description="Progress info (when running)")
    result: Optional[AnalysisResult] = Field(None, description="Results (when completed)")
    error_message: Optional[str] = Field(None, description="Error details (when failed)")


class JobCreatedResponse(BaseModel):
    """Response when a new job is created."""

    job_id: str = Field(..., description="Unique job identifier")
    status: Literal["queued"] = Field(default="queued", description="Initial status")


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy"] = "healthy"
    version: str = "1.0.0"
