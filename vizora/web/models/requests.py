"""
Request models for Vizora Web API.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Request body for starting an analysis job."""

    mode: Literal["eda", "predictive", "hybrid", "forecast"] = Field(
        ...,
        description="Analysis mode: eda (exploratory), predictive (modeling), hybrid (both), or forecast (time series)"
    )
    goal: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="User's analysis goal/objective"
    )
    target_column: Optional[str] = Field(
        None,
        description="Target column name for predictive modeling or forecast target (optional)"
    )
    # Forecast-specific fields
    forecast_horizon: Optional[int] = Field(
        None,
        ge=1,
        le=365,
        description="Number of periods to forecast (for forecast mode)"
    )
    forecast_frequency: Optional[Literal["daily", "weekly", "monthly"]] = Field(
        None,
        description="Forecast frequency (for forecast mode)"
    )
    date_column: Optional[str] = Field(
        None,
        description="Date/time column name (for forecast mode, auto-detected if not provided)"
    )
