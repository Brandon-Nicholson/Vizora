"""
Request models for Vizora Web API.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Request body for starting an analysis job."""

    mode: Literal["eda", "predictive", "hybrid"] = Field(
        ...,
        description="Analysis mode: eda (exploratory), predictive (modeling), or hybrid (both)"
    )
    goal: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="User's analysis goal/objective"
    )
    target_column: Optional[str] = Field(
        None,
        description="Target column name for predictive modeling (optional)"
    )
