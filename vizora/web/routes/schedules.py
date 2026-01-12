"""
Schedules API Routes

Endpoints for managing scheduled reports.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr

from vizora.web.auth.dependencies import get_current_user
from vizora.web.billing.service import billing_service
from vizora.web.scheduler.service import scheduler_service


router = APIRouter(prefix="/api/schedules", tags=["schedules"])


class CreateScheduleRequest(BaseModel):
    """Request to create a scheduled report."""
    name: str
    job_id: Optional[str] = None  # Optional if using Google Sheets
    frequency: str  # daily, weekly, monthly
    email: EmailStr
    hour: int = 9
    day_of_week: Optional[int] = None  # 0-6, Monday-Sunday
    # Google Sheets data source fields
    spreadsheet_id: Optional[str] = None
    sheet_name: Optional[str] = None
    analysis_mode: Optional[str] = None
    analysis_goal: Optional[str] = None
    target_column: Optional[str] = None


class UpdateScheduleRequest(BaseModel):
    """Request to update a schedule."""
    name: Optional[str] = None
    frequency: Optional[str] = None
    email: Optional[EmailStr] = None
    hour: Optional[int] = None
    day_of_week: Optional[int] = None
    is_active: Optional[bool] = None
    spreadsheet_id: Optional[str] = None
    sheet_name: Optional[str] = None
    analysis_mode: Optional[str] = None
    analysis_goal: Optional[str] = None
    target_column: Optional[str] = None


class ScheduleResponse(BaseModel):
    """Schedule response."""
    id: str
    name: str
    job_id: Optional[str] = None
    frequency: str
    email: str
    hour: int
    day_of_week: Optional[int] = None
    is_active: bool
    next_run_at: Optional[str] = None
    last_run_at: Optional[str] = None
    run_count: int
    created_at: str
    # Google Sheets fields
    spreadsheet_id: Optional[str] = None
    sheet_name: Optional[str] = None
    analysis_mode: Optional[str] = None
    analysis_goal: Optional[str] = None
    target_column: Optional[str] = None
    data_source: Optional[str] = None  # 'model' or 'google_sheets'


@router.post("", response_model=ScheduleResponse)
async def create_schedule(
    request: CreateScheduleRequest,
    user=Depends(get_current_user),
):
    """
    Create a new scheduled report.

    Requires Pro subscription.
    Supports two data source modes:
    - Model-based: Re-runs stored analysis (requires job_id)
    - Google Sheets: Fetches fresh data each run (requires spreadsheet_id)
    """
    user_id = user.id

    # Check if user is Pro
    tier = billing_service.get_user_tier(user_id)
    if tier != "pro":
        raise HTTPException(
            status_code=402,
            detail="Scheduled reports require a Pro subscription",
        )

    # Validate data source - must have either job_id or spreadsheet_id
    if not request.job_id and not request.spreadsheet_id:
        raise HTTPException(
            status_code=400,
            detail="Must provide either job_id (for saved model) or spreadsheet_id (for Google Sheets)",
        )

    # Validate Google Sheets requirements
    if request.spreadsheet_id and not request.analysis_mode:
        raise HTTPException(
            status_code=400,
            detail="analysis_mode is required when using Google Sheets as data source",
        )

    # Validate frequency
    if request.frequency not in ["daily", "weekly", "monthly"]:
        raise HTTPException(
            status_code=400,
            detail="Frequency must be daily, weekly, or monthly",
        )

    # Validate hour
    if not 0 <= request.hour <= 23:
        raise HTTPException(
            status_code=400,
            detail="Hour must be between 0 and 23",
        )

    # Validate day_of_week for weekly
    if request.frequency == "weekly" and request.day_of_week is not None:
        if not 0 <= request.day_of_week <= 6:
            raise HTTPException(
                status_code=400,
                detail="Day of week must be between 0 (Monday) and 6 (Sunday)",
            )

    try:
        schedule = scheduler_service.create_schedule(
            user_id=user_id,
            name=request.name,
            job_id=request.job_id or "",
            frequency=request.frequency,
            email=request.email,
            hour=request.hour,
            day_of_week=request.day_of_week,
            spreadsheet_id=request.spreadsheet_id,
            sheet_name=request.sheet_name,
            analysis_mode=request.analysis_mode,
            analysis_goal=request.analysis_goal,
            target_column=request.target_column,
        )

        # Determine data source type
        data_source = "google_sheets" if schedule.get("spreadsheet_id") else "model"

        return ScheduleResponse(
            id=schedule["id"],
            name=schedule["name"],
            job_id=schedule.get("job_id"),
            frequency=schedule["frequency"],
            email=schedule["email"],
            hour=schedule.get("hour", 9),
            day_of_week=schedule.get("day_of_week"),
            is_active=schedule.get("is_active", True),
            next_run_at=schedule.get("next_run_at"),
            last_run_at=schedule.get("last_run_at"),
            run_count=schedule.get("run_count", 0),
            created_at=schedule.get("created_at", ""),
            spreadsheet_id=schedule.get("spreadsheet_id"),
            sheet_name=schedule.get("sheet_name"),
            analysis_mode=schedule.get("analysis_mode"),
            analysis_goal=schedule.get("analysis_goal"),
            target_column=schedule.get("target_column"),
            data_source=data_source,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_schedules(user=Depends(get_current_user)):
    """
    List all schedules for the current user.
    """
    schedules = scheduler_service.get_user_schedules(user.id)
    return {"schedules": schedules}


@router.get("/{schedule_id}")
async def get_schedule(
    schedule_id: str,
    user=Depends(get_current_user),
):
    """
    Get a specific schedule.
    """
    schedule = scheduler_service.get_schedule(schedule_id, user.id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return schedule


@router.patch("/{schedule_id}")
async def update_schedule(
    schedule_id: str,
    request: UpdateScheduleRequest,
    user=Depends(get_current_user),
):
    """
    Update a schedule.
    """
    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    # Validate hour if provided
    if "hour" in updates and not 0 <= updates["hour"] <= 23:
        raise HTTPException(
            status_code=400,
            detail="Hour must be between 0 and 23",
        )

    # Validate day_of_week if provided
    if "day_of_week" in updates and updates["day_of_week"] is not None:
        if not 0 <= updates["day_of_week"] <= 6:
            raise HTTPException(
                status_code=400,
                detail="Day of week must be between 0 and 6",
            )

    result = scheduler_service.update_schedule(schedule_id, user.id, updates)
    if not result:
        raise HTTPException(status_code=404, detail="Schedule not found")

    return result


@router.delete("/{schedule_id}")
async def delete_schedule(
    schedule_id: str,
    user=Depends(get_current_user),
):
    """
    Delete a schedule.
    """
    success = scheduler_service.delete_schedule(schedule_id, user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"status": "deleted"}


@router.post("/{schedule_id}/pause")
async def pause_schedule(
    schedule_id: str,
    user=Depends(get_current_user),
):
    """
    Pause a schedule.
    """
    success = scheduler_service.pause_schedule(schedule_id, user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"status": "paused"}


@router.post("/{schedule_id}/resume")
async def resume_schedule(
    schedule_id: str,
    user=Depends(get_current_user),
):
    """
    Resume a paused schedule.
    """
    success = scheduler_service.resume_schedule(schedule_id, user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"status": "resumed"}


@router.post("/{schedule_id}/run-now")
async def run_schedule_now(
    schedule_id: str,
    user=Depends(get_current_user),
):
    """
    Manually trigger a scheduled report immediately.
    """
    schedule = scheduler_service.get_schedule(schedule_id, user.id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    try:
        # Execute the report directly
        scheduler_service._execute_scheduled_report(schedule_id)
        return {"status": "triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
