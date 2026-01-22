"""
Google Integration API Routes

Endpoints for Google OAuth and Google Sheets access.
"""

import io
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from vizora.web.auth.dependencies import get_current_user
from vizora.web.billing.service import billing_service
from vizora.web.google.oauth import google_oauth
from vizora.web.google.sheets import sheets_service


def get_frontend_url() -> str:
    """Get the primary frontend URL from FRONTEND_URL env var.

    Handles comma-separated list of URLs by returning only the first one.
    """
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
    # Handle comma-separated URLs - use the first one
    if "," in frontend_url:
        frontend_url = frontend_url.split(",")[0].strip()
    return frontend_url


router = APIRouter(prefix="/api/google", tags=["google"])


class AuthUrlResponse(BaseModel):
    """Google OAuth authorization URL response."""
    auth_url: str


class ConnectionStatus(BaseModel):
    """Google connection status."""
    connected: bool
    configured: bool


class SpreadsheetInfo(BaseModel):
    """Spreadsheet information."""
    id: str
    name: str
    modified_at: Optional[str] = None
    owner: Optional[str] = None


class SheetInfo(BaseModel):
    """Sheet within a spreadsheet."""
    id: int
    name: str
    row_count: int
    column_count: int


class SpreadsheetDetails(BaseModel):
    """Detailed spreadsheet information."""
    id: str
    name: str
    sheets: list[SheetInfo]


class SheetPreview(BaseModel):
    """Sheet data preview."""
    columns: list[str]
    rows: list[list]
    total_rows: int


@router.get("/status", response_model=ConnectionStatus)
async def get_connection_status(user=Depends(get_current_user)):
    """
    Check if Google Sheets is connected for the current user.
    """
    return ConnectionStatus(
        connected=google_oauth.is_connected(user.id),
        configured=google_oauth.is_configured,
    )


@router.get("/connect", response_model=AuthUrlResponse)
async def get_auth_url(user=Depends(get_current_user)):
    """
    Get Google OAuth authorization URL.

    Requires Pro subscription.
    """
    user_id = user.id

    # Check if user is Pro
    tier = billing_service.get_user_tier(user_id)
    if tier != "pro":
        raise HTTPException(
            status_code=402,
            detail="Google Sheets integration requires a Pro subscription",
        )

    if not google_oauth.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Google OAuth is not configured",
        )

    try:
        auth_url = google_oauth.get_authorization_url(user_id)
        return AuthUrlResponse(auth_url=auth_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/callback")
async def oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    error: Optional[str] = Query(None),
):
    """
    Handle Google OAuth callback.

    Redirects to frontend with success or error status.
    """
    frontend_url = get_frontend_url()

    if error:
        return RedirectResponse(f"{frontend_url}/sheets?error={error}")

    try:
        # State contains user_id
        user_id = state
        google_oauth.exchange_code(code, user_id)
        return RedirectResponse(f"{frontend_url}/sheets?connected=true")
    except Exception as e:
        return RedirectResponse(f"{frontend_url}/sheets?error={str(e)}")


@router.post("/disconnect")
async def disconnect_google(user=Depends(get_current_user)):
    """
    Disconnect Google account.
    """
    success = google_oauth.disconnect(user.id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to disconnect")
    return {"status": "disconnected"}


@router.get("/spreadsheets", response_model=list[SpreadsheetInfo])
async def list_spreadsheets(
    user=Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),
):
    """
    List user's Google Spreadsheets.
    """
    if not google_oauth.is_connected(user.id):
        raise HTTPException(status_code=400, detail="Google not connected")

    spreadsheets = sheets_service.list_spreadsheets(user.id, limit)
    return [SpreadsheetInfo(**s) for s in spreadsheets]


@router.get("/spreadsheets/{spreadsheet_id}", response_model=SpreadsheetDetails)
async def get_spreadsheet(
    spreadsheet_id: str,
    user=Depends(get_current_user),
):
    """
    Get spreadsheet details including sheet names.
    """
    if not google_oauth.is_connected(user.id):
        raise HTTPException(status_code=400, detail="Google not connected")

    info = sheets_service.get_spreadsheet_info(user.id, spreadsheet_id)
    if not info:
        raise HTTPException(status_code=404, detail="Spreadsheet not found")

    return SpreadsheetDetails(
        id=info["id"],
        name=info["name"],
        sheets=[SheetInfo(**s) for s in info["sheets"]],
    )


@router.get("/spreadsheets/{spreadsheet_id}/preview", response_model=SheetPreview)
async def preview_sheet(
    spreadsheet_id: str,
    user=Depends(get_current_user),
    sheet: Optional[str] = Query(None, description="Sheet name"),
    rows: int = Query(5, ge=1, le=20),
):
    """
    Get a preview of sheet data.
    """
    if not google_oauth.is_connected(user.id):
        raise HTTPException(status_code=400, detail="Google not connected")

    preview = sheets_service.get_sheet_preview(
        user.id,
        spreadsheet_id,
        sheet,
        rows,
    )

    if not preview:
        raise HTTPException(status_code=404, detail="Sheet not found")

    return SheetPreview(**preview)


@router.post("/spreadsheets/{spreadsheet_id}/import")
async def import_sheet(
    spreadsheet_id: str,
    user=Depends(get_current_user),
    sheet: Optional[str] = Query(None, description="Sheet name to import"),
):
    """
    Import a Google Sheet as CSV data for analysis.

    Returns the sheet data as a virtual file that can be used for analysis.
    """
    if not google_oauth.is_connected(user.id):
        raise HTTPException(status_code=400, detail="Google not connected")

    # Get spreadsheet info for filename
    info = sheets_service.get_spreadsheet_info(user.id, spreadsheet_id)
    if not info:
        raise HTTPException(status_code=404, detail="Spreadsheet not found")

    # Read sheet data
    df = sheets_service.read_sheet_data(user.id, spreadsheet_id, sheet)
    if df is None:
        raise HTTPException(status_code=500, detail="Failed to read sheet data")

    # Convert to CSV
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # Generate filename
    filename = f"{info['name']}"
    if sheet:
        filename += f"_{sheet}"
    filename += ".csv"
    filename = filename.replace(" ", "_")

    return {
        "filename": filename,
        "data": csv_data.decode("utf-8"),
        "rows": len(df),
        "columns": list(df.columns),
    }
