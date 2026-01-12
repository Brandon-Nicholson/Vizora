"""
Authentication Routes

API endpoints for user authentication using Supabase.
Note: Most auth logic is handled client-side with Supabase JS SDK.
These routes provide server-side token verification and user profile access.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from vizora.web.auth.dependencies import get_current_user, get_user_profile
from vizora.web.auth.schemas import UserResponse, UserProfile
from vizora.web.auth.supabase_client import supabase_client

router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get the current authenticated user's information.

    Requires a valid Bearer token in the Authorization header.
    """
    return current_user


@router.get("/profile", response_model=UserProfile)
async def get_current_user_profile(
    profile: UserProfile = Depends(get_user_profile)
):
    """
    Get the current user's full profile including billing information.

    Returns:
        - User info (id, email, name)
        - Subscription tier (free/pro)
        - Monthly usage count
        - Monthly limit
    """
    return profile


@router.post("/logout")
async def logout(current_user: UserResponse = Depends(get_current_user)):
    """
    Server-side logout endpoint.

    Note: The actual logout is handled client-side by Supabase JS SDK.
    This endpoint can be used to perform server-side cleanup if needed.
    """
    return {"message": "Logged out successfully"}


@router.get("/status")
async def auth_status():
    """
    Check if authentication is properly configured.

    Returns configuration status without exposing sensitive details.
    """
    is_configured = supabase_client.is_configured()

    return {
        "auth_enabled": is_configured,
        "provider": "supabase" if is_configured else None,
        "message": (
            "Authentication is configured and ready"
            if is_configured
            else "Authentication is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env"
        ),
    }
