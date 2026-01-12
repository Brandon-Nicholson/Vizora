"""
Authentication Dependencies

FastAPI dependencies for protecting routes and extracting user information.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Optional
from datetime import datetime

from fastapi import Depends, HTTPException, Header, status
from jose import jwt, JWTError
from dotenv import load_dotenv

from vizora.web.auth.schemas import UserResponse, TokenPayload, UserProfile
from vizora.web.auth.supabase_client import supabase_client

load_dotenv()

# Supabase JWT settings
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
ALGORITHM = "HS256"


def _decode_token_claims(token: str) -> dict:
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="SUPABASE_JWT_SECRET not configured",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify signature but skip exp validation to handle string-based exp values.
    claims = jwt.decode(
        token,
        SUPABASE_JWT_SECRET,
        algorithms=[ALGORITHM],
        options={"verify_aud": False, "verify_exp": False},
    )

    exp = claims.get("exp")
    if isinstance(exp, str) and exp.isdigit():
        exp = int(exp)
        claims["exp"] = exp

    if isinstance(exp, int):
        now_ts = int(datetime.utcnow().timestamp())
        if exp < now_ts:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return claims


def _fetch_user_from_supabase(token: str) -> dict | None:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_key:
        return None

    req = urllib.request.Request(
        f"{supabase_url}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": supabase_key,
        },
    )

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        raise


async def get_current_user(
    authorization: str = Header(..., description="Bearer token")
) -> UserResponse:
    """
    Validate the JWT token and return the current user.

    This dependency should be used for routes that require authentication.

    Args:
        authorization: The Authorization header containing the Bearer token

    Returns:
        UserResponse with user information

    Raises:
        HTTPException: If token is invalid or missing
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.replace("Bearer ", "")

    try:
        # Verify the token with Supabase
        # Supabase handles JWT verification internally
        user_response = supabase_client.auth.get_user(token)

        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user = user_response.user

        user_id = user.id
        email = user.email or ""
        name = user.user_metadata.get("name") if user.user_metadata else None
        created_at = (
            datetime.fromisoformat(user.created_at.replace("Z", "+00:00"))
            if user.created_at
            else None
        )
    except HTTPException:
        raise
    except Exception:
        # If the client fails (e.g. token parsing bug), call Supabase auth API directly.
        try:
            user_data = _fetch_user_from_supabase(token)
            if user_data:
                user_id = user_data.get("id")
                email = user_data.get("email", "")
                user_metadata = user_data.get("user_metadata") or {}
                name = user_metadata.get("name")
                created_at = (
                    datetime.fromisoformat(user_data["created_at"].replace("Z", "+00:00"))
                    if user_data.get("created_at")
                    else None
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        except HTTPException:
            raise
        except Exception:
            # Fallback to local JWT verification for mismatched token parsing.
            try:
                claims = _decode_token_claims(token)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Could not validate credentials: {str(e)}",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            user_id = claims.get("sub")
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token subject",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            email = claims.get("email", "")
            user_metadata = claims.get("user_metadata") or {}
            name = user_metadata.get("name")
            created_at = None

    # Get user tier from profiles table
    tier = "free"
    try:
        profile = (
            supabase_client.table("profiles")
            .select("tier")
            .eq("id", user_id)
            .single()
            .execute()
        )
        if profile.data:
            tier = profile.data.get("tier", "free")
    except Exception:
        # Profile might not exist yet, default to free
        pass

    return UserResponse(
        id=user_id,
        email=email,
        name=name,
        created_at=created_at,
        tier=tier,
    )


async def get_optional_user(
    authorization: Optional[str] = Header(None, description="Bearer token")
) -> Optional[UserResponse]:
    """
    Optionally validate the JWT token and return the current user.

    This dependency should be used for routes where authentication is optional.

    Args:
        authorization: The optional Authorization header

    Returns:
        UserResponse if token is valid, None otherwise
    """
    if not authorization:
        return None

    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None


async def get_user_profile(
    current_user: UserResponse = Depends(get_current_user)
) -> UserProfile:
    """
    Get the full user profile including billing information.

    Args:
        current_user: The authenticated user

    Returns:
        UserProfile with extended information
    """
    try:
        # Get subscription info
        subscription = None
        try:
            subscription = (
                supabase_client.table("subscriptions")
                .select("*")
                .eq("user_id", current_user.id)
                .single()
                .execute()
            )
        except Exception:
            subscription = None

        # Get usage count for current month
        start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        usage = (
            supabase_client.table("usage_logs")
            .select("id", count="exact")
            .eq("user_id", current_user.id)
            .gte("created_at", start_of_month.isoformat())
            .execute()
        )

        tier = "free"
        stripe_customer_id = None
        current_period_start = None
        subscription_status = None

        if subscription and subscription.data:
            subscription_status = subscription.data.get("status")
            if subscription_status == "active":
                tier = subscription.data.get("tier", "free")
            stripe_customer_id = subscription.data.get("stripe_customer_id")
            if subscription.data.get("current_period_start"):
                current_period_start = datetime.fromisoformat(
                    subscription.data["current_period_start"].replace("Z", "+00:00")
                )

        if subscription_status != "active":
            try:
                profile = (
                    supabase_client.table("profiles")
                    .select("tier")
                    .eq("id", current_user.id)
                    .single()
                    .execute()
                )
                if profile.data and profile.data.get("tier"):
                    tier = profile.data.get("tier", "free")
            except Exception:
                pass

        # Set limits based on tier
        free_limit = os.getenv("FREE_TIER_MONTHLY_LIMIT", "5")
        monthly_limit = int(free_limit) if tier == "free" else float("inf")

        return UserProfile(
            id=current_user.id,
            email=current_user.email,
            name=current_user.name,
            tier=tier,
            stripe_customer_id=stripe_customer_id,
            monthly_usage=usage.count or 0,
            monthly_limit=monthly_limit if monthly_limit != float("inf") else -1,  # -1 for unlimited
            current_period_start=current_period_start,
        )

    except Exception:
        # Return basic profile if database query fails
        return UserProfile(
            id=current_user.id,
            email=current_user.email,
            name=current_user.name,
            tier="free",
            monthly_usage=0,
            monthly_limit=5,
        )
