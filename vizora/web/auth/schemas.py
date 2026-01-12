"""
Authentication Schemas

Pydantic models for authentication requests and responses.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr


class TokenPayload(BaseModel):
    """JWT token payload structure."""
    sub: str  # User ID
    email: Optional[str] = None
    exp: Optional[int] = None
    iat: Optional[int] = None


class UserResponse(BaseModel):
    """User information returned to the client."""
    id: str
    email: str
    name: Optional[str] = None
    created_at: Optional[datetime] = None
    tier: str = "free"  # free, pro


class UserProfile(BaseModel):
    """Extended user profile with billing info."""
    id: str
    email: str
    name: Optional[str] = None
    tier: str = "free"
    stripe_customer_id: Optional[str] = None
    monthly_usage: int = 0
    monthly_limit: int = 5
    current_period_start: Optional[datetime] = None


class SignUpRequest(BaseModel):
    """Sign up request payload."""
    email: EmailStr
    password: str
    name: Optional[str] = None


class SignInRequest(BaseModel):
    """Sign in request payload."""
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    """Authentication response with tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse
