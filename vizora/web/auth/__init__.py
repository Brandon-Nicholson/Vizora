"""
Vizora Authentication Module

Provides Supabase-based authentication for the Vizora API.
"""

from vizora.web.auth.supabase_client import supabase_client
from vizora.web.auth.dependencies import get_current_user, get_optional_user
from vizora.web.auth.schemas import UserResponse, TokenPayload

__all__ = [
    "supabase_client",
    "get_current_user",
    "get_optional_user",
    "UserResponse",
    "TokenPayload",
]
