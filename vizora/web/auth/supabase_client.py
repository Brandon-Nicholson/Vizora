"""
Supabase Client Wrapper

Initializes and provides access to the Supabase client for authentication
and database operations.
"""

import os
from functools import lru_cache
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


@lru_cache()
def get_supabase_client() -> Client:
    """
    Create and cache a Supabase client instance.

    Returns:
        Supabase Client instance

    Raises:
        ValueError: If Supabase credentials are not configured
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Use service role for backend

    if not url or not key or url == "your-supabase-url":
        raise ValueError(
            "Supabase credentials not configured. "
            "Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env"
        )

    return create_client(url, key)


# Lazy-loaded client instance
class SupabaseClientProxy:
    """
    Proxy class for lazy-loading the Supabase client.
    Allows the app to start even if Supabase is not configured.
    """

    _client: Client | None = None
    _initialized: bool = False

    def _get_client(self) -> Client:
        if not self._initialized:
            try:
                self._client = get_supabase_client()
            except ValueError:
                self._client = None
            self._initialized = True

        if self._client is None:
            raise ValueError(
                "Supabase client not available. "
                "Please configure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
            )
        return self._client

    @property
    def auth(self):
        return self._get_client().auth

    @property
    def table(self):
        return self._get_client().table

    @property
    def from_(self):
        return self._get_client().from_

    def is_configured(self) -> bool:
        """Check if Supabase is properly configured."""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        return bool(url and key and url != "your-supabase-url")


# Global client instance
supabase_client = SupabaseClientProxy()
