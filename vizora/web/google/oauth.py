"""
Google OAuth2 Service

Handles Google OAuth2 authentication flow for Google Sheets access.
"""

import os
from typing import Optional
from urllib.parse import urlencode

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from vizora.web.auth.supabase_client import supabase_client


# OAuth2 scopes needed for Google Sheets read access
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


class GoogleOAuth:
    """
    Manages Google OAuth2 authentication.
    """

    def __init__(self):
        self._client_id = os.getenv("GOOGLE_CLIENT_ID")
        self._client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self._redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/google/callback")

    @property
    def is_configured(self) -> bool:
        """Check if Google OAuth is configured."""
        return bool(self._client_id and self._client_secret)

    def get_authorization_url(self, user_id: str) -> str:
        """
        Generate Google OAuth2 authorization URL.

        Args:
            user_id: User ID to include in state

        Returns:
            Authorization URL to redirect user to
        """
        if not self.is_configured:
            raise ValueError("Google OAuth not configured")

        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self._redirect_uri],
                }
            },
            scopes=SCOPES,
        )
        flow.redirect_uri = self._redirect_uri

        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=user_id,  # Include user_id in state for callback
        )

        return authorization_url

    def exchange_code(self, code: str, user_id: str) -> dict:
        """
        Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback
            user_id: User ID from state

        Returns:
            Token data including access_token and refresh_token
        """
        if not self.is_configured:
            raise ValueError("Google OAuth not configured")

        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self._redirect_uri],
                }
            },
            scopes=SCOPES,
        )
        flow.redirect_uri = self._redirect_uri

        flow.fetch_token(code=code)
        credentials = flow.credentials

        # Store tokens in database
        token_data = {
            "access_token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": list(credentials.scopes) if credentials.scopes else SCOPES,
        }

        self._store_tokens(user_id, token_data)

        return token_data

    def _store_tokens(self, user_id: str, token_data: dict):
        """Store Google tokens in database."""
        try:
            supabase_client.table("google_tokens").upsert(
                {
                    "user_id": user_id,
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token"),
                    "scopes": token_data.get("scopes", SCOPES),
                },
                on_conflict="user_id",
            ).execute()
        except Exception as e:
            print(f"Failed to store Google tokens: {e}")
            raise

    def get_credentials(self, user_id: str) -> Optional[Credentials]:
        """
        Get Google credentials for a user.

        Args:
            user_id: User ID

        Returns:
            Google Credentials object or None if not connected
        """
        try:
            result = (
                supabase_client.table("google_tokens")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )

            if not result.data:
                return None

            row = result.data[0] if isinstance(result.data, list) else result.data
            if not row:
                return None

            credentials = Credentials(
                token=row["access_token"],
                refresh_token=row.get("refresh_token"),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self._client_id,
                client_secret=self._client_secret,
                scopes=row.get("scopes", SCOPES),
            )

            # Refresh if expired
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(self._get_request())
                # Update stored token
                self._store_tokens(user_id, {
                    "access_token": credentials.token,
                    "refresh_token": credentials.refresh_token,
                    "scopes": list(credentials.scopes) if credentials.scopes else SCOPES,
                })

            return credentials

        except Exception as e:
            print(f"Failed to get Google credentials: {e}")
            return None

    def _get_request(self):
        """Get a google.auth.transport.Request for token refresh."""
        import google.auth.transport.requests
        return google.auth.transport.requests.Request()

    def is_connected(self, user_id: str) -> bool:
        """
        Check if user has connected their Google account.

        Args:
            user_id: User ID

        Returns:
            True if connected
        """
        credentials = self.get_credentials(user_id)
        return credentials is not None

    def disconnect(self, user_id: str) -> bool:
        """
        Disconnect user's Google account.

        Args:
            user_id: User ID

        Returns:
            True if disconnected successfully
        """
        try:
            supabase_client.table("google_tokens").delete().eq("user_id", user_id).execute()
            return True
        except Exception:
            return False


# Global instance
google_oauth = GoogleOAuth()
