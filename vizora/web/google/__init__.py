"""
Google Integration Module

Handles Google OAuth2 and Google Sheets API integration.
"""

from vizora.web.google.oauth import google_oauth
from vizora.web.google.sheets import sheets_service

__all__ = ["google_oauth", "sheets_service"]
