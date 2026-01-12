"""
Google Sheets Service

Handles reading data from Google Sheets.
"""

import io
from typing import Optional

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from vizora.web.google.oauth import google_oauth


class SheetsService:
    """
    Service for interacting with Google Sheets API.
    """

    def _get_service(self, user_id: str):
        """
        Get authenticated Sheets service for user.

        Args:
            user_id: User ID

        Returns:
            Google Sheets API service or None
        """
        credentials = google_oauth.get_credentials(user_id)
        if not credentials:
            return None

        return build("sheets", "v4", credentials=credentials)

    def _get_drive_service(self, user_id: str):
        """
        Get authenticated Drive service for user.

        Args:
            user_id: User ID

        Returns:
            Google Drive API service or None
        """
        credentials = google_oauth.get_credentials(user_id)
        if not credentials:
            return None

        return build("drive", "v3", credentials=credentials)

    def list_spreadsheets(self, user_id: str, page_size: int = 20) -> list:
        """
        List user's Google Spreadsheets.

        Args:
            user_id: User ID
            page_size: Number of results to return

        Returns:
            List of spreadsheet metadata
        """
        drive_service = self._get_drive_service(user_id)
        if not drive_service:
            return []

        try:
            results = (
                drive_service.files()
                .list(
                    q="mimeType='application/vnd.google-apps.spreadsheet'",
                    pageSize=page_size,
                    fields="files(id, name, modifiedTime, owners)",
                    orderBy="modifiedTime desc",
                )
                .execute()
            )

            files = results.get("files", [])
            return [
                {
                    "id": f["id"],
                    "name": f["name"],
                    "modified_at": f.get("modifiedTime"),
                    "owner": f.get("owners", [{}])[0].get("displayName", "Unknown"),
                }
                for f in files
            ]

        except HttpError as e:
            print(f"Failed to list spreadsheets: {e}")
            return []

    def get_spreadsheet_info(self, user_id: str, spreadsheet_id: str) -> Optional[dict]:
        """
        Get information about a spreadsheet including sheet names.

        Args:
            user_id: User ID
            spreadsheet_id: Google Spreadsheet ID

        Returns:
            Spreadsheet metadata or None
        """
        service = self._get_service(user_id)
        if not service:
            return None

        try:
            spreadsheet = (
                service.spreadsheets()
                .get(spreadsheetId=spreadsheet_id)
                .execute()
            )

            sheets = [
                {
                    "id": sheet["properties"]["sheetId"],
                    "name": sheet["properties"]["title"],
                    "row_count": sheet["properties"]["gridProperties"].get("rowCount", 0),
                    "column_count": sheet["properties"]["gridProperties"].get("columnCount", 0),
                }
                for sheet in spreadsheet.get("sheets", [])
            ]

            return {
                "id": spreadsheet_id,
                "name": spreadsheet.get("properties", {}).get("title", ""),
                "sheets": sheets,
            }

        except HttpError as e:
            print(f"Failed to get spreadsheet info: {e}")
            return None

    def read_sheet_data(
        self,
        user_id: str,
        spreadsheet_id: str,
        sheet_name: Optional[str] = None,
        range_notation: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Read data from a Google Sheet into a pandas DataFrame.

        Args:
            user_id: User ID
            spreadsheet_id: Google Spreadsheet ID
            sheet_name: Name of sheet to read (defaults to first sheet)
            range_notation: A1 notation range (e.g., "A1:Z100")

        Returns:
            DataFrame with sheet data or None
        """
        service = self._get_service(user_id)
        if not service:
            return None

        try:
            # Build range string
            if sheet_name:
                if range_notation:
                    range_str = f"'{sheet_name}'!{range_notation}"
                else:
                    range_str = f"'{sheet_name}'"
            elif range_notation:
                range_str = range_notation
            else:
                # Get first sheet name
                info = self.get_spreadsheet_info(user_id, spreadsheet_id)
                if info and info["sheets"]:
                    range_str = f"'{info['sheets'][0]['name']}'"
                else:
                    range_str = "Sheet1"

            result = (
                service.spreadsheets()
                .values()
                .get(
                    spreadsheetId=spreadsheet_id,
                    range=range_str,
                    valueRenderOption="UNFORMATTED_VALUE",
                    dateTimeRenderOption="FORMATTED_STRING",
                )
                .execute()
            )

            values = result.get("values", [])

            if not values:
                return pd.DataFrame()

            # First row is headers
            headers = values[0]
            data = values[1:] if len(values) > 1 else []

            # Pad rows to match header length
            padded_data = []
            for row in data:
                if len(row) < len(headers):
                    row = row + [None] * (len(headers) - len(row))
                elif len(row) > len(headers):
                    row = row[:len(headers)]
                padded_data.append(row)

            df = pd.DataFrame(padded_data, columns=headers)
            return df

        except HttpError as e:
            print(f"Failed to read sheet data: {e}")
            return None

    def read_sheet_as_csv(
        self,
        user_id: str,
        spreadsheet_id: str,
        sheet_name: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Read a Google Sheet and return as CSV bytes.

        Args:
            user_id: User ID
            spreadsheet_id: Google Spreadsheet ID
            sheet_name: Name of sheet to read

        Returns:
            CSV bytes or None
        """
        df = self.read_sheet_data(user_id, spreadsheet_id, sheet_name)
        if df is None:
            return None

        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer.getvalue()

    def get_sheet_preview(
        self,
        user_id: str,
        spreadsheet_id: str,
        sheet_name: Optional[str] = None,
        num_rows: int = 5,
    ) -> Optional[dict]:
        """
        Get a preview of sheet data.

        Args:
            user_id: User ID
            spreadsheet_id: Google Spreadsheet ID
            sheet_name: Name of sheet to preview
            num_rows: Number of data rows to include

        Returns:
            Preview data with columns and sample rows
        """
        service = self._get_service(user_id)
        if not service:
            return None

        try:
            # Get limited range for preview
            range_str = f"'{sheet_name}'!A1:Z{num_rows + 1}" if sheet_name else f"A1:Z{num_rows + 1}"

            result = (
                service.spreadsheets()
                .values()
                .get(
                    spreadsheetId=spreadsheet_id,
                    range=range_str,
                    valueRenderOption="FORMATTED_VALUE",
                )
                .execute()
            )

            values = result.get("values", [])

            if not values:
                return {"columns": [], "rows": [], "total_rows": 0}

            headers = values[0]
            data = values[1:] if len(values) > 1 else []

            # Get total row count
            info = self.get_spreadsheet_info(user_id, spreadsheet_id)
            total_rows = 0
            if info and sheet_name:
                for sheet in info["sheets"]:
                    if sheet["name"] == sheet_name:
                        total_rows = sheet["row_count"] - 1  # Subtract header
                        break
            elif info and info["sheets"]:
                total_rows = info["sheets"][0]["row_count"] - 1

            return {
                "columns": headers,
                "rows": data,
                "total_rows": total_rows,
            }

        except HttpError as e:
            print(f"Failed to get sheet preview: {e}")
            return None


# Global instance
sheets_service = SheetsService()
