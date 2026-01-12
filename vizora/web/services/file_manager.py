"""
Temporary file management for uploaded datasets.
"""

import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


class FileManager:
    """
    Manages temporary file storage with automatic cleanup.

    Files are stored in a dedicated directory and automatically
    cleaned up after max_age_hours.
    """

    def __init__(self, base_dir: Optional[str] = None, max_age_hours: int = 1):
        """
        Initialize the file manager.

        Args:
            base_dir: Base directory for uploads. Defaults to system temp.
            max_age_hours: Maximum age of files before cleanup.
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path(tempfile.gettempdir()) / "vizora_uploads"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)

    def save_upload(self, file_content: bytes, job_id: str) -> Path:
        """
        Save uploaded file content to disk.

        Args:
            file_content: Raw bytes of the uploaded file.
            job_id: Unique job identifier for the filename.

        Returns:
            Path to the saved file.
        """
        filepath = self.base_dir / f"{job_id}.csv"
        filepath.write_bytes(file_content)
        return filepath

    def get_file_path(self, job_id: str) -> Optional[Path]:
        """
        Get the path to a job's CSV file if it exists.

        Args:
            job_id: The job identifier.

        Returns:
            Path if file exists, None otherwise.
        """
        filepath = self.base_dir / f"{job_id}.csv"
        return filepath if filepath.exists() else None

    def delete_file(self, job_id: str) -> bool:
        """
        Delete a job's CSV file.

        Args:
            job_id: The job identifier.

        Returns:
            True if file was deleted, False if it didn't exist.
        """
        filepath = self.base_dir / f"{job_id}.csv"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def cleanup_old_files(self) -> int:
        """
        Remove files older than max_age.

        Returns:
            Number of files deleted.
        """
        now = datetime.now()
        deleted = 0

        for f in self.base_dir.glob("*.csv"):
            try:
                file_time = datetime.fromtimestamp(f.stat().st_mtime)
                if file_time < now - self.max_age:
                    f.unlink()
                    deleted += 1
            except (OSError, FileNotFoundError):
                pass

        return deleted


# Global instance
file_manager = FileManager()
