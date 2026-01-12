"""
Scheduler Service

Manages scheduled report jobs using APScheduler.
Supports dynamic data sources including Google Sheets.
"""

import io
import os
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from vizora.web.auth.supabase_client import supabase_client
from vizora.web.scheduler.email_service import email_service


class SchedulerService:
    """
    Manages scheduled analysis jobs.
    """

    def __init__(self):
        self._scheduler: Optional[BackgroundScheduler] = None

    @property
    def scheduler(self) -> BackgroundScheduler:
        """Lazy-load scheduler."""
        if self._scheduler is None:
            self._scheduler = BackgroundScheduler()
            self._scheduler.start()
            # Load existing schedules from database
            self._load_schedules()
        return self._scheduler

    def _load_schedules(self):
        """Load active schedules from database on startup."""
        try:
            result = (
                supabase_client.table("schedules")
                .select("*")
                .eq("is_active", True)
                .execute()
            )

            for schedule in result.data or []:
                self._add_job_from_record(schedule)

        except Exception as e:
            print(f"Failed to load schedules: {e}")

    def _add_job_from_record(self, schedule: dict):
        """Add a scheduler job from a database record."""
        schedule_id = schedule["id"]
        frequency = schedule["frequency"]
        cron_expression = schedule.get("cron_expression")

        trigger = self._create_trigger(frequency, cron_expression)
        if trigger:
            self.scheduler.add_job(
                self._execute_scheduled_report,
                trigger=trigger,
                id=schedule_id,
                args=[schedule_id],
                replace_existing=True,
            )

    def _create_trigger(self, frequency: str, cron_expression: Optional[str] = None):
        """Create APScheduler trigger from frequency."""
        if cron_expression:
            return CronTrigger.from_crontab(cron_expression)

        triggers = {
            "daily": IntervalTrigger(days=1),
            "weekly": IntervalTrigger(weeks=1),
            "monthly": CronTrigger(day=1, hour=9),  # 1st of month at 9am
        }
        return triggers.get(frequency)

    def create_schedule(
        self,
        user_id: str,
        name: str,
        job_id: str,
        frequency: str,
        email: str,
        cron_expression: Optional[str] = None,
        hour: int = 9,
        day_of_week: Optional[int] = None,
        spreadsheet_id: Optional[str] = None,
        sheet_name: Optional[str] = None,
        analysis_mode: Optional[str] = None,
        analysis_goal: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> dict:
        """
        Create a new scheduled report.

        Args:
            user_id: Owner user ID
            name: Schedule name
            job_id: Analysis job ID to re-run (or use as template)
            frequency: daily, weekly, monthly, or custom
            email: Email to send reports to
            cron_expression: Custom cron expression (if frequency is custom)
            hour: Hour to run (0-23)
            day_of_week: Day of week for weekly (0=Monday)
            spreadsheet_id: Google Sheet ID for dynamic data source
            sheet_name: Sheet name within the spreadsheet
            analysis_mode: Analysis mode (explore, predict, etc.)
            analysis_goal: User's analysis goal/question
            target_column: Target column for prediction tasks

        Returns:
            Created schedule record
        """
        schedule_id = str(uuid4())

        # Build cron expression if not custom
        if frequency != "custom":
            if frequency == "daily":
                cron_expression = f"0 {hour} * * *"
            elif frequency == "weekly":
                dow = day_of_week or 0
                cron_expression = f"0 {hour} * * {dow}"
            elif frequency == "monthly":
                cron_expression = f"0 {hour} 1 * *"

        # Calculate next run
        next_run = self._calculate_next_run(frequency, cron_expression, hour, day_of_week)

        # Store in database
        record = {
            "id": schedule_id,
            "user_id": user_id,
            "name": name,
            "job_id": job_id,
            "frequency": frequency,
            "cron_expression": cron_expression,
            "email": email,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_active": True,
            "next_run_at": next_run.isoformat() if next_run else None,
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": sheet_name,
            "analysis_mode": analysis_mode,
            "analysis_goal": analysis_goal,
            "target_column": target_column,
        }

        try:
            supabase_client.table("schedules").insert(record).execute()

            # Add to scheduler
            trigger = self._create_trigger(frequency, cron_expression)
            if trigger:
                self.scheduler.add_job(
                    self._execute_scheduled_report,
                    trigger=trigger,
                    id=schedule_id,
                    args=[schedule_id],
                    replace_existing=True,
                )

            # Send confirmation email
            frequency_desc = self._get_frequency_description(frequency, hour, day_of_week)
            email_service.send_schedule_confirmation(
                to_email=email,
                schedule_name=name,
                frequency=frequency_desc,
                next_run=next_run.strftime("%B %d, %Y at %I:%M %p") if next_run else "Soon",
            )

            return record

        except Exception as e:
            print(f"Failed to create schedule: {e}")
            raise

    def _calculate_next_run(
        self,
        frequency: str,
        cron_expression: Optional[str],
        hour: int,
        day_of_week: Optional[int],
    ) -> Optional[datetime]:
        """Calculate when the schedule will next run."""
        now = datetime.now()

        if frequency == "daily":
            next_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run

        elif frequency == "weekly":
            next_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            days_ahead = (day_of_week or 0) - now.weekday()
            if days_ahead <= 0 or (days_ahead == 0 and next_run <= now):
                days_ahead += 7
            return next_run + timedelta(days=days_ahead)

        elif frequency == "monthly":
            next_run = now.replace(day=1, hour=hour, minute=0, second=0, microsecond=0)
            if next_run <= now:
                # Move to next month
                if now.month == 12:
                    next_run = next_run.replace(year=now.year + 1, month=1)
                else:
                    next_run = next_run.replace(month=now.month + 1)
            return next_run

        return None

    def _get_frequency_description(
        self, frequency: str, hour: int, day_of_week: Optional[int]
    ) -> str:
        """Get human-readable frequency description."""
        hour_str = f"{hour:02d}:00"
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        if frequency == "daily":
            return f"Daily at {hour_str}"
        elif frequency == "weekly":
            day = days[day_of_week] if day_of_week is not None else "Monday"
            return f"Every {day} at {hour_str}"
        elif frequency == "monthly":
            return f"1st of each month at {hour_str}"
        return frequency

    def _execute_scheduled_report(self, schedule_id: str):
        """Execute a scheduled report job."""
        try:
            # Get schedule details
            result = (
                supabase_client.table("schedules")
                .select("*")
                .eq("id", schedule_id)
                .single()
                .execute()
            )
            schedule = result.data
            if not schedule or not schedule.get("is_active"):
                return

            # Check if this schedule uses Google Sheets as data source
            spreadsheet_id = schedule.get("spreadsheet_id")

            if spreadsheet_id:
                # Dynamic data source: fetch fresh data from Google Sheets
                self._execute_with_fresh_data(schedule)
            else:
                # Static data source: use stored job result
                self._execute_with_stored_result(schedule)

            # Update last run and next run
            next_run = self._calculate_next_run(
                schedule["frequency"],
                schedule.get("cron_expression"),
                schedule.get("hour", 9),
                schedule.get("day_of_week"),
            )

            supabase_client.table("schedules").update({
                "last_run_at": datetime.now().isoformat(),
                "next_run_at": next_run.isoformat() if next_run else None,
                "run_count": (schedule.get("run_count") or 0) + 1,
                "last_error": None,
                "last_error_at": None,
            }).eq("id", schedule["id"]).execute()

        except Exception as e:
            print(f"Failed to execute scheduled report {schedule_id}: {e}")
            # Log the error
            try:
                supabase_client.table("schedules").update({
                    "last_error": str(e),
                    "last_error_at": datetime.now().isoformat(),
                }).eq("id", schedule_id).execute()
            except Exception:
                pass

    def _execute_with_fresh_data(self, schedule: dict):
        """
        Execute scheduled report by fetching fresh data from Google Sheets.
        Runs a new analysis on the latest data.
        """
        from vizora.web.google.sheets import sheets_service
        from vizora.web.services.pdf_generator import pdf_generator
        from vizora.pipeline import run_pipeline

        user_id = schedule["user_id"]
        spreadsheet_id = schedule["spreadsheet_id"]
        sheet_name = schedule.get("sheet_name")

        # Fetch fresh data from Google Sheets
        df = sheets_service.read_sheet_data(user_id, spreadsheet_id, sheet_name)
        if df is None or df.empty:
            raise Exception("Failed to fetch data from Google Sheets or sheet is empty")

        # Convert DataFrame to CSV bytes for the pipeline
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        csv_bytes = csv_buffer.getvalue()

        # Get analysis parameters
        mode = schedule.get("analysis_mode", "explore")
        goal = schedule.get("analysis_goal", "Analyze the latest data")
        target_column = schedule.get("target_column")

        # Run the analysis pipeline
        result = run_pipeline(
            file_bytes=csv_bytes,
            filename=f"{schedule['name']}_data.csv",
            mode=mode,
            goal=goal,
            target_column=target_column,
        )

        # Generate PDF from fresh analysis
        metadata = {
            "goal": goal,
            "mode": mode,
            "user_email": schedule["email"],
            "data_source": "Google Sheets (Live)",
            "spreadsheet_id": spreadsheet_id,
            "run_date": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        }

        pdf_bytes = pdf_generator.generate(result, metadata)

        # Get summary for email
        summary = result.get("summary_markdown", "Analysis completed on fresh data.")
        if len(summary) > 500:
            summary = summary[:497] + "..."

        # Send email with fresh analysis
        email_service.send_report_email(
            to_email=schedule["email"],
            subject=f"Vizora Report: {schedule['name']} (Latest Data)",
            analysis_name=schedule["name"],
            summary=summary,
            pdf_bytes=pdf_bytes,
        )

    def _execute_with_stored_result(self, schedule: dict):
        """
        Execute scheduled report using stored job result.
        This is the original behavior for static data schedules.
        """
        from vizora.web.services.pdf_generator import pdf_generator

        # Get the original job result
        job_result = (
            supabase_client.table("job_results")
            .select("*")
            .eq("job_id", schedule["job_id"])
            .single()
            .execute()
        )

        if not job_result.data:
            raise Exception(f"No job result found for job_id {schedule['job_id']}")

        # Generate PDF
        result_data = job_result.data.get("result", {})
        metadata = {
            "goal": job_result.data.get("goal", ""),
            "mode": job_result.data.get("mode", ""),
            "user_email": schedule["email"],
        }

        pdf_bytes = pdf_generator.generate(result_data, metadata)

        # Get summary for email
        summary = result_data.get("summary_markdown", "Analysis completed.")
        if len(summary) > 500:
            summary = summary[:497] + "..."

        # Send email
        email_service.send_report_email(
            to_email=schedule["email"],
            subject=f"Vizora Report: {schedule['name']}",
            analysis_name=schedule["name"],
            summary=summary,
            pdf_bytes=pdf_bytes,
        )

    def get_user_schedules(self, user_id: str) -> list:
        """Get all schedules for a user."""
        try:
            result = (
                supabase_client.table("schedules")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            return result.data or []
        except Exception:
            return []

    def get_schedule(self, schedule_id: str, user_id: str) -> Optional[dict]:
        """Get a specific schedule."""
        try:
            result = (
                supabase_client.table("schedules")
                .select("*")
                .eq("id", schedule_id)
                .eq("user_id", user_id)
                .single()
                .execute()
            )
            return result.data
        except Exception:
            return None

    def update_schedule(
        self,
        schedule_id: str,
        user_id: str,
        updates: dict,
    ) -> Optional[dict]:
        """Update a schedule."""
        try:
            # Verify ownership
            existing = self.get_schedule(schedule_id, user_id)
            if not existing:
                return None

            # Update database
            result = (
                supabase_client.table("schedules")
                .update(updates)
                .eq("id", schedule_id)
                .execute()
            )

            # Update scheduler job if active
            updated = result.data[0] if result.data else None
            if updated and updated.get("is_active"):
                self._add_job_from_record(updated)
            elif updated:
                # Remove job if deactivated
                try:
                    self.scheduler.remove_job(schedule_id)
                except Exception:
                    pass

            return updated

        except Exception as e:
            print(f"Failed to update schedule: {e}")
            return None

    def delete_schedule(self, schedule_id: str, user_id: str) -> bool:
        """Delete a schedule."""
        try:
            # Verify ownership
            existing = self.get_schedule(schedule_id, user_id)
            if not existing:
                return False

            # Remove from scheduler
            try:
                self.scheduler.remove_job(schedule_id)
            except Exception:
                pass

            # Delete from database
            supabase_client.table("schedules").delete().eq("id", schedule_id).execute()
            return True

        except Exception:
            return False

    def pause_schedule(self, schedule_id: str, user_id: str) -> bool:
        """Pause a schedule."""
        result = self.update_schedule(schedule_id, user_id, {"is_active": False})
        return result is not None

    def resume_schedule(self, schedule_id: str, user_id: str) -> bool:
        """Resume a paused schedule."""
        result = self.update_schedule(schedule_id, user_id, {"is_active": True})
        return result is not None


# Global instance
scheduler_service = SchedulerService()
