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
        cron_expression = schedule.get("cron_expression")

        # Always use cron expression - it should be set during creation
        if not cron_expression:
            # Fallback: generate cron from stored frequency/hour/day_of_week
            frequency = schedule["frequency"]
            hour = schedule.get("hour", 9)
            day_of_week = schedule.get("day_of_week", 0)
            cron_expression = self._generate_cron_expression(frequency, hour, day_of_week)

        if cron_expression:
            trigger = CronTrigger.from_crontab(cron_expression)
            self.scheduler.add_job(
                self._execute_scheduled_report,
                trigger=trigger,
                id=schedule_id,
                args=[schedule_id],
                replace_existing=True,
            )
            print(f"Loaded schedule '{schedule.get('name')}' with cron: {cron_expression}")

    def _generate_cron_expression(self, frequency: str, hour: int, day_of_week: Optional[int]) -> Optional[str]:
        """Generate cron expression from frequency settings."""
        if frequency == "daily":
            return f"0 {hour} * * *"
        elif frequency == "weekly":
            dow = day_of_week if day_of_week is not None else 0
            return f"0 {hour} * * {dow}"
        elif frequency == "monthly":
            return f"0 {hour} 1 * *"
        return None

    def _create_trigger(self, frequency: str, cron_expression: Optional[str] = None):
        """Create APScheduler trigger from frequency."""
        if cron_expression:
            return CronTrigger.from_crontab(cron_expression)

        # Fallback - shouldn't be used since we always generate cron expressions
        triggers = {
            "daily": CronTrigger(hour=9, minute=0),
            "weekly": CronTrigger(day_of_week=0, hour=9, minute=0),
            "monthly": CronTrigger(day=1, hour=9, minute=0),
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

    def _execute_scheduled_report(self, schedule_id: str, raise_on_error: bool = False):
        """
        Execute a scheduled report job.

        Args:
            schedule_id: The schedule ID to execute
            raise_on_error: If True, re-raise exceptions instead of just logging them.
                           Use True for manual "Run Now" triggers to show errors to user.
        """
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
            if not schedule:
                raise Exception(f"Schedule {schedule_id} not found")
            if not schedule.get("is_active"):
                raise Exception(f"Schedule {schedule_id} is paused")

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
            error_msg = str(e)
            print(f"Failed to execute scheduled report {schedule_id}: {error_msg}")
            # Log the error to database
            try:
                supabase_client.table("schedules").update({
                    "last_error": error_msg,
                    "last_error_at": datetime.now().isoformat(),
                }).eq("id", schedule_id).execute()
            except Exception:
                pass

            # Re-raise if requested (for manual triggers)
            if raise_on_error:
                raise

    def _execute_with_fresh_data(self, schedule: dict):
        """
        Execute scheduled report by fetching fresh data from Google Sheets.
        Runs a new analysis on the latest data.
        """
        from vizora.web.google.sheets import sheets_service
        from vizora.web.google.oauth import google_oauth
        from vizora.web.services.pdf_generator import pdf_generator
        from vizora.web.services.analysis import analysis_service

        user_id = schedule["user_id"]
        spreadsheet_id = schedule["spreadsheet_id"]
        sheet_name = schedule.get("sheet_name")

        # Check if user has Google connected
        if not google_oauth.is_connected(user_id):
            raise Exception(
                "Google account not connected. Please reconnect your Google account "
                "in Settings to continue receiving scheduled reports."
            )

        # Fetch fresh data from Google Sheets
        df = sheets_service.read_sheet_data(user_id, spreadsheet_id, sheet_name)
        if df is None:
            raise Exception(
                f"Failed to fetch data from Google Sheets. "
                f"Check that you have access to spreadsheet {spreadsheet_id}"
            )
        if df.empty:
            raise Exception("Google Sheet is empty - no data to analyze")

        # Get analysis parameters
        mode = schedule.get("analysis_mode", "eda")
        # Map legacy mode names to current ones
        mode_mapping = {"explore": "eda", "predict": "predictive"}
        mode = mode_mapping.get(mode, mode)

        goal = schedule.get("analysis_goal", "Analyze the latest data")
        target_column = schedule.get("target_column")

        # Run the analysis using the analysis service
        result = analysis_service.run_analysis(
            df=df,
            mode=mode,
            goal=goal,
            target_column=target_column,
        )

        # Generate PDF from fresh analysis
        # Convert figures to the format expected by pdf_generator
        figures = [
            {
                'type': fig.type,
                'name': fig.name,
                'base64_png': fig.base64_png
            }
            for fig in result.figures
        ] if result.figures else []

        metadata = {
            "goal": goal,
            "mode": mode,
            "user_email": schedule["email"],
            "data_source": "Google Sheets (Live)",
            "spreadsheet_id": spreadsheet_id,
            "run_date": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "target": target_column,
            }
        }

        pdf_bytes = pdf_generator.generate(
            figures=figures,
            metrics=result.metrics,
            summary_markdown=result.summary_markdown or '',
            plan=result.plan or {},
            metadata=metadata
        )

        # Get summary for email
        summary = result.summary_markdown or "Analysis completed on fresh data."
        if len(summary) > 500:
            summary = summary[:497] + "..."

        # Send email with fresh analysis
        if not email_service.is_configured:
            raise Exception(
                "Email service not configured. Set RESEND_API_KEY environment variable."
            )

        email_sent = email_service.send_report_email(
            to_email=schedule["email"],
            subject=f"Vizora Report: {schedule['name']} (Latest Data)",
            analysis_name=schedule["name"],
            summary=summary,
            pdf_bytes=pdf_bytes,
        )

        if not email_sent:
            raise Exception(f"Failed to send email to {schedule['email']}")

    def _execute_with_stored_result(self, schedule: dict):
        """
        Execute scheduled report using stored job result.
        This is the original behavior for static data schedules.

        NOTE: This currently requires results to be stored in the job_results table,
        which is not automatically done by the analysis routes. Consider migrating
        to Google Sheets-based schedules for persistent scheduled reports.
        """
        from vizora.web.services.pdf_generator import pdf_generator

        job_id = schedule.get("job_id")
        if not job_id:
            raise Exception("Schedule has no associated job_id")

        # Try to get the job result from the database
        try:
            job_result = (
                supabase_client.table("job_results")
                .select("*")
                .eq("job_id", job_id)
                .single()
                .execute()
            )
        except Exception as e:
            # Table might not exist or job not found
            raise Exception(
                f"Cannot find stored results for job {job_id}. "
                "Saved Analysis schedules require persisted job results. "
                "Consider using Google Sheets as the data source for scheduled reports."
            )

        if not job_result.data:
            raise Exception(
                f"No stored results found for job {job_id}. "
                "The original analysis results may have been lost. "
                "Consider recreating this schedule with Google Sheets as the data source."
            )

        # Generate PDF
        result_data = job_result.data.get("result", {})

        # Extract figures in the expected format
        figures = result_data.get("figures", [])

        metadata = {
            "goal": job_result.data.get("goal", ""),
            "mode": job_result.data.get("mode", ""),
            "user_email": schedule["email"],
            "dataset_info": result_data.get("dataset_info", {}),
        }

        pdf_bytes = pdf_generator.generate(
            figures=figures,
            metrics=result_data.get("metrics"),
            summary_markdown=result_data.get("summary_markdown", ""),
            plan=result_data.get("plan", {}),
            metadata=metadata
        )

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
