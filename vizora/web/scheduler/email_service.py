"""
Email Service

Handles sending scheduled report emails using Resend.
"""

import os
from typing import Optional
import base64


class EmailService:
    """
    Email service for sending scheduled reports.
    Uses Resend API for delivery.
    """

    def __init__(self):
        self._client = None
        self._from_email = os.getenv("RESEND_FROM_EMAIL", "reports@vizora.ai")

    @property
    def client(self):
        """Lazy-load Resend client."""
        if self._client is None:
            api_key = os.getenv("RESEND_API_KEY")
            if api_key:
                import resend
                resend.api_key = api_key
                self._client = resend
        return self._client

    @property
    def is_configured(self) -> bool:
        """Check if email service is configured."""
        return self.client is not None

    def send_report_email(
        self,
        to_email: str,
        subject: str,
        analysis_name: str,
        summary: str,
        pdf_bytes: Optional[bytes] = None,
    ) -> bool:
        """
        Send a scheduled report email.

        Args:
            to_email: Recipient email address
            subject: Email subject
            analysis_name: Name of the analysis
            summary: Brief summary text
            pdf_bytes: Optional PDF report attachment

        Returns:
            True if sent successfully
        """
        if not self.is_configured:
            print("Email service not configured - skipping email send")
            return False

        try:
            html_content = self._build_report_html(analysis_name, summary)

            params = {
                "from": self._from_email,
                "to": [to_email],
                "subject": subject,
                "html": html_content,
            }

            # Add PDF attachment if provided
            if pdf_bytes:
                params["attachments"] = [
                    {
                        "filename": f"{analysis_name.replace(' ', '_')}_report.pdf",
                        "content": base64.b64encode(pdf_bytes).decode("utf-8"),
                    }
                ]

            self.client.Emails.send(params)
            return True

        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

    def _build_report_html(self, analysis_name: str, summary: str) -> str:
        """Build HTML email content."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #1a1a2e;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px 12px 0 0;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .content {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 0 0 12px 12px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }}
        .footer {{
            text-align: center;
            margin-top: 20px;
            color: #666;
            font-size: 14px;
        }}
        .cta {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Vizora Scheduled Report</h1>
        <p>{analysis_name}</p>
    </div>
    <div class="content">
        <div class="summary">
            <h3>Analysis Summary</h3>
            <p>{summary}</p>
        </div>
        <p>Your scheduled analysis has completed. The full PDF report is attached to this email.</p>
        <center>
            <a href="https://vizora.ai/results" class="cta">View in Vizora</a>
        </center>
    </div>
    <div class="footer">
        <p>This is an automated report from Vizora.</p>
        <p>To manage your scheduled reports, visit your dashboard.</p>
    </div>
</body>
</html>
"""

    def send_schedule_confirmation(
        self,
        to_email: str,
        schedule_name: str,
        frequency: str,
        next_run: str,
    ) -> bool:
        """
        Send confirmation when a schedule is created.

        Args:
            to_email: Recipient email
            schedule_name: Name of the schedule
            frequency: Schedule frequency description
            next_run: Next scheduled run time

        Returns:
            True if sent successfully
        """
        if not self.is_configured:
            return False

        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #1a1a2e;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .badge {{
            display: inline-block;
            background: #e8f5e9;
            color: #2e7d32;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .detail {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }}
        .detail-label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <div class="card">
        <span class="badge">Schedule Created</span>
        <h2>{schedule_name}</h2>
        <p>Your scheduled report has been set up successfully.</p>

        <div class="detail">
            <div class="detail-label">Frequency</div>
            <strong>{frequency}</strong>
        </div>

        <div class="detail">
            <div class="detail-label">Next Run</div>
            <strong>{next_run}</strong>
        </div>

        <p>You'll receive an email with your PDF report at each scheduled time.</p>
    </div>
</body>
</html>
"""
            self.client.Emails.send({
                "from": self._from_email,
                "to": [to_email],
                "subject": f"Schedule Created: {schedule_name}",
                "html": html_content,
            })
            return True

        except Exception as e:
            print(f"Failed to send confirmation email: {e}")
            return False


# Global instance
email_service = EmailService()
