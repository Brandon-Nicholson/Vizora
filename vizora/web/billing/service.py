"""
Billing Service

Business logic for subscription management, usage tracking, and billing operations.
"""

import os
from datetime import datetime
from typing import Optional

from vizora.web.auth.supabase_client import supabase_client
from vizora.web.billing.stripe_client import stripe_client

# Load limits from environment
FREE_TIER_LIMIT = int(os.getenv("FREE_TIER_MONTHLY_LIMIT", "5"))


class BillingService:
    """
    Handles subscription and usage management.
    """

    def get_user_tier(self, user_id: str) -> str:
        """
        Get the user's current subscription tier.

        Args:
            user_id: Supabase user ID

        Returns:
            'free' or 'pro'
        """
        try:
            result = (
                supabase_client.table("subscriptions")
                .select("tier, status")
                .eq("user_id", user_id)
                .single()
                .execute()
            )

            if result.data and result.data.get("status") == "active":
                return result.data.get("tier", "free")
        except Exception:
            result = None

        try:
            profile = (
                supabase_client.table("profiles")
                .select("tier")
                .eq("id", user_id)
                .single()
                .execute()
            )
            if profile.data and profile.data.get("tier"):
                return profile.data.get("tier", "free")
        except Exception:
            pass

        return "free"

    def get_monthly_usage(self, user_id: str) -> int:
        """
        Get the user's usage count for the current month.

        Args:
            user_id: Supabase user ID

        Returns:
            Number of analyses run this month
        """
        try:
            start_of_month = datetime.now().replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )

            result = (
                supabase_client.table("usage_logs")
                .select("id", count="exact")
                .eq("user_id", user_id)
                .gte("created_at", start_of_month.isoformat())
                .execute()
            )

            return result.count or 0
        except Exception:
            return 0

    def get_usage_limit(self, tier: str) -> int:
        """
        Get the monthly usage limit for a tier.

        Args:
            tier: 'free' or 'pro'

        Returns:
            Monthly limit (-1 for unlimited)
        """
        if tier == "pro":
            return -1  # Unlimited
        return FREE_TIER_LIMIT

    def can_run_analysis(self, user_id: str) -> tuple[bool, str]:
        """
        Check if a user can run another analysis.

        Args:
            user_id: Supabase user ID

        Returns:
            Tuple of (allowed, reason)
        """
        tier = self.get_user_tier(user_id)
        limit = self.get_usage_limit(tier)

        if limit == -1:  # Unlimited
            return True, "ok"

        usage = self.get_monthly_usage(user_id)

        if usage >= limit:
            return False, f"Monthly limit of {limit} analyses reached. Upgrade to Pro for unlimited analyses."

        return True, "ok"

    def log_usage(
        self,
        user_id: str,
        job_id: str,
        analysis_mode: str,
    ) -> bool:
        """
        Log an analysis usage event.

        Args:
            user_id: Supabase user ID
            job_id: Analysis job ID
            analysis_mode: Mode (eda, predictive, hybrid)

        Returns:
            True if logged successfully
        """
        try:
            supabase_client.table("usage_logs").insert({
                "user_id": user_id,
                "job_id": job_id,
                "analysis_mode": analysis_mode,
            }).execute()
            return True
        except Exception:
            return False

    def get_subscription_details(self, user_id: str) -> dict:
        """
        Get detailed subscription information for a user.

        Args:
            user_id: Supabase user ID

        Returns:
            Dict with tier, status, usage, limit, etc.
        """
        tier = self.get_user_tier(user_id)
        usage = self.get_monthly_usage(user_id)
        limit = self.get_usage_limit(tier)

        subscription_data = {
            "tier": tier,
            "status": "active",
            "monthly_usage": usage,
            "monthly_limit": limit,
            "stripe_customer_id": None,
            "current_period_end": None,
        }

        try:
            result = (
                supabase_client.table("subscriptions")
                .select("*")
                .eq("user_id", user_id)
                .single()
                .execute()
            )

            if result.data:
                subscription_data["stripe_customer_id"] = result.data.get("stripe_customer_id")
                subscription_data["current_period_end"] = result.data.get("current_period_end")
                subscription_data["status"] = result.data.get("status", "active")
        except Exception:
            pass

        return subscription_data

    def create_or_update_subscription(
        self,
        user_id: str,
        stripe_customer_id: str,
        stripe_subscription_id: str,
        tier: str,
        status: str,
        current_period_end: Optional[datetime] = None,
    ) -> bool:
        """
        Create or update a user's subscription record.

        Args:
            user_id: Supabase user ID
            stripe_customer_id: Stripe customer ID
            stripe_subscription_id: Stripe subscription ID
            tier: Subscription tier
            status: Subscription status
            current_period_end: End of current billing period

        Returns:
            True if successful
        """
        try:
            data = {
                "user_id": user_id,
                "stripe_customer_id": stripe_customer_id,
                "stripe_subscription_id": stripe_subscription_id,
                "tier": tier,
                "status": status,
            }

            if current_period_end:
                data["current_period_end"] = current_period_end.isoformat()

            # Upsert based on user_id
            supabase_client.table("subscriptions").upsert(
                data,
                on_conflict="user_id"
            ).execute()

            # Also update the profiles table tier
            supabase_client.table("profiles").upsert(
                {"id": user_id, "tier": tier},
                on_conflict="id"
            ).execute()

            return True
        except Exception as e:
            print(f"Error updating subscription: {e}")
            return False

    def downgrade_to_free(self, user_id: str) -> bool:
        """
        Downgrade a user to the free tier.

        Args:
            user_id: Supabase user ID

        Returns:
            True if successful
        """
        try:
            supabase_client.table("subscriptions").update({
                "tier": "free",
                "status": "canceled",
            }).eq("user_id", user_id).execute()

            supabase_client.table("profiles").update({
                "tier": "free"
            }).eq("id", user_id).execute()

            return True
        except Exception:
            return False

    def can_export_pdf(self, user_id: str) -> bool:
        """
        Check if a user can export PDF reports (Pro feature).

        Args:
            user_id: Supabase user ID

        Returns:
            True if user is Pro tier
        """
        return self.get_user_tier(user_id) == "pro"


# Global service instance
billing_service = BillingService()
