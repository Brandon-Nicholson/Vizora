"""
Stripe Client Configuration

Initializes and provides access to the Stripe API client.
"""

import os
import stripe
from dotenv import load_dotenv

load_dotenv()


class StripeClient:
    """
    Stripe API client wrapper with lazy initialization.
    """

    def __init__(self):
        self._initialized = False
        self._api_key = None
        self._webhook_secret = None
        self._pro_price_id = None

    def _initialize(self):
        """Initialize Stripe with API key."""
        if self._initialized:
            return

        self._api_key = os.getenv("STRIPE_SECRET_KEY")
        self._webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        self._pro_price_id = os.getenv("STRIPE_PRO_PRICE_ID")

        if self._api_key and not self._api_key.startswith("sk_test_your"):
            stripe.api_key = self._api_key

        self._initialized = True

    def is_configured(self) -> bool:
        """Check if Stripe is properly configured."""
        self._initialize()
        return bool(
            self._api_key
            and self._api_key.startswith("sk_")
            and not self._api_key.startswith("sk_test_your")
        )

    @property
    def api_key(self) -> str | None:
        self._initialize()
        return self._api_key

    @property
    def webhook_secret(self) -> str | None:
        self._initialize()
        return self._webhook_secret

    @property
    def pro_price_id(self) -> str | None:
        self._initialize()
        return self._pro_price_id

    def create_checkout_session(
        self,
        customer_email: str,
        success_url: str,
        cancel_url: str,
        customer_id: str | None = None,
        metadata: dict | None = None,
    ) -> stripe.checkout.Session:
        """
        Create a Stripe Checkout session for Pro subscription.

        Args:
            customer_email: Customer's email address
            success_url: URL to redirect to after successful payment
            cancel_url: URL to redirect to if checkout is cancelled
            customer_id: Existing Stripe customer ID (optional)
            metadata: Additional metadata to attach to the session

        Returns:
            Stripe Checkout Session object
        """
        self._initialize()

        if not self.is_configured():
            raise ValueError("Stripe is not configured")

        if not self._pro_price_id:
            raise ValueError("Pro price ID not configured")

        session_params = {
            "mode": "subscription",
            "payment_method_types": ["card"],
            "line_items": [
                {
                    "price": self._pro_price_id,
                    "quantity": 1,
                }
            ],
            "success_url": success_url,
            "cancel_url": cancel_url,
            "metadata": metadata or {},
        }

        if customer_id:
            session_params["customer"] = customer_id
        else:
            session_params["customer_email"] = customer_email

        return stripe.checkout.Session.create(**session_params)

    def create_customer_portal_session(
        self,
        customer_id: str,
        return_url: str,
    ) -> stripe.billing_portal.Session:
        """
        Create a Stripe Customer Portal session for subscription management.

        Args:
            customer_id: Stripe customer ID
            return_url: URL to redirect to after portal session

        Returns:
            Stripe Billing Portal Session object
        """
        self._initialize()

        if not self.is_configured():
            raise ValueError("Stripe is not configured")

        return stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )

    def construct_webhook_event(
        self,
        payload: bytes,
        signature: str,
    ) -> stripe.Event:
        """
        Construct and verify a Stripe webhook event.

        Args:
            payload: Raw request body
            signature: Stripe signature header

        Returns:
            Verified Stripe Event object

        Raises:
            stripe.error.SignatureVerificationError: If signature is invalid
        """
        self._initialize()

        if not self._webhook_secret:
            raise ValueError("Webhook secret not configured")

        return stripe.Webhook.construct_event(
            payload,
            signature,
            self._webhook_secret,
        )

    def get_subscription(self, subscription_id: str) -> stripe.Subscription:
        """Retrieve a subscription by ID."""
        self._initialize()
        return stripe.Subscription.retrieve(subscription_id)

    def cancel_subscription(self, subscription_id: str) -> stripe.Subscription:
        """Cancel a subscription at period end."""
        self._initialize()
        return stripe.Subscription.modify(
            subscription_id,
            cancel_at_period_end=True,
        )


# Global client instance
stripe_client = StripeClient()
