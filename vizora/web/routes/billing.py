"""
Billing Routes

API endpoints for Stripe billing, checkout, and subscription management.
"""

import os
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vizora.web.auth.dependencies import get_current_user


def get_frontend_url() -> str:
    """Get the primary frontend URL from FRONTEND_URL env var.

    Handles comma-separated list of URLs by returning only the first one.
    """
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
    # Handle comma-separated URLs - use the first one
    if "," in frontend_url:
        frontend_url = frontend_url.split(",")[0].strip()
    return frontend_url
from vizora.web.auth.schemas import UserResponse
from vizora.web.billing.stripe_client import stripe_client
from vizora.web.billing.service import billing_service

router = APIRouter(prefix="/api/billing", tags=["billing"])


class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str


class SubscriptionResponse(BaseModel):
    tier: str
    status: str
    monthly_usage: int
    monthly_limit: int
    current_period_end: str | None


class PortalResponse(BaseModel):
    portal_url: str


@router.get("/status")
async def billing_status():
    """
    Check if billing is properly configured.
    """
    is_configured = stripe_client.is_configured()

    return {
        "billing_enabled": is_configured,
        "provider": "stripe" if is_configured else None,
        "message": (
            "Billing is configured and ready"
            if is_configured
            else "Billing is not configured. Set STRIPE_SECRET_KEY in .env"
        ),
    }


@router.post("/create-checkout", response_model=CheckoutResponse)
async def create_checkout(
    current_user: UserResponse = Depends(get_current_user),
):
    """
    Create a Stripe Checkout session for Pro subscription.

    Redirects the user to Stripe's hosted checkout page.
    """
    if not stripe_client.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Billing is not configured"
        )

    # Get frontend URL for redirects
    frontend_url = get_frontend_url()

    # Check if user already has an active subscription
    subscription = billing_service.get_subscription_details(current_user.id)
    if subscription["tier"] == "pro" and subscription["status"] == "active":
        raise HTTPException(
            status_code=400,
            detail="You already have an active Pro subscription"
        )

    try:
        session = stripe_client.create_checkout_session(
            customer_email=current_user.email,
            customer_id=subscription.get("stripe_customer_id"),
            success_url=f"{frontend_url}/mode?checkout=success",
            cancel_url=f"{frontend_url}/pricing?checkout=cancelled",
            metadata={"user_id": current_user.id},
        )

        return CheckoutResponse(
            checkout_url=session.url,
            session_id=session.id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create checkout session: {str(e)}"
        )


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    current_user: UserResponse = Depends(get_current_user),
):
    """
    Get the current user's subscription status and usage.
    """
    details = billing_service.get_subscription_details(current_user.id)

    return SubscriptionResponse(
        tier=details["tier"],
        status=details["status"],
        monthly_usage=details["monthly_usage"],
        monthly_limit=details["monthly_limit"],
        current_period_end=details.get("current_period_end"),
    )


@router.get("/usage")
async def get_usage(
    current_user: UserResponse = Depends(get_current_user),
):
    """
    Get the current user's usage statistics.
    """
    tier = billing_service.get_user_tier(current_user.id)
    usage = billing_service.get_monthly_usage(current_user.id)
    limit = billing_service.get_usage_limit(tier)

    can_analyze, reason = billing_service.can_run_analysis(current_user.id)

    return {
        "monthly_usage": usage,
        "monthly_limit": limit,
        "can_analyze": can_analyze,
        "reason": reason if not can_analyze else None,
        "tier": tier,
    }


@router.post("/portal", response_model=PortalResponse)
async def create_portal_session(
    current_user: UserResponse = Depends(get_current_user),
):
    """
    Create a Stripe Customer Portal session for subscription management.

    Allows users to update payment methods, view invoices, and cancel subscriptions.
    """
    if not stripe_client.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Billing is not configured"
        )

    subscription = billing_service.get_subscription_details(current_user.id)

    if not subscription.get("stripe_customer_id"):
        raise HTTPException(
            status_code=400,
            detail="No billing account found. Please subscribe first."
        )

    frontend_url = get_frontend_url()

    try:
        session = stripe_client.create_customer_portal_session(
            customer_id=subscription["stripe_customer_id"],
            return_url=f"{frontend_url}/mode",
        )

        return PortalResponse(portal_url=session.url)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create portal session: {str(e)}"
        )


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
):
    """
    Handle Stripe webhook events.

    This endpoint receives events from Stripe about subscription changes,
    payment status, etc.
    """
    if not stripe_client.is_configured():
        return JSONResponse(
            status_code=503,
            content={"error": "Billing not configured"}
        )

    if not stripe_signature:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing Stripe-Signature header"}
        )

    payload = await request.body()

    try:
        event = stripe_client.construct_webhook_event(payload, stripe_signature)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid signature: {str(e)}"}
        )

    # Handle the event
    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        # Payment successful, activate subscription
        await handle_checkout_completed(data)

    elif event_type == "customer.subscription.updated":
        # Subscription changed (upgrade, downgrade, renewal)
        await handle_subscription_updated(data)

    elif event_type == "customer.subscription.deleted":
        # Subscription cancelled
        await handle_subscription_deleted(data)

    elif event_type == "invoice.payment_failed":
        # Payment failed
        await handle_payment_failed(data)

    return {"status": "ok", "event_type": event_type}


async def handle_checkout_completed(session: dict):
    """Handle successful checkout session."""
    user_id = session.get("metadata", {}).get("user_id")
    customer_id = session.get("customer")
    subscription_id = session.get("subscription")

    if not user_id:
        print(f"Checkout completed but no user_id in metadata: {session.get('id')}")
        return

    # Get subscription details from Stripe
    if subscription_id:
        subscription = stripe_client.get_subscription(subscription_id)
        current_period_end = datetime.fromtimestamp(subscription.current_period_end)
    else:
        current_period_end = None

    billing_service.create_or_update_subscription(
        user_id=user_id,
        stripe_customer_id=customer_id,
        stripe_subscription_id=subscription_id,
        tier="pro",
        status="active",
        current_period_end=current_period_end,
    )

    print(f"User {user_id} upgraded to Pro")


async def handle_subscription_updated(subscription: dict):
    """Handle subscription update (renewal, plan change)."""
    customer_id = subscription.get("customer")
    subscription_id = subscription.get("id")
    status = subscription.get("status")

    # Find user by customer ID
    # Note: In production, you'd query your database for the user
    # For now, we'll rely on the metadata from the original checkout
    current_period_end = datetime.fromtimestamp(subscription.get("current_period_end", 0))

    # Update based on status
    if status == "active":
        print(f"Subscription {subscription_id} renewed/updated")
    elif status == "past_due":
        print(f"Subscription {subscription_id} is past due")
    elif status == "canceled":
        print(f"Subscription {subscription_id} was canceled")


async def handle_subscription_deleted(subscription: dict):
    """Handle subscription cancellation."""
    customer_id = subscription.get("customer")
    subscription_id = subscription.get("id")

    print(f"Subscription {subscription_id} deleted for customer {customer_id}")

    # In production, find user by customer_id and downgrade
    # billing_service.downgrade_to_free(user_id)


async def handle_payment_failed(invoice: dict):
    """Handle failed payment."""
    customer_id = invoice.get("customer")
    subscription_id = invoice.get("subscription")

    print(f"Payment failed for customer {customer_id}, subscription {subscription_id}")

    # In production, you might want to notify the user or take action
