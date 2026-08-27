from typing import List

from pydantic import BaseModel, Field


# ============================================================
# REAL TRANSACTION REQUEST
# ============================================================

class TransactionRequest(BaseModel):
    """
    Raw transaction submitted to TRACE-X.

    Bank identifiers are strings because leading zeros
    are meaningful in the original dataset.
    """

    timestamp: str = Field(
        ...,
        description="Transaction timestamp",
    )

    from_bank: str = Field(
        ...,
        description="Sending bank identifier",
    )

    sender_account: str = Field(
        ...,
        description="Sending account identifier",
    )

    to_bank: str = Field(
        ...,
        description="Receiving bank identifier",
    )

    receiver_account: str = Field(
        ...,
        description="Receiving account identifier",
    )

    amount_received: float = Field(
        ...,
        ge=0,
        description="Amount received",
    )

    receiving_currency: str = Field(
        ...,
        description="Receiving currency",
    )

    amount_paid: float = Field(
        ...,
        ge=0,
        description="Amount paid",
    )

    payment_currency: str = Field(
        ...,
        description="Payment currency",
    )

    payment_format: str = Field(
        ...,
        description="Payment format",
    )


# ============================================================
# DIRECT MODEL REQUEST
# ============================================================

class PredictionRequest(BaseModel):

    features: List[float] = Field(
        ...,
        min_length=38,
        max_length=38,
        description="Exactly 38 TRACE-X model features",
    )


# ============================================================
# MODEL RESPONSE
# ============================================================

class PredictionResponse(BaseModel):

    model: str
    risk_score: float
    threshold: float
    decision: str


# ============================================================
# HEALTH RESPONSE
# ============================================================

class HealthResponse(BaseModel):

    status: str
    model_loaded: bool