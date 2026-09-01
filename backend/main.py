"""
TRACE-X Investigation API

TRACE-X V1 remains the authoritative ML decision engine.

The GenAI layer is an explanation and investigation
assistant only. It does NOT modify the TRACE-X decision.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import duckdb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    TransactionRequest,
)
from backend.inference import trace_x_model


# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    Path(__file__).resolve().parent / ".env"
)


# ============================================================
# PATHS
# ============================================================

ANALYTICS_DB = Path(
    os.getenv(
        "TRACE_X_ANALYTICS_DB",
        str(
            PROJECT_ROOT
            / "models"
            / "TRACE_X_FINAL_MODEL"
            / "trace_x_data.duckdb"
        ),
    )
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="TRACE-X Investigation API",
    description=(
        "TRACE-X financial intelligence backend for "
        "transaction risk scoring, historical analysis, "
        "dashboard analytics, investigation support and "
        "grounded GenAI assistance."
    ),
    version="3.1.0",
)


# ============================================================
# CORS
# ============================================================

_default_origins = [
    "http://localhost:5179",
    "http://127.0.0.1:5179",
    "http://localhost:5178",
    "http://127.0.0.1:5178",
    "http://localhost:5177",
    "http://127.0.0.1:5177",
    "http://localhost:5176",
    "http://127.0.0.1:5176",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

_env_origins = [
    origin.strip()
    for origin in os.getenv(
        "TRACE_X_ALLOWED_ORIGINS",
        "",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_env_origins + _default_origins,
    allow_origin_regex=os.getenv(
        "TRACE_X_ALLOWED_ORIGIN_REGEX",
        r"https://.*\.trycloudflare\.com",
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

def get_analytics_connection():
    """
    Open the TRACE-X analytics database in read-only mode.

    Each request gets a short-lived DuckDB connection.
    """

    if not ANALYTICS_DB.exists():
        raise FileNotFoundError(
            "TRACE-X analytics database not found:\n"
            f"{ANALYTICS_DB}"
        )

    return duckdb.connect(
        str(ANALYTICS_DB),
        read_only=True,
    )


# ============================================================
# HELPERS
# ============================================================

def transaction_to_dict(
    request: TransactionRequest,
) -> dict[str, Any]:
    return {
        "Timestamp": request.timestamp,
        "From_Bank": str(request.from_bank),
        "Sender_Account": str(request.sender_account),
        "To_Bank": str(request.to_bank),
        "Receiver_Account": str(request.receiver_account),
        "Amount_Received": float(request.amount_received),
        "Receiving_Currency": str(request.receiving_currency),
        "Amount_Paid": float(request.amount_paid),
        "Payment_Currency": str(request.payment_currency),
        "Payment_Format": str(request.payment_format),
    }


def http500(message: str, exc: Exception):
    raise HTTPException(
        status_code=500,
        detail=f"{message}: {exc}",
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "TRACE-X Investigation API",
        "status": "operational",
        "model_ready": True,
        "analytics_ready": ANALYTICS_DB.exists(),
        "genai_enabled": bool(
            os.getenv("GEMINI_API_KEY")
        ),
        "version": "3.1.0",
        "port": int(os.getenv("PORT", "8001")),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=True,
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": True,
        "analytics_ready": ANALYTICS_DB.exists(),
    }


# ============================================================
# MODEL STATUS
# ============================================================

@app.get("/api/v1/model")
def model_status() -> dict[str, Any]:
    try:
        return trace_x_model.status()
    except Exception as exc:
        http500("Unable to read model status", exc)


# ============================================================
# SYSTEM STATUS
# ============================================================

@app.get("/api/v1/system")
def system_status() -> dict[str, Any]:
    try:
        model_data = trace_x_model.status()
        analytics_ready = ANALYTICS_DB.exists()
        genai_configured = bool(
            os.getenv("GEMINI_API_KEY")
        )

        historical_ready = (
            model_data.get("historical_engine") == "ready"
        )

        feature_ready = (
            model_data.get("feature_service") == "ready"
        )

        return {
            "status": (
                "OPERATIONAL"
                if analytics_ready
                else "DEGRADED"
            ),
            "model": model_data,
            "engines": {
                "ml": "READY",
                "historical": (
                    "READY"
                    if historical_ready
                    else "ERROR"
                ),
                "feature_service": (
                    "READY"
                    if feature_ready
                    else "ERROR"
                ),
                "analytics": (
                    "READY"
                    if analytics_ready
                    else "NOT_READY"
                ),
                "graph": "READY",
                "rules": "READY",
                "investigation": "READY",
                "genai": (
                    "READY"
                    if genai_configured
                    else "NOT_CONFIGURED"
                ),
            },
        }

    except Exception as exc:
        return {
            "status": "DEGRADED",
            "model": "ERROR",
            "engines": {
                "ml": "ERROR",
                "historical": "ERROR",
                "feature_service": "ERROR",
                "analytics": "ERROR",
                "graph": "READY",
                "rules": "READY",
                "investigation": "READY",
                "genai": "ERROR",
            },
            "error": str(exc),
        }


# ============================================================
# DIRECT VECTOR PREDICTION
# ============================================================

@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    try:
        result = trace_x_model.predict_vector(
            request.features
        )

        return PredictionResponse(
            model=result["model"],
            risk_score=float(result["risk_score"]),
            threshold=float(result["threshold"]),
            decision=result["decision"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except Exception as exc:
        http500("Prediction failed", exc)


# ============================================================
# REAL TRANSACTION PREDICTION
# ============================================================

@app.post("/api/v1/predict/transaction")
def predict_transaction(
    request: TransactionRequest,
) -> dict[str, Any]:
    try:
        transaction = transaction_to_dict(request)

        return trace_x_model.predict_transaction(
            transaction
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except Exception as exc:
        print(
            "TRACE-X TRANSACTION ERROR:",
            repr(exc),
        )
        http500(
            "Transaction prediction failed",
            exc,
        )


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

@app.get("/api/v1/dashboard/summary")
def dashboard_summary() -> dict[str, Any]:
    con = None

    try:
        con = get_analytics_connection()

        row = con.execute(
            """
            SELECT
                COUNT(*) AS total_transactions,

                COALESCE(
                    SUM(
                        CASE
                            WHEN "Is Laundering" = 1
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS flagged_transactions,

                COALESCE(
                    SUM(
                        CASE
                            WHEN "Is Laundering" = 0
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS normal_transactions,

                COALESCE(
                    SUM("Amount Received"),
                    0
                ) AS total_received,

                COALESCE(
                    SUM("Amount Paid"),
                    0
                ) AS total_paid,

                COALESCE(
                    AVG("Amount Received"),
                    0
                ) AS average_transaction

            FROM transactions
            """
        ).fetchone()

        (
            total,
            flagged,
            normal,
            received,
            paid,
            average,
        ) = row

        total = int(total or 0)
        flagged = int(flagged or 0)
        normal = int(normal or 0)

        return {
            "total_transactions": total,
            "flagged_transactions": flagged,
            "normal_transactions": normal,
            "laundering_rate": (
                flagged / total
                if total > 0
                else 0.0
            ),
            "total_received": float(
                received or 0
            ),
            "total_paid": float(
                paid or 0
            ),
            "average_transaction": float(
                average or 0
            ),
        }

    except Exception as exc:
        http500(
            "Dashboard summary failed",
            exc,
        )

    finally:
        if con is not None:
            con.close()


# ============================================================
# TRANSACTION TRENDS
# ============================================================

@app.get("/api/v1/dashboard/trends")
def dashboard_trends() -> dict[str, Any]:
    con = None

    try:
        con = get_analytics_connection()

        rows = con.execute(
            """
            SELECT
                DATE_TRUNC(
                    'hour',
                    "Timestamp"
                ) AS period,

                COUNT(*) AS transactions,

                SUM(
                    CASE
                        WHEN "Is Laundering" = 1
                        THEN 1
                        ELSE 0
                    END
                ) AS flagged

            FROM transactions

            GROUP BY 1
            ORDER BY 1

            LIMIT 500
            """
        ).fetchall()

        return {
            "data": [
                {
                    "period": str(row[0]),
                    "transactions": int(
                        row[1] or 0
                    ),
                    "flagged": int(
                        row[2] or 0
                    ),
                }
                for row in rows
            ]
        }

    except Exception as exc:
        http500(
            "Dashboard trends failed",
            exc,
        )

    finally:
        if con is not None:
            con.close()


# ============================================================
# DECISION DISTRIBUTION
# ============================================================

@app.get("/api/v1/dashboard/distribution")
def dashboard_distribution() -> dict[str, Any]:
    con = None

    try:
        con = get_analytics_connection()

        rows = con.execute(
            """
            SELECT
                CASE
                    WHEN "Is Laundering" = 1
                    THEN 'FLAG'
                    ELSE 'NORMAL'
                END AS decision,

                COUNT(*) AS count

            FROM transactions

            GROUP BY 1
            ORDER BY 1
            """
        ).fetchall()

        return {
            "data": [
                {
                    "decision": str(row[0]),
                    "count": int(row[1] or 0),
                }
                for row in rows
            ]
        }

    except Exception as exc:
        http500(
            "Decision distribution failed",
            exc,
        )

    finally:
        if con is not None:
            con.close()


# ============================================================
# TOP SUSPICIOUS BANKS
# ============================================================

@app.get("/api/v1/dashboard/top-banks")
def dashboard_top_banks() -> dict[str, Any]:
    con = None

    try:
        con = get_analytics_connection()

        rows = con.execute(
            """
            SELECT
                "From Bank" AS bank,

                COUNT(*) AS transactions,

                SUM(
                    CASE
                        WHEN "Is Laundering" = 1
                        THEN 1
                        ELSE 0
                    END
                ) AS flagged,

                COALESCE(
                    SUM("Amount Received"),
                    0
                ) AS amount

            FROM transactions

            GROUP BY 1

            ORDER BY flagged DESC

            LIMIT 10
            """
        ).fetchall()

        return {
            "data": [
                {
                    "bank": str(row[0]),
                    "transactions": int(
                        row[1] or 0
                    ),
                    "flagged": int(
                        row[2] or 0
                    ),
                    "amount": float(
                        row[3] or 0
                    ),
                }
                for row in rows
            ]
        }

    except Exception as exc:
        http500(
            "Top banks query failed",
            exc,
        )

    finally:
        if con is not None:
            con.close()


# ============================================================
# INVESTIGATION QUEUE
# ============================================================

@app.get("/api/v1/dashboard/queue")
def dashboard_queue(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
) -> dict[str, Any]:

    con = None

    try:
        con = get_analytics_connection()

        rows = con.execute(
            """
            SELECT
                "Timestamp",
                "From Bank",
                "Account",
                "To Bank",
                "Account_1",
                "Amount Received",
                "Receiving Currency",
                "Amount Paid",
                "Payment Currency",
                "Payment Format",
                "Is Laundering"

            FROM transactions

            WHERE "Is Laundering" = 1

            ORDER BY "Timestamp" DESC

            LIMIT ?
            """,
            [limit],
        ).fetchall()

        data = []

        for row in rows:
            data.append(
                {
                    "timestamp": str(row[0]),
                    "from_bank": str(row[1]),
                    "sender_account": str(row[2]),
                    "to_bank": str(row[3]),
                    "receiver_account": str(row[4]),
                    "amount_received": float(
                        row[5] or 0
                    ),
                    "receiving_currency": str(row[6]),
                    "amount_paid": float(
                        row[7] or 0
                    ),
                    "payment_currency": str(row[8]),
                    "payment_format": str(row[9]),
                    "is_laundering": int(
                        row[10] or 0
                    ),
                }
            )

        return {
            "count": len(data),
            "data": data,
        }

    except Exception as exc:
        http500(
            "Investigation queue failed",
            exc,
        )

    finally:
        if con is not None:
            con.close()


# ============================================================
# NETWORK INTELLIGENCE
# ============================================================

@app.get("/api/v1/dashboard/network")
def dashboard_network(
    bank: str | None = None,
    limit: int = Query(
        default=40,
        ge=1,
        le=200,
    ),
) -> dict[str, Any]:

    con = None

    try:
        con = get_analytics_connection()

        if bank:
            rows = con.execute(
                """
                SELECT
                    "From Bank",
                    "Account",
                    "To Bank",
                    "Account_1",

                    COUNT(*) AS tx_count,

                    COALESCE(
                        SUM("Amount Received"),
                        0
                    ) AS total_amount

                FROM transactions

                WHERE
                    "From Bank" = ?
                    OR "To Bank" = ?

                GROUP BY
                    1, 2, 3, 4

                ORDER BY
                    tx_count DESC

                LIMIT ?
                """,
                [
                    bank,
                    bank,
                    limit,
                ],
            ).fetchall()

        else:
            rows = con.execute(
                """
                SELECT
                    "From Bank",
                    "Account",
                    "To Bank",
                    "Account_1",

                    COUNT(*) AS tx_count,

                    COALESCE(
                        SUM("Amount Received"),
                        0
                    ) AS total_amount

                FROM transactions

                GROUP BY
                    1, 2, 3, 4

                ORDER BY
                    tx_count DESC

                LIMIT ?
                """,
                [limit],
            ).fetchall()

        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        for row in rows:

            from_bank = str(row[0])
            sender = str(row[1])
            to_bank = str(row[2])
            receiver = str(row[3])

            sender_id = f"bank:{from_bank}"
            receiver_id = f"bank:{to_bank}"

            nodes[sender_id] = {
                "id": sender_id,
                "label": f"Bank {from_bank}",
                "type": "bank",
            }

            nodes[receiver_id] = {
                "id": receiver_id,
                "label": f"Bank {to_bank}",
                "type": "bank",
            }

            edges.append(
                {
                    "id": (
                        f"{sender_id}"
                        f"->{receiver_id}"
                        f":{sender}"
                        f":{receiver}"
                    ),
                    "source": sender_id,
                    "target": receiver_id,
                    "sender_account": sender,
                    "receiver_account": receiver,
                    "transactions": int(
                        row[4] or 0
                    ),
                    "amount": float(
                        row[5] or 0
                    ),
                }
            )

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
        }

    except Exception as exc:
        http500(
            "Network intelligence failed",
            exc,
        )

    finally:
        if con is not None:
            con.close()


# ============================================================
# INVESTIGATION WORKFLOW
# ============================================================

@app.post("/api/v1/investigate")
def investigate(
    request: TransactionRequest,
) -> dict[str, Any]:

    try:
        transaction = transaction_to_dict(request)

        historical = (
            trace_x_model
            .historical_engine
            .get_history(transaction)
        )

        prediction = (
            trace_x_model
            .predict_transaction(transaction)
        )

        risk_score = float(
            prediction.get(
                "risk_score",
                0.0,
            )
        )

        threshold = float(
            prediction.get(
                "threshold",
                0.76,
            )
        )

        return {
            "transaction": transaction,
            "model": prediction,
            "historical_features": historical,
            "investigation": {
                "risk_level": (
                    "HIGH"
                    if risk_score >= threshold
                    else "NORMAL"
                ),
                "model_triggered": (
                    risk_score >= threshold
                ),
                "next_step": (
                    "REVIEW"
                    if risk_score >= threshold
                    else "NO_IMMEDIATE_ACTION"
                ),
            },
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except Exception as exc:
        print(
            "TRACE-X INVESTIGATION ERROR:",
            repr(exc),
        )
        http500(
            "Investigation failed",
            exc,
        )


# ============================================================
# GEMINI AI INVESTIGATION
# ============================================================

@app.post("/api/v1/ai/investigate")
def ai_investigate(
    request: TransactionRequest,
) -> dict[str, Any]:

    try:

        # Lazy import so normal TRACE-X inference
        # still works when Gemini is unavailable.
        from backend.ai_service import (
            get_ai_investigation_service,
        )

        transaction = transaction_to_dict(request)

        historical = (
            trace_x_model
            .historical_engine
            .get_history(transaction)
        )

        prediction = (
            trace_x_model
            .predict_transaction(transaction)
        )

        evidence_pack = {
            "transaction": transaction,

            "model": {
                "name": prediction.get(
                    "model",
                    "TRACE-X V1",
                ),
                "risk_score": float(
                    prediction.get(
                        "risk_score",
                        0.0,
                    )
                ),
                "threshold": float(
                    prediction.get(
                        "threshold",
                        0.76,
                    )
                ),
                "decision": prediction.get(
                    "decision",
                    "UNKNOWN",
                ),
                "feature_count": 38,
            },

            "historical": historical,

            # Reserved for grounded network evidence.
            "network": {},

            # Reserved for grounded rules.
            "rules": {},
        }

        ai_service = (
            get_ai_investigation_service()
        )

        ai_result = ai_service.investigate(
            evidence_pack
        )

        return {
            "status": "success",
            "trace_x": evidence_pack["model"],
            "historical_features": historical,
            "ai": ai_result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except Exception as exc:
        print(
            "TRACE-X GENAI ERROR:",
            repr(exc),
        )
        http500(
            "AI investigation failed",
            exc,
        )


# ============================================================
# DEVELOPMENT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=os.getenv(
            "HOST",
            "127.0.0.1",
        ),
        port=int(
            os.getenv(
                "PORT",
                "8001",
            )
        ),
        reload=False,
    )