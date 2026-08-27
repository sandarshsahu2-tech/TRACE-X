"""
TRACE-X
Historical Feature Engine

IMPORTANT:
This version does NOT open the persistent trace_x_data.duckdb file.

It queries the authoritative LI-Small_Trans.csv dataset through
an in-memory DuckDB connection.

Why:
- avoids Windows DuckDB file-lock conflicts
- allows multiple TRACE-X processes safely
- preserves past-only historical logic
- prevents future-data leakage
- no model retraining
"""

from __future__ import annotations

from pathlib import Path

import duckdb


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "DATA"
    / "LI-Small_Trans.csv"
)


# ============================================================
# HISTORICAL FEATURE ENGINE
# ============================================================

class HistoricalFeatureEngine:

    def __init__(self) -> None:

        if not DATA_PATH.exists():
            raise FileNotFoundError(
                "TRACE-X transaction dataset not found:\n"
                f"{DATA_PATH}"
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # In-memory DuckDB.
        #
        # We intentionally DO NOT open:
        #
        # models/TRACE_X_FINAL_MODEL/trace_x_data.duckdb
        #
        # This eliminates the Windows file-lock problem.
        # ----------------------------------------------------

        self.con = duckdb.connect(
            database=":memory:"
        )

        self.source = (
            str(DATA_PATH.resolve())
            .replace("\\", "/")
        )

    # ========================================================
    # HISTORY
    # ========================================================

    def get_history(
        self,
        transaction: dict,
    ) -> dict:

        timestamp = transaction[
            "Timestamp"
        ]

        from_bank = str(
            transaction["From_Bank"]
        )

        sender = str(
            transaction["Sender_Account"]
        )

        to_bank = str(
            transaction["To_Bank"]
        )

        receiver = str(
            transaction["Receiver_Account"]
        )

        # ----------------------------------------------------
        # Normalize bank identifiers.
        #
        # Dataset:
        # From Bank -> 3 digits
        # To Bank   -> 6 digits
        #
        # This preserves values such as:
        # 070
        # 022661
        # ----------------------------------------------------

        from_bank = from_bank.zfill(3)
        to_bank = to_bank.zfill(6)

        sender_key = (
            f"{from_bank}|{sender}"
        )

        receiver_key = (
            f"{to_bank}|{receiver}"
        )

        pair_key = (
            f"{from_bank}|{sender}"
            f"->{to_bank}|{receiver}"
        )

        # ====================================================
        # AUTHORITATIVE DATA QUERY
        # ====================================================

        query = f"""
        WITH base AS (

            SELECT

                *,

                CAST(
                    "Timestamp"
                    AS TIMESTAMP
                ) AS ts,

                LPAD(
                    CAST(
                        "From Bank"
                        AS VARCHAR
                    ),
                    3,
                    '0'
                )
                || '|'
                || CAST(
                    "Account"
                    AS VARCHAR
                )
                AS sender_key,

                LPAD(
                    CAST(
                        "To Bank"
                        AS VARCHAR
                    ),
                    6,
                    '0'
                )
                || '|'
                || CAST(
                    "Account_1"
                    AS VARCHAR
                )
                AS receiver_key,

                LPAD(
                    CAST(
                        "From Bank"
                        AS VARCHAR
                    ),
                    3,
                    '0'
                )
                || '|'
                || CAST(
                    "Account"
                    AS VARCHAR
                )
                || '->'
                || LPAD(
                    CAST(
                        "To Bank"
                        AS VARCHAR
                    ),
                    6,
                    '0'
                )
                || '|'
                || CAST(
                    "Account_1"
                    AS VARCHAR
                )
                AS pair_key

            FROM read_csv_auto(
                '{self.source}',
                header = true
            )
        )

        SELECT

            -- ==============================================
            -- SENDER HISTORY
            -- ==============================================

            COUNT(*) FILTER (
                WHERE
                    sender_key = ?
                    AND ts < CAST(
                        ? AS TIMESTAMP
                    )
            )
            AS Sender_Prior_Tx_Count,

            COALESCE(
                SUM(
                    "Amount Received"
                ) FILTER (
                    WHERE
                        sender_key = ?
                        AND ts < CAST(
                            ? AS TIMESTAMP
                        )
                ),
                0
            )
            AS Sender_Prior_Total_Received,

            COALESCE(
                AVG(
                    "Amount Received"
                ) FILTER (
                    WHERE
                        sender_key = ?
                        AND ts < CAST(
                            ? AS TIMESTAMP
                        )
                ),
                0
            )
            AS Sender_Prior_Avg_Received,


            -- ==============================================
            -- RECEIVER HISTORY
            -- ==============================================

            COUNT(*) FILTER (
                WHERE
                    receiver_key = ?
                    AND ts < CAST(
                        ? AS TIMESTAMP
                    )
            )
            AS Receiver_Prior_Tx_Count,

            COALESCE(
                SUM(
                    "Amount Received"
                ) FILTER (
                    WHERE
                        receiver_key = ?
                        AND ts < CAST(
                            ? AS TIMESTAMP
                        )
                ),
                0
            )
            AS Receiver_Prior_Total_Received,

            COALESCE(
                AVG(
                    "Amount Received"
                ) FILTER (
                    WHERE
                        receiver_key = ?
                        AND ts < CAST(
                            ? AS TIMESTAMP
                        )
                ),
                0
            )
            AS Receiver_Prior_Avg_Received,


            -- ==============================================
            -- SENDER / RECEIVER PAIR
            -- ==============================================

            COUNT(*) FILTER (
                WHERE
                    pair_key = ?
                    AND ts < CAST(
                        ? AS TIMESTAMP
                    )
            )
            AS Pair_Prior_Tx_Count,

            COALESCE(
                SUM(
                    "Amount Paid"
                ) FILTER (
                    WHERE
                        pair_key = ?
                        AND ts < CAST(
                            ? AS TIMESTAMP
                        )
                ),
                0
            )
            AS Pair_Prior_Total_Paid,

            COALESCE(
                AVG(
                    "Amount Paid"
                ) FILTER (
                    WHERE
                        pair_key = ?
                        AND ts < CAST(
                            ? AS TIMESTAMP
                        )
                ),
                0
            )
            AS Pair_Prior_Avg_Paid

        FROM base
        """

        parameters = [

            # Sender
            sender_key,
            timestamp,

            sender_key,
            timestamp,

            sender_key,
            timestamp,

            # Receiver
            receiver_key,
            timestamp,

            receiver_key,
            timestamp,

            receiver_key,
            timestamp,

            # Pair
            pair_key,
            timestamp,

            pair_key,
            timestamp,

            pair_key,
            timestamp,
        ]

        result = self.con.execute(
            query,
            parameters,
        ).fetchone()

        if result is None:
            raise RuntimeError(
                "TRACE-X historical query returned no result."
            )

        (
            sender_count,
            sender_total,
            sender_avg,

            receiver_count,
            receiver_total,
            receiver_avg,

            pair_count,
            pair_total,
            pair_avg,

        ) = result

        # ====================================================
        # NORMALIZE RESULTS
        # ====================================================

        sender_count = int(
            sender_count or 0
        )

        receiver_count = int(
            receiver_count or 0
        )

        pair_count = int(
            pair_count or 0
        )

        sender_total = float(
            sender_total or 0
        )

        sender_avg = float(
            sender_avg or 0
        )

        receiver_total = float(
            receiver_total or 0
        )

        receiver_avg = float(
            receiver_avg or 0
        )

        pair_total = float(
            pair_total or 0
        )

        pair_avg = float(
            pair_avg or 0
        )

        # ====================================================
        # FEATURE RESULT
        # ====================================================

        return {

            "Sender_Prior_Tx_Count":
                sender_count,

            "Sender_Prior_Total_Received":
                sender_total,

            "Sender_Prior_Avg_Received":
                sender_avg,

            "Receiver_Prior_Tx_Count":
                receiver_count,

            "Receiver_Prior_Total_Received":
                receiver_total,

            "Receiver_Prior_Avg_Received":
                receiver_avg,

            "Pair_Prior_Tx_Count":
                pair_count,

            "Pair_Prior_Total_Paid":
                pair_total,

            "Pair_Prior_Avg_Paid":
                pair_avg,

            "Is_First_Sender_Tx":
                int(
                    sender_count == 0
                ),

            "Is_First_Receiver_Tx":
                int(
                    receiver_count == 0
                ),

            "Is_New_Sender_Receiver_Pair":
                int(
                    pair_count == 0
                ),
        }

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:

        if self.con is not None:

            self.con.close()

            self.con = None