"""
TRACE-X
Production Feature Construction Service

Purpose
-------
Build the exact 38 numerical features expected by TRACE-X V1.

Design principles
-----------------
1. Match the training definitions exactly.
2. Preserve the exact model feature order.
3. Use TRAIN-only frequency lookup.
4. Use past-only historical features supplied by the history engine.
5. Never retrain or alter the frozen model.
6. Fail loudly on malformed transactions or missing model features.

TRACE-X V1:
- 38 features
- 800 boosting rounds
- threshold handled by inference layer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "TRACE_X_FINAL_MODEL"
)

FREQUENCY_PATH = (
    MODEL_DIR
    / "frequency_lookup.parquet"
)

FEATURE_PATH = (
    MODEL_DIR
    / "TRACE_X_MODEL_FEATURES.json"
)


# ============================================================
# FEATURE SERVICE
# ============================================================

class FeatureService:

    # --------------------------------------------------------
    # CATEGORICAL COLUMNS
    # --------------------------------------------------------

    CATEGORICAL_COLUMNS = [
        "From_Bank",
        "To_Bank",
        "Sender_Account",
        "Receiver_Account",
        "Receiving_Currency",
        "Payment_Currency",
        "Payment_Format",
    ]

    # --------------------------------------------------------
    # EXACT TRAINING FEATURE CONTRACT
    # --------------------------------------------------------

    FEATURE_COLUMNS = [
        "Amount_Received",
        "Amount_Paid",

        "Log_Amount_Received",
        "Log_Amount_Paid",

        "Absolute_Amount_Difference",
        "Relative_Amount_Difference",
        "Received_Paid_Ratio",

        "Sender_Prior_Tx_Count",
        "Sender_Prior_Total_Received",
        "Sender_Prior_Avg_Received",

        "Receiver_Prior_Tx_Count",
        "Receiver_Prior_Total_Received",
        "Receiver_Prior_Avg_Received",

        "Pair_Prior_Tx_Count",
        "Pair_Prior_Total_Paid",
        "Pair_Prior_Avg_Paid",

        "Is_First_Sender_Tx",
        "Is_First_Receiver_Tx",
        "Is_New_Sender_Receiver_Pair",

        "Transaction_Hour",
        "Transaction_DayOfWeek",
        "Transaction_Day",
        "Transaction_Month",

        "Is_Weekend",
        "Is_Night",
        "Is_Evening",
        "Is_Cross_Bank_Cross_Currency",

        "Hour_Sin",
        "Hour_Cos",
        "DayOfWeek_Sin",
        "DayOfWeek_Cos",

        "From_Bank_Freq",
        "To_Bank_Freq",
        "Sender_Account_Freq",
        "Receiver_Account_Freq",
        "Receiving_Currency_Freq",
        "Payment_Currency_Freq",
        "Payment_Format_Freq",
    ]

    # --------------------------------------------------------
    # REQUIRED RAW TRANSACTION COLUMNS
    # --------------------------------------------------------

    REQUIRED_TRANSACTION_COLUMNS = [
        "Timestamp",
        "From_Bank",
        "To_Bank",
        "Sender_Account",
        "Receiver_Account",
        "Amount_Received",
        "Receiving_Currency",
        "Amount_Paid",
        "Payment_Currency",
        "Payment_Format",
    ]

    # --------------------------------------------------------
    # EXPECTED HISTORICAL COLUMNS
    # --------------------------------------------------------

    HISTORICAL_COLUMNS = [
        "Sender_Prior_Tx_Count",
        "Sender_Prior_Total_Received",
        "Sender_Prior_Avg_Received",

        "Receiver_Prior_Tx_Count",
        "Receiver_Prior_Total_Received",
        "Receiver_Prior_Avg_Received",

        "Pair_Prior_Tx_Count",
        "Pair_Prior_Total_Paid",
        "Pair_Prior_Avg_Paid",

        "Is_First_Sender_Tx",
        "Is_First_Receiver_Tx",
        "Is_New_Sender_Receiver_Pair",
    ]

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self) -> None:

        self.frequency_maps: dict[
            str,
            dict[str, float]
        ] = {}

        self._load_frequency_lookup()
        self._validate_feature_contract()

    # ========================================================
    # LOAD FREQUENCY LOOKUP
    # ========================================================

    def _load_frequency_lookup(self) -> None:

        if not FREQUENCY_PATH.exists():
            raise FileNotFoundError(
                "TRACE-X frequency lookup not found:\n"
                f"{FREQUENCY_PATH}"
            )

        lookup = pd.read_parquet(
            FREQUENCY_PATH
        )

        required_columns = {
            "value",
            "feature",
            "frequency",
        }

        missing = (
            required_columns
            - set(lookup.columns)
        )

        if missing:
            raise ValueError(
                "Frequency lookup is missing columns: "
                + ", ".join(sorted(missing))
            )

        for feature in self.CATEGORICAL_COLUMNS:

            rows = lookup.loc[
                lookup["feature"] == feature,
                ["value", "frequency"],
            ]

            mapping: dict[str, float] = {}

            for value, frequency in rows.itertuples(
                index=False,
                name=None,
            ):

                # Unknown/invalid lookup keys are skipped.
                if pd.isna(value):
                    continue

                try:
                    numeric_frequency = float(
                        frequency
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if not np.isfinite(
                    numeric_frequency
                ):
                    continue

                mapping[
                    str(value)
                ] = numeric_frequency

            self.frequency_maps[
                feature
            ] = mapping

        # Final sanity check
        missing_maps = [
            column
            for column in self.CATEGORICAL_COLUMNS
            if column not in self.frequency_maps
        ]

        if missing_maps:
            raise RuntimeError(
                "Frequency maps failed to initialize for: "
                + ", ".join(missing_maps)
            )

    # ========================================================
    # FEATURE CONTRACT VALIDATION
    # ========================================================

    def _validate_feature_contract(
        self
    ) -> None:

        if len(
            self.FEATURE_COLUMNS
        ) != 38:
            raise RuntimeError(
                "TRACE-X V1 requires exactly 38 features, "
                f"found {len(self.FEATURE_COLUMNS)}."
            )

        # Verify packaged feature contract when available.
        if FEATURE_PATH.exists():

            import json

            with open(
                FEATURE_PATH,
                "r",
                encoding="utf-8",
            ) as file:

                packaged_features = json.load(
                    file
                )

            if packaged_features != self.FEATURE_COLUMNS:

                raise RuntimeError(
                    "Feature contract mismatch.\n\n"
                    f"Packaged:\n{packaged_features}\n\n"
                    f"Service:\n{self.FEATURE_COLUMNS}"
                )

    # ========================================================
    # BASIC TRANSACTION VALIDATION
    # ========================================================

    def _validate_transaction(
        self,
        transaction: Mapping[str, Any],
    ) -> None:

        missing = [
            column
            for column in self.REQUIRED_TRANSACTION_COLUMNS
            if column not in transaction
        ]

        if missing:
            raise ValueError(
                "Transaction is missing required fields: "
                + ", ".join(missing)
            )

        for column in (
            "Amount_Received",
            "Amount_Paid",
        ):

            try:
                value = float(
                    transaction[column]
                )
            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    f"{column} must be numeric."
                ) from exc

            if not np.isfinite(value):

                raise ValueError(
                    f"{column} must be finite."
                )

    # ========================================================
    # FREQUENCY FEATURES
    # ========================================================

    def _add_frequency_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        dataframe = dataframe.copy()

        for column in self.CATEGORICAL_COLUMNS:

            mapping = self.frequency_maps[
                column
            ]

            dataframe[
                column + "_Freq"
            ] = (
                dataframe[column]
                .astype(str)
                .map(mapping)
                .fillna(0.0)
                .astype(np.float32)
            )

        return dataframe

    # ========================================================
    # HISTORICAL FEATURES
    # ========================================================

    def _add_historical_features(
        self,
        dataframe: pd.DataFrame,
        history: Mapping[str, Any],
    ) -> pd.DataFrame:

        dataframe = dataframe.copy()

        for column in self.HISTORICAL_COLUMNS:

            value = history.get(
                column,
                0,
            )

            if value is None:
                value = 0

            dataframe[
                column
            ] = value

        return dataframe

    # ========================================================
    # BUILD EXACT FEATURE VECTOR
    # ========================================================

    def build(
        self,
        transaction: Mapping[str, Any],
        history: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:

        if history is None:
            history = {}

        self._validate_transaction(
            transaction
        )

        df = pd.DataFrame(
            [dict(transaction)]
        )

        # ----------------------------------------------------
        # Amount features
        #
        # These match the training pipeline:
        #
        # log(1 + abs(amount))
        # abs(received - paid)
        # difference / max(abs(received), abs(paid), 1e-9)
        # received / max(abs(paid), 1e-9)
        # ----------------------------------------------------

        received = float(
            df["Amount_Received"].iloc[0]
        )

        paid = float(
            df["Amount_Paid"].iloc[0]
        )

        df[
            "Log_Amount_Received"
        ] = np.log1p(
            max(abs(received), 0.0)
        )

        df[
            "Log_Amount_Paid"
        ] = np.log1p(
            max(abs(paid), 0.0)
        )

        difference = abs(
            received - paid
        )

        df[
            "Absolute_Amount_Difference"
        ] = difference

        df[
            "Relative_Amount_Difference"
        ] = (
            difference
            / max(
                abs(received),
                abs(paid),
                1e-9,
            )
        )

        df[
            "Received_Paid_Ratio"
        ] = (
            received
            / max(
                abs(paid),
                1e-9,
            )
        )

        # ----------------------------------------------------
        # Historical features
        # ----------------------------------------------------

        df = self._add_historical_features(
            df,
            history,
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp = pd.to_datetime(
            df["Timestamp"].iloc[0],
            errors="coerce",
        )

        if pd.isna(timestamp):
            raise ValueError(
                "Invalid transaction timestamp."
            )

        hour = int(
            timestamp.hour
        )

        # IMPORTANT:
        #
        # Training uses DuckDB EXTRACT(DOW):
        # Sunday = 0
        # Monday = 1
        # ...
        # Saturday = 6
        #
        # pandas weekday():
        # Monday = 0
        # ...
        # Sunday = 6
        #
        # Convert pandas -> DuckDB convention.
        weekday = int(
            (timestamp.weekday() + 1) % 7
        )

        day = int(
            timestamp.day
        )

        month = int(
            timestamp.month
        )

        df[
            "Transaction_Hour"
        ] = hour

        df[
            "Transaction_DayOfWeek"
        ] = weekday

        df[
            "Transaction_Day"
        ] = day

        df[
            "Transaction_Month"
        ] = month

        # ----------------------------------------------------
        # Temporal flags
        #
        # EXACTLY aligned with training:
        #
        # Weekend = Sunday or Saturday
        # Night   = hour < 6
        # Evening = 18 <= hour <= 23
        # ----------------------------------------------------

        df[
            "Is_Weekend"
        ] = int(
            weekday in (0, 6)
        )

        df[
            "Is_Night"
        ] = int(
            hour < 6
        )

        df[
            "Is_Evening"
        ] = int(
            18 <= hour <= 23
        )

        # ----------------------------------------------------
        # Cross-bank / cross-currency
        # ----------------------------------------------------

        from_bank = str(
            df["From_Bank"].iloc[0]
        )

        to_bank = str(
            df["To_Bank"].iloc[0]
        )

        receiving_currency = str(
            df[
                "Receiving_Currency"
            ].iloc[0]
        )

        payment_currency = str(
            df[
                "Payment_Currency"
            ].iloc[0]
        )

        df[
            "Is_Cross_Bank_Cross_Currency"
        ] = int(
            (
                from_bank
                != to_bank
            )
            and
            (
                receiving_currency
                != payment_currency
            )
        )

        # ----------------------------------------------------
        # Cyclical time features
        # ----------------------------------------------------

        df[
            "Hour_Sin"
        ] = np.sin(
            2.0
            * np.pi
            * hour
            / 24.0
        )

        df[
            "Hour_Cos"
        ] = np.cos(
            2.0
            * np.pi
            * hour
            / 24.0
        )

        df[
            "DayOfWeek_Sin"
        ] = np.sin(
            2.0
            * np.pi
            * weekday
            / 7.0
        )

        df[
            "DayOfWeek_Cos"
        ] = np.cos(
            2.0
            * np.pi
            * weekday
            / 7.0
        )

        # ----------------------------------------------------
        # Frequency features
        # ----------------------------------------------------

        df = self._add_frequency_features(
            df
        )

        # ----------------------------------------------------
        # Final feature existence check
        # ----------------------------------------------------

        missing = [
            column
            for column in self.FEATURE_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise RuntimeError(
                "TRACE-X feature construction failed. "
                "Missing features:\n"
                + "\n".join(missing)
            )

        # ----------------------------------------------------
        # Exact ordering
        # ----------------------------------------------------

        result = df[
            self.FEATURE_COLUMNS
        ].copy()

        # ----------------------------------------------------
        # Numeric cleaning
        # ----------------------------------------------------

        result = result.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        result = result.fillna(
            0.0
        )

        result = result.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Final numeric safety check
        # ----------------------------------------------------

        values = result.to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(
            values
        ).all():

            raise RuntimeError(
                "TRACE-X generated non-finite feature values."
            )

        # ----------------------------------------------------
        # Final 38-feature assertion
        # ----------------------------------------------------

        if result.shape[1] != 38:

            raise RuntimeError(
                "TRACE-X generated an invalid feature count: "
                f"{result.shape[1]}"
            )

        return result

    # ========================================================
    # CONVENIENCE METHOD
    # ========================================================

    def build_vector(
        self,
        transaction: Mapping[str, Any],
        history: Mapping[str, Any] | None = None,
    ) -> list[float]:

        dataframe = self.build(
            transaction,
            history,
        )

        return (
            dataframe
            .iloc[0]
            .astype(np.float32)
            .tolist()
        )

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> dict[str, Any]:

        return {
            "feature_service": "ready",
            "feature_count": len(
                self.FEATURE_COLUMNS
            ),
            "frequency_lookup": (
                FREQUENCY_PATH.name
            ),
            "frequency_lookup_exists": (
                FREQUENCY_PATH.exists()
            ),
            "categorical_feature_count": len(
                self.CATEGORICAL_COLUMNS
            ),
            "historical_feature_count": len(
                self.HISTORICAL_COLUMNS
            ),
        }


# ============================================================
# SINGLETON
# ============================================================

feature_service = FeatureService()