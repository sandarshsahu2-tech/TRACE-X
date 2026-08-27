"""
TRACE-X
Live feature construction service.

Builds the exact 38-feature contract expected by TRACE-X V1.

Sources:
- Past-only historical features
- TRAIN-only frequency lookup
- Transaction-level amount/time features

No retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FREQUENCY_PATH = (
    PROJECT_ROOT
    / "models"
    / "TRACE_X_FINAL_MODEL"
    / "frequency_lookup.parquet"
)


class FeatureService:

    CATEGORICAL_COLUMNS = [
        "From_Bank",
        "To_Bank",
        "Sender_Account",
        "Receiver_Account",
        "Receiving_Currency",
        "Payment_Currency",
        "Payment_Format",
    ]

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

    def __init__(self) -> None:

        if not FREQUENCY_PATH.exists():
            raise FileNotFoundError(
                f"Frequency lookup not found:\n{FREQUENCY_PATH}"
            )

        lookup = pd.read_parquet(
            FREQUENCY_PATH
        )

        self.frequency_maps: dict[
            str,
            dict[str, float]
        ] = {}

        for feature in self.CATEGORICAL_COLUMNS:

            rows = lookup[
                lookup["feature"] == feature
            ]

            self.frequency_maps[feature] = dict(
                zip(
                    rows["value"].astype(str),
                    rows["frequency"].astype(float),
                )
            )

    # ============================================================
    # FREQUENCY FEATURES
    # ============================================================

    def _add_frequency_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        for column in self.CATEGORICAL_COLUMNS:

            df[
                column + "_Freq"
            ] = (
                df[column]
                .astype(str)
                .map(
                    self.frequency_maps[
                        column
                    ]
                )
                .fillna(0.0)
                .astype(np.float32)
            )

        return df

    # ============================================================
    # BUILD FEATURES
    # ============================================================

    def build(
        self,
        transaction: dict[str, Any],
        history: dict[str, Any],
    ) -> pd.DataFrame:

        df = pd.DataFrame(
            [transaction]
        )

        # --------------------------------------------------------
        # Amount features
        # --------------------------------------------------------

        received = float(
            df["Amount_Received"].iloc[0]
        )

        paid = float(
            df["Amount_Paid"].iloc[0]
        )

        df["Log_Amount_Received"] = np.log1p(
            max(abs(received), 0.0)
        )

        df["Log_Amount_Paid"] = np.log1p(
            max(abs(paid), 0.0)
        )

        difference = abs(
            received - paid
        )

        df["Absolute_Amount_Difference"] = (
            difference
        )

        df["Relative_Amount_Difference"] = (
            difference
            / max(
                abs(received),
                abs(paid),
                1e-9,
            )
        )

        df["Received_Paid_Ratio"] = (
            received
            / max(
                abs(paid),
                1e-9,
            )
        )

        # --------------------------------------------------------
        # Historical features
        # --------------------------------------------------------

        for column in self.FEATURE_COLUMNS:

            if column in history:

                df[column] = history[column]

        # --------------------------------------------------------
        # Temporal features
        # --------------------------------------------------------

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

        weekday = int(
            timestamp.weekday()
        )

        day = int(
            timestamp.day
        )

        month = int(
            timestamp.month
        )

        df["Transaction_Hour"] = hour
        df["Transaction_DayOfWeek"] = weekday
        df["Transaction_Day"] = day
        df["Transaction_Month"] = month

        df["Is_Weekend"] = int(
            weekday >= 5
        )

        df["Is_Night"] = int(
            hour < 6 or hour >= 22
        )

        df["Is_Evening"] = int(
            18 <= hour < 22
        )

        from_bank = str(
            df["From_Bank"].iloc[0]
        )

        to_bank = str(
            df["To_Bank"].iloc[0]
        )

        receiving_currency = str(
            df["Receiving_Currency"].iloc[0]
        )

        payment_currency = str(
            df["Payment_Currency"].iloc[0]
        )

        df["Is_Cross_Bank_Cross_Currency"] = int(
            from_bank != to_bank
            and
            receiving_currency != payment_currency
        )

        df["Hour_Sin"] = np.sin(
            2 * np.pi * hour / 24
        )

        df["Hour_Cos"] = np.cos(
            2 * np.pi * hour / 24
        )

        df["DayOfWeek_Sin"] = np.sin(
            2 * np.pi * weekday / 7
        )

        df["DayOfWeek_Cos"] = np.cos(
            2 * np.pi * weekday / 7
        )

        # --------------------------------------------------------
        # Frequency features
        # --------------------------------------------------------

        df = self._add_frequency_features(
            df
        )

        # --------------------------------------------------------
        # Final model contract
        # --------------------------------------------------------

        missing = [
            column
            for column in self.FEATURE_COLUMNS
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing model features: "
                + ", ".join(missing)
            )

        return (
            df[
                self.FEATURE_COLUMNS
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .astype(np.float32)
        )