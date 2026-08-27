"""
TRACE-X
Frozen V1 inference service.

Pipeline:
    Transaction
        ↓
    HistoricalFeatureEngine
        ↓
    FeatureService
        ↓
    38 model features
        ↓
    Frozen XGBoost V1

IMPORTANT:
- No retraining
- Threshold remains locked at 0.76
- Model remains frozen
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# TRACE-X SERVICES
# ============================================================

from backend.feature_service import FeatureService
from backend.historical_engine import HistoricalFeatureEngine


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "TRACE_X_FINAL_MODEL"
)

MODEL_PATH = (
    MODEL_DIR
    / "TRACE_X_XGBOOST_GPU.json"
)

FEATURE_PATH = (
    MODEL_DIR
    / "TRACE_X_MODEL_FEATURES.json"
)

AUDIT_PATH = (
    MODEL_DIR
    / "TRACE_X_FINAL_MODEL_AUDIT.json"
)


# ============================================================
# LOCKED MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "TRACE-X V1"
THRESHOLD = 0.76


# ============================================================
# MODEL SERVICE
# ============================================================

class TraceXInference:

    def __init__(self) -> None:

        self.model = None
        self.feature_columns = None
        self.audit = None

        self.feature_service = None
        self.historical_engine = None

        self._load()
        self._load_feature_services()


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    def _load(self) -> None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"TRACE-X model not found:\n{MODEL_PATH}"
            )

        if not FEATURE_PATH.exists():
            raise FileNotFoundError(
                f"Feature contract not found:\n{FEATURE_PATH}"
            )

        if not AUDIT_PATH.exists():
            raise FileNotFoundError(
                f"Model audit not found:\n{AUDIT_PATH}"
            )

        self.model = xgb.Booster()

        self.model.load_model(
            str(MODEL_PATH)
        )

        import json

        with open(
            FEATURE_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            self.feature_columns = json.load(f)

        with open(
            AUDIT_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            self.audit = json.load(f)


    # --------------------------------------------------------
    # LOAD FEATURE SERVICES
    # --------------------------------------------------------

    def _load_feature_services(self) -> None:

        self.feature_service = FeatureService()

        self.historical_engine = (
            HistoricalFeatureEngine()
        )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def status(self) -> dict[str, Any]:

        return {
            "model": MODEL_NAME,
            "status": "loaded",
            "frozen": True,
            "boosting_rounds": (
                self.model.num_boosted_rounds()
            ),
            "feature_count": len(
                self.feature_columns
            ),
            "threshold": THRESHOLD,
            "model_file": MODEL_PATH.name,
            "historical_engine": "ready",
            "feature_service": "ready",
        }


    # --------------------------------------------------------
    # DIRECT VECTOR PREDICTION
    # --------------------------------------------------------

    def predict_vector(
        self,
        features: list[float],
    ) -> dict[str, Any]:

        if len(features) != len(
            self.feature_columns
        ):

            raise ValueError(
                "Feature count mismatch. "
                f"Expected "
                f"{len(self.feature_columns)}, "
                f"received "
                f"{len(features)}."
            )

        X = np.asarray(
            features,
            dtype=np.float32,
        ).reshape(1, -1)

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        prediction = float(
            self.model.predict(
                xgb.DMatrix(X)
            )[0]
        )

        decision = (
            "FLAG"
            if prediction >= THRESHOLD
            else "NORMAL"
        )

        return {
            "model": MODEL_NAME,
            "risk_score": prediction,
            "threshold": THRESHOLD,
            "decision": decision,
        }


    # --------------------------------------------------------
    # REAL TRANSACTION PREDICTION
    # --------------------------------------------------------

    def predict_transaction(
        self,
        transaction: dict[str, Any],
    ) -> dict[str, Any]:

        if self.feature_service is None:
            raise RuntimeError(
                "Feature service is not initialized."
            )

        if self.historical_engine is None:
            raise RuntimeError(
                "Historical engine is not initialized."
            )

        # ----------------------------------------------------
        # 1. Historical features
        # ----------------------------------------------------

        history = (
            self.historical_engine.get_history(
                transaction
            )
        )

        # ----------------------------------------------------
        # 2. Build model features
        # ----------------------------------------------------

        feature_frame = (
            self.feature_service.build(
                transaction=transaction,
                history=history,
            )
        )

        # ----------------------------------------------------
        # 3. Verify exact feature contract
        # ----------------------------------------------------

        actual_columns = (
            feature_frame.columns.tolist()
        )

        if actual_columns != (
            self.feature_columns
        ):

            raise ValueError(
                "Model feature contract mismatch.\n"
                f"Expected: {self.feature_columns}\n"
                f"Received: {actual_columns}"
            )

        # ----------------------------------------------------
        # 4. Convert to model matrix
        # ----------------------------------------------------

        X = (
            feature_frame
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .astype(np.float32)
            .to_numpy()
        )

        # ----------------------------------------------------
        # 5. Frozen model prediction
        # ----------------------------------------------------

        prediction = float(
            self.model.predict(
                xgb.DMatrix(X)
            )[0]
        )

        decision = (
            "FLAG"
            if prediction >= THRESHOLD
            else "NORMAL"
        )

        # ----------------------------------------------------
        # 6. Return investigation-ready result
        # ----------------------------------------------------

        return {
            "model": MODEL_NAME,
            "risk_score": prediction,
            "threshold": THRESHOLD,
            "decision": decision,
            "feature_count": len(
                actual_columns
            ),
            "historical_features": history,
        }


# ============================================================
# SINGLETON MODEL
# ============================================================

trace_x_model = TraceXInference()