from pathlib import Path


# ============================================================
# TRACE-X | CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

REFERENCE_DB = (
    MODEL_DIR
    / "TRACE_X_REFERENCE.duckdb"
)

THRESHOLD = 0.76

MODEL_NAME = "TRACE-X V1"

CATEGORICAL_COLUMNS = [
    "From_Bank",
    "To_Bank",
    "Sender_Account",
    "Receiver_Account",
    "Receiving_Currency",
    "Payment_Currency",
    "Payment_Format",
]