
# ============================================================
# TRACE-X
# PRODUCTION INFERENCE PIPELINE
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "TRACE_X_XGBOOST_GPU.json"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "TRACE_X_MODEL_FEATURES.json"
)

AUDIT_PATH = os.path.join(
    BASE_DIR,
    "TRACE_X_FINAL_MODEL_AUDIT.json"
)

THRESHOLD = 0.76

TARGET = "Is_Laundering"

CATEGORICAL_COLUMNS = [
    "From_Bank",
    "To_Bank",
    "Sender_Account",
    "Receiver_Account",
    "Receiving_Currency",
    "Payment_Currency",
    "Payment_Format",
]

DROP_COLUMNS = [
    TARGET,
    "Timestamp",
] + CATEGORICAL_COLUMNS


def load_model():

    model = xgb.Booster()

    model.load_model(
        MODEL_PATH
    )

    return model


def load_feature_list():

    with open(
        FEATURE_PATH,
        "r"
    ) as f:

        return json.load(f)


def add_frequency_features(
    dataframe,
    training_dataframe
):

    dataframe = dataframe.copy()

    for column in CATEGORICAL_COLUMNS:

        counts = (
            training_dataframe[column]
            .value_counts(
                dropna=False
            )
        )

        dataframe[
            column + "_Freq"
        ] = (
            dataframe[column]
            .map(counts)
            .fillna(0)
            .astype(np.float32)
            / len(training_dataframe)
        )

    return dataframe


def prepare_features(
    dataframe,
    training_dataframe
):

    dataframe = add_frequency_features(
        dataframe,
        training_dataframe
    )

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in DROP_COLUMNS
    ]

    X = (
        dataframe[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .astype(np.float32)
        .to_numpy()
    )

    return X, feature_columns


def predict(
    dataframe,
    training_dataframe
):

    model = load_model()

    X, feature_columns = prepare_features(
        dataframe,
        training_dataframe
    )

    predictions = model.predict(
        xgb.DMatrix(X)
    )

    result = dataframe.copy()

    result["TRACE_X_Risk_Score"] = (
        predictions
    )

    result["TRACE_X_Prediction"] = np.where(
        predictions >= THRESHOLD,
        "FLAG",
        "NORMAL"
    )

    result["TRACE_X_Threshold"] = (
        THRESHOLD
    )

    return result


if __name__ == "__main__":

    print(
        "TRACE-X inference pipeline loaded."
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Threshold:",
        THRESHOLD
    )
