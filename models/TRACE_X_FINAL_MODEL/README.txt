TRACE-X FINAL MODEL
===================

Selected model:
    XGBoost V1

Primary metric:
    AUPRC

Final test performance:
    AUPRC     : 0.214005
    ROC-AUC   : 0.965293
    Precision : 0.439791
    Recall    : 0.209476
    F1        : 0.283784

Locked threshold:
    0.76

Dataset:
    6,924,049 transactions
    3,565 laundering transactions

Temporal split:
    TRAIN      : 2022-09-01 -> 2022-09-07
    VALIDATION : 2022-09-08
    TEST       : 2022-09-09 -> 2022-09-17

Training policy:
    - No random split
    - No oversampling
    - No synthetic transactions
    - Train-only categorical frequency encoding
    - Past-only historical features
    - Validation used for threshold selection
    - Test used only for final evaluation

Model:
    TRACE_X_XGBOOST_GPU.json

Audit:
    TRACE_X_FINAL_MODEL_AUDIT.json

Inference:
    prediction_pipeline.py

Important:
    The inference pipeline requires the same
    feature-generation assumptions used during
    model development.