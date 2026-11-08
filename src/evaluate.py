
"""Regression evaluation metrics."""

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true_log, y_pred_log):

    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    return {
        "MAE_INR": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE_INR": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2_INR": float(
            r2_score(y_true, y_pred)
        ),
    }
