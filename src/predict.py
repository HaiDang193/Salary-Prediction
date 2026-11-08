
"""Load model and predict car prices."""

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT_DIR
from src.preprocessing import preprocess_raw


MODEL_PATH = ROOT_DIR / "models" / "car_price_xgboost.joblib"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Model does not contain feature names. "
            "Please check how it was trained."
        )

    features = model.feature_names_in_.tolist()

    return model, features


def predict_prices(cars: pd.DataFrame) -> np.ndarray:
    model, features = load_model()

    X = preprocess_raw(cars, features)

    predictions_log = model.predict(X)

    return np.expm1(predictions_log)
