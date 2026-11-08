
"""XGBoost model training."""

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold,
)
from xgboost import XGBRegressor

from src.config import DATA_DIR


PARAM_GRID = {
    "n_estimators": [100, 200, 300, 400, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def train_tuned_model(n_iter=180):

    df = pd.read_csv(
        DATA_DIR / "processed" / "car_processed.csv"
    )

    X = df.drop(columns="log_selling_price")
    y = df["log_selling_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    search = RandomizedSearchCV(
        estimator=XGBRegressor(
            random_state=42,
            n_jobs=1,
        ),
        param_distributions=PARAM_GRID,
        n_iter=n_iter,
        cv=KFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        ),
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
        refit="r2",
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    return (
        search.best_estimator_,
        X_test,
        y_test,
        search.best_params_,
    )
