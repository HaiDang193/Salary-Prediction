
"""Optional: retrain and save XGBoost model."""

import json
import joblib

from src.config import ROOT_DIR
from src.train import train_tuned_model
from src.evaluate import regression_metrics


def main():

    model, X_test, y_test, params = train_tuned_model()

    metrics = regression_metrics(
        y_test,
        model.predict(X_test),
    )

    model_dir = ROOT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        model_dir / "car_price_xgboost.joblib",
    )

    metadata = {
        "features": X_test.columns.tolist(),
        "target_transform": "log1p",
        "reference_year": 2026,
    }

    (model_dir / "model_features.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "best_params": params,
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    print("[SUCCESS] Model saved")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
