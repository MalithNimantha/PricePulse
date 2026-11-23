"""Score next-price forecasts and save their underlying held-out records."""

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.features import TARGET
from src.training import ENSEMBLES, predict_next

NAIVE = "Naive Baseline"
PREDICTIONS = {
    "Random Forest": "random_forest_lkr",
    "Gradient Boosting": "gradient_boosting_lkr",
    "Stacking": "stacking_lkr",
    NAIVE: "naive_baseline_lkr",
}


def score_partition(frame, models, split):
    records = frame[["product", "date", "target_date", "price_lkr", TARGET]].rename(
        columns={"date": "origin_date", "price_lkr": "latest_price_lkr", TARGET: "actual_price_lkr"}
    )
    for name in ENSEMBLES:
        records[PREDICTIONS[name]] = predict_next(models[name], frame)
    records[PREDICTIONS[NAIVE]] = records["latest_price_lkr"]

    actual = records["actual_price_lkr"]
    metrics = pd.DataFrame(
        [
            {
                "split": split,
                "model": name,
                "mae_lkr": mean_absolute_error(actual, records[column]),
                "rmse_lkr": root_mean_squared_error(actual, records[column]),
                "r2": r2_score(actual, records[column]),
                "observations": len(records),
            }
            for name, column in PREDICTIONS.items()
        ]
    )
    return metrics, records


def select_ensemble(validation_metrics):
    eligible = validation_metrics[validation_metrics["model"].isin(ENSEMBLES)]
    return eligible.loc[eligible["mae_lkr"].idxmin(), "model"]


def save_results(directory, metrics, test_predictions):
    directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(directory / "model_metrics.csv", index=False)
    test_predictions.sort_values(["target_date", "product"]).to_csv(
        directory / "test_predictions.csv", index=False, date_format="%Y-%m-%d"
    )
