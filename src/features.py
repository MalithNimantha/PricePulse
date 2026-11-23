"""Build point-in-time features and the next observed price for each product."""

import pandas as pd

FEATURES = [
    "product",
    "price_lkr",
    "lag_1",
    "lag_2",
    "lag_3",
    "change_1",
    "previous_change",
    "rolling_mean_4",
    "rolling_std_4",
    "days_since_previous",
    "month",
    "week_of_year",
]
TARGET = "target_price_lkr"


def build_features(observed):
    frame = observed.sort_values(["product", "date"], ignore_index=True).copy()
    groups = frame.groupby("product", sort=False)
    prices = groups["price_lkr"]

    for lag in (1, 2, 3):
        frame[f"lag_{lag}"] = prices.shift(lag)
    frame["change_1"] = frame["price_lkr"] - frame["lag_1"]
    frame["previous_change"] = frame["lag_1"] - frame["lag_2"]
    frame["rolling_mean_4"] = prices.transform(lambda series: series.rolling(4).mean())
    frame["rolling_std_4"] = prices.transform(lambda series: series.rolling(4).std())
    # Keep the actual elapsed days between observed reports, including skipped periods.
    gaps = groups["date"].diff().dt.days
    if gaps.eq(0).any():
        raise ValueError("Each product must have at most one observed price per reporting date")
    frame["days_since_previous"] = gaps
    frame["month"] = frame["date"].dt.month
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype("int64")

    # The only forward shifts create labels. Keep the last, unlabeled row for inference.
    frame[TARGET] = prices.shift(-1)
    frame["target_date"] = groups["date"].shift(-1)
    return frame.dropna(subset=FEATURES).sort_values(["date", "product"], ignore_index=True)


def chronological_partitions(frame, validation_start, test_start):
    labeled = frame.dropna(subset=[TARGET, "target_date"])
    train = labeled[labeled["target_date"].lt(validation_start)]
    validation = labeled[
        labeled["date"].ge(validation_start) & labeled["target_date"].lt(test_start)
    ]
    test = labeled[labeled["date"].ge(test_start)]
    return {"Train": train, "Validation": validation, "Test": test}


def partition_summary(partitions):
    return pd.DataFrame.from_dict(
        {
            name: {
                "Rows": len(frame),
                "First origin": frame["date"].min().date(),
                "Last origin": frame["date"].max().date(),
                "Last target": frame["target_date"].max().date(),
            }
            for name, frame in partitions.items()
        },
        orient="index",
    )
