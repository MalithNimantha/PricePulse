"""Read Colombo retail prices and select product series using early coverage."""

import re

import numpy as np
import pandas as pd

PRODUCT_COUNT = 24
MINIMUM_COVERAGE = 0.80
TRAIN_SHARE = 0.70
VALIDATION_END_SHARE = 0.85
CURRENCY_SUFFIX = " (Rs)"
UNIT_PATTERN = re.compile(
    r"\s*(\(?(?:\d+\s*(?:kg\s*\.?\s*pkt|kg|ml|g|leaves|nuts)|each|bunch)\)?)$",
    re.IGNORECASE,
)


def read_retail_csv(path, *, expected_products=()):
    """Load the CSV using the saved model's product labels when supplied."""
    wide = pd.read_csv(path, comment="#", encoding="utf-8-sig")
    if "period" not in wide:
        raise ValueError("Expected a period column in the retail-price CSV")
    # Keep Macro Lab's currency suffix when it is part of the trained category.
    if not any(product.endswith(CURRENCY_SUFFIX) for product in expected_products):
        wide.columns = wide.columns.str.removesuffix(CURRENCY_SUFFIX)
    if wide.columns.has_duplicates:
        raise ValueError("Product headings must be unique in the retail-price CSV")
    missing = [product for product in expected_products if product not in wide.columns]
    if missing:
        raise ValueError(
            f"The retail-price CSV at {path} is missing product columns required by the saved model: "
            f"{', '.join(missing)}"
        )
    wide["period"] = pd.to_datetime(wide["period"], format="%Y-%m-%d", errors="raise")
    if wide["period"].isna().any() or wide["period"].duplicated().any():
        raise ValueError("Reporting periods must be present and unique")
    return wide.sort_values("period", ignore_index=True)


def observed_prices(wide):
    long = wide.rename(columns={"period": "date"}).melt(
        id_vars="date", var_name="product", value_name="price_lkr"
    )
    long["price_lkr"] = pd.to_numeric(long["price_lkr"], errors="coerce")
    valid = np.isfinite(long["price_lkr"]) & long["price_lkr"].gt(0)
    return long.loc[valid].sort_values(["product", "date"], ignore_index=True)


def split_dates(wide):
    """Return the first origin date in validation and test, respectively."""
    dates = wide["period"]
    return dates.iloc[int(len(dates) * TRAIN_SHARE)], dates.iloc[int(len(dates) * VALIDATION_END_SHARE)]


def choose_products(observed, wide, validation_start):
    early = observed.loc[observed["date"].lt(validation_start)]
    periods = wide["period"].lt(validation_start).sum()
    coverage = (early.groupby("product", sort=True).size() / periods).sort_values(
        ascending=False, kind="stable"
    )
    return coverage[coverage.ge(MINIMUM_COVERAGE)].head(PRODUCT_COUNT)


def product_name_and_unit(product):
    product = product.removesuffix(CURRENCY_SUFFIX)
    match = UNIT_PATTERN.search(product)
    if not match:
        return product, "as reported"
    unit = match[1].strip("()").lower()
    unit = re.sub(r"^(\d+)\s*(?=[a-z])", r"\1 ", unit)
    return product[:match.start()].strip(), unit
