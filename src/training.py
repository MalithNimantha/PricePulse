"""Fit three ensemble regressors with chronological out-of-fold stacking."""

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import FEATURES, TARGET

SEED = 42
FOREST = "Random Forest"
BOOSTING = "Gradient Boosting"
STACKING = "Stacking"
BASE_MODELS = (FOREST, BOOSTING)
ENSEMBLES = (*BASE_MODELS, STACKING)


def make_base_model(name):
    estimators = {
        FOREST: RandomForestRegressor(
            n_estimators=96, max_depth=8, min_samples_leaf=14,
            max_features=0.8, random_state=SEED,
        ),
        BOOSTING: GradientBoostingRegressor(
            loss="huber", n_estimators=120, learning_rate=0.04,
            max_depth=2, min_samples_leaf=16, random_state=SEED,
        ),
    }
    encoder = ColumnTransformer(
        [("product", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["product"])],
        remainder="passthrough",
    )
    return make_pipeline(encoder, estimators[name])


def price_change(frame):
    return frame[TARGET] - frame["price_lkr"]


def fit_base_model(name, frame):
    return make_base_model(name).fit(frame[FEATURES], price_change(frame))


class ChronologicalStackingRegressor:
    """A Ridge regressor on base-model predictions from later, unseen dates."""

    def fit(self, train, fitted_base_models):
        dates = pd.Index(train["date"].unique()).sort_values()
        predictions, changes, audit = [], [], []
        for number, (fit_idx, holdout_idx) in enumerate(TimeSeriesSplit(n_splits=5).split(dates), 1):
            holdout_dates = dates[holdout_idx]
            fit = train[
                train["date"].isin(dates[fit_idx]) & train["target_date"].lt(holdout_dates[0])
            ]
            holdout = train[train["date"].isin(holdout_dates)]
            fold_models = {name: fit_base_model(name, fit) for name in BASE_MODELS}
            predictions.append(self._base_predictions(fold_models, holdout[FEATURES]))
            changes.append(price_change(holdout).to_numpy())
            audit.append(
                {
                    "Fold": number,
                    "Last fit target": fit["target_date"].max().date(),
                    "First fold origin": holdout_dates[0].date(),
                    "Last fold origin": holdout_dates[-1].date(),
                    "Fit rows": len(fit),
                    "Out-of-fold rows": len(holdout),
                }
            )

        self.meta_model_ = Ridge(alpha=100.0).fit(np.vstack(predictions), np.concatenate(changes))
        self.base_models_ = {name: fitted_base_models[name] for name in BASE_MODELS}
        self.fold_summary_ = pd.DataFrame(audit).set_index("Fold")
        return self

    @staticmethod
    def _base_predictions(models, features):
        return np.column_stack([models[name].predict(features) for name in BASE_MODELS])

    def predict(self, features):
        return self.meta_model_.predict(self._base_predictions(self.base_models_, features))


def predict_next(model, frame, feature_columns=FEATURES):
    """Convert a learned LKR change into the next reported price."""
    return frame["price_lkr"].to_numpy() + model.predict(frame[feature_columns])


def save_models(path, models, selected_model, products, train):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "selected_model": selected_model,
            "feature_columns": FEATURES.copy(),
            "products": sorted(products),
            "last_training_target": train["target_date"].max().date().isoformat(),
            "sklearn_version": sklearn.__version__,
        },
        path,
        compress=3,
    )
