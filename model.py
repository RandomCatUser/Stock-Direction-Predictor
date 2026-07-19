"""
model.py
========
Trains and persists classification models that predict next-period stock
direction (up/down) from the engineered feature table.

Key methodological choices (the "reasoning" the user asked for):

1. NO RANDOM SHUFFLING / NO K-FOLD CV.
   Financial data is a time series. A normal random train/test split would
   let the model "see the future" (train on day 500, test on day 300),
   which inflates accuracy in a way that will NOT hold up in real trading.
   We always split chronologically: train on the past, test on the future.

2. WALK-FORWARD VALIDATION.
   Instead of one static split, we roll the split forward in folds -- train
   on an expanding window, test on the next chunk, repeat. This tells us
   whether performance is consistent over different market regimes (bull,
   bear, sideways) rather than lucky on one particular test window.

3. MULTIPLE MODEL FAMILIES, COMPARED FAIRLY.
   - Logistic Regression: a simple, highly interpretable linear baseline.
     If fancier models can't beat this, the extra complexity isn't earning
     its keep.
   - Random Forest: captures non-linear interactions between indicators
     (e.g. "RSI oversold AND volume spike") without much tuning.
   - Gradient Boosting: usually the strongest tabular-data performer, at
     the cost of being slower and more prone to overfitting without care.

4. A "NAIVE" BASELINE IS ALWAYS REPORTED ALONGSIDE MODELS.
   Predicting "tomorrow moves the same direction as today" or "always
   predict the majority class" are the bars a real model must clear. Many
   published stock-prediction demos skip this and report bare accuracy,
   which is misleading -- 51% accuracy on a ~50/50 balanced target is not
   impressive without a baseline for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split a time-ordered dataframe into train/test WITHOUT shuffling.
    The test set is always the most recent `test_size` fraction of rows."""
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def get_model_zoo(random_state: int = 42) -> dict[str, Pipeline]:
    """Return a dict of name -> sklearn Pipeline (scaler + classifier).

    All models share a StandardScaler step. Tree-based models don't
    strictly need scaling, but keeping the pipeline uniform makes the
    comparison code simpler and doesn't hurt them.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=20,  # guards against overfitting to noise
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,  # stochastic GB, reduces overfitting
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


@dataclass
class NaiveBaselines:
    """Two simple baselines every real model must beat to be worth using."""

    majority_class: int
    persistence_acc_train: float

    @staticmethod
    def compute(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        majority = int(train_df["target"].mode()[0])
        majority_acc = (test_df["target"] == majority).mean()
        # "persistence": predict today's direction repeats tomorrow
        if "return_1d" in test_df.columns:
            naive_pred = (test_df["return_1d"] > 0).astype(int)
            persistence_acc = (naive_pred == test_df["target"]).mean()
        else:
            persistence_acc = np.nan
        return {"majority_class_accuracy": majority_acc, "persistence_accuracy": persistence_acc}


def walk_forward_splits(n_rows: int, n_folds: int = 5, min_train_size: float = 0.4):
    """Yield (train_idx, test_idx) index arrays for expanding-window
    walk-forward validation. Each fold trains on everything up to a point
    and tests on the next equally-sized chunk."""
    start = int(n_rows * min_train_size)
    remaining = n_rows - start
    fold_size = remaining // n_folds
    for i in range(n_folds):
        train_end = start + i * fold_size
        test_end = train_end + fold_size if i < n_folds - 1 else n_rows
        yield np.arange(0, train_end), np.arange(train_end, test_end)


def train_model(pipeline: Pipeline, train_df: pd.DataFrame, feature_cols: list[str]):
    X_train, y_train = train_df[feature_cols], train_df["target"]
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline: Pipeline, path: str = "models/model.joblib"):
    joblib.dump(pipeline, path)


def load_model(path: str = "models/model.joblib") -> Pipeline:
    return joblib.load(path)


if __name__ == "__main__":
    from data_loader import fetch_data
    from features import build_feature_table

    raw = fetch_data("DEMO", start="2015-01-01", end="2024-01-01", source="synthetic")
    table, feature_cols = build_feature_table(raw)
    train_df, test_df = chronological_split(table, test_size=0.2)

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print("Baselines:", NaiveBaselines.compute(train_df, test_df))

    for name, pipe in get_model_zoo().items():
        trained = train_model(pipe, train_df, feature_cols)
        acc = trained.score(test_df[feature_cols], test_df["target"])
        print(f"{name}: test accuracy = {acc:.4f}")
