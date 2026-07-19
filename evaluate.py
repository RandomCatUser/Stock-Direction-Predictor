"""
evaluate.py
===========
Honest evaluation of a stock-direction classifier: standard classification
metrics, walk-forward robustness checks, and a simple long/flat strategy
backtest translated into $ terms (not just accuracy %).

Why go beyond accuracy?
-----------------------
Accuracy alone can be misleading for trading:
  - A model can have >50% accuracy but still lose money if it's right on
    small moves and wrong on big ones.
  - A model can look great on one lucky test window and fail elsewhere --
    that's what walk-forward validation and reporting variance across
    folds is for.
  - Transaction costs are ignored in almost every "amazing 90% accuracy"
    stock prediction tutorial online; we include a cost assumption here to
    stay honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report_dict(y_true, y_pred, y_proba=None) -> dict:
    """Compute the core metrics we care about, all in one dict."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = np.nan
    return metrics


def confusion_matrix_df(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=["Actual: Down", "Actual: Up"],
        columns=["Predicted: Down", "Predicted: Up"],
    )


def walk_forward_evaluate(pipeline_factory, table: pd.DataFrame, feature_cols: list[str], n_folds=5):
    """Run walk-forward validation and return a DataFrame of per-fold metrics.

    `pipeline_factory` is a zero-arg callable returning a fresh, untrained
    sklearn Pipeline (so each fold trains from scratch, no leakage between
    folds).
    """
    from model import walk_forward_splits

    rows = []
    for i, (train_idx, test_idx) in enumerate(walk_forward_splits(len(table), n_folds=n_folds)):
        train_df, test_df = table.iloc[train_idx], table.iloc[test_idx]
        pipe = pipeline_factory()
        pipe.fit(train_df[feature_cols], train_df["target"])
        preds = pipe.predict(test_df[feature_cols])
        proba = pipe.predict_proba(test_df[feature_cols])[:, 1]
        m = classification_report_dict(test_df["target"], preds, proba)
        m["fold"] = i + 1
        m["train_size"] = len(train_df)
        m["test_size"] = len(test_df)
        m["test_start"] = test_df.index.min()
        m["test_end"] = test_df.index.max()
        rows.append(m)
    return pd.DataFrame(rows).set_index("fold")


def backtest_strategy(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Simulate a simple strategy: go long (hold the stock) on days the
    model predicts UP, stay in cash on days it predicts DOWN. Compare
    cumulative return against passive buy-and-hold.

    cost_bps: round-trip transaction cost in basis points charged whenever
    the strategy changes position (enter or exit). This is a simplification
    but far more honest than ignoring costs entirely, which is the norm in
    naive online tutorials.
    """
    df = test_df.copy()
    df["prediction"] = predictions
    df["market_return"] = df["future_return"]  # actual next-period return
    df["strategy_gross_return"] = np.where(df["prediction"] == 1, df["market_return"], 0.0)

    position_changes = df["prediction"].diff().abs().fillna(1)  # 1 = trade at start
    trading_cost = position_changes * (cost_bps / 10_000)
    df["strategy_net_return"] = df["strategy_gross_return"] - trading_cost

    df["strategy_cum_return"] = (1 + df["strategy_net_return"]).cumprod() - 1
    df["buy_hold_cum_return"] = (1 + df["market_return"]).cumprod() - 1
    return df[
        [
            "prediction",
            "market_return",
            "strategy_net_return",
            "strategy_cum_return",
            "buy_hold_cum_return",
        ]
    ]


def summarize_backtest(bt: pd.DataFrame) -> dict:
    n_days = len(bt)
    ann_factor = 252 / n_days if n_days else np.nan
    strat_total = bt["strategy_cum_return"].iloc[-1]
    hold_total = bt["buy_hold_cum_return"].iloc[-1]
    strat_daily_std = bt["strategy_net_return"].std()
    sharpe = (
        (bt["strategy_net_return"].mean() / strat_daily_std) * np.sqrt(252)
        if strat_daily_std > 0
        else np.nan
    )
    return {
        "strategy_total_return": strat_total,
        "buy_hold_total_return": hold_total,
        "annualized_strategy_return_est": (1 + strat_total) ** ann_factor - 1,
        "sharpe_ratio_est": sharpe,
        "n_test_days": n_days,
    }


if __name__ == "__main__":
    from data_loader import fetch_data
    from features import build_feature_table
    from model import chronological_split, get_model_zoo, train_model

    raw = fetch_data("DEMO", start="2015-01-01", end="2024-01-01", source="synthetic")
    table, feature_cols = build_feature_table(raw)
    train_df, test_df = chronological_split(table, test_size=0.2)

    pipe = train_model(get_model_zoo()["gradient_boosting"], train_df, feature_cols)
    preds = pipe.predict(test_df[feature_cols])
    proba = pipe.predict_proba(test_df[feature_cols])[:, 1]

    print("Metrics:", classification_report_dict(test_df["target"], preds, proba))
    print("\nConfusion matrix:")
    print(confusion_matrix_df(test_df["target"], preds))

    bt = backtest_strategy(test_df, preds)
    print("\nBacktest summary:", summarize_backtest(bt))
