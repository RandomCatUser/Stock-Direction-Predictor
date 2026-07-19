from __future__ import annotations

import argparse
import os

import pandas as pd

from data_loader import fetch_data
from evaluate import (
    backtest_strategy,
    classification_report_dict,
    confusion_matrix_df,
    summarize_backtest,
    walk_forward_evaluate,
)
from features import build_feature_table
from model import chronological_split, get_model_zoo, save_model, train_model, NaiveBaselines


def parse_args():
    p = argparse.ArgumentParser(description="Stock direction prediction pipeline")
    p.add_argument("--ticker", default="AAPL", help="Stock ticker symbol")
    p.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument(
        "--source",
        default="yfinance",
        choices=["yfinance", "synthetic"],
        help="Data source. Use 'synthetic' if you have no internet access.",
    )
    p.add_argument("--horizon", type=int, default=1, help="Prediction horizon in trading days")
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction of data held out for final test")
    p.add_argument("--folds", type=int, default=5, help="Number of walk-forward validation folds")
    p.add_argument("--cost-bps", type=float, default=5.0, help="Round-trip trading cost in basis points")
    p.add_argument("--out-dir", default="models", help="Where to save the trained model")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[1/5] Fetching data for {args.ticker} (source={args.source})...")
    raw = fetch_data(args.ticker, start=args.start, end=args.end, source=args.source)
    print(f"    -> {len(raw)} rows from {raw.index.min().date()} to {raw.index.max().date()}")

    print("\n[2/5] Engineering features...")
    table, feature_cols = build_feature_table(raw, horizon=args.horizon)
    print(f"    -> {len(table)} usable rows, {len(feature_cols)} features")
    print(f"    -> target balance: {table['target'].value_counts(normalize=True).to_dict()}")

    print("\n[3/5] Chronological train/test split + baselines...")
    train_df, test_df = chronological_split(table, test_size=args.test_size)
    baselines = NaiveBaselines.compute(train_df, test_df)
    print(f"    -> {baselines}")

    print("\n[4/5] Training & comparing models (final holdout test)...")
    results = {}
    fitted = {}
    for name, pipe in get_model_zoo().items():
        trained = train_model(pipe, train_df, feature_cols)
        preds = trained.predict(test_df[feature_cols])
        proba = trained.predict_proba(test_df[feature_cols])[:, 1]
        metrics = classification_report_dict(test_df["target"], preds, proba)
        results[name] = metrics
        fitted[name] = trained
        print(f"    {name:20s} acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}  auc={metrics['roc_auc']:.3f}")

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = fitted[best_name]
    print(f"\n    Best model by ROC-AUC on holdout: {best_name}")
    print("\n    Confusion matrix (best model):")
    print(confusion_matrix_df(test_df["target"], best_model.predict(test_df[feature_cols])))

    print(f"\n[5/5] Walk-forward validation ({args.folds} folds) for {best_name}...")
    wf = walk_forward_evaluate(
        lambda: get_model_zoo()[best_name], table, feature_cols, n_folds=args.folds
    )
    print(wf[["accuracy", "precision", "recall", "f1", "roc_auc", "test_start", "test_end"]])
    print(f"\n    Mean walk-forward accuracy: {wf['accuracy'].mean():.3f} (+/- {wf['accuracy'].std():.3f})")

    print("\nBacktest (best model, holdout period, net of trading costs):")
    preds = best_model.predict(test_df[feature_cols])
    bt = backtest_strategy(test_df, preds, cost_bps=args.cost_bps)
    summary = summarize_backtest(bt)
    for k, v in summary.items():
        print(f"    {k}: {v}")

    model_path = os.path.join(args.out_dir, f"{args.ticker}_{best_name}.joblib")
    save_model(best_model, model_path)
    print(f"\nSaved best model to {model_path}")
    print(
        "\nReminder: this is an educational project. Directional accuracy hovering "
        "near ~50-55% is the REALISTIC outcome for public daily OHLCV data -- "
        "markets are close to efficient, and this is not investment advice."
    )


if __name__ == "__main__":
    main()
