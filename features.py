

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index: momentum oscillator bounded 0-100.
    >70 conventionally "overbought", <30 "oversold"."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    """Moving Average Convergence/Divergence: trend-following momentum."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _bollinger(close: pd.Series, window=20, n_std=2):
    """Bollinger Bands: volatility envelope around a moving average."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    # %B: where price sits within the band, 0 = lower band, 1 = upper band
    pct_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid
    return pct_b, bandwidth


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a standard set of technical-analysis features to an OHLCV frame."""
    out = df.copy()
    close, volume = out["Close"], out["Volume"]

    # --- Trend: moving averages and price relative to them ---
    for w in (5, 10, 20, 50):
        out[f"sma_{w}"] = close.rolling(w).mean()
        out[f"close_to_sma_{w}"] = close / out[f"sma_{w}"] - 1  # % above/below MA
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # --- Momentum ---
    out["rsi_14"] = _rsi(close, 14)
    macd_line, signal_line, hist = _macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    for w in (1, 3, 5, 10):
        out[f"return_{w}d"] = close.pct_change(w)

    # --- Volatility ---
    out["volatility_10"] = close.pct_change().rolling(10).std()
    out["volatility_20"] = close.pct_change().rolling(20).std()
    pct_b, bandwidth = _bollinger(close)
    out["bb_pct_b"] = pct_b
    out["bb_bandwidth"] = bandwidth
    high_low_range = (out["High"] - out["Low"]) / out["Close"]
    out["hl_range"] = high_low_range

    # --- Volume pressure ---
    out["volume_sma_10"] = volume.rolling(10).mean()
    out["volume_ratio"] = volume / out["volume_sma_10"]
    out["volume_change"] = volume.pct_change()

    return out


def add_lag_features(df: pd.DataFrame, columns: list[str], lags=(1, 2, 3)) -> pd.DataFrame:
    """Add lagged copies of selected columns so the model can see recent
    history explicitly (tree models don't have built-in memory of order)."""
    out = df.copy()
    for col in columns:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def make_classification_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add a binary target: 1 if Close `horizon` days ahead is higher than
    today's Close, else 0. Uses `shift(-horizon)` (future data) ONLY for the
    label column -- this is fine for a label, but that column must never be
    used as a feature."""
    out = df.copy()
    future_close = out["Close"].shift(-horizon)
    out["future_return"] = future_close / out["Close"] - 1
    out["target"] = (out["future_return"] > 0).astype(int)
    return out


def build_feature_table(
    raw: pd.DataFrame, horizon: int = 1, lag_columns=("return_1d", "rsi_14", "volume_ratio")
) -> tuple[pd.DataFrame, list[str]]:
    """Full pipeline: raw OHLCV -> indicators -> lags -> target -> clean table.

    Returns (dataframe, feature_column_names).
    """
    df = add_technical_indicators(raw)
    df = add_lag_features(df, list(lag_columns))
    df = make_classification_target(df, horizon=horizon)

    # Feature columns = everything engineered, excluding raw OHLCV and target
    # columns (raw price/volume levels are non-stationary and leak scale;
    # future_return/target are the label, must not be used as features).
    exclude = {"Open", "High", "Low", "Close", "Volume", "future_return", "target"}
    feature_cols = [c for c in df.columns if c not in exclude]

    # Drop rows with any NaN (from rolling windows at the start, and from the
    # shifted target at the very end of the series). NOTE: `future_return`
    # must be included here explicitly -- (NaN > 0) evaluates to False rather
    # than NaN in pandas, so without this the final `horizon` rows would be
    # silently mislabeled as "down" instead of being dropped.
    clean = df.dropna(subset=feature_cols + ["target", "future_return"]).copy()
    return clean, feature_cols


if __name__ == "__main__":
    from data_loader import fetch_data

    raw = fetch_data("DEMO", start="2020-01-01", end="2024-01-01", source="synthetic")
    table, cols = build_feature_table(raw)
    print(f"Feature table shape: {table.shape}")
    print(f"Number of features: {len(cols)}")
    print(f"Target balance:\n{table['target'].value_counts(normalize=True)}")
