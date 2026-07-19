

from __future__ import annotations

import numpy as np
import pandas as pd


def fetch_data(
    ticker: str = "AAPL",
    start: str = "2015-01-01",
    end: str | None = None,
    source: str = "yfinance",
    seed: int = 42,
) -> pd.DataFrame:
    """Fetch historical daily OHLCV data for a ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. "AAPL", "MSFT", "SPY".
    start, end : str
        Date range in "YYYY-MM-DD" format. `end=None` means "today".
    source : {"yfinance", "synthetic"}
        "yfinance" pulls real data from Yahoo Finance (requires internet).
        "synthetic" generates a realistic fake series for offline use.
    seed : int
        RNG seed for synthetic data, for reproducibility.

    Returns
    -------
    pd.DataFrame indexed by Date with columns: Open, High, Low, Close, Volume
    """
    if source == "yfinance":
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError(
                f"No data returned for {ticker}. Check your internet connection, "
                "the ticker symbol, or try source='synthetic' for an offline demo."
            )
        # yfinance sometimes returns MultiIndex columns for a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        return df

    if source == "synthetic":
        return _generate_synthetic_ohlcv(start=start, end=end, seed=seed, ticker=ticker)

    raise ValueError(f"Unknown source '{source}'. Use 'yfinance' or 'synthetic'.")


def _generate_synthetic_ohlcv(
    start: str, end: str | None, seed: int, ticker: str
) -> pd.DataFrame:
    """Generate a plausible daily OHLCV series.

    Model: log-price follows a random walk with drift, where volatility
    itself follows a simple GARCH-like clustering process (today's volatility
    depends partly on yesterday's), which is a much closer approximation to
    real markets than plain Gaussian noise (real markets show "volatility
    clustering" -- calm periods and turbulent periods, not uniform noise).
    """
    rng = np.random.default_rng(seed + abs(hash(ticker)) % 1000)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    dates = pd.bdate_range(start=start, end=end)  # business days only
    n = len(dates)

    # --- volatility clustering (simple GARCH(1,1)-style process) ---
    omega, alpha, beta = 1e-6, 0.08, 0.90
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    shocks = rng.standard_normal(n)
    returns = np.zeros(n)
    drift = 0.0003  # slight long-run upward drift, like broad equity markets
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = drift + np.sqrt(sigma2[t]) * shocks[t]

    log_price = np.cumsum(returns) + np.log(100)  # start around $100
    close = np.exp(log_price)

    # Build Open/High/Low around Close with a small intraday range
    intraday_noise = rng.uniform(0.002, 0.02, size=n)
    open_ = close * (1 + rng.normal(0, 0.004, size=n))
    high = np.maximum(open_, close) * (1 + intraday_noise)
    low = np.minimum(open_, close) * (1 - intraday_noise)
    volume = rng.integers(2_000_000, 20_000_000, size=n).astype(float)
    # Volume tends to spike with volatility -- bake that relationship in
    volume *= 1 + 3 * (sigma2 / sigma2.max())

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df.round(2)


if __name__ == "__main__":
    # Quick manual smoke test
    df = fetch_data("DEMO", start="2020-01-01", end="2024-01-01", source="synthetic")
    print(df.head())
    print(df.tail())
    print(f"\n{len(df)} rows generated.")
