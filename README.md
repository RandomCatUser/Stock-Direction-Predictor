# Stock Direction Predictor

A full Python machine-learning pipeline that predicts whether a stock's
price will go **up or down** over the next N trading days, using technical
indicators derived from historical OHLCV (Open/High/Low/Close/Volume) data.

This is an **educational project**. Read the "Honest expectations" section
before treating any output as investment advice — it isn't.

## What's in here

| File | Purpose |
|---|---|
| `data_loader.py` | Fetches real data via `yfinance`, or generates realistic synthetic data for offline use |
| `features.py` | Builds ~35 technical-indicator features (RSI, MACD, Bollinger Bands, moving averages, momentum, volatility, volume pressure) and the up/down target label |
| `model.py` | Chronological train/test split, walk-forward validation, 3 model families (Logistic Regression, Random Forest, Gradient Boosting) |
| `evaluate.py` | Classification metrics, confusion matrix, walk-forward robustness report, and a cost-aware backtest vs. buy-and-hold |
| `main.py` | CLI that runs the entire pipeline end-to-end |
| `docs/stock_prediction_walkthrough.ipynb` | The same pipeline, run step-by-step with full reasoning/explanation at each stage |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Real data (requires internet access to Yahoo Finance)
python main.py --ticker AAPL --start 2015-01-01

# Offline demo with synthetic data (no internet needed)
python main.py --ticker DEMO --source synthetic

# Predict 5 trading days ahead instead of 1, with more validation folds
python main.py --ticker MSFT --horizon 5 --folds 8
```

Or open the notebook for the full guided walkthrough:

```bash
jupyter notebook docs/stock_prediction_walkthrough.ipynb
```

## Key design decisions (the "why")

1. **Classification, not regression.** We predict direction (up/down)
   rather than an exact future price. Predicting exact prices on noisy
   financial data tends to collapse into a model that just repeats
   yesterday's price — great-looking error metrics, zero practical use.

2. **Chronological splits only, never random shuffling.** Shuffling would
   let a model train on the future and test on the past, inflating
   accuracy in a way that is fake and won't survive live trading.

3. **Walk-forward validation.** A single train/test split can be lucky or
   unlucky depending on which market regime it lands in. We roll the split
   forward across multiple folds — bull markets, bear markets, sideways —
   to see if performance holds up.

4. **Naive baselines are always reported.** "51% accuracy" means nothing
   without knowing that guessing the majority class already gets ~50% on a
   balanced target. Every result is shown next to this floor.

5. **Backtest includes trading costs.** A strategy that "wins" on raw
   accuracy can still lose money after realistic transaction costs. The
   backtest here charges a basis-point cost on every position change.

## Honest expectations

Public daily-close technical data is one of the most heavily studied,
close-to-efficiently-priced datasets that exists. Expect directional
accuracy in roughly the **50-55%** range, not 90%+. If a stock prediction
tutorial online claims dramatically higher accuracy, it almost always has
a data leak — very often a random (non-chronological) train/test split, or
a feature that accidentally contains future information.

This project is for learning applied ML and time-series evaluation
discipline. **It is not financial advice**, and no model here should be
used to make real trading decisions without a great deal of additional
rigor (proper risk management, realistic slippage/costs, out-of-sample
testing across many tickers and periods, etc.).


