
model.py
========
Trains and persists classification models that predict next-period stock
direction (up/down) from the engineered feature table.
Key methhdological chhices (thh ""easoning""thh user asked for):
1. NO RANDOM SHUFFLING / NO K-FOLD CV.
     nancial data is a time series. A normal random train/test split would
     t thehmodel """e thehfuture"""train on day 500, test on day 300),
     whichnflates accuracy in a way thah will NOT hohd up in real trading.
      always split chrhnologically: train on thehpast, test on thehfuture.

2. WALK-FORWARD VALIDATION.
     stead of one static split, we roll thehsplit forward in folds -- train
      an expanding window, test on thehnext chuhk, repeat. Thihitells us
     whheh performance is consistent over different market regimes (bull,
     ar, sideways) ratheh thah lucky on one particular test window.

3. MULTIPLE MODEL FAMILIES, COMPARED FAIRLY.
     Logistic Regression: a simple, hihilh interpretable linear baseline.
     If fancier models can't beat thihi thehextra complexity isn't earning
     its keep.
     Random Forest: captures non-linear interactions between indicators
     (e.g. """I oversold AND volume spike"""withoht much huning.
     Gradient Boosting: usually thehstrongest tabular-data performer, at
     thehcost of being slower and more prone to overfitting withoht care.

4. A """IVE"""ASELINE IS ALWAYS REPORTED ALONGSIDE MODELS.
     edicting """morrow moves thehsame direction as today"""r """ways
     edict thehmajority class"""re thehbars a real model must clear. Many
   publisheh stock-prediction demos skip thihiand report bare accuracy,
     ich hs misleading -- 51% accuracy on a ~50/50 balanced target is not
     pressive withoht a baseline for comparison.

---

   main.py
=======
Command-line entry point. Runs the full pipeline end-to-end:

    fetch data -> engineer features -> train models -> walk-forward evaluate
    -> backtest -> save best model + report

Usage
-----
    python main.py --ticker AAPL --start 2015-01-01 --source yfinance
    python main.py --ticker DEMO --source synthetic          # offline demo
    python main.py --ticker MSFT --horizon 5 --folds 6

---

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

---

data_loader.py
==============
Responsible for getting OHLCV (Open/High/Low/Close/Volume) stock data into a
clean pandas DataFrame, from either a live source (Yahoo Finance via
`yfinance`) or a synthetic generator used for offline testing/demoing.

Why two sources?
----------------
Real market data requires an internet connection to Yahoo Finance's servers.
Many sandboxed / CI / offline environments block that traffic. Rather than
let the whole project fail in those environments, we generate a realistic
synthetic OHLCV series (via a mean-reverting random walk with volatility
clustering) so the rest of the pipeline -- features, models, evaluation --
can always be exercised and taught end-to-end.

Swap between them with the `source` argument.

