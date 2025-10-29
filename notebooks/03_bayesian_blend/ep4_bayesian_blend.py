import os, math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

pd.options.display.float_format = "{:,.6f}".format

DATA_DIR = "data/raw"
FACTOR_DIR = "results/ep2_factor_engine"
TFT_DIR = "results/ep3_tft_alpha"
OUTDIR = "results/ep4_bayesian_blend"
os.makedirs(OUTDIR, exist_ok=True)

prices = pd.read_csv(os.path.join(DATA_DIR, "prices_sp100.csv"), index_col=0, parse_dates=True)
fwd1 = prices.pct_change().shift(-1)

mom_curve = pd.read_csv(os.path.join(FACTOR_DIR, "MOM_6M_skip1w_curves.csv"), index_col=0)
def_curve = pd.read_csv(os.path.join(FACTOR_DIR, "DEF_vol_1M_curves.csv"), index_col=0)
val_curve = pd.read_csv(os.path.join(FACTOR_DIR, "VAL_proxy_curves.csv"), index_col=0)

preds_path = os.path.join(TFT_DIR, "tft_predictions.csv")
if not os.path.exists(preds_path):
    raise FileNotFoundError(f"Missing {preds_path} — make sure to save TFT out-of-sample predictions.")
tft_preds = pd.read_csv(preds_path, parse_dates=["date"])
tft_preds = tft_preds.pivot(index="date", columns="ticker", values="prediction")

common_dates = prices.index.intersection(tft_preds.index)
prices = prices.loc[common_dates]
tft_preds = tft_preds.loc[common_dates]

def zscore(df):
    return df.sub(df.mean(axis=1))
                  
np.random.seed(42)
tickers = prices.columns[:50]
dates = common_dates
factor_signals = {
    "mom": pd.DataFrame(np.random.randn(len(dates), len(tickers)), index=dates, columns=tickers),
    "def": pd.DataFrame(np.random.randn(len(dates), len(tickers)), index=dates, columns=tickers),
    "val": pd.DataFrame(np.random.randn(len(dates), len(tickers)), index=dates, columns=tickers),
    "tft": tft_preds[tickers]
}

for k, v in factor_signals.items():
    factor_signals[k] = zscore(v)