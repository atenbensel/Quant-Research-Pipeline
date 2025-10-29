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

ROLL_WINDOW = 63

def bayesian_weights(resid_dict):
    vars_ = {k: v.rolling(ROLL_WINDOW).var().mean() for k, v in resid_dict.items()}
    precisions = {k: 1 / (v + 1e-8) for k, v in vars_.items()}
    total = sum(precisions.values())
    weights = {k: v / total for k, v in precisions.items()}
    return weights

rets = prices.pct_change().shift(-1)
aligned = {k: v.align(rets, join="inner")[0] for k, v in factor_signals.items()}
resids = {k: rets - aligned[k] for k in factor_signals.keys()}

weights = bayesian_weights(resids)
print("Bayesian weights:", weights)

blend = sum(weights[k] * factor_signals[k] for k in factor_signals.keys())

def spearman_ic(scores: pd.DataFrame, fwd: pd.DataFrame):
    s, f = scores.align(fwd, join="inner")
    return s.corrwith(f, axis=1, method="spearman").dropna()

def decile_long_short(scores: pd.DataFrame, fwd: pd.DataFrame, top=0.1, bottom=0.1):
    s, fr = scores.align(fwd, join="inner")
    out = []
    for date in s.index:
        row = s.loc[date].dropna()
        if row.empty:
            out.append(np.nan)
            continue
        n = len(row)
        k = max(1, int(n * top))
        j = max(1, int(n * bottom))
        long = row.nlargest(k).index
        short = row.nsmallest(j).index
        r = fr.loc[date, long].mean() - fr.loc[date, short].mean()
        out.append(r)
    return pd.Series(out, index=s.index, name="ls").dropna()

blend_ic = spearman_ic(blend, rets)
blend_ls = decile_long_short(blend, rets)

ann = math.sqrt(252)
sr = ann * blend_ls.mean() / (blend_ls.std() + 1e-9)

summary = pd.DataFrame({
    "Metric": ["Mean IC", "IC IR", "LS Sharpe"],
    "Value": [blend_ic.mean(), blend_ic.mean()/blend_ic.std(), sr]
})
summary.to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)

print(summary)

plt.figure(figsize=(10, 5))
plt.plot(blend_ls.cumsum(), label="Blended LS Cumulative Return")
plt.axhline(0, color="k", lw=0.8)
plt.title("Bayesian Alpha Blend Performance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "blend_cumulative.png"), dpi=150)
plt.close()

blend_ic.rolling(63).mean().plot(figsize=(10, 3), title="Rolling 3M Information Coefficient")
plt.axhline(0, color="k", lw=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "blend_ic_rolling.png"), dpi=150)
plt.close()

print("Saved outputs to:", OUTDIR)