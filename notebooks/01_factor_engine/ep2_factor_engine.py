import os, math
from dataclasses import dataclass
from typing import List, Dict
import io, re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

pd.options.display.float_format = "{:,.6f}".format

PRICES_CSV = "data/raw/prices_sp100.csv"
FF5_CSV    = "data/raw/F-F_Research_Data_5_Factors_2x3.CSV"
MOM_CSV    = "data/raw/F-F_Momentum_Factor.CSV"

START = "2015-01-01"
END   = "2025-01-01"

COST_BPS      = 5
TRAIN_YEARS   = 4
TEST_MONTHS   = 6
EMBARGO_DAYS  = 5

OUTDIR = "results/ep2_factor_engine"
os.makedirs(OUTDIR, exist_ok=True)

def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Make sure you ran the earlier download commands to create the local CSVs."
        )

require_file(PRICES_CSV)
require_file(FF5_CSV)
require_file(MOM_CSV)

prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True).sort_index()
prices = prices.loc[(prices.index >= START) & (prices.index <= END)]
print("Prices:", prices.shape, prices.index.min().date(), "→", prices.index.max().date())

def _parse_ff_monthly(csv_path: str) -> pd.DataFrame:
    with open(csv_path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        first = line.strip().split(",")[0].strip()
        if re.fullmatch(r"\d{6}", first):
            start = i
            break
    if start is None:
        raise ValueError(f"Could not locate monthly data block in {csv_path}")

    end = None
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if stripped == "" or stripped.lower().startswith("annual"):
            end = j
            break

    data_str = "".join(lines[start:end]) if end else "".join(lines[start:])

    df = pd.read_csv(io.StringIO(data_str))
    ym = df.iloc[:, 0].astype(str).str.strip()
    df = df.set_index(ym)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    idx = pd.PeriodIndex(df.index, freq="M").to_timestamp(how="end")
    df.index = idx
    df = df.sort_index()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df

ff5m = _parse_ff_monthly(FF5_CSV)
mom_m = _parse_ff_monthly(MOM_CSV)

if len(mom_m.columns) == 1:
    mom_m.columns = ["MOM"]

ff_monthly = ff5m.join(mom_m, how="outer").sort_index()
ff_daily = ff_monthly.reindex(pd.date_range(prices.index.min(), prices.index.max(), freq="B")).ffill()

ff_cols = [c for c in ff5m.columns if c.strip() != ""]
ff5m = ff5m[ff_cols]
if len(mom_m.columns) == 1:
    mom_m.columns = ["MOM"]

ff_monthly = ff5m.join(mom_m, how="outer").sort_index()

ff_daily = ff_monthly.reindex(pd.date_range(prices.index.min(), prices.index.max(), freq="B")).ffill()
print("FF monthly → daily shape:", ff_daily.shape)

def compute_returns(prices: pd.DataFrame, lookahead: int = 1):
    rets = prices.pct_change()
    fwd  = rets.shift(-lookahead)
    return rets, fwd

rets, fwd1 = compute_returns(prices, lookahead=1)

def xsec_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional (per date) z-score of ranks for scale-invariant long/short."""
    ranks = df.rank(axis=1, method="average")
    z = ranks.apply(lambda row: pd.Series(stats.zscore(row, nan_policy="omit"), index=row.index), axis=1)
    return z

def factor_momentum(prices: pd.DataFrame, lookback=126, skip=5) -> pd.DataFrame:
    """
    6M momentum with a ~1-week skip to reduce short-term reversal.
    score_t = r(t-126→t) / (1 + r(t-5→t))
    """
    raw = prices.pct_change(lookback)
    if skip > 0:
        raw = raw / (1.0 + prices.pct_change(skip))
    return xsec_rank(raw)

def factor_defensive_vol(prices: pd.DataFrame, window=21) -> pd.DataFrame:
    """Low recent volatility ranks higher (defensive tilt)."""
    vol = prices.pct_change().rolling(window).std()
    return xsec_rank(-vol)

def factor_value_proxy(prices: pd.DataFrame, window=252) -> pd.DataFrame:
    """
    Proxy for 'cheapness' without fundamentals:
    score_t = (1/Price_t) / sigma_{t-window}(returns)
    """
    vol = prices.pct_change().rolling(window).std()
    inv_px = (1.0 / prices).replace([np.inf, -np.inf], np.nan)
    proxy = inv_px / (vol + 1e-8)
    return xsec_rank(proxy)

mom = factor_momentum(prices, lookback=126, skip=5)
defv = factor_defensive_vol(prices, window=21)
valp = factor_value_proxy(prices, window=252)

@dataclass
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

def rolling_splits(
    dates: pd.DatetimeIndex,
    train_years: int = TRAIN_YEARS,
    test_months: int = TEST_MONTHS,
    embargo_days: int = EMBARGO_DAYS,
) -> List[Split]:
    dates = pd.DatetimeIndex(dates)
    start = dates.min()
    end   = dates.max()
    cursor = start + pd.DateOffset(years=train_years)
    splits: List[Split] = []
    while cursor + pd.DateOffset(months=test_months) < end:
        tr_start = start
        tr_end   = cursor - pd.Timedelta(days=embargo_days)
        te_start = cursor + pd.Timedelta(days=embargo_days)
        te_end   = cursor + pd.DateOffset(months=test_months)
        splits.append(Split(tr_start, tr_end, te_start, te_end))
        cursor = cursor + pd.DateOffset(months=test_months)
    return splits

splits = rolling_splits(prices.index, TRAIN_YEARS, TEST_MONTHS, EMBARGO_DAYS)
print("Num splits:", len(splits))
for s in splits[:2]:
    print(s)

def spearman_ic(scores: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Per-date cross-sectional rank correlation between factor scores and next-day returns."""
    s, f = scores.align(fwd, join="inner")
    return s.corrwith(f, axis=1, method="spearman").dropna()

def decile_long_short(scores: pd.DataFrame, fwd: pd.DataFrame, top=0.1, bottom=0.1) -> pd.Series:
    """Equal-weight long top-decile, short bottom-decile by score each day."""
    s, fr = scores.align(fwd, join="inner")
    out = []
    for date in s.index:
        row = s.loc[date].dropna()
        if row.empty:
            out.append(np.nan); continue
        n = len(row)
        k = max(1, int(n * top))
        j = max(1, int(n * bottom))
        long  = row.nlargest(k).index
        short = row.nsmallest(j).index
        r = fr.loc[date, long].mean() - fr.loc[date, short].mean()
        out.append(r)
    return pd.Series(out, index=s.index, name="ls").dropna()

def apply_costs(ret: pd.Series, turnover=0.5, cost_bps=COST_BPS) -> pd.Series:
    """
    Simple cost model: expected per-period cost = turnover * cost_bps.
    Here we assume 50% turnover as a placeholder; refine in execution episodes.
    """
    return ret - (turnover * cost_bps / 1e4)

def run_eval(name: str, scores: pd.DataFrame, fwd: pd.DataFrame) -> Dict:
    ic_all, ls_all = [], []
    for s in splits:
        te = (scores.index >= s.test_start) & (scores.index <= s.test_end)
        sc_te = scores.loc[te]
        fwd_te = fwd.loc[te]
        ic = spearman_ic(sc_te, fwd_te)
        ls = decile_long_short(sc_te, fwd_te)
        ic_all.append(ic); ls_all.append(ls)
    ic_series = pd.concat(ic_all).sort_index()
    ls_series = pd.concat(ls_all).sort_index()
    gross = ls_series
    net   = apply_costs(gross, turnover=0.5, cost_bps=COST_BPS)
    ann = math.sqrt(252)
    return {
        "name": name,
        "IC_mean": ic_series.mean(),
        "IC_IR": ic_series.mean() / (ic_series.std() + 1e-12),
        "LS_gross_SR": ann * gross.mean() / (gross.std() + 1e-12),
        "LS_net_SR":   ann * net.mean()   / (net.std() + 1e-12),
        "gross_curve": gross.cumsum(),
        "net_curve":   net.cumsum(),
        "ic_series":   ic_series,
    }

results = [
    run_eval("MOM_6M_skip1w", mom, fwd1),
    run_eval("DEF_vol_1M",    defv, fwd1),
    run_eval("VAL_proxy",     valp, fwd1),
]

summary = pd.DataFrame([{k:v for k,v in r.items() if k not in ["gross_curve","net_curve","ic_series"]} for r in results])
print("Summary:")

if any(c.lower().strip().startswith("mkt") for c in ff_daily.columns):
    mkt_col = [c for c in ff_daily.columns if c.lower().strip().startswith("mkt")][0]
    ff_daily["MKT_roll_vol_63"] = ff_daily[mkt_col].rolling(63).std()

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.ravel()
for i, r in enumerate(results):
    axes[2*i].plot(r["gross_curve"], label=f'{r["name"]} gross')
    axes[2*i].plot(r["net_curve"],   label=f'{r["name"]} net', alpha=0.85)
    axes[2*i].legend()
    axes[2*i].set_title(f'{r["name"]} L/S cumulative (daily)')
    axes[2*i].axhline(0, color="k", lw=0.8)

    ic_roll = r["ic_series"].rolling(63).mean()
    axes[2*i+1].plot(ic_roll)
    axes[2*i+1].axhline(0, color="k", lw=0.8)
    axes[2*i+1].set_title(f'{r["name"]} Rolling IC (63d)')

plt.tight_layout()
plt.show()

summary_path = os.path.join(OUTDIR, "summary.csv")
summary.to_csv(summary_path, index=False)
for r in results:
    pd.DataFrame({"gross": r["gross_curve"], "net": r["net_curve"]}).to_csv(
        os.path.join(OUTDIR, f"{r['name']}_curves.csv")
    )
print("Saved:", summary_path)