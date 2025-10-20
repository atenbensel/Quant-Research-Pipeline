import os, math
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

pd.options.display.float_format = "{:,.6f}".format

PRICES_CSV = "data/raw/prices_sp100.csv"
FF5_CSV = "data/raw/F-F_Research_Data_5_Factors_2x3.csv"
MOM_CSV = "data/raw/F-F_Momentum_Factor.csv"

START = "2015-01-01"
END = "2025-01-01"

COST_BPS = 5
TRAIN_YEARS = 4
TEST_MONTHS = 6
EMBARGO_DAYS = 5

OUTDIR = "results/ep2_factor_engine"
os.makedirs(OUTDIR, exist_ok=True)

def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing File: {path}\n"
        )
    
require_file(PRICES_CSV)
require_file(FF5_CSV)
require_file(MOM_CSV)

prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True).sort.index()
prices = prices.loc[(prices.index >= START) & (prices.index <= END)]

def _parse_ff_monthly(csv_path: str, date_col: str = None) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    start_idx = None
    for i, v in enumerate(raw.iloc[:, 0].astype(str).tolist()):
        if v.isdigit() and len(v) == 6:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"Could not locate monthly data block in {csv_path}")
    df = pd.read_csv(csv_path, skiprows=start_idx, nrows=None)
    ym = df.iloc[:, 0].astype(str)
    df = df[pd.to_numeric(ym, errors="coerce").notna()]
    ym = df.iloc[:, 0].astype(str)
    df = df.set_index(ym)
    df = df.apply(pd.to_numeric, errors="coerce") / 100.00
    idx = pd.PeriodIndex(df.index, freq="M").to_timestamp(how="end")
    df.index = idxdf = df.sort_index()
    return df

