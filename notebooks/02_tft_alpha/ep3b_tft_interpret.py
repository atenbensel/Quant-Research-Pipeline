import os, warnings, io
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import torch
try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)

pl.seed_everything(42)

PRICES_CSV = "data/raw/prices_sp100.csv"
OUTDIR = "results/ep3_tft_alpha"
INTERP_DIR = os.path.join(OUTDIR, "interpret")
os.makedirs(INTERP_DIR, exist_ok=True)

MAX_ENCODER_LENGTH = 126
MAX_PRED_LENGTH = 5
TARGET_NAME = "ret_fwd_5d"

TRAIN_END = "2022-12-30"
VAL_END = "2023-12-29"
TEST_END =  "2024-12-31"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pct_change(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return df.pct_change(periods)

def rolling_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.rolling(window).std()

def momentum(prices: pd.DataFrame, lookback: int = 126, skip: int = 5) -> pd.DataFrame:
    raw = prices.pct_change(lookback)
    if skip > 0:
        raw = raw / (1 + prices.pct_change(skip))
    return raw

def make_forward_sum(ret: pd.DateFrame, horizon: int) -> pd.DataFrame:
    return ret.shift(-horizon + 1).rolling(horizon).sum()

def to_long(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    df = panel.copy()
    df = df.stack().rename(name).to_frame()
    df.index.set_names(["date", "ticker"], inplace=True)
    return df

