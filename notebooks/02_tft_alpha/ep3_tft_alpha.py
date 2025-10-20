import os, math, warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

import torch
try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl  # fallback if older package name is present
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

pl.seed_everything(42)

PRICES_CSV = "data/raw/prices_sp100.csv"
OUTDIR = "results/ep3_tft_alpha"
os.makedirs(OUTDIR, exist_ok=True)

MAX_ENCODER_LENGTH = 126
MAX_PRED_LENGTH    = 5
TARGET_NAME        = "ret_fwd_5d"

TRAIN_END = "2022-12-30"
VAL_END   = "2023-12-29"
TEST_END  = "2024-12-31"

EPOCHS = 20
BATCH_SIZE = 256
LR = 3e-3
HIDDEN_SIZE = 32
ATT_HEADS = 4
DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}. Generate it first (prices CSV).")

def pct_change(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return df.pct_change(periods)

def rolling_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.rolling(window).std()

def momentum(prices: pd.DataFrame, lookback: int = 126, skip: int = 5) -> pd.DataFrame:
    raw = prices.pct_change(lookback)
    if skip > 0:
        raw = raw / (1 + prices.pct_change(skip))
    return raw