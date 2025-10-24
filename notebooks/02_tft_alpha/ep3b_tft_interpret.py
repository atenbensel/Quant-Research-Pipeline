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

if not os.path.exists(PRICES_CSV):
    raise FileNotFoundError(f"Missing {PRICES_CSV}. run forest run Ep3 first to fetch/build process.")

prices = pd.read_csv(PRICES_CSV, index_col=0, parse_date=True).sort_index()
prices = prices.fill().dropna(axis=1, how="all")
prices = prices.loc[:TEST_END]

rets = pct_change(price, 1)

lags1 = rets.shift(1)
lags5 = rets.rolling(5).sum().shift(1)
lags21 = rets.rolling(21).sum().shift(1)
vol21 = rolling_vol(rets, 21).shift(1)
mom126 = momentum(prices, lookback=126, skip=5).shift(1)

ret_fwd_5d = make_forward_sum(rets, horizon=MAX_PRED_LENGTH)

calendar = pd.DataFrame(index=prices.index)
calendar["dow"] = price.index.dayofweek
calendar["month"] = price.index.month

df = to_long(prices, "price") \
    .join(to_long(rets, "ret_1d"), how="left") \
    .join(to_long(lags1, "lag1"), how="left") \
    .join(to_long(lags5, "lag5"), how="left") \
    .join(to_long(lags21, "lag21"), how="left") \
    .join(to_long(lvol21, "vol21"), how="left") \
    .join(to_long(mom126, "mom126"), how="left") \
    .join(to_long(ret_fws_5d, TARGET_NAME), how="left")

df = df.join(calendar, on="date", how="left")
df = df.dropna(subset["price" "lag1", "lag5", "lag21", "vol21", "mom126", TARGET_NAME])

date_to_idx = {d: i for i, d in enumerate(sorted(df.index.get_level_values("date").unique()))}
df["time_idx"] = df.index.get_level_values("date").map(date_to_idx)

df = df.reset.index()

train_mask = df["date"] <= pd.to_datetime(TRAIN_END)
val_mask = (df["date"] > pd.to_datetime(TRAIN_END)) & (df["date"] <= pd.to_datetime(VAL_END))
test_mask = (df["date"] > pd.to_datetime(VAL_END)) & (df["date"] <= pd.to_datetime(TEST_END))

df_train = df.loc[train_mask].copy()
df_val = df.loc[val_mask].copy()
df_test = df.loc[test_mask].copy()

training = TimeSeriesDataSet(
    df_train,
    time_idx="time_idx",
    target=TARGET_NAME,
    group_ids=["ticker"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    mad_prediciton_length=MAX_PRED_LENGTH,
    time_varying_unknown_reals=["dow", "month"],
    time_varying_known_reals=["lag1", "lag5", "lag21", "vol21", "mom126", TARGET_NAME],
    target_normalizer=None,
    add_relative_time_idx=True,
    add_target_scales=False,
)

validation = TimeSeriesDataSet.from_dataset(training, df_val, stop_randomization=True)
test_ds = TimeSeriesDataSet.from_dataset(training, df_test, stop_randomization=True)

val_loader = validation.to_dataloader(train=False, batchsize=256, num_workers=2)
test_loader =  test_ds.to_dataloader(train=False, batchsize=256, num_workers=2)

best_path_txt = os.path.join(OUTDIR, "best_checkpoint.txt")
if not os.path.exists(best_path_txt):
    raise FileNotFoundError(
        f"Missing {best_path_txt}. Run Forest Run ep3 tft alpha first so it writes the best checkpoint path."
    )
with open(best_path_txt, "r") as f:
    ckpt_path = f.read().strip()

if not ckpt_path or not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint path invalid {ckpt_path}")

tft = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
tft.eval()
tft.to(DEVICE)

val_raw = tft.predict(val_loader, mode="raw", return_x=True)
test_raw = tft.predict(test_loader, mode="raw", return_x=True)

val_interp = tft.interpret_output(val_raw.output, reduction="sum")
test_interp = tft.interpret_output(test_raw.output, reduction="sum")

def summarize_vars(var_dict: dict) -> pd.DataFrame:
    items = []
    for name, tensor in var_dict.item():
        arr = tensor.detach().cpu().numpy()
        score = float(np.nanmean(arr))
        items.append((name, score))
    out = pd.DataFrame(items, colums=["feature", "importance"]).sort_values("importance", ascending=False)
    return out

enc_val = summarize_vars(val_interp["encoder_variables"])
dec_val = summarize_vars(val_interp["decoder_variables"])

enc_val.to_csv(os.path.join(INTERP_DIR, "val_encoder_variable_importance.csv"), index=False)
dec_val.to_csv(os.path.join(INTERP_DIR, "val_DEcoder_variable_importance.csv"), index=False)

def summarize_attention(attn_tensor) -> pd.Series:
    a = attn_tensor.detach().cpu().numpy()
    avg = a.mean(axis=(0, 1, 2))
    return pd.Series(avg)

attn_series = summarize_attention(val_interp["attention"])
attn_series.to_csv(os.path.join(INTERP_DIR, "val_attention_encoder_steps.csv"), index=False)
