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
    import pytorch_lightning as pl 
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

pl.seed_everything(42)

PRICES_CSV = "data/raw/prices_sp100.csv"
OUTDIR = "results/ep3_tft_alpha"
os.makedirs(OUTDIR, exist_ok=True)

MAX_ENCODER_LENGTH = 126
MAX_PRED_LENGTH = 5
TARGET_NAME = "ret_fwd_5d"

TRAIN_END = "2022-12-30"
VAL_END = "2023-12-29"
TEST_END = "2024-12-31"

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

def make_forward_sum(ret: pd.DataFrame, horizon: int) -> pd.DataFrame:
    fwd = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    fwd = ret.shift(-horizon+1).rolling(horizon).sum()
    return fwd

def spearman_ic(scores_row: pd.Series, target_row: pd.Series) -> float:
    s = scores_row.dropna()
    t = target_row.reindex(s.index).dropna()
    common = s.index.intersection(t.index)
    if len(common) < 5:
        return np.nan
    return stats.spearmanr(s.loc[common], t.loc[common]).correlation

ensure_file(PRICES_CSV)
prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True).sort_index()
prices = prices.ffill().dropna(axis=1, how="all")
prices = prices.loc[:TEST_END]
print("Prices:", prices.shape, prices.index.min().date(), "→", prices.index.max().date())

rets = pct_change(prices, 1)

feat = pd.DataFrame(index=prices.index)
lags1  = rets.shift(1)
lags5  = rets.rolling(5).sum().shift(1)
lags21 = rets.rolling(21).sum().shift(1)
vol21  = rolling_vol(rets, 21).shift(1)
mom126 = momentum(prices, lookback=126, skip=5).shift(1)

ret_fwd_5d = make_forward_sum(rets, horizon=MAX_PRED_LENGTH)

calendar = pd.DataFrame(index=prices.index)
calendar["dow"] = prices.index.dayofweek
calendar["month"] = prices.index.month

def to_long(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    df = panel.copy()
    df = df.stack().rename(name).to_frame()
    df.index.set_names(["date", "ticker"], inplace=True)
    return df

df = to_long(prices, "price") \
    .join(to_long(rets, "ret_1d"), how="left") \
    .join(to_long(lags1, "lag1"), how="left") \
    .join(to_long(lags5, "lag5"), how="left") \
    .join(to_long(lags21, "lag21"), how="left") \
    .join(to_long(vol21, "vol21"), how="left") \
    .join(to_long(mom126, "mom126"), how="left") \
    .join(to_long(ret_fwd_5d, TARGET_NAME), how="left")

df = df.join(calendar, on="date", how="left")

df = df.dropna(subset=["price", "lag1", "lag5", "lag21", "vol21", "mom126", TARGET_NAME])

date_to_idx = {d: i for i, d in enumerate(sorted(df.index.get_level_values("date").unique()))}
df["time_idx"] = df.index.get_level_values("date").map(date_to_idx)

df = df.reset_index()

train_mask = df["date"] <= pd.to_datetime(TRAIN_END)
val_mask   = (df["date"] > pd.to_datetime(TRAIN_END)) & (df["date"] <= pd.to_datetime(VAL_END))
test_mask  = (df["date"] > pd.to_datetime(VAL_END))   & (df["date"] <= pd.to_datetime(TEST_END))

df_train = df.loc[train_mask].copy()
df_val   = df.loc[val_mask].copy()
df_test  = df.loc[test_mask].copy()

print("Split sizes:", len(df_train), len(df_val), len(df_test))

training = TimeSeriesDataSet(
    df_train,
    time_idx="time_idx",
    target=TARGET_NAME,
    group_ids=["ticker"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PRED_LENGTH,
    time_varying_known_reals=["dow", "month"],
    time_varying_unknown_reals=["lag1", "lag5", "lag21", "vol21", "mom126", TARGET_NAME],
    target_normalizer=None,
    add_relative_time_idx=True,
    add_target_scales=False,
)

validation = TimeSeriesDataSet.from_dataset(training, df_val, stop_randomization=True)
test_ds = TimeSeriesDataSet.from_dataset(training, df_test, stop_randomization=True)

train_loader = training.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=2)
val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=2)
test_loader = test_ds.to_dataloader(train=False,  batch_size=BATCH_SIZE, num_workers=2)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LR,
    hidden_size=HIDDEN_SIZE,
    attention_head_size=ATT_HEADS,
    dropout=DROPOUT,
    hidden_continuous_size=HIDDEN_SIZE,
    loss=QuantileLoss(),
    log_interval=50,
    reduce_on_plateau_patience=4,
)

early_stop = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, mode="min"
)
checkpoint = pl.callbacks.ModelCheckpoint(
    dirpath=OUTDIR, filename="tft-{epoch:02d}-{val_loss:.4f}", save_top_k=1, monitor="val_loss", mode="min"
)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    devices="auto",
    callbacks=[early_stop, checkpoint],
    gradient_clip_val=0.1,
    enable_progress_bar=True,
    log_every_n_steps=50,
    default_root_dir=OUTDIR,
)

trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

best_ckpt = checkpoint.best_model_path
print("Best checkpoint:", best_ckpt)

val_pred = tft.predict(val_loader, return_x=True)
test_pred = tft.predict(test_loader, return_x=True)

def unpack_preds(pred_out) -> Tuple[np.ndarray, pd.DataFrame]:
    yhat, x = pred_out
    preds = yhat.cpu().numpy()
    index = []
    for i in range(len(x["decoder_time_idx"])):
        t_idx = int(x["decoder_time_idx"][i][-1].item())
        ticker = x["group_ids"][i][0]
        index.append((t_idx, ticker))
    pred_df = pd.DataFrame(preds, columns=[f"pred_t+{k+1}" for k in range(preds.shape[1])])
    meta = pd.DataFrame(index, columns=["time_idx","ticker"]).reset_index(drop=True)
    meta["time_idx"] = [t for t, _ in index]
    meta["ticker"] = [tick for _, tick in index]
    out = pd.concat([meta, pred_df], axis=1)
    return preds, out

_, val_pred_df = unpack_preds(val_pred)
_, test_pred_df = unpack_preds(test_pred)

idx_to_date = {v:k for k,v in {d:i for i,d in enumerate(sorted(df["date"].unique()))}.items()}
val_pred_df["date"] = val_pred_df["time_idx"].map(idx_to_date)
test_pred_df["date"] = test_pred_df["time_idx"].map(idx_to_date)

VAL_COL = "pred_t+5" if "pred_t+5" in val_pred_df.columns else val_pred_df.columns[-1]
TEST_COL = "pred_t+5" if "pred_t+5" in test_pred_df.columns else test_pred_df.columns[-1]

real = df.set_index(["time_idx","ticker"])[[TARGET_NAME]].reset_index()
val_eval = val_pred_df.merge(real, on=["time_idx","ticker"], how="left")
test_eval = test_pred_df.merge(real, on=["time_idx","ticker"], how="left")

def point_metrics(eval_df: pd.DataFrame, pred_col: str) -> dict:
    x = eval_df[pred_col].astype(float)
    y = eval_df[TARGET_NAME].astype(float)
    mse = np.mean((x - y) ** 2)
    mae = np.mean(np.abs(x - y))
    return {"MSE": float(mse), "MAE": float(mae)}

val_point = point_metrics(val_eval, VAL_COL)
test_point = point_metrics(test_eval, TEST_COL)

def daily_ic(eval_df: pd.DataFrame, pred_col: str) -> pd.Series:
    out = []
    for d, part in eval_df.groupby("date"):
        ic = spearman_ic(part.set_index("ticker")[pred_col], part.set_index("ticker")[TARGET_NAME])
        out.append((d, ic))
    ic_series = pd.Series({d: v for d, v in out}).sort_index()
    return ic_series

def decile_ls(eval_df: pd.DataFrame, pred_col: str, top=0.1, bottom=0.1) -> pd.Series:
    out = []
    for d, part in eval_df.groupby("date"):
        s = part.set_index("ticker")[pred_col].dropna()
        if s.empty: 
            out.append((d, np.nan)); 
            continue
        n = len(s)
        k = max(1, int(n*top))
        j = max(1, int(n*bottom))
        long = s.nlargest(k).index
        short = s.nsmallest(j).index
        realized = part.set_index("ticker")[TARGET_NAME]
        r = realized.reindex(long).mean() - realized.reindex(short).mean()
        out.append((d, r))
    return pd.Series({d:v for d,v in out}).sort_index()

val_ic = daily_ic(val_eval, VAL_COL)
test_ic = daily_ic(test_eval, TEST_COL)

val_ls = decile_ls(val_eval, VAL_COL)
test_ls = decile_ls(test_eval, TEST_COL)

ann = math.sqrt(252/5)
val_sr = ann * val_ls.mean() / (val_ls.std() + 1e-12)
test_sr = ann * test_ls.mean() / (test_ls.std() + 1e-12)

summary = pd.DataFrame([
    {"split": "val",  "MSE": val_point["MSE"],  "MAE": val_point["MAE"],  "IC_mean": float(np.nanmean(val_ic)),  "LS_SR": float(val_sr)},
    {"split": "test", "MSE": test_point["MSE"], "MAE": test_point["MAE"], "IC_mean": float(np.nanmean(test_ic)), "LS_SR": float(test_sr)},
])
print("\nSummary:\n", summary)

summary.to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)
val_eval.to_csv(os.path.join(OUTDIR, "val_predictions.csv"), index=False)
test_eval.to_csv(os.path.join(OUTDIR, "test_predictions.csv"), index=False)
with open(os.path.join(OUTDIR, "best_checkpoint.txt"), "w") as f:
    f.write(best_ckpt or "")

val_ls.cumsum().to_csv(os.path.join(OUTDIR, "val_ls_curve.csv"))
test_ls.cumsum().to_csv(os.path.join(OUTDIR, "test_ls_curve.csv"))

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes = axes.ravel()
axes[0].plot(val_ic.rolling(20).mean()); axes[0].axhline(0, color="k", lw=1); axes[0].set_title("VAL Rolling IC (20)")
axes[1].plot(test_ic.rolling(20).mean()); axes[1].axhline(0, color="k", lw=1); axes[1].set_title("TEST Rolling IC (20)")
axes[2].plot(val_ls.cumsum()); axes[2].axhline(0, color="k", lw=1); axes[2].set_title("VAL L/S Cumulative (5D target)")
axes[3].plot(test_ls.cumsum()); axes[3].axhline(0, color="k", lw=1); axes[3].set_title("TEST L/S Cumulative (5D target)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "tft_eval_plots.png"), dpi=150)
plt.close()

print(f"\nSaved artifacts to: {OUTDIR}")
print("Done")