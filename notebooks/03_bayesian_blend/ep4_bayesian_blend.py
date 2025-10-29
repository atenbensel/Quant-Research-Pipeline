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