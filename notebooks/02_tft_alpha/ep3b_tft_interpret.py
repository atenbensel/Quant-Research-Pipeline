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