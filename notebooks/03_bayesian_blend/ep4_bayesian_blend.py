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