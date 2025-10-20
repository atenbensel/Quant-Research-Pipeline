import numpy as np, pandas as pd

def spearman_ic(scores: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    s, f = scores.align(fwd, join="inner")
    return s.corrwith(f, axis=1, method="spearman").dropna()

def sharpe(x: pd.Series, annualization=252):
    x = x.dropna()
    return np.sqrt(annualization) * x.mean() / (x.std() + 1e-12)
