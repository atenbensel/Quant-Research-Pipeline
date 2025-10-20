import pandas as pd, numpy as np
from scipy import stats

def xsec_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, method="average").apply(stats.zscore, axis=1)

def momentum(prices: pd.DataFrame, lookback=126, skip=5):
    mom = prices.pct_change(lookback)
    if skip > 0:
        mom = mom / (1 + prices.pct_change(skip))
    return xsec_rank(mom)

def defensive_vol(prices: pd.DataFrame, window=21):
    vol = prices.pct_change().rolling(window).std()
    return xsec_rank(-vol)
