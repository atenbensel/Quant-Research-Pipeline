import pandas as pd, pathlib as _pl

DATA_DIR = _pl.Path("data")
def ensure_dir(p): _pl.Path(p).mkdir(parents=True, exist_ok=True)
def save_csv(df: pd.DataFrame, path: str): ensure_dir(_pl.Path(path).parent); df.to_csv(path)
