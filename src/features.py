import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    p = df.copy()
    for c in ["open","high","low","close","volume"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=["close","volume"])

    p["ret1"]  = p["close"].pct_change(1)
    p["ret5"]  = p["close"].pct_change(5)
    p["ma5"]   = p["close"].rolling(5).mean()
    p["ma10"]  = p["close"].rolling(10).mean()
    p["ma20"]  = p["close"].rolling(20).mean()
    p["std20"] = p["close"].rolling(20).std()
    p["vol20"] = p["close"].pct_change().rolling(20).std()

    delta = p["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up, roll_down = up.rolling(14).mean(), down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    p["rsi14"] = 100 - 100/(1+rs)

    p["next_close"] = p["close"].shift(-1)
    p["y"] = (p["next_close"] > p["close"]).astype(int)
    p["v20"] = p["volume"].rolling(20).mean()

    return p.dropna().reset_index(drop=True)
