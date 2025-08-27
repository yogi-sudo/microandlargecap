import os, time, math
from typing import List, Dict
import requests
import pandas as pd

EOD_KEY = os.getenv("EODHD_API_KEY", "")
CAP_TTL = int(os.getenv("CAP_TTL", "86400"))  # cache age for caps (s)
CAPS_CACHE = "out/eodhd_caps_cache.csv"

LARGE_MIN_B = float(os.getenv("LARGE_MIN_B", "15"))  # >= 15B = Large
MID_MIN_B   = float(os.getenv("MID_MIN_B", "5"))     # >= 5B  = Mid, else Micro

os.makedirs("out", exist_ok=True)

def _read_cache() -> pd.DataFrame:
    try:
        df = pd.read_csv(CAPS_CACHE)
        if "ts" in df.columns and (time.time() - df["ts"].max()) <= CAP_TTL:
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=["Ticker","MktCapB","ts"])

def _write_cache(df: pd.DataFrame) -> None:
    try:
        df.to_csv(CAPS_CACHE, index=False)
    except Exception:
        pass

def _fetch_cap_one(ticker: str) -> float:
    # EODHD fundamentals endpoint – MarketCapitalization in USD
    # Ticker should be like BHP.AX
    if not EOD_KEY:
        return float("nan")
    url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={EOD_KEY}&fmt=json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        js = r.json() or {}
        hi = js.get("Highlights") or {}
        cap = hi.get("MarketCapitalization", None)
        if cap is None:
            return float("nan")
        return float(cap) / 1e9  # USD billions
    except Exception:
        return float("nan")

def get_caps(tickers: List[str]) -> pd.DataFrame:
    cache = _read_cache()
    need = []
    out_rows = []
    now = time.time()
    cache_map: Dict[str, float] = {str(k): v for k, v in zip(cache.get("Ticker", []), cache.get("MktCapB", []))}
    for t in tickers:
        if t in cache_map and not math.isnan(cache_map[t]):
            out_rows.append({"Ticker": t, "MktCapB": cache_map[t], "ts": now})
        else:
            need.append(t)
    for t in need:
        capb = _fetch_cap_one(t)
        out_rows.append({"Ticker": t, "MktCapB": capb, "ts": now})
        time.sleep(0.25)  # be gentle
    df = pd.DataFrame(out_rows)
    # Keep best/last values per ticker
    df = df.sort_values("ts").drop_duplicates("Ticker", keep="last")
    _write_cache(df)
    return df

def tag_tiers(caps_df: pd.DataFrame) -> pd.DataFrame:
    df = caps_df.copy()
    def _tier(x):
        if pd.isna(x): return "unknown"
        if x >= LARGE_MIN_B: return "large"
        if x >= MID_MIN_B:   return "mid"
        return "micro"
    df["Tier"] = df["MktCapB"].apply(_tier)
    return df[["Ticker","MktCapB","Tier"]]
