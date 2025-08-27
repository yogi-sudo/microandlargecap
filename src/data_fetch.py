#!/usr/bin/env python3
import os
import datetime as dt
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf

# Folders (match your project layout)
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
OUT_DIR   = os.getenv("OUT_DIR", "out")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Defaults (overridable via env OR function args)
DEFAULT_YEARS    = int(os.getenv("TRAIN_YEARS", 3))
DEFAULT_MIN_ROWS = int(os.getenv("MIN_ROWS", 150))
DEFAULT_MIN_PX   = float(os.getenv("MIN_PRICE", 0.2))
DEFAULT_MIN_VOL  = float(os.getenv("W_MIN_VOL", 10_000))  # avg shares/day
DEFAULT_UNI_MAX  = int(os.getenv("UNIVERSE_MAX", 0))      # 0 = no cap

TODAY = dt.date.today()

def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_ohlc.csv")

def fetch_prices(ticker: str, years: int = DEFAULT_YEARS) -> pd.DataFrame:
    """
    Returns OHLCV with columns: date, open, high, low, close, volume (all numeric), ascending by date.
    Uses cache if present, otherwise downloads via yfinance and writes cache.
    """
    fn = _cache_path(ticker)
    # Try cache first
    if os.path.exists(fn):
        try:
            df = pd.read_csv(fn, parse_dates=["date"])
            need = {"date", "open", "high", "low", "close", "volume"}
            if not df.empty and need.issubset(df.columns):
                # ensure numerics
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["date", "close", "volume"])
                return df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    # Download fresh
    start = TODAY - dt.timedelta(days=365 * years + 7)
    end   = TODAY + dt.timedelta(days=1)
    y = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
    if y is None or y.empty:
        return pd.DataFrame()

    y = y.reset_index().rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low":  "low",
        "Close":"close",
        "Volume":"volume"
    })
    keep = ["date", "open", "high", "low", "close", "volume"]
    for c in keep:
        if c != "date":
            y[c] = pd.to_numeric(y[c], errors="coerce")
    y = y.dropna(subset=["date", "close", "volume"])[keep].copy()
    try:
        y.to_csv(fn, index=False)
    except Exception:
        # best effort cache write
        pass
    return y.sort_values("date").reset_index(drop=True)

def build_dataset(
    tickers: List[str],
    years: int = None,
    min_rows: int = None,
    min_price: float = None,
    min_vol: float = None,
    universe_max: int = None,
) -> pd.DataFrame:
    """
    Build a stacked OHLCV dataframe for a list of tickers, with basic filtering.
    Accepts keyword arguments (so main.py can call build_dataset(tickers=...)).

    Returns columns: ticker, date, open, high, low, close, volume
    """
    years       = DEFAULT_YEARS    if years is None else years
    min_rows    = DEFAULT_MIN_ROWS if min_rows is None else min_rows
    min_price   = DEFAULT_MIN_PX   if min_price is None else min_price
    min_vol     = DEFAULT_MIN_VOL  if min_vol is None else min_vol
    universe_max= DEFAULT_UNI_MAX  if universe_max is None else universe_max

    if universe_max and universe_max > 0:
        tickers = tickers[:universe_max]

    frames = []
    for i, t in enumerate(tickers, 1):
        px = fetch_prices(t, years=years)
        print(f"\rDownload: {i}/{len(tickers)}", end="", flush=True)
        if px.empty or len(px) < min_rows:
            continue
        # Filters
        last_close = float(px["close"].iloc[-1])
        if last_close < min_price:
            continue
        v20 = px["volume"].rolling(20).mean().iloc[-1]
        if pd.isna(v20) or v20 < min_vol:
            continue

        px = px.copy()
        px["ticker"] = t
        frames.append(px)

    print()
    if not frames:
        raise SystemExit("No usable histories after filters. Lower MIN_ROWS/W_MIN_VOL or widen universe.")
    data = pd.concat(frames, ignore_index=True)
    # Normalize dtypes
    data["date"] = pd.to_datetime(data["date"])
    for c in ["open","high","low","close","volume"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=["date","close","volume"])
    return data.sort_values(["ticker","date"]).reset_index(drop=True)

# ======================  ASX symbols + market caps (EODHD)  ======================
import os, json, time, math, datetime as _dt
from typing import List, Dict
import pandas as _pd
import requests as _req
from concurrent.futures import ThreadPoolExecutor, as_completed

_EOD_API = os.getenv("EODHD_API_KEY", "")
_DATA_DIR = "data"
_UNIVERSE_CSV = os.path.join(_DATA_DIR, "nextday_universe.csv")
_CAPS_CSV     = os.path.join(_DATA_DIR, "universe_caps.csv")
_CAPS_CACHE   = os.path.join(_DATA_DIR, "fundamentals_cache.json")
_ASX_ALL      = os.path.join(_DATA_DIR, "asx_symbols_full.csv")

def _safe_upper(s): 
    try: return str(s).upper()
    except: return str(s)

def _ensure_dirs():
    os.makedirs(_DATA_DIR, exist_ok=True)

def _eod_get(url, params=None, timeout=30):
    if not params: params = {}
    params["api_token"] = _EOD_API
    params["fmt"] = "json"
    r = _req.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        # soft-fail: return None, caller decides
        return None
    try:
        return r.json()
    except Exception:
        return None

def fetch_asx_symbol_list(limit:int=20000) -> _pd.DataFrame:
    """
    Pulls ALL AU symbols from EODHD and saves to data/asx_symbols_full.csv.
    Filters to common stock / ETF / REIT / fund.
    """
    _ensure_dirs()
    if not _EOD_API:
        raise RuntimeError("EODHD_API_KEY not set in env")

    url = "https://eodhd.com/api/exchange-symbol-list/AU"
    js = _eod_get(url)
    if not js:
        raise RuntimeError("Failed to fetch AU symbol list (EODHD)")

    df = _pd.DataFrame(js)
    if df.empty: 
        raise RuntimeError("EODHD symbol list returned empty")

    # normalize
    df["ticker"] = df["Code"].astype(str).str.upper()
    df["name"]   = df.get("Name","").astype(str)
    typ = df.get("Type","").astype(str).str.lower()
    ok_types = {"common stock","etf","fund","reit","cefs"}  # CEFs rare on AU; harmless
    mask_ok = typ.isin(ok_types)
    df_ok = df[mask_ok].copy()

    # strip suffixes like ".AU" if present (EODHD’s Code usually already sans suffix)
    df_ok["ticker"] = (df_ok["ticker"]
                        .str.replace(r"\.ASX$","",regex=True)
                        .str.replace(r"\.AX$","",regex=True)
                        .str.replace(r"\.AU$","",regex=True))
    df_ok = df_ok.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    df_ok.to_csv(_ASX_ALL, index=False)
    return df_ok

def merge_universe_with_isin(isin_path:str, base: _pd.DataFrame) -> _pd.DataFrame:
    """
    Optional: union with your ISIN.xlsx/.xls/.csv; we only extract Ticker/Code-like columns.
    """
    if not isin_path or not os.path.exists(isin_path):
        return base
    ext = os.path.splitext(isin_path)[1].lower()
    if ext in (".xls",".xlsx"):
        try:
            extra = _pd.read_excel(isin_path)
        except Exception as e:
            print("[WARN] ISIN read failed:", e)
            return base
    else:
        extra = _pd.read_csv(isin_path)

    # Find probable ticker columns
    candidates = [c for c in extra.columns if str(c).strip().lower() in ("ticker","code","symbol")]
    if not candidates:
        # heuristic: any column with short uppercase tokens
        for c in extra.columns:
            s = extra[c].astype(str)
            if (s.str.len()<=5).mean() > 0.5:
                candidates = [c]; break
    if not candidates:
        return base

    out = base.copy()
    tick = (extra[candidates[0]].astype(str).str.upper()
              .str.replace(r"\.ASX$","",regex=True)
              .str.replace(r"\.AX$","",regex=True)
              .str.replace(r"\.AU$","",regex=True))
    add = _pd.DataFrame({"ticker": tick})
    out = _pd.concat([out[["ticker"]], add[["ticker"]]], ignore_index=True)
    out = out[out["ticker"].str.len().between(1, 10)].drop_duplicates()
    return out.reset_index(drop=True)

def build_nextday_universe(isin_path:str=None) -> _pd.DataFrame:
    """
    Writes data/nextday_universe.csv with ASX tradeable equities + optional ISIN merge.
    """
    _ensure_dirs()
    df = fetch_asx_symbol_list()
    uni = _pd.DataFrame({"ticker": df["ticker"].astype(str)})
    uni = merge_universe_with_isin(isin_path, uni)
    uni = uni.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    uni.to_csv(_UNIVERSE_CSV, index=False)
    return uni

def _load_caps_cache() -> Dict[str, dict]:
    if os.path.exists(_CAPS_CACHE):
        try:
            return json.load(open(_CAPS_CACHE,"r"))
        except Exception:
            return {}
    return {}

def _save_caps_cache(d: Dict[str, dict]):
    try:
        json.dump(d, open(_CAPS_CACHE,"w"))
    except Exception:
        pass

def _fetch_one_cap(sym: str) -> dict:
    url = f"https://eodhd.com/api/fundamentals/{sym}.AU"
    js = _eod_get(url)
    if not js:
        return {"ticker": sym, "market_cap_m": None, "sector": None}
    gen = js.get("General", {}) if isinstance(js, dict) else {}
    mc  = gen.get("MarketCapitalization")
    sec = gen.get("Sector")
    try:
        mc_m = float(mc)/1e6 if mc is not None else None
    except Exception:
        mc_m = None
    return {"ticker": sym, "market_cap_m": mc_m, "sector": sec}

def fetch_market_caps(universe: List[str], workers:int=8, max_names:int=0) -> _pd.DataFrame:
    """
    Parallel caps fetch via EODHD fundamentals. Uses JSON cache to avoid refetch.
    """
    _ensure_dirs()
    if not _EOD_API:
        raise RuntimeError("EODHD_API_KEY not set in env")

    cache = _load_caps_cache()
    out_rows = []
    todo = []
    # what’s already cached?
    for sym in universe:
        key = f"{sym}.AU"
        rec = cache.get(key)
        if rec and ("market_cap_m" in rec):
            out_rows.append({"ticker": sym, 
                             "market_cap_m": rec.get("market_cap_m"),
                             "sector": rec.get("sector")})
        else:
            todo.append(sym)
    if max_names and max_names>0:
        todo = todo[:max_names]

    if todo:
        with ThreadPoolExecutor(max_workers=max(1,workers)) as ex:
            fut = {ex.submit(_fetch_one_cap, s): s for s in todo}
            for f in as_completed(fut):
                s = fut[f]
                try:
                    rec = f.result()
                except Exception:
                    rec = {"ticker": s, "market_cap_m": None, "sector": None}
                out_rows.append(rec)
                # update cache
                cache[f"{s}.AU"] = {"market_cap_m": rec["market_cap_m"], "sector": rec["sector"]}
        _save_caps_cache(cache)

    caps = _pd.DataFrame(out_rows)
    if not caps.empty:
        caps = caps.drop_duplicates(subset=["ticker"], keep="last")
        caps.to_csv(_CAPS_CSV, index=False)
    return caps

def ensure_universe_and_caps(isin_path:str=None, workers:int=8, max_caps:int=0):
    """
    Public entry for main.py or any caller:
      - builds data/nextday_universe.csv (ASX list +/- ISIN merge)
      - builds data/universe_caps.csv (market_cap_m + sector)
    """
    uni = build_nextday_universe(isin_path=isin_path)
    tickers = uni["ticker"].astype(str).tolist()
    caps = fetch_market_caps(tickers, workers=workers, max_names=max_caps)
    # friendly print
    print(f"[UNIVERSE] {len(tickers)} symbols -> {_UNIVERSE_CSV}")
    print(f"[CAPS] rows={0 if caps is None else len(caps)} -> {_CAPS_CSV}")

# ======================  END symbols + market caps block  ======================

