#!/usr/bin/env python3
import os, time, json, hashlib, datetime as dt, pandas as pd, requests
from typing import Dict, Any, List, Optional

EODHD_API_KEY = os.getenv("EODHD_API_KEY", "").strip()
BASE = "https://eodhd.com/api"
CACHE_DIR = os.getenv("EOD_CACHE_DIR", "cache_eodhd")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------- simple disk cache ----------
def _cache_path(tag: str, params: Dict[str, Any]) -> str:
    h = hashlib.md5((tag + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{tag}_{h}.json")

def _get_json(tag: str, path: str, params: Dict[str, Any], ttl_sec: int = 3600) -> Any:
    if not EODHD_API_KEY:
        raise RuntimeError("Missing EODHD_API_KEY in environment/.env")
    params = dict(params or {})
    params.setdefault("api_token", EODHD_API_KEY)
    params.setdefault("fmt", "json")

    p = _cache_path(tag, params)
    if os.path.exists(p) and (time.time() - os.path.getmtime(p) < ttl_sec):
        with open(p, "r") as f:
            return json.load(f)

    url = f"{BASE}{path}"
    last_exc = None
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                with open(p, "w") as f:
                    json.dump(data, f)
                return data
            last_exc = RuntimeError(f"{r.status_code} {r.text[:200]}")
        except Exception as e:
            last_exc = e
        time.sleep(1.2)
    raise last_exc

# ---------- datasets you asked for ----------
def exchanges_list() -> pd.DataFrame:
    data = _get_json("exchanges", "/exchanges-list", {})
    return pd.DataFrame(data)

def symbols_by_exchange(code: str) -> pd.DataFrame:
    data = _get_json(f"sym_{code}", "/exchange-symbol-list/" + code, {})
    return pd.DataFrame(data)

def eod(symbol: str, start: Optional[str] = None, end: Optional[str] = None, period: str = "d") -> pd.DataFrame:
    params = {"period": period}
    if start: params["from"] = start
    if end:   params["to"] = end
    data = _get_json(f"eod_{symbol}", f"/eod/{symbol}", params)
    df = pd.DataFrame(data)
    if df.empty: return df
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    cols = ["symbol","date","open","high","low","close","adjusted_close","volume"]
    for c in cols:
        if c not in df.columns:
            if c == "adjusted_close" and "adjusted_close" not in df.columns:
                df["adjusted_close"] = df.get("close")
            else:
                df[c] = pd.NA
    return df[cols]

def technicals(symbol: str, function: str = "rsi", period: int = 14) -> pd.DataFrame:
    # EODHD technicals: /technical/indicator/{symbol}?function=rsi&period=14
    data = _get_json(f"tech_{symbol}_{function}_{period}", f"/technical/indicator/{symbol}", {"function": function, "period": period})
    # typical payload: {"symbol":"XXX","values":[{"date":"2024-01-02","rsi":...}, ...]}
    values = data.get("values", [])
    df = pd.DataFrame(values)
    if df.empty: return df
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    return df

def fundamentals(symbol: str) -> Dict[str, Any]:
    return _get_json(f"fund_{symbol}", f"/fundamentals/{symbol}", {})

def splits(symbol: str) -> pd.DataFrame:
    data = _get_json(f"splits_{symbol}", f"/splits/{symbol}", {})
    df = pd.DataFrame(data)
    if df.empty: return df
    for c in ("date","split"):
        if c not in df.columns: df[c] = pd.NA
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = symbol
    return df[["symbol","date","split"]]

def dividends(symbol: str) -> pd.DataFrame:
    data = _get_json(f"divs_{symbol}", f"/div/{symbol}", {})
    df = pd.DataFrame(data)
    if df.empty: return df
    df["date"] = pd.to_datetime(df.get("date") or df.get("exDate"))
    df["symbol"] = symbol
    # normalize dividend per share if present
    if "value" in df.columns:
        df.rename(columns={"value":"dividend"}, inplace=True)
    elif "amount" in df.columns:
        df.rename(columns={"amount":"dividend"}, inplace=True)
    else:
        df["dividend"] = pd.NA
    return df[["symbol","date","dividend"]]

def earnings_calendar(symbols: List[str], from_date: str, to_date: str) -> pd.DataFrame:
    # EODHD calendar has multiple endpoints; we’ll call /calendar/earnings with filters per day window
    out = []
    for s in symbols:
        payload = _get_json(f"cal_{s}_{from_date}_{to_date}", "/calendar/earnings", {"symbols": s, "from": from_date, "to": to_date})
        items = payload if isinstance(payload, list) else payload.get("earnings", [])
        for it in items:
            it["symbol"] = s
            out.append(it)
        time.sleep(0.25)
    return pd.DataFrame(out)

def tick_data(symbol: str, date_str: str) -> pd.DataFrame:
    # Intraday trades/quotes: /ticks/{symbol}?date=YYYY-MM-DD
    data = _get_json(f"tick_{symbol}_{date_str}", f"/ticks/{symbol}", {"date": date_str})
    df = pd.DataFrame(data)
    if df.empty: return df
    # normalize timestamp
    if "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df["symbol"] = symbol
    return df