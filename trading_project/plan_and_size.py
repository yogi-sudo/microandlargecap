import os, sys, math, time, datetime as dt, json
from pathlib import Path
import pandas as pd, numpy as np
import requests
from tqdm import tqdm

# -------- Config (env overridable) --------
EODHD_API_KEY = os.getenv("EODHD_API_KEY", "").strip()
PORTFOLIO_EQUITY = float(os.getenv("PORTFOLIO_EQUITY", 50000))
RISK_PCT_PER_TRADE = float(os.getenv("RISK_PCT_PER_TRADE", 0.005))  # 0.5%
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", 0.10))
LOOKBACK_YEARS = int(os.getenv("LOOKBACK_YEARS", 3))
W_MIN_VOL = float(os.getenv("W_MIN_VOL", 20000))
MIN_PRICE = float(os.getenv("MIN_PRICE", 0.2))
CACHE_DIR = Path("cache/ohlc")
OUT_DIR = Path("out")
TODAY = dt.date.today()

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Helpers --------
def to_ax(t): return f"{t}.AX"
def to_eod(t): return f"{t}.AU"

def daterange_start(years):
    return (TODAY - dt.timedelta(days=int(365.25*years))).isoformat()

def fetch_eodhd_daily(ticker):
    if not EODHD_API_KEY:
        return None
    url = f"https://eodhd.com/api/eod/{to_eod(ticker)}"
    params = {
        "api_token": EODHD_API_KEY,
        "from": daterange_start(LOOKBACK_YEARS),
        "to": TODAY.isoformat(),
        "period": "d",
        "fmt": "json"
    }
    try:
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        js = r.json()
        if not isinstance(js, list) or not js:
            return None
        df = pd.DataFrame(js)
        if df.empty: return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.rename(columns={"adjusted_close":"close"})
        need_cols = ["date","open","high","low","close","volume"]
        return df[[c for c in need_cols if c in df.columns]]
    except Exception:
        return None

def fetch_yf_daily(ticker):
    try:
        import yfinance as yf
        df = yf.download(to_ax(ticker), start=daterange_start(LOOKBACK_YEARS),
                         end=(TODAY + dt.timedelta(days=1)).isoformat(),
                         progress=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
        })
        df = df[["open","high","low","close","volume"]].copy()
        df["date"] = pd.to_datetime(df.index)
        df = df.reset_index(drop=True).sort_values("date")
        return df
    except Exception:
        return None

def load_cached_or_fetch(ticker):
    fp = CACHE_DIR / f"{ticker}_ohlc.csv"
    if fp.exists():
        try:
            df = pd.read_csv(fp, parse_dates=["date"])
            if not df.empty and df["date"].max().date() >= TODAY - dt.timedelta(days=3):
                return df
        except Exception:
            pass
    df = fetch_eodhd_daily(ticker)
    if df is None:
        df = fetch_yf_daily(ticker)
    if df is None or df.empty:
        return None
    df.to_csv(fp, index=False)
    return df

def atr14(df):
    c = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - c).abs(),
        (df["low"] - c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def last_features(df):
    p = df.copy()
    p["vma20"] = p["volume"].rolling(20).mean()
    p["vratio"] = p["volume"] / (p["vma20"] + 1e-9)
    p["ret5"] = p["close"].pct_change(5)
    p["hh_252"] = p["close"].rolling(252, min_periods=60).max()
    p["dist_to_hh"] = p["close"]/p["hh_252"] - 1.0
    p["ma20"] = p["close"].rolling(20).mean()
    p["ma50"] = p["close"].rolling(50).mean()
    p["trend_up"] = (p["ma20"] > p["ma50"]).astype(int)
    p["atr14"] = atr14(p)
    p = p.dropna().copy()
    if p.empty: return None
    row = p.iloc[-1]
    return {
        "close": float(row["close"]),
        "vratio": float(row["vratio"]),
        "ret5": float(row["ret5"]),
        "dist_to_hh": float(row["dist_to_hh"]),
        "trend_up": int(row["trend_up"]),
        "atr14": float(row["atr14"]),
        "avgvol20": float(row["vma20"])
    }

def normalize(series):
    s = pd.Series(series, dtype=float)
    lo, hi = np.nanpercentile(s, 5), np.nanpercentile(s, 95)
    s = (s - lo) / max(1e-9, (hi - lo))
    return s.clip(0,1)

def tier_from_cap(b):
    if b >= 20: return "large"
    if b >= 7: return "mid"
    return "micro"

def rating_from_score(score):
    if score >= 0.70: return "Strong Buy"
    if score >= 0.55: return "Buy"
    if score >= 0.45: return "Watch"
    return "Avoid"

def plan_position(price, atr, rating):
    if price <= 0 or atr <= 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    r_mult = 2.0 if rating != "Strong Buy" else 2.5
    stop = max(0.01, price - 2.0*atr)
    rr = price - stop
    if rr <= 0:
        return 0, stop, price, price, price
    risk_dollars = PORTFOLIO_EQUITY * RISK_PCT_PER_TRADE
    shares = int(max(0, math.floor(risk_dollars / rr)))
    max_dollars = PORTFOLIO_EQUITY * MAX_POS_PCT
    if shares*price > max_dollars:
        shares = int(max_dollars // price)
    take_profit = price + r_mult * rr
    dollars = shares*price
    return shares, stop, take_profit, rr, dollars

# -------- Main --------
def main():
    caps = pd.read_csv("data/asx_caps.csv")
    caps["Ticker"] = caps["Ticker"].astype(str).str.strip()
    caps["Company"] = caps["Company"].astype(str)
    caps["Price"] = pd.to_numeric(caps["Price"], errors="coerce")
    caps["MarketCapB"] = pd.to_numeric(caps["MarketCapB"], errors="coerce")
    caps = caps.dropna(subset=["Ticker","MarketCapB"]).reset_index(drop=True)

    rows = []
    tickers = caps["Ticker"].tolist()
    for t in tqdm(tickers, desc="OHLCV"):
        df = load_cached_or_fetch(t)
        if df is None or len(df) < 120:
            continue
        f = last_features(df)
        if f is None: 
            continue
        if f["close"] < MIN_PRICE or f["avgvol20"] < W_MIN_VOL:
            continue
        rows.append({
            "Ticker": t,
            "Close": f["close"],
            "VRatio": f["vratio"],
            "Ret5": f["ret5"],
            "DistToHH": f["dist_to_hh"],
            "TrendUp": f["trend_up"],
            "ATR14": f["atr14"],
            "AvgVol20": f["avgvol20"]
        })

    if not rows:
        print("No usable histories. Try lowering W_MIN_VOL or MIN_PRICE.")
        sys.exit(0)

    feat = pd.DataFrame(rows)
    joined = caps.merge(feat, on="Ticker", how="inner")
    joined["Tier"] = joined["MarketCapB"].apply(tier_from_cap)

    joined["_n_ret5"] = normalize(joined["Ret5"])
    joined["_n_vr"] = normalize(joined["VRatio"])
    joined["_n_hh"] = 1 - normalize(joined["DistToHH"])
    joined["_n_trend"] = joined["TrendUp"].astype(float)

    joined["Score"] = (
        0.35*joined["_n_hh"] +
        0.25*joined["_n_trend"] +
        0.20*joined["_n_vr"] +
        0.20*joined["_n_ret5"]
    )

    joined["Rating"] = joined["Score"].apply(rating_from_score)

    plans = []
    for _, r in joined.iterrows():
        shares, stop, tp, rr, dollars = plan_position(r["Close"], r["ATR14"], r["Rating"])
        plans.append({
            "Ticker": r["Ticker"],
            "Company": r["Company"],
            "Tier": r["Tier"],
            "Price": round(r["Close"], 4),
            "MarketCapB": round(r["MarketCapB"], 3),
            "Score": round(r["Score"], 3),
            "Rating": r["Rating"],
            "VRatio": round(r["VRatio"], 2),
            "Ret5_pct": round(100*r["Ret5"], 2),
            "DistToHH_pct": round(100*r["DistToHH"], 2),
            "TrendUp": int(r["TrendUp"]),
            "ATR14": round(r["ATR14"], 4),
            "AvgVol20": int(r["AvgVol20"]),
            "PosShares": shares,
            "PosDollars": round(dollars, 2),
            "Stop": round(stop, 4),
            "TakeProfit": round(tp, 4),
            "RR_$": round(rr, 4)
        })
    out = pd.DataFrame(plans)
    if out.empty:
        print("No positions suggested under risk rules.")
        sys.exit(0)

    out = out.sort_values(["Tier","Score","MarketCapB"], ascending=[True, False, False]).reset_index(drop=True)

    print("\n=== Large (top 10) ===")
    print(out[out["Tier"]=="large"].head(10)[["Ticker","Company","Price","Score","Rating","PosShares","Stop","TakeProfit","PosDollars"]].to_string(index=False))
    print("\n=== Mid (top 10) ===")
    print(out[out["Tier"]=="mid"].head(10)[["Ticker","Company","Price","Score","Rating","PosShares","Stop","TakeProfit","PosDollars"]].to_string(index=False))
    print("\n=== Micro (top 10) ===")
    print(out[out["Tier"]=="micro"].head(10)[["Ticker","Company","Price","Score","Rating","PosShares","Stop","TakeProfit","PosDollars"]].to_string(index=False))

    outf = OUT_DIR / f"trade_plan_{TODAY.isoformat()}.csv"
    out.to_csv(outf, index=False)
    print(f"\nSaved: {outf}")

if __name__ == "__main__":
    main()
