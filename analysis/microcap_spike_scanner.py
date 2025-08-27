#!/usr/bin/env python3
import os, argparse, sys, math, time
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

# --------------- Helpers ---------------

def _norm_ticker(s: str) -> str:
    if pd.isna(s): return ""
    return (
        str(s).upper()
        .replace(".ASX","")
        .replace(".AX","")
        .strip()
    )

def ensure_universe(path: str) -> str:
    """
    Ensure we have a universe CSV with a 'ticker' column.
    If missing, try to build it from data/nextday_universe.csv or data/universe_caps.csv.
    Returns the path to a valid universe CSV.
    """
    if os.path.exists(path):
        return path

    candidates = [
        "data/nextday_universe.csv",
        "data/universe_caps.csv",
    ]
    base = None
    for c in candidates:
        if os.path.exists(c):
            base = c
            break
    if base is None:
        print(f"[ERROR] No universe file. Provide --universe or create one of: {candidates}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(base)
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    df[col] = df[col].map(_norm_ticker)
    df = df[df[col].str.len() > 0].drop_duplicates(subset=[col]).reset_index(drop=True)
    out = "data/nextday_universe_valid.csv"
    os.makedirs("data", exist_ok=True)
    df[[col]].rename(columns={col:"ticker"}).to_csv(out, index=False)
    print(f"[UNIVERSE] {len(df)} symbols -> {out}")
    return out

def load_universe(path: str) -> List[str]:
    path = ensure_universe(path)
    df = pd.read_csv(path)
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    tickers = df[col].map(_norm_ticker)
    tickers = tickers[tickers.str.len() > 0].drop_duplicates().tolist()
    return tickers

def load_caps(path: Optional[str]) -> Dict[str, float]:
    """
    Return {ticker: market_cap_m} if available, else {}.
    Accepts columns like 'market_cap_m' or 'market_cap' (in AUD) or 'MarketCap'.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        caps = pd.read_csv(path)
        # normalize column names
        cols = {c.lower(): c for c in caps.columns}
        tcol = "ticker" if "ticker" in cols else list(caps.columns)[0]
        caps["ticker"] = caps[tcol].map(_norm_ticker)
        # find a cap column
        cap_col = None
        for c in ["market_cap_m","marketcap_m","mkt_cap_m","mktcap_m","cap_m","market_cap"]:
            if c in cols:
                cap_col = cols[c]
                break
        if cap_col is None:
            # try MarketCap in absolute dollars, convert to millions
            for c in caps.columns:
                if c.lower() in ("marketcap","market_capitalization"):
                    cap_col = c
                    caps["market_cap_m"] = pd.to_numeric(caps[cap_col], errors="coerce") / 1e6
                    break
        else:
            # already in millions
            caps["market_cap_m"] = pd.to_numeric(caps[cap_col], errors="coerce")

        out = caps.dropna(subset=["ticker"]).set_index("ticker")["market_cap_m"].to_dict()
        return out
    except Exception:
        return {}

def load_events(path: Optional[str], hours: int = 96) -> set:
    """
    Read events.csv and return the set of tickers with news in the last `hours`.
    """
    if not path or not os.path.exists(path):
        return set()
    try:
        ev = pd.read_csv(path)
        if "ticker" not in ev.columns:
            return set()
        ev["ticker"] = ev["ticker"].map(_norm_ticker)
        if "published_at" in ev.columns:
            ev["published_at"] = pd.to_datetime(ev["published_at"], utc=True, errors="coerce")
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=hours)
            ev = ev[ev["published_at"] >= cutoff]
        return set(ev["ticker"].dropna().unique().tolist())
    except Exception:
        return set()

def read_prices_for(ticker: str, prices_dir: str) -> Optional[pd.DataFrame]:
    """
    Load intraday CSV for ticker (file name '<TICKER>.csv') with columns including:
    timestamp/date, close, volume.
    Returns a DataFrame with lowercase columns or None if not found/invalid.
    """
    p1 = os.path.join(prices_dir, f"{ticker}.csv")
    if not os.path.exists(p1):
        return None
    try:
        df = pd.read_csv(p1)
        df.columns = [c.lower() for c in df.columns]
        # harmonize timestamp
        ts_col = "timestamp" if "timestamp" in df.columns else ("date" if "date" in df.columns else None)
        if ts_col is None or "close" not in df.columns:
            return None
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col,"close"]).sort_values(ts_col)
        # volume may be missing on some feeds
        if "volume" not in df.columns:
            df["volume"] = np.nan
        return df
    except Exception:
        return None

def latest_stats(df: pd.DataFrame) -> Optional[Dict]:
    """
    Compute:
      price (last close)
      prev_close (previous bar close)
      gap_pct = price / prev_close - 1
      vol_last, vol_avg20, rel_vol
      dollar_vol = price * vol_last
    Return dict or None if insufficient data.
    """
    if df is None or df.empty:
        return None
    if "close" not in df.columns:
        return None

    n = len(df)
    if n < 3:
        return None

    price = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])

    # volumes
    vol = pd.to_numeric(df.get("volume", pd.Series(index=df.index, dtype=float)), errors="coerce")
    vol_last = float(vol.iloc[-1]) if not np.isnan(vol.iloc[-1]) else np.nan
    vol_avg20 = float(vol.tail(20).mean()) if len(vol) >= 1 else np.nan

    # rel vol
    if (vol_avg20 is None) or (vol_avg20 == 0) or math.isnan(vol_avg20):
        rel_vol = np.nan
    else:
        rel_vol = (vol_last / vol_avg20) if not math.isnan(vol_last) else np.nan

    # gap %
    gap_pct = (price / prev_close - 1.0) if prev_close > 0 else 0.0

    # dollar volume (approx)
    dollar_vol = price * (vol_last if not math.isnan(vol_last) else 0.0)

    return {
        "price": price,
        "prev_close": prev_close,
        "gap_pct": gap_pct,
        "rel_vol": rel_vol,
        "dollar_vol": dollar_vol,
    }

# --------------- Main scan ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="data/nextday_universe_valid.csv")
    ap.add_argument("--prices_dir", default="data/prices")
    ap.add_argument("--caps", default="data/universe_caps.csv")
    ap.add_argument("--events_csv", default="data/events.csv")
    ap.add_argument("--news_window_hours", type=int, default=96)

    ap.add_argument("--min_price", type=float, default=0.02)
    ap.add_argument("--max_price", type=float, default=5.00)
    ap.add_argument("--max_cap_m", type=float, default=1000.0)
    ap.add_argument("--min_relvol", type=float, default=1.8)
    ap.add_argument("--min_gap", type=float, default=0.05)
    ap.add_argument("--min_dollar_vol", type=float, default=150000.0)

    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out_csv", default="artifacts/microcap_candidates.csv")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Load universe, caps, and events
    tickers = load_universe(args.universe)
    caps_map = load_caps(args.caps)
    news_set = load_events(args.events_csv, hours=args.news_window_hours)

    rows = []
    pbar = tqdm(tickers, desc="Scan microcaps", ncols=100)
    for t in pbar:
        df = read_prices_for(t, args.prices_dir)
        st = latest_stats(df)
        if not st:
            continue

        price      = st["price"]
        gap_pct    = st["gap_pct"]
        rel_vol    = st["rel_vol"]
        dollar_vol = st["dollar_vol"]

        # Market cap (millions); if missing, set NaN so it can still pass if user wants
        cap_m = caps_map.get(t, np.nan)

        # Filters
        if not (args.min_price <= price <= args.max_price):
            continue
        if not np.isnan(cap_m) and cap_m > args.max_cap_m:
            continue
        if np.isnan(rel_vol) or rel_vol < args.min_relvol:
            continue
        if gap_pct < args.min_gap:
            continue
        if dollar_vol < args.min_dollar_vol:
            continue

        has_news = 1 if t in news_set else 0

        # Entry/TP/SL (quick template)
        entry = price
        tp    = round(price * 1.08, 6)  # +8%
        sl    = round(price * 0.94, 6)  # -6%

        # Score: combine rel_vol, gap, dollar_vol (scaled)
        score = (
            (rel_vol if not np.isnan(rel_vol) else 0) * 10.0
            + (gap_pct * 100.0) * 2.0
            + (dollar_vol / 100000.0)
            + (15.0 if has_news else 0.0)
        )

        rows.append({
            "ticker": t,
            "price": price,
            "gap_%": gap_pct * 100.0,
            "rel_vol": rel_vol,
            "dollar_vol": dollar_vol,
            "market_cap_m": cap_m,
            "has_news": has_news,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "score": score,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No microcap spikes passing filters.")
        # still write an empty file with headers for downstream consistency
        cols = ["ticker","price","gap_%","rel_vol","dollar_vol","market_cap_m","has_news","entry","tp","sl","score"]
        pd.DataFrame(columns=cols).to_csv(args.out_csv, index=False)
        print(f"Saved -> {args.out_csv}")
        return

    # Sort and limit
    out = out.sort_values(["score","gap_%","rel_vol"], ascending=False).head(args.top)

    # Pretty print
    def f2(x): 
        try: return f"{x:,.2f}"
        except: return str(x)
    def f4(x):
        try: return f"{x:.4f}"
        except: return str(x)

    print("\n=== Microcap Spike Candidates ===")
    view_cols = ["ticker","price","gap_%","rel_vol","dollar_vol","market_cap_m","has_news","entry","tp","sl","score"]
    print(out[view_cols].to_string(index=False, 
                                   formatters={
                                       "price": f4, "gap_%": f4, "rel_vol": f4,
                                       "dollar_vol": f2, "market_cap_m": f2,
                                       "entry": f4, "tp": f4, "sl": f4, "score": f2
                                   }))

    out.to_csv(args.out_csv, index=False)
    print(f"\nSaved -> {args.out_csv}")

if __name__ == "__main__":
    main()