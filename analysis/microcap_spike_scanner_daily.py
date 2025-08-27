#!/usr/bin/env python3
import os, argparse, pandas as pd, numpy as np, re

def load_caps(path):
    if not os.path.exists(path): return {}
    df = pd.read_csv(path)
    tcol = "ticker" if "ticker" in df.columns else df.columns[0]
    if "market_cap_m" not in df.columns: return {}
    m = df[[tcol, "market_cap_m"]].dropna()
    m[tcol] = (m[tcol].astype(str).str.upper()
               .str.replace(r"\.AX$", "", regex=True))
    m["market_cap_m"] = pd.to_numeric(m["market_cap_m"], errors="coerce")
    return dict(zip(m[tcol], m["market_cap_m"]))

def has_recent_news(events_csv, ticker, hours=48):
    if not os.path.exists(events_csv): return 0
    ev = pd.read_csv(events_csv)
    if "ticker" not in ev.columns or "published_at" not in ev.columns: return 0
    ev["ticker"] = (ev["ticker"].astype(str).str.upper()
                    .str.replace(r"\.AX$", "", regex=True))
    ev["published_at"] = pd.to_datetime(ev["published_at"], errors="coerce", utc=True)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=hours)
    sub = ev[(ev["ticker"] == ticker) & (ev["published_at"] >= cutoff)]
    return int(len(sub) > 0)

def _to_num(x):
    # handle "1,234.56" and any stray strings
    if isinstance(x, str):
        x = re.sub(r"[,\s]", "", x)
    return pd.to_numeric(x, errors="coerce")

def last_two_daily(csv_path):
    """
    Returns (prev_close:float, last_close:float, last_date_iso:str)
    Works with:
      - Yahoo daily: columns [date, close, volume]
      - Our intraday: columns [timestamp, close] (collapsed to daily last)
    """
    try:
        df = pd.read_csv(csv_path)
        if "close" not in df.columns:
            return None
        # coerce close to numeric
        df["close"] = df["close"].apply(_to_num)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            g = df.dropna(subset=["date", "close"]).sort_values("date")
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp", "close"])
            df["date"] = df["timestamp"].dt.date
            g = (df.groupby("date", as_index=False)["close"].last()
                   .dropna(subset=["date","close"]).sort_values("date"))
        else:
            return None
        if len(g) < 2:
            return None
        prev = float(g.iloc[-2]["close"])
        last = float(g.iloc[-1]["close"])
        dlast = pd.to_datetime(g.iloc[-1]["date"]).date().isoformat()
        return prev, last, dlast
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--prices_dir", default="data/prices_daily")   # fallback daily location
    ap.add_argument("--caps", default="data/universe_caps.csv")
    ap.add_argument("--events_csv", default="data/events.csv")
    ap.add_argument("--min_price", type=float, default=0.02)
    ap.add_argument("--max_price", type=float, default=10.00)
    ap.add_argument("--max_cap_m", type=float, default=2000)
    ap.add_argument("--min_gap", type=float, default=0.03)  # 3%
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out_csv", default="artifacts/microcap_candidates.csv")
    args = ap.parse_args()

    os.makedirs("artifacts", exist_ok=True)

    # Universe
    u = pd.read_csv(args.universe)
    tcol = "ticker" if "ticker" in u.columns else u.columns[0]
    tickers = (u[tcol].astype(str).str.upper()
               .str.replace(r"\.AX$", "", regex=True)).unique()

    # Caps
    caps = load_caps(args.caps)

    rows = []
    for t in tickers:
        cap = caps.get(t, np.nan)
        if not np.isnan(cap) and cap > args.max_cap_m:
            continue

        # prefer intraday collapsed if present, else daily file
        path_intr = os.path.join("data/prices", f"{t}.csv")
        path_daily = os.path.join(args.prices_dir, f"{t}.csv")
        data_path = path_intr if os.path.exists(path_intr) else path_daily
        if not os.path.exists(data_path):
            continue

        pair = last_two_daily(data_path)
        if pair is None:
            continue
        prev, last, dlast = pair
        if not (isinstance(prev, float) and isinstance(last, float)):
            continue
        if prev <= 0:
            continue

        # price window filter
        if last < args.min_price or last > args.max_price:
            continue

        gap = (last - prev) / prev
        if abs(gap) < args.min_gap:
            continue

        rows.append({
            "ticker": t,
            "price": round(last, 4),
            "gap_%": round(gap * 100.0, 4),
            "market_cap_m": cap if not np.isnan(cap) else None,
            "has_news": has_recent_news(args.events_csv, t, hours=48),
            "ts_last": dlast
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No microcap gaps passing filters.")
        out.to_csv(args.out_csv, index=False)
        print("Saved ->", args.out_csv)
        return

    # Simple score: gap plus a small news boost
    out["score"] = out["gap_%"] + out["has_news"] * 2.0
    out = out.sort_values(["score", "gap_%"], ascending=False).head(args.top)

    print("\n=== Microcap Gap Candidates ===")
    cols = ["ticker","price","gap_%","market_cap_m","has_news","ts_last","score"]
    print(out[cols].to_string(index=False))

    out.to_csv(args.out_csv, index=False)
    print("\nSaved ->", args.out_csv)

if __name__ == "__main__":
    main()
