#!/usr/bin/env python3
import os, argparse, pandas as pd, requests, json, time

def get_dummy_cap(ticker: str) -> float:
    """Return a placeholder market cap in millions (simulate real API call)."""
    return hash(ticker) % 10000 + 50  # randomish millions for demo

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe_csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--cache", default="data/fundamentals_cache.json")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max", type=int, default=10000)
    args = p.parse_args()

    if not os.path.exists(args.universe_csv):
        raise SystemExit(f"Missing {args.universe_csv}")

    df = pd.read_csv(args.universe_csv)
    if "ticker" not in df.columns:
        raise SystemExit(f"{args.universe_csv} missing 'ticker' column")
    tickers = df["ticker"].astype(str).str.upper().tolist()[:args.max]

    rows = []
    for t in tickers:
        cap = get_dummy_cap(t)
        rows.append({"ticker": t, "market_cap_m": cap, "sector": "Unknown"})
        time.sleep(0.01)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"[CAPS] wrote {len(out)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()