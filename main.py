#!/usr/bin/env python3
import os
import pandas as pd
from dotenv import load_dotenv

# --- Load .env (if present next to this file) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.exists(DOTENV_PATH):
    load_dotenv(DOTENV_PATH)
    print(f"Loaded environment from {DOTENV_PATH}")
else:
    print("WARNING: .env not found, using system environment variables.")

# --- Project imports ---
from src.universe import get_universe
from src.data_fetch import build_dataset
from src.features import add_features
from src.ml_model import train_and_eval, walkforward_backtest
from src.plan import generate_trade_plan
from src.sentiment import get_news_sentiment
from src.pnl import log_from_plan
from src.data_fetch import ensure_universe_and_caps

# build/refresh universe & caps (set ISIN path if you want it merged)
ensure_universe_and_caps(
    isin_path="data/ISIN.xls",   # or None if you don’t want to merge
    workers=8,                   # parallel fundamentals calls
    max_caps=0                   # 0 = no limit
)

# --- Config (env-overridable) ---
TOPN        = int(os.getenv("TOPN", 12))
CAPITAL     = float(os.getenv("CAPITAL", 3000))
SENT_W      = float(os.getenv("SENT_W", 0.35))       # weight for sentiment in blend [0..1]
BACKTEST_N  = int(os.getenv("BACKTEST_DAYS", 30))
OUT_DIR     = os.getenv("OUT_DIR", "out")

def main():
    # 1) Universe
    tickers = get_universe()
    print(f"Universe size: {len(tickers)}")

    # 2) Raw OHLCV (uses cache when available)
    raw = build_dataset(tickers=tickers)
    last_date = pd.to_datetime(raw["date"]).max().date() if not raw.empty else None
    print(f"Built dataset: {len(raw):,} rows | tickers: {raw['ticker'].nunique()} | last: {last_date}")

    # 3) Features
    f = raw.groupby("ticker", group_keys=False).apply(add_features)
    print(f"Featurized: {len(f):,} rows")

    # 4) Train + holdout eval
    model, feats = train_and_eval(f)

    # 5) Walk-forward backtest (last N trading days, next-day exit)
    walkforward_backtest(model, feats, f, days=BACKTEST_N, topN=TOPN)

    # 6) Warm up news-sentiment cache for the last trading day
    try:
        day = pd.to_datetime(f["date"]).max().date()
        day_tickers = sorted(f.loc[f["date"] == pd.to_datetime(day), "ticker"].unique())
        if day_tickers:
            _ = get_news_sentiment(day_tickers, day)  # writes out/sent_cache/*.json + out/news_sentiment.csv
    except Exception as e:
        print(f"(sentiment skipped: {e})")

    # 7) Trade plan (ML + cached sentiment + sizing / stops / targets)
    plan_df = generate_trade_plan(model, feats, f, topN=TOPN, capital=CAPITAL, sent_w=SENT_W)

    # 8) Paper PnL logging (T+1 close from cache) → out/performance.csv
    try:
        plan_date = pd.to_datetime(f["date"]).max().date()
        plan_csv = os.path.join(OUT_DIR, f"trade_plan_{plan_date}.csv")
        if os.path.exists(plan_csv):
            log_from_plan(plan_csv)
        else:
            print(f"[PNL] plan CSV not found at {plan_csv}")
    except Exception as e:
        print(f"[PNL] skipped: {e}")

if __name__ == "__main__":
    main()