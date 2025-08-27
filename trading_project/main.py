#!/usr/bin/env python3
import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from src.ml_daily_train_predict import build_dataset, train_model
from src.plan import generate_trade_plan

TOPN     = int(os.getenv("TOPN", "12"))
CAPITAL  = float(os.getenv("CAPITAL", "3000"))
SENT_W   = float(os.getenv("SENT_W", "0.30"))  # 0..1
OUT_DIR  = os.getenv("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("Loaded environment from", os.path.join(os.getcwd(), ".env") if os.path.exists(".env") else "(no .env)")

    # 1) data → features
    raw = build_dataset()  # your existing function; prints its own progress
    feats, feat_cols = raw.groupby("ticker", group_keys=False).apply(
        lambda df: df  # if you have add_features, keep it; otherwise identity
    ), [c for c in raw.columns if c not in ("date","ticker","close")]

    # 2) train + quick backtest (your existing utilities)
    model, holdout_stats = train_model(feats, feature_cols=feat_cols)
    if holdout_stats:
        print(holdout_stats)

    # 3) plan with sentiment
    plan_df = generate_trade_plan(
        model, feat_cols, feats,
        topN=TOPN, capital=CAPITAL, sent_w=SENT_W
    )

if __name__ == "__main__":
    main()
