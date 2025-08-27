#!/usr/bin/env python3
import os, datetime as dt
from typing import List
import numpy as np
import pandas as pd
from .sentiment import get_news_sentiment

OUT_DIR   = os.getenv("OUT_DIR", "out")
TODAY     = dt.date.today()

os.makedirs(OUT_DIR, exist_ok=True)

def _load_sentiment_map(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if {"Date","Ticker","Sentiment"}.issubset(df.columns):
            # keep only today’s rows
            today = dt.date.today().isoformat()
            df = df[df["Date"] == today]
            s = df.set_index("Ticker")["Sentiment"].astype(float)
            return s.to_dict()
    except Exception:
        pass
    return {}

def _clamp01(v, lo=-1.0, hi=1.0):
    try:
        v = float(v)
    except Exception:
        v = 0.0
    return max(lo, min(hi, v))

def generate_trade_plan(
    model,
    feat_cols: List[str],
    data: pd.DataFrame,
    topN: int = 10,
    capital: float = 3000.0,
    sent_w: float = 0.30,   # weight for sentiment [0..1]
) -> pd.DataFrame:

    if data.empty:
        raise ValueError("generate_trade_plan: empty dataset")

    last_day = pd.to_datetime(data["date"]).max()
    block = data[data["date"] == last_day].copy()
    if block.empty:
        raise ValueError("generate_trade_plan: no rows for the last day")

    # ML probability
    prob = model.predict_proba(block[feat_cols].values)[:, 1]
    block = block.assign(MLProb=prob)

    # Blend sentiment: load cache first, fill gaps live
    sent_csv = os.path.join(OUT_DIR, "news_sentiment.csv")
    sent_map = _load_sentiment_map(sent_csv)
    sent_list = []
    for t in block["ticker"]:
        if t in sent_map:
            sent_list.append(_clamp01(sent_map[t]))
        else:
            # Live fetch (cached to CSV for next run)
            sent_list.append(get_news_sentiment(t))
    block["Sentiment"] = [ _clamp01(x) for x in sent_list ]

    # Normalize prob to [-1,1] edge around 0.5 then blend with sentiment
    edge = (block["MLProb"] - 0.5) / 0.5
    block["Score"] = (1.0 - sent_w) * edge + sent_w * block["Sentiment"]

    # Rank
    block = block.sort_values(["Score","MLProb"], ascending=False).head(topN).copy()

    # Stops/targets scaled by recent volatility if available
    std20 = block.get("std20", pd.Series([np.nan]*len(block), index=block.index))
    vol_k = 1.0 + (std20.fillna(std20.median()).clip(lower=0) / (block["close"] + 1e-9)).fillna(0)
    stop_pct = (0.04 * vol_k).clip(upper=0.072)    # up to ~7.2%
    tp1_pct  = (0.03 * vol_k).clip(upper=0.048)
    tp2_pct  = (0.06 * vol_k).clip(upper=0.12)

    block["BuyPrice"] = block["close"]
    block["Stop"]     = (block["close"] * (1.0 - stop_pct)).round(4)
    block["Target1"]  = (block["close"] * (1.0 + tp1_pct)).round(4)
    block["Target2"]  = (block["close"] * (1.0 + tp2_pct)).round(4)

    # Sizing: equal capital per trade
    per_trade = capital / max(1, topN)
    qty = np.maximum(1, np.floor(per_trade / block["close"]))
    block["Qty"]     = qty.astype(int)
    block["Capital"] = (block["Qty"] * block["close"]).round(2)

    out = block[[
        "ticker","close","Score","MLProb","Sentiment",
        "BuyPrice","Stop","Target1","Target2","Qty","Capital"
    ]].rename(columns={"ticker":"Ticker","close":"Close"})

    # Optional Tier if present
    if "Tier" in block.columns:
        out.insert(0, "Tier", block["Tier"].values)

    # Save & print
    out_csv = os.path.join(OUT_DIR, f"trade_plan_{pd.to_datetime(block['date']).iloc[0].date().isoformat()}.csv")
    out.to_csv(out_csv, index=False)

    fmt = {
        "Close":     lambda x: f"{x:.2f}",
        "Score":     lambda x: f"{x:.3f}",
        "MLProb":    lambda x: f"{x:.2f}",
        "Sentiment": lambda x: f"{x:.2f}",
        "BuyPrice":  lambda x: f"{x:.2f}",
        "Stop":      lambda x: f"{x:.2f}",
        "Target1":   lambda x: f"{x:.2f}",
        "Target2":   lambda x: f"{x:.2f}",
        "Capital":   lambda x: f"{x:.2f}",
    }
    cols = [c for c in ["Tier","Ticker","Close","Score","MLProb","Sentiment",
                        "BuyPrice","Stop","Target1","Target2","Qty","Capital"] if c in out.columns]

    print("\n=== Trade Plan (topN) ===")
    print(out[cols].to_string(index=False, formatters=fmt))
    print(f"\nSaved: {out_csv}")
    return out
