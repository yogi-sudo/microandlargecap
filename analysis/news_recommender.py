#!/usr/bin/env python3
import os, pandas as pd, re
from datetime import datetime, timezone

COMBINED = "artifacts/nextday_combined.csv"
EVENTS = "data/events.csv"
OUT = "artifacts/nextday_tradeplan.csv"

# Keywords for catalyst detection
POSITIVE_WORDS = {"record","beats","surge","contract","approval","profit","upgrade","strong"}
NEGATIVE_WORDS = {"downgrade","loss","delay","lawsuit","weak","miss","fraud"}

def simple_sentiment(text: str) -> float:
    if not text or not isinstance(text,str): return 0.0
    t = text.lower()
    score = sum(+1 for w in POSITIVE_WORDS if w in t) - sum(1 for w in NEGATIVE_WORDS if w in t)
    return score

# 1. Load ML picks
if not os.path.exists(COMBINED):
    raise SystemExit(f"No combined file: {COMBINED}")
df = pd.read_csv(COMBINED)
df["ticker"] = df["ticker"].astype(str).str.upper()

# 2. Load news
news = pd.read_csv(EVENTS) if os.path.exists(EVENTS) else pd.DataFrame(columns=["ticker","headline","ts"])
if not news.empty:
    news["ticker"] = news["ticker"].astype(str).str.upper()
    news["sentiment"] = news["headline"].map(simple_sentiment)
    news = news.sort_values("ts", ascending=False)
    # Aggregate
    agg = news.groupby("ticker").agg(
        news_score=("sentiment","mean"),
        headline_top=("headline","first")
    ).reset_index()
else:
    agg = pd.DataFrame(columns=["ticker","news_score","headline_top"])

# 3. Merge
out = df.merge(agg, on="ticker", how="left")

# 4. Recommendation logic
out["news_score"] = out["news_score"].fillna(0)
out["has_catalyst"] = out["news_score"].apply(lambda s: "YES" if s>0 else "NO")
out["rec"] = out.apply(lambda r: 
    "STRONG BUY" if r.get("prob_pct",0)>60 and r["news_score"]>0 else
    "WATCH" if r.get("prob_pct",0)>50 else
    "IGNORE", axis=1)

# 5. Save final table
keep = ["ticker","cap_band","prob_pct","news_score","headline_top","rec"]
out[keep].to_csv(OUT, index=False)
print(f"[news_recommender] wrote {len(out)} rows -> {OUT}")
print(out[keep].head(20).to_string(index=False))
