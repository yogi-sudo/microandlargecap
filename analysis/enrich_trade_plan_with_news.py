#!/usr/bin/env python3
import os, glob, pandas as pd

def norm_tk(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.upper()
            .str.replace(r"\.AX$|\.ASX$","",regex=True)
            .str.replace(r"[^0-9A-Z]+","",regex=True))

def main():
    files = sorted(glob.glob("out/trade_plan*.csv"))
    if not files:
        print("[enrich] no trade_plan*.csv found")
        return
    p = files[-1]
    plan = pd.read_csv(p)
    if "Ticker" in plan.columns:
        plan["ticker"] = norm_tk(plan["Ticker"])
    else:
        plan["ticker"] = norm_tk(plan.get("ticker", ""))

    news_path = "artifacts/news_sentiment_today.csv"
    news = pd.read_csv(news_path) if os.path.exists(news_path) else pd.DataFrame(columns=["ticker","sentiment","label","headline_sample"])
    if not news.empty:
        news["ticker"] = norm_tk(news["ticker"])

    out = plan.merge(news[["ticker","sentiment","label","headline_sample"]], on="ticker", how="left")
    out["Sentiment"] = out["sentiment"].fillna(0.0).round(2)
    out["News"] = out["label"].fillna("none")
    out["Headline"] = out["headline_sample"].fillna("")

    keep = ["Ticker","close","Score","MLProb","Sentiment","BuyPrice","Stop","Target1","Target2","Qty","Capital","News"]
    keep = [c for c in keep if c in out.columns]
    print("\n=== Trade Plan (topN) with News ===")
    fmts = {
        "close": lambda x: f"{x:.2f}",
        "Score": lambda x: f"{x:.3f}",
        "MLProb": lambda x: f"{x:.2f}",
        "Sentiment": lambda x: f"{x:.2f}",
        "BuyPrice": lambda x: f"{x:.2f}",
        "Stop": lambda x: f"{x:.2f}",
        "Target1": lambda x: f"{x:.2f}",
        "Target2": lambda x: f"{x:.2f}",
        "Capital": lambda x: f"{x:.2f}",
    }
    print(out[keep].to_string(index=False, formatters=fmts))

    out_path = "out/trade_plan_with_news.csv"
    out.to_csv(out_path, index=False)
    print("\n[enrich] wrote ->", out_path)

if __name__ == "__main__":
    main()
