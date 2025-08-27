#!/usr/bin/env python3
import os, json, argparse
import pandas as pd
from datetime import datetime

COMBINED = "artifacts/nextday_combined.csv"
EVENTS   = "data/events.csv"
SENTS    = "out/news_sentiment.csv"

def norm_tk(s):
    return (s.astype(str).str.upper()
            .str.replace(r"\.AX$|\.ASX$","", regex=True)
            .str.replace(r"[^0-9A-Z]+","", regex=True))

def load_sentiments(path):
    if not os.path.exists(path): return {}
    df = pd.read_csv(path)
    if not {"Date","Ticker","Sentiment"}.issubset(df.columns): return {}
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    today = datetime.utcnow().date()
    df = df[df["Date"]==today]
    return df.set_index("Ticker")["Sentiment"].to_dict()

def main():
    ap = argparse.ArgumentParser(description="Merge latest RSS headline + sentiment into combined report")
    ap.add_argument("--combined", default=COMBINED)
    ap.add_argument("--events",   default=EVENTS)
    ap.add_argument("--sents",    default=SENTS)
    args = ap.parse_args()

    if not os.path.exists(args.combined):
        raise SystemExit(f"[merge] combined file not found: {args.combined}")

    df = pd.read_csv(args.combined)
    if "ticker" not in df.columns:
        raise SystemExit("[merge] combined has no 'ticker' column")

    df["ticker"] = norm_tk(df["ticker"])

    latest = {}
    if os.path.exists(args.events):
        ev = pd.read_csv(args.events)
        if {"ts_utc","title","tickers"}.issubset(ev.columns):
            ev = ev.copy()
            ev["ts_utc"] = pd.to_datetime(ev["ts_utc"], errors="coerce")
            ev = ev.dropna(subset=["ts_utc"])
            rows = []
            for _, r in ev.iterrows():
                try:
                    tks = json.loads(r["tickers"]) if isinstance(r["tickers"], str) else []
                except Exception:
                    tks = []
                for t in tks:
                    rows.append({"ticker": str(t).upper(), "ts_utc": r["ts_utc"], "title": r["title"], "source": "rss"})
            evx = pd.DataFrame(rows)
            if not evx.empty:
                evx = evx.sort_values("ts_utc", ascending=False)
                latest = evx.drop_duplicates(subset=["ticker"], keep="first").set_index("ticker")[["title","source"]].to_dict(orient="index")

    sent_map = load_sentiments(args.sents)

    headlines, sources, sents = [], [], []
    if "headline" not in df.columns: df["headline"] = ""
    if "source" not in df.columns:   df["source"]   = ""

    for t, h, s in zip(df["ticker"], df["headline"], df["source"]):
        if t in latest:
            headlines.append(latest[t]["title"])
            sources.append("rss")
        else:
            headlines.append(h)
            sources.append(s)
        sents.append(sent_map.get(t, None))

    df["headline"] = headlines
    df["source"]   = sources
    if "sentiment" not in df.columns:
        df.insert(df.columns.get_loc("source")+1, "sentiment", sents)
    else:
        df["sentiment"] = sents

    df["has_news"] = df["headline"].notna() & (df["headline"].astype(str).str.len()>0)
    df.to_csv(args.combined, index=False)
    print(f"[merge] updated -> {args.combined}")
    print(df.head(12)[["group","ticker","headline","source","sentiment","has_news"]].to_string(index=False))

if __name__=="__main__":
    main()
