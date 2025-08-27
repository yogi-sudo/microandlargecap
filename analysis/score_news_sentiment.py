#!/usr/bin/env python3
import os, re, argparse, datetime as dt
import pandas as pd

POS = [
  "beats", "beat", "upgrade", "raises guidance", "record profit", "surge",
  "contract win", "contract awarded", "approval", "licence granted", "license granted",
  "positive", "strong", "partnership", "merger", "takeover", "acquisition",
  "buyback", "dividend increase", "higher dividend", "guidance raised"
]
NEG = [
  "misses", "miss", "downgrade", "cuts guidance", "profit warning", "class action",
  "fraud", "halt", "trading halt", "investigation", "recall", "layoffs", "termination",
  "dilutive", "placement at discount", "capital raising", "suspend", "bankruptcy",
  "adminstration", "insolvency", "loss", "negative"
]
WORST = [
  "fraud", "bankruptcy", "insolvency", "administration", "trading halt",
  "class action", "criminal", "suspend", "de-list", "delist"
]
CATALYST_TAGS = {
  "earnings beat": ["beats expectations","beats","record profit","above guidance"],
  "earnings miss": ["misses","below guidance","profit warning"],
  "placement/raise": ["capital raising","placement","rights issue","share issue","raising"],
  "contract win": ["contract win","awarded","tender win"],
  "partnership/M&A": ["partnership","merger","acquisition","takeover"],
  "regulatory": ["approval","licence","license","clearance","permit"],
}

def norm_tk(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.upper()
            .str.replace(r"\.AX$|\.ASX$","",regex=True)
            .str.replace(r"[^0-9A-Z]+","",regex=True))

def keep_recent(df, hours):
    if "ts" not in df.columns: return df
    try:
        t = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    except Exception:
        return df
    now = pd.Timestamp.utcnow()
    return df[t >= now - pd.Timedelta(hours=hours)]

def score_headlines(heads):
    htxt = " ".join(h.lower() for h in heads)
    pos = sum(any(w in h.lower() for w in POS) for h in heads)
    neg = sum(any(w in h.lower() for w in NEG) for h in heads)
    worst_hit = any(w in htxt for w in WORST)

    total = max(1, pos + neg)
    raw = (pos - neg) / total
    raw = max(-1.0, min(1.0, raw))

    if worst_hit: label = "worst"
    elif raw >= 0.2: label = "good"
    elif raw <= -0.2: label = "bad"
    else: label = "mixed"

    tags = []
    for tag, keys in CATALYST_TAGS.items():
        if any(k in htxt for k in keys): tags.append(tag)
    return raw, label, ", ".join(sorted(set(tags))) if tags else ""

def best_headline(heads):
    if not heads: return ""
    # prefer headlines with any catalyst word
    pri = [h for h in heads if any(k in h.lower() for ks in CATALYST_TAGS.values() for k in ks)]
    return (pri[0] if pri else heads[0])[:180]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_csv", default="data/events.csv")
    ap.add_argument("--hours", type=int, default=int(os.getenv("NEWS_WINDOW_HOURS","96")))
    ap.add_argument("--out_csv", default="artifacts/news_sentiment_today.csv")
    args = ap.parse_args()

    if not os.path.exists(args.events_csv):
        os.makedirs("artifacts", exist_ok=True)
        pd.DataFrame(columns=["ticker","sentiment","label","catalysts","headline_sample"]).to_csv(args.out_csv, index=False)
        print("[news] no events.csv; wrote empty", args.out_csv)
        return

    df = pd.read_csv(args.events_csv)
    if "ticker" not in df.columns or "headline" not in df.columns:
        pd.DataFrame(columns=["ticker","sentiment","label","catalysts","headline_sample"]).to_csv(args.out_csv, index=False)
        print("[news] events.csv missing columns; wrote empty", args.out_csv)
        return

    df["ticker"] = norm_tk(df["ticker"])
    df = keep_recent(df, args.hours)
    if df.empty:
        pd.DataFrame(columns=["ticker","sentiment","label","catalysts","headline_sample"]).to_csv(args.out_csv, index=False)
        print("[news] no recent headlines; wrote empty", args.out_csv)
        return

    rows = []
    for t, g in df.groupby("ticker"):
        heads = [str(x) for x in g["headline"].dropna().tolist()]
        if not heads: continue
        s, lab, cats = score_headlines(heads)
        rows.append({
            "ticker": t,
            "sentiment": s,
            "label": lab,
            "catalysts": cats,
            "pos": sum(any(w in h.lower() for w in POS) for h in heads),
            "neg": sum(any(w in h.lower() for w in NEG) for h in heads),
            "total": len(heads),
            "headline_sample": best_headline(heads)
        })

    out = pd.DataFrame(rows).sort_values("sentiment", ascending=False)
    out.to_csv(args.out_csv, index=False)
    print(f"[news] wrote {len(out)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()
