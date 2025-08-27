#!/usr/bin/env python3
import os, argparse, datetime as dt
import pandas as pd

def _import_getter():
    try:
        from src.sentiment import get_news_sentiment
        return get_news_sentiment
    except Exception:
        from sentiment import get_news_sentiment
        return get_news_sentiment

def _today_au():
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(os.getenv("TIMEZONE","Australia/Sydney"))
        return dt.datetime.now(tz).date()
    except Exception:
        return dt.date.today()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="")
    ap.add_argument("--report_csv", default="")
    ap.add_argument("--out_csv", default="out/news_sentiment.csv")
    ap.add_argument("--date", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.strip().upper().replace(".AX","") for t in args.tickers.split(",") if t.strip()]
    elif args.report_csv and os.path.exists(args.report_csv):
        df = pd.read_csv(args.report_csv)
        col = "ticker" if "ticker" in df.columns else df.columns[0]
        tickers = sorted(set(df[col].astype(str).str.upper().str.replace(r"\.AX$","",regex=True)))
    else:
        print("Provide --tickers CSV or --report_csv")
        return

    D = _today_au() if not args.date else dt.date.fromisoformat(args.date)
    get_news_sentiment = _import_getter()
    scores = get_news_sentiment(tickers, D)
    out = pd.DataFrame([{"Date": D.isoformat(), "Ticker": k, "Sentiment": v} for k,v in scores.items()])
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    if args.verbose:
        print(out.sort_values("Sentiment", ascending=False).head(20).to_string(index=False))
    print("Saved ->", args.out_csv)

if __name__ == "__main__":
    main()
