#!/usr/bin/env python3
import argparse, os, datetime as dt, requests, pandas as pd, time

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY","")
EODHD_KEY   = os.getenv("EODHD_API_KEY","")

def newsapi_fetch(t, hours, max_hits=20):
    if not NEWSAPI_KEY: return []
    since = (dt.datetime.utcnow() - dt.timedelta(hours=hours)).date().isoformat()
    q = f'"{t} ASX" OR "{t}.AX"'
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q, "from": since, "language": "en",
        "sortBy": "publishedAt", "pageSize": max_hits, "apiKey": NEWSAPI_KEY
    }
    rows = []
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        for a in r.json().get("articles",[]) or []:
            title = (a.get("title") or "").strip()
            if not title: continue
            src = (a.get("source") or {}).get("name") or "newsapi"
            ts  = a.get("publishedAt") or dt.datetime.utcnow().isoformat()
            rows.append({"ticker": t, "headline": title, "source": src, "ts": ts})
    except Exception:
        pass
    return rows

def eodhd_fetch(t, hours, max_hits=50):
    if not EODHD_KEY: return []
    since = (dt.datetime.utcnow() - dt.timedelta(hours=hours)).strftime("%Y-%m-%d")
    # EODHD news endpoint pattern; symbol tries both T.AX and T
    rows = []
    for sym in (f"{t}.AX", t):
        try:
            url = f"https://eodhd.com/api/news"
            params = {"s": sym, "offset": 0, "limit": max_hits, "api_token": EODHD_KEY}
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            for a in r.json() or []:
                title = (a.get("title") or "").strip()
                if not title: continue
                dtp = a.get("date") or ""
                if dtp and dtp[:10] < since: continue
                src = a.get("source") or "eodhd"
                ts  = a.get("date") or dt.datetime.utcnow().isoformat()
                rows.append({"ticker": t, "headline": title, "source": src, "ts": ts})
            if rows: break
        except Exception:
            pass
        time.sleep(0.2)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True, help="CSV with 'ticker' column or comma-list")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--hours", type=int, default=96)
    args = ap.parse_args()

    if args.tickers.endswith(".csv"):
        df = pd.read_csv(args.tickers)
        tickers = sorted(set(df[df.columns[0]].astype(str).str.upper().str.replace(r"\.AX$","",regex=True)))
    else:
        tickers = [x.strip().upper().replace(".AX","") for x in args.tickers.split(",") if x.strip()]

    all_rows = []
    for t in tickers:
        all_rows += newsapi_fetch(t, args.hours)
        all_rows += eodhd_fetch(t, args.hours)

    if not all_rows:
        pd.DataFrame(columns=["ticker","headline","source","ts"]).to_csv(args.out_csv, index=False)
        print("wrote", args.out_csv, "rows=0"); return

    out = pd.DataFrame(all_rows).drop_duplicates()
    out.to_csv(args.out_csv, index=False)
    print("wrote", args.out_csv, "rows=", len(out))

if __name__ == "__main__":
    main()
