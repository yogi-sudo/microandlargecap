import argparse, os, re, sys, time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Set, Optional
from urllib.parse import urlparse

import feedparser
import pandas as pd

def norm_token(s: str) -> str:
    return re.sub(r"[^0-9A-Z]+", "", s.upper())

def norm_ticker(s: str) -> str:
    s = s.upper()
    s = re.sub(r"\.(AX|ASX)$", "", s)
    return re.sub(r"[^0-9A-Z]+", "", s)

def load_tickers(path: str) -> Set[str]:
    df = pd.read_csv(path)
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    return set(norm_ticker(x) for x in df[col].astype(str))

def load_aliases(path: Optional[str]) -> Dict[str, Set[str]]:
    amap: Dict[str, Set[str]] = {}
    if not path or not os.path.exists(path):
        return amap
    df = pd.read_csv(path)
    if not {"alias", "ticker"} <= set(df.columns):
        return amap
    for _, r in df.iterrows():
        a = norm_token(str(r["alias"]))
        tk = norm_ticker(str(r["ticker"]))
        if not a or not tk:
            continue
        amap.setdefault(a, set()).add(tk)
    return amap

def load_sources(path: Optional[str]) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

def text_fields_from_entry(e) -> str:
    parts = []
    for k in ("title", "summary"):
        v = getattr(e, k, None)
        if v:
            parts.append(str(v))
    # include category/tags if any
    for tag in getattr(e, "tags", []) or []:
        t = getattr(tag, "term", None)
        if t:
            parts.append(str(t))
    return " ".join(parts)

def parse_time(e) -> Optional[datetime]:
    for attr in ("published_parsed", "updated_parsed"):
        tup = getattr(e, attr, None)
        if tup:
            try:
                return datetime.fromtimestamp(time.mktime(tup), tz=timezone.utc)
            except Exception:
                pass
    return None

def match_headline_to_tickers(text: str, tickers: Set[str], alias_map: Dict[str, Set[str]]) -> Set[str]:
    s = norm_token(text)
    hits: Set[str] = set()
    # ticker literal
    for tk in tickers:
        if tk and tk in s:
            hits.add(tk)
    # alias expansion
    # only apply aliases that are reasonably long to avoid noise
    for alias, tks in alias_map.items():
        if alias and len(alias) >= 3 and alias in s:
            hits.update(tks)
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True, help="CSV with a 'ticker' column")
    ap.add_argument("--out_csv", required=True, help="Output CSV")
    ap.add_argument("--hours", type=int, default=96, help="Lookback window hours")
    ap.add_argument("--sources", required=False, help="Text file with one RSS URL per line")
    ap.add_argument("--aliases", required=False, help="CSV with columns: alias,ticker")
    args = ap.parse_args()

    tickers = load_tickers(args.tickers)
    alias_map = load_aliases(args.aliases)
    sources = load_sources(args.sources)
    if not sources:
        sources = [
            "https://www.marketindex.com.au/news/rss.xml",
            "https://au.finance.yahoo.com/rss/",
            "https://www.afr.com/rss",
            "https://www.theaustralian.com.au/business/rss",
            "https://www.livewiremarkets.com/feeds/rss",
        ]

    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    rows: List[Dict[str, str]] = []

    for src in sources:
        try:
            fp = feedparser.parse(src)
            dom = urlparse(src).netloc or "rss"
            for e in getattr(fp, "entries", []) or []:
                ts = parse_time(e)
                if not ts or ts < cutoff:
                    continue
                title = getattr(e, "title", "") or ""
                if not title:
                    continue
                text = text_fields_from_entry(e)
                link = getattr(e, "link", "") or ""
                matched = match_headline_to_tickers(text, tickers, alias_map)
                for tk in matched:
                    rows.append({
                        "ticker": tk,
                        "headline": title,
                        "source": dom,
                        "ts": ts.isoformat(),
                        "url": link,
                        "domain": dom,
                    })
        except Exception as exc:
            print(f"[rss] error {src}: {exc}", file=sys.stderr)

    df = pd.DataFrame(rows).drop_duplicates()
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[rss] wrote {len(df)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()
