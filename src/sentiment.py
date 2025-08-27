#!/usr/bin/env python3
import os, json, time, datetime as dt
from typing import Dict, List, Union
import pandas as pd
import requests

# Optional OpenAI (new SDK)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Directories / env
OUT_DIR = os.getenv("OUT_DIR", "out")
CACHE_DIR = os.path.join(OUT_DIR, "sent_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

NEWSAPI_KEY     = os.getenv("NEWSAPI_KEY", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------- cache helpers ----------
def _sent_cache_path(ticker: str, d: dt.date) -> str:
    safe = str(ticker).replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{d.isoformat()}.json")

def _csv_cache_path() -> str:
    return os.path.join(OUT_DIR, "news_sentiment.csv")

def _load_csv_cache() -> pd.DataFrame:
    p = _csv_cache_path()
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            if set(["Date","Ticker","Sentiment"]).issubset(df.columns):
                df["Date"] = pd.to_datetime(df["Date"]).dt.date
                return df
        except Exception:
            pass
    return pd.DataFrame(columns=["Date","Ticker","Sentiment"])

def _save_csv_cache(df: pd.DataFrame) -> None:
    df = df.copy()
    df = df[["Date","Ticker","Sentiment"]]
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(_csv_cache_path(), index=False)

# ---------- data fetchers ----------
def _newsapi_headlines(ticker: str, d: dt.date, max_n: int = 8) -> List[str]:
    if not NEWSAPI_KEY:
        return []
    base = str(ticker).replace(".AX","").strip()
    url  = "https://newsapi.org/v2/everything"
    # same-day window (you can widen as needed)
    day = d.isoformat()
    params = {
        "q": base,
        "from": day,
        "to": day,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max_n,
        "apiKey": NEWSAPI_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        arts = js.get("articles", []) or []
        heads = [a.get("title","") for a in arts if a.get("title")]
        return [h for h in heads if h][:max_n]
    except Exception:
        return []

def _gpt_score(headlines: List[str]) -> float:
    """
    Return a float in [-1,1] from GPT given a list of headlines.
    If no key/SDK or no headlines, returns 0.0.
    """
    if not headlines or not OPENAI_API_KEY or OpenAI is None:
        return 0.0
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "You are a concise financial sentiment rater. "
            "Given ASX news headlines for one company, return ONE number between -1 and 1 "
            "indicating short-horizon (1–3 days) sentiment for the stock.\n"
            "-1 = strongly negative, 0 = neutral, +1 = strongly positive.\n"
            "Return ONLY the number.\n\n"
            "Headlines:\n" + "\n".join(f"- {h}" for h in headlines)
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content": prompt}],
            temperature=0.0,
            max_tokens=8,
        )
        txt = (resp.choices[0].message.content or "").strip()
        val = float(str(txt).split()[0])
        return max(-1.0, min(1.0, val))
    except Exception:
        return 0.0

# ---------- public API ----------
def get_news_sentiment(
    tickers: Union[str, List[str]],
    d: dt.date = None
) -> Union[float, Dict[str, float]]:
    """
    Returns a sentiment score in [-1,1] for ticker(s) on date d.

    If `tickers` is a string → returns a float.
    If `tickers` is a list  → returns {ticker: float} dict.

    Uses:
    - CSV cache at out/news_sentiment.csv
    - per-ticker JSON cache at out/sent_cache/TICKER_YYYY-MM-DD.json
    - NewsAPI for headlines
    - GPT (optional) to turn headlines into a single score
    """
    if d is None:
        d = dt.date.today()

    # Normalize input to list for processing
    single = False
    if isinstance(tickers, str):
        single = True
        tick_list = [tickers]
    else:
        tick_list = list(tickers)

    df_cache = _load_csv_cache()
    results: Dict[str, float] = {}
    new_rows = []

    for t in tick_list:
        # 1) CSV cache has today's sentiment?
        row = df_cache[(df_cache["Date"] == d) & (df_cache["Ticker"] == t)]
        if not row.empty:
            s = float(row["Sentiment"].iloc[0])
            results[t] = max(-1.0, min(1.0, s))
            continue

        # 2) JSON cache for this (ticker, date)?
        pj = _sent_cache_path(t, d)
        if os.path.exists(pj):
            try:
                js = json.load(open(pj, "r"))
                s = float(js.get("sentiment", 0.0))
                s = max(-1.0, min(1.0, s))
                results[t] = s
                new_rows.append({"Date": d, "Ticker": t, "Sentiment": s})
                continue
            except Exception:
                pass

        # 3) Fresh headlines → GPT score (optional)
        heads = _newsapi_headlines(t, d, max_n=8)
        s = _gpt_score(heads) if heads else 0.0
        results[t] = s

        # write JSON cache
        try:
            json.dump({"ticker": t, "date": d.isoformat(), "sentiment": s, "headlines": heads},
                      open(pj, "w"))
        except Exception:
            pass

        # stage CSV row
        new_rows.append({"Date": d, "Ticker": t, "Sentiment": s})

        # gentle pacing to be polite to APIs
        time.sleep(0.2)

    # Merge & persist CSV cache
    if new_rows:
        add = pd.DataFrame(new_rows)
        add["Date"] = pd.to_datetime(add["Date"]).dt.date
        df_cache = pd.concat([df_cache, add], ignore_index=True)
        df_cache = df_cache.drop_duplicates(subset=["Date","Ticker"], keep="last")
        _save_csv_cache(df_cache)

    return results[tick_list[0]] if single else results