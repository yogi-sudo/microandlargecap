#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config (tweak as needed)
# =========================
NEWS_WINDOW_HOURS="${NEWS_WINDOW_HOURS:-96}"
TOP_SWING="${TOP_SWING:-12}"          # how many swing picks to print
TOP_MICRO="${TOP_MICRO:-50}"          # how many microcap spikes to keep
PYTHON_BIN="${PYTHON_BIN:-python3}"   # or "python"
REQS_FILE="requirements.txt"

# Progress: force Python to show tqdm bars even if not TTY
export PYTHONUNBUFFERED=1
export PYTHONPATH="."
export TQDM_DISABLE=0
export TQDM_MININTERVAL=0.2
export TQDM_FILE=1        # send bars to stdout

# =========================
# Helpers
# =========================
log() { printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }
have() { [[ -f "$1" ]]; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

STEP=0
next() { STEP=$((STEP+1)); log "STEP ${STEP}/10: $*"; }

# =========================
# Dirs & venv
# =========================
next "Setup folders & virtualenv"
mkdir -p data data/prices data/prices_daily artifacts out signals models

if [[ -d .venv ]]; then
  log "Using existing venv"
else
  log "Creating venv"
  ${PYTHON_BIN} -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
if [[ -f "$REQS_FILE" ]]; then
  log "Installing requirements"
  pip install -r "$REQS_FILE"
fi

# =========================
# Universe
# =========================
next "Build/verify trading universe"
if [[ -f data/nextday_universe_valid.csv ]]; then
  log "Universe already present: data/nextday_universe_valid.csv"
else
  if [[ -f universe_ax.txt ]]; then
    log "Building universe from universe_ax.txt"
    awk 'NF{print toupper($0)}' universe_ax.txt \
      | sed -E 's/[[:space:]]+//g' \
      | sed -E 's/\.AX$//; s/\.ASX$//' \
      | awk 'BEGIN{print "ticker"} {print}' \
      > data/nextday_universe_valid.csv
  else
    log "No universe file found; deriving from cached OHLC files"
    ls -1 cache_*_ohlc.csv 2>/dev/null \
      | sed -E 's/^cache_([A-Z0-9]+)\.AX_ohlc\.csv$/\1/' \
      | awk 'BEGIN{print "ticker"} {print}' \
      > data/nextday_universe_valid.csv || true
  fi
fi
log "Universe size: $(tail -n +2 data/nextday_universe_valid.csv | wc -l | tr -d ' ')"

# Ensure an events.csv exists so downstream never breaks
if [[ ! -f data/events.csv ]]; then
  log "Creating empty events.csv"
  echo "ticker,headline,source,ts" > data/events.csv
fi

# =========================
# Run ML swing model (main.py)
# =========================
next "Run ML swing model (main.py) + convert with headlines"
if [[ -f main.py ]]; then
  python main.py || true
  python - <<'PY'
import os, glob, pandas as pd

OUT_REPORT = "artifacts/nextday_report.csv"
NEWS_FILE_CANDIDATES = ["data/events.csv","data/events_api.csv","data/events_rss.csv","artifacts/news_events.csv"]
NEWS_WINDOW_HOURS = int(os.getenv("NEWS_WINDOW_HOURS", "96"))
TOP_SWING = int(os.getenv("TOP_SWING", "12"))

def norm_tk(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.upper()
            .str.replace(r"\.AX$|\.ASX$", "", regex=True)
            .str.replace(r"[^0-9A-Z]+", "", regex=True))

# 1) find latest trade_plan*.csv
candidates = sorted(glob.glob("out/trade_plan*.csv"))
if not candidates:
    pd.DataFrame(columns=["ticker","label","prob_pct","exp_move_pct","side","entry","tp","sl","headline","source"]).to_csv(OUT_REPORT, index=False)
    print(f"[swing] no trade_plan files; wrote empty {OUT_REPORT}")
    raise SystemExit(0)

plan_path = candidates[-1]
df = pd.read_csv(plan_path)

# 2) normalize to standard report schema
out = pd.DataFrame({
    "ticker": df.get("Ticker", df.get("ticker","")),
    "label": "bullish",
    "prob_pct": (df.get("MLProb", 0.0)*100).round(1),
    "exp_move_pct": None,
    "side": "long",
    "entry": df.get("BuyPrice", df.get("Close", 0.0)),
    "tp": df.get("Target1", None),
    "sl": df.get("Stop", None),
    "headline": "Model swing pick",
    "source": "model",
})
out["ticker"] = norm_tk(out["ticker"])

# 3) enrich headline per ticker if any recent news is available
news = None
for p in NEWS_FILE_CANDIDATES:
    if os.path.exists(p):
        try:
            cand = pd.read_csv(p)
            if "ticker" in cand.columns and "headline" in cand.columns:
                news = cand
                break
        except Exception:
            pass

if news is not None and not news.empty:
    if "ticker" not in news.columns:
        for c in ["symbol","Symbol","Ticker","code","Code"]:
            if c in news.columns:
                news["ticker"] = news[c]; break
    news["ticker"] = norm_tk(news["ticker"])
    if "ts" in news.columns:
        news["ts"] = pd.to_datetime(news["ts"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=NEWS_WINDOW_HOURS)
        news = news[news["ts"].notna() & (news["ts"] >= cutoff)]
        news = news.sort_values("ts", ascending=False)
    news = news.dropna(subset=["headline"]).drop_duplicates("ticker", keep="first")
    news_map = dict(zip(news["ticker"], news["headline"].astype(str)))
    out["headline"] = out["ticker"].map(news_map).fillna(out["headline"])
    out.loc[out["ticker"].isin(news_map.keys()), "source"] = "model+rss"

# 4) save
out.to_csv(OUT_REPORT, index=False)
print(f"[swing] wrote -> {OUT_REPORT} rows:", len(out))

# 5) pretty print top
cols = ["ticker","entry","tp","sl","prob_pct","headline"]
show = out[cols].head(TOP_SWING).copy()
f2 = lambda v: (f"{float(v):.2f}" if pd.notna(v) else "")
f1 = lambda v: (f"{float(v):.1f}" if pd.notna(v) else "")
def trim(s, n=90):
    s = "" if pd.isna(s) else str(s).replace("\n"," ").strip()
    return s if len(s)<=n else s[:n-1]+"…"
show["entry"]=show["entry"].map(f2); show["tp"]=show["tp"].map(f2); show["sl"]=show["sl"].map(f2)
show["prob_pct"]=show["prob_pct"].map(f1); show["headline"]=show["headline"].map(trim)
print("\n=== Trade Plan (topN) ===")
print(show.rename(columns={"ticker":"Ticker","entry":"BuyPrice","tp":"Target1","sl":"Stop","prob_pct":"Prob%","headline":"Headline"}).to_string(index=False))
PY
else
  log "No main.py found; writing empty swing report"
  echo "ticker,label,prob_pct,exp_move_pct,side,entry,tp,sl,headline,source" > artifacts/nextday_report.csv
fi

# =========================
# Microcap scanner
# =========================
next "Scan microcaps (daily preferred, else intraday)"
MICRO_OUT="artifacts/microcap_candidates.csv"
if [[ -f analysis/microcap_spike_scanner_daily.py ]]; then
  python analysis/microcap_spike_scanner_daily.py \
    --universe data/nextday_universe_valid.csv \
    --prices_dir data/prices_daily \
    --caps data/universe_caps.csv \
    --events_csv data/events.csv \
    --min_price 0.02 --max_price 50 \
    --max_cap_m 5000 \
    --min_gap 0.02 \
    --top "${TOP_MICRO}" \
    --out_csv "${MICRO_OUT}" || true
elif [[ -f analysis/microcap_spike_scanner.py ]]; then
  python analysis/microcap_spike_scanner.py \
    --universe data/nextday_universe_valid.csv \
    --prices_dir data/prices \
    --caps data/universe_caps.csv \
    --events_csv data/events.csv \
    --min_price 0.02 --max_price 10.0 \
    --max_cap_m 2000 \
    --min_relvol 1.2 \
    --min_gap 0.03 \
    --min_dollar_vol 50000 \
    --top "${TOP_MICRO}" \
    --out_csv "${MICRO_OUT}" || true
else
  log "No microcap scanner found; writing empty microcap file"
  echo "ticker,price,gap_%,rel_vol,dollar_vol,market_cap_m,has_news,entry,tp,sl,score" > "${MICRO_OUT}"
fi

# =========================
# Combine (adds bands later)
# =========================
next "Combine swing + microcap into nextday_combined.csv"
if [[ -f analysis/combine_reports.py ]]; then
  python analysis/combine_reports.py || true
else
  python - <<'PY'
import pandas as pd, os
s="artifacts/nextday_report.csv"; m="artifacts/microcap_candidates.csv"
dfs=[]
if os.path.exists(s):
    a=pd.read_csv(s); a.insert(0,"group","Daily Swing (Large/Mid)"); dfs.append(a)
if os.path.exists(m):
    b=pd.read_csv(m)
    if not b.empty:
        for c in ["entry","tp","sl"]:
            if c not in b.columns: b[c]=None
        b = b.rename(columns={"gap_%":"exp_move_pct"})
        b["label"]="momentum"; b["prob_pct"]=None; b["side"]="long"
        b["headline"]="Momentum (no news)"
        b["source"]="microcap_scanner"
        b.insert(0,"group","Intraday Spikes (Microcap)")
        b=b[["group","ticker","label","prob_pct","exp_move_pct","side","entry","tp","sl","headline","source"]]
        dfs.append(b)
df=pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["group","ticker","label","prob_pct","exp_move_pct","side","entry","tp","sl","headline","source"])
df.to_csv("artifacts/nextday_combined.csv", index=False)
print("[combine] wrote -> artifacts/nextday_combined.csv (rows=",len(df),")")
PY
fi

# =========================
# Caps for combined tickers
# =========================
next "Fetch market caps for combined tickers"
if [[ -f artifacts/nextday_combined.csv ]]; then
  python - <<'PY'
import pandas as pd
df=pd.read_csv("artifacts/nextday_combined.csv")
pd.DataFrame({"ticker":sorted(set(df["ticker"].astype(str)))})\
  .to_csv("artifacts/combined_tickers.csv", index=False)
print("tickers:", len(set(df["ticker"].astype(str))))
PY

  if [[ -f analysis/fetch_market_caps.py ]]; then
    python analysis/fetch_market_caps.py \
      --universe_csv artifacts/combined_tickers.csv \
      --out_csv data/universe_caps.csv \
      --cache data/fundamentals_cache.json \
      --workers 8 --max 10000 || true
  else
    log "No fetch_market_caps.py — caps may be missing (band will be Unclassified)"
  fi

  next "Merge caps & add cap bands"
  python - <<'PY'
import pandas as pd, os
comb="artifacts/nextday_combined.csv"
df=pd.read_csv(comb)

def norm(s):
    return (s.astype(str).str.upper()
            .str.replace(r"\.AX$|\.ASX$","",regex=True)
            .str.replace(r"[^0-9A-Z]+","",regex=True))
df["ticker"]=norm(df["ticker"])

for c in ("market_cap_m","sector","cap_band"):
    if c in df.columns:
        df.drop(columns=[c], inplace=True)

if os.path.exists("data/universe_caps.csv"):
    caps=pd.read_csv("data/universe_caps.csv")
    if "ticker" not in caps.columns:
        for c in ["Ticker","symbol","Symbol","code","Code"]:
            if c in caps.columns:
                caps["ticker"]=caps[c]; break
    caps["ticker"]=norm(caps["ticker"])

    cap_col=None
    for c in ["market_cap_m","market_cap","marketCap","MarketCap","mktcap","cap_m",
              "market_capitalization","market_capitalisation","MarketCapitalisation"]:
        if c in caps.columns: cap_col=c; break
    if cap_col:
        v=pd.to_numeric(caps[cap_col], errors="coerce")
        med=v.dropna().median()
        if pd.notna(med) and med>1e6: v=v/1_000_000.0
        caps["market_cap_m"]=v
    else:
        caps["market_cap_m"]=pd.NA

    sec=None
    for c in ["sector","Sector","industry","Industry","GICS_Sector","GICS Sector"]:
        if c in caps.columns: sec=c; break
    caps["sector"]=caps[sec] if sec else pd.NA

    caps=caps[["ticker","market_cap_m","sector"]].drop_duplicates("ticker")
    df=df.merge(caps, on="ticker", how="left")

def band(x):
    try: x=float(x)
    except: return "Unclassified"
    if x>=5000: return "Large-cap"
    if x>=500:  return "Mid-cap"
    return "Micro-cap"

if "market_cap_m" not in df.columns:
    df["market_cap_m"]=pd.NA
df["cap_band"]=df["market_cap_m"].map(band)
df.to_csv(comb, index=False)
print("[OK] wrote caps & bands -> artifacts/nextday_combined.csv")
print(df.head(8).to_string(index=False))
PY
fi

# =========================
# News: RSS + API, merge into combined
# =========================
next "Fetch news (RSS/API) and merge headlines"
if [[ -f artifacts/combined_tickers.csv && -f artifacts/nextday_combined.csv ]]; then
  [[ -f analysis/fetch_news_rss.py ]] && python analysis/fetch_news_rss.py --tickers artifacts/combined_tickers.csv --out_csv data/events_rss.csv --hours "${NEWS_WINDOW_HOURS}" || true
  [[ -f analysis/fetch_news_api.py ]] && python analysis/fetch_news_api.py --tickers artifacts/combined_tickers.csv --out_csv data/events_api.csv --hours "${NEWS_WINDOW_HOURS}" || true

  python - <<'PY'
import os, pandas as pd
outs=[]
for p in ("data/events_api.csv","data/events_rss.csv"):
    if os.path.exists(p):
        try: outs.append(pd.read_csv(p))
        except: pass
if outs:
    df=pd.concat(outs, ignore_index=True).drop_duplicates()
    df.to_csv("data/events.csv", index=False)
    print("merged -> data/events.csv rows=", len(df))
else:
    if not os.path.exists("data/events.csv"):
        pd.DataFrame(columns=["ticker","headline","source","ts"]).to_csv("data/events.csv", index=False)
        print("wrote empty data/events.csv")
PY

  if [[ -f analysis/add_news_to_combined.py ]]; then
    python analysis/add_news_to_combined.py --combined artifacts/nextday_combined.csv --events data/events.csv || true
  fi
fi

# =========================
# Optional: news-based recommender stage
# =========================
if [[ -f analysis/news_recommender.py ]]; then
  next "Generate final trade plan with news"
  python analysis/news_recommender.py || true
fi

# =========================
# Final Combined Table (Cap Band + News)
# =========================
next "Show final combined table (cap bands + news)"
python - <<'PY'
import pandas as pd

comb = "artifacts/nextday_combined.csv"
try:
    df = pd.read_csv(comb)
except FileNotFoundError:
    print(f"[WARN] {comb} not found")
    raise SystemExit(0)

cols = [c for c in ["ticker","cap_band","prob_pct","entry","tp","sl","headline"] if c in df.columns]
def fmt(v):
    if pd.isna(v): return ""
    s = str(v).strip()
    return s[:90] + "…" if len(s) > 90 else s

for c in cols:
    df[c] = df[c].map(fmt)

print("\n=== Final Combined Picks (with Cap Bands & News) ===")
print(df[cols].head(30).to_string(index=False))
PY

# =========================
# Summary
# =========================
next "Summary & output"
ls -lh artifacts/nextday_report.csv 2>/dev/null || true
ls -lh artifacts/microcap_candidates.csv 2>/dev/null || true
ls -lh artifacts/nextday_combined.csv 2>/dev/null || true
log "Done."