#!/usr/bin/env bash
set -euo pipefail

# ---- Config (tweak if you like) --------------------------------------------
NEWS_WINDOW_HOURS="${NEWS_WINDOW_HOURS:-96}"
TOP_SWING="${TOP_SWING:-12}"                # how many swing picks to print
TOP_MICRO="${TOP_MICRO:-50}"                # how many microcap spikes to keep
PYTHON_BIN="${PYTHON_BIN:-python3}"         # or "python"
REQS_FILE="requirements.txt"

# ---- Helpers ----------------------------------------------------------------
log() { printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }
exists() { command -v "$1" >/dev/null 2>&1; }
have() { [[ -f "$1" ]]; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---- Dirs -------------------------------------------------------------------
mkdir -p data data/prices data/prices_daily artifacts out signals models

# ---- Venv -------------------------------------------------------------------
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
export PYTHONUNBUFFERED=1
export PYTHONPATH="."

# ---- Universe (tries best source available) ---------------------------------
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

# ---- (Optional) News step (placeholder) -------------------------------------
if [[ ! -f data/events.csv ]]; then
  log "Creating empty events.csv"
  echo "ticker,headline,source,ts" > data/events.csv
fi

# ---- Run ML swing model (main.py) -------------------------------------------
if [[ -f main.py ]]; then
  log "Running ML swing model (main.py)"
  python main.py || true
  LAST_PLAN="$(ls -1 out/trade_plan*.csv 2>/dev/null | sort | tail -n1 || true)"
  if [[ -n "${LAST_PLAN}" && -f "${LAST_PLAN}" ]]; then
    log "Converting ${LAST_PLAN} -> artifacts/nextday_report.csv"
    python - <<'PY'
import os, pandas as pd, glob
out_dir="out"
arts="artifacts/nextday_report.csv"
files=sorted(glob.glob(os.path.join(out_dir,"trade_plan*.csv")))
if not files:
    print("[swing] no trade_plan*.csv; writing empty report")
    pd.DataFrame(columns=["ticker","label","prob_pct","exp_move_pct","side","entry","tp","sl","headline","source"]).to_csv(arts,index=False)
else:
    p=files[-1]
    df=pd.read_csv(p)
    out=pd.DataFrame({
        "ticker": df.get("Ticker", df.get("ticker","")),
        "label": "bullish",
        "prob_pct": (df.get("MLProb",0.0)*100).round(1),
        "exp_move_pct": None,
        "side": "long",
        "entry": df.get("BuyPrice", df.get("Close",0.0)),
        "tp": df.get("Target1", None),
        "sl": df.get("Stop", None),
        "headline": "Model swing pick",
        "source": "model",
    })
    out.to_csv(arts, index=False)
    print(f"[swing] wrote -> {arts} rows:", len(out))
PY
  else
    log "No trade plan found in out/, writing empty swing report"
    echo "ticker,label,prob_pct,exp_move_pct,side,entry,tp,sl,headline,source" > artifacts/nextday_report.csv
  fi
else
  log "No main.py found; writing empty swing report"
  echo "ticker,label,prob_pct,exp_move_pct,side,entry,tp,sl,headline,source" > artifacts/nextday_report.csv
fi

# ---- Microcap scanner (daily first; fall back to intraday) ------------------
MICRO_OUT="artifacts/microcap_candidates.csv"
if [[ -f analysis/microcap_spike_scanner_daily.py ]]; then
  log "Scanning microcaps (daily)"
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
  log "Scanning microcaps (intraday)"
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

# ---- Combine (adds bands if caps file present) ------------------------------
if [[ -f analysis/combine_reports.py ]]; then
  log "Combining reports"
  python analysis/combine_reports.py || true
else
  log "No combine_reports.py; creating naive combined file"
  python - <<'PY'
import pandas as pd, os
s="artifacts/nextday_report.csv"; m="artifacts/microcap_candidates.csv"
dfs=[]
if os.path.exists(s):
    a=pd.read_csv(s); a.insert(0,"group","Daily Swing (Large/Mid)"); dfs.append(a)
if os.path.exists(m):
    b=pd.read_csv(m); 
    if not b.empty:
        keep=["ticker","entry","tp","sl"]
        for c in keep:
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

# ---- Fetch caps for just combined tickers (if helper exists) ----------------
if [[ -f artifacts/nextday_combined.csv ]]; then
  log "Extracting combined tickers"
  python - <<'PY'
import pandas as pd, os
df=pd.read_csv("artifacts/nextday_combined.csv")
tick=sorted(set(df["ticker"].astype(str)))
pd.DataFrame({"ticker":tick}).to_csv("artifacts/combined_tickers.csv", index=False)
print("tickers:", len(tick))
PY

  if [[ -f analysis/fetch_market_caps.py ]]; then
    log "Fetching market caps for combined tickers"
    python analysis/fetch_market_caps.py \
      --universe_csv artifacts/combined_tickers.csv \
      --out_csv data/universe_caps.csv \
      --cache data/fundamentals_cache.json \
      --workers 8 --max 10000 || true
  else
    log "No fetch_market_caps.py — caps may be missing (band will be Unclassified)"
  fi

  # Add bands into combined file (robust to missing caps)
  log "Merging caps & adding cap bands"
  python - <<'PY'
import pandas as pd, os, re
comb="artifacts/nextday_combined.csv"
df=pd.read_csv(comb)
def norm(s):
    return (s.astype(str).str.upper()
            .str.replace(r"\.AX$|\.ASX$","",regex=True)
            .str.replace(r"[^0-9A-Z]+","",regex=True))
df["ticker"]=norm(df["ticker"])
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
if "market_cap_m" not in df.columns: df["market_cap_m"]=pd.NA
def band(x):
    try: x=float(x)
    except: return "Unclassified"
    if x>=5000: return "Large-cap"
    if x>=500:  return "Mid-cap"
    return "Micro-cap"
df["cap_band"]=df["market_cap_m"].map(band)
df.to_csv(comb, index=False)
print("[OK] wrote caps & bands -> artifacts/nextday_combined.csv")
print(df.head(8).to_string(index=False))
PY
fi

# ---- Summary ----------------------------------------------------------------
log "Done."
log "Files:"
ls -lh artifacts/nextday_report.csv 2>/dev/null || true
ls -lh artifacts/microcap_candidates.csv 2>/dev/null || true
ls -lh artifacts/nextday_combined.csv 2>/dev/null || true
