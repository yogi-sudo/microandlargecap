
# ASX Twin‑Track Pipeline — FULL Command Guide

_Last updated: 2025-08-27 16:49 UTC_

This doc contains **every command** to operate the project: setup, daily runs, individual steps, microcap spikes, combining reports, and pushing to GitHub.

---

## 0) One‑time Project Setup

```bash
# clone
git clone git@github.com:yogi-sudo/microandlargecap.git
cd microandlargecap

# (optional) set your git identity for this repo
git config user.name "yogi-sudo"
git config user.email "28231757+yogi-sudo@users.noreply.github.com"

# Python venv
python3 -m venv .venv
source .venv/bin/activate

# install deps
pip install --upgrade pip wheel
pip install -r requirements.txt

# Git LFS (only once per machine; skip if already done)
git lfs install
```

### Environment
Copy `.env.example` to `.env` and fill keys:
```
EODHD_API_KEY=...
NEWSAPI_KEY=...           # optional (sentiment & headlines)
OPENAI_API_KEY=...        # optional (GPT sentiment)
OPENAI_MODEL=gpt-4o-mini  # or gpt-4o-mini, etc.
TIMEZONE=Australia/Sydney
```

---

## 1) Run EVERYTHING in one go (recommended)

```bash
# from repo root
chmod +x run_all.sh
./run_all.sh
```

**What it does:**
1) Builds/refreshes universe & market caps  
2) Runs daily swing ML and writes `out/trade_plan_*.csv`  
3) Normalizes into `artifacts/nextday_report.csv`  
4) Scans microcap gaps (daily) → `artifacts/microcap_candidates.csv`  
5) Combines both → `artifacts/nextday_combined.csv` (adds `market_cap_m` + `cap_band`)  

---

## 2) If you prefer step‑by‑step

```bash
# (A) Create / refresh universe list (broad)
python3 - <<'PY'
import pandas as pd, os
os.makedirs("data", exist_ok=True)
# edit this list to your preferred seeds or import from a file
tickers = ["CBA","BHP","RIO","FMG","CSL","ANZ","WBC","NAB","WES","WDS","REA","ZIP","PNV"]
pd.DataFrame({"ticker": tickers}).to_csv("data/nextday_universe.csv", index=False)
print("Universe -> data/nextday_universe.csv (", len(tickers), ")")
PY

# (B) Fetch market caps for the current universe
python3 analysis/fetch_market_caps.py \
  --universe_csv data/nextday_universe.csv \
  --out_csv data/universe_caps.csv \
  --cache data/fundamentals_cache.json \
  --workers 8 --max 10000

# (C) Run the daily swing ML (produces plan under out/)
PYTHONPATH="." python main.py

# (D) Normalize ML plan into nextday_report.csv
python3 - <<'PY'
import os, glob, pandas as pd
os.makedirs("artifacts", exist_ok=True)
paths = sorted(glob.glob("out/trade_plan_*.csv"))
if not paths: raise SystemExit("No trade_plan_*.csv found in out/")
src = paths[-1]
df = pd.read_csv(src)
df = df.rename(columns={
    "Ticker":"ticker","Close":"entry","Score":"prob_pct","MLProb":"ml_prob","Sentiment":"sentiment"
})
df["label"] = "bullish"
df["tp"] = (df["entry"] * 1.03).round(2)
df["sl"] = (df["entry"] * 0.97).round(2)
df["exp_move_pct"] = None
df["rel_vol"] = None
df["dollar_vol"] = None
df["headline"] = "Model swing pick"
out_cols = ["ticker","label","prob_pct","exp_move_pct","entry","tp","sl","rel_vol","dollar_vol","headline"]
df[out_cols].to_csv("artifacts/nextday_report.csv", index=False)
print("Wrote artifacts/nextday_report.csv from", src, "rows:", len(df))
PY

# (E) Microcap gap scanner (daily bars) → adjust thresholds as desired
python3 analysis/microcap_spike_scanner_daily.py \
  --universe data/nextday_universe_valid.csv \
  --prices_dir data/prices_daily \
  --caps data/universe_caps.csv \
  --events_csv data/events.csv \
  --min_price 0.02 --max_price 50 \
  --max_cap_m 5000 \
  --min_gap 0.02 \
  --top 50 \
  --out_csv artifacts/microcap_candidates.csv

# (F) Combine swing + microcap into one CSV with caps & bands
python3 analysis/combine_reports.py
```

The combined file is at:
```
artifacts/nextday_combined.csv
```
Columns include: `group, cap_band, ticker, label, prob_pct, exp_move_pct, entry, tp, sl, market_cap_m, sector, headline, source, rel_vol, dollar_vol`

---

## 3) Useful “refresh” commands

```bash
# Rebuild universe from combined picks
python3 - <<'PY'
import pandas as pd, os
df = pd.read_csv("artifacts/nextday_combined.csv")
ticks = sorted(set(df["ticker"].astype(str)))
os.makedirs("artifacts", exist_ok=True)
pd.DataFrame({"ticker":ticks}).to_csv("artifacts/combined_tickers.csv", index=False)
print("tickers:", len(ticks))
PY

# Fetch caps only for combined tickers
python3 analysis/fetch_market_caps.py \
  --universe_csv artifacts/combined_tickers.csv \
  --out_csv data/universe_caps.csv \
  --cache data/fundamentals_cache.json \
  --workers 8 --max 10000
```

---

## 4) Optional: Sentiment runtime (NewsAPI + GPT)

```bash
# run sentiment routine for specific tickers (optional)
python3 analysis/news_sentiment_runtime.py --tickers CSL,BHP,RIO,FMG --verbose
```

It writes/updates: `out/news_sentiment.csv` and caches JSON under `out/sent_cache/`

---

## 5) Git commands (SSH remote)

```bash
# ensure SSH works
ssh -T git@github.com

# set remote (SSH)
git remote set-url origin git@github.com:yogi-sudo/microandlargecap.git
git remote -v

# commit and push
git add -A
git commit -m "Update data & artifacts"
git push -u origin main
```

If you see `Permission denied (publickey)`, follow the SSH tips at the bottom of this doc.

---

## 6) Microcap Intraday (1h) Scanner (optional)

```bash
python3 analysis/microcap_spike_scanner.py \
  --universe data/nextday_universe_valid.csv \
  --prices_dir data/prices \
  --caps data/universe_caps.csv \
  --events_csv data/events.csv \
  --min_price 0.02 --max_price 100 \
  --max_cap_m 10000 \
  --min_relvol 1.0 \
  --min_gap 0.01 \
  --min_dollar_vol 10000 \
  --top 100 \
  --out_csv artifacts/microcap_candidates.csv
```

---

## 7) SSH Key Quick‑fix

```bash
# create a new ed25519 key (no password, press Enter)
ssh-keygen -t ed25519 -C "your_email@users.noreply.github.com"

# start agent and add key
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519 2>/dev/null || ssh-add -K ~/.ssh/id_ed25519

# copy the public key to clipboard (macOS)
pbcopy < ~/.ssh/id_ed25519.pub

# now add that key in GitHub → Settings → SSH and GPG keys → "New SSH key"
# test
ssh -T git@github.com
```

---

## Outputs to look at

- `out/trade_plan_*.csv` — raw daily swing ML picks  
- `artifacts/nextday_report.csv` — normalized swing report  
- `artifacts/microcap_candidates.csv` — microcap gap list  
- `artifacts/nextday_combined.csv` — final merged output (with market_cap_m + cap_band)

---

## Troubleshooting

- _“No microcap spikes passing filters”_ → lower `--min_gap` to `0.02` and `--min_price` to `0.02`.  
- _“Permission denied (publickey)”_ → re-add SSH key steps above.  
- _Missing caps_ → run the caps fetch step against `artifacts/combined_tickers.csv`.
