#!/usr/bin/env python3
import os, datetime as dt, warnings
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
import yfinance as yf
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# ----------------------- Config -----------------------
TODAY = dt.date.today()

TRAIN_YEARS   = int(os.getenv("TRAIN_YEARS", 3))
MIN_ROWS      = int(os.getenv("MIN_ROWS", 150))
MIN_PRICE     = float(os.getenv("MIN_PRICE", 0.2))
W_MIN_VOL     = float(os.getenv("W_MIN_VOL", 10000))
UNIVERSE_MAX  = int(os.getenv("UNIVERSE_MAX", 200))
TOPN          = int(os.getenv("TOPN", 10))
THRESH_PROB   = float(os.getenv("THRESH_PROB", 0.55))
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", 30))

CAPITAL       = float(os.getenv("CAPITAL", 3000))
PER_TRADE     = float(os.getenv("PER_TRADE", 300))

CACHE_DIR = "cache"
OUT_DIR   = "out"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "picks_history"), exist_ok=True)

# ----------------------- Universe -----------------------
FALLBACK_ASX = [
    "CBA.AX","BHP.AX","CSL.AX","NAB.AX","WBC.AX","ANZ.AX","WES.AX","MQG.AX","GMG.AX","FMG.AX",
    "TLS.AX","WDS.AX","TCL.AX","ALL.AX","RIO.AX","WOW.AX","WTC.AX","BXB.AX","REA.AX","SIG.AX"
]

def load_universe() -> List[str]:
    paths = [
        os.path.join(OUT_DIR, "universe_all_clean.ax.csv"),
        os.path.join(OUT_DIR, "tier_combined.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            for col in ["Ticker","ticker","Code","symbol"]:
                if col in df.columns:
                    vals = df[col].astype(str).str.strip()
                    vals = np.where(vals.str.endswith(".AX"), vals, vals + ".AX")
                    uniq = sorted(pd.Series(vals).unique().tolist())
                    return uniq[:UNIVERSE_MAX] if UNIVERSE_MAX > 0 else uniq
    return FALLBACK_ASX[:UNIVERSE_MAX] if UNIVERSE_MAX > 0 else FALLBACK_ASX

# ----------------------- Data -----------------------
def cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_ohlc.csv")

def _safe_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            s = df[c]
            if isinstance(s, (pd.Series,)):
                df[c] = pd.to_numeric(s, errors="coerce")
    return df

def fetch_prices(ticker: str, years: int = TRAIN_YEARS) -> pd.DataFrame:
    """Return columns: date, open, high, low, close, volume (numeric), or empty if bad."""
    fn = cache_path(ticker)

    # Try cache first
    if os.path.exists(fn):
        try:
            df = pd.read_csv(fn, parse_dates=["date"])
            # Make sure expected columns are present
            need = {"date","open","high","low","close","volume"}
            if need.issubset(df.columns):
                df = _safe_to_numeric(df, ["open","high","low","close","volume"])
                df = df.dropna(subset=["date","close","volume"])
                if not df.empty:
                    return df.sort_values("date")
        except Exception:
            pass  # fall back to download

    # Fresh download
    start = TODAY - dt.timedelta(days=365*years + 7)
    end   = TODAY + dt.timedelta(days=1)
    try:
        y = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            group_by="column",
            auto_adjust=True,  # yfinance recent default anyway; keeps it explicit
        )
    except Exception:
        return pd.DataFrame()

    if y is None or len(y) == 0:
        return pd.DataFrame()

    # Some tickers may come multi-index or missing cols; normalize
    if isinstance(y.columns, pd.MultiIndex):
        y = y.copy()
        y.columns = [c[0] for c in y.columns]

    # Required columns present?
    needed = ["Open","High","Low","Close","Volume"]
    if not all(col in y.columns for col in needed):
        return pd.DataFrame()

    y = y.reset_index().rename(columns={
        "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
    })

    keep = ["date","open","high","low","close","volume"]
    y = y[keep].copy()
    y = _safe_to_numeric(y, ["open","high","low","close","volume"])
    y = y.dropna(subset=["date","close","volume"])
    if y.empty:
        return pd.DataFrame()

    try:
        y.to_csv(fn, index=False)
    except Exception:
        pass

    return y.sort_values("date")

# ----------------------- Features -----------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    p = df.copy()
    p = _safe_to_numeric(p, ["open","high","low","close","volume"])
    p = p.dropna(subset=["close","volume"])
    if p.empty:
        return p

    p["ret1"]  = p["close"].pct_change(1)
    p["ret5"]  = p["close"].pct_change(5)
    p["ma5"]   = p["close"].rolling(5).mean()
    p["ma10"]  = p["close"].rolling(10).mean()
    p["ma20"]  = p["close"].rolling(20).mean()
    p["std20"] = p["close"].rolling(20).std()
    p["vol20"] = p["close"].pct_change().rolling(20).std()

    # RSI(14)
    delta = p["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up, roll_down = up.rolling(14).mean(), down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    p["rsi14"] = 100 - 100/(1 + rs)

    # targets
    p["next_close"] = p["close"].shift(-1)
    p["y"] = (p["next_close"] > p["close"]).astype(int)
    p["v20"] = p["volume"].rolling(20).mean()

    return p.dropna().reset_index(drop=True)

def build_dataset(tickers: List[str]) -> pd.DataFrame:
    frames = []
    print(f"Fetching {len(tickers)} tickers...")
    for t in tqdm(tickers, unit="stk"):
        px = fetch_prices(t)
        if px.empty or len(px) < MIN_ROWS:
            continue
        # latest close/volume checks must be numeric by now
        last_close = float(px["close"].iloc[-1])
        if last_close < MIN_PRICE:
            continue
        v20 = px["volume"].rolling(20).mean().iloc[-1]
        if pd.isna(v20) or v20 < W_MIN_VOL:
            continue

        f = add_features(px)
        if f.empty:
            continue
        f["ticker"] = t
        frames.append(f)

    if not frames:
        raise SystemExit("No usable histories after filters. Lower MIN_ROWS or W_MIN_VOL, or reduce UNIVERSE_MAX.")

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ticker","date"]).reset_index(drop=True)
    return data

# ----------------------- ML -----------------------
FEATS = ["close","ret1","ret5","ma5","ma10","ma20","std20","vol20","rsi14"]

def train_and_eval(data: pd.DataFrame):
    cutoff = data["date"].max() - pd.Timedelta(days=BACKTEST_DAYS + 5)
    train, test = data[data["date"] <= cutoff], data[data["date"] > cutoff]

    Xtr, ytr = train[FEATS].values, train["y"].values
    Xte, yte = test[FEATS].values,  test["y"].values

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", n_jobs=4, eval_metric="auc"
    )
    model.fit(Xtr, ytr)

    if len(Xte) > 0:
        prob = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, prob)
        acc = accuracy_score(yte, (prob >= 0.5).astype(int))
        cov = (prob >= THRESH_PROB).mean()
        hit = ((prob >= THRESH_PROB) & (yte == 1)).sum() / max(1, (prob >= THRESH_PROB).sum())
        print(f"Holdout: AUC={auc:.3f} Acc={acc:.3f} Coverage@{THRESH_PROB:.2f}={cov:.2%} Hit@Thr={hit:.2%}")
    else:
        print("Holdout: no test split (small dataset); trained on all.")

    return model, FEATS

# ----------------------- Tomorrow Picks -----------------------
def picks_for_tomorrow(model, feat_cols, data, topN=TOPN, capital=CAPITAL, per_trade=PER_TRADE):
    last_day = data["date"].max()
    block = data[data["date"] == last_day].copy()
    prob = model.predict_proba(block[feat_cols].values)[:, 1]
    block["prob"] = prob

    picks = block.sort_values("prob", ascending=False).head(topN).copy()
    picks["Qty"] = np.maximum(1, np.floor(per_trade / picks["close"]))
    picks["Capital"] = picks["Qty"] * picks["close"]

    show = picks[["ticker","date","close","prob","Qty","Capital"]].rename(
        columns={"ticker":"Ticker","date":"Date","close":"Close","prob":"ProbUp_T1"}
    ).sort_values("ProbUp_T1", ascending=False)

    print("\nTomorrow picks:")
    print(show.to_string(index=False, formatters={
        "Close":     lambda x: f"{x:.2f}",
        "ProbUp_T1": lambda x: f"{x:.3f}",
        "Capital":   lambda x: f"{x:.2f}",
    }))

    out_csv = os.path.join(OUT_DIR, f"ml_tomorrow_picks_{TODAY.isoformat()}.csv")
    show.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

# ----------------------- Main -----------------------
def main():
    tickers = load_universe()
    if UNIVERSE_MAX and len(tickers) > UNIVERSE_MAX:
        tickers = tickers[:UNIVERSE_MAX]
    print(f"Universe size: {len(tickers)}")

    data = build_dataset(tickers)
    print(f"Dataset rows: {len(data):,} | tickers: {data['ticker'].nunique()} | last: {pd.to_datetime(data['date']).max().date()}")

    model, feat_cols = train_and_eval(data)
    picks_for_tomorrow(model, feat_cols, data)

if __name__ == "__main__":
    main()
