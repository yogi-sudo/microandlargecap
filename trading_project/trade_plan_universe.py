#!/usr/bin/env python3
import os, sys, json, datetime as dt, warnings, requests
import pandas as pd, numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Optional: yfinance fallback
try:
    import yfinance as yf
except Exception:
    yf = None

TODAY = dt.date.today()
OUT_DIR, CACHE_DIR = "out", "cache"
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)

# Env knobs
EODHD_KEY = os.getenv("EODHD_API_KEY")           # optional
EQUITY     = float(os.getenv("EQUITY", 100000))  # account equity
RISK_PCT   = float(os.getenv("RISK_PCT", 0.01))  # risk per trade (1%)
ATR_SL     = float(os.getenv("ATR_SL", 2.0))     # stop: 2x ATR
ATR_TP     = float(os.getenv("ATR_TP", 3.0))     # take profit: 3x ATR
TRAIN_YEARS= int(os.getenv("TRAIN_YEARS", 3))
HOLDOUT_D  = int(os.getenv("HOLDOUT_DAYS", 60))
MIN_ROWS   = int(os.getenv("MIN_ROWS", 150))     # require this much history

UNIVERSE_FILE = "universe_ax.txt"                # one .AX ticker per line

def load_universe():
    if not os.path.exists(UNIVERSE_FILE):
        sys.exit(f"Universe file not found: {UNIVERSE_FILE}. Create it with .AX tickers (one per line).")
    tickers=[]
    with open(UNIVERSE_FILE,"r") as f:
        for line in f:
            s=line.strip()
            if not s or s.startswith("#"): continue
            # normalize: ensure .AX suffix only once
            s = s.upper()
            if not s.endswith(".AX"): s += ".AX"
            tickers.append(s)
    if not tickers:
        sys.exit("Universe file is empty.")
    return tickers

def daterange_start(years=3):
    return (TODAY - dt.timedelta(days=365*years)).isoformat()

def fetch_ohlcv_ax(ticker_ax: str, years=3):
    """
    Cached OHLCV for `TICKER.AX`. Try cache -> EODHD (if key) -> yfinance.
    Returns DataFrame with columns: date,open,high,low,close,volume
    """
    assert ticker_ax.endswith(".AX")
    base = ticker_ax[:-3]  # 'CBA' from 'CBA.AX'
    cache_file = f"{CACHE_DIR}/{ticker_ax}_ohlc.csv".replace("/", "_")
    # cache
    if os.path.exists(cache_file):
        try:
            d = pd.read_csv(cache_file, parse_dates=["date"])
            need = {"date","open","high","low","close","volume"}
            if not d.empty and need.issubset(d.columns):
                return d
        except Exception:
            pass

    start = daterange_start(years)

    # EODHD (optional)
    if EODHD_KEY:
        try:
            url = f"https://eodhd.com/api/eod/{base}.AU"
            params = {"api_token":EODHD_KEY,"fmt":"json","period":"d","from":start}
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 200:
                d = pd.DataFrame(r.json())
                if not d.empty and {"date","open","high","low","close","volume"}.issubset(d.columns):
                    d["date"] = pd.to_datetime(d["date"])
                    d = d[["date","open","high","low","close","volume"]]
                    d.to_csv(cache_file, index=False)
                    return d
        except Exception:
            pass

    # yfinance fallback
    if yf is not None:
        try:
            d = yf.download(ticker_ax, start=start, end=(TODAY+dt.timedelta(days=1)).isoformat(), progress=False)
            if not d.empty:
                d = d.reset_index()[["Date","Open","High","Low","Close","Volume"]]
                d.columns = ["date","open","high","low","close","volume"]
                d.to_csv(cache_file, index=False)
                return d
        except Exception:
            pass

    return None

def atr(df, n=14):
    h,l,c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def compute_features(df):
    p = df.copy()
    # some pandas/yf versions can hand back DataFrame columns; flatten to Series
    for col in ["open","high","low","close","volume"]:
        if isinstance(p[col], pd.DataFrame):
            p[col] = p[col].iloc[:,0]
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p = p.dropna(subset=["open","high","low","close","volume"]).copy()

    p["ret1"]  = p["close"].pct_change()
    p["ret5"]  = p["close"].pct_change(5)
    p["ma20"]  = p["close"].rolling(20).mean()
    p["vol20"] = p["close"].pct_change().rolling(20).std()
    p["atr14"] = atr(p,14)
    p["y_next_up"] = (p["close"].shift(-1) > p["close"]).astype(int)
    p = p.dropna().reset_index(drop=True)
    return p

def cheap_sentiment(_ticker_ax:str)->float:
    """Placeholder 0.5 until you wire NewsAPI/LLM. Keeps pipeline stable."""
    return 0.5

def get_model():
    # Try XGBoost first, fallback to HistGB if not installed
    try:
        import xgboost as xgb
        return ("xgb", xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="auc", n_jobs=-1
        ))
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return ("hgb", HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.06, max_iter=400, validation_fraction=None
        ))

def main():
    tickers = load_universe()
    print(f"Universe: {len(tickers)} tickers from {UNIVERSE_FILE}")

    frames=[]
    for t in tqdm(tickers, desc="OHLCV"):
        d = fetch_ohlcv_ax(t, years=TRAIN_YEARS)
        if d is None or len(d) < MIN_ROWS:
            continue
        f = compute_features(d)
        if f.empty: 
            continue
        f["ticker"] = t
        frames.append(f)

    if not frames:
        sys.exit("No usable histories — check your universe and cache/API.")

    data = pd.concat(frames).sort_values(["ticker","date"])
    feat_cols = ["close","ret1","ret5","ma20","vol20","atr14"]

    cutoff = data["date"].max() - dt.timedelta(days=HOLDOUT_D)
    train = data[data["date"]<=cutoff]
    test  = data[data["date"]> cutoff]
    if train.empty or test.empty:
        sys.exit("Not enough data to train/test; reduce HOLDOUT_D or increase TRAIN_YEARS/MIN_ROWS.")

    model_name, model = get_model()
    Xtr, ytr = train[feat_cols], train["y_next_up"]
    Xte, yte = test[feat_cols],  test["y_next_up"]
    model.fit(Xtr, ytr)

    # eval
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score
        if hasattr(model, "predict_proba"):
            pte = model.predict_proba(Xte)[:,1]
        else:
            from scipy.special import expit
            pte = expit(getattr(model,"decision_function")(Xte))
        auc = roc_auc_score(yte, pte)
        acc = accuracy_score(yte, (pte>=0.5).astype(int))
        print(f"Holdout → AUC={auc:.3f}  Acc={acc:.3f}  (model={model_name})")
    except Exception as e:
        print(f"Holdout metrics unavailable: {e}")

    # live scores
    latest = data.groupby("ticker").tail(1).copy()
    if hasattr(model, "predict_proba"):
        latest["ml_prob"] = model.predict_proba(latest[feat_cols])[:,1]
    else:
        from scipy.special import expit
        latest["ml_prob"] = expit(getattr(model,"decision_function")(latest[feat_cols]))

    latest["sentiment"] = latest["ticker"].apply(cheap_sentiment)
    latest["blended"]   = latest["ml_prob"]*0.7 + latest["sentiment"]*0.3

    # position sizing via ATR
    plans=[]
    for _,r in latest.iterrows():
        close = float(r["close"]); atr14 = float(r["atr14"])
        if not np.isfinite(atr14) or atr14<=0: 
            atr14 = max(0.01, np.nanstd(latest["close"].values[-20:]))
        sl = max(0.01, close - ATR_SL*atr14)
        tp = close + ATR_TP*atr14
        risk_per_share = max(0.01, close - sl)
        size = int((EQUITY*RISK_PCT)/risk_per_share)
        plans.append({
            "Ticker":        r["ticker"],
            "Date":          r["date"].date(),
            "Close":         round(close,2),
            "ML_Prob":       round(float(r["ml_prob"]),3),
            "Sentiment":     round(float(r["sentiment"]),2),
            "Blended":       round(float(r["blended"]),3),
            "StopLoss":      round(sl,2),
            "TakeProfit":    round(tp,2),
            "Risk/Share":    round(risk_per_share,2),
            "PositionSize":  max(size,0)
        })

    plan = pd.DataFrame(plans).sort_values(["Blended","ML_Prob"], ascending=False)

    # If you want to clip by market cap tiers later, you can keep separate lists/files.
    out_path = f"{OUT_DIR}/trade_plan_{TODAY.isoformat()}.csv"
    plan.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("\nTop 15 preview:")
    print(plan.head(15).to_string(index=False,
          formatters={"ML_Prob":lambda x:f"{x:.3f}","Blended":lambda x:f"{x:.3f}","Close":lambda x:f"{x:.2f}"}))

if __name__ == "__main__":
    main()
