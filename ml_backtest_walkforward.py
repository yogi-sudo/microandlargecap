#!/usr/bin/env python3
import os, datetime as dt, warnings
import numpy as np, pandas as pd, yfinance as yf
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# --- CONFIG ---
TODAY = dt.date.today()
TRAIN_YEARS = int(os.getenv("TRAIN_YEARS", 3))
MIN_ROWS = int(os.getenv("MIN_ROWS", 150))
MIN_PRICE = float(os.getenv("MIN_PRICE", 0.20))
W_MIN_VOL = float(os.getenv("W_MIN_VOL", 10000))
UNIVERSE_MAX = int(os.getenv("UNIVERSE_MAX", 500))
TOPN = int(os.getenv("TOPN", 10))
PER_TRADE = float(os.getenv("PER_TRADE", 300))
STOP_PCT = float(os.getenv("STOP_PCT", 0.02))
TP_PCT = float(os.getenv("TP_PCT", 0.06))
BACK_DAYS = int(os.getenv("BACK_DAYS", 30))

CACHE_DIR, OUT_DIR = "cache", "out"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# --- UNIVERSE ---
FALLBACK_ASX = ["CBA.AX","BHP.AX","CSL.AX","NAB.AX","WBC.AX","ANZ.AX","MQG.AX","WOW.AX"]

def load_universe():
    for fn in ["out/universe_all_clean.ax.csv","out/tier_combined.csv"]:
        if os.path.exists(fn):
            df = pd.read_csv(fn)
            for c in ["Ticker","ticker","Code","symbol"]:
                if c in df.columns:
                    vals = df[c].astype(str).str.strip()
                    vals = np.where(vals.str.endswith(".AX"), vals, vals+".AX")
                    return sorted(set(vals))[:UNIVERSE_MAX]
    return FALLBACK_ASX

# --- DATA (with cache) ---
def cache_path(t): return os.path.join(CACHE_DIR,f"{t}_ohlc.csv")

def fetch_prices(t):
    fn = cache_path(t)
    if os.path.exists(fn):
        try:
            df = pd.read_csv(fn, parse_dates=["date"])
            if not df.empty and {"date","open","high","low","close","volume"}.issubset(df.columns):
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df.dropna().sort_values("date")
        except: pass
    start = TODAY - dt.timedelta(days=365*TRAIN_YEARS+7)
    end = TODAY+dt.timedelta(days=1)
    y = yf.download(t, start=start.isoformat(), end=end.isoformat(), progress=False)
    if y.empty: return pd.DataFrame()
    y = y.reset_index().rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    keep=["date","open","high","low","close","volume"]
    for c in keep: 
        if c!="date": y[c]=pd.to_numeric(y[c],errors="coerce")
    y = y.dropna(subset=["date","close","volume"])[keep]
    y.to_csv(fn,index=False)
    return y.sort_values("date")

# --- FEATURES ---
def add_features(df):
    p=df.copy()
    p["ret1"]=p["close"].pct_change(1)
    p["ret5"]=p["close"].pct_change(5)
    p["ma5"]=p["close"].rolling(5).mean()
    p["ma10"]=p["close"].rolling(10).mean()
    p["ma20"]=p["close"].rolling(20).mean()
    p["std20"]=p["close"].rolling(20).std()
    p["vol20"]=p["close"].pct_change().rolling(20).std()
    delta=p["close"].diff()
    up,down=delta.clip(lower=0),-delta.clip(upper=0)
    rs=up.rolling(14).mean()/(down.rolling(14).mean()+1e-9)
    p["rsi14"]=100-100/(1+rs)
    p["next_close"]=p["close"].shift(-1)
    p["y"]=(p["next_close"]>p["close"]).astype(int)
    return p.dropna().reset_index(drop=True)

def build_dataset(tickers):
    frames=[]
    for t in tickers:
        px=fetch_prices(t)
        if px.empty or len(px)<MIN_ROWS: continue
        if float(px["close"].iloc[-1])<MIN_PRICE: continue
        if px["volume"].rolling(20).mean().iloc[-1]<W_MIN_VOL: continue
        f=add_features(px)
        if not f.empty:
            f["ticker"]=t
            frames.append(f)
    if not frames: raise SystemExit("No usable histories.")
    df=pd.concat(frames,ignore_index=True)
    df["date"]=pd.to_datetime(df["date"])
    return df.sort_values(["ticker","date"]).reset_index(drop=True)

# --- TRAIN MODEL ---
FEATS=["close","ret1","ret5","ma5","ma10","ma20","std20","vol20","rsi14"]
def train_model(df, cutoff):
    train=df[df["date"]<=cutoff]
    if train.empty: return None
    X,y=train[FEATS].values,train["y"].values
    model=xgb.XGBClassifier(n_estimators=400,max_depth=5,learning_rate=0.05,
                             subsample=0.8,colsample_bytree=0.8,tree_method="hist",
                             reg_lambda=1.0,n_jobs=4,eval_metric="auc")
    model.fit(X,y)
    return model

# --- WALK-FORWARD BACKTEST ---
def walkforward(df):
    days=sorted(df["date"].unique())[-(BACK_DAYS+1):]
    logs=[]
    for d in days[:-1]:
        cutoff=d
        model=train_model(df,cutoff)
        block=df[df["date"]==cutoff]
        prob=model.predict_proba(block[FEATS].values)[:,1]
        block["prob"]=prob
        picks=block.sort_values("prob",ascending=False).head(TOPN).copy()
        nxt=df[df["date"]==days[days.index(d)+1]][["ticker","open","high","low","close"]]
        joined=picks.merge(nxt,on="ticker",how="left",suffixes=("","_next"))
        joined["stop"]=joined["close"]*(1-STOP_PCT)
        joined["tp"]=joined["close"]*(1+TP_PCT)
        def pnl(row):
            if pd.isna(row["close_next"]): return np.nan
            if row["low_next"]<=row["stop"]: return (row["stop"]/row["close"]-1)
            if row["high_next"]>=row["tp"]: return (row["tp"]/row["close"]-1)
            return (row["close_next"]/row["close"]-1)
        joined["ret1d"]=joined.apply(pnl,axis=1)
        joined["Qty"]=np.floor(PER_TRADE/joined["close"])
        joined["Pnl$"]=joined["ret1d"]*joined["Qty"]*joined["close"]
        for _,r in joined.iterrows():
            logs.append({"date":d.date(),"ticker":r["ticker"],"prob":r["prob"],
                         "ret1d":r["ret1d"],"Pnl$":r["Pnl$"]})
    bt=pd.DataFrame(logs)
    fn=os.path.join(OUT_DIR,"backtest_walkforward_30d.csv")
    bt.to_csv(fn,index=False)
    if not bt.empty:
        print(f"Walk-forward {BACK_DAYS}d → {len(bt)} trades | Win%={(bt['ret1d']>0).mean():.2%} | AvgRet={bt['ret1d'].mean():.3%} | TotalPnl=${bt['Pnl$'].sum():.2f}")
        print(f"Saved: {fn}")
    else:
        print("No trades logged.")

# --- MAIN ---
def main():
    tickers=load_universe()
    print(f"Universe: {len(tickers)} tickers")
    df=build_dataset(tickers)
    walkforward(df)

if __name__=="__main__":
    main()
