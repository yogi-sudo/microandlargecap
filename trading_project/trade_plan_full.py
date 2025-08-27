#!/usr/bin/env python3
import os, sys, json, datetime as dt, requests
import pandas as pd, numpy as np
from tqdm import tqdm
import yfinance as yf
from dotenv import load_dotenv
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ================== CONFIG ==================
TODAY = dt.date.today()
OUT_DIR = "out"
CACHE_DIR = "cache"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

load_dotenv()
EODHD_KEY = os.getenv("EODHD_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

UNIVERSE_FILE = "universe_caps.csv"  # combined Large/Mid/Micro list you gave me

RISK_PER_TRADE = 0.01   # 1% of equity
EQUITY = 100000         # default account size
ATR_MULT_SL = 2.0       # Stop-loss = entry - ATR*2
ATR_MULT_TP = 3.0       # Take-profit = entry + ATR*3

# ================== HELPERS ==================
def load_universe():
    df = pd.read_csv(UNIVERSE_FILE)
    df = df.dropna(subset=["Ticker","Approx. Market Cap ($B)"])
    df["Approx. Market Cap ($B)"] = df["Approx. Market Cap ($B)"].astype(float)
    df = df.sort_values("Approx. Market Cap ($B)", ascending=False)
    return df

def fetch_ohlcv(ticker, years=3):
    """Try EODHD, fallback to Yahoo"""
    start = (TODAY - dt.timedelta(days=365*years)).isoformat()
    cache_file = f"{CACHE_DIR}/{ticker}_ohlc.csv"
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=["date"])
    df = None
    if EODHD_KEY:
        url = f"https://eodhd.com/api/eod/{ticker}.AU"
        params = {"api_token": EODHD_KEY, "fmt":"json","period":"d","from":start}
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code==200:
                d = pd.DataFrame(r.json())
                if not d.empty:
                    d.rename(columns={"date":"date"}, inplace=True)
                    d["date"] = pd.to_datetime(d["date"])
                    d.to_csv(cache_file,index=False)
                    return d[["date","open","high","low","close","volume"]]
        except: pass
    # fallback to yfinance
    try:
        d = yf.download(f"{ticker}.AX", start=start, end=(TODAY+dt.timedelta(days=1)).isoformat(), progress=False)
        if not d.empty:
            d = d.reset_index()[["Date","Open","High","Low","Close","Volume"]]
            d.columns = ["date","open","high","low","close","volume"]
            d.to_csv(cache_file,index=False)
            return d
    except: return None
    return None

def atr(df, n=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def news_sentiment(ticker):
    """Simple sentiment via NewsAPI headlines"""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q":ticker,"apiKey":NEWSAPI_KEY,"pageSize":5,"sortBy":"publishedAt"}
        r = requests.get(url, params=params, timeout=20)
        arts = r.json().get("articles",[])
        if not arts: return 0.5
        text = " ".join([a["title"] for a in arts])
        # crude scoring (pos words / total)
        pos_words = sum(w.lower() in text.lower() for w in ["up","gain","profit","bullish","growth","surge"])
        return min(1.0, pos_words/max(1,len(arts)))
    except: return 0.5

def compute_features(df):
    df = df.copy()
    if isinstance(df["close"], pd.DataFrame):
        df["close"] = df["close"].iloc[:,0]
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["ma20"] = df["close"].rolling(20).mean()
    df["vol20"] = df["close"].pct_change().rolling(20).std()
    df["atr14"] = atr(df,14)
    df.dropna(inplace=True)
    return df

# ================== MAIN ==================
def main():
    uni = load_universe()
    print(f"Universe loaded: {len(uni)} stocks")

    # === Build dataset for ML ===
    dfs=[]
    for t in tqdm(uni["Ticker"], desc="OHLCV"):
        d = fetch_ohlcv(t)
        if d is not None and len(d)>100:
            f = compute_features(d)
            f["ticker"] = t
            dfs.append(f)
    if not dfs:
        sys.exit("No price data available.")
    data = pd.concat(dfs)
    data.sort_values(["ticker","date"], inplace=True)

    # target = next-day positive return
    data["target"] = (data.groupby("ticker")["close"].shift(-1) > data["close"]).astype(int)
    feat_cols = ["close","ret1","ret5","ma20","vol20","atr14"]

    # train/test split
    cutoff = data["date"].max() - dt.timedelta(days=60)
    train = data[data["date"]<=cutoff]
    test = data[data["date"]>cutoff]

    Xtr, ytr = train[feat_cols], train["target"]
    Xte, yte = test[feat_cols], test["target"]

    model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    model.fit(Xtr,ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:,1])
    print(f"Holdout AUC={auc:.3f}")

    # === Latest snapshot ===
    latest = data.sort_values(["ticker","date"]).groupby("ticker").tail(1).copy()
    latest["ml_prob"] = model.predict_proba(latest[feat_cols])[:,1]
    latest["sentiment"] = latest["ticker"].apply(news_sentiment)
    latest["blended"] = (latest["ml_prob"]*0.7 + latest["sentiment"]*0.3)

    # Position sizing
    out=[]
    for _,r in latest.iterrows():
        sl = max(0.01, r["atr14"])*ATR_MULT_SL
        tp = r["close"] + r["atr14"]*ATR_MULT_TP
        risk_per_share = r["close"] - sl
        size = int((EQUITY*RISK_PER_TRADE)/max(0.01,risk_per_share))
        out.append({
            "Ticker":r["ticker"],"Close":r["close"],"Prob":round(r["ml_prob"],3),
            "Sentiment":round(r["sentiment"],2),"Blended":round(r["blended"],3),
            "StopLoss":round(sl,2),"TakeProfit":round(tp,2),"Size":size
        })
    df_out=pd.DataFrame(out).sort_values("Blended",ascending=False)
    fname=f"{OUT_DIR}/trade_plan_{TODAY.isoformat()}.csv"
    df_out.to_csv(fname,index=False)
    print(f"\nSaved trade plan: {fname}")
    print(df_out.head(15).to_string(index=False))

if __name__=="__main__":
    main()