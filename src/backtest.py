import pandas as pd, os, numpy as np

def walkforward_backtest(model, feats, data, days=30, topN=10):
    dates = sorted(data["date"].unique())[-days-1:]
    logs=[]
    for d in dates[:-1]:
        block = data[data["date"]==d].copy()
        prob = model.predict_proba(block[feats].values)[:,1]
        picks = block.assign(prob=prob).sort_values("prob",ascending=False).head(topN)
        nxt = data[data["date"]==pd.to_datetime(d)+pd.Timedelta(days=1)][["ticker","close"]]
        joined = picks.merge(nxt,on="ticker",how="left",suffixes=("","_next"))
        joined["ret1d"] = joined["close_next"]/joined["close"]-1
        logs.append(joined)
    bt = pd.concat(logs)
    win=(bt["ret1d"]>0).mean(); avg=bt["ret1d"].mean()
    print(f"Walk-forward {days}d → {len(bt)} trades | Win%={win:.2%} | AvgRet={avg:.3%}")
    os.makedirs("out",exist_ok=True)
    bt.to_csv("out/backtest_walkforward.csv",index=False)
