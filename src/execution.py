import os, pandas as pd, numpy as np, datetime as dt

def generate_trade_plan(model, feats, data, topN=10, capital=3000):
    last_day=data["date"].max()
    block=data[data["date"]==last_day].copy()
    prob=model.predict_proba(block[feats].values)[:,1]
    picks=block.assign(prob=prob).sort_values("prob",ascending=False).head(topN)
    picks["Qty"]=np.floor((capital/topN)/picks["close"]).astype(int)
    picks["Capital"]=picks["Qty"]*picks["close"]
    out=picks[["ticker","close","prob","Qty","Capital"]].rename(
        columns={"ticker":"Ticker","close":"Close","prob":"ProbUp_T1"})
    fn=f"out/trade_plan_{dt.date.today().isoformat()}.csv"
    out.to_csv(fn,index=False)
    print(f"Trade plan saved: {fn}")
