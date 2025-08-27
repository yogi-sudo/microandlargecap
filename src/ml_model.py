import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

FEATS = ["close","ret1","ret5","ma5","ma10","ma20","std20","vol20","rsi14"]

def train_and_eval(data: pd.DataFrame, backtest_days: int = 30, thresh: float = 0.55):
    cutoff = data["date"].max() - pd.Timedelta(days=backtest_days+5)
    train = data[data["date"] <= cutoff]
    test  = data[data["date"] > cutoff]
    Xtr, ytr = train[FEATS].values, train["y"].values
    Xte, yte = test[FEATS].values,  test["y"].values

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", n_jobs=4, eval_metric="auc"
    )
    model.fit(Xtr, ytr)

    if len(Xte) > 0:
        prob = model.predict_proba(Xte)[:,1]
        auc = roc_auc_score(yte, prob)
        acc = accuracy_score(yte, (prob>=0.5).astype(int))
        hr  = ((prob>=thresh) & (yte==1)).sum() / max(1, (prob>=thresh).sum())
        cov = (prob>=thresh).mean()
        print(f"Holdout: AUC={auc:.3f} Acc={acc:.3f} Hit@{thresh:.2f}={hr:.3f} Coverage={cov:.3f}")
    else:
        print("Holdout: not enough rows after split; trained on all.")

    return model, FEATS

def walkforward_backtest(model, feat_cols, data: pd.DataFrame, days: int = 30, topN: int = 10):
    last_days = sorted(data["date"].unique())[-days-1:]
    logs = []
    for d in last_days[:-1]:
        block = data[data["date"] == d].copy()
        if block.empty: 
            continue
        prob = model.predict_proba(block[feat_cols].values)[:,1]
        block["prob"] = prob
        picks = block.sort_values("prob", ascending=False).head(topN).copy()
        nextd = pd.to_datetime(d) + pd.Timedelta(days=1)
        nxt = data[data["date"] == nextd][["ticker","close"]].rename(columns={"close":"close_next"})
        joined = picks.merge(nxt, on="ticker", how="left")
        joined["ret1d"] = (joined["close_next"]/joined["close"] - 1.0)
        logs.extend(joined[["ticker","prob","ret1d"]].assign(date=pd.to_datetime(d).date()).to_dict("records"))

        # save day picks for audit trail
        picks[["ticker","close","prob"]].to_csv(os.path.join("out","picks_history",f"picks_{pd.to_datetime(d).date()}.csv"), index=False)

    if not logs:
        print("Backtest: no logs.")
        return

    bt = pd.DataFrame(logs).dropna(subset=["ret1d"])
    winrate = (bt["ret1d"] > 0).mean()
    avg_ret = bt["ret1d"].mean()
    print(f"Walk-forward {days}d → picks: {len(bt)} | Win%={winrate:.2%} | Avg 1d ret={avg_ret:.3%}")
    bt.to_csv(os.path.join("out", f"backtest_{days}d_pnl.csv"), index=False)
