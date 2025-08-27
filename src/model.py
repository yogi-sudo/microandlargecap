import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

FEATS=["close","ret1","ret5","ma5","ma10","ma20","std20","vol20","rsi14"]

def train_and_eval(data):
    cutoff = data["date"].max() - pd.Timedelta(days=35)
    train = data[data["date"]<=cutoff]
    test  = data[data["date"]>cutoff]
    Xtr, ytr = train[FEATS].values, train["y"].values
    Xte, yte = test[FEATS].values, test["y"].values
    model = xgb.XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,
                              subsample=0.8,colsample_bytree=0.8,tree_method="hist")
    model.fit(Xtr, ytr)
    if len(Xte):
        prob = model.predict_proba(Xte)[:,1]
        print(f"AUC={roc_auc_score(yte,prob):.3f} Acc={accuracy_score(yte,prob>0.5):.3f}")
    return model, FEATS
