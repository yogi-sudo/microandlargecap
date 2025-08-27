#!/usr/bin/env python3
import os, datetime as dt
import pandas as pd
import numpy as np

OUT_DIR = os.getenv("OUT_DIR","out")
CACHE_DIR = os.getenv("CACHE_DIR","cache")

def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_ohlc.csv")

def _close_on(ticker: str, d: dt.date) -> float:
    """Get close for ticker on date d from cached OHLC CSV."""
    p = _cache_path(ticker)
    if not os.path.exists(p):
        return np.nan
    df = pd.read_csv(p, parse_dates=["date"])
    row = df[df["date"].dt.date == d]
    if row.empty:
        return np.nan
    return float(row["close"].iloc[0])

def log_from_plan(plan_csv: str, exit_next_day: bool = True) -> str:
    """
    Read trade_plan_YYYY-MM-DD.csv and compute realized PnL = (T+1 close / Buy - 1) * Capital.
    Append to out/performance.csv and return the path.
    """
    if not os.path.exists(plan_csv):
        raise FileNotFoundError(plan_csv)

    plan = pd.read_csv(plan_csv)
    # infer plan date from filename
    try:
        base = os.path.basename(plan_csv)
        date_str = base.replace("trade_plan_","").replace(".csv","")
        trade_date = pd.to_datetime(date_str).date()
    except Exception:
        # fallback: today
        trade_date = dt.date.today()

    exit_date = trade_date + dt.timedelta(days=1) if exit_next_day else trade_date

    # Ensure columns exist
    need = {"Ticker","BuyPrice","Capital"}
    missing = need - set(plan.columns)
    if missing:
        # Try alternative naming used earlier:
        rename = {}
        if "Close" in plan.columns and "BuyPrice" not in plan.columns: rename["Close"]="BuyPrice"
        plan = plan.rename(columns=rename)
        missing = need - set(plan.columns)
        if missing:
            raise ValueError(f"Plan missing columns: {missing}")

    out_rows = []
    for _, r in plan.iterrows():
        tkr = str(r["Ticker"])
        buy = float(r["BuyPrice"])
        cap = float(r["Capital"])
        sell = _close_on(tkr, exit_date)
        if np.isnan(sell):
            pnl = np.nan
            ret = np.nan
        else:
            ret = (sell / buy) - 1.0
            pnl = ret * cap
        out_rows.append({
            "TradeDate": trade_date,
            "ExitDate": exit_date,
            "Ticker": tkr,
            "Buy": buy,
            "ExitClose": sell,
            "Ret": ret,
            "PnL": pnl,
            "Capital": cap
        })

    perf = pd.DataFrame(out_rows)
    perf_path = os.path.join(OUT_DIR, "performance.csv")
    if os.path.exists(perf_path):
        old = pd.read_csv(perf_path, parse_dates=["TradeDate","ExitDate"])
        old["TradeDate"] = old["TradeDate"].dt.date
        old["ExitDate"]  = old["ExitDate"].dt.date
        perf = pd.concat([old, perf], ignore_index=True)
        perf = perf.drop_duplicates(subset=["TradeDate","Ticker"], keep="last")
    perf.to_csv(perf_path, index=False)

    # quick terminal summary
    val = perf[perf["TradeDate"]==trade_date].dropna(subset=["Ret"])
    if not val.empty:
        win = (val["Ret"]>0).mean()*100
        avg = val["Ret"].mean()*100
        tot = val["PnL"].sum()
        print(f"\n[PNL] {trade_date} → Exit {exit_date} | Trades={len(val)} | Win%={win:.1f}% | AvgRet={avg:.2f}% | TotalPnL=${tot:,.2f}")
    else:
        print(f"\n[PNL] {trade_date} → Exit {exit_date} | No valid closes in cache yet.")
    return perf_path
