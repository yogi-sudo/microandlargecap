#!/usr/bin/env python3
import os
from typing import List, Optional
import pandas as pd
import numpy as np

# Fallback if no universe CSVs are present
FALLBACK_ASX = [
    "CBA.AX","BHP.AX","CSL.AX","NAB.AX","WBC.AX","ANZ.AX","WES.AX","MQG.AX","GMG.AX","FMG.AX",
    "TLS.AX","WDS.AX","TCL.AX","ALL.AX","RIO.AX","WOW.AX","WTC.AX","BXB.AX","REA.AX","SIG.AX",
    "QBE.AX","PME.AX","COL.AX","XRO.AX","NST.AX","STO.AX","RMD.AX","SUN.AX","ORG.AX","CPU.AX",
    "SCG.AX","IAG.AX","COH.AX","FPH.AX","QAN.AX","EVN.AX","SOL.AX","SGP.AX","CAR.AX","MPL.AX",
    "LYC.AX","TNE.AX","JHX.AX","S32.AX","TLC.AX","JBH.AX","ASX.AX","VCX.AX","SHL.AX","APA.AX"
]

OUT_DIR = os.getenv("OUT_DIR", "out")
UNIVERSE_MAX = int(os.getenv("UNIVERSE_MAX", "0"))  # 0 = no cap

def _read_any_ticker_column(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Try common ticker column names and return a Series if found, else None.
    """
    for col in ["Ticker", "ticker", "Code", "code", "Symbol", "symbol"]:
        if col in df.columns:
            return df[col].astype(str)
    return None

def _ensure_ax_suffix(series: pd.Series) -> pd.Series:
    s = series.str.strip()
    return np.where(s.str.endswith(".AX"), s, s + ".AX")

def _load_from_file(path: str) -> Optional[List[str]]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        tser = _read_any_ticker_column(df)
        if tser is None:
            return None
        tickers = _ensure_ax_suffix(tser)
        uniq = sorted(pd.Series(tickers).dropna().unique().tolist())
        return uniq
    except Exception:
        return None

def get_universe(universe_max: Optional[int] = None) -> List[str]:
    """
    Priority:
      1) out/universe_all_clean.ax.csv         (your curated universe)
      2) out/tier_combined.csv                 (built by your tier pipeline)
      3) FALLBACK_ASX                          (hardcoded blue-chips)
    The .AX suffix is enforced. Duplicates removed. Optional cap via env or arg.
    """
    if universe_max is None:
        universe_max = UNIVERSE_MAX

    paths = [
        os.path.join(OUT_DIR, "universe_all_clean.ax.csv"),
        os.path.join(OUT_DIR, "tier_combined.csv"),
    ]

    tickers: Optional[List[str]] = None
    for p in paths:
        tickers = _load_from_file(p)
        if tickers:
            break

    if not tickers:
        tickers = FALLBACK_ASX.copy()

    if universe_max and universe_max > 0:
        tickers = tickers[:universe_max]

    return tickers
