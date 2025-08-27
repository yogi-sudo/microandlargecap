#!/usr/bin/env python3
import os, sys, subprocess, shlex, tempfile
import pandas as pd
from pathlib import Path

SWING_CSV     = "artifacts/nextday_report.csv"
MICRO_CSV     = "artifacts/microcap_candidates.csv"
CAPS_CSV      = "data/universe_caps.csv"
OUT_COMBINED  = "artifacts/nextday_combined.csv"
FETCH_CAPS    = "analysis/fetch_market_caps.py"

def norm_tk(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.upper()
              .str.replace(r"\.AX$|\.ASX$", "", regex=True)
              .str.replace(r"[^0-9A-Z]+","", regex=True))

def load_swing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalize common columns -> our schema
    rename = {
        "prob_%":"prob_pct", "exp_move_%":"exp_move_pct",
        "ticker":"ticker", "side":"side", "entry":"entry",
        "tp":"tp", "sl":"sl", "headline":"headline", "source":"source"
    }
    for k in list(rename.keys()):
        if k not in df.columns and k.capitalize() in df.columns:
            df.rename(columns={k.capitalize():rename[k]}, inplace=True)
    # ensure required cols exist
    for c in ["prob_pct","exp_move_pct","side","entry","tp","sl","headline"]:
        if c not in df.columns: df[c] = pd.NA
    df["ticker"] = norm_tk(df.get("ticker", pd.Series([])))
    df["group"]  = "Daily Swing (Large/Mid)"
    df["label"]  = df.get("label", pd.Series(["bullish"]*len(df)))
    df["rel_vol"] = pd.NA
    df["dollar_vol"] = pd.NA
    # keep common view
    keep = ["group","ticker","label","prob_pct","exp_move_pct","side",
            "entry","tp","sl","headline","rel_vol","dollar_vol"]
    return df[[c for c in keep if c in df.columns]]

def load_micro(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Standardize column names
    rename = {
        "gap_%":"exp_move_pct",   # treat gap as exp move (%) for display
        "price":"entry"
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    df["ticker"]   = norm_tk(df.get("ticker", pd.Series([])))
    df["group"]    = "Intraday Spikes (Microcap)"
    df["label"]    = "momentum"
    df["side"]     = "long"
    df["tp"]       = df.get("tp", pd.NA)
    df["sl"]       = df.get("sl", pd.NA)
    if "headline" not in df.columns:
        df["headline"] = "Momentum (no news)"
    # keep common view
    keep = ["group","ticker","label","prob_pct","exp_move_pct","side",
            "entry","tp","sl","headline","rel_vol","dollar_vol"]
    for c in keep:
        if c not in df.columns: df[c] = pd.NA
    return df[keep]

def ensure_caps_for_tickers(tickers):
    """
    Make sure CAPS_CSV exists and has rows for the specified tickers.
    If not, try to call analysis/fetch_market_caps.py to produce it.
    """
    Path("data").mkdir(parents=True, exist_ok=True)

    have = pd.DataFrame()
    if os.path.exists(CAPS_CSV):
        try:
            have = pd.read_csv(CAPS_CSV)
        except Exception:
            have = pd.DataFrame()

    col_tk = None
    for c in ["ticker","Ticker","symbol","Symbol","code","Code"]:
        if c in have.columns:
            col_tk = c
            break

    need_fetch = False
    if have.empty or col_tk is None:
        need_fetch = True
    else:
        got_set = set(norm_tk(have[col_tk].fillna("")))
        miss = [t for t in tickers if t not in got_set]
        if len(miss) > 0:
            need_fetch = True

    if not need_fetch:
        return  # all good

    if not os.path.exists(FETCH_CAPS):
        # can't auto-fetch; leave it to user
        print(f"[caps] {CAPS_CSV} missing or incomplete and {FETCH_CAPS} not found. "
              f"Proceeding without caps (bands will be Unclassified).")
        return

    # Write a temp universe file for just these tickers
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
        pd.DataFrame({"ticker": tickers}).to_csv(tmp_path, index=False)

    cmd = f'python3 {shlex.quote(FETCH_CAPS)} --universe_csv {shlex.quote(tmp_path)} ' \
          f'--out_csv {shlex.quote(CAPS_CSV)} --cache data/fundamentals_cache.json --workers 8 --max 10000'
    print(f"[caps] fetching… ({len(tickers)} tickers)")
    try:
        subprocess.run(shlex.split(cmd), check=False)
    except Exception as e:
        print(f"[caps] fetch attempt failed: {e}")

def merge_caps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # normalize tickers
    df = df.copy()
    df["ticker"] = norm_tk(df["ticker"])

    if not os.path.exists(CAPS_CSV):
        df["market_cap_m"] = pd.NA
        df["sector"] = pd.NA
        return df

    caps = pd.read_csv(CAPS_CSV)
    # find ticker column
    tcol = next((c for c in ["ticker","Ticker","symbol","Symbol","code","Code"] if c in caps.columns), None)
    if tcol is None:
        df["market_cap_m"] = pd.NA
        df["sector"] = pd.NA
        return df
    caps["ticker"] = norm_tk(caps[tcol])

    # find cap column and coerce to millions
    ccol = next((c for c in [
        "market_cap_m","market_cap","marketCap","MarketCap","mktcap","cap_m",
        "market_capitalization","market_capitalisation","MarketCapitalisation"
    ] if c in caps.columns), None)
    if ccol:
        caps["market_cap_m"] = pd.to_numeric(caps[ccol], errors="coerce")
        med = caps["market_cap_m"].dropna().median()
        # if looks like raw dollars, convert to millions
        if pd.notna(med) and med > 1e6:
            caps["market_cap_m"] = caps["market_cap_m"] / 1_000_000.0
    else:
        caps["market_cap_m"] = pd.NA

    scol = next((c for c in ["sector","Sector","industry","Industry","GICS_Sector","GICS Sector"]
                 if c in caps.columns), None)
    caps["sector"] = caps[scol] if scol else pd.NA

    caps = caps[["ticker","market_cap_m","sector"]].drop_duplicates("ticker")
    return df.merge(caps, on="ticker", how="left")

def band_cap(x):
    try:
        x = float(x)
    except Exception:
        return "Unclassified"
    if x >= 5000: return "Large-cap"
    if x >= 500:  return "Mid-cap"
    return "Micro-cap"

def print_section(title: str, df: pd.DataFrame):
    print(f"\n=== {title} (rows={len(df)}) ===")
    if df.empty:
        print("None"); return
    cols = ["ticker","cap_band","label","prob_pct","exp_move_pct","entry","tp","sl","rel_vol","dollar_vol","headline"]
    cols = [c for c in cols if c in df.columns]
    # order a bit
    order = ["ticker","cap_band","label","prob_pct","exp_move_pct","entry","tp","sl","rel_vol","dollar_vol","headline"]
    cols_sorted = [c for c in order if c in cols]
    print(df[cols_sorted].to_string(index=False))

def main():
    os.makedirs("artifacts", exist_ok=True)

    swing = load_swing(SWING_CSV)
    micro = load_micro(MICRO_CSV)

    if swing.empty and micro.empty:
        print("[combine] Nothing to combine — run the two generators first.")
        sys.exit(0)

    combined = pd.concat([swing, micro], ignore_index=True)
    combined["ticker"] = norm_tk(combined["ticker"])
    tickers = sorted(set(combined["ticker"].dropna()))
    # ensure caps present or generate them
    ensure_caps_for_tickers(tickers)

    # merge caps and band
    combined = merge_caps(combined)
    if "market_cap_m" not in combined.columns:
        combined["market_cap_m"] = pd.NA
    combined["cap_band"] = combined["market_cap_m"].map(band_cap)

    # write output
    combined.to_csv(OUT_COMBINED, index=False)

    # Print sections
    swing_view  = combined[combined["group"]=="Daily Swing (Large/Mid)"].copy()
    micro_view  = combined[combined["group"]=="Intraday Spikes (Microcap)"].copy()

    # sorting inside sections
    if "prob_pct" in swing_view.columns:
        swing_view = swing_view.sort_values(["prob_pct","exp_move_pct"], ascending=False, na_position="last")
    if "exp_move_pct" in micro_view.columns:
        micro_view = micro_view.sort_values(["exp_move_pct"], ascending=False, na_position="last")

    print_section("Daily Swing (Large/Mid)", swing_view)
    print_section("Intraday Spikes (Microcap)", micro_view)

    print(f"\n[OK] wrote -> {OUT_COMBINED}")

if __name__ == "__main__":
    main()