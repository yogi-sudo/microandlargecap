import os, sys, pandas as pd

p = "artifacts/nextday_combined.csv"
if not os.path.exists(p):
    print("Missing file:", p)
    print("Run: bash ./run_all.sh")
    sys.exit(1)

df = pd.read_csv(p)

# Only the swing group
df = df[df.get("group","") == "Daily Swing (Large/Mid)"].copy()

# Columns to show (only those that exist)
cols = [c for c in ["ticker","cap_band","prob_pct","entry","tp","sl","headline"] if c in df.columns]
if not cols:
    print("No expected columns found in", p)
    sys.exit(1)

df = df[cols].head(12)

# Numeric formatting
for c in ("entry","tp","sl","prob_pct"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
fmt_price = lambda v: f"{v:.2f}" if pd.notna(v) else ""
fmt_prob  = lambda v: f"{v:.1f}" if pd.notna(v) else ""
if "entry" in df: df["entry"] = df["entry"].map(fmt_price)
if "tp" in df:    df["tp"]    = df["tp"].map(fmt_price)
if "sl" in df:    df["sl"]    = df["sl"].map(fmt_price)
if "prob_pct" in df: df["prob_pct"] = df["prob_pct"].map(fmt_prob)

# Trim headline
if "headline" in df:
    def trim(s, n=90):
        if pd.isna(s): return ""
        s = str(s).replace("\n"," ").strip()
        return s if len(s)<=n else s[:n-1]+"…"
    df["headline"] = df["headline"].map(trim)

print("\n=== TOMORROW'S TRADE PLAN (Swing) ===")
print(df.to_string(index=False))
