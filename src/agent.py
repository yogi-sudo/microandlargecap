#!/usr/bin/env python3
import os, re, subprocess, sys, traceback, time, json

PROJECT_FILES = [
    "main.py",
    "src/universe.py","src/data_fetch.py","src/features.py",
    "src/ml_model.py","src/plan.py","src/sentiment.py","src/pnl.py"
]

FIX_PATTERNS = [
    # (file, regex, replacement, description)
    ("src/data_fetch.py", r'\["Volume"\]', '["volume"]', "volume column lower-case"),
    ("src/ml_model.py", r'from typing import ([^\n]+)', r'from typing import Optional, List, Tuple', "ensure Optional is imported"),
]

def _apply_patch(path, pattern, repl):
    try:
        s = open(path, "r", encoding="utf-8").read()
        ns = re.sub(pattern, repl, s, flags=re.M)
        if ns != s:
            open(path, "w", encoding="utf-8").write(ns)
            return True
        return False
    except Exception:
        return False

def run_with_self_heal(cmd):
    """Run main; if it fails with a known pattern, apply fix and retry once."""
    try:
        return subprocess.call(cmd)
    except Exception:
        pass

    # If Python raised, capture text
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode == 0:
            print(out.stdout)
            return 0
        err = (out.stderr or "") + "\n" + (out.stdout or "")
    except Exception as e:
        err = traceback.format_exc()

    print("[agent] detected failure, attempting simple fixes…")
    fixed_any = False
    for fp, pat, repl, desc in FIX_PATTERNS:
        if os.path.exists(fp) and re.search(pat, open(fp,encoding="utf-8").read(), flags=re.M):
            ok = _apply_patch(fp, pat, repl)
            if ok:
                print(f"[agent] applied fix: {desc} → {fp}")
                fixed_any = True

    if fixed_any:
        print("[agent] re-running…")
        return subprocess.call(cmd)
    else:
        print("[agent] no known safe fix matched; printing last error:")
        print(err)
        return 1

if __name__ == "__main__":
    sys.exit(run_with_self_heal([sys.executable, "main.py"]))
