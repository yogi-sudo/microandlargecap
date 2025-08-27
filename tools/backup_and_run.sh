#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR=".backup_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"/{analysis,src,cache,cache_eodhd,out,artifacts,data}

shopt -s nullglob
mv analysis/*.bak                "$BACKUP_DIR/analysis/" 2>/dev/null || true
mv analysis/*_bak.py             "$BACKUP_DIR/analysis/" 2>/dev/null || true
mv analysis/*_py.bak             "$BACKUP_DIR/analysis/" 2>/dev/null || true
mv analysis/*.py.bak             "$BACKUP_DIR/analysis/" 2>/dev/null || true
mv analysis/*~                   "$BACKUP_DIR/analysis/" 2>/dev/null || true
mv analysis/*.log                "$BACKUP_DIR/analysis/" 2>/dev/null || true
mv src/*.bak                     "$BACKUP_DIR/src/"      2>/dev/null || true
mv src/*_bak.py                  "$BACKUP_DIR/src/"      2>/dev/null || true
mv src/*_py.bak                  "$BACKUP_DIR/src/"      2>/dev/null || true
mv src/*.py.bak                  "$BACKUP_DIR/src/"      2>/dev/null || true
mv src/*~                        "$BACKUP_DIR/src/"      2>/dev/null || true
mv src/*.log                     "$BACKUP_DIR/src/"      2>/dev/null || true
mv *.bak                         "$BACKUP_DIR/"          2>/dev/null || true
mv *~                            "$BACKUP_DIR/"          2>/dev/null || true
mv *.log                         "$BACKUP_DIR/"          2>/dev/null || true

if ! grep -qE '(^|/)\.backup_|\.bak$' .gitignore 2>/dev/null; then
  {
    echo ""
    echo "# Backups and editor crud"
    echo ".backup_*"
    echo "*.bak"
    echo "*~"
    echo "*.log"
  } >> .gitignore
fi

git add -A
git commit -m "Chore: move backup/old files into ${BACKUP_DIR} and ignore *.bak" || true

chmod +x run_all.sh
./run_all.sh
