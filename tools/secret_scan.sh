#!/usr/bin/env bash
# Secret / private-info scan over ALL commits of the repo (not only HEAD). Prints only file:line and the pattern class.
set -uo pipefail
cd "${1:-.}"
PAT='(sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|[0-9]{8,10}:AA[A-Za-z0-9_-]{30,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}|api[_-]?key\s*[:=]\s*["'\'']?[A-Za-z0-9_-]{16,}|password\s*[:=]\s*["'\''][^"'\'' ]{6,}|token\s*[:=]\s*["'\''][A-Za-z0-9._-]{16,})'
# private infra: generic tailnet/LAN patterns; add your own host names, user paths etc. one regex per line in a
# file outside the repo and pass it as SECRET_SCAN_PRIVATE=<file>
PRIV='(taild[0-9a-f]+\.ts\.net|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]+\.[0-9]+|/home/[a-z_][a-z0-9_-]*/)'
if [ -n "${SECRET_SCAN_PRIVATE:-}" ] && [ -s "$SECRET_SCAN_PRIVATE" ]; then
  PRIV="${PRIV%)}|$(grep -v '^#' "$SECRET_SCAN_PRIVATE" | grep . | paste -sd'|'))"
fi
echo "== secrets (all history)"
git log --all -p --no-color | grep -nEi "$PAT" | sed -E 's/(eyJ[A-Za-z0-9_-]{6})[A-Za-z0-9._-]+/\1…/g; s/(sk-[A-Za-z0-9-]{4})[A-Za-z0-9_-]+/\1…/g' | head -40
echo "== private infra / personal (HEAD files, counts)"
git grep -nEi -c "$PRIV" HEAD -- . | head -40
echo "== files with suspicious names (all history)"
git log --all --name-only --pretty=format: | sort -u | grep -Ei '(\.env|secret|token|cookie|credential|id_rsa|\.pem|\.key$|state\.json|config\.json)' || echo none
echo "== large blobs (all history, >1 MB)"
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '$1=="blob" && $3>1000000 {print $3, $4}' | sort -rn | head
