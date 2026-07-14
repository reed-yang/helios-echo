#!/usr/bin/env bash
# Publish probe JSON outputs into the static site (webui/data/) and write a
# manifest the SPA reads. Usage: bash webui/build_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=webui/data
mkdir -p "$OUT"

# name|file  (source under helios/analysis/out) -> copied into webui/data
declare -a SETS=(
  "Helios-Base|base_full.json"
  "Helios-Distilled|distilled_full.json"
)

MANI="["
first=1
for s in "${SETS[@]}"; do
  name="${s%%|*}"; file="${s##*|}"
  src="helios/analysis/out/$file"
  if [ -f "$src" ]; then
    cp "$src" "$OUT/$file"
    [ $first -eq 1 ] || MANI+=","
    MANI+="{\"name\":\"$name\",\"file\":\"$file\"}"
    first=0
    echo "published $name -> $OUT/$file ($(du -h "$OUT/$file" | cut -f1))"
  else
    echo "skip $name (missing $src)"
  fi
done
MANI+="]"
echo "$MANI" > "$OUT/manifest.json"
echo "wrote $OUT/manifest.json: $MANI"
