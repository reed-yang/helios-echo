#!/usr/bin/env bash
# Serve the static site and expose it via a Cloudflare quick tunnel (no creds).
# Usage: bash webui/run_tunnel.sh [PORT]
set -euo pipefail
PORT="${1:-8790}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CFD="$(command -v cloudflared || echo /home/xiangbo/bin/cloudflared)"
LOG=/tmp/helios_attn_tunnel.log

# static server
pkill -f "http.server ${PORT}" 2>/dev/null || true
( cd "$HERE" && python -m http.server "$PORT" >/tmp/helios_attn_http.log 2>&1 & )
sleep 1
echo "[serve] http://localhost:${PORT} (root: $HERE)"

# quick tunnel
pkill -f "cloudflared tunnel --url" 2>/dev/null || true
nohup "$CFD" tunnel --url "http://localhost:${PORT}" --no-autoupdate > "$LOG" 2>&1 &
echo "[tunnel] starting cloudflared (pid $!) ..."
for i in $(seq 1 30); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOG" | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done
if [ -n "${URL:-}" ]; then
  echo "[tunnel] PUBLIC URL: $URL"
else
  echo "[tunnel] URL not ready yet; tail $LOG"; tail -5 "$LOG" || true
fi
