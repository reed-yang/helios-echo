#!/bin/bash
# Keepalive for the P2 preview site: restart the static server and the
# Cloudflare quick tunnel when either dies. Current public URL is kept in
# $SITE/.current_url (quick tunnels get a fresh hostname on restart).
SITE=/mnt/beegfs/siyuan/workspace/helios-echo/results/p2_site
CF=/home/siyuan/.local/bin/cloudflared

while true; do
  if ! curl -s -o /dev/null --max-time 5 http://127.0.0.1:8437/index.html; then
    pkill -f "http-server $SITE" 2>/dev/null
    setsid nohup npx --yes http-server "$SITE" -p 8437 -a 127.0.0.1 -c-1 --silent >> "$SITE/.server.log" 2>&1 &
    sleep 10
  fi
  if ! pgrep -f "cloudflared tunnel --url http://127.0.0.1:8437" > /dev/null; then
    setsid nohup "$CF" tunnel --url http://127.0.0.1:8437 --no-autoupdate --logfile "$SITE/.tunnel.log" > /dev/null 2>&1 &
    sleep 20
    url=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$SITE/.tunnel.log" | tail -1)
    [ -n "$url" ] && echo "$url" > "$SITE/.current_url"
  fi
  sleep 60
done
