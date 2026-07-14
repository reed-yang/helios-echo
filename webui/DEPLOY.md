# Deploying the Helios Attention Explorer

The site is a fully self-contained static SPA (`index.html` + `app.js` + `style.css`
+ `data/*.json`). No backend, no build step, no external CDN. Any static host works.

## Contents
```
webui/
├── index.html          # UI shell (4 tabs, toggles)
├── app.js              # all logic + hand-rolled SVG rendering (no libraries)
├── style.css
├── data/
│   ├── manifest.json   # [{name, file}] datasets the UI can load
│   └── base_full.json  # aggregated probe output (meta + by_step_layer + full_maps)
├── build_data.sh       # copies helios/analysis/out/*.json -> data/ + writes manifest
└── DEPLOY.md
```
Regenerate `data/` after a new probe run:
```bash
bash webui/build_data.sh
```

## Option A — Cloudflare Pages (durable, recommended)
Requires a Cloudflare account + API token with the **Pages:Edit** permission
(or `wrangler login` in a browser session).

```bash
# one-time
npm i -g wrangler         # or: npx wrangler ...
export CLOUDFLARE_API_TOKEN=<token>
export CLOUDFLARE_ACCOUNT_ID=<account id>

# deploy the static dir
cd helios-team
wrangler pages deploy webui --project-name helios-attention --commit-dirty=true
```
`wrangler` prints the production URL (`https://helios-attention.pages.dev`) and a
per-deploy preview URL. To wire it to a custom/official domain, add the domain
under Pages → Custom domains in the dashboard.

Git-based alternative: push this repo, connect it in the Pages dashboard, set
**build command = (none)** and **output dir = `webui`**.

## Option B — Instant public URL via quick tunnel (no credentials)
Serves `webui/` from this machine and exposes it through Cloudflare's
`*.trycloudflare.com`. Ephemeral (dies with the process) but zero-setup — this is
what `run_tunnel.sh` does:
```bash
bash webui/run_tunnel.sh 8790     # prints the https://<random>.trycloudflare.com URL
```
