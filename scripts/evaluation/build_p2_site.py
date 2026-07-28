#!/usr/bin/env python3
"""Build a static preview site for P2-interim long-video A/B results.

Scans results/p2_interim/*/{on,off}/prompt_*.manifest.json plus the matching
metrics JSONs, copies videos, and emits a self-contained index.html. Re-run
after every new campaign; the page lists whatever campaigns exist on disk.

Usage: python scripts/evaluation/build_p2_site.py [--out results/p2_site]
"""

import argparse
import html
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P2_ROOT = REPO / "results" / "p2_interim"

CAMPAIGN_META = {
    "p2-interim-r1": {
        "off": ("r1 · off(无记忆基线)", "Distilled 基座,真·无记忆。经典漂移:运动失稳 + 去饱和崩塌。"),
        "on": ("r1 · on(未训练记忆)", "全新 M₀,未经训练。静态塌缩:motion→0,画面冻结。"),
    },
    "p2-interim-r2-pilot1000": {
        "on": ("r2 · pilot ckpt-1000(训练后记忆)", "Stage A pilot 语料(2,869 clips)训练 1000 步。饱和度持平,motion 存活。"),
    },
    "p2-interim-r2-slice512-500": {
        "on": ("r2 · slice512 ckpt-500(训练后记忆)", "Stage A 57k-clip 语料训练 500 步。饱和度过冲上行,motion 存活。"),
    },
}


def window_mean(values, start, end):
    chunk = values[start:end]
    return sum(chunk) / len(chunk) if chunk else float("nan")


def collect_cards():
    cards = []
    for manifest_path in sorted(P2_ROOT.glob("*/*/prompt_*.manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        run_id, arm, idx = manifest["run_id"], manifest["arm"], manifest["prompt_index"]
        video = Path(manifest["artifacts"]["video"])
        if not video.exists():
            continue
        metrics_path = P2_ROOT / run_id / "metrics" / f"{arm}_prompt_{idx:02d}.json"
        metrics = None
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            mot, sat = m["per_chunk"]["motion"], m["per_chunk"]["saturation"]
            metrics = {
                "mot_first": window_mean(mot, 0, 10),
                "mot_last": window_mean(mot, len(mot) - 10, len(mot)),
                "sat_first": window_mean(sat, 0, 10),
                "sat_last": window_mean(sat, len(sat) - 10, len(sat)),
                "sat_slope": m["slopes"]["saturation"]["theil_sen"],
            }
        label, blurb = CAMPAIGN_META.get(run_id, {}).get(arm, (f"{run_id} · {arm}", ""))
        partial = manifest.get("memory_partial")
        cards.append({
            "order": list(CAMPAIGN_META).index(run_id) if run_id in CAMPAIGN_META else 99,
            "run_id": run_id, "arm": arm, "idx": idx,
            "label": label, "blurb": blurb,
            "prompt": manifest["prompt"], "seed": manifest["seed"],
            "created": manifest["created_at_utc"][:16].replace("T", " "),
            "ckpt": Path(partial["path"]).parent.name if partial else "—",
            "sha": partial["sha256"][:12] if partial else "—",
            "video_src": video,
            "video_name": f"{run_id.replace('p2-interim-', '')}_{arm}_p{idx}.mp4",
        })
    cards.sort(key=lambda c: (c["order"], c["arm"] != "off", c["idx"]))
    return cards


def render(cards):
    def fmt(v, nd=2):
        return f"{v:.{nd}f}" if isinstance(v, float) else "—"

    groups = {}
    for c in cards:
        groups.setdefault((c["order"], c["label"], c["blurb"]), []).append(c)

    sections = []
    for (_, label, blurb), items in sorted(groups.items(), key=lambda kv: kv[0][0]):
        tiles = []
        for c in items:
            m = c.get("metrics") or {}
            metric_row = ""
            mp = P2_ROOT / c["run_id"] / "metrics" / f"{c['arm']}_prompt_{c['idx']:02d}.json"
            if mp.exists():
                mm = json.loads(mp.read_text())
                mot, sat = mm["per_chunk"]["motion"], mm["per_chunk"]["saturation"]
                metric_row = (
                    f"<div class='metrics'>motion {window_mean(mot,0,10):.2f} → {window_mean(mot,len(mot)-10,len(mot)):.2f}"
                    f" · 饱和度 {window_mean(sat,0,10):.0f} → {window_mean(sat,len(sat)-10,len(sat)):.0f}"
                    f" · 斜率 {mm['slopes']['saturation']['theil_sen']:+.2f}</div>"
                )
            tiles.append(f"""
      <div class="card">
        <video controls preload="metadata" src="videos/{c['video_name']}"></video>
        <div class="meta">
          <div class="prompt">P{c['idx']} · {html.escape(c['prompt'])}</div>
          {metric_row}
          <div class="sub">seed {c['seed']} · ckpt {html.escape(c['ckpt'])} · sha {c['sha']} · {c['created']} UTC</div>
        </div>
      </div>""")
        sections.append(f"""
    <section>
      <h2>{html.escape(label)}</h2>
      <p class="blurb">{html.escape(blurb)}</p>
      <div class="grid">{''.join(tiles)}
      </div>
    </section>""")

    return f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Helios-Echo · P2 长视频漂移评测预览</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; background: #0d1017; color: #e6e9ef; font: 15px/1.6 -apple-system, "Segoe UI", "PingFang SC", sans-serif; }}
  header {{ padding: 28px 32px 8px; border-bottom: 1px solid #1f2430; }}
  h1 {{ margin: 0 0 6px; font-size: 21px; }}
  .note {{ color: #8b93a5; font-size: 13px; }}
  section {{ padding: 18px 32px; }}
  h2 {{ font-size: 17px; margin: 8px 0 2px; color: #9ecbff; }}
  .blurb {{ color: #8b93a5; margin: 2px 0 12px; font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 18px; }}
  .card {{ background: #141926; border: 1px solid #222a3a; border-radius: 10px; overflow: hidden; }}
  video {{ width: 100%; display: block; background: #000; aspect-ratio: 640/384; }}
  .meta {{ padding: 10px 14px 12px; }}
  .prompt {{ font-size: 13.5px; }}
  .metrics {{ margin-top: 6px; font-size: 12.5px; color: #7ee2a8; }}
  .sub {{ margin-top: 6px; font-size: 12px; color: #6b7385; }}
</style>
</head>
<body>
<header>
  <h1>Helios-Echo · P2-interim 长视频漂移评测(90.75s / 66 sections / seed 配对)</h1>
  <div class="note">四臂对照:off 基线漂移 → on 未训练记忆静态塌缩 → Stage A 训练后记忆(两 ckpt)。指标为逐 chunk 首末 10 段均值与 Theil-Sen 斜率。</div>
</header>
{''.join(sections)}
<footer style="padding:20px 32px;color:#4c5364;font-size:12px">generated by scripts/evaluation/build_p2_site.py · helios-echo</footer>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "p2_site")
    args = parser.parse_args()

    cards = collect_cards()
    if not cards:
        raise SystemExit("no manifests found under results/p2_interim")
    videos_dir = args.out / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    for c in cards:
        dst = videos_dir / c["video_name"]
        if not dst.exists() or dst.stat().st_size != c["video_src"].stat().st_size:
            shutil.copy2(c["video_src"], dst)
    (args.out / "index.html").write_text(render(cards))
    total = sum(f.stat().st_size for f in videos_dir.iterdir()) / 1e6
    print(f"SITE_OK cards={len(cards)} out={args.out} videos={total:.0f}MB")


if __name__ == "__main__":
    main()
