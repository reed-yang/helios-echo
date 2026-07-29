#!/usr/bin/env python3
"""Build the static preview site for P2 long-video A/B results.

Two pages share one videos/ directory:
  index.html  — event-switch protocol (r4): the memory-critical scenario
  static.html — static-prompt drift protocol (r3), kept as the sub-page

Scans results/p2_interim/<run_id>/<arm>/prompt_*.manifest.json plus matching
metrics JSONs. Only campaigns listed in a page's meta appear (voided raw-prompt
campaigns stay off both pages). Re-run after every campaign.

Usage: python scripts/evaluation/build_p2_site.py [--out results/p2_site]
"""

import argparse
import html
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P2_ROOT = REPO / "results" / "p2_interim"

R4_META = {
    "p2-rep50-r4evsw-off": {
        "off": ("r4 · off(无记忆基线)", "6 事件 × 7 sections 硬切换。切换后旧事件内容离开历史窗口——无记忆时一致性只能靠运气。"),
    },
    "p2-rep50-r4evsw-on-untrained": {
        "on": ("r4 · on(未训练记忆)", "全新 M₀ 参与读写。对照未训练先验在切换场景的行为。"),
    },
    "p2-rep50-r4evsw-on-pilot1500": {
        "on": ("r4 · on(训练后记忆 pilot@1500)", "Stage A 1500 步。记忆是跨切换保持人物/背景一致性的唯一通道——核心价值场景。"),
    },
    "p2-rep50-r4evsw-on-pilot2000": {
        "on": ("r4+ · on(pilot@2000,步数演化)", "2000 步:饱和斜率 −0.09(五臂中最接近零,优于 off 基线 −0.34),motion 最活跃。"),
    },
    "p2-rep50-r4evsw-on-slice512-2000": {
        "on": ("r4+ · on(slice512@2000,语料对照)", "57k 语料同步数对照:贴合基线(−0.38),零病理;校准进度落后于多轮复读的 pilot。"),
    },
    "p2-rep50-r4evsw-on-pilot4000": {
        "on": ("r4★ · on(pilot@4000,训练终态)", "Stage A 终态验收:pilot 完整 4000 步。与 1500/2000 演化链对照。"),
    },
    "p2-rep50-r4evsw-on-slice512-4000": {
        "on": ("r4★ · on(slice512@4000,训练终态)", "Stage A 终态验收:57k 语料完整 4000 步(~2.2 epochs)。"),
    },
    "p2-rep50-r4evsw-on-full4000": {
        "on": ("r4◆ · on(full@4000,全量语料 ~0.76ep)", "168k 全量语料 4000 步:斜率 −0.19 优于基线与 slice512——语料多样性可部分替代重复暴露。"),
    },
    "p2-rep50-r4evsw-on-full8000": {
        "on": ("r4◆ · on(full@8000,~1.52ep)", "斜率 −0.26:未复现 pilot 式单调归零,仍优于基线,零病理;@12000 为决定点。"),
    },
}
R3_META = {
    "p2-rep50-r5long-off": {
        "off": ("r5 ★ 8 分钟 · off(基线纯漂移)", "349 sections = 11,517 帧 = 8.0 分钟单 prompt 长滚动。漂移的完整发展过程,与右下记忆臂逐位种子配对。"),
    },
    "p2-rep50-r5long-on-pilot4000": {
        "on": ("r5 ★ 8 分钟 · on(pilot@4000 终态记忆)", "同 case 同 seed 的记忆臂:347 次记忆写入(迄今最长状态机演化)。对照观察 8 分钟尺度的画面保持。"),
    },
    "p2-rep50-r5long-on-full8000": {
        "on": ("r5 ◆ 8 分钟 · on(full@8000,全量语料)", "多样性谱系的 8 分钟回复力检验:与上两组同 case 同 seed 配对。"),
    },
    "p2-rep50-r3-off": {
        "off": ("r3 · off(无记忆基线)", "rep50 结构化 prompt,静态单 prompt 90.75s。分布内基线漂移温和。"),
    },
    "p2-rep50-r3-on-untrained": {
        "on": ("r3 · on(未训练记忆)", "3/8 case 静态冻结,1 case 饱和爆冲——未训练记忆有害。"),
    },
    "p2-rep50-r3-on-pilot1000": {
        "on": ("r3 · on(训练后记忆 pilot@1000)", "冻结全部消除(8/8 存活),指标与基线持平。"),
    },
    "p2-rep50-r3-on-pilot4000": {
        "on": ("r3★ · on(pilot@4000,训练终态)", "Stage A 终态验收:静态协议下的 4000 步终态行为。"),
    },
    "p2-rep50-r3-on-slice512-4000": {
        "on": ("r3★ · on(slice512@4000,训练终态)", "Stage A 终态验收:大语料终态,静态协议。"),
    },
}

PAGES = {
    "index.html": {
        "meta": R4_META,
        # Auto-discover any compliant campaign of this protocol family so new
        # checkpoints appear on rebuild without manual registration. Voided
        # raw-prompt campaigns (p2-interim-*, p2-dress) never match.
        "auto": lambda rid: rid.startswith("p2-rep50-") and "r4evsw" in rid,
        "title": "Helios-Echo · Event-Switch 评测(6 事件硬切换 · 57.75s)",
        "note": "记忆主场:每 7 个 section 硬切换事件 prompt,旧事件内容滑出历史窗口后,跨段一致性只能来自演进记忆。三臂同 case 同 seed 配对。",
        "other": ('static.html', '→ 静态 prompt 漂移协议(r3)子页'),
    },
    "static.html": {
        "meta": R3_META,
        "auto": lambda rid: rid.startswith("p2-rep50-") and "r4evsw" not in rid,
        "title": "Helios-Echo · 静态 Prompt 漂移评测(r3 · 90.75s / r5 · 8min)",
        "note": "单 prompt 长滚动,度量漂移签名(motion / 饱和度轨迹)。rep50 结构化 prompt(段 0)。",
        "other": ('index.html', '← 返回 Event-Switch 主页'),
    },
}


def window_mean(values, start, end):
    chunk = values[start:end]
    return sum(chunk) / len(chunk) if chunk else float("nan")


def prompt_html(prompt):
    """Structural prompt(s): show <event> summaries, full structural text folded.

    The model receives the FULL structural prompt (header/event/role/Background/
    style/scene tags, verbatim from the rep50 CSV — see each manifest); the
    event line shown on the card is a display summary only.
    """
    if isinstance(prompt, list):
        items = []
        for i, seg in enumerate(prompt):
            m = re.search(r"<event>(.*?)</event>", seg, re.S)
            items.append(f"<li><b>E{i}</b> {html.escape((m.group(1) if m else seg).strip())}</li>")
        summary = f"<ol class='events'>{''.join(items)}</ol>"
        full = "<hr>".join(html.escape(seg) for seg in prompt)
    else:
        m = re.search(r"<event>(.*?)</event>", prompt, re.S)
        summary = f"<div class='prompt'>{html.escape((m.group(1) if m else prompt).strip())}</div>"
        full = html.escape(prompt)
    return (
        f"{summary}<details class='fullprompt'><summary>完整结构化 prompt(实际输入)</summary>"
        f"<div class='fulltext'>{full}</div></details>"
    )


def collect_cards(meta, auto=None):
    cards = []
    for manifest_path in sorted(P2_ROOT.glob("*/*/prompt_*.manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        run_id, arm, idx = manifest["run_id"], manifest["arm"], manifest["prompt_index"]
        if run_id in meta and arm in meta[run_id]:
            label, blurb = meta[run_id][arm]
        elif auto is not None and auto(run_id):
            label = f"{run_id.replace('p2-rep50-', '')} · {arm}(自动收录)"
            blurb = "新 campaign,自动上站;判读见对应 verdict 文档。"
        else:
            continue
        video = Path(manifest["artifacts"]["video"])
        if not video.exists():
            continue
        partial = manifest.get("memory_partial")
        # Curated campaigns keep their meta order; auto-discovered ones go last,
        # ordered by run_id so they group deterministically.
        order = list(meta).index(run_id) if run_id in meta else 500
        cards.append({
            "order": order,
            "run_id": run_id, "arm": arm, "idx": idx,
            "label": label, "blurb": blurb,
            "prompt": manifest["prompt"], "seed": manifest["seed"],
            "created": manifest["created_at_utc"][:16].replace("T", " "),
            "ckpt": Path(partial["path"]).parent.name if partial else "—",
            "video_src": video,
            "video_name": f"{run_id.replace('p2-rep50-', '').replace('p2-interim-', '')}_{arm}_p{idx}.mp4",
        })
    cards.sort(key=lambda c: (c["order"], c["run_id"], c["idx"]))
    return cards


def render(cards, page):
    groups = {}
    for c in cards:
        groups.setdefault((c["order"], c["label"], c["blurb"]), []).append(c)

    sections = []
    for (_, label, blurb), items in sorted(groups.items(), key=lambda kv: kv[0][0]):
        tiles = []
        for c in items:
            metric_row = ""
            mp = P2_ROOT / c["run_id"] / "metrics" / f"{c['arm']}_prompt_{c['idx']:02d}.json"
            if mp.exists():
                mm = json.loads(mp.read_text())
                mot, sat = mm["per_chunk"]["motion"], mm["per_chunk"]["saturation"]
                n = len(mot)
                metric_row = (
                    f"<div class='metrics'>motion {window_mean(mot,0,10):.2f} → {window_mean(mot,n-10,n):.2f}"
                    f" · 饱和度 {window_mean(sat,0,10):.0f} → {window_mean(sat,n-10,n):.0f}"
                    f" · 斜率 {mm['slopes']['saturation']['theil_sen']:+.2f}</div>"
                )
            tiles.append(f"""
      <div class="card">
        <video controls preload="metadata" src="videos/{c['video_name']}"></video>
        <div class="meta">
          <div class="case">case {c['idx']}</div>
          {prompt_html(c['prompt'])}
          {metric_row}
          <div class="sub">seed {c['seed']} · ckpt {html.escape(c['ckpt'])} · {c['created']} UTC</div>
        </div>
      </div>""")
        sections.append(f"""
    <section>
      <h2>{html.escape(label)}</h2>
      <p class="blurb">{html.escape(blurb)}</p>
      <div class="grid">{''.join(tiles)}
      </div>
    </section>""")

    other_href, other_text = page["other"]
    return f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>{html.escape(page['title'])}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; background: #0d1017; color: #e6e9ef; font: 15px/1.6 -apple-system, "Segoe UI", "PingFang SC", sans-serif; }}
  header {{ padding: 28px 32px 12px; border-bottom: 1px solid #1f2430; }}
  h1 {{ margin: 0 0 6px; font-size: 21px; }}
  .note {{ color: #8b93a5; font-size: 13px; max-width: 900px; }}
  .navlink {{ display: inline-block; margin-top: 8px; color: #9ecbff; font-size: 13px; text-decoration: none; }}
  section {{ padding: 18px 32px; }}
  h2 {{ font-size: 17px; margin: 8px 0 2px; color: #9ecbff; }}
  .blurb {{ color: #8b93a5; margin: 2px 0 12px; font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 18px; }}
  .card {{ background: #141926; border: 1px solid #222a3a; border-radius: 10px; overflow: hidden; }}
  video {{ width: 100%; display: block; background: #000; aspect-ratio: 640/384; }}
  .meta {{ padding: 10px 14px 12px; }}
  .case {{ color: #d9b96b; font-size: 12px; margin-bottom: 4px; }}
  .prompt {{ font-size: 13.5px; }}
  .events {{ margin: 0; padding-left: 18px; font-size: 12.5px; }}
  .events li {{ margin: 1px 0; }}
  .metrics {{ margin-top: 6px; font-size: 12.5px; color: #7ee2a8; }}
  .fullprompt {{ margin-top: 6px; }}
  .fullprompt summary {{ font-size: 12px; color: #9ecbff; cursor: pointer; }}
  .fulltext {{ margin-top: 4px; font-size: 11.5px; color: #a8b0c0; white-space: pre-wrap; max-height: 220px; overflow-y: auto; background: #0f1420; border-radius: 6px; padding: 8px; }}
  .sub {{ margin-top: 6px; font-size: 12px; color: #6b7385; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(page['title'])}</h1>
  <div class="note">{html.escape(page['note'])}</div>
  <a class="navlink" href="{other_href}">{html.escape(other_text)}</a>
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

    videos_dir = args.out / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    total_cards = 0
    for page_name, page in PAGES.items():
        cards = collect_cards(page["meta"], auto=page.get("auto"))
        for c in cards:
            dst = videos_dir / c["video_name"]
            if not dst.exists() or dst.stat().st_size != c["video_src"].stat().st_size:
                shutil.copy2(c["video_src"], dst)
        (args.out / page_name).write_text(render(cards, page))
        total_cards += len(cards)
        print(f"PAGE {page_name}: {len(cards)} cards")
    print(f"SITE_OK pages={len(PAGES)} cards={total_cards}")


if __name__ == "__main__":
    main()
