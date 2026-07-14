"""Compute the findings numbers from a probe JSON. Usage: python -m helios.analysis.summarize <json>"""
import json, sys
from collections import defaultdict

CATS = ["sink", "short_highres", "mid", "long_lowres", "noisy"]
HIST = CATS[:4]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main(path):
    d = json.load(open(path))
    R = d["by_step_layer"]
    meta = d["meta"]
    prompts = sorted(set(r["prompt"] for r in R))
    layers = sorted(set(r["layer"] for r in R))
    chunks = sorted(set(r["chunk"] for r in R))
    print(f"== {meta['checkpoint']} | {meta['resolution']} | {meta['num_layers']}L x {meta['num_heads']}H | "
          f"steps={meta['num_inference_steps']} recorded={meta['recorded_steps']} ==")
    print(f"prompts={prompts} chunks={chunks} n_records={len(R)}")

    def agg(recs, field):
        acc = {c: 0.0 for c in CATS}
        n = 0
        for r in recs:
            n += 1
            for c in CATS:
                acc[c] += r[field][c]
        return {c: acc[c] / max(n, 1) for c in CATS}

    print("\n-- GLOBAL mean MASS (share%) --")
    g = agg(R, "mass")
    tot = sum(g.values())
    for c in CATS:
        print(f"   {c:14s} {g[c]:.4f}  ({g[c]/tot*100:4.1f}%)")
    print(f"   history_share = {sum(g[c] for c in HIST)/tot:.3f}")

    print("\n-- GLOBAL mean PER-TOKEN (x1e5) ranking --")
    pt = agg(R, "per_token")
    for c, v in sorted(pt.items(), key=lambda kv: -kv[1]):
        print(f"   {c:14s} {v*1e5:7.3f}")

    print("\n-- history_share by LAYER (mean over chunks>=1, steps, prompts) --")
    byL = defaultdict(list)
    for r in R:
        if r["chunk"] >= 1:
            byL[r["layer"]].append(r["history_share"])
    hs_layer = {l: mean(byL[l]) for l in layers}
    peak = max(hs_layer, key=hs_layer.get)
    for l in layers:
        if l % 4 == 0 or l == peak:
            bar = "#" * int(hs_layer[l] * 200)
            print(f"   L{l:2d} {hs_layer[l]:.3f} {bar}{'  <-- peak' if l==peak else ''}")
    print(f"   PEAK history_share at layer {peak} = {hs_layer[peak]:.3f}; "
          f"L0={hs_layer[layers[0]]:.3f} Llast={hs_layer[layers[-1]]:.3f}")

    print("\n-- history_share by CHUNK (mean over layers, steps, prompts) --")
    byC = defaultdict(list)
    for r in R:
        byC[r["chunk"]].append(r["history_share"])
    for c in chunks:
        sec = c * 33 / 24
        print(f"   chunk {c} (~{sec:4.1f}s): {mean(byC[c]):.3f}")

    print("\n-- per-category MASS share by layer band --")
    for band, (lo, hi) in [("early L0-9", (0, 9)), ("mid L10-29", (10, 29)), ("late L30-39", (30, 39))]:
        recs = [r for r in R if lo <= r["layer"] <= hi and r["chunk"] >= 1]
        a = agg(recs, "mass"); t = sum(a.values())
        print(f"   {band:14s}: " + "  ".join(f"{c.split('_')[0]}={a[c]/t*100:4.1f}%" for c in CATS))

    print("\n-- per-token: is recent-hi-res / sink individually >= noisy? (mid-layers, chunk>=1) --")
    recs = [r for r in R if 10 <= r["layer"] <= 29 and r["chunk"] >= 1]
    a = agg(recs, "per_token")
    print("   " + "  ".join(f"{c.split('_')[0]}={a[c]*1e5:.2f}" for c in CATS))
    print(f"   recent/noisy per-token ratio = {a['short_highres']/a['noisy']:.2f}, "
          f"sink/noisy = {a['sink']/a['noisy']:.2f}, long/noisy = {a['long_lowres']/a['noisy']:.3f}")

    print("\n-- entropy (nats) by layer band, mean --")
    for band, (lo, hi) in [("early", (0, 9)), ("mid", (10, 29)), ("late", (30, 39))]:
        print(f"   {band}: {mean(r['entropy'] for r in R if lo<=r['layer']<=hi):.2f}")

    print("\n-- cross-prompt consistency: history_share peak layer per prompt --")
    for p in prompts:
        bl = defaultdict(list)
        for r in R:
            if r["prompt"] == p and r["chunk"] >= 1:
                bl[r["layer"]].append(r["history_share"])
        pk = max(layers, key=lambda l: mean(bl[l]))
        print(f"   {p:8s}: peak L{pk} ({mean(bl[pk]):.3f}), global hist_share={mean(r['history_share'] for r in R if r['prompt']==p and r['chunk']>=1):.3f}")

    print("\n-- head specialization: at peak layer, top sink-head vs mean (chunk>=1) --")
    import statistics
    ph_sink = defaultdict(list)
    for r in R:
        if r["layer"] == peak and r["chunk"] >= 1 and r.get("per_head_mass"):
            for h, v in enumerate(r["per_head_mass"]["sink"]):
                ph_sink[h].append(v)
    if ph_sink:
        means = {h: mean(v) for h, v in ph_sink.items()}
        top = max(means, key=means.get)
        print(f"   layer {peak}: max-sink head #{top}={means[top]:.3f} vs head-mean={statistics.mean(means.values()):.4f} "
              f"(ratio {means[top]/max(statistics.mean(means.values()),1e-9):.1f}x)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "helios/analysis/out/base_full.json")
