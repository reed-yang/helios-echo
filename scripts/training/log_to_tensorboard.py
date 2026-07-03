#!/usr/bin/env python3
"""Export Helios training metrics (loss / grad_norm / lr) from a train_*.out log to
TensorBoard event files, so they can be viewed without wandb cloud / a login.

The trainer logs every step's metrics three ways: the tqdm postfix (in the .out log),
and `accelerator.log(...)` -> the offline wandb run. This parses the .out log (the most
robust, dependency-free source) and writes scalars to a TensorBoard logdir.

Idempotent: rewrites the whole logdir each run (cheap — a few thousand points), so just
re-run it to pick up new steps. View with:
    tensorboard --logdir <out-dir> --port 6006 --bind_all

Usage:
    python log_to_tensorboard.py --log /mnt/beegfs/xiangbo/helios_runs/logs/train_vs24_1290.out \
                                 --out /mnt/beegfs/xiangbo/helios_runs/tb_vs24
"""
import argparse
import os
import re
import shutil

from torch.utils.tensorboard import SummaryWriter

# matches e.g.  "1551/9900 [12:10:13<66:35:46, 28.72s/it, grad_norm=0.00667, loss=0.171, lr=3e-05]"
STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[")
KV_RE = re.compile(r"(grad_norm|loss|lr)=([0-9.eE+-]+)")


def parse(log_paths):
    """Merge (step -> {metric: value}) across one or more logs; later files/lines win
    (so a resumed job's re-run steps overwrite the original job's)."""
    rows = {}
    raw = ""
    for p in log_paths:
        try:
            with open(p, "rb") as f:
                raw += f.read().decode("utf-8", "ignore") + "\n"
        except FileNotFoundError:
            continue
    # tqdm uses \r to overwrite; split on both so every update becomes a line
    for line in re.split(r"[\r\n]", raw):
        if "grad_norm=" not in line:  # training bar only (skip the step-0 validation bar)
            continue
        m = STEP_RE.search(line)
        if not m:
            continue
        step, total = int(m.group(1)), int(m.group(2))
        if total < 100 or step > total:  # exclude /50 val bar and truncated/garbled lines
            continue
        kvs = {}
        for k, v in KV_RE.findall(line):
            try:
                kvs[k] = float(v)
            except ValueError:
                continue  # truncated tqdm refresh (e.g. 'lr=3e-')
        if not kvs:
            continue
        # merge so a truncated later refresh of a step doesn't drop earlier-parsed keys
        rows.setdefault(step, {}).update(kvs)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, nargs="+", help="train_*.out log file(s), in chronological order")
    ap.add_argument("--out", required=True, help="tensorboard logdir")
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        print(f"no parseable steps in {args.log}")
        return
    # fresh logdir each run so there are no duplicate/stale scalars
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)
    w = SummaryWriter(args.out)
    metrics = set()
    for step in sorted(rows):
        for k, v in rows[step].items():
            w.add_scalar(f"train/{k}", v, step)
            metrics.add(k)
    w.flush()
    w.close()
    lo, hi = min(rows), max(rows)
    print(f"wrote {len(rows)} steps (step {lo}..{hi}) metrics={sorted(metrics)} -> {args.out}")


if __name__ == "__main__":
    main()
