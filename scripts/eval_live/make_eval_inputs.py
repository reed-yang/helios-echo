#!/usr/bin/env python3
"""Split the eval inputs into ONE-ITEM-PER-FILE units so the queue scheduler can run each video on a
single GPU (world_size=1, no NCCL) and keep every GPU busy.

  Group A (no-switch, 2-min t2v): prompts5.txt (5 lines) -> qa/0.txt .. qa/4.txt  (one prompt each).
       infer_helios writes line-index 0.mp4 inside its --output_folder; the worker renames -> <i>.mp4.
  Group B (event-switch, 1-min):  evtsw.csv -> qb/<id>.csv  (header + that id's 6 event rows).
       infer_helios writes <id>.mp4 directly.

Outputs under eval_prompts_cfr/{qa,qb}/. Idempotent.
"""
import csv, os

BASE = "/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr"
PROMPTS_A = os.path.join(BASE, "prompts5.txt")
CSV_B = os.path.join(BASE, "evtsw.csv")
QA = os.path.join(BASE, "qa")
QB = os.path.join(BASE, "qb")


def main():
    os.makedirs(QA, exist_ok=True)
    os.makedirs(QB, exist_ok=True)

    # Group A: one prompt per file
    with open(PROMPTS_A) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i, line in enumerate(lines):
        with open(os.path.join(QA, f"{i}.txt"), "w") as g:
            g.write(line + "\n")
    print(f"group A: wrote {len(lines)} single-prompt files -> {QA}")

    # Group B: one id per CSV (header + that id's rows, preserving prompt_index order)
    rows = list(csv.DictReader(open(CSV_B)))
    ids = []
    seen = set()
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"]); ids.append(r["id"])
    for vid in ids:
        sub = [r for r in rows if r["id"] == vid]
        sub.sort(key=lambda r: int(r["prompt_index"]))
        with open(os.path.join(QB, f"{vid}.csv"), "w", newline="") as g:
            w = csv.writer(g)
            w.writerow(["id", "prompt_index", "prompt"])
            for r in sub:
                w.writerow([r["id"], r["prompt_index"], r["prompt"]])
    print(f"group B: wrote {len(ids)} single-id CSVs ({len(rows)} rows total) -> {QB}")
    print("ids:", ids)


if __name__ == "__main__":
    main()
