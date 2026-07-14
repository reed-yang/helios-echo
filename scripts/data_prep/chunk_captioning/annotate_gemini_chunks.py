#!/usr/bin/env python3
"""Chunk-aligned Gemini captioning (1 caption = 1 bidirectional 33-frame chunk).

Implements docs/CHUNK_ALIGNED_CAPTIONING.md over the sample produced by
sample_30k.py. One Gemini call per clip:

  * frames are extracted at EXACT indices of the training chunk grid
    (chunk k = RGB frames [cut_start + 33k, cut_start + 33(k+1)), 24fps CFR),
    4 frames per chunk (offsets 0/11/22/32), interleaved with "CHUNK k" labels;
  * the response is structured JSON: global static fields (header / role /
    background / style, shared by all chunks so entity naming is consistent)
    + one {event, scene} pair PER chunk;
  * the final per-chunk caption is assembled offline in the training HTML-tag
    schema: shared header/role/Background/style + that chunk's event/scene.

Two backends (--backend):
  forge  (default): OpenAI-compatible gateway api.forge.tensorblock.co/v1, model
         tensorblock/gemini-3-flash-preview. Key: FORGE_API_KEY env or
         /mnt/beegfs/xiangbo/.config/forge_api_key.
  google: native google-genai SDK. Key: GEMINI_API_KEY / GOOGLE_API_KEY env.

Run (pilot):
  PYTHONPATH=/mnt/beegfs/xiangbo/pylibs_genai \
  /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python annotate_gemini_chunks.py \
      --sample /mnt/beegfs/dataset/video_single_24FPS/chunk_captions_gemini_30k/sample_30k.jsonl \
      --outdir /mnt/beegfs/dataset/video_single_24FPS/chunk_captions_gemini_30k/shards \
      --limit 20 --concurrency 4

Production: drop --limit, raise --concurrency (16-32), optionally split with
--shard i/N across processes/machines. Output is one JSONL per (shard,run),
atomic appends, resumable (clips already OK in the outdir are skipped).
"""
import base64
import argparse
import asyncio
import io
import json
import os
import re
import sys
import time
from pathlib import Path

# NOTE: everything here is FRAME-indexed, never time-indexed. The corpus is per-clip
# constant-fps but NOT uniformly 24 fps (mostly 25, also 24/30/...), so a 33-frame
# chunk is ~1.1-1.4 s depending on the clip — never hardcode seconds.
FRAMES_PER_CHUNK = 33

SYSTEM_PROMPT = (
    "You are a professional video annotator building a training corpus for an "
    "autoregressive text-to-video model that generates video in consecutive "
    "short chunks of 33 frames (roughly 1.1-1.4 seconds each). You annotate "
    "each chunk with exactly what is visible in it. Describe only concrete "
    "visual content, in present tense. Never hedge (no \"appears to\", "
    "\"possibly\", \"seems\") and never use meta-framing like \"the video "
    "shows\" or \"in this chunk\"."
)

INSTRUCTIONS_TMPL = """The clip above is divided into {n} consecutive chunks of 33 frames each (roughly 1.1-1.4 s per chunk depending on the clip's frame rate). For each chunk you saw 4 frames in temporal order (start / one-third / two-thirds / end of the chunk), preceded by a "CHUNK k" label.

An automatic (often noisy) description of the whole clip is provided as a HINT only — trust the frames over the text, and silently drop anything the frames do not support:
\"\"\"{hint}\"\"\"

Return a JSON object with these fields:

- "header": definition and speed tags for the whole clip. Definition: "high definition" if sharp/clean else "low definition". Speed: "high speed" only if sped-up/time-lapsed, else "normal speed". Optionally add other obvious capture tags (e.g. "vertical", "black and white").
- "role": first state how many distinct entities (people/animals/salient agents) appear anywhere in the clip. Then each one's STATIC appearance only (no actions), naming each with an id tag <ID_1>, <ID_2>, ... These ids are the ONLY way you refer to entities later.
- "background": the environment across the clip: location, props, scenery, lighting, on-screen text, weather, time of day.
- "style": "realistic" or "unrealistic" (animation/CGI/stylized), plus a few words on visual style.
- "chunks": an array of EXACTLY {n} objects, one per chunk in order. Each object:
    - "event": ONE self-contained sentence: which entity (<ID_x>) does what, where, during THIS chunk only.
    - "scene": cinematography of this chunk (shot scale, camera angle, camera motion, depth of field), then the subjects' motion/action WITHIN this chunk, referring to entities as <ID_x>, then finer detail.

Hard rules for "chunks":
1. Describe ONLY what happens inside that chunk's own time window. Do not mention earlier or later chunks and do not use connective words like "then", "next", "begins to", "continues", "still", "again", "starts", "finishes".
2. Every chunk entry must be understandable on its own, as if it were the only prompt a video model receives for that 1.375 s.
3. Use the same <ID_x> ids consistently in every chunk.
4. If the content barely changes between chunks, that is fine — describe the near-static action plainly (it is OK for neighboring entries to be similar).
5. If the action visibly changes in the middle of a chunk, describe the chunk as it is, including the change (e.g. "<ID_1> lifts the cup and turns her head to the left").
6. The array must have exactly {n} entries."""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "header": {"type": "STRING"},
        "role": {"type": "STRING"},
        "background": {"type": "STRING"},
        "style": {"type": "STRING"},
        "chunks": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "event": {"type": "STRING"},
                    "scene": {"type": "STRING"},
                },
                "required": ["event", "scene"],
            },
        },
    },
    "required": ["header", "role", "background", "style", "chunks"],
}

FRAME_OFFSETS = (0, 11, 22, 32)


def extract_chunk_frames(video_path, cut_start, n_chunks, max_side):
    """Decode the exact chunk-grid frames and JPEG-encode them.

    Returns list of (chunk_idx, jpeg_bytes) in temporal order.
    """
    from PIL import Image
    from video_reader import PyVideoReader

    indices = []
    for k in range(n_chunks):
        base = cut_start + k * FRAMES_PER_CHUNK
        indices.extend(base + off for off in FRAME_OFFSETS)
    vr = PyVideoReader(video_path, threads=0)
    frames = vr.get_batch(indices)  # (N, H, W, 3) uint8
    out = []
    for i, arr in enumerate(frames):
        img = Image.fromarray(arr)
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        out.append((i // len(FRAME_OFFSETS), buf.getvalue()))
    return out


def _intro_text(rec):
    return (
        f"A video clip of {rec['n_chunks'] * FRAMES_PER_CHUNK} frames, shown as "
        f"{rec['n_chunks']} chunks x {len(FRAME_OFFSETS)} sampled frames:"
    )


def _extract_json(text):
    """Parse model output as JSON, tolerating markdown fences / leading prose."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def build_openai_messages(rec, frames):
    """OpenAI-compatible chat messages (forge backend), images as data URLs."""
    content = [{"type": "text", "text": _intro_text(rec)}]
    cur = -1
    for chunk_idx, jpeg in frames:
        if chunk_idx != cur:
            content.append({"type": "text", "text": f"CHUNK {chunk_idx}:"})
            cur = chunk_idx
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
                },
            }
        )
    content.append(
        {
            "type": "text",
            "text": INSTRUCTIONS_TMPL.format(
                n=rec["n_chunks"], hint=rec.get("hint", "") or "(none)"
            )
            + "\n\nReturn ONLY the JSON object, no markdown fences, no commentary.",
        }
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_contents(rec, frames, types):
    parts = [types.Part.from_text(text=_intro_text(rec))]
    cur = -1
    for chunk_idx, jpeg in frames:
        if chunk_idx != cur:
            parts.append(types.Part.from_text(text=f"CHUNK {chunk_idx}:"))
            cur = chunk_idx
        parts.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
    parts.append(
        types.Part.from_text(
            text=INSTRUCTIONS_TMPL.format(n=rec["n_chunks"], hint=rec.get("hint", "") or "(none)")
        )
    )
    return parts


def _as_text(x):
    """The model occasionally returns a field as a list (or nested values)
    despite the schema — coerce everything to one clean string."""
    if isinstance(x, (list, tuple)):
        return " ".join(_as_text(v) for v in x if v is not None).strip()
    if isinstance(x, dict):
        return " ".join(_as_text(v) for v in x.values() if v is not None).strip()
    return str(x).strip()


def assemble_captions(data):
    """Global fields + per-chunk event/scene -> per-chunk HTML-schema captions."""
    caps = []
    for ch in data["chunks"]:
        caps.append(
            f"<header>{_as_text(data['header'])}</header>\n"
            f"<event>{_as_text(ch['event'])}</event>\n"
            f"<role>{_as_text(data['role'])}</role>\n"
            f"<Background>{_as_text(data['background'])}</Background>\n"
            f"<style>{_as_text(data['style'])}</style>\n"
            f"<scene>{_as_text(ch['scene'])}</scene>"
        )
    return caps


BANNED_CONNECTIVES = re.compile(
    r"\b(then|begins? to|beginning to|continues?|continuing|starts? to|next,)\b", re.I
)


def soft_lint(data):
    """Non-fatal quality flags recorded alongside the output."""
    flags = []
    n_conn = sum(
        1
        for ch in data["chunks"]
        if BANNED_CONNECTIVES.search(ch["event"] + " " + ch["scene"])
    )
    if n_conn:
        flags.append(f"connectives_in_{n_conn}_chunks")
    if "<ID_1>" not in data["role"]:
        flags.append("no_id_tags_in_role")
    return flags


async def annotate_one(client, types, rec, args, decode_sem):
    t0 = time.time()
    out = {
        "clip_id": rec["clip_id"],
        "uttid": rec["uttid"],
        "video": rec["video"],
        "cut": rec["cut"],
        "n_chunks": rec["n_chunks"],
        "model": args.model,
        "error": "",
        "flags": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    try:
        async with decode_sem:
            frames = await asyncio.to_thread(
                extract_chunk_frames, rec["video"], rec["cut"][0], rec["n_chunks"], args.max_side
            )
    except Exception as exc:
        out["error"] = f"decode:{type(exc).__name__}: {str(exc)[:160]}"
        out["latency_s"] = round(time.time() - t0, 2)
        return out

    if args.backend == "google":
        contents = build_contents(rec, frames, types)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=args.temperature,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            max_output_tokens=args.max_output_tokens,
        )
    else:
        messages = build_openai_messages(rec, frames)

    data, last_err = None, ""
    for attempt in range(args.retries):
        try:
            if args.backend == "google":
                resp = await client.aio.models.generate_content(
                    model=args.model, contents=contents, config=config
                )
                if resp.usage_metadata:
                    out["prompt_tokens"] = resp.usage_metadata.prompt_token_count or 0
                    out["completion_tokens"] = resp.usage_metadata.candidates_token_count or 0
                text = resp.text or ""
            else:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_output_tokens,
                    response_format={"type": "json_object"},
                )
                if resp.usage:
                    out["prompt_tokens"] = resp.usage.prompt_tokens or 0
                    out["completion_tokens"] = resp.usage.completion_tokens or 0
                text = resp.choices[0].message.content or ""
            cand = _extract_json(text)
            if len(cand.get("chunks", [])) != rec["n_chunks"]:
                last_err = f"chunk_count:{len(cand.get('chunks', []))}!={rec['n_chunks']}"
                continue
            missing = [k for k in ("header", "role", "background", "style") if not cand.get(k)]
            if missing:
                last_err = f"missing_fields:{','.join(missing)}"
                continue
            for k in ("header", "role", "background", "style"):
                cand[k] = _as_text(cand[k])
            for ch in cand["chunks"]:
                ch["event"] = _as_text(ch.get("event", ""))
                ch["scene"] = _as_text(ch.get("scene", ""))
            data = cand
            break
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {str(exc)[:160]}"
            # generous backoff for 429 / transient 5xx
            await asyncio.sleep(min(5 * (attempt + 1), 30))

    if data is None:
        out["error"] = last_err or "no_response"
    else:
        out.update(
            header=data["header"],
            role=data["role"],
            background=data["background"],
            style=data["style"],
            chunk_events=[c["event"] for c in data["chunks"]],
            chunk_scenes=[c["scene"] for c in data["chunks"]],
            captions=assemble_captions(data),
        )
        out["flags"] = soft_lint(data)
    out["latency_s"] = round(time.time() - t0, 2)
    return out


def load_done(outdir):
    done = set()
    for p in Path(outdir).glob("part-*.jsonl"):
        with p.open() as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("captions") and not d.get("error"):
                    done.add(d["clip_id"])
    return done


FORGE_BASE_URL = "https://api.forge.tensorblock.co/v1"
FORGE_KEY_FILE = "/mnt/beegfs/xiangbo/.config/forge_api_key"


def _make_client(args):
    if args.backend == "google":
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            sys.exit("ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        return genai.Client(api_key=api_key), types
    from openai import AsyncOpenAI

    api_key = os.environ.get("FORGE_API_KEY")
    if not api_key and os.path.exists(FORGE_KEY_FILE):
        api_key = open(FORGE_KEY_FILE).read().strip()
    if not api_key:
        sys.exit(f"ERROR: set FORGE_API_KEY or create {FORGE_KEY_FILE}.")
    return AsyncOpenAI(api_key=api_key, base_url=FORGE_BASE_URL, timeout=300.0), None


async def main_async(args):
    client, types = _make_client(args)

    records = [json.loads(l) for l in open(args.sample)]
    if args.num_shards > 1:
        records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard]
    done = load_done(args.outdir)
    todo = [r for r in records if r["clip_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"shard {args.shard}/{args.num_shards}: {len(records)} records, "
          f"{len(done)} done, {len(todo)} to run")
    if not todo:
        return

    os.makedirs(args.outdir, exist_ok=True)
    out_path = Path(args.outdir) / f"part-{args.shard:02d}-{int(time.time())}.jsonl"
    fh = open(out_path, "a")
    write_lock = asyncio.Lock()
    api_sem = asyncio.Semaphore(args.concurrency)
    decode_sem = asyncio.Semaphore(args.decode_workers)
    stat = {"ok": 0, "err": 0, "ptok": 0, "ctok": 0, "t0": time.time()}

    async def run_one(rec):
        async with api_sem:
            try:
                out = await annotate_one(client, types, rec, args, decode_sem)
            except Exception as exc:  # belt-and-braces: one bad record must not kill the run
                out = {
                    "clip_id": rec["clip_id"], "uttid": rec["uttid"], "video": rec["video"],
                    "cut": rec["cut"], "n_chunks": rec["n_chunks"], "model": args.model,
                    "error": f"unhandled:{type(exc).__name__}: {str(exc)[:160]}",
                    "flags": [], "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
                }
        async with write_lock:
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            stat["ok" if not out["error"] else "err"] += 1
            stat["ptok"] += out["prompt_tokens"]
            stat["ctok"] += out["completion_tokens"]
            n = stat["ok"] + stat["err"]
            if n % 25 == 0 or n == len(todo):
                dt = time.time() - stat["t0"]
                print(f"[{n}/{len(todo)}] ok={stat['ok']} err={stat['err']} "
                      f"tok(in/out)={stat['ptok']}/{stat['ctok']} "
                      f"{n/dt*60:.1f} clips/min", flush=True)

    await asyncio.gather(*(run_one(r) for r in todo))
    fh.close()
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--backend", choices=["forge", "google"], default="forge")
    ap.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "tensorblock/gemini-3-flash-preview"),
        help="forge default: tensorblock/gemini-3-flash-preview; for --backend google use e.g. gemini-2.5-flash",
    )
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max_output_tokens", type=int, default=8192)
    ap.add_argument("--max_side", type=int, default=512, help="resize frames so long side <= this")
    ap.add_argument("--concurrency", type=int, default=8, help="concurrent Gemini requests")
    ap.add_argument("--decode_workers", type=int, default=4, help="concurrent video decodes")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
