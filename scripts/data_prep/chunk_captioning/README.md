# Chunk 对齐 Gemini 标注（30k 子集）

实现 `docs/CHUNK_ALIGNED_CAPTIONING.md`：对 368x640 latent 语料随机采样的 30k 条视频，
按 **1 caption = 1 个 33 帧 bidirectional chunk** 做精细标注。caption 沿用训练语料的
HTML-tag schema（`<header><event><role><Background><style><scene>`，同
`live_model_data_preprocessing/captioning/prompts.json`）：header/role/Background/style
为全片共享（保证实体 <ID_x> 命名一致），event/scene 为每 chunk 独立。

数据根目录：`/mnt/beegfs/dataset/video_single_24FPS/chunk_captions_gemini_30k/`
- `latent_listing.txt` — 429,634 条 368x640 latent 文件名（采样总体）
- `sample_30k.jsonl` — seed 1001 采出的 30k（158,292 chunks，均值 5.28/clip）；
  字段 `clip_id / uttid / video / cut / n_chunks / hint`（hint = v3 manifest 里的旧 caption）
- `shards/part-*.jsonl` — 标注输出（原子追加、断点续跑）
- `captions_chunks.jsonl` — merge 后的最终结果

## 跑

```bash
export GEMINI_API_KEY=...          # 不落盘、不提交
cd scripts/data_prep/chunk_captioning
PY=/mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python
export PYTHONPATH=/mnt/beegfs/xiangbo/pylibs_genai   # google-genai 装在这个 overlay

# 1) 采样（已跑过，重跑同 seed 结果一致）
$PY sample_30k.py --listing .../latent_listing.txt \
    --manifest /mnt/beegfs/dataset/video_single_24FPS/train_manifest_combined_v3.jsonl \
    --out .../sample_30k.jsonl --n 30000 --seed 1001

# 2) 先导（20 条，人工过目 captions 质量）
$PY annotate_gemini_chunks.py --sample .../sample_30k.jsonl --outdir .../shards \
    --limit 20 --concurrency 4

# 3) 全量（可多进程 --shard i --num_shards N；重复跑自动跳过已完成 clip）
$PY annotate_gemini_chunks.py --sample .../sample_30k.jsonl --outdir .../shards \
    --concurrency 24

# 4) 合并 + 补漏
$PY merge.py --sample .../sample_30k.jsonl --shards .../shards --out .../captions_chunks.jsonl
```

## 设计要点

- **帧网格**：chunk k = CFR 视频帧（逐 clip 恒定 fps，25 为主，非统一 24——一切按帧号，不按秒） `[cut_start + 33k, cut_start + 33(k+1))`，
  每 chunk 抽 4 帧（offset 0/11/22/32），带 "CHUNK k" 标签交错送入 Gemini —— 与训练
  编码使用同一帧数学，无时间戳歧义。**不要**把结果挂到旧 latent 上（旧 latent 有
  未记录的随机起点偏移，见 doc 坑 #11）；训练前需按 `start=cut_start` 重新编码。
- **一条视频一次调用**，structured JSON output（response_schema 强制），chunk 数
  不等于 n_chunks 视为失败并重试。
- **措辞纪律**在 prompt 里硬性规定（无 then/begins/continues、每条自足、<ID_x> 一致），
  另有 soft_lint 把违规打进 `flags` 字段便于事后统计。
- 输出字段：`captions`（组装好的每 chunk HTML caption 列表，训练直接用）+
  `chunk_events/chunk_scenes/header/role/background/style`（原始件，便于重组/去重）。
- 成本（gemini-2.5-flash，帧长边 512）：每 clip ≈ 4×n_chunks×~258 输入 token + ~1-2k
  输出；30k clip（约 63.3 万帧）估算 **$60–120 美元级**，先导会给出实测 token 数。
