# long_video_eval：独立长视频评测模块

`long_video_eval` 是一个独立的长视频生成评测框架。它不把某个 repo 的脚本路径写死在代码里，而是通过 config 显式声明：

- 输入视频在哪里；
- prompt 文件在哪里；
- 要评测哪些模型；
- 要用哪些 GPU；
- 每个 metric 需要哪些代码、权重和 Python 包；
- 每个 metric backend 应该执行什么命令。

这样别人拿到这个目录后，可以在自己的机器上改 config，而不是改代码。

## 设计目标

1. **输入协议清晰且可适配**

   最简单的输入方式是使用标准目录。单个模型也可以评测：

   ```text
   SOURCE_ROOT/
     my_model/videos/0000.mp4
     my_model/videos/0001.mp4
   ```

   其中 `my_model` 只是模型名字，可以是 `base`、`distilled`，也可以是 `krea14b`、`wan21`、`my_method` 等任意名字。prompt 文件按行和视频编号对齐：第 1 行对应 `0000.mp4`。

   如果要横向比较多个模型，可以在同一个 `SOURCE_ROOT` 下放多个模型目录：

   ```text
   SOURCE_ROOT/
     my_model/videos/0000.mp4
     my_model/videos/0001.mp4
     another_model/videos/0000.mp4
     another_model/videos/0001.mp4
   ```

   如果已有输出不长这样，不需要手动重排目录。可以在 config 里显式声明任意视频目录：

   ```json
   {
     "defaults": {
       "models": ["krea14b"],
       "video_roots": {
         "krea14b": "/path/to/krea/output/videos"
       }
     }
   }
   ```

   `run` 时 suite 会在 `OUTPUT_ROOT/_inputs/DATASET_NAME/source_root` 下创建一个标准化 symlink input view，供现有 metric backend 使用；原始视频目录不会被移动或复制。要比较多个模型时，只需要在 `models` 和 `video_roots` 里列出多个 key。

2. **资源显式声明**

   DOVER、HPSv3、PickScore、VBench、HeliosBench 这类 metric 都不是零依赖的。它们需要代码、Python 包和 checkpoint。这个模块不会假装这些东西不存在，而是把它们写进 config 的 `resources` 和 `python_packages`。

   当前 60s/90s 长视频 suite **不默认评测 VideoAlign**。原因是 VideoAlign 更适合短视频或固定帧数输入的 reward / alignment 评估；直接用于 60 秒以上视频时，需要重新决定全量输入还是采样输入、采样多少帧、是否和训练时帧数口径一致，以及显存是否可承受。为了避免把一个不稳定的采样策略混进长视频主评测，这版 suite 先不包含 VideoAlign。

3. **先检查，再安装/下载，再运行**

   推荐流程：

   ```bash
   python -m long_video_eval doctor --config configs/helios_60s90s.local.example.json
   python -m long_video_eval setup  --config configs/helios_60s90s.local.example.json --yes
   python -m long_video_eval run    --config configs/helios_60s90s.local.example.json
   ```

   `run` 默认不会偷偷下载权重或 clone 代码，避免正式评测跑到一半被网络或权限问题卡住。

4. **评测 backend 可替换**

   当前第一版实现了通用 `external_command` backend。也就是说，metric 可以是：

   - 当前 repo 里的已有 wrapper；
   - 官方 metric repo 的 CLI；
   - 你自己写的 Python 脚本；
   - 未来内置的纯 Python metric adapter。

   模块负责统一检查输入、生成 manifest、展开命令、记录配置；具体 metric 的实现由 config 绑定。

## 安装方式

在当前 repo 中可以直接用：

```bash
cd /path/to/helios-team/tools/long_video_eval
python -m long_video_eval --help
```

也可以 editable install：

```bash
cd /path/to/helios-team/tools/long_video_eval
pip install -e .
long-video-eval --help
```

如果要让 `setup` 自动下载 Hugging Face 权重，需要安装可选依赖：

```bash
pip install -e '.[setup]'
```

## 命令说明

### doctor

只检查，不修改环境：

```bash
python -m long_video_eval doctor \
  --config configs/helios_60s90s.local.example.json \
  --models base \
  --expected-videos 100
```

检查内容：

- `SOURCE_ROOT` 是否存在，或 `video_roots` 中声明的视频目录是否存在；
- 每个 model 对应的 `*.mp4` 是否存在；
- prompt 和视频编号是否能对齐；
- Python import 是否可用；
- config 中声明的代码目录和 checkpoint 是否存在；
- checkpoint 如果配置了 `sha256`，会校验 hash。

### setup

根据 config 下载或准备资源：

```bash
python -m long_video_eval setup \
  --config configs/helios_60s90s.local.example.json \
  --yes
```

支持的资源类型：

- `local_path`：只检查路径，不自动下载。
- `git`：如果路径不存在，可以 clone 指定 repo，并 checkout 到指定 revision。
- `hf_file`：用 `huggingface_hub.hf_hub_download` 下载单个文件。
- `hf_snapshot`：用 `huggingface_hub.snapshot_download` 下载整个 HF repo snapshot。

如果需要 HF token，先在 shell 里设置：

```bash
export HF_TOKEN=...
```

### prepare

只生成输入 manifest，不跑 metric：

```bash
python -m long_video_eval prepare \
  --config configs/helios_60s90s.local.example.json \
  --models base \
  --expected-videos 100
```

输出：

```text
OUTPUT_ROOT/_manifests/DATASET_NAME.manifest.json
```

manifest 会记录：

- 输入视频目录；
- prompt 文件；
- model 列表；
- GPU 列表；
- 每个 model 的视频数量；
- prompt 和视频匹配情况；
- 缺失视频预览。

### run

运行 config 中定义的所有 enabled steps：

```bash
python -m long_video_eval run \
  --config configs/helios_60s90s.local.example.json \
  --models base \
  --expected-videos 100
```

dry-run：

```bash
python -m long_video_eval run \
  --config configs/helios_60s90s.local.example.json \
  --models base \
  --dry-run \
  --allow-missing-resources
```

`run` 会先生成：

```text
OUTPUT_ROOT/_manifests/DATASET_NAME.manifest.json
OUTPUT_ROOT/_generated_prompts/DATASET_NAME.jsonl
```

然后按 config 中的 `run.steps` 执行命令。

## Config 结构

示例文件：

```text
configs/helios_60s90s.local.example.json
```

另有一个只用于对齐当前 60s/90s launcher 的核对配置：

```text
configs/helios_60s_long100_original_base.pipeline_parity.json
```

这个 parity config 固定复刻当前正在运行的第一个组合：

```text
duration = 60s
suite    = long100_original
model    = base
```

它的作用不是作为通用模板，而是用于 dry-run 比对：确认独立 suite 传给 HeliosBench 和 `segment_drift_eval` backend 的 `SOURCE_ROOT`、prompt、`OUTPUT_ROOT`、metric 列表、GPU 分片和 drift 参数是否符合预期。

这里的 `segment_drift_eval` 指的是：对完整视频、开头 15% 片段、结尾 15% 片段分别跑 DOVER / PickScore / HPSv3 / VBench custom-video，然后基于开头和结尾的差异分析 temporal drift。底层复用的历史 wrapper 文件名仍然包含 `ratio`，因为切片比例由 `DRIFT_RATIO=0.15` 控制。

VideoAlign 没有放进这个默认 suite。它可以作为短视频 reward 对齐、训练前后 sanity check、或单独的 VideoAlign 专项分析来跑；如果后续确实要用于 60s/90s 视频，需要先明确官方/训练口径下的帧采样策略和显存边界，再作为单独 backend 加入 config。

核对命令：

```bash
cd /path/to/helios-team/tools/long_video_eval
python -m long_video_eval run \
  --config configs/helios_60s_long100_original_base.pipeline_parity.json \
  --dry-run \
  --allow-missing-resources
```

`dry-run` 会打印每个 backend 实际收到的环境变量，但不会启动真实评测。

关键字段：

```json
{
  "defaults": {
    "source_root": "...",
    "video_roots": {
      "optional_model_name": "/path/to/existing/videos"
    },
    "prompt_file": "...",
    "output_root": "...",
    "dataset_name": "helios_60s_long100_original",
    "duration_frames": 1452,
    "models": ["base", "distilled"],
    "gpus": ["0", "1", "2", "3", "4", "5", "6", "7"]
  },
  "python_packages": [
    {"import": "torch", "pip": "torch"}
  ],
  "resources": [
    {"id": "dover_weights", "type": "local_path", "path": ".../DOVER.pth"}
  ],
  "run": {
    "steps": [
      {
        "name": "segment_drift_eval",
        "kind": "external_command",
        "env": {
          "SOURCE_ROOT": "{source_root}",
          "PROMPT_FILE": "{prompt_file}",
          "MODELS": "{models_space}",
          "GPUS": "{gpus_comma}"
        },
        "command": ["bash", "scripts/evaluation/run_helios_long_ratio_eval_sharded_visko3.sh"]
      }
    ]
  }
}
```

支持的占位符包括：

- `{project_root}`
- `{source_root}`
- `{prompt_file}`
- `{prompt_jsonl}`
- `{output_root}`
- `{dataset_name}`
- `{duration_frames}`
- `{models_space}`
- `{gpus_space}`
- `{gpus_comma}`
- `{num_shards}`
- `{cache_dir}`
- `{manifest_json}`

## 和当前 repo 的关系

这个模块本身不直接 import 当前 repo 的 `scripts/evaluation/*`。当前示例 config 会调用随本 PR 一起加入的 backend wrapper，是为了复用已经验证过的评测流程。

如果别人要在自己的 repo 里用，有两种方式：

1. 修改 config，把 `run.steps` 指向他们自己的 metric backend；
2. 保留这个模块，另外安装 DOVER/VBench/HPSv3/HeliosBench 的官方代码，并在 config 中写清楚路径和 checkpoint。

如果别人的视频输出已经在其他目录，例如：

```text
/runs/krea14b/videos/*.mp4
```

不需要改原始目录结构，只需要在 `defaults.video_roots` 中把模型名映射到这些目录。suite 会为 backend 自动创建统一视图：

```text
OUTPUT_ROOT/_inputs/DATASET_NAME/source_root/
  krea14b/videos/*.mp4 -> /runs/krea14b/videos/*.mp4
```

如果要同时比较多个模型，再添加更多映射即可：

```json
{
  "defaults": {
    "models": ["krea14b", "my_method"],
    "video_roots": {
      "krea14b": "/runs/krea14b/videos",
      "my_method": "/runs/my_method/samples"
    }
  }
}
```

## 后续可扩展方向

- 把 DOVER、PickScore、HPSv3 做成内置 Python adapter，减少 external command。
- 增加 YAML config 支持。
- 增加 per-metric result schema 校验。
- 增加 result merge 和 plot 子命令。
- 给每个 metric 的资源加固定 revision 和 sha256，进一步提高可复现性。
