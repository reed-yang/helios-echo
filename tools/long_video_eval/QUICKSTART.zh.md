# long_video_eval 快速上手

## 一句话介绍

`long_video_eval` 是一个独立的长视频生成评测 suite。给它一组视频、prompt、模型名字和 config，它会先检查输入和评测资源，再统一调度 HeliosBench、DOVER、PickScore、HPSv3、VBench custom-video 等指标，并支持 GPU 分片和 temporal drift 分析。

## 它解决什么问题

长视频生成评测通常会遇到几个实际问题：

- metric 依赖分散：不同指标需要不同代码、checkpoint、Python 包。
- 输入格式容易混乱：视频编号、prompt 行号、model 名字需要严格对齐。
- 长视频评测需要切片：不仅要看 full video，还要看开头和结尾是否发生质量或语义漂移。
- 多 GPU 跑评测时容易写出一次性脚本，后续复现实验比较困难。

这个 suite 的设计目标是把这些约定放进一个 config 里，使评测流程可检查、可复用、可迁移。

## 输入格式

最简单的输入方式是使用标准目录。单个模型也可以评测：

```text
SOURCE_ROOT/
  my_model/videos/0000.mp4
  my_model/videos/0001.mp4
```

prompt 文件按行和视频编号对齐：

```text
prompts.txt
  line 1 -> 0000.mp4
  line 2 -> 0001.mp4
```

也就是说，`0000.mp4` 使用 prompt 文件第 1 行，`0001.mp4` 使用第 2 行。`my_model` 只是模型名字，可以是 `base`、`distilled`，也可以是 `krea14b`、`wan21`、`my_method` 等。

如果要横向比较多个模型，可以在同一个 `SOURCE_ROOT` 下放多个模型目录：

```text
SOURCE_ROOT/
  my_model/videos/0000.mp4
  my_model/videos/0001.mp4
  another_model/videos/0000.mp4
  another_model/videos/0001.mp4
```

多模型评测时，不同模型下面的视频编号应该和同一个 prompt 文件对齐。

如果已有输出不在这个结构里，不需要手动移动或复制视频。可以在 config 里用 `video_roots` 显式声明每个模型的视频目录：

```json
{
  "defaults": {
    "models": ["krea14b"],
    "video_roots": {
      "krea14b": "/runs/krea14b/videos"
    }
  }
}
```

运行时 suite 会在 `OUTPUT_ROOT/_inputs/DATASET_NAME/source_root` 下创建标准化 symlink input view，供现有 backend 使用；原始输出目录不会被改动。如果要比较多个模型，再在 `models` 和 `video_roots` 里列出多个 key。

## 快速运行

在当前 repo 里可以直接运行：

```bash
cd /path/to/helios-team/tools/long_video_eval
```

第一步，检查输入、Python import、metric 代码路径和 checkpoint：

```bash
python -m long_video_eval doctor \
  --config configs/helios_60s90s.local.example.json \
  --models base,distilled \
  --expected-videos 100
```

第二步，如果缺少资源，根据 config 下载或准备资源：

```bash
python -m long_video_eval setup \
  --config configs/helios_60s90s.local.example.json \
  --yes
```

第三步，正式运行评测：

```bash
python -m long_video_eval run \
  --config configs/helios_60s90s.local.example.json \
  --models base,distilled \
  --expected-videos 100
```

如果只是想确认会执行什么命令，不启动真实评测：

```bash
python -m long_video_eval run \
  --config configs/helios_60s90s.local.example.json \
  --models base,distilled \
  --dry-run \
  --allow-missing-resources
```

## 换成自己的数据

复制一份 config，然后主要改 `defaults`。如果你的输出已经是 `SOURCE_ROOT/<model>/videos/*.mp4`，可以这样写：

```json
{
  "defaults": {
    "source_root": "/path/to/generated/videos",
    "prompt_file": "/path/to/prompts.txt",
    "output_root": "/path/to/eval_outputs",
    "dataset_name": "my_long_video_eval",
    "duration_frames": 1452,
    "models": ["base", "distilled"],
    "gpus": ["0", "1", "2", "3", "4", "5", "6", "7"]
  }
}
```

如果不同模型的视频分散在任意目录，用 `video_roots`：

```json
{
  "defaults": {
    "prompt_file": "/path/to/prompts.txt",
    "output_root": "/path/to/eval_outputs",
    "dataset_name": "my_long_video_eval",
    "duration_frames": 1452,
    "models": ["krea14b", "my_method"],
    "video_roots": {
      "krea14b": "/runs/krea14b/videos",
      "my_method": "/runs/my_method/samples"
    },
    "gpus": ["0", "1", "2", "3", "4", "5", "6", "7"]
  }
}
```

关键字段含义：

- `source_root`：标准输入根目录；不使用 `video_roots` 时，下面应该有 `<model>/videos/*.mp4`。
- `video_roots`：可选。把任意模型名映射到任意已有视频目录；设置后不需要手动整理成标准目录。
- `prompt_file`：逐行 prompt 文件，行号和视频编号对齐。
- `output_root`：评测输出目录。
- `dataset_name`：这轮评测的名字，会写进 manifest 和结果目录。
- `duration_frames`：视频帧数，例如 60 秒 24 fps 的 Helios 设置是 1452 frames。
- `models`：要评测的模型名字。标准目录模式下对应 `SOURCE_ROOT/<model>/videos`；`video_roots` 模式下对应 `video_roots` 的 key。
- `gpus`：用于评测分片的 GPU id。

## 当前包含哪些评测 backend

### heliosbench

`heliosbench` backend 复用 HeliosBench 风格的评测流程，当前用于计算：

- `semantic`：视频和 prompt 的语义匹配。
- `aesthetic`：视频视觉美学质量。
- `motion_amplitude`：视频运动幅度。
- `motion_smoothness`：运动平滑度。
- `drifting_aesthetic`：美学质量从开头到结尾的漂移。
- `drifting_motion_smoothness`：运动平滑度从开头到结尾的漂移。
- `drifting_semantic`：语义匹配从开头到结尾的漂移。

这些指标适合长视频，因为它们不仅看整体表现，也关心视频后段是否退化。

### segment_drift_eval

`segment_drift_eval` 是面向长视频的片段评测 backend。它会对三个时间范围分别评测：

- `full`：完整视频。
- `start15`：开头 15%。
- `end15`：结尾 15%。

然后基于 `start15` 和 `end15` 的差异分析 temporal drift。

当前接入的指标包括：

- `DOVER`：无参考视频质量 / 美学质量评估。
- `PickScore`：图文偏好分数，用于衡量视频采样帧和 prompt 的匹配偏好。
- `HPSv3`：图文偏好 / reward 风格指标。
- `VBench custom-video`：对自定义视频集合跑 VBench 维度，例如 subject consistency、background consistency、aesthetic quality、imaging quality、motion smoothness、dynamic degree 等。

注意：底层复用的历史 wrapper 文件名仍然叫 `run_helios_long_ratio_eval_sharded_visko3.sh`，因为切片比例由 `DRIFT_RATIO=0.15` 控制。但在 suite 语义里，这一步叫 `segment_drift_eval`，更准确。

### 为什么默认不测 VideoAlign

这版 60s/90s 长视频 suite 默认不包含 VideoAlign。原因不是 VideoAlign 不重要，而是它更适合短视频或固定帧数输入的 reward / alignment 评估。对 60 秒以上视频直接使用 VideoAlign，需要先明确几个问题：

- 是输入全量帧，还是采样帧；
- 如果采样，采样多少帧、间隔如何设定；
- 这个采样口径是否和训练时 VideoAlign reward 的输入一致；
- 60s/90s 视频的全量或高密度输入是否会超过显存和模型上下文能力。

为了避免把一个不稳定的采样口径混进长视频主评测，当前 suite 只把 VideoAlign 留作单独专项分析。短视频 sanity check、reward 训练前后对齐、或 5 秒/10 秒视频评估仍然可以单独配置 VideoAlign backend。

## 输出内容

运行时会先写 manifest：

```text
OUTPUT_ROOT/_manifests/DATASET_NAME.manifest.json
```

manifest 会记录：

- source root；
- prompt 文件；
- model 列表；
- GPU 列表；
- 每个 model 的视频数量；
- 缺失视频；
- prompt 和视频编号对齐情况。

各 backend 的结果会写到 config 中定义的输出目录下。当前本地示例会把 HeliosBench 和 segment drift 结果分别写到不同子目录，方便后续分析和画图。

## 和当前 repo 的关系

`long_video_eval` 本身是独立模块，不直接 import 当前 repo 的 `scripts/evaluation/*`。当前示例 config 会调用随本 PR 一起加入的 backend wrapper，是为了复用已经验证过的评测流程。

如果别人要迁移到自己的项目，有两种方式：

1. 保留这个 suite，修改 config 里的 `run.steps`，让它调用自己项目里的 metric 脚本。
2. 把 DOVER、PickScore、HPSv3、VBench、HeliosBench 的代码和权重路径写进 config，继续用统一的 `doctor/setup/run` 流程。

## 推荐给合作者的使用方式

第一次使用时，推荐先只跑：

```bash
python -m long_video_eval doctor --config your_config.json
python -m long_video_eval run --config your_config.json --dry-run --allow-missing-resources
```

确认输入和命令都正确后，再执行：

```bash
python -m long_video_eval setup --config your_config.json --yes
python -m long_video_eval run --config your_config.json
```

这样可以避免正式评测跑到一半才发现 prompt 对不上、checkpoint 缺失、GPU 分片参数不对，或者输出目录写错。

## 本地验证配置

`configs/helios_60s_long100_original_base.pipeline_parity.json` 是一个专门用于验证的 config。它固定复刻当前 60s/90s launcher 的第一个组合：

```text
duration = 60s
suite    = long100_original
model    = base
```

它不是通用模板。给别人使用时，应该从：

```text
configs/helios_60s90s.local.example.json
```

复制并修改。
