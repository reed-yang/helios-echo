# 为什么旧 VAE latent 不能直接转换成 chunk 对齐的新 latent

问题：`latents_cfr_int_30b_368x640` 已经有 43 万条现成 latent，为什么给 chunk 对齐 caption
配套时必须从原视频重新 encode（`latents_chunkcap_368x640`），而不能从旧 latent"换算"出来？

一句话答案：**旧 latent 覆盖原视频的哪一段帧根本没有被记录下来（起点是随机数），而且即使知道
起点，Wan VAE 的时间压缩也让"平移几帧"在 latent 空间没有对应运算。** 下面分四层解释。

## 1. 旧 latent 的时间起点是随机的、且没落盘（致命，与 VAE 无关）

旧 encode 走 `BucketedFeatureDataset`（`helios/dataset/dataloader_mp4_dist.py:421-425`）：

```python
start_frame = random.randint(cut_start_frame, max_start_frame)   # 每次运行都不同
end_frame   = start_frame + bucket_num_frame                     # bucket ∈ {21,41,61,...,501}
```

`start_frame` 用完即弃，`.pt` 文件名里只有 `cut`（`{clip}_{cs}-{ce}_{nf}_368_640.pt`），
没有真实起点。偏移范围 = `cut_len − bucket_len`：长度桶步长 20，所以 ≤520 帧的 clip 偏移
∈ [0,19]；超过 501 帧的 clip 全部落到 501 桶，偏移可达 `cut_len − 501`（任意大）。

**为什么"不是从第一帧开始"？** 因为离线 encoder（get_short-latents.py）直接复用了*训练*
dataloader 的取帧代码，而训练里随机时间裁剪是有意的数据增广 —— 单 prompt 数据切哪一段都
和 caption 匹配。离线落盘时这个随机性被无声地继承了下来。

**实测证据（2026-07-03，逐 offset 重编码 vs 存盘 latent 的 MSE 谷值）**：

| 旧 latent | cut | bucket | 真实起点 |
|---|---|---|---|
| `00005594…_0-188_181` | 0-188 | 181 | **第 6 帧**（MSE 0.131→0.000→0.019，谷在 6） |
| `0000776d…_0-171_161` | 0-171 | 161 | **第 5 帧** |
| `0001a9bc…_0-167_161` | 0-167 | 161 | **第 3 帧** |

三条 clip 三个不同起点，全都不是 0 —— 随机性坐实。而新 caption 的合同是
caption_k ↔ `[cut_start+33k, cut_start+33(k+1))`；起点不明，caption 和 chunk 的对应关系
就无从谈起。这一条就已经判死刑：信息丢了，任何"转换"都无米下锅。
（理论上可以像上面这样逐 clip 暴力搜 offset 找回起点，但每条要跑 `max_off+1` 次 VAE
encode，比直接重 encode 一次还贵，长 clip 还要搜几百个候选 —— 不划算。）

## 2. 就算知道起点，长度桶也不是 33 的倍数

bucket 长度 21/41/61/…/501 均 ≡ 相对 33 不整除。旧 encode 拿到 141 帧后做
`num_chunks = 141 // 33 = 4`，实际只 encode 了前 132 帧，**尾巴 9 帧被丢弃**。新网格需要的
某些帧（比如 caption 网格里第 4 个 chunk 的 132-165 帧）可能根本没被 encode 进旧 latent —— 
不存在的信息无法恢复。

## 3. 就算起点对齐、长度也够，latent 也不能"平移"

Wan VAE 做 **4× 时间下采样 + 因果 3D 卷积**：33 帧 RGB → 9 帧 latent，规则是
`第 1 帧单独成 latent，之后每 4 帧压成 1 帧 latent`。一个 latent 帧是它对应 RGB 窗口的
**非线性压缩摘要**，不是逐帧采样。假设旧 latent 从第 26 帧开始、新网格从第 0 帧开始，
你需要的是"往左平移 26 帧后的 latent"。但：

- latent 时间轴上 1 步 = RGB 4 帧，26 帧 = 6.5 个 latent 步，连整数平移都不是；
- 即使是整 4 帧的平移，latent[t] 编码的是"分组边界固定"的那 4 帧的联合压缩，平移后分组
  重新划分（原来 {27,28,29,30} 一组，现在 {24,25,26,27} 一组），新分组的联合信息在旧 latent
  里只存在被压缩、混叠过的版本 —— **VAE 编码不是时移等变的（not shift-equivariant）**，
  不存在一个线性/闭式算子把旧 latent 映射到新 latent。

## 4. chunk-wise 独立 encode 让跨界更不可能

Helios 的 latent 是每 33 帧**独立**过一次 VAE（chunk 之间没有信息流动）。新网格的一个 chunk
（如 `[0,33)`）在旧网格（起点 26）下横跨旧 chunk₀ 的尾部和"被丢弃的头 26 帧"，
或横跨两个旧 chunk。每个旧 chunk 的第 1 帧还享受"单帧成 latent"的特殊处理 —— 拼接两个旧
chunk 的 latent 片段不等于新 chunk 的 latent。

## 那 decode 回 RGB 再 re-encode 行不行？

技术上能跑通（`latent → VAE.decode → RGB → 按新网格 VAE.encode`），但没有意义：

1. **起点仍然未知**（第 1 条），decode 出的 132 帧你不知道锚在原视频哪里，caption 对不上；
2. decode/re-encode 引入一次有损往返，画质白白降一档；
3. 成本并不省：decode+encode ≈ 2 次 VAE 前向 > 直接从 mp4 读帧 encode 的 1 次。

**结论：从源 mp4 按 `[cut_start, cut_start+n*33)` 确定性重 encode 是唯一正确做法**，
且很便宜（24,847 条 clip，48×H200 约 1 小时，`encode_chunk_latents.sbatch`）。

## 反过来记住这个教训

任何"latent 复用"的前提是 **latent 的帧覆盖范围被精确记录且和新用途的网格重合**。
以后落盘 latent 时把真实 `start_frame` 写进文件名或 sidecar（新 encoder 已经这样做：
文件名里的帧数就是真实覆盖 `[cut_start, cut_start+n*33)`），旧数据就不会再报废一次。

相关：`docs/CHUNK_ALIGNED_CAPTIONING.md` §4 坑 #11、§7 encode 方式总结。
