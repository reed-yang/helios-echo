# Session Checkpoint — 2026-07-29(Stage A 全闭环 + 全量 run 在跑)

> 压缩重载/用户回归单一入口,取代 2026-07-27 checkpoint。读序:本文 → `CLAUDE.md` → `logs/progress.md` 末六节 → 各 verdict。

## 一、正在跑的东西(回来第一眼)

| 任务 | 位置 | 状态 |
|---|---|---|
| **全量 Stage A**(168,431 clips × 12k 步) | **sbatch job 5823** @ c-node05+mc-node01(16×H200,accum1 全局 batch 32) | RUNNING ~3000+/12000,7.8-8.4 s/it,ETA 07-29 晚;log `results/stageA_full_run1.log`;每 500 步 ckpt 已验证 |
| 预览站(双页 106 卡) | 登录节点 http-server:8437 + cloudflared 隧道 + 保活(全部脱离会话) | URL 见 `results/p2_site/.current_url`(当前 scope-elegant-aviation-garcia.trycloudflare.com) |
| watcher(session 内,重载后需重布) | 5823 里程碑 br054yf0d;站点可用性 b6sna071j | session 结束即失效 |

## 二、Stage A 已全闭环(2026-07-28)

1. **双轨训练完成**:pilot(2,869 clips)与 slice512(57,741)均 4000/4000,ckpt 链 9×2,不变量全程零破坏(pilot 尾部 SIGABRT = 拆除竞态噪声)。终态 loss `results/stageA_*_loss_final.csv`。
2. **终态验收 PASS**(`logs/research/stage-a-terminal-verdict-2026-07-28.md`):pilot 谱系切换协议饱和斜率单调收敛 −0.30→−0.09→**−0.002**(基线 −0.343)= 零净漂移;slice512@4000 静态平价、切换欠校准(−0.624)。**Stage B 选基推荐 pilot@4000**。
3. **证据链三协议**(全部 rep50 结构化 prompt,种子跨臂配对,基线可复用):
   - r3 静态 66-section/90.75s(`p2-interim-verdict-r3.md`):分布内基线漂移温和;untrained 冻结 3/8;训练后无害化。
   - r4 切换 6×7 硬切换/57.75s(`p2-eventswitch-verdict-r4.md` + cadence 附录):untrained 爆冲 +1.29;pilot 演化收敛;管线原生 `use_interpolate_prompt`/`interpolation_steps=0`。
   - r5 8min 349-section(`p2-longhorizon-verdict-r5-2026-07-28.md`):漂移形态翻转为中后段过饱和;off 2/5 失控→200;记忆臂斜率减半且**全部末段回收**——记忆=阻尼+回复力,"回复力→稳定器"列为 Stage B 核心验收目标。347 次写入压力测试通过。
4. **作废口径**:r1/r2(vs24_long raw prompt,OOD)已 VOID,不上站不引用。
5. 评测工具链例行化:`run_p2_interim_drift_ab.py`(--prompt-set rep50 / --segments / --sections / --memory-partial 载入排除 patch convs)+ `sbatch_p2_r3_infer.sbatch`(可选 sections)/`sbatch_p2_r4_eventswitch.sbatch`;metrics = `tools/long_video_eval/scripts/run_helios_long_timeseries_metric.py`;站点 `build_p2_site.py`(R3/R4 META 控页面)。

## 三、环境事实(新增)

- 集群泄漏 Prohibited 坏卡进 cgroup(c-node08、mc-node02 各一)⇒ 所有 eval sbatch 内置"选最大空余 Default 卡 + 60GB 门槛"重映射;同事进程会运行时膨胀(63→116GB 实测)。
- c-node08 "Kill task failed" 自动 drain 慢性病(3 次)⇒ 核净后 `sudo scontrol update nodename=... state=resume`。
- 叠加实测可行(与 xiangbo 同卡 41.7+50GB);pinghe 不可叠(memory 文件有完整规则)。
- 双节点 accelerate:每节点一 launcher(--machine_rank $SLURM_NODEID),NCCL 走 IB,7.8 s/it(16 卡 accum1)。
- 全量语料 168,431 文件/2.3TiB 已齐(v2 cache 重扫正常,scan 一次 build 多 rank load)。

## 四、等用户决策

1. Stage B 启动时机与设计(基座 pilot@4000;"回复力→稳定器" = U=4 unroll 验收目标;slice512@4000 对照臂可选)。
2. 全量 run 各 cadence ckpt(4000/8000/12000)追踪评测——4000 落地我会按例行跑(r4 切换 + 可选 8min)。
3. CLIP 身份/背景一致性轨迹指标(v5)是否立项;8min 素材上信息量最大。
4. 永久域名:Cloudflare Pages 需 API token(Pages:Edit);当前 trycloudflare 随进程重启换 URL。
5. 旧项:PR#1 ping xiangbo、job_submit.lua、mc-node01 GPU2、slurmdbd。

## 五、Session 元信息

- 本窗口(07-28 全天至 07-29 凌晨)提交序列见 git log;全部已推送 `private`(github.com/reed-yang/helios-echo)。
- 战役统计:r3 重建 24+1 / r4 24 / cadence-2000 16 / 终态 32 / r5 10 = **107 支推理,4 轮连续零失败**;GPU 峰值并发 30 卡。
- 长期 memory:`cluster-gpu-etiquette.md`(节点规则全集)。
