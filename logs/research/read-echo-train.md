# Echo-Infinity Training Pipeline: Code-Grounded Report

## Scope and evidence rules

**OBSERVED.** This report deep-reads the requested training entry point, trainer, DMD models, rollout pipelines, loss/data utilities, and released training configs. It also reads the memory encoder and causal transformer sections required to trace the actual memory write and gradient paths. Echo-Infinity exposes only one trainer entry point: `train.py` merges a small default config into the selected stage config, instantiates `ScoreDistillationTrainer` when `trainer == "score_distillation"`, and calls `train()` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/train.py:7-32`).

**Important artifact limitation.** The referenced checkpoints are not present in this checkout, so their tensor contents and provenance cannot be independently inspected. The report distinguishes configuration/README labels from executable-code evidence.

## 1. Training stages

### 1.1 Stage 1: short-video initialization

**OBSERVED.** The release documentation calls the first run “Stage 1 — Init,” launches it with `scripts/train_echo_infinity_init.sh`, and names checkpoint step 400 as the example output reused by Stage 2 (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/README.md:123-141`). The script selects `configs/echo_infinity.yaml` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/scripts/train_echo_infinity_init.sh:26-29`) and invokes distributed `train.py` with it (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/scripts/train_echo_infinity_init.sh:41-49`).

**OBSERVED.** Stage 1 initializes its generator from `checkpoints/chunkwise/causal_forcing.pt` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:1`), which the README labels “Stage-2 (Causal ODE) init from upstream Causal-Forcing” (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/README.md:60-67`). The executable loader merely loads the checkpoint’s `generator` or `model` dictionary with `strict=False`; it does not perform ODE training or teacher conversion itself (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:357-389`). It separately loads a `critic` if one exists (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:392-397`).

**OBSERVED.** This stage generates and supervises 21 **latent** frames, uses three latent frames per causal block, and always samples exactly 21 because both minimum and maximum training lengths equal 21 (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:43-48`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:52-53`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:66-68`). The VAE maps one initial pixel frame plus groups of four later pixel frames to latent time positions (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/vae.py:328-338`), and validation video is written at 16 fps (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:1011-1024`). Thus 21 latent frames correspond to 81 decoded pixel frames, approximately 5.06 seconds at 16 fps.

**OBSERVED.** Stage 1 is not configured for the trainer’s cross-chunk streaming wrapper: `streaming_training` is absent from `echo_infinity.yaml`, while the trainer defaults it to false (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:435-443`). Instead, each DMD sample runs a complete 21-frame self-forcing rollout through `SelfForcingTrainingPipeline` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/base.py:117-126`).

**OBSERVED.** Stage 1 enables the query memory with three query frames, 1,560 tokens per frame, two encoder layers, hidden width 1,536, 12 heads, gate bias 2, and `bptt_clips: 1` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:70-88`). Its local cache is 12 latent frames plus a 3-frame sink (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:54-57`).

### 1.2 Stage 2: long streaming tuning

**OBSERVED.** The release documentation calls the second run “Stage 2 — Long-Video Tuning”; its script selects `configs/echo_infinity-long.yaml` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/README.md:137-143`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/scripts/train_echo_infinity_long.sh:26-29`).

**OBSERVED.** Stage 2 initializes the base generator and critic from `checkpoints/echo_infinity.pt`, the Stage-1 checkpoint path (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:2`). Because an adapter block is present, the trainer loads this checkpoint before wrapping, extracts embedded query-memory weights, loads the base generator and critic, and then applies LoRA to the generator and fake-score networks (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:83-137`). With no explicit `init_from_ema` in the long config, it uses the ordinary `generator` key rather than `generator_ema` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:95-107`).

**OBSERVED.** Stage 2 sets a sequence length and streaming maximum of 240 latent frames, a chunk size of 21 latent frames, and a minimum of 18 newly generated latent frames per chunk (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:45-50`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:88-93`). Through the VAE’s 4× temporal expansion and 16-fps output, 240 latent frames decode to 957 pixel frames, approximately 59.8 seconds (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/vae.py:328-338`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:1011-1024`).

**OBSERVED.** The actual chunks are not always non-overlapping 21-frame clips. After the first chunk, the code samples the new-frame count from `range(18, max_new_frames, 3)`, retains overlap from the previous stored chunk, and builds a full 21-frame training clip when possible (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/streaming_training.py:153-189`). Only the newly generated region is marked for DMD/critic supervision (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/streaming_training.py:190-218`).

**OBSERVED.** Stage 2 additionally changes the distribution class from `dmd` to `dmd_switch`, supplies paired first- and second-prompt files, randomly chooses a switch point from latent-frame indices 21 through 201, and enables relative RoPE plus prompt-switch recaching (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:32-33`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:51-62`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:72-88`). `TwoTextDataset` enforces line-by-line prompt pairing (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/utils/dataset.py:32-45`), and the trainer selects a valid configured switch point and broadcasts it across ranks (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:548-566`).

### 1.3 What is and is not distilled

**OBSERVED.** The real score network is the noncausal Wan2.1-T2V-14B, the fake score network is the noncausal Wan2.1-T2V-1.3B, and the generator is the causal Wan wrapper (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:2-7`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/base.py:24-35`). The wrapper constructs noncausal models with `WanModel` and the generator with `CausalWanModel` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/utils/wan_wrapper.py:106-124`).

**INFERRED.** Therefore DMD trains a causal student/generator using score differences involving a frozen bidirectional 14B model, but this is **distribution matching**, not a direct teacher-output reconstruction loss.

**ABSENT.** No requested file implements an explicit preliminary causal-distillation stage from the bidirectional teacher. No teacher-forced teacher/student loss is called by the released training path. A teacher-forcing mask helper exists in the causal model (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:603-632`), but the model’s `_forward_train` immediately raises `NotImplementedError`, making that path unreachable here (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:842-845`). The executable Echo stages use cached autoregressive inference forwards instead.

## 2. Causal-forcing DMD mechanics

### 2.1 Generator, critic, and real-score roles

**OBSERVED.** `BaseModel` marks the causal generator and fake score trainable, freezes the real score, and also freezes the text encoder and VAE (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/base.py:24-39`). The models are set to evaluation mode during each update, but evaluation mode does not override `requires_grad` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:613-640`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:719-765`).

**OBSERVED.** For the generator DMD loss, the fake score predicts conditional clean latents; with the released configs its guidance scale is zero, so no fake unconditional pass is used. The real score always performs conditional and negative-prompt passes and combines them with guidance scale 3 (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/dmd.py:31-36`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/dmd.py:46-56`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:17`). The DMD gradient is `pred_fake_x0 - pred_real_x0`, normalized by mean absolute `generated_x0 - pred_real_x0`, with NaNs converted to finite values (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/dmd.py:56-62`).

**OBSERVED.** The score timestep is sampled uniformly for the whole video/clip, shifted by the Wan schedule, and clamped to 2–98% of the 1,000-step range (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/dmd.py:28-44`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/dmd.py:67-76`). The generator receives the surrogate target `(generated - grad).detach()` through a half-MSE, so backpropagation gives the prescribed DMD gradient without differentiating through either score model (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/dmd.py:77-82`).

**OBSERVED.** The fake score is trained as a denoiser on separately generated samples. Generation is inside `torch.no_grad()`, then the generated latent is noised, the fake score predicts clean latent/flow, and the configured flow-prediction MSE is optimized (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/dmd.py:99-130`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/utils/loss.py:35-45`). The streaming variant explicitly detaches the generated chunk and persistent cache tensors before critic computation (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/streaming_training.py:234-252`).

**OBSERVED.** This “critic” is therefore a learned fake-distribution score/denoiser, not a binary real/fake discriminator. The training loop updates it every iteration, while the generator and memory encoder update when `step % 5 == 0` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:42`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:769-839`).

### 2.2 Rollout inside training

**OBSERVED.** The causal generator is sampled with four denoising timestep values derived from `[1000, 750, 500, 250]`; `warp_denoising_step` remaps these through the scheduler (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:8-15`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/base.py:18-22`). For every three-frame causal block, one exit step is randomly selected and synchronized; `same_step_across_blocks: true` makes all blocks exit at the same selected denoising step (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/self_forcing_training.py:45-55`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/self_forcing_training.py:146-170`).

**OBSERVED.** Earlier denoising steps run under `no_grad`; only the selected exit forward for blocks in the supervised suffix runs with gradients (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/self_forcing_training.py:157-170`). After producing a block, the pipeline adds configured context noise—zero in the defaults—and performs another `no_grad` generator forward to write the produced context into persistent KV and memory state (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/self_forcing_training.py:171-176`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/default_config.yaml:9-10`).

**OBSERVED.** Stage 1 requests gradients only over the last 21 latent frames of a 21-frame rollout (`slice_last_frames: 21` and fixed 21-frame length), so all selected exit forwards are gradient-enabled (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:66-68`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/self_forcing_training.py:124-141`). The generated blocks still become their own subsequent context through the explicit recache forward, so training follows the inference-time autoregressive state evolution rather than teacher-forcing ground-truth video.

**OBSERVED.** Stage 2 preserves one streaming state across optimizer iterations. `start_new_sequence()` samples prompts, embeds them, creates the streaming state, and only restarts after the configured maximum length is reached (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:676-717`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:719-728`). `train_first_chunk: true` means the initial chunk is itself used for generator training rather than consumed only as a gradient-free primer (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:92-95`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:735-744`).

### 2.3 How self-generated history reaches memory writes

**OBSERVED.** Every chunk/block’s selected clean prediction is re-noised at context timestep zero and fed back through the generator under `no_grad` with the same persistent KV cache (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/streaming_training.py:72-77`). Inside causal self-attention, new K/V are inserted into a sink-preserving rolling local cache; when the local budget overflows, old non-sink tokens are shifted out (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:123-168`).

**OBSERVED.** After all transformer blocks finish, the causal model computes how far the oldest recent-frame boundary moved. If frames exited the local window, it clones the **last transformer block’s** evicted K/V, clones sink K/V, and calls `query_memory_encoder.update(...)` before committing cache updates (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:785-828`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:837-840`). Thus memory is written from evicted KV produced by self-generated latent context, not from ground-truth videos.

**OBSERVED.** Query-memory update cross-attends the current query state to those evicted K/V, projects the result, and applies the elementwise retention gate
`new_state = gate * old_state + (1 - gate) * projected` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:124-158`). Once history exists, `get_kv()` projects the evolving state to memory K/V (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:190-212`); each causal self-attention layer concatenates `[sink, memory, local]` when memory is active (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:283-310`).

**OBSERVED.** `use_evicted_kv: false` appears in both configs (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:81`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:117`) but the current `QueryMemoryEncoder` does not read that field. Its update always consumes the evicted K/V supplied by the causal model (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:38-68`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:124-142`). The flag is therefore inert in this implementation.

## 3. Gradient flow into memory modules

### 3.1 Parameters and learning rates

**OBSERVED.** In Stage 1, all causal generator parameters and all fake-score parameters are trainable; real score, text encoder, and VAE are frozen (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/base.py:29-38`). The query-memory encoder is created outside FSDP, attached by `object.__setattr__`, synchronized manually across ranks, and given all-reduce hooks (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:209-250`). It receives a distinct AdamW optimizer at `generator_lr × encoder_lr_multiplier` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:290-299`).

**OBSERVED.** Stage-1 rates are generator `2e-6`, critic `4e-7`, and memory encoder `1e-5`; all use AdamW with weight decay 0.01, β1=0, β2=0.999 (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:25-30`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:87`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/default_config.yaml:3`).

**OBSERVED.** In Stage 2, rank-256 LoRA with α=256 and no dropout is applied to every linear submodule inside generator `CausalWanAttentionBlock`s and critic `WanAttentionBlock`s (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:97-104`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:941-968`). The memory encoder is not placed inside PEFT and remains separately fully trainable (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:209-250`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:290-299`).

**OBSERVED.** Stage-2 optimizer rates are generator LoRA `1e-5`, critic LoRA `2e-6`, and memory encoder `5e-5`, again with AdamW, weight decay 0.01, and β=(0, 0.999) (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:26-31`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:123`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/default_config.yaml:3`). The generator and memory optimizers step only once per five trainer iterations, while the critic steps every iteration (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:774-839`).

### 3.2 Exact detach boundaries and BPTT span

**OBSERVED.** The memory state itself is initialized as a clone of the learned query parameter without detaching, so gradients from a sequence can reach `query_init` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:102-118`). `update()` also does not internally detach the old or new state (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:124-192`).

**OBSERVED.** The explicit truncation boundary is in `StreamingTrainingModel.generate_next_chunk`: after every generated chunk, it increments a chunk counter and calls `encoder.detach_state()` when the count is divisible by `encoder.bptt_clips` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/streaming_training.py:200-217`). `detach_state()` detaches the evolving query tensor(s), but it does not reset them (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:214-220`). Both released stages configure `bptt_clips: 1` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:85`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:121`).

**OBSERVED.** This configured detach is executed only by the Stage-2 streaming wrapper. Stage 1 does not instantiate `StreamingTrainingModel`, so no `detach_state()` call occurs inside its fixed 21-frame rollout (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:435-443`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/base.py:117-126`). Consequently, Stage 1 permits gradient penetration through memory-state evolution throughout the entire 21-latent-frame/5-second sample, subject to the additional `no_grad` boundaries below.

**OBSERVED.** Stage 2 detaches the query state **after every generated streaming chunk**, i.e. approximately each 21-latent-frame/5-second subclip, exactly matching the claimed cross-subclip truncation (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/streaming_training.py:153-218`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:89-93`). Since detachment happens before `compute_generator_loss()` is called, the current chunk’s DMD loss cannot backpropagate through the carried query state into memory updates made by prior chunks (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:735-747`).

**OBSERVED.** The write-side recache forward is under `torch.no_grad()` in both rollout pipelines (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/self_forcing_training.py:171-176`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/streaming_training.py:72-77`). This prevents gradients from flowing backward through the generated context tensor or through generator K/V-producing operations used for the write. However, `torch.no_grad()` does not detach the already-existing query state; the encoder’s arithmetic can produce an updated state connected to prior differentiable query history when invoked under grad-enabled forwards.

**OBSERVED.** Cache writes are applied through tensor assignment into persistent cache tensors (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:652-684`), and the streaming critic path explicitly detaches any cache K/V that still require gradients (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/streaming_training.py:234-246`). Hence persistent raw K/V are not an intended cross-iteration BPTT carrier; the explicit recurrent carrier is the query-memory state.

**INFERRED.** “Stage-1 full penetration” should be interpreted narrowly as no explicit query-state truncation within the 5-second sample—not fully differentiable autoregressive sampling through every denoising and cache-write operation. Most denoising steps and every context-recache write are deliberately under `no_grad`.

### 3.3 A code-path caveat in Stage 2

**OBSERVED.** A chunk’s `frames_to_save` is detached before becoming `previous_frames`, so overlap/history frames cannot carry image-level gradients into later chunks (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/streaming_training.py:186-200`). The full chunk’s gradient mask then supervises only new frames (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/streaming_training.py:193-218`).

**OBSERVED.** The memory detach occurs after generation but before loss computation (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/streaming_training.py:212-218`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/streaming_training.py:223-232`). This is a strict per-chunk truncated-BPTT boundary, not merely a boundary between successive optimizer iterations.

## 4. Gate and encoder initialization

**OBSERVED.** The learned query initialization is sampled from `Normal(0, initializer_range)` with `initializer_range=0.014`; three query frames × 1,560 tokens produce 4,680 memory query tokens (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:40-55`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:67-83`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:72-78`).

**OBSERVED.** Every gate linear bias is overwritten with +2 (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:71-87`). Because the update retains old state with `sigmoid(gate)` and writes projected content with `1-sigmoid(gate)`, zero-valued gate input begins near 0.881 old-state retention and 0.119 new-state write (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:144-155`). This favors conservative early writes but does not make the update identity.

**OBSERVED.** The gate weights, connector MLP, cross-attention projections, output K/V projections, and encoder FFNs have no memory-specific zero initialization in `QueryMemoryEncoder`; only gate biases receive a special initializer (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:68-94`). They therefore use PyTorch module defaults.

**ABSENT.** There is no zero-initialized memory output projection or zero-initialized connector that initially disables memory. The causal Wan backbone’s generic `init_weights()` zeros all linear biases and the final diffusion head weight (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:921-935`), but pretrained backbone loading replaces those initial backbone values, and this routine does not initialize the separately created query-memory encoder (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:204-216`).

**OBSERVED.** Stage 1 enables RMS normalization of projected memory keys; Stage 2 disables it (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:88`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:124`). Stage 2 enables `memory_recache`, used to reset and reconstruct query memory after a prompt switch (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:125`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/streaming_switch_training.py:178-192`).

## 5. Losses and regularizers beyond DMD

**OBSERVED.** The released generator objective is only the DMD surrogate half-MSE (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/dmd.py:64-82`). The fake-score objective is the selected denoising loss; both released configs choose flow prediction (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:18`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:19`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/utils/loss.py:35-45`).

**ABSENT.** There is no GAN adversarial loss in the released DMD/DMDSwitch training path. Although `wan_wrapper.py` contains unused classification/GAN branch construction helpers (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/utils/wan_wrapper.py:128-142`), neither `DMD` nor the trainer calls them; the fake-score “critic” is trained only by denoising regression (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/dmd.py:99-130`).

**ABSENT.** There is no direct real-video reconstruction loss, perceptual loss, feature loss, temporal consistency penalty, gate entropy/sparsity penalty, query-state norm penalty, memory-attention regularizer, or explicit memory reconstruction target in the requested training path. The dataset itself supplies text prompts rather than videos in both released stages (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:305-315`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/utils/dataset.py:11-45`).

**OBSERVED.** The memory encoder contains an optional variational-information-bottleneck branch and computes a KL value (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:92-94`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:159-192`). However, both released configs set `use_vib: false` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:84`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:120`).

**ABSENT.** Even if VIB were enabled, the returned KL scalar is discarded at every normal memory update call (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:822-828`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/streaming_switch_training.py:178-192`). There is no coefficient or addition of this KL to the training objective.

**OBSERVED.** The only generic optimizer regularization is AdamW weight decay 0.01 (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/default_config.yaml:3`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:290-299`). Generator and critic gradient norms are clipped at default 10, while the separately optimized memory encoder has no explicit gradient clipping (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:431-433`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:793-807`).

## 6. Curriculum ordering, warmup, and stabilization

### 6.1 Actual curriculum

**OBSERVED.** The release’s complete training curriculum is:

1. Start from the external Causal-Forcing “Causal ODE” generator checkpoint (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:1`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/README.md:60-67`).
2. Run short, fixed 21-latent-frame DMD with full generator, fake-score, and newly initialized memory encoder trainable (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:43-53`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:290-299`).
3. Initialize long streaming tuning from the Stage-1 Echo checkpoint (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:2`).
4. Train 240-latent-frame streaming sequences with LoRA generator/critic updates, a fully trainable memory encoder, paired prompt switches, and per-chunk query-state detachment (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:45-62`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:74-125`).

### 6.2 Denoising-step and update curricula

**OBSERVED.** There is no temporal progression over denoising steps: `last_step_only` is false, and every causal block randomly selects among all four available exit steps from the start (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/default_config.yaml:6`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/self_forcing_training.py:45-55`).

**OBSERVED.** Generator/memory updates are intentionally five times less frequent than critic updates, because `dfake_gen_update_ratio` is 5 and the fake score steps every trainer iteration (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:42`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:769-839`). Each step accumulates two microbatches; with 32 workers/GPUs and batch size one, the documented effective batch is 64 (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:32-36`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/README.md:123-127`).

### 6.3 Learning-rate and EMA warmup

**ABSENT.** There is no LR scheduler, linear warmup, cosine decay, stage-local ramp, gate-bias schedule, or memory-loss coefficient schedule. The trainer constructs fixed-rate AdamW optimizers and calls `.step()` directly (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:290-300`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:769-839`).

**OBSERVED.** Non-LoRA Stage 1 uses generator EMA decay 0.99, created at step 200; EMA is disabled entirely in LoRA mode, so Stage 2 has no EMA despite retaining the shared EMA config values (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:34-35`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:275-284`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:842-848`).

### 6.4 What stabilizes early memory training

**OBSERVED.** The concrete stabilizers are: initialization from an already causal generator rather than from bidirectional Wan weights directly (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:1`); conservative +2 retention-gate bias (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:47-87`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:152-155`); a short 5-second first stage before 60-second streaming (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:66-68`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:88-93`); a frozen 14B real score and pretrained fake score/backbones (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/base.py:24-38`); five critic updates per generator/memory update (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:774-839`); and truncated cross-chunk memory BPTT in Stage 2 (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/streaming_training.py:212-217`).

**ABSENT.** There is no “static queries first” phase: the query-memory update path is enabled from the start of Stage 1, and no config freezes its encoder or delays writes (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:70-88`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:209-250`). There is also no teacher-forcing pre-phase inside Echo training, no gradual unroll-length schedule, and no gradual transition from ground-truth to generated history.

## 7. What the “switch” variant adds

**OBSERVED.** `DMDSwitch` does not alter the DMD equation. Its only model-level override substitutes `StreamingSwitchTrainingPipeline` and forwards global-sink, APR, recache, and attention-window options (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/dmd_switch.py:11-14`).

**OBSERVED.** During a switch chunk, the switch pipeline delays gradient-enabled exit forwards until the switch frame, changes the text conditioning, and recaches recent history under the new condition (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/streaming_switch_training.py:23-68`). It can preserve global sink K/V, clears cross-attention cache, recaches up to 21 recent frames, and resets/rebuilds query memory when `memory_recache` is enabled (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/streaming_switch_training.py:81-118`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/streaming_switch_training.py:131-192`).

**OBSERVED.** The released long config enables global sink and memory recache but does not enable APR; APR defaults false in `DMDSwitch` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:52`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:125`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/dmd_switch.py:13-14`). Therefore APR blending code is present but inactive in the released Stage-2 recipe.

## 8. Porting-relevant conclusions for Helios

**INFERRED.** Echo’s central train-as-inference property is not simply “DMD loss on a causal model.” It depends on the recurrent state being updated from K/V evicted by self-generated context forwards, while subsequent generated blocks/chunks read that state. A Helios port without persistent KV must preserve the analogous causal dependency through its re-patchified latent-history path.

**INFERRED.** Echo’s Stage-1 gradient claim is compatible with a Helios within-section recurrent memory update, but Echo itself does not differentiate through all sampling/write operations. The portable invariant is “do not explicitly detach the evolving memory state inside the short training sample,” not “retain a full denoising graph.”

**INFERRED.** Echo Stage 2 gives a precise truncation analogue for Helios: retain memory values across roughly 5-second units but detach the recurrent memory state at each unit boundary. Raw video/latent overlap is detached separately and should not be mistaken for memory-state BPTT.

**INFERRED.** No auxiliary memory target can be copied from Echo because none exists in the release. Memory learns only through downstream generator DMD gradients, with conservative gate initialization and a higher memory LR.

## TL;DR

- Stage 1 runs fixed 21-latent-frame (~5 s) DMD from an external Causal-Forcing “Causal ODE” checkpoint.
- Stage 2 initializes from Stage 1 and runs 240 latent frames (~60 s) as stateful 21-frame streaming chunks.
- The frozen real score is bidirectional Wan2.1-14B; the trainable fake score is a 1.3B denoiser; the generator is causal.
- The generator DMD gradient is normalized `fake_x0 - real_x0`; the fake score learns flow denoising on generated samples.
- Self-generated outputs are re-fed at context timestep zero; evicted last-layer K/V update the evolving query memory.
- Stage 1 has no explicit query-state detach within its 5-second sample.
- Stage 2 calls `detach_state()` after every chunk because `bptt_clips=1`, truncating memory BPTT across ~5-second subclips.
- Stage 1 trains the full generator, full fake score, and full memory encoder; Stage 2 trains generator/critic LoRA plus the full memory encoder.
- Memory LRs are `1e-5` in Stage 1 and `5e-5` in Stage 2; generator LRs are `2e-6` and `1e-5`.
- Gate bias is +2, initially favoring ~88% old-state retention; no memory projection is zero-initialized.
- There is no GAN loss, consistency loss, direct memory loss, active VIB KL, LR warmup, static-query phase, or teacher-forcing curriculum.
- The switch variant adds paired prompts, switch-time recaching, global-sink preservation, relative RoPE, and memory reconstruction—not a new DMD objective.