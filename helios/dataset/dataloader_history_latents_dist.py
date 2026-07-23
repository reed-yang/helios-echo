import multiprocessing
import os
import pickle
import random
from collections import defaultdict

import torch
from einops import rearrange
from torch.utils.data import Dataset, Sampler


class BucketedFeatureDataset(Dataset):
    def __init__(
        self,
        feature_folders,
        history_sizes=[16, 2, 1],
        is_keep_x0=True,
        force_rebuild=False,
        return_all_vae_latent=False,
        return_prompt_raw=False,
        return_evicted_latent=False,
        return_rollout_metadata=False,
        num_rollout_sections=3,
        single_res=False,
        single_height=384,
        single_width=640,
        seed=42,
    ):
        self.history_sizes = history_sizes
        self.is_keep_x0 = is_keep_x0
        self.force_rebuild = force_rebuild
        self.return_all_vae_latent = return_all_vae_latent
        self.return_prompt_raw = return_prompt_raw
        self.return_evicted_latent = return_evicted_latent
        self.return_rollout_metadata = return_rollout_metadata
        self.num_rollout_sections = num_rollout_sections
        self.single_res = single_res
        self.single_height = single_height
        self.single_width = single_width
        # Rewritten caption versions that may be stored per .pt as prompt_embed_<v>.
        # If present, __getitem__ picks one uniformly at random per sample.
        self.caption_versions = ["ultra_short", "short", "medium", "long"]
        assert self.is_keep_x0, "is_keep_x0 need to be True now!"

        self.base_seed = seed
        # Shared-memory epoch: persistent DataLoader workers hold forked dataset
        # copies, so a plain-int epoch set by the main process after worker
        # creation never reaches them (verified against StatefulDataLoader) and
        # every per-(seed, epoch, idx) draw silently collapses to the creation-
        # time epoch. A multiprocessing.Value is inherited by fork-started
        # workers, so set_epoch stays live. Under spawn-pickling __getstate__
        # degrades it to a frozen int (status quo ante, no worse).
        self._epoch_shared = multiprocessing.Value("i", 0)

        if isinstance(feature_folders, str):
            self.feature_folders = [feature_folders]
        else:
            self.feature_folders = feature_folders

        self.samples = []
        self.buckets = defaultdict(list)

        for folder in self.feature_folders:
            cache_file = os.path.join(folder, "dataset_cache_v2.pkl")
            self._process_folder(folder, cache_file)

        # Resolution filtering is configuration-dependent, so apply it only to the
        # in-memory view of the configuration-independent v2 cache superset.
        if self.single_res:
            allowed_resolutions = {
                (self.single_height, self.single_width),
                (self.single_height // 2, self.single_width // 2),
                (self.single_height // 4, self.single_width // 4),
            }
            kept = [
                sample
                for sample in self.samples
                if (sample["height"], sample["width"]) in allowed_resolutions
            ]
            if len(kept) != len(self.samples):
                print(
                    f"Resolution filter: dropped {len(self.samples) - len(kept)} samples "
                    f"({len(kept)} remain)"
                )
                self.samples = kept
                self.buckets = defaultdict(list)
                for sample_idx, sample_info in enumerate(self.samples):
                    self.buckets[sample_info["bucket_key"]].append(sample_idx)

        # Rollout-length filter (design ch.2 D6): clean_all needs num_rollout_sections
        # consecutive sections; the offline encoder cuts consecutive 33-RGB-frame chunks
        # (get_short-latents.py: frame_window_size = (9-1)*4+1), so sections ==
        # num_frame // 33. Filtered IN MEMORY after cache load — the on-disk
        # dataset_cache_v2.pkl keeps the unfiltered superset so shared data folders
        # stay valid for readers with other U.
        # No-op for the current guarantee (num_frame >= 121 -> 3 sections >= default U=3).
        if self.return_all_vae_latent:
            frame_window_size = 33
            required = self.num_rollout_sections
            kept = [s for s in self.samples if s["num_frame"] // frame_window_size >= required]
            if len(kept) != len(self.samples):
                print(
                    f"Rollout filter: dropped {len(self.samples) - len(kept)} samples with "
                    f"fewer than {required} sections ({len(kept)} remain)"
                )
                self.samples = kept
                self.buckets = defaultdict(list)
                for sample_idx, sample_info in enumerate(self.samples):
                    self.buckets[sample_info["bucket_key"]].append(sample_idx)

    def _process_folder(self, folder, cache_file):
        cached_data = None
        if not self.force_rebuild and os.path.exists(cache_file):
            print(f"Loading cached metadata from: {folder}")
            try:
                with open(cache_file, "rb") as f:
                    candidate = pickle.load(f)
            except (OSError, pickle.UnpicklingError, EOFError):
                candidate = None
            if isinstance(candidate, dict) and candidate.get("schema") == 2:
                cached_data = candidate

        if cached_data is None:
            print(f"Building metadata cache for folder: {folder}")
            folder_samples, folder_buckets = self._build_folder_metadata(folder)
            cached_data = {"schema": 2, "samples": folder_samples, "buckets": folder_buckets}
            if not self.force_rebuild:
                # First-build coordination across ranks is intentionally deferred to
                # a later trainer-side rank-0/barrier improvement.
                print(f"Saving metadata cache for folder: {folder}")
                with open(cache_file, "wb") as f:
                    pickle.dump(cached_data, f)
            print(f"Cached {len(folder_samples)} samples from {folder}\n")
        else:
            folder_samples = cached_data["samples"]
            folder_buckets = cached_data["buckets"]
            print(f"Loaded {len(folder_samples)} samples from cache: {folder}\n")

        sample_idx_offset = len(self.samples)
        self.samples.extend(folder_samples)

        for bucket_key, indices in folder_buckets.items():
            adjusted_indices = [idx + sample_idx_offset for idx in indices]
            self.buckets[bucket_key].extend(adjusted_indices)

    def _build_folder_metadata(self, folder):
        feature_files = [f for f in os.listdir(folder) if f.endswith(".pt")]
        samples = []
        buckets = defaultdict(list)
        sample_idx = 0

        print(f"Processing {len(feature_files)} files in {folder}...")

        for i, feature_file in enumerate(feature_files):
            if i % 10000 == 0:
                print(f"  Processed {i}/{len(feature_files)} files")

            feature_path = os.path.join(folder, feature_file)

            # Parse filename
            parts = feature_file.split("_")
            uttid = "_".join(parts[:-3])
            num_frame = int(parts[-3])
            height = int(parts[-2])
            width = int(parts[-1].replace(".pt", ""))

            # keep length >= 121
            if num_frame < 121:
                continue

            bucket_key = (num_frame, height, width)

            sample_info = {
                "uttid": uttid,
                "dataset_name": folder.rstrip("/"),
                "file_path": feature_path,
                "bucket_key": bucket_key,
                "num_frame": num_frame,
                "height": height,
                "width": width,
            }

            samples.append(sample_info)
            buckets[bucket_key].append(sample_idx)
            sample_idx += 1

        return samples, buckets

    @property
    def _epoch(self):
        shared = getattr(self, "_epoch_shared", 0)
        return shared.value if hasattr(shared, "value") else int(shared)

    @_epoch.setter
    def _epoch(self, epoch):
        shared = getattr(self, "_epoch_shared", None)
        if hasattr(shared, "value"):
            shared.value = int(epoch)
        else:
            self._epoch_shared = multiprocessing.Value("i", int(epoch))

    def __getstate__(self):
        # multiprocessing.Value cannot be pickled (spawn-started workers);
        # degrade to a frozen int — identical to the old plain-int behavior.
        state = self.__dict__.copy()
        shared = state.get("_epoch_shared")
        if hasattr(shared, "value"):
            state["_epoch_shared"] = shared.value
        return state

    def set_epoch(self, epoch):
        self._epoch = epoch

    @staticmethod
    def _compute_eviction(
        evicted_timeline, history_timeline, choice_idx, latent_window_size, history_window_size
    ):
        """Eviction slices for the memory write path (design ch.2 D4/D5).

        Advancing the target section from k-1 to k rolls the oldest
        latent_window_size frames out of section (k-1)'s history window:
        timeline[:, W*(k-1) : W*k) with W=latent_window_size. Because both
        timelines are [history_window_size zeros] + real frames, the number
        of REAL frames among the evicted ones is
        min(W, max(0, W*k - history_window_size)) — the single source of
        truth for whether/how much to write (never hardcode k boundaries).

        Two timelines because the two slices play different roles in the D5
        write forward: the evicted frames are its X_Noisy and must be at the
        SAMPLE'S OWN bucket resolution (evicted_timeline = continue_vae_latent),
        while the preceding history conditions the forward exactly like the
        normal training history path, which uses the full-res source timeline
        (history_timeline = continue_source_latent). For full-res samples the
        two timelines are the same tensor.

        Returns (evicted_latent [C,W,h,w], evicted_history [C,Hw,H,W'],
        evicted_valid_frames int). evicted_history is the Hw-frame slice
        preceding the evicted block, left-padded with zeros on underflow.
        choice_idx == 0 has no predecessor section: all-zero tensors, 0 valid.
        """
        ev_channels, _, ev_height, ev_width = evicted_timeline.shape
        hist_channels, _, hist_height, hist_width = history_timeline.shape
        evicted_start = (choice_idx - 1) * latent_window_size
        evicted_valid_frames = int(
            min(latent_window_size, max(0, choice_idx * latent_window_size - history_window_size))
        )

        if choice_idx <= 0:
            evicted_latent = evicted_timeline.new_zeros(
                ev_channels, latent_window_size, ev_height, ev_width
            )
            evicted_history = history_timeline.new_zeros(
                hist_channels, history_window_size, hist_height, hist_width
            )
            return evicted_latent, evicted_history, evicted_valid_frames

        evicted_latent = evicted_timeline[
            :, evicted_start : evicted_start + latent_window_size
        ].clone()

        hist_start = evicted_start - history_window_size
        pad = max(0, -hist_start)
        real_history = history_timeline[:, max(0, hist_start) : evicted_start]
        if pad > 0:
            zero_pad = history_timeline.new_zeros(hist_channels, pad, hist_height, hist_width)
            evicted_history = torch.cat([zero_pad, real_history], dim=1)
        else:
            evicted_history = real_history.clone()

        return evicted_latent, evicted_history, evicted_valid_frames

    def prepare_stage1_latent(self, vae_latent, idx, base_vae_latent=None):
        source_latent = base_vae_latent if base_vae_latent is not None else vae_latent

        x0_latent = None
        if self.is_keep_x0:
            x0_latent = source_latent[0, :, :1, :, :].clone()
        total_sections = source_latent.shape[0]
        latent_window_size = source_latent.shape[2]
        history_window_size = sum(self.history_sizes)
        section_size = history_window_size + latent_window_size

        temp_source_latent = rearrange(source_latent, "b c t h w -> c (b t) h w")
        zero_padding_source = torch.zeros(
            temp_source_latent.shape[0],
            history_window_size,
            temp_source_latent.shape[2],
            temp_source_latent.shape[3],
            device=temp_source_latent.device,
            dtype=temp_source_latent.dtype,
        )
        continue_source_latent = torch.cat([zero_padding_source, temp_source_latent], dim=1)

        temp_vae_latent = rearrange(vae_latent, "b c t h w -> c (b t) h w")
        zero_padding_vae = torch.zeros(
            temp_vae_latent.shape[0],
            history_window_size,
            temp_vae_latent.shape[2],
            temp_vae_latent.shape[3],
            device=temp_vae_latent.device,
            dtype=temp_vae_latent.dtype,
        )
        continue_vae_latent = torch.cat([zero_padding_vae, temp_vae_latent], dim=1)

        # One seeded generator per (seed, epoch, idx): the first draw reproduces the
        # previous per-sample choice_idx exactly; subsequent draws share the same
        # deterministic stream (fixes divergence F3 where start_section_idx used the
        # process-global RNG, decoupled from the seeded per-sample discipline).
        sample_seed = self.base_seed + self._epoch * 1000000 + idx
        sample_generator = torch.Generator().manual_seed(sample_seed)
        choice_idx = torch.randint(0, total_sections, (1,), generator=sample_generator).item()
        if choice_idx == 0 and x0_latent is not None:
            x0_latent = torch.zeros_like(x0_latent)

        clean_all_vae_latent = None
        start_section_idx = None
        if self.return_all_vae_latent:
            max_start_idx = total_sections - self.num_rollout_sections
            if max_start_idx < 0:
                raise ValueError(
                    f"Not enough sections: total_sections={total_sections}, num_rollout_sections={self.num_rollout_sections}"
                )
            start_section_idx = torch.randint(
                0, max_start_idx + 1, (1,), generator=sample_generator
            ).item()
            start_indice = start_section_idx * latent_window_size
            end_indice = start_indice + history_window_size + self.num_rollout_sections * latent_window_size
            clean_all_vae_latent = continue_source_latent[:, start_indice:end_indice, :, :]

        start_indice = choice_idx * latent_window_size
        end_indice = start_indice + section_size

        history_latent = continue_source_latent[:, start_indice : start_indice + history_window_size, :, :]
        target_latent = continue_vae_latent[:, start_indice + history_window_size : end_indice, :, :]

        eviction = None
        if self.return_evicted_latent:
            eviction = self._compute_eviction(
                continue_vae_latent,
                continue_source_latent,
                choice_idx,
                latent_window_size=latent_window_size,
                history_window_size=history_window_size,
            )

        # choice_idx is the index of the target chunk (0-based over total_sections). It is
        # returned so the multi-event path can look up which event owns this chunk.
        # start_section_idx is the ABSOLUTE first section of the clean_all slice —
        # the trainer needs it to derive per-section eviction validity and prompts.
        return (
            x0_latent,
            history_latent,
            target_latent,
            clean_all_vae_latent,
            choice_idx,
            start_section_idx,
            eviction,
        )

    def __len__(self):
        return len(self.samples)

    def _apply_tag_embed_override(self, feature_data, file_path):
        """Override prompt embeds with tagged-caption sidecars (reweight_v2 run).

        Active only when HELIOS_TAG_EMBED_DIR is set. Sidecars (built by
        scripts/data/encode_tag_sidecars.py) hold the UMT5 embedding of
        "<dataset*>\n<caption>" trimmed to seq_len; they are keyed by the
        RESOLVED source basename, so every reweight symlink replica
        (*_repK_*) maps to the same sidecar. A missing sidecar raises: this
        run must never silently train on an untagged caption (the generic
        retry loop in __getitem__ will surface the path in a .txt breadcrumb
        and the error spam makes the failure obvious).
        """
        tag_dir = os.environ.get("HELIOS_TAG_EMBED_DIR")
        if not tag_dir:
            return feature_data
        sidecar_path = os.path.join(tag_dir, os.path.basename(os.path.realpath(file_path)))
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        emb = sidecar["prompt_embed"]
        max_len = 512
        padded = torch.cat([emb, emb.new_zeros(max_len - emb.shape[0], emb.shape[1])], dim=0)
        for key in list(feature_data.keys()):
            if key == "prompt_embed" or key.startswith("prompt_embed_"):
                feature_data[key] = padded
        if "event_prompt_embeds" in feature_data:
            # demo multi-event .pt in this mix all hold exactly 1 event whose text
            # equals prompt_raw, so the tagged single-prompt embed substitutes 1:1.
            n_events = feature_data["event_prompt_embeds"].shape[0]
            if n_events != 1:
                raise ValueError(f"{file_path}: {n_events} events; tag override only supports 1")
            feature_data["event_prompt_embeds"] = padded.unsqueeze(0)
        feature_data["prompt_raw"] = sidecar["prompt_raw_tagged"]
        return feature_data

    def _apply_text_sidecar(self, feature_data, file_path):
        """cap-v3 text/embedding sidecar override (2026-07-10, plan: helios_organized
        workspace/doc/plans/PLAN_CAPTION_CLEANUP_RECAPTION.md §6, approved option B).

        Active only when HELIOS_TEXT_SIDECAR_DIR is set (":"-separated dirs, first
        hit wins). A sidecar <dir>/<basename of resolved .pt> holds
        {"prompt_raw": str, "prompt_embed": (L,4096) trimmed to true length}.
        The embed is re-padded to 512 and replaces every prompt_embed* key (and a
        single-event event_prompt_embeds), mirroring _apply_tag_embed_override.
        Unlike the tag override, a MISSING sidecar silently falls back to the
        in-.pt embed — rollout is incremental: only clips whose caption changed
        get a sidecar; everything else keeps training on the baked-in cap-v0.
        Mutually exclusive with HELIOS_TAG_EMBED_DIR (raises if both are set).
        """
        sidecar_dirs = os.environ.get("HELIOS_TEXT_SIDECAR_DIR")
        if not sidecar_dirs:
            return feature_data
        if os.environ.get("HELIOS_TAG_EMBED_DIR"):
            raise ValueError("HELIOS_TEXT_SIDECAR_DIR and HELIOS_TAG_EMBED_DIR are mutually exclusive")
        base = os.path.basename(os.path.realpath(file_path))
        sidecar_path = None
        for d in sidecar_dirs.split(":"):
            cand = os.path.join(d, base)
            if os.path.exists(cand):
                sidecar_path = cand
                break
        if sidecar_path is None:
            return feature_data  # fallback: keep in-.pt text/embed
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        emb = sidecar["prompt_embed"]
        max_len = 512
        if emb.shape[0] > max_len:
            raise ValueError(f"{sidecar_path}: embed len {emb.shape[0]} > {max_len}")
        padded = torch.cat([emb, emb.new_zeros(max_len - emb.shape[0], emb.shape[1])], dim=0)
        for key in list(feature_data.keys()):
            if key == "prompt_embed" or key.startswith("prompt_embed_"):
                feature_data[key] = padded.to(feature_data[key].dtype)
        if "event_prompt_embeds" in feature_data:
            n_events = feature_data["event_prompt_embeds"].shape[0]
            if n_events != 1:
                raise ValueError(f"{file_path}: {n_events} events; text sidecar only supports 1")
            feature_data["event_prompt_embeds"] = padded.to(
                feature_data["event_prompt_embeds"].dtype).unsqueeze(0)
        feature_data["prompt_raw"] = sidecar["prompt_raw"]
        if isinstance(feature_data.get("prompt_raws"), list) and len(feature_data["prompt_raws"]) == 1:
            feature_data["prompt_raws"] = [sidecar["prompt_raw"]]
        return feature_data

    def _pick_prompt_embed(self, feature_data, choice_idx=None, caption_version=None):
        """Select the prompt embed for the sampled target chunk.

        Multi-event (prompt-switching) data stores one embed per event plus a
        chunk->event map; we return the embed of the event that owns the chunk at
        ``choice_idx``. This is what teaches Helios to switch prompts at chunk
        boundaries (the transformer/forward/loss are unchanged). Falls back to the
        single-prompt caption-mixing path for ordinary Stage-1 .pt files;
        ``caption_version`` pins the fallback to one version (rollout consumers
        pass it so a multi-section unroll does not switch captions mid-rollout).
        """
        if (
            choice_idx is not None
            and "event_idx_per_chunk" in feature_data
            and "event_prompt_embeds" in feature_data
        ):
            event_per_chunk = feature_data["event_idx_per_chunk"]
            event_idx = int(event_per_chunk[min(int(choice_idx), len(event_per_chunk) - 1)].item())
            return feature_data["event_prompt_embeds"][event_idx]
        # uniformly pick one rewritten caption version if available; else legacy embed.
        available = [v for v in self.caption_versions if f"prompt_embed_{v}" in feature_data]
        if available:
            if caption_version not in available:
                caption_version = random.choice(available)
            return feature_data[f"prompt_embed_{caption_version}"]
        return feature_data["prompt_embed"]

    def __getitem__(self, idx):
        anchor_f = self.samples[idx]["num_frame"]
        anchor_h = self.samples[idx]["height"]
        anchor_w = self.samples[idx]["width"]
        while True:
            sample_info = self.samples[idx]

            if (
                anchor_f != sample_info["num_frame"]
                or anchor_h != sample_info["height"]
                or anchor_w != sample_info["width"]
            ):
                # Retry within the SAME (num_frame,h,w) bucket so the replacement always matches the
                # anchor dims; random-global retry can spiral forever when the anchor frame-count is rare.
                _bk = self.buckets.get((anchor_f, anchor_h, anchor_w))
                idx = random.choice(_bk) if _bk else random.randint(0, len(self.samples) - 1)
                print("Try to find a same dim sample, retrying...")
                continue

            try:
                base_vae_latent = None
                if (anchor_h, anchor_w) in [
                    (self.single_height // 2, self.single_width // 2),
                    (self.single_height // 4, self.single_width // 4),
                ]:
                    base_file_path = (
                        sample_info["file_path"]
                        .replace("/mid", "")
                        .replace("/low", "")
                        .replace(
                            f"{self.single_height // 2}_{self.single_width // 2}",
                            f"{self.single_height}_{self.single_width}",
                        )
                        .replace(
                            f"{self.single_height // 4}_{self.single_width // 4}",
                            f"{self.single_height}_{self.single_width}",
                        )
                    )
                    base_vae_latent = torch.load(base_file_path, map_location="cpu", weights_only=False)["vae_latent"]

                feature_data = torch.load(sample_info["file_path"], map_location="cpu", weights_only=False)
                feature_data = self._apply_tag_embed_override(feature_data, sample_info["file_path"])
                feature_data = self._apply_text_sidecar(feature_data, sample_info["file_path"])
                (
                    x0_latent,
                    history_latent,
                    target_latent,
                    clean_all_vae_latent,
                    choice_idx,
                    start_section_idx,
                    eviction,
                ) = self.prepare_stage1_latent(feature_data["vae_latent"], idx, base_vae_latent)
                if self.return_prompt_raw:
                    prompt_raws = feature_data["prompt_raw"]
                break
            except Exception:
                # Same-bucket retry (see note above): a corrupt/transiently-unreadable .pt must not send us
                # spiralling over random global indices looking for a same-dim replacement.
                _bk = self.buckets.get((anchor_f, anchor_h, anchor_w))
                idx = random.choice(_bk) if _bk else random.randint(0, len(self.samples) - 1)
                print(f"Error loading {sample_info['file_path']}, retrying...")
                file_name = os.path.basename(sample_info["file_path"])
                txt_name = f"{file_name}.txt"
                with open(txt_name, "w") as f:
                    f.write(sample_info["file_path"] + "\n")

        output_dict = {
            "uttid": sample_info["uttid"],
            "bucket_key": sample_info["bucket_key"],
            "dataset_name": sample_info["dataset_name"],
            "num_frame": sample_info["num_frame"],
            "height": sample_info["height"],
            "width": sample_info["width"],
            "x0_latents": x0_latent,
            "history_latents": history_latent,
            "target_latents": target_latent,
            "clean_all_latents": clean_all_vae_latent,
            # Per-chunk prompt for multi-event data (else random caption-version mixing).
            "prompt_embeds": self._pick_prompt_embed(feature_data, choice_idx),
            "prompt_attention_masks": feature_data.get("prompt_attention_mask", None),
        }

        if self.return_prompt_raw:
            output_dict["prompt_raws"] = prompt_raws

        if self.return_evicted_latent:
            evicted_latent, evicted_history, evicted_valid_frames = eviction
            output_dict["evicted_latents"] = evicted_latent
            output_dict["evicted_history_latents"] = evicted_history
            output_dict["evicted_valid_frames"] = evicted_valid_frames

        if self.return_rollout_metadata:
            # Unroll consumers need the ABSOLUTE start section (per-section eviction
            # validity depends on the position relative to the zero prefix) and one
            # prompt per unrolled section (multi-event data maps each chunk to its
            # owning event; design ch.2 D6).
            assert start_section_idx is not None, (
                "return_rollout_metadata requires return_all_vae_latent"
            )
            output_dict["start_section_idx"] = start_section_idx
            # Draw the non-event caption version ONCE per rollout: per-section
            # independent draws switch captions mid-rollout with probability
            # 1 - (1/V)^(U-1) (~98% at V=4, U=4), injecting text-conditioning
            # churn into the exact loss that trains temporal continuity.
            # Event samples are unaffected (per-chunk event mapping wins).
            available_versions = [
                v for v in self.caption_versions if f"prompt_embed_{v}" in feature_data
            ]
            rollout_caption_version = (
                random.choice(available_versions) if available_versions else None
            )
            output_dict["section_prompt_embeds"] = [
                self._pick_prompt_embed(
                    feature_data,
                    start_section_idx + u,
                    caption_version=rollout_caption_version,
                )
                for u in range(self.num_rollout_sections)
            ]

        return output_dict


class BucketedSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        drop_last=False,
        shuffle=True,
        seed=42,
        dataset_sampling_ratios=None,
        num_sp_groups=1,
        sp_world_size=1,
        global_rank=0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.generator = torch.Generator()
        self.buckets = dataset.buckets
        self._epoch = 0

        # Distributed parameters
        self.num_sp_groups = num_sp_groups
        self.sp_world_size = sp_world_size
        self.global_rank = global_rank
        self.ith_sp_group = self.global_rank // self.sp_world_size

        self.dataset_sampling_ratios = (
            {key.rstrip("/"): value for key, value in dataset_sampling_ratios.items()}
            if dataset_sampling_ratios is not None
            else {}
        )
        self._prepare_dataset_buckets()

    def _prepare_dataset_buckets(self):
        self.dataset_buckets = {}

        for bucket_key, sample_indices in self.buckets.items():
            dataset_groups = {}
            for idx in sample_indices:
                dataset_name = self.dataset.samples[idx]["dataset_name"]
                if dataset_name not in dataset_groups:
                    dataset_groups[dataset_name] = []
                dataset_groups[dataset_name].append(idx)
            self.dataset_buckets[bucket_key] = dataset_groups

    def set_epoch(self, epoch):
        self._epoch = epoch

    def _shard_indices_for_sp_group(self, indices):
        """
        Shard indices across SP groups, similar to DP_SP_BatchSampler.
        Each SP group gets a disjoint subset of the data.
        """
        if self.num_sp_groups == 1:
            return indices

        # Convert to tensor if it's a list
        if isinstance(indices, list):
            indices_tensor = torch.tensor(indices, dtype=torch.long)
        else:
            indices_tensor = indices

        # Pad indices if necessary to make it divisible by num_sp_groups
        total_size = len(indices_tensor)
        if total_size % self.num_sp_groups != 0:
            if not self.drop_last:
                padding_size = self.num_sp_groups - (total_size % self.num_sp_groups)
                indices_tensor = torch.cat([indices_tensor, indices_tensor[:padding_size]])
        else:
            # If drop_last, truncate to be divisible
            if self.drop_last:
                truncate_size = (total_size // self.num_sp_groups) * self.num_sp_groups
                indices_tensor = indices_tensor[:truncate_size]

        # Shard: each SP group gets every num_sp_groups-th element
        sp_group_indices = indices_tensor[self.ith_sp_group :: self.num_sp_groups]

        return sp_group_indices.tolist()

    def _apply_global_ratio_sampling(self):
        if not self.dataset_sampling_ratios:
            return

        dataset_sample_map = {}
        for bucket_key, dataset_groups in self.dataset_buckets.items():
            for dataset_name, indices in dataset_groups.items():
                if dataset_name not in dataset_sample_map:
                    dataset_sample_map[dataset_name] = {"indices": [], "buckets": []}
                dataset_sample_map[dataset_name]["indices"].extend(indices)
                dataset_sample_map[dataset_name]["buckets"].extend([bucket_key] * len(indices))

        total_samples = sum(len(info["indices"]) for info in dataset_sample_map.values())
        total_ratio = sum(self.dataset_sampling_ratios.values())

        sampled_dataset_map = {}
        for dataset_name, info in dataset_sample_map.items():
            if dataset_name in self.dataset_sampling_ratios:
                ratio = self.dataset_sampling_ratios[dataset_name] / total_ratio
                target_samples = max(1, int(total_samples * ratio))

                indices = info["indices"]
                buckets = info["buckets"]

                if len(indices) >= target_samples:
                    selected = torch.randperm(len(indices), generator=self.generator)[:target_samples].tolist()
                    sampled_indices = [indices[i] for i in selected]
                    sampled_buckets = [buckets[i] for i in selected]
                else:
                    sampled_indices = []
                    sampled_buckets = []
                    remaining = target_samples

                    while remaining > 0:
                        repeat_count = min(remaining, len(indices))
                        selected = torch.randperm(len(indices), generator=self.generator)[:repeat_count].tolist()
                        sampled_indices.extend([indices[i] for i in selected])
                        sampled_buckets.extend([buckets[i] for i in selected])
                        remaining -= repeat_count

                sampled_dataset_map[dataset_name] = {"indices": sampled_indices, "buckets": sampled_buckets}
            else:
                sampled_dataset_map[dataset_name] = info

        new_dataset_buckets = {}
        for bucket_key in self.dataset_buckets.keys():
            new_dataset_buckets[bucket_key] = {}

        for dataset_name, info in sampled_dataset_map.items():
            indices = info["indices"]
            buckets = info["buckets"]

            for idx, bucket_key in zip(indices, buckets):
                if dataset_name not in new_dataset_buckets[bucket_key]:
                    new_dataset_buckets[bucket_key][dataset_name] = []
                new_dataset_buckets[bucket_key][dataset_name].append(idx)

        self.dataset_buckets = new_dataset_buckets

    def __iter__(self):
        # Use epoch-level seed for reproducibility
        epoch_seed = self.seed + self._epoch
        self.generator.manual_seed(epoch_seed)

        if self.dataset_sampling_ratios:
            self._apply_global_ratio_sampling()

        bucket_iterators = {}
        bucket_batches = {}

        for bucket_key, dataset_groups in self.dataset_buckets.items():
            balanced_indices = self._create_balanced_indices(dataset_groups)

            # Global shuffle before sharding (important for distributed consistency)
            if self.shuffle:
                perm = torch.randperm(len(balanced_indices), generator=self.generator).tolist()
                balanced_indices = [balanced_indices[i] for i in perm]

            # Shard indices for this SP group
            sp_group_indices = self._shard_indices_for_sp_group(balanced_indices)

            batches = []
            for i in range(0, len(sp_group_indices), self.batch_size):
                batch = sp_group_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

            if batches:
                bucket_batches[bucket_key] = batches
                bucket_iterators[bucket_key] = iter(batches)

        remaining_buckets = list(bucket_iterators.keys())

        while remaining_buckets:
            idx = torch.randint(len(remaining_buckets), (1,), generator=self.generator).item()
            bucket_key = remaining_buckets[idx]
            bucket_iter = bucket_iterators[bucket_key]

            try:
                batch = next(bucket_iter)
                yield batch
            except StopIteration:
                remaining_buckets.remove(bucket_key)

    def _create_balanced_indices(self, dataset_groups):
        return sum(dataset_groups.values(), [])

    def _equal_sampling(self, dataset_groups):
        all_indices = []
        dataset_names = list(dataset_groups.keys())

        if len(dataset_names) <= 1:
            return sum(dataset_groups.values(), [])

        min_samples = min(len(indices) for indices in dataset_groups.values())

        for dataset_name, indices in dataset_groups.items():
            if len(indices) > min_samples:
                selected = torch.randperm(len(indices), generator=self.generator)[:min_samples].tolist()
                sampled_indices = [indices[i] for i in selected]
            else:
                sampled_indices = indices
            all_indices.extend(sampled_indices)

        return all_indices

    def _ratio_sampling(self, dataset_groups):
        return sum(dataset_groups.values(), [])

    def __len__(self):
        if self.dataset_sampling_ratios:
            temp_generator = torch.Generator()
            temp_generator.manual_seed(self.seed)

            dataset_sample_map = {}
            for bucket_key, dataset_groups in self.dataset_buckets.items():
                for dataset_name, indices in dataset_groups.items():
                    if dataset_name not in dataset_sample_map:
                        dataset_sample_map[dataset_name] = []
                    dataset_sample_map[dataset_name].extend(indices)

            total_samples = sum(len(indices) for indices in dataset_sample_map.values())
            total_ratio = sum(self.dataset_sampling_ratios.values())

            sampled_total = 0
            for dataset_name, indices in dataset_sample_map.items():
                if dataset_name in self.dataset_sampling_ratios:
                    ratio = self.dataset_sampling_ratios[dataset_name] / total_ratio
                    target_samples = max(1, int(total_samples * ratio))
                    sampled_total += target_samples
                else:
                    sampled_total += len(indices)

            # Account for SP group sharding
            sp_group_samples = sampled_total // self.num_sp_groups
            if not self.drop_last and sampled_total % self.num_sp_groups != 0:
                sp_group_samples += 1

            total_batches = sp_group_samples // self.batch_size
            if not self.drop_last and sp_group_samples % self.batch_size != 0:
                total_batches += 1
            return total_batches
        else:
            total_batches = 0
            for bucket_key, dataset_groups in self.dataset_buckets.items():
                balanced_indices = self._create_balanced_indices(dataset_groups)

                # Account for SP group sharding
                sp_group_size = len(balanced_indices) // self.num_sp_groups
                if not self.drop_last and len(balanced_indices) % self.num_sp_groups != 0:
                    sp_group_size += 1

                num_batches = sp_group_size // self.batch_size
                if not self.drop_last and sp_group_size % self.batch_size != 0:
                    num_batches += 1
                total_batches += num_batches
            return total_batches


def collate_fn(batch):
    return {
        key: torch.stack([d[key] for d in batch])
        if isinstance(batch[0][key], torch.Tensor)
        else [d[key] for d in batch]
        for key in batch[0]
    }


if __name__ == "__main__":
    import torch.distributed.checkpoint as dcp
    from accelerate import Accelerator
    from torchdata.stateful_dataloader import StatefulDataLoader

    feature_folder = [
        "demo_data/ultravideo-long",
    ]
    dataloader_num_workers = 0
    batch_size = 2
    num_train_epochs = 2
    seed = 0
    output_dir = "accelerate_checkpoints"
    checkpoint_dirs = (
        [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if os.path.exists(output_dir)
        else []
    )

    dataset_ratios = {}
    # dataset_ratios = {
    #     "demo_data/ultravideo-long": 0.9,
    # }

    accelerator = Accelerator()
    print(accelerator.process_index, accelerator.num_processes)

    dataset = BucketedFeatureDataset(
        feature_folder,
        force_rebuild=True,
        return_all_vae_latent=True,
        return_prompt_raw=True,
        single_res=True,
        single_height=384,
        single_width=640,
        seed=seed,
    )
    sampler = BucketedSampler(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        dataset_sampling_ratios=dataset_ratios,
        seed=seed,
        # num_sp_groups=get_world_size() // get_sp_world_size(),
        # sp_world_size=get_sp_world_size(),
        # global_rank=get_world_rank(),
        num_sp_groups=accelerator.num_processes // 1,
        sp_world_size=1,
        global_rank=accelerator.process_index,
    )
    dataloader = StatefulDataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=dataloader_num_workers
    )

    print(len(dataset), len(dataloader))
    print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")

    step = 0
    global_step = 0
    first_epoch = 0
    num_update_steps_per_epoch = len(dataloader)
    if checkpoint_dirs:
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
        checkpoint_path = os.path.join(output_dir, latest_checkpoint)
        print(f"Found checkpoint: {checkpoint_path}")

        accelerator.load_state(checkpoint_path)
        global_step = int(latest_checkpoint.split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch

        states = {
            "dataloader": dataloader,
        }
        dcp_dir = os.path.join(checkpoint_path, "distributed_checkpoint")
        dcp.load(states, checkpoint_id=dcp_dir)

        print(f"Resuming from step {global_step}, epoch {first_epoch}")

    print("Testing dataloader...")
    step = global_step
    dataset_counts = defaultdict(int)
    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        for i, batch in enumerate(dataloader):
            # Get metadata
            uttid = batch["uttid"]
            num_frame = batch["num_frame"]
            height = batch["height"]
            width = batch["width"]
            bucket_key = batch["bucket_key"]

            # Get feature
            x0_latents = batch["x0_latents"]
            history_latents = batch["history_latents"]
            target_latents = batch["target_latents"]
            prompt_embeds = batch["prompt_embeds"]

            if accelerator.process_index == 0:
                # print info
                print(f" Step {step}:")
                print(f"  Batch {i}:")
                # print(f"  Data Name: {batch['dataset_name']}")
                print(f"  Batch size: {len(uttid)}")
                print(f"  Uttids: {uttid}")
                print(f"  Dimensions - frames: {num_frame[0]}, height: {height[0]}, width: {width[0]}")
                print(f"  Bucket key: {bucket_key[0]}")
                print(f"  X0 latent shape: {x0_latents.shape}")
                print(f"  History latent shape: {history_latents.shape}")
                print(f"  Context latent shape: {target_latents.shape}")
                print(f"  Prompt embed shape: {prompt_embeds.shape}")
                # print(f"  Prompt attention mask shape: {prompt_attention_masks.shape}")

                # verify
                assert all(nf == num_frame[0] for nf in num_frame), "Frame numbers not consistent in batch"
                assert all(h == height[0] for h in height), "Heights not consistent in batch"
                assert all(w == width[0] for w in width), "Widths not consistent in batch"

                print("  ✓ Batch dimensions are consistent")

            for dataset_name in batch["dataset_name"]:
                dataset_counts[dataset_name] += 1

            step += 1

            # if step == 20:
            #     checkpoint_dir = f"checkpoint-{step}"
            #     save_path = os.path.join(output_dir, checkpoint_dir)
            #     os.makedirs(save_path, exist_ok=True)

            #     if accelerator.is_main_process:
            #         print(f"Saving checkpoint at step {step}")

            #         accelerator.save_state(save_path)

            #     print(accelerator.process_index, accelerator.num_processes)
            #     states = {
            #         "dataloader": dataloader,
            #     }
            #     dcp_dir = os.path.join(save_path, "distributed_checkpoint")
            #     dcp.save(states, checkpoint_id=dcp_dir)

    print("实际采样统计:", dict(dataset_counts))
