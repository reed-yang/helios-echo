"""
Non-invasive attention probe for Helios streaming inference.

Goal: measure, for the *noisy* query tokens of each autoregressive chunk, how
attention mass is distributed over the five key-token categories:

    sink          -- first-frame anchor (x0 prefix, is_keep_x0), high-res patch_short
    short_highres -- most-recent 1x frame,        high-res patch_short
    mid           -- 2x patch_mid history
    long_lowres   -- far 4x patch_long history
    noisy         -- the current chunk being generated (self)

Key facts about the *actual* inference path (helios/diffusers_version):
  * infer_helios.py runs `helios.diffusers_version.transformer_helios_diffusers`.
  * Self-attention (`attn1`, is_cross_attention=False) does FULL bidirectional
    attention over the concatenated sequence, laid out (see forward()):
        [ long_lowres | mid | short(=[sink, recent_1x]) | noisy ]
    i.e. history first (long,mid,short), noisy last.
  * The real kernel is `dispatch_attention_fn` (flash / SDPA) which never
    returns attention weights, so we add an EAGER softmax(QK^T/sqrt(d)) branch,
    guarded by PROBE.enabled, that aggregates by category immediately and never
    materialises a full [Lq,Lk] matrix except for a few representative cells.

Everything here is opt-in: nothing runs unless `PROBE.enabled` is set by
install()+run_probe. Normal inference path is untouched when the probe is off.
"""

import math

import torch

from helios.diffusers_version import transformer_helios_diffusers as T

CATEGORIES = ["sink", "short_highres", "mid", "long_lowres", "noisy"]

# Fixed order in the concatenated key/query sequence (history first, noisy last):
#   [ long_lowres | mid | sink | short_highres | noisy ]


class _Probe:
    def __init__(self):
        self.enabled = False
        # per-forward segment token counts captured by conv forward-hooks
        self.seg = {}            # {'noisy':n, 'short':n, 'mid':n, 'long':n, 'short_T':Ts}
        # generation-time cursor, driven by the forward wrapper
        self.cur_chunk = -1
        self.cur_step = -1
        self.cur_t = None
        self.prev_t = None
        # config
        self.n_query_sample = 512
        self.record_steps = None      # set[int] of within-chunk denoise steps to record; None = all
        # Counterfactual "amplify" scale applied to ALL history keys when the
        # model's own is_amplify_history mechanism is OFF (which is the case for
        # the released Base & Distilled weights). Lets the UI show natural vs.
        # mechanism-forced attention. 1.0 => disabled.
        self.sim_amplify_scale = 4.0
        self.full_map_cells = set()   # {(chunk, step, layer)} for which to dump a downsampled QxK map
        self.full_map_q = 64
        self.full_map_k = 256
        # outputs
        self.records = []
        self.full_maps = []
        self.meta = {}
        self.boundaries_per_step = {}  # {(chunk,step): {cat:[s,e]}}
        self._warned = False

    def reset_run(self):
        self.seg = {}
        self.cur_chunk = -1
        self.cur_step = -1
        self.cur_t = None
        self.prev_t = None
        self.records = []
        self.full_maps = []
        self.boundaries_per_step = {}

    # ----- boundaries -----
    def compute_boundaries(self, total, ocl):
        """Return {cat: (start,end)} covering [0,total) with no overlap.

        Cross-validated: long+mid+short == total-ocl (history_seq_len) and
        the noisy block is exactly the last `ocl` tokens.
        """
        n_long = self.seg["long"]
        n_mid = self.seg["mid"]
        n_short = self.seg["short"]
        Ts = self.seg["short_T"]
        n_noisy = ocl
        # sink = first temporal slice of the short block (x0 prefix is prepended
        # along the temporal axis and patch_short has temporal stride 1).
        n_sink = n_short // Ts
        n_short_hr = n_short - n_sink
        hist = n_long + n_mid + n_short
        assert hist + n_noisy == total, (
            f"boundary mismatch: long{n_long}+mid{n_mid}+short{n_short}+noisy{n_noisy} != total{total}"
        )
        o = 0
        b = {}
        b["long_lowres"] = (o, o + n_long); o += n_long
        b["mid"] = (o, o + n_mid); o += n_mid
        b["sink"] = (o, o + n_sink); o += n_sink
        b["short_highres"] = (o, o + n_short_hr); o += n_short_hr
        b["noisy"] = (o, o + n_noisy); o += n_noisy
        assert o == total
        return b, hist

    def query_indices(self, ocl, device):
        n = min(self.n_query_sample, ocl)
        idx = torch.linspace(0, ocl - 1, n, device=device).round().long().unique()
        return idx


PROBE = _Probe()


# ---------------------------------------------------------------------------
# eager aggregation
# ---------------------------------------------------------------------------
def _aggregate(qsel, key, boundaries, scale):
    """qsel: [nq,H,D]  key: [Lk,H,D]  -> per-head mass & entropy.

    Returns dict with per-head mass (list over cats), head-avg mass/per_token,
    entropy (head-avg), computed only over the sampled noisy queries.
    """
    H = qsel.shape[1]
    # scores: [H, nq, Lk]
    scores = torch.einsum("nhd,khd->hnk", qsel.float(), key.float()) * scale
    probs = torch.softmax(scores, dim=-1)            # rows sum to 1
    del scores
    row_sum = probs.sum(-1).mean().item()            # ~1.0 sanity

    per_head_mass = {}
    for cat, (s, e) in boundaries.items():
        if e > s:
            m = probs[:, :, s:e].sum(-1).mean(dim=1)  # [H]
        else:
            m = torch.zeros(H, device=probs.device)
        per_head_mass[cat] = m                        # tensor [H]

    # entropy (nats), head-averaged over sampled queries
    ent = (-(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)).mean().item()
    del probs
    return per_head_mass, ent, row_sum


def _downsample_map(qsel, key, boundaries, scale, nq_bins, nk_bins):
    """Head-averaged, pooled [nq_bins x nk_bins] attention map for one cell."""
    H = qsel.shape[1]
    Lk = key.shape[0]
    scores = torch.einsum("nhd,khd->hnk", qsel.float(), key.float()) * scale
    probs = torch.softmax(scores, dim=-1).mean(0)     # [nq, Lk] head-avg
    del scores
    nq = probs.shape[0]
    # pool queries
    qb = torch.linspace(0, nq, nq_bins + 1).round().long()
    kb = torch.linspace(0, Lk, nk_bins + 1).round().long()
    out = torch.zeros(nq_bins, nk_bins)
    for i in range(nq_bins):
        qs, qe = qb[i].item(), max(qb[i + 1].item(), qb[i].item() + 1)
        rowblock = probs[qs:qe]
        for j in range(nk_bins):
            ks, ke = kb[j].item(), max(kb[j + 1].item(), kb[j].item() + 1)
            out[i, j] = rowblock[:, ks:ke].sum(-1).mean()
    # boundary positions in kbin coords
    bnd_bins = {c: [int(round(s / Lk * nk_bins)), int(round(e / Lk * nk_bins))]
                for c, (s, e) in boundaries.items()}
    return out.tolist(), bnd_bins


# ---------------------------------------------------------------------------
# patched attention processor __call__ (mirror of the original + eager branch)
# ---------------------------------------------------------------------------
def _make_probing_call():
    def probing_call(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, rotary_emb=None, original_context_length=None):
        query, key, value = T._get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = T.apply_rotary_emb_transposed(query, rotary_emb)
            key = T.apply_rotary_emb_transposed(key, rotary_emb)

        key_pre = None
        scale_key_val = None
        if not attn.is_cross_attention and attn.is_amplify_history:
            history_seq_len = hidden_states.shape[1] - original_context_length
            if history_seq_len > 0:
                key_pre = key  # pre-amplify keys (rope already applied)
                scale_key = 1.0 + torch.sigmoid(attn.history_key_scale) * (attn.max_scale - 1.0)
                scale_key_val = scale_key.detach().float().cpu().tolist()
                if attn.history_scale_mode == "per_head":
                    scale_key = scale_key.view(1, 1, -1, 1)
                key = torch.cat([key[:, :history_seq_len] * scale_key, key[:, history_seq_len:]], dim=1)

        if PROBE.enabled and not attn.is_cross_attention:
            try:
                _record(attn, query, key, value, original_context_length, key_pre, scale_key_val)
            except Exception as e:  # never break inference because of the probe
                if not PROBE._warned:
                    print(f"[attn_probe] record failed (suppressing further): {e}")
                    PROBE._warned = True

        hidden_states = T.dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            backend=self._attention_backend,
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    return probing_call


@torch.no_grad()
def _record(attn, query, key, value, ocl, key_pre, scale_key_val):
    layer = getattr(attn, "_probe_layer_idx", -1)
    if layer < 0 or not PROBE.seg:
        return
    total = query.shape[1]
    # only record chosen denoise steps
    if PROBE.record_steps is not None and PROBE.cur_step not in PROBE.record_steps:
        return

    boundaries, hist = PROBE.compute_boundaries(total, ocl)
    key = key[0]              # [Lk,H,D]  (real: post real-amplify if mechanism on)
    q = query[0]             # [Lq,H,D]
    D = q.shape[-1]
    scale = 1.0 / math.sqrt(D)

    idx = PROBE.query_indices(ocl, q.device)
    qsel = q[hist + idx]      # sampled noisy queries [nq,H,D]

    # ---- natural distribution (the released model's actual behavior) ----
    if key_pre is not None:
        nat_key = key_pre[0]          # real mechanism ON: natural = pre-amplify keys
    else:
        nat_key = key                 # mechanism OFF: the real keys are already natural
    per_head_nat, ent, row_sum = _aggregate(qsel, nat_key, boundaries, scale)

    # ---- amplified distribution (real mechanism, else counterfactual sim) ----
    amp_simulated = False
    amp_scale_note = scale_key_val
    if key_pre is not None:
        amp_key = key                 # real amplified keys
    elif PROBE.sim_amplify_scale and PROBE.sim_amplify_scale != 1.0:
        amp_key = nat_key.clone()
        amp_key[:hist] = amp_key[:hist] * PROBE.sim_amplify_scale  # boost all history keys
        amp_simulated = True
        amp_scale_note = PROBE.sim_amplify_scale
    else:
        amp_key = nat_key
    if amp_key is nat_key:
        per_head_amp = per_head_nat
    else:
        per_head_amp, _, _ = _aggregate(qsel, amp_key, boundaries, scale)

    n_tok = {c: (e - s) for c, (s, e) in boundaries.items()}

    def headavg(phm):
        return {c: float(phm[c].mean().item()) for c in CATEGORIES}

    mass_nat = headavg(per_head_nat)
    mass_amp = headavg(per_head_amp)
    per_token_nat = {c: (mass_nat[c] / n_tok[c] if n_tok[c] > 0 else 0.0) for c in CATEGORIES}
    per_token_amp = {c: (mass_amp[c] / n_tok[c] if n_tok[c] > 0 else 0.0) for c in CATEGORIES}
    hist_share = float(1.0 - mass_nat["noisy"])

    rec = {
        "chunk": PROBE.cur_chunk,
        "denoise_step": PROBE.cur_step,
        "t": PROBE.cur_t,
        "layer": layer,
        "mass": mass_nat,
        "per_token": per_token_nat,
        "mass_amplified": mass_amp,
        "per_token_amplified": per_token_amp,
        "per_head_mass": {c: [round(float(x), 6) for x in per_head_nat[c].tolist()] for c in CATEGORIES},
        "n_tokens": n_tok,
        "entropy": round(ent, 5),
        "history_share": round(hist_share, 6),
        "row_sum": round(row_sum, 5),
        "amplify_active": bool(key_pre is not None),
        "amplify_simulated": amp_simulated,
        "amplify_scale": amp_scale_note,
    }
    PROBE.records.append(rec)
    PROBE.boundaries_per_step[(PROBE.cur_chunk, PROBE.cur_step)] = {
        c: [boundaries[c][0], boundaries[c][1]] for c in CATEGORIES
    }

    cell = (PROBE.cur_chunk, PROBE.cur_step, layer)
    if cell in PROBE.full_map_cells:
        mat, bnd_bins = _downsample_map(qsel, key, boundaries, scale,
                                        PROBE.full_map_q, PROBE.full_map_k)
        PROBE.full_maps.append({
            "chunk": PROBE.cur_chunk, "denoise_step": PROBE.cur_step, "layer": layer,
            "matrix": mat, "kbin_boundaries": bnd_bins,
            "nq_bins": PROBE.full_map_q, "nk_bins": PROBE.full_map_k,
        })


# ---------------------------------------------------------------------------
# conv forward-hooks: capture exact per-segment token counts (no guessing)
# ---------------------------------------------------------------------------
def _seg_hook(name):
    def hook(module, inp, out):
        if not PROBE.enabled:
            return
        # out: [B, C, T, H, W]
        B, C, Tt, Hh, Ww = out.shape
        n = Tt * Hh * Ww
        if name == "noisy":
            PROBE.seg["noisy"] = n
        elif name == "short":
            PROBE.seg["short"] = n
            PROBE.seg["short_T"] = Tt
        elif name == "mid":
            PROBE.seg["mid"] = n
        elif name == "long":
            PROBE.seg["long"] = n
    return hook


# ---------------------------------------------------------------------------
# forward wrapper: track (chunk, denoise_step) from the timestep schedule
# ---------------------------------------------------------------------------
def _wrap_forward():
    orig = T.HeliosTransformer3DModel.forward

    def wrapped(self, hidden_states, timestep, *args, **kwargs):
        if PROBE.enabled:
            try:
                t = float(timestep.reshape(-1)[0].item())
            except Exception:
                t = None
            if t is not None:
                # within a chunk the timestep strictly decreases; a jump back up
                # marks the start of a new autoregressive chunk.
                if PROBE.prev_t is None or t > PROBE.prev_t + 1e-6:
                    PROBE.cur_chunk += 1
                    PROBE.cur_step = 0
                else:
                    PROBE.cur_step += 1
                PROBE.prev_t = t
                PROBE.cur_t = t
        return orig(self, hidden_states, timestep, *args, **kwargs)

    T.HeliosTransformer3DModel.forward = wrapped
    return orig


_INSTALLED = False


def install(transformer):
    """Patch the processor, wrap forward, register conv hooks, tag layer ids."""
    global _INSTALLED
    # tag each self-attn with its layer index
    for i, block in enumerate(transformer.blocks):
        block.attn1._probe_layer_idx = i

    # register conv hooks (idempotent-ish: only once)
    if not getattr(transformer, "_probe_hooks", None):
        h = []
        h.append(transformer.patch_embedding.register_forward_hook(_seg_hook("noisy")))
        if hasattr(transformer, "patch_short"):
            h.append(transformer.patch_short.register_forward_hook(_seg_hook("short")))
            h.append(transformer.patch_mid.register_forward_hook(_seg_hook("mid")))
            h.append(transformer.patch_long.register_forward_hook(_seg_hook("long")))
        transformer._probe_hooks = h

    if not _INSTALLED:
        T.HeliosAttnProcessor.__call__ = _make_probing_call()
        _wrap_forward()
        _INSTALLED = True
    return transformer
