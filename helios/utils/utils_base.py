import gc
import html
import math
import os
import random
from typing import List, Literal, Optional, Union

import ftfy
import regex as re
import torch
from accelerate.logging import get_logger


logger = get_logger(__name__)

NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


# ======================================== memory monitoring ========================================
def get_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return {"allocated": allocated, "reserved": reserved, "max_allocated": max_allocated}
    return None


def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()


# ======================================== initialize ========================================
def get_config_value(args, name):
    if hasattr(args, name):
        return getattr(args, name)
    elif hasattr(args, "training_config") and hasattr(args.training_config, name):
        return getattr(args.training_config, name)
    else:
        raise AttributeError(f"Neither args nor args.training_config has attribute '{name}'")


def compare_configs(existing_conf, current_conf, path="", ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = set()

    mismatches = []

    all_keys = set(existing_conf.keys()) | set(current_conf.keys())

    for key in all_keys:
        current_path = f"{path}.{key}" if path else key

        if current_path in ignore_keys or key in ignore_keys:
            continue

        if key not in existing_conf:
            mismatches.append(f"Key '{current_path}' missing in existing config")
        elif key not in current_conf:
            mismatches.append(f"Key '{current_path}' missing in current config")
        else:
            existing_val = existing_conf[key]
            current_val = current_conf[key]

            if isinstance(existing_val, dict) and isinstance(current_val, dict):
                mismatches.extend(compare_configs(existing_val, current_val, current_path, ignore_keys))
            elif existing_val != current_val:
                mismatches.append(f"Key '{current_path}': existing={existing_val} vs current={current_val}")

    return mismatches


def get_optimizer(args, accelerator, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimizer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.training_config.learning_rate,
            betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
            eps=args.training_config.adam_epsilon,
            weight_decay=args.training_config.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.training_config.optimizer.lower() not in supported_optimizers:
        accelerator.print(
            f"Unsupported choice of optimizer: {args.training_config.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.training_config.optimizer = "adamw"

    if args.training_config.use_8bit_adam and args.training_config.optimizer.lower() not in ["adam", "adamw"]:
        accelerator.print(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.training_config.optimizer.lower()}"
        )

    if args.training_config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.training_config.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.training_config.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
            eps=args.training_config.adam_epsilon,
            weight_decay=args.training_config.adam_weight_decay,
        )
    elif args.training_config.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.training_config.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
            eps=args.training_config.adam_epsilon,
            weight_decay=args.training_config.adam_weight_decay,
        )
    elif args.training_config.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.training_config.learning_rate <= 0.1:
            accelerator.print(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
            beta3=args.training_config.prodigy_beta3,
            weight_decay=args.training_config.adam_weight_decay,
            eps=args.training_config.adam_epsilon,
            decouple=args.training_config.prodigy_decouple,
            use_bias_correction=args.training_config.prodigy_use_bias_correction,
            safeguard_warmup=args.training_config.prodigy_safeguard_warmup,
        )

    return optimizer


# ======================================== checkpoints related ========================================
def save_extra_components(args, model=None, model_state_dict=None, output_dir=None):
    if model is None and model_state_dict is None:
        raise ValueError("Either 'model' or 'model_state_dict' must be provided")

    if output_dir is None:
        raise ValueError("output_dir must be provided")

    os.makedirs(output_dir, exist_ok=True)
    state_dict = {}

    # Determine whether to use model or model_state_dict
    use_state_dict = model_state_dict is not None

    # 1. Save patch_short, patch_mid, patch_long (formerly multi_term_memory_patchg)
    if args.training_config.is_enable_stage1 and (
        args.training_config.is_train_full_multi_term_memory_patchg
        or args.training_config.is_train_lora_multi_term_memory_patchg
    ):
        patch_names = ["patch_short", "patch_mid", "patch_long"]

        if use_state_dict:
            # Extract from state_dict
            for k, v in model_state_dict.items():
                if any(k.startswith(f"{p}.") for p in patch_names):
                    state_dict[k] = v.detach().clone().cpu() if torch.is_tensor(v) else v
        else:
            # Extract from model
            for p in patch_names:
                if hasattr(model, p):
                    patch_module = getattr(model, p)
                    for k, v in patch_module.state_dict().items():
                        state_dict[f"{p}.{k}"] = v.detach().clone().cpu()

    # 1b. Save patch_embedding when it is FULLY fine-tuned (is_train_full_patch_embedding). The PEFT
    # LoRA state-dict does not include it and save_lora_weights won't capture it, so persist it here.
    if getattr(args.training_config, "is_train_full_patch_embedding", False):
        if use_state_dict:
            for k, v in model_state_dict.items():
                if k.startswith("patch_embedding."):
                    state_dict[k] = v.detach().clone().cpu() if torch.is_tensor(v) else v
        elif hasattr(model, "patch_embedding"):
            for k, v in model.patch_embedding.state_dict().items():
                state_dict[f"patch_embedding.{k}"] = v.detach().clone().cpu()

    # 2. Save LoRA layers from all transformer blocks
    if args.training_config.restrict_self_attn and args.training_config.is_train_restrict_lora:
        if use_state_dict:
            # Extract LoRA parameters from state_dict
            for k, v in model_state_dict.items():
                if any(lora_key in k for lora_key in [".q_loras.", ".k_loras.", ".v_loras."]):
                    state_dict[k] = v.detach().clone().cpu() if torch.is_tensor(v) else v
        else:
            # Extract from model
            for block_idx, block in enumerate(model.blocks):
                if hasattr(block.attn1, "q_loras"):
                    for k, v in block.attn1.q_loras.state_dict().items():
                        state_dict[f"blocks.{block_idx}.attn1.q_loras.{k}"] = v.detach().clone().cpu()

                if hasattr(block.attn1, "k_loras"):
                    for k, v in block.attn1.k_loras.state_dict().items():
                        state_dict[f"blocks.{block_idx}.attn1.k_loras.{k}"] = v.detach().clone().cpu()

                if hasattr(block.attn1, "v_loras"):
                    for k, v in block.attn1.v_loras.state_dict().items():
                        state_dict[f"blocks.{block_idx}.attn1.v_loras.{k}"] = v.detach().clone().cpu()

    # 3. Save History Scale parameters
    if args.training_config.is_amplify_history:
        if use_state_dict:
            # Extract history_key_scale from state_dict
            for k, v in model_state_dict.items():
                if "history_key_scale" in k:
                    state_dict[k] = v.detach().clone().cpu() if torch.is_tensor(v) else v
        else:
            # Extract from model
            for block_idx, block in enumerate(model.blocks):
                if hasattr(block.attn1, "history_key_scale"):
                    state_dict[f"blocks.{block_idx}.attn1.history_key_scale"] = (
                        block.attn1.history_key_scale.detach().clone().cpu()
                    )

    # 4. Save GAN parameters
    if args.training_config.is_use_gan:
        if use_state_dict:
            # Extract GAN parameters from state_dict
            for k, v in model_state_dict.items():
                if k.startswith("gan_heads.") or k.startswith("gan_final_head."):
                    state_dict[k] = v.detach().clone().cpu() if torch.is_tensor(v) else v
        else:
            # Extract from model
            if hasattr(model, "gan_heads"):
                for hook_name, gan_head in model.gan_heads.items():
                    for k, v in gan_head.state_dict().items():
                        state_dict[f"gan_heads.{hook_name}.{k}"] = v.detach().clone().cpu()

            if hasattr(model, "gan_final_head"):
                for k, v in model.gan_final_head.state_dict().items():
                    state_dict[f"gan_final_head.{k}"] = v.detach().clone().cpu()

    # 5. Save evolving-memory components (full-rank, outside the PEFT adapter).
    # Gated on ENABLE (not train) so Stage-C frozen variants persist too
    # (design ch.2 D12). The rolling query_state is a plain attribute and never
    # appears in any state_dict by construction.
    if getattr(args.training_config, "is_enable_evolving_memory", False):
        if use_state_dict:
            for k, v in model_state_dict.items():
                if k.startswith("evolving_memory.") or "memory_key_scale" in k:
                    state_dict[k] = v.detach().clone().cpu() if torch.is_tensor(v) else v
        else:
            if hasattr(model, "evolving_memory"):
                for k, v in model.evolving_memory.state_dict().items():
                    state_dict[f"evolving_memory.{k}"] = v.detach().clone().cpu()
            for block_idx, block in enumerate(model.blocks):
                if hasattr(block.attn1, "memory_key_scale"):
                    state_dict[f"blocks.{block_idx}.attn1.memory_key_scale"] = (
                        block.attn1.memory_key_scale.detach().clone().cpu()
                    )

    torch.save(state_dict, os.path.join(output_dir, "transformer_partial.pth"))
    print(f"Saved checkpoint with {len(state_dict)} parameters to {output_dir}/transformer_partial.pth")


def load_extra_components(args, model, checkpoint_path):
    """
    Load patch_short, patch_mid, patch_long, q_loras, k_loras, v_loras into the model
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    loaded_keys = set()

    # Load patch modules (formerly multi_term_memory_patchg)
    if args.training_config.is_enable_stage1:
        patch_names = ["patch_short", "patch_mid", "patch_long"]

        for p_name in patch_names:
            patch_keys_in_sd = [k for k in state_dict.keys() if k.startswith(f"{p_name}.")]
            if patch_keys_in_sd and hasattr(model, p_name):
                patch_state = {
                    k.replace(f"{p_name}.", ""): v for k, v in state_dict.items() if k.startswith(f"{p_name}.")
                }
                patch_module = getattr(model, p_name)
                load_info = patch_module.load_state_dict(patch_state, strict=False)
                loaded_keys.update(patch_keys_in_sd)

                print(f"Loaded {len(patch_keys_in_sd)} parameters for {p_name}")
                if load_info.missing_keys:
                    print(f"  Missing keys in {p_name}: {load_info.missing_keys}")
                if load_info.unexpected_keys:
                    print(f"  Unexpected keys in {p_name}: {load_info.unexpected_keys}")

        # Load fully fine-tuned patch_embedding if present in the checkpoint.
        pe_keys = [k for k in state_dict.keys() if k.startswith("patch_embedding.")]
        if pe_keys and hasattr(model, "patch_embedding"):
            pe_state = {k.replace("patch_embedding.", ""): v for k, v in state_dict.items() if k.startswith("patch_embedding.")}
            info = model.patch_embedding.load_state_dict(pe_state, strict=False)
            loaded_keys.update(pe_keys)
            print(f"Loaded {len(pe_keys)} parameters for patch_embedding")
            if info.unexpected_keys:
                print(f"  Unexpected keys in patch_embedding: {info.unexpected_keys}")

    # Load LoRA layers
    lora_keys_count = 0
    if args.training_config.restrict_self_attn:
        for block_idx, block in enumerate(model.blocks):
            # Load q_loras
            q_lora_keys_in_sd = [k for k in state_dict.keys() if k.startswith(f"blocks.{block_idx}.attn1.q_loras.")]
            if q_lora_keys_in_sd:
                q_lora_state = {
                    k.replace(f"blocks.{block_idx}.attn1.q_loras.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith(f"blocks.{block_idx}.attn1.q_loras.")
                }
                load_info = block.attn1.q_loras.load_state_dict(q_lora_state, strict=False)
                loaded_keys.update(q_lora_keys_in_sd)
                lora_keys_count += len(q_lora_keys_in_sd)
                if load_info.missing_keys:
                    print(f"  Missing keys in blocks.{block_idx}.attn1.q_loras: {load_info.missing_keys}")
                if load_info.unexpected_keys:
                    print(f"  Unexpected keys in blocks.{block_idx}.attn1.q_loras: {load_info.unexpected_keys}")

            # Load k_loras
            k_lora_keys_in_sd = [k for k in state_dict.keys() if k.startswith(f"blocks.{block_idx}.attn1.k_loras.")]
            if k_lora_keys_in_sd:
                k_lora_state = {
                    k.replace(f"blocks.{block_idx}.attn1.k_loras.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith(f"blocks.{block_idx}.attn1.k_loras.")
                }
                load_info = block.attn1.k_loras.load_state_dict(k_lora_state, strict=False)
                loaded_keys.update(k_lora_keys_in_sd)
                lora_keys_count += len(k_lora_keys_in_sd)
                if load_info.missing_keys:
                    print(f"  Missing keys in blocks.{block_idx}.attn1.k_loras: {load_info.missing_keys}")
                if load_info.unexpected_keys:
                    print(f"  Unexpected keys in blocks.{block_idx}.attn1.k_loras: {load_info.unexpected_keys}")

            # Load v_loras
            v_lora_keys_in_sd = [k for k in state_dict.keys() if k.startswith(f"blocks.{block_idx}.attn1.v_loras.")]
            if v_lora_keys_in_sd:
                v_lora_state = {
                    k.replace(f"blocks.{block_idx}.attn1.v_loras.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith(f"blocks.{block_idx}.attn1.v_loras.")
                }
                load_info = block.attn1.v_loras.load_state_dict(v_lora_state, strict=False)
                loaded_keys.update(v_lora_keys_in_sd)
                lora_keys_count += len(v_lora_keys_in_sd)
                if load_info.missing_keys:
                    print(f"  Missing keys in blocks.{block_idx}.attn1.v_loras: {load_info.missing_keys}")
                if load_info.unexpected_keys:
                    print(f"  Unexpected keys in blocks.{block_idx}.attn1.v_loras: {load_info.unexpected_keys}")

        print(f"Loaded {lora_keys_count} parameters for Restrict Self Attn LoRA")

    # Load History Scale layers
    history_keys_count = 0
    if args.training_config.is_amplify_history:
        for block_idx, block in enumerate(model.blocks):
            history_key_scale_key = f"blocks.{block_idx}.attn1.history_key_scale"
            if history_key_scale_key in state_dict:
                block.attn1.history_key_scale.data = state_dict[history_key_scale_key].to(
                    block.attn1.history_key_scale.device
                )
                loaded_keys.add(history_key_scale_key)
                history_keys_count += 1

        print(f"Loaded {history_keys_count} parameters for History Scale")

    # Load evolving-memory components (design ch.2 D12: fail loudly when the
    # checkpoint carries memory keys but the model was built without the module
    # — silent skipping would look like a successful lineage handoff).
    memory_keys = [k for k in state_dict.keys() if k.startswith("evolving_memory.")]
    if memory_keys:
        assert hasattr(model, "evolving_memory"), (
            "checkpoint contains evolving_memory.* keys but the model was constructed "
            "without is_enable_evolving_memory"
        )
        memory_state = {
            k.replace("evolving_memory.", "", 1): v for k, v in state_dict.items() if k.startswith("evolving_memory.")
        }
        model.evolving_memory.load_state_dict(memory_state, strict=True)
        loaded_keys.update(memory_keys)
        print(f"Loaded {len(memory_keys)} parameters for evolving_memory")
    memory_scale_count = 0
    for block_idx, block in enumerate(model.blocks):
        memory_scale_key = f"blocks.{block_idx}.attn1.memory_key_scale"
        if memory_scale_key in state_dict:
            assert hasattr(block.attn1, "memory_key_scale"), (
                f"checkpoint contains {memory_scale_key} but the block was constructed without is_amplify_memory"
            )
            block.attn1.memory_key_scale.data = state_dict[memory_scale_key].to(
                block.attn1.memory_key_scale.device
            )
            loaded_keys.add(memory_scale_key)
            memory_scale_count += 1
    if memory_scale_count:
        print(f"Loaded {memory_scale_count} parameters for Memory Scale")

    # Load GAN
    gan_keys_count = 0
    if args.training_config.is_use_gan:
        # Load intermediate gan_heads
        if hasattr(model, "gan_heads"):
            for hook_name, gan_head in model.gan_heads.items():
                gan_head_prefix = f"gan_heads.{hook_name}."
                gan_head_keys_in_sd = [k for k in state_dict.keys() if k.startswith(gan_head_prefix)]

                if gan_head_keys_in_sd:
                    gan_head_state = {
                        k.replace(gan_head_prefix, ""): v
                        for k, v in state_dict.items()
                        if k.startswith(gan_head_prefix)
                    }
                    load_info = gan_head.load_state_dict(gan_head_state, strict=False)
                    loaded_keys.update(gan_head_keys_in_sd)
                    gan_keys_count += len(gan_head_keys_in_sd)
                    if load_info.missing_keys:
                        print(f"  Missing keys in gan_heads.{hook_name}: {load_info.missing_keys}")
                    if load_info.unexpected_keys:
                        print(f"  Unexpected keys in gan_heads.{hook_name}: {load_info.unexpected_keys}")

        # Load final gan head
        if hasattr(model, "gan_final_head"):
            gan_final_keys_in_sd = [k for k in state_dict.keys() if k.startswith("gan_final_head.")]

            if gan_final_keys_in_sd:
                gan_final_state = {
                    k.replace("gan_final_head.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("gan_final_head.")
                }
                load_info = model.gan_final_head.load_state_dict(gan_final_state, strict=False)
                loaded_keys.update(gan_final_keys_in_sd)
                gan_keys_count += len(gan_final_keys_in_sd)
                if load_info.missing_keys:
                    print(f"  Missing keys in gan_final_head: {load_info.missing_keys}")
                if load_info.unexpected_keys:
                    print(f"  Unexpected keys in gan_final_head: {load_info.unexpected_keys}")

        if gan_keys_count > 0:
            print(f"Loaded {gan_keys_count} parameters for GAN components")

    if not loaded_keys:
        print("No extra components were loaded from the checkpoint.")
        return

    all_sd_keys = set(state_dict.keys())
    unmatched_keys = all_sd_keys - loaded_keys

    print("\nCheckpoint loading completed.")
    print(f"Total loaded keys: {len(loaded_keys)}")
    if unmatched_keys:
        print(f"The following keys in the checkpoint were not loaded into the model: {sorted(unmatched_keys)}\n")
    else:
        print("Load extra module successfully! All keys in the checkpoint were successfully processed or matched.\n")


def save_model_checkpoint(
    transformer,
    args,
    save_path,
    weight_dtype=None,
    unwrap_model_fn=None,
    get_peft_model_state_dict_fn=None,
    collate_lora_metadata_fn=None,
    save_extra_components_fn=None,
    pipeline_class=None,
    norm_layer_prefixes=None,
):
    modules_to_save = {}
    model_to_save = unwrap_model_fn(transformer) if unwrap_model_fn else transformer

    transformer_lora_layers = get_peft_model_state_dict_fn(model_to_save)

    if args.model_config.train_norm_layers:
        norm_prefixes = norm_layer_prefixes or []
        transformer_norm_layers = {
            f"transformer.{name}": param
            for name, param in model_to_save.named_parameters()
            if any(k in name for k in norm_prefixes)
        }
        transformer_lora_layers = {
            **transformer_lora_layers,
            **transformer_norm_layers,
        }

    modules_to_save["transformer"] = model_to_save

    if pipeline_class and hasattr(pipeline_class, "save_lora_weights"):
        lora_metadata = collate_lora_metadata_fn(modules_to_save) if collate_lora_metadata_fn else {}
        pipeline_class.save_lora_weights(
            save_directory=save_path,
            transformer_lora_layers=transformer_lora_layers,
            **lora_metadata,
        )

    if save_extra_components_fn:
        save_extra_components_fn(args=args, model=model_to_save, output_dir=save_path)

    modules_to_save = None
    lora_metadata = None
    transformer_norm_layers = None
    transformer_lora_layers = None
    del modules_to_save
    del lora_metadata
    del transformer_norm_layers
    del transformer_lora_layers


def load_model_checkpoint(
    args,
    checkpoint_path,
    transformer,
    pipeline_class=None,
    norm_layer_prefixes=None,
    convert_unet_state_dict_to_peft_fn=None,
    set_peft_model_state_dict_fn=None,
    cast_training_params_fn=None,
):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    lora_state_dict = None
    if pipeline_class and hasattr(pipeline_class, "load_lora_weights"):
        lora_state_dict = pipeline_class.lora_state_dict(checkpoint_path)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft_fn(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict_fn(transformer, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        print(f"load lora from {checkpoint_path} successfully!")

    if args.model_config.train_norm_layers and lora_state_dict and norm_layer_prefixes:
        transformer_norm_state_dict = {
            k: v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.") and any(norm_k in k for norm_k in norm_layer_prefixes)
        }
        transformer._transformer_norm_layers = pipeline_class._load_norm_into_transformer(
            transformer_norm_state_dict,
            transformer=transformer,
            discard_original_layers=False,
        )

    load_extra_components(args, transformer, os.path.join(checkpoint_path, "transformer_partial.pth"))

    if args.training_config.mixed_precision != "fp32":
        models = [transformer]
        cast_training_params_fn(models)


# ======================================== sigmas & timesteps ========================================
def get_sigmas(noise_scheduler, timesteps, n_dim=4, device="cuda", dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def apply_schedule_shift(
    sigmas,
    noise,
    sigmas_two=None,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    exp_max: float = 7.0,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    mu: float = None,
    return_mu: bool = False,
):
    if mu is None:
        # Resolution-dependent shifting of timestep schedules as per section 5.3.2 of SD3 paper
        image_seq_len = (noise.shape[-1] * noise.shape[-2] * noise.shape[-3]) // 4  # patch size 1,2,2
        mu = calculate_shift(
            image_seq_len,
            base_seq_len if base_seq_len is not None else 256,
            max_seq_len if max_seq_len is not None else 4096,
            base_shift if base_shift is not None else 0.5,
            max_shift if max_shift is not None else 1.15,
        )
        if time_shift_type == "exponential":
            mu = min(mu, math.log(exp_max))
            mu = math.exp(mu)

    if sigmas_two is not None:
        sigmas = (sigmas * mu) / (1 + (mu - 1) * sigmas)
        sigmas_two = (sigmas_two * mu) / (1 + (mu - 1) * sigmas_two)
        if return_mu:
            return sigmas, sigmas_two, mu
        else:
            return sigmas, sigmas_two
    else:
        sigmas = (sigmas * mu) / (1 + (mu - 1) * sigmas)
        if return_mu:
            return sigmas, mu
        else:
            return sigmas


# ======================================== clean prompt ========================================


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def _get_t5_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]] = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    caption_dropout_p: float = 0.0,
    device: Optional[torch.device] = "cuda",
    dtype: Optional[torch.dtype] = torch.bfloat16,
):
    device = device
    dtype = dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    if random.random() < caption_dropout_p:
        prompt_embeds.fill_(0)
        mask.fill_(False)
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds, text_inputs.attention_mask


def encode_prompt(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 512,
    caption_dropout_p: float = 0.0,
    device: Optional[torch.device] = "cuda",
    dtype: Optional[torch.dtype] = torch.bfloat16,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_embeds is None:
        prompt_embeds, prompt_attention_mask = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            caption_dropout_p=caption_dropout_p,
            device=device,
            dtype=dtype,
        )

    return prompt_embeds, prompt_attention_mask


# ======================================== other techniques ========================================


class AdaptiveAntiDrifting:
    def __init__(
        self,
        rho_mu: float = 0.9,
        rho_sigma: float = 0.9,
        delta_mu: float = 0.15,
        delta_sigma: float = 0.15,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            rho_mu: EMA coefficient for mean (momentum parameter)
            rho_sigma: EMA coefficient for variance (momentum parameter)
            delta_mu: Threshold for mean drift detection
            delta_sigma: Threshold for variance drift detection
            device: Device for tensor operations
            dtype: Data type for tensors
        """
        self.rho_mu = rho_mu
        self.rho_sigma = rho_sigma
        self.delta_mu = delta_mu
        self.delta_sigma = delta_sigma
        self.device = device
        self.dtype = dtype

        # Global statistics (initialized on first chunk)
        self.global_mean = None
        self.global_var = None
        self.is_initialized = False

    def compute_latent_statistics(self, latent_chunk: torch.Tensor) -> tuple:
        # Shape: (B, C, T, H, W) -> (B, C)
        mean = latent_chunk.mean(dim=[2, 3, 4])
        var = latent_chunk.var(dim=[2, 3, 4])

        return mean, var

    def update_global_statistics(self, current_mean: torch.Tensor, current_var: torch.Tensor):
        if not self.is_initialized:
            self.global_mean = current_mean.clone()
            self.global_var = current_var.clone()
            self.is_initialized = True
        else:
            self.global_mean = self.rho_mu * self.global_mean + (1 - self.rho_mu) * current_mean
            self.global_var = self.rho_sigma * self.global_var + (1 - self.rho_sigma) * current_var

    def detect_drift(self, current_mean: torch.Tensor, current_var: torch.Tensor) -> bool:
        if not self.is_initialized:
            return False

        mean_drift = torch.norm(current_mean - self.global_mean, p=2, dim=-1).mean().item()
        var_drift = torch.norm(current_var - self.global_var, p=2, dim=-1).mean().item()

        has_drift = (mean_drift > self.delta_mu) and (var_drift > self.delta_sigma)

        return has_drift

    def apply_frame_aware_corruption(
        self,
        history_latents: torch.Tensor,
        corruption_strength: float = 0.1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(history_latents, generator=generator, device=history_latents.device)
        corrupted_latents = history_latents + corruption_strength * noise

        return corrupted_latents

    def reset(self):
        self.global_mean = None
        self.global_var = None
        self.is_initialized = False


class HistoryProjector:
    """Training-free projection of the per-section history latents, applied
    after the [long|mid|short] tiers are split and before they are patchified.

    Long rollouts accumulate a slow bias in the history latents (visible as an
    over-saturation runaway). Unlike AdaptiveAntiDrifting, which compares a
    chunk against an EMA that drifts along with it and then injects noise into
    the generated latents, these projections map the *conditioning* history
    back onto a fixed support every section, so a sub-threshold per-section
    increment cannot compound.

    Modes:
        "none"      no-op.
        "quantize"  uniform scalar grid, ``round(z / step) * step``. The
                    dead zone of ``step / 2`` erases small increments.
        "codebook"  FramePack history discretization (arXiv 2504.12626 v3,
                    Eq. 6): every latent pixel is replaced by the nearest
                    centroid of a K-means codebook fitted on real latents.
                    Note the paper applies this during training; using it on a
                    checkpoint trained with continuous history is an
                    inference-only variant.
        "jitter"    TRAINING-ONLY augmentation: a RANDOM per-channel affine of
                    magnitude `jitter`. Inference-time renorm applies a slightly
                    different deterministic transform each section, and the model
                    reads that statistics change as content, which steps the
                    output at every chunk boundary (measured 2026-07-30: the step
                    grows with alpha). Training on random rescalings teaches the
                    model that history statistics carry no signal, which should
                    remove the step. It corrects nothing by itself, so it is not
                    offered as an inference mode.
        "renorm"    per-channel affine match onto a frozen reference: the
                    statistics of the first fully populated history window are
                    recorded once, and every later history is mapped back onto
                    them.

    Every mode blends with the untouched history through ``alpha`` (0 = no-op,
    1 = full projection). Padding frames are passed through untouched, which
    keeps them out of the statistics and out of the codebook lookup (a centroid
    is a real latent vector, so projecting padding would inject content into a
    history window that is still empty). Padding is detected structurally: the
    pipeline builds it with ``torch.zeros``, so a frame counts as padding only
    when every element is within ``padding_tol`` of zero. A statistical test
    would also catch legitimate near-constant frames and skip them silently.
    The caller is responsible for excluding the fixed x0 anchor.
    """

    MODES = ("none", "quantize", "codebook", "renorm", "jitter")

    def __init__(
        self,
        mode: str = "none",
        step: float = 0.0,
        codebook: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        padding_tol: float = 0.0,
        smooth: float = 0.0,
        jitter: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"unknown history projection mode {mode!r}, expected one of {self.MODES}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"history projection alpha must be in [0, 1], got {alpha}")
        if mode == "quantize" and not step > 0.0:
            raise ValueError(f"quantize mode needs a positive step, got {step}")
        if mode == "codebook":
            if codebook is None:
                raise ValueError("codebook mode needs a codebook tensor")
            if codebook.ndim != 2:
                raise ValueError(f"codebook must be 2-D (K, C), got shape {tuple(codebook.shape)}")
        if padding_tol < 0.0:
            raise ValueError(f"padding_tol must be non-negative, got {padding_tol}")
        if not 0.0 <= smooth < 1.0:
            raise ValueError(f"smooth must be in [0, 1), got {smooth}")
        if mode == "jitter" and not jitter > 0.0:
            raise ValueError(f"jitter mode needs a positive jitter magnitude, got {jitter}")
        if jitter < 0.0:
            raise ValueError(f"jitter must be non-negative, got {jitter}")

        self.mode = mode
        self.step = float(step)
        self.codebook = codebook
        self.alpha = float(alpha)
        self.padding_tol = float(padding_tol)
        # Read-time renorm recomputes its transform from the sliding window every
        # section, so the SAME past frame is presented to the model under a
        # slightly different transform on consecutive steps. That shows up as a
        # step at every chunk boundary (a human reported a flicker at the 1.375s
        # chunk cadence, 2026-07-30). Smoothing the transform across sections
        # damps the step; the write-time site removes the cause instead.
        self.smooth = float(smooth)
        self.previous_transform = None
        self.jitter = float(jitter)
        self.generator = generator

        # Frozen per-channel reference for "renorm", set on the first fully
        # populated history window.
        self.reference = None
        self.num_calls = 0
        self.num_applied = 0
        self.reference_call = None
        # Effect size: a projection that fires without changing a single value
        # is indistinguishable from no projection in the artifacts.
        self.num_changed = 0
        self.max_abs_delta = 0.0
        self.num_frames_seen = 0
        self.num_frames_padding = 0

    @property
    def enabled(self) -> bool:
        return self.mode != "none" and self.alpha > 0.0

    def stats(self) -> dict:
        """Counters for the run manifest. num_applied only proves the branch
        ran; num_changed and max_abs_delta prove it changed the history."""
        return {
            "mode": self.mode,
            "step": self.step,
            "alpha": self.alpha,
            "padding_tol": self.padding_tol,
            "smooth": self.smooth,
            "jitter": self.jitter,
            "codebook_size": None if self.codebook is None else int(self.codebook.shape[0]),
            "num_calls": self.num_calls,
            "num_applied": self.num_applied,
            "reference_call": self.reference_call,
            "num_changed": self.num_changed,
            "max_abs_delta": self.max_abs_delta,
            "num_frames_seen": self.num_frames_seen,
            "num_frames_padding": self.num_frames_padding,
        }

    def __call__(self, tiers: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project a section's history tiers, each shaped (B, C, T, H, W).

        Returns the tiers in the same order; tensors are returned unchanged
        when the projector is disabled or the history is still all padding.
        """
        if not self.enabled:
            return list(tiers)

        self.num_calls += 1
        masks = [t.abs().amax(dim=(1, 3, 4)) > self.padding_tol for t in tiers]
        for mask in masks:
            self.num_frames_seen += int(mask.numel())
            self.num_frames_padding += int((~mask).sum())
        if not any(bool(mask.any()) for mask in masks):
            return list(tiers)

        if self.mode == "renorm":
            projected = self._renorm(tiers, masks)
        elif self.mode == "jitter":
            projected = self._jitter(tiers, masks)
        else:
            projected = [self._blend(t, self._pointwise(t), mask) for t, mask in zip(tiers, masks)]
        if projected is None:
            return list(tiers)

        for original, result in zip(tiers, projected):
            delta = (result - original).abs()
            self.num_changed += int((delta > 0).sum())
            self.max_abs_delta = max(self.max_abs_delta, float(delta.max()))

        self.num_applied += 1
        return projected

    def _blend(self, latents: torch.Tensor, projected: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # At full strength the result must be the projection exactly: the point
        # of the codebook mode is that the history lands on a finite support,
        # and latents + 1.0 * (projected - latents) only rounds back to it.
        blended = projected if self.alpha == 1.0 else latents + self.alpha * (projected - latents)
        keep = mask.view(mask.shape[0], 1, mask.shape[1], 1, 1)
        return torch.where(keep, blended, latents)

    def _pointwise(self, latents: torch.Tensor) -> torch.Tensor:
        if self.mode == "quantize":
            return torch.round(latents / self.step) * self.step
        return self._nearest_centroid(latents)

    def _nearest_centroid(self, latents: torch.Tensor, chunk_size: int = 16384) -> torch.Tensor:
        book = self.codebook.to(device=latents.device, dtype=latents.dtype)
        b, c, f, h, w = latents.shape
        flat = latents.permute(0, 2, 3, 4, 1).reshape(-1, c)
        out = torch.empty_like(flat)
        # ||x - m||^2 = ||x||^2 - 2 x.m + ||m||^2; the ||x||^2 term is constant
        # per pixel and does not change the argmin.
        book_sq = (book * book).sum(dim=1)
        for start in range(0, flat.shape[0], chunk_size):
            block = flat[start : start + chunk_size]
            index = (book_sq - 2.0 * (block @ book.t())).argmin(dim=1)
            out[start : start + chunk_size] = book[index]
        return out.view(b, f, h, w, c).permute(0, 4, 1, 2, 3)

    def _jitter(self, tiers: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        """One random per-channel affine, shared across the section's tiers so the
        history stays internally consistent: the point is to rescale the whole
        window's statistics, not to decorrelate its frames."""
        reference = tiers[0]
        channels = reference.shape[1]
        noise = torch.randn(
            2, channels, generator=self.generator, device=reference.device, dtype=reference.dtype
        )
        scale = torch.exp(noise[0] * self.jitter)
        shift = noise[1] * self.jitter
        out = []
        for latents, mask in zip(tiers, masks):
            projected = latents * scale.view(1, -1, 1, 1, 1) + shift.view(1, -1, 1, 1, 1)
            out.append(self._blend(latents, projected, mask))
        return out

    def _renorm(self, tiers: List[torch.Tensor], masks: List[torch.Tensor]) -> Optional[List[torch.Tensor]]:
        values = self._channel_values(tiers, masks)
        mean = values.mean(dim=1)
        std = values.std(dim=1)

        if self.reference is None:
            # Freeze the reference on the first history window that carries no
            # zero padding, so the reference describes real generated content.
            if not all(bool(mask.all()) for mask in masks):
                return None
            self.reference = (mean.detach().clone(), std.detach().clone())
            self.reference_call = self.num_calls
            return None

        ref_mean, ref_std = self.reference
        ref_mean = ref_mean.to(device=mean.device, dtype=mean.dtype)
        ref_std = ref_std.to(device=std.device, dtype=std.dtype)
        scale = ref_std / std.clamp_min(1e-4)
        shift = ref_mean - mean * scale
        if self.smooth > 0.0 and self.previous_transform is not None:
            previous_scale, previous_shift = self.previous_transform
            scale = self.smooth * previous_scale.to(scale.device) + (1.0 - self.smooth) * scale
            shift = self.smooth * previous_shift.to(shift.device) + (1.0 - self.smooth) * shift
        self.previous_transform = (scale.detach().clone(), shift.detach().clone())

        out = []
        for latents, mask in zip(tiers, masks):
            projected = latents * scale.view(1, -1, 1, 1, 1) + shift.view(1, -1, 1, 1, 1)
            out.append(self._blend(latents, projected, mask))
        return out

    @staticmethod
    def _channel_values(tiers: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor:
        """Per-channel values of every valid history frame, shaped (C, N)."""
        columns = []
        for latents, mask in zip(tiers, masks):
            for b in range(latents.shape[0]):
                selected = latents[b][:, mask[b]]
                if selected.shape[1] == 0:
                    continue
                columns.append(selected.reshape(selected.shape[0], -1))
        return torch.cat(columns, dim=1)


def load_history_codebook(path: str) -> torch.Tensor:
    """Load a (K, C) K-means codebook saved by scripts/evaluation/fit_history_codebook.py."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    codebook = payload["codebook"] if isinstance(payload, dict) else payload
    if not torch.is_tensor(codebook) or codebook.ndim != 2:
        raise ValueError(f"{path} does not contain a 2-D (K, C) codebook")
    return codebook.float()


def build_transformer_param_groups(transformer, base_lr, memory_lr):
    """Split trainable transformer params into base vs evolving-memory groups
    (design ch.2 D11). Memory params (full-rank encoder + per-block
    memory_key_scale) get their own lr; each group carries a 'role' tag that
    the HELIOS_FORCE_LR resume path uses to restore per-group lrs instead of
    flattening them."""
    base_params, memory_params = [], []
    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            continue
        if "evolving_memory" in name or "memory_key_scale" in name:
            memory_params.append(param)
        else:
            base_params.append(param)
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr, "role": "base"})
    if memory_params:
        groups.append({"params": memory_params, "lr": memory_lr, "role": "memory"})
    assert groups, "no trainable parameters found for the optimizer"
    return groups
