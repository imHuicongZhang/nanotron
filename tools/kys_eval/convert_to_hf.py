#!/usr/bin/env python
"""
Standalone Nanotron(LLaMA, tp=1/pp=1) -> HuggingFace LlamaForCausalLM converter.

Reads the checkpoint's sharded safetensors directly (each leaf holds a single `data`
tensor) and maps them to HF parameter names. Needs ONLY torch + safetensors +
transformers -- no nanotron, no flash-attn, no torchrun, no GPU (runs on CPU).

Why this is correct (matches nanotron's official examples/llama/convert_nanotron_to_hf.py):
  * qkv_proj is a merged [ (n_q+n_kv+n_kv)*d_head , hidden ] tensor with Q heads first,
    then K, then V. With rope_interleaved=False (our config) the official converter takes
    the RAW slices (interleave_qkv=False) -> q/k/v map directly to HF q_proj/k_proj/v_proj
    with no permutation. Nanotron non-interleaved RoPE == HF GPT-NeoX rotary layout.
  * gate_up_proj is a merged [ 2*intermediate , hidden ] tensor: gate first, then up.
  * down_proj / o_proj / layernorms / embedding / lm_head map 1:1.
  * tie_word_embeddings=False -> lm_head is stored separately and copied as-is.

Usage:
  python tools/kys_eval/convert_to_hf.py --checkpoint_path <step_dir> --save_path <hf_out> \
      --tokenizer /path/to/llama2-tokenizer [--dtype bfloat16]
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def _load(path: Path) -> torch.Tensor:
    return load_file(str(path))["data"]


def _find(root: Path, *parts: str) -> Path:
    """Find the single safetensors file under root/parts (filename varies by shard tag)."""
    d = root.joinpath(*parts)
    files = sorted(d.glob("*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected exactly one safetensors in {d}, found {files}")
    return files[0]


def convert(checkpoint_path: Path, save_path: Path, tokenizer_name: str, dtype: torch.dtype):
    with open(checkpoint_path / "model_config.json") as f:
        nt = json.load(f)

    hidden = nt["hidden_size"]
    n_heads = nt["num_attention_heads"]
    n_kv = nt["num_key_value_heads"]
    d_head = hidden // n_heads
    inter = nt["intermediate_size"]
    n_layers = nt["num_hidden_layers"]

    q_size = n_heads * d_head
    k_size = n_kv * d_head  # == v_size

    mm = checkpoint_path / "model" / "model"
    sd = {}

    tie = nt.get("tie_word_embeddings", False)

    # Embeddings + final norm + lm_head.
    # When tie_word_embeddings is True (the rewrite settings), nanotron stores NO separate
    # lm_head — the output projection reuses the input embedding. We skip lm_head here and
    # re-tie it to embed_tokens after loading. When False (e.g. the pilot baselines) lm_head
    # is a distinct safetensors leaf and is loaded directly.
    sd["model.embed_tokens.weight"] = _load(_find(mm, "token_position_embeddings", "pp_block", "token_embedding"))
    sd["model.norm.weight"] = _load(_find(mm, "final_layer_norm", "pp_block"))
    if not tie:
        sd["lm_head.weight"] = _load(_find(mm, "lm_head", "pp_block"))

    for i in range(n_layers):
        blk = ("decoder", str(i), "pp_block")
        qkv = _load(_find(mm, *blk, "attn", "qkv_proj"))  # [q_size+2*k_size, hidden]
        q = qkv[0:q_size]
        k = qkv[q_size : q_size + k_size]
        v = qkv[q_size + k_size : q_size + 2 * k_size]
        gate_up = _load(_find(mm, *blk, "mlp", "gate_up_proj"))  # [2*inter, hidden]
        gate = gate_up[0:inter]
        up = gate_up[inter : 2 * inter]

        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = q
        sd[p + "self_attn.k_proj.weight"] = k
        sd[p + "self_attn.v_proj.weight"] = v
        sd[p + "self_attn.o_proj.weight"] = _load(_find(mm, *blk, "attn", "o_proj"))
        sd[p + "mlp.gate_proj.weight"] = gate
        sd[p + "mlp.up_proj.weight"] = up
        sd[p + "mlp.down_proj.weight"] = _load(_find(mm, *blk, "mlp", "down_proj"))
        sd[p + "input_layernorm.weight"] = _load(_find(mm, *blk, "input_layernorm"))
        sd[p + "post_attention_layernorm.weight"] = _load(_find(mm, *blk, "post_attention_layernorm"))

    sd = {key: val.to(dtype).contiguous() for key, val in sd.items()}

    cfg = LlamaConfig(
        vocab_size=nt["vocab_size"],
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        hidden_act=nt["hidden_act"],
        max_position_embeddings=nt["max_position_embeddings"],
        rms_norm_eps=nt["rms_norm_eps"],
        rope_theta=nt["rope_theta"],
        attention_bias=nt.get("attention_bias", False),
        tie_word_embeddings=nt.get("tie_word_embeddings", False),
        bos_token_id=nt.get("bos_token_id", 1),
        eos_token_id=nt.get("eos_token_id", 2),
        pad_token_id=nt.get("pad_token_id", None),
        rope_scaling=nt.get("rope_scaling", None),
        torch_dtype=str(dtype).split(".")[-1],
    )

    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if tie:
        # assign=True replaced embed_tokens.weight with a fresh tensor, breaking the
        # meta-init tie; re-point lm_head.weight back at the real input embedding.
        model.tie_weights()
    allowed_missing = {"lm_head.weight"} if tie else set()
    assert not (set(missing) - allowed_missing) and not unexpected, f"missing={missing} unexpected={unexpected}"
    model = model.to(dtype)
    if tie:
        model.tie_weights()

    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=True)
    AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(save_path)
    print(f"[convert_to_hf] saved {n_layers}-layer model ({sum(p.numel() for p in sd.values())/1e9:.3f}B params) -> {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", type=Path, required=True)
    ap.add_argument("--save_path", type=Path, required=True)
    ap.add_argument("--tokenizer", type=str, required=True)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    args = ap.parse_args()
    convert(args.checkpoint_path, args.save_path, args.tokenizer, getattr(torch, args.dtype))
