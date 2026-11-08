---
license: apache-2.0
tags:
  - kys
  - know-your-sources
  - nanotron
  - reference-checkpoint
---

# KYS-1.5B-Rewritten-v2

Checkpoints from the **v2 evaluation** of the Know-Your-Sources rewritten arm: 1.5B Llama models
pretrained from scratch on a 10B-token mixture, repeated for three epochs on the
**branched schedule**, with epoch boundaries at steps **4768 / 9537 / 14305**.

## Reference checkpoint

`rewrite-1p5b/seed42/diversity_oriented/ep3/` is the **reference checkpoint of
[kys-eval](https://github.com/imHuicongZhang/kys-eval)**: the checkpoint an external operator
re-scores to confirm their pipeline reproduces ours before scoring new models.

```bash
python -m kys_eval.reference_check \
    --model hf://wytro/KYS-1.5B-Rewritten-v2/rewrite-1p5b/seed42/diversity_oriented/ep3/hf
```

Expected scores (`acc_norm`, percent, 0-shot, full splits, NVIDIA H100 80GB HBM3):

| ARC-e | HellaSwag | PIQA | SIQA | OBQA | CSQA | **Mean6** | **MMLU** |
|---|---|---|---|---|---|---|---|
| 51.52 | 49.82 | 72.03 | 39.10 | 35.20 | 31.45 | **46.52** | **30.97** |

`model.safetensors` sha256: `733262aec4ca52fe94b69df0acf7ef1faa7cd0fc1f42ee79b91004a022db3f45`

## Not the paper-era checkpoint

These are **not** the checkpoints behind the numbers printed in the paper's tables. The paper-era
models sit at steps 4770 / 9540 / 14305, and at **14147** for Diversity Oriented; they are released
separately as `blab-jhu/KYS-1.5B-*`. The v2 checkpoints here were trained on the branched schedule
and re-scored; per-cell scores differ from the paper's by up to about 1 accuracy point. Use this
repo for kys-eval comparisons, and the `blab-jhu/KYS-1.5B-*` repos to reproduce the paper's tables.

## Layout

```
rewrite-1p5b/seed<S>/<setting>/ep<N>/hf/         HuggingFace export (LlamaForCausalLM, bf16)
rewrite-1p5b/seed<S>/<setting>/ep<N>/nanotron/   Nanotron checkpoint directory, as written by training
```

`hf/` holds `config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json` and
`tokenizer_config.json`. `nanotron/` holds `model/`, `optimizer/`, `lr_scheduler/`, `random/`,
`config.yaml`, `model_config.json` and `checkpoint_metadata.json`.

Settings are `quality_base`, `quality_first`, `diversity_oriented`, `disagreement_aware`,
`wrap_inspired`, `rewire_inspired`; seeds are 42, 43, 44; epochs `ep1` (step 4768), `ep2` (9537)
and `ep3` (14305).

## Architecture

Llama 2 architecture: 28 layers, hidden size 2048, FFN 5632 (SwiGLU), 16 heads, RoPE theta 10^4,
RMSNorm, vocabulary 32000 (Llama 2 tokenizer), tied embeddings, 1.504B parameters, bf16.
