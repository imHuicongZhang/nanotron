# `tools/kys7b/` — the 7B forks

These three scripts are forks of the 1.5B tooling in `tools/`, for the 7B grid in
`configs/know-your-sources-7b/` deployed from `deploy/clusters_7b.yaml`.

The file names are deliberately identical to `tools/`, so every command in `SOP.md` changes
only by path:

```bash
python tools/kys7b/generate_configs.py  --out configs/know-your-sources-7b
python tools/kys7b/render_config.py     --template configs/know-your-sources-7b/<x>.yaml --cluster b300 --seed 42 --out rendered/<x>.yaml
python tools/kys7b/assert_invariants.py --config rendered/<x>.yaml --check-resume
```

**The 1.5B files are untouched.** `tools/*.py`, `deploy/clusters.yaml`,
`configs/know-your-sources/` and the existing markdown still describe the 1.5B grid exactly as
it was run. Nothing here modifies them.

Every guard from the 1.5B tooling is preserved verbatim: accum is derived as
`global_batch_seq / (mbs * dp)` and refused if non-integer; mbs must agree across all clusters
named in `seed_assignment`; the grid splits by seed only; `wandb.project` must match
`general.project`; offline mode refuses a set entity; corpus paths are composed from
`data_root` through a hardcoded setting map; branches resume from the trunk **step directory**
and never from the trunk folder.

## Constants that differ from `tools/`

### `generate_configs.py`

| Constant | 1.5B (`tools/`) | 7B (here) | Why |
|---|---|---|---|
| `MODEL.model_config.hidden_size` | 2048 | **4096** | Llama-2 7B geometry |
| `MODEL.model_config.num_hidden_layers` | 28 | **32** | " |
| `MODEL.model_config.num_attention_heads` | 16 | **32** | " |
| `MODEL.model_config.num_key_value_heads` | 16 | **32** | full MHA, no GQA (`KV == A`) |
| `MODEL.model_config.intermediate_size` | 5632 | **11008** | " |
| `MODEL.model_config.tie_word_embeddings` | `True` | **`False`** | untied; adds 131,072,000 params for the lm_head |
| `TRAIN_STEPS` | 14305 | **71526** | 50B unique tokens x 3 epochs at 2,097,152 tokens/step |
| `WARMUP` | 500 | **2000** | longer run, larger model |
| `LR` | 5.0e-4 | **1.8e-4** | lower peak LR at 7B |
| `BRANCHES['ep1']` | `(4292, 476, 4768)` | **`(21458, 2384, 23842)`** | epoch-1 branch point + 10% cooldown |
| `BRANCHES['ep2']` | `(8583, 954, 9537)` | **`(42916, 4768, 47684)`** | epoch-2 |
| `BRANCHES['ep3']` | `(12875, 1430, 14305)` | **`(64373, 7153, 71526)`** | epoch-3 |
| `TRUNK_SEGMENTS` | `[4292, 8583, 12875]` | **`[21458, 42916, 64373]`** | segment ends == branch points |
| `TRUNK_CKPT_INTERVAL` | 1500 | **5000** | restart insurance; lands on no branch point |
| `PROJECT` | `zhc-1p5b-10b-wsd` | **`zhc-7b-50b-wsd`** | separate wandb project |
| data stage `name` | `S0_top10B` | **`S0_top50B`** | 50B corpora |
| trunk init reference | `_init_1.5B_seed<S>/0` | **`_init_7B_seed<S>/0`** | 7B init checkpoints |

Unchanged: `BRANCH_CKPT_INTERVAL` (100_000), `SETTINGS` (the same six), `SEEDS` (42/43/44),
`OPT_FACTORY` (adamW, beta1 0.9, beta2 0.95, eps 1e-8, fused), `weight_decay` 0.1,
`clip_grad` 1.0, `accumulate_grad_in_fp32` true, and the rest of `MODEL` — `vocab_size` 32000,
`max_position_embeddings` 2048, `rope_theta` 10000.0, `rms_norm_eps` 1.0e-5, `attention_bias`
false, `hidden_act` silu, `init_method.std` 0.02, `make_vocab_size_divisible_by` 1, `dtype`
bfloat16, `ddp_bucket_cap_mb` 25.

Also unchanged, and stated in the generated `HEADER` of every template: accum is derived as
`1024 / (mbs * dp)`, and the global batch is 1024 sequences = **2,097,152 tokens/step**.

**Checkpoint-interval check** (asserted in the module docstring, restated here):
`42916 = 2 x 21458` exactly, but `64373 = 3 x 21458 - 1`, so `gcd(21458, 42916, 64373) = 1` —
the three trunk segment ends share no common `checkpoint_interval` above 1, which is why each
segment has to *end* on its branch point. And `5000` lands on none of them
(`21458 = 4x5000 + 1458`, `42916 = 8x5000 + 2916`, `64373 = 12x5000 + 4373`), so a restart
checkpoint can never be mistaken for a branch point.

### `render_config.py`

| Thing | 1.5B | 7B |
|---|---|---|
| `HERE` | `Path(__file__).resolve().parent.parent` | `.parent.parent.parent` — this file is one directory deeper |
| `DEFAULT_PROFILE` | `deploy/clusters.yaml` | `deploy/clusters_7b.yaml` |
| usage examples | `configs/know-your-sources/`, `--cluster h200` | `configs/know-your-sources-7b/`, `--cluster b300` |
| corpora repo named in comments | `wytro/Know-Your-Sources-tokenized` | `wytro/Know-Your-Sources-7B-tokenized` — **`# PLACEHOLDER`**, does not exist yet |
| trunk latest step in the branch-resume comment | 12875 | 64373 |

`SETTING_CORPUS`, `CORPUS_LEAF`, `OWNED` and **all logic** are unchanged — the diff against
`tools/render_config.py` is comments, docstrings and those two path constants only.

### `assert_invariants.py`

| Constant | 1.5B | 7B |
|---|---|---|
| `EXPECTED_MBS` | 16 | **8** |
| `EXPECTED_TOK_PER_STEP` | 2_097_152 | 2_097_152 (unchanged) |
| `EXPECTED_CORPUS` | six real `(tokens, shards)` tuples | **all six `None`, `# PLACEHOLDER`** |

The corpus table is all-null because the 50B corpora have not been tokenized. A new
`check_corpus_table()` refuses to pass while any value is `None` and names each unfilled
setting. It runs **unconditionally** — not gated on the inherited `--skip-corpus`, whose
purpose is "the corpora are not on this host", not "declare an unverified grid verified". No
skip flag was added. Filling a plausible default instead of `None` would be strictly worse: it
would let the grid start against an unverified corpus.

The log-banner regex, `--check-resume`, `check_env`, `check_tokenizer_metadata` and every
other check are unchanged (the banner's *comment* now shows `mbs: 8 | grad_accum: 32`, but the
regex itself is untouched).

## Why mbs is 8 here and was 16 there

Verified in Phase 0 (2026-09-02): the **1.5B grid was actually run on B300 at dp 4, mbs 16,
accum 16**, read from `tokens.micro_batch_size` in the `config.yaml` of the published 1.5B
checkpoints at
`/home/jhu/zhuicon1/bvandur1-project/zhuicon1/checkpoints/rewrite-1p5b/seed42/quality_base`
(all three step directories agree). Note this means the `dp 8 / accum 8` H200 layout described
in `deploy/clusters.yaml` and `SOP.md` was never executed, even though its mbs 16 was.

The 7B grid runs at **mbs 8** because 16 does not fit a 7B model on a B300 at `zero_stage 0`
(~308 GiB estimated against a 241 GiB budget — see `deploy/clusters_7b.yaml`). The
`masked_mean` per-token weighting therefore differs between the two grids by up to **7.04e-4**
relative.

That affects no comparison the experiment makes. Every six-way between-setting comparison is
**within one grid at one mbs**, and the two grids already differ in LR, warmup, token budget,
embedding tying and parameter count. Within the 7B grid mbs must be identical across all 72
runs, and every guard that enforces that is still in place.
