# KYS grid — handover package

Everything needed to run the 72-run per-epoch-WSD grid on a fresh cluster. Sizes measured
2026-08-09.

## 1. Code — ~32 MB

| item | size | note |
|---|---:|---|
| `nanotron-kys` worktree | 14.4 MB | branch `kys/epoch-wsd` |
| `.git` | 17.3 MB | carries tags `upstream-pin-2411b022`, `kys-patched-base` |

Ship as a git bundle so history and tags survive:

```bash
git bundle create nanotron-kys.bundle --all
# receiving side:
git clone nanotron-kys.bundle nanotron-kys && cd nanotron-kys && git checkout kys/epoch-wsd
```

Pinned upstream commit `2411b022a75fb7f7561a1bb4166706da5e1b76de` (2026-04-07), which is
**still upstream `main` HEAD** as of 2026-08-09 — a fresh clone of `huggingface/nanotron`
gives byte-identical source. 9 patches on top; see `INSTALL.md` §4 for which and why.

Includes:
- `INSTALL.md` — install instructions (H200 primary, Blackwell appendix)
- `tools/probe_blackwell.py` — 30-second on-hardware go/no-go check
- `tools/gen_kys_configs.py` — emits the 108 experiment templates
- `tools/render_config.py` + `deploy/clusters.yaml` — per-cluster deployment stamping
- `tools/hash_init_checkpoint.py` — init-checkpoint verification
- `tools/assert_invariants.py` — preflight + step-200 batch-invariant assertion
- `SOP.md` — launch procedure, the mbs rationale, and the full wandb setup
- `PATCH_NOTES.md`, `UPSTREAM_PIN.md`, `COMPATIBILITY.md`

## 2. Configs — <2 MB

108 templates = 18 trunks × 3 segments + 54 branches, covering 72 logical runs
(6 settings × 3 seeds × (1 trunk + 3 cooldown branches)).

Templates deliberately do **not** parse on their own: `parallelism`, `micro_batch_size`,
`batch_accumulation_per_replica`, `zero_stage` and `sequence_length` are stamped at launch by
`render_config.py`, which derives `accum = 1024 / (mbs × dp)` and refuses non-integers, so the
2,097,152-token global batch is invariant across every run and every cluster.

## 3. Tokenizer — 3.6 MB

`tokenizers/llama2-unsloth-tokenizer/` (llama-2 32000 vocab, token_size 2 bytes).

## 4. Tokenized corpora — ~120.4 GB

| setting | folder | size |
|---|---|---:|
| quality-base | `10B-base-shuf42/tokenized` | ~20.06 GB |
| quality-first | `quality-first/tokenized` | 20.12 GB |
| diversity-first | `diversity-first/tokenized` | 19.88 GB |
| wrap | `wrap/tokenized` | 20.14 GB |
| rewrite | `rewrite/tokenized` | 20.13 GB |
| signal-disagreement-λ0.5 | `signal-disagreement-lambda05/tokenized` | 20.10 GB |

16 `.ds` shards each plus `.ds.index` / `.ds.metadata`. **Ship the `.ds.metadata` files** —
nanotron's config validator reads `vocab_size` from them and refuses to start without it
(`Model's vocab_size (32000) does not match dataset's (None)`).

`10B-base-shuf42` is the re-shuffled quality-base corpus (seed 42, matching the other five);
it replaces the old unshuffled `10B-base`. Do not ship the old one.

## 5. Init checkpoints — 27.1 GB

`_init_1.5B_seed{42,43,44}/0/` at 9.03 GB each. Each trunk directory must be pre-seeded with
its seed's init as step 0 plus a `latest.txt` containing `0`, so the trunk's latest.txt
auto-resume works for both first launch and crash-restart.

Verify after transfer **and** after nanotron loads them:

```bash
python tools/hash_init_checkpoint.py <ckpt>/0 --check init_1.5B_seedNN.hash.json
python tools/hash_init_checkpoint.py <ckpt>/0 --check init_1.5B_seedNN.hash.json --mode loaded
```

| checkpoint | parameters | rolling sha256 |
|---|---:|---|
| `_init_1.5B_seed42` | 1,504,299,008 | `2ede6612b2e48d7529f867f0e74ca0a7d9ba79cd635d4f803f8022d9b0113aba` |
| `_init_1.5B_seed43` | 1,504,299,008 | `78ab44e2b2ac954ee441ea340e35969c82cf78bf3f2266f3c8cd0590ce3b8aa3` |
| `_init_1.5B_seed44` | 1,504,299,008 | `a967df1cae0538c63bf1be412d6e3db0082e75936680f98b11b9a84d49642247` |

The hash is layout-independent: it keys on the logical parameter path with the
`_pp-rank-N-of-M_tp-rank-N-of-M` suffix stripped, and hashes raw storage bytes via a uint8
reinterpret (numpy has no bfloat16). It is therefore unaffected by `dp`, which is correct —
with `tp=1, pp=1, zero_stage=0` the `model/`, `optimizer/` and `lr_scheduler/` payloads carry
no `dp` in their filenames and `load_random_states()` is never called, so a checkpoint written
at dp=4 resumes at any dp.

The init optimizer state is empty (`state_dict["state"] == {}` — Adam has never stepped), so
`load_optimizer: true` on the first trunk segment is equivalent to a fresh optimizer.

## 6. Total

| | |
|---|---:|
| code + configs + tokenizer | ~36 MB |
| tokenized corpora | ~120.4 GB |
| init checkpoints | 27.1 GB |
| **total** | **~148 GB** |

Unchanged by the stack move — the Blackwell upgrade adds no shipped bytes, only different
pip pins.

## 7. Disk to reserve on the receiving side

| | per (setting,seed) | grid (×18) |
|---|---:|---:|
| checkpoints, peak | 295 GB | 5.31 TB |
| after pruning trunk restart points | 126 GB | 2.27 TB |

14 checkpoints per chain at 21.06 GB each (optimizer 18.05 + model 3.01): 6 must be kept
(3 branch points 4292/8583/12875 + 3 annealed finals 4768/9537/14305), 8 are trunk restart
insurance and can be deleted once their segment completes. Stripping optimizer state from the
3 annealed finals — the only ones eval needs — takes each from 21.06 GB to 3.01 GB.

## 8. What is and is not comparable across layouts

Measured 2026-08-09 against the real corpora and the real sampler.

**`dp` is free.** `MegatronPretrainingSampler.__iter__` accumulates `mbs × dp` consecutive
sample indices and hands rank *d* the slice `[d*mbs : (d+1)*mbs]`, so the partition of the
1024-sequence global batch into micro-batches is **consecutive blocks of size `mbs`,
regardless of `dp`**. The accumulated gradient is
`(1/dp)·(1/accum)·Σ_j loss_j` = the mean over all `1024/mbs` micro-batches, which has no `dp`
in it. Verified numerically on 1024 real sequences: per-token weights for
`(mbs=4, dp=4, accum=64)` vs `(mbs=4, dp=8, accum=32)` differ by **0.000e+00**. Only
floating-point rounding (the all-reduce tree) differs.

**`micro_batch_size` is not free.** `Loss.forward` computes
`masked_mean(loss, label_mask) = (loss*mask).sum() / mask.sum()`, and `label_mask` drops the
token before every document boundary (`return_positions` defaults True; datatrove supplies
`positions`; the collator masks `position_ids == 0`). Measured on quality-first: **2.8 masked
tokens per 2048-token sequence, range 0–10**. So `masked_mean`'s denominator `n_j` varies
between micro-batches — relative std 6.2e-4 at mbs=4, 3.9e-4 at mbs=16 — and the effective
per-token weight is `1/(n_micro · n_j)`. Regrouping the same 1024 sequences into
different-sized micro-batches therefore changes the objective:

| comparison | max relative per-token weight difference |
|---|---:|
| mbs=4 dp=4 accum=64 **vs** mbs=4 dp=8 accum=32 | **0.000e+00** (identical) |
| mbs=4 **vs** mbs=8 | 1.04e-3 |
| mbs=4 **vs** mbs=16 | 1.43e-3 |
| mbs=8 **vs** mbs=16 | 7.04e-4 |

That is ~4 orders of magnitude above fp32 rounding — a genuine difference in the optimization
objective, not a rounding artefact.

**Consequence.** `micro_batch_size` must be identical across all 72 runs;
`dp` may differ per cluster. `render_config.py` enforces both: it refuses to render when the
clusters named in `seed_assignment` disagree about `micro_batch_size`, and refuses an `mbs`
that will not fit a cluster's HBM. H100 (80 GB) caps `mbs` at 4; H200 (141 GB) allows 16 —
so a grid spanning both must run mbs=4 on both sides.
