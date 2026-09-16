# 1.5B raw-selected baselines

> Operators on an external cluster: start with [RUNBOOK.md](RUNBOOK.md). Data counts live in the
> data repo's `manifest.json` (blab-jhu/KYS-Pre-Rewritten); values below marked pending are filled
> by follow-up commits.

No-rewrite controls for the Know-Your-Sources 1.5B grid. Each trains on the shared 5B anchor plus
the **original, unrewritten** text of the documents one rewritten arm rewrote, subsampled at the
document level to the same 5B strategy budget, so the difference from that arm isolates the effect
of rewriting from the effect of source selection.

| setting | rewritten counterpart | strategy half |
|---|---|---|
| `raw_diversity_oriented` | `diversity_oriented` | original text of the documents the Diversity-Oriented arm rewrote |
| `raw_disagreement_aware` | `disagreement_aware` | original text of the documents the Disagreement-Aware arm rewrote |
| `raw_random` | `wrap_inspired` | original text of the uniform random sample the WRAP-Inspired arm rewrote (no new sample drawn) |
| `raw_rewire_inspired` | `rewire_inspired` | original text of the source documents of the 5B kept by REWIRE's post-rewrite filter |

`raw_random` is the **no-selection reference** — its sources are a uniform random sample — and the
no-rewrite control for `wrap_inspired`. `raw_rewire_inspired` starts from a random pool too, but its
sources are only those whose rewrites passed REWIRE's post-rewrite quality filter.

## Layout

    configs/1.5B-baseline/README.md, RUNBOOK.md
    configs/1.5B-baseline-seed<S>/                  S = 42, 43, 44
        templates/<setting>_seed<S>_<kind>.yaml       24 experiment templates (tools/generate_configs.py)
        <setting>_seed<S>_<kind>.yaml + .env          24 rendered configs with {{PLACEHOLDERS}} (render_placeholders.py)
        filled/                                        produced on the training cluster (fill_placeholders.py)

`<kind>` is `trunk1`, `trunk2`, `trunk3` (stable phase, ending at steps 4292 / 8583 / 12875) and
`ep1`, `ep2`, `ep3` (decay branches from those trunk steps, ending at 4768 / 9537 / 14305).

**What differs between settings.** Against its rewritten counterpart, each rendered config differs
in exactly four fields (all 24 seed-42 configs rendered for skipjack and diffed): `general.run`, the
dataset folder, the checkpoint paths (setting name only), and `parallelism.recompute_layer`. Model,
LR schedule, steps, global batch, micro batch, seeds and data order are identical.

**What differs between seeds.** Exactly what differs in the original grid: `general.seed`,
`data_stages[0].data.seed`, the run name and the seed directory in the checkpoint paths, plus the
seed's init checkpoint (seeded into the trunk directory by `plan_submit.py`). Data order does not
change: the corpora were shuffled and subsampled once, with seed 42, and nanotron's data seed only
reorders files when `shuffle_files` is true, which it never is here (verified in
`src/nanotron/config/config.py` and the released seed-43 checkpoint configs).

**Placeholders.** The committed configs are complete except for site values: `{{DATA_ROOT}}`,
`{{TOKENIZER_PATH}}`, `{{CKPT_ROOT}}`, `{{WANDB_ENTITY}}`, `{{WANDB_DIR}}`, `{{CLUSTER}}`,
`{{RECOMPUTE_LAYER}}`. `tools/assert_invariants.py` refuses any config or `.env` that still contains
one. Fill the `marc-cluster` entry in `deploy/clusters.yaml`, then
`tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed <S>` re-renders the templates with
those values (renderer guards included) and `tools/kys_raw/plan_submit.py` writes the SLURM chain.

## Data: blab-jhu/KYS-Pre-Rewritten (raw text)

    raw_text/<setting>/part-00000.parquet ... part-00015.parquet    orig_doc_id, source (anchor|strategy), text
    manifest.json   README.md

Published as raw text in the exact post-shuffle document order, anchor merged in, 16 contiguous files
per setting. The consumer tokenizes with `tools/kys_raw/tokenize_raw_text.sh` (RUNBOOK.md, "Tokenizing
from raw_text"): 16 datatrove tasks give task i exactly file i, so shards `00000`…`00015` concatenate to
the file order, and the summed token count must equal the manifest's `expected_total_tokens`.

**Source set.** For each arm, the `orig_doc_id` of every non-anchor row (`source_prompt != 'original'`)
of the published `wytro/Know-Your-Sources/<arm>/*.parquet`, deduplicated across the rewriting
prompts. Text is read only from the 100M DCLM-RefinedWeb pool by position (shard `id // 500000`, row
`id % 500000`); no published or rewritten text builds a corpus. This is not the full pre-rewrite
selection: source documents whose rewrites failed (or, for REWIRE, were filtered out) never reached
the arm. For Disagreement-Aware the recorded selection is 5,602,476 documents, of which 5,592,424
are in the source set.

**Token budget rule.** Tokens per document = `len(llama2_tokenizer(text, add_special_tokens=False)) + 1`
(the +1 is the EOS appended at tokenization). A source set above 5,000,000,000 tokens is cut by taking
documents in the order of `np.random.default_rng(42).permutation` over the sorted unique ids and
keeping the shortest prefix reaching 5B — whole documents, at most one document of overshoot. At or
below 5B it is used as is.

**Anchor.** 4,120,164 documents / 5,000,002,332 tokens: the `source_prompt == 'original'` rows,
identical ids and counts in all four published arms; text read from the pool by position.

| setting | rewrite rows | source docs (dedup) | raw tokens before | docs after | tokens after |
|---|---:|---:|---:|---:|---:|
| `raw_diversity_oriented` | 8,336,411 | 5,867,876 | 9,728,111,104 | 3,018,451 | 5,000,000,553 |
| `raw_disagreement_aware` | 8,442,273 | 5,592,424 | pending | pending | pending |
| `raw_random` | 13,022,091 | 10,573,523 | pending | pending | pending |
| `raw_rewire_inspired` | 12,290,444 | pending | pending | pending | pending |

Merged corpus per setting = anchor + subsampled strategy documents (e.g. `raw_diversity_oriented`:
7,138,615 documents, 10,000,002,885 tokens), shuffled together at the document level with seed 42 by
`pp_io.bucketed_shuffle` (the function every published arm used).

**Leakage checks** (`tools/kys_raw/verify_raw_corpus.py`; a setting is uploaded only after all pass):
(1) every parquet read in the build code classified by line — text only from the pool; (2) all
4,120,164 anchor texts equal the pool and the published anchor text and total exactly 5,000,002,332
tokens; (3) 200 sampled documents per setting equal the pool and none of the rewrites of the same
source; (4) markdown-heading, list-marker and no-URL/boilerplate rates sit closer to the raw pool
than to the rewritten arm; (5) strategy documents are longer than the rewrites. Results per setting
are recorded in manifest.json.

Build path: `tools/kys_raw/build_raw_sources.py` → `assemble_raw_corpus.py` → `verify_raw_corpus.py`
→ `publish_raw_text.py` (driven per setting by `run_pipeline.sh`).

## Micro batch and parallelism

Global batch 1024 x 2048 tokens; the grid's dp 4 / **mbs 32** / accum 8, zero_stage 0.

**What the original grid ran**, read from the `config.yaml` of all 54 released checkpoints: dp 4,
mbs 32, accum 8 in 53 of 54. **The one exception is seed-42 `quality_base`** (ep1-ep3), which ran
mbs 16 / accum 16. The mbs 16 previously pinned in `deploy/clusters.yaml`, `SOP.md` and
`tools/assert_invariants.py` was never the grid's value; those now record 32.

**On 80 GB H100s** (one-GPU probe, skipjack job 424229, 64 sequences per step, steps 2-12):

| probe | s/it | tokens/s | peak allocated | peak reserved | loss @ step 12 |
|---|---:|---:|---:|---:|---:|
| mbs 4, no recompute | 3.26 | 40.2K | 46.8 GiB | 47.4 GiB | 8.74 |
| mbs 16 + recompute | 4.02 | 32.6K | 37.7 GiB | 40.6 GiB | 8.74 |
| mbs 32 + recompute | 4.02 | 32.6K | 49.6 GiB | 55.3 GiB | 8.74 |

Decision: mbs 32 with full layer recomputation, accum 8 (49.6 GiB peak on one H100, 23% slower than
mbs 4). Without recomputation mbs 32 needs ~191 GiB. Recomputation changes no math, so the raw
baselines keep the counterparts' `masked_mean` weighting exactly. 4 x H100 s/it: pending.

**Sanity check** (seed-42 init, rewritten `diversity_oriented`, 20 steps, full 1024 x 2048 batch, one
H100 each with dp 1 and accum scaled to match; skipjack jobs 426597 / 426598):

| | mbs 32 + recompute | mbs 4, no recompute |
|---|---:|---:|
| s/it (full 1024-seq step) | **63.9** | 72.2 |
| peak GPU memory | 57.9 GiB | 50.3 GiB |
| lm_loss at steps 1 / 10 / 20 | 10.8 / 8.83 / 8.18 | 10.8 / 8.83 / 8.18 |

The loss curves agree at every step (a single step differs by 0.01 at three significant figures),
which is what dp-freedom and recomputation predict. At the full step mbs 32 is ~11% **faster** than
mbs 4 — the opposite of the 64-sequence probe — because mbs 4 needs 256 micro-batches per step
against 32. Measured 4 x H100 s/it: pending (job queued); ~16 s/it scaled from one GPU.

## Environment notes (skipjack; see INSTALL.md)

`grouped_gemm` is required even for the dense Llama (nanotron imports it unconditionally). Compute
nodes without Python headers need `C_INCLUDE_PATH` set for Triton and a prebuilt dataset helper.
Both are documented in INSTALL.md, and `deploy/slurm/kys_segment.sbatch` takes the include path from
the cluster entry.
