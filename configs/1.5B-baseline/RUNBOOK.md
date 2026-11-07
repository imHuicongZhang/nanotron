# Know-Your-Sources raw-selected baselines: runbook

**Start with [`WORKFLOW_RAW_BASELINES.md`](WORKFLOW_RAW_BASELINES.md).** It is the single, self-contained entry
point: what to do, in what order, and who does it. This runbook is a detailed reference for the same procedure. If
the two ever disagree, the workflow document wins; please report the discrepancy.

## Purpose

Seven raw (unrewritten) 1.5B training corpora, in two families. The global Top-10B selections are documented in
[`reports/GLOBAL_TOP10B_SELECTION_REPORT.md`](reports/GLOBAL_TOP10B_SELECTION_REPORT.md).

**Strategy-linked controls (4, published earlier; audited, data unchanged).** Each is the shared 5B anchor + a seed-42
uniform random 5B of the unique source documents of one rewritten arm's final rewritten half: an **equal-token-budget**
control, not an identical-document one. `raw_random`'s half is restricted to sources WRAP rewrote successfully;
`raw_rewire_inspired`'s is **conditional on REWIRE's post-rewrite filter**. Audit: [reports/strategy_linked_audit/RAW_SELECTED_BASELINES_PROVENANCE.md](reports/strategy_linked_audit/RAW_SELECTED_BASELINES_PROVENANCE.md).

**Global Top-10B controls (3).** The whole ~10B corpus is one global Top-10B selection over the Quality-Base
universe, by one score. There is no anchor. They are compared against the existing fastText Quality-Base.

| setting | family | anchor | comparator (v2) |
|---|---|---|---|
| `raw_diversity_oriented` | strategy-linked | yes | `diversity_oriented` |
| `raw_disagreement_aware` | strategy-linked | yes | `disagreement_aware` |
| `raw_random` | strategy-linked (uniform-sample population, successful WRAP rewrites only) | yes | `wrap_inspired` |
| `raw_rewire_inspired` | strategy-linked, **conditional on REWIRE's post-rewrite filter** | yes | `rewire_inspired` |
| `raw_top10b_fineweb_edu` | global Top-10B by `fineweb-edu-ranking-v2` | no | `quality_base` |
| `raw_top10b_modernbert` | global Top-10B by `modernbert-ranking-v2` | no | `quality_base` |
| `raw_top10b_consensus` | global Top-10B by mean of the three v2 percentiles | no | `quality_base` |

The list lives in `tools/kys_raw/registry.py`. Every tool reads it from there.

## What you need

- **Repo:** `github.com/imHuicongZhang/nanotron`, branch `huicong-dev`. The data repo's `manifest.json` records the
  code commits that generated each corpus (`code.commit` for the four strategy-linked, `global_top10b.code.commit`
  for the three new).
- **Data:** `blab-jhu/KYS-Pre-Rewritten`, which holds `raw_text/<setting>/` (16 parquet files each, in training
  order), `tokenizer/`, `manifest.json` (counts, `expected_total_tokens`, and sha256 of every file), `selection/`
  and `reports/`. It is published as raw text; you tokenize it (below).
- **Init checkpoints:** `wytro/Know-Your-Sources-init`.

  | seed | directory | hash manifest | rolling sha256 |
  |---|---|---|---|
  | 42 | `_init_1.5B_seed42/0/` | `init_1.5B_seed42.hash.json` | `2ede6612b2e48d7529f867f0e74ca0a7d9ba79cd635d4f803f8022d9b0113aba` |
  | 43 | `_init_1.5B_seed43/0/` | `init_1.5B_seed43.hash.json` | `78ab44e2b2ac954ee441ea340e35969c82cf78bf3f2266f3c8cd0590ce3b8aa3` |
  | 44 | `_init_1.5B_seed44/0/` | `init_1.5B_seed44.hash.json` | `a967df1cae0538c63bf1be412d6e3db0082e75936680f98b11b9a84d49642247` |

  Each has 1,504,299,008 parameters and 180 files. Check with
  `python tools/hash_init_checkpoint.py <dir> --check <manifest>`.
- **Environment:** INSTALL.md, exactly, including `grouped_gemm` at the pinned commit. Run the smoke tests of
  INSTALL.md §3 (the 20-step run is §3.4).

## Order of work

**What runs now is the rebuttal set: nine chains.** Each of these settings runs at seeds 42, 43 and 44:
`raw_diversity_oriented`, `raw_disagreement_aware` and `signal-disagreement-lambda0`. That is 54 segments and 27
annealed finals. The other five raw settings (`raw_random`, `raw_rewire_inspired`, `raw_top10b_fineweb_edu`,
`raw_top10b_modernbert`, `raw_top10b_consensus`) are **not** part of it and must not be queued: always pass
`--settings` to `plan_submit.py`, and tokenize only the two raw settings above. The full commands are in WORKFLOW §1b.

Launch order on a 32-GPU allocation, with four GPUs per chain:
- **Wave 1, eight chains at once, all at dp 4 / mbs 32 / accum 8:** `raw_diversity_oriented` and
  `raw_disagreement_aware` at seeds 42, 43 and 44, plus `signal-disagreement-lambda0` at seeds 42 and 43. The seeds
  run concurrently; this replaces the serial-seed rule below for the rebuttal set.
- **Wave 2:** `signal-disagreement-lambda0` seed 44, once wave 1 leaves the GPUs free.
  - On one 8-GPU node, use dp 8 / mbs 32 / accum 4. The launcher supports this: one node per job, and the GPU count
    is the config's `dp`. It needs a `marc-cluster-8gpu` entry rendered into `filled-dp8/`.
  - Otherwise, use four GPUs at dp 4 like wave 1.
  - dp 8 is not supported across two 4-GPU nodes.

The structure below is that of the full grid; the rebuttal set uses it unchanged for its nine chains.

1. Full grid only: seeds run one after another: 42, then 43, then 44. Each seed starts only after the previous seed's
   24 finals are uploaded.
2. Within a seed, the 8 chains run in parallel. Each chain has 6 segments:
   - `trunk1` 1→4292, `trunk2` →8583, `trunk3` →12875: the stable phase, chained through `latest.txt`;
   - `ep1` 4292→4768, `ep2` 8583→9537, `ep3` 12875→14305: linear decay to 0, each resuming from its trunk step
     directory.
3. Dependencies (`afterok`): trunk2 after trunk1, trunk3 after trunk2, ep1 after trunk1, ep2 after trunk2, ep3 after
   trunk3.
4. Totals: full grid 8 settings × 3 seeds = 24 chains, 144 segments, 72 annealed finals. Rebuttal set: 9 chains,
   54 segments, 27 annealed finals.

## Tokenizing from raw_text

This is CPU only, once per setting.

```python
from huggingface_hub import snapshot_download
data_root = snapshot_download("blab-jhu/KYS-Pre-Rewritten", repo_type="dataset", local_dir="<data_root>")
```

```bash
for s in raw_diversity_oriented raw_disagreement_aware raw_random raw_rewire_inspired \
         raw_top10b_fineweb_edu raw_top10b_modernbert raw_top10b_consensus; do
  tools/kys_raw/tokenize_raw_text.sh <data_root> $s
done
```

**What the script does, per setting:**
1. Checks the tokenizer files and all 16 parquet files against `manifest.json` sha256.
2. Runs:
   ```bash
   python tools/preprocess_data_parquet.py --tokenizer-name-or-path <data_root>/tokenizer/tokenizer.json \
       --eos-token "</s>" --output-folder <data_root>/<s>/tokenized --logging-dir <data_root>/<s>/tokenize_logs \
       --n-tasks 16 parquet --dataset <data_root>/raw_text/<s> --column text --glob-pattern "part-*.parquet"
   ```
   With 16 files and 16 tasks, task i reads file i and nothing is shuffled.
3. Runs `python tools/fix_ds_metadata.py --output-folder <data_root>/<s>/tokenized --tokenizer-dir <data_root>/tokenizer`.
4. Requires 16 shards and a token total exactly equal to `settings.<s>.expected_total_tokens`.

All seven totals were confirmed this way on the preparation cluster (selection report §8; strategy-linked audit
§R5). Each document is its text + one `</s>`; no `<s>` is added.

**Do not:**
- merge, shuffle, re-split or change the task count;
- use another tokenizer.

Use `<data_root>` as `data_root` and `<data_root>/tokenizer` as `tokenizer_path`. `tools/assert_invariants.py` reads
the expected totals from `<data_root>/manifest.json`.

## Additional setting: `signal-disagreement-lambda0`

An eighth setting in `tools/kys_raw/registry.py` (family `lambda_grid`). It is **not** a raw baseline. It is the
λ = 0 arm of the Disagreement-Aware rewritten grid: the shared 5B anchor plus the 5B rewritten half chosen by
`u = q + 0·√var3`, i.e. pure quality. The published `disagreement_aware` arm is λ = 0.5, and it is this setting's
comparator. It runs on the identical schedule, batch (mbs 32 / accum 8 / dp 4), init checkpoints and seeds, so
the rest of this runbook applies unchanged. (Exception in the rebuttal set: seed 44 may run at dp 8 / accum 4 on one
8-GPU node; see Order of work.) It adds 3 chains (18 segments, 9 annealed finals).

- **Configs:** `configs/1.5B-baseline-seed<S>/signal-disagreement-lambda0_seed<S>_<segment>.{yaml,env}` and
  `templates/`, produced by `generate_configs.py` → `render_placeholders.py`. They are identical to
  `raw_random`'s except for the setting name and the comment line naming the template path.
- **The data ships tokenized, not as raw_text.** Do not run `tokenize_raw_text.sh` for it. Place the 26 shards at
  `<data_root>/signal-disagreement-lambda0/tokenized/` (`00000_unshuffled.ds` … `00025_unshuffled.ds`, each with
  `.ds.index` and `.ds.metadata`). Then run
  `python tools/fix_ds_metadata.py --output-folder <data_root>/signal-disagreement-lambda0/tokenized --tokenizer-dir <data_root>/tokenizer`,
  with the same `--tokenizer-dir` as the raw settings.
- **Expected total: 10,000,002,556 tokens in 26 shards (12,566,224 documents).** `assert_invariants.py` reads
  this from the registry's `expected_corpus`, not from `manifest.json`. The assembly manifests give a budget of
  10,000,002,558 tokens over 12,566,226 rows. The 2-token difference is two wikipedia rewrites with empty text
  (`doc_id` 79663246 in `shuffled/part_00020`, 84276725 in `part_00023`). The budget counts each of them as one
  `</s>`, but datatrove writes no document for an empty text. Budget composition: the anchor has
  4,120,164 docs / 5,000,002,332 tokens, the wikipedia rewrites 5,527,293 / 3,514,196,820, and the distill
  rewrites 2,918,769 / 1,485,803,406. Each document is its text plus one `</s>`; no `<s>` is added.
- **Source on the preparation cluster:** `/weka/scratch/jhu/bvandur1/zhuicon1/datasets/rewrite/signal-disagreement-lambda0/`.
  It holds `shuffled/` (26 parquet × 500,000 rows, in training order), `tokenized/` (78 files, 19 GB),
  `tokenized.SHA256SUMS`, `_assembly_manifest.json` and `_pretrain_manifest.json`.
- **Tokenization (2026-11-06):** `tools/preprocess_data_parquet.py --eos-token "</s>" --n-tasks 26 --workers 4
  parquet --dataset <…>/shuffled --column text --glob-pattern "part_*.parquet"`, with the KYS-Pre-Rewritten
  `tokenizer/tokenizer.json` (sha256 `81bb383c…d8e5`), then `fix_ds_metadata.py`. With 26 files and 26 tasks,
  shard i is file i and nothing is shuffled.

**Provenance checks run on 2026-11-06:**
1. *Anchor identity.* The `source_prompt == 'original'` rows are 4,120,164 unique `orig_doc_id`s. As a set they
   are identical to the `source == 'anchor'` rows of KYS-Pre-Rewritten `raw_text/raw_disagreement_aware`, whose
   16 files were verified against `manifest.json` sha256. The intersection is 4,120,164 documents, and the anchor
   holds 5,000,002,332 tokens.
2. *Shuffle.* Recomputing `pp_io.bucketed_shuffle(seed=42)` (B = 16 buckets, inputs `shared-top-5B/` +
   `rewritten/`) reproduces the `shuffled/` row order exactly. The anchor share is 0.327–0.329 in every shard. So
   this is the document-shuffled corpus, not the unshuffled layout behind the retired `10B-base` sawtooth.
3. *Token convention.* 6,000 rows sampled from shards 0, 7, 13 and 25, covering all three sources, were
   re-tokenized with that tokenizer. In every row the stored `train_tokens` equals
   `len(tokens(text, add_special_tokens=False))`.
4. *Tokenized output.* There are 26 shards, `0000i_unshuffled.ds` = `shuffled/part_0000i.parquet`. For each shard,
   the `.ds.metadata` count equals `.ds` bytes / 2, which equals Σ(`train_tokens` + 1) of its parquet file. The
   exceptions are shards 20 and 23, which are 1 token and 1 document short, from the empty rewrites above. In
   every shard, the first document's token ids equal its parquet file's first text tokenized + `</s>`. The total
   is 10,000,002,556. After `fix_ds_metadata.py`, line 1 of all 26 metadata files is byte-identical to the raw
   baselines' line 1.
5. *Configs.* The 18 configs were rendered against this cluster's data with the grid profile (dp 4 / mbs 32 /
   accum 8). The full `assert_invariants.py` (corpus, tokenizer metadata, batch, environment, placeholders) passes
   on all 18. After renaming, the shipped `.env` files are identical to `raw_random`'s. A `raw_random` config still
   passes with its total read from `manifest.json`.

This setting was trained once before, in July 2026 (seeds 42/43/44, from the earlier `examples/` configs). Its
tokenized data and checkpoints from that run are deleted, so this entry re-creates it from the data above.

## Setup steps

1. **Init.** Verify it (above). trunk1 loads the init as **weights only** (`load_optimizer: false`,
   `load_lr_scheduler: false`): the init's optimizer file holds no Adam state, and nanotron refuses to load an empty
   one. Later segments load full state from the trunk.
2. **Cluster entry.** Fill every field of `marc-cluster` in `deploy/clusters.yaml`, then run:
   ```bash
   python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed <S>
   ```
   This re-renders the 48 templates of that seed (or the `--settings` subset) into `configs/1.5B-baseline-seed<S>/filled/`, derives accum and
   checks the memory fit. Never edit rendered configs by hand. Placeholders in the shipped configs:
   `{{DATA_ROOT}} {{TOKENIZER_PATH}} {{CKPT_ROOT}} {{WANDB_ENTITY}} {{WANDB_DIR}} {{CLUSTER}} {{RECOMPUTE_LAYER}}`.
3. **Parallelism.** Global batch 1024 × 2048; dp = GPUs per node; **mbs 32** with `recompute_layer: true` on 80 GB
   cards; accum is derived.
   - Why mbs stays 32: nanotron normalizes the loss per micro-batch (`src/nanotron/models/llama.py:984-1006`), so a
     different mbs changes per-token weights. The recorded effect is 1.43e-3 relative for mbs 4 vs 16
     (`deploy/clusters.yaml`, h200 entry).
   - Recomputation changes speed, not the math.
   - If mbs 32 cannot fit, pass `--expected-mbs N` to `fill_placeholders.py` and `plan_submit.py`, and report it.
   - The v2 Quality-Base comparator's seed 42 itself ran mbs 16 / accum 16; every other v2 run used 32 / 8.
4. **Measured speed** (skipjack, 20 steps, seed-42 init, full batch):

   | configuration | s/it | peak memory |
   |---|---:|---:|
   | 4 × H100, dp 4, mbs 32 + recompute | 22.8 | 59.4 GiB |
   | 4 × H100, dp 4, mbs 4, no recompute | 17.9 | 53.7 GiB |
   | 1 × H100, mbs 32 + recompute | 63.9 | 57.9 GiB |
   | 1 × H100, mbs 4 | 72.2 | 50.3 GiB |

   Loss at steps 1 / 10 / 20 was 10.8 / 8.83 / 8.18–8.19 in all four (agreement to about 0.01 at three significant
   figures; no bitwise claim).
   - Wall time at 22.8 s/it: 27.2 h per trunk segment, 3.0 / 6.0 / 9.1 h for ep1 / ep2 / ep3.
   - Per chain: 15,735 steps, 99.7 node-hours. Whole grid (24 chains): 2,392 node-hours. Rebuttal set (9 chains):
     897 node-hours.
   - **`slurm.time` must exceed the longest stretch without a checkpoint** (about 9.5 h at 22.8 s/it) with margin,
     so ≥ 12 h. `plan_submit.py --s-per-it` enforces this.
5. **Invariants.** `fill_placeholders.py` runs `assert_invariants.py --skip-corpus --skip-env` on every config.
   After tokenizing, run it fully per config (`python tools/assert_invariants.py --config <filled cfg>`, after
   sourcing the `.env`). The launcher repeats it with `--check-resume` before every segment.
6. **Submission.**
   ```bash
   python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed <S> --s-per-it <measured>
   ```
   This writes `submit_seed<S>.sh` and prints the plan. The script refuses to submit anything if one of its job
   names is already queued for `$USER`, seeds each trunk with the init (step 0 + `latest.txt`), and submits
   `deploy/slurm/kys_segment.sbatch` per segment with `afterok` dependencies and `--requeue`.
   - `--test-only` asks SLURM to validate each segment without queueing.
   - `--settings a,b` restricts it to a subset.
   - `--submit` runs the script.
   - Run it from the clone set as `repo_dir`.
7. **Requeue.** A trunk resumes via `latest.txt`, losing at most 1500 steps; a branch reruns from its branch point; a
   segment whose final checkpoint exists exits 0. Do **not** resubmit while jobs of the seed are still queued or
   running: the final-checkpoint skip does not protect an unfinished segment.

## Logging

Use your own wandb project, or `mode: offline` with `wandb.dir` on shared storage. The run name is
`<setting>_seed<S>_<segment>` (= `general.run`); the `.env` sets `WANDB_RUN_GROUP=<setting>_seed<S>`,
`WANDB_JOB_TYPE=<segment>` and tags for mbs/dp/accum/cluster. nanotron passes `general.project`
(`zhc-1p5b-10b-wsd`) to `wandb.init`. The deliverable is the exported per-segment loss CSV, not wandb.

## Deliverables per seed

- For each setting, each ep final in nanotron format **and** as an HF export, uploaded to
  `blab-jhu/KYS-1.5B-Raw-Selected-Baselines`.
  - The finals are ep1 4768, ep2 9537 and ep3 14305, at `<ckpt_root>/seed<S>/<setting>/ep<N>/<setting>/seed<S>/<step>/`.
  - Export: `python tools/kys_eval/convert_to_hf.py --checkpoint_path <step dir> --save_path <out> --tokenizer <tokenizer dir>` (CPU).
  - Upload layout: `rewrite-1p5b/seed<S>/<setting>/ep<N>/nanotron/`, `.../hf/`, and `rewrite-1p5b/seed<S>/<setting>/logs/`.
- Loss CSVs per segment.
- The filled configs and the `marc-cluster` entry on branch `marc-runs`. Never commit tokens.
- A short note: mbs, s/it, requeues, and any deviation.

## Evaluation

Use `github.com/imHuicongZhang/kys-eval` at `main` `c8aa8ff` or later, which covers `signal-disagreement-lambda0`
(paired with `disagreement_aware`). The full grid is 72 cells; the rebuttal set is 27, selected with
`--settings raw_diversity_oriented raw_disagreement_aware signal-disagreement-lambda0`. Run the reference check
first, then `run_grid`, then `aggregate`. The commands
are in WORKFLOW §14 and the kys-eval README. Upload `results/` and `reports/` to `rewrite-1p5b/seed<S>/eval/`.

## Contact

Questions go to Huicong Zhang. Do not change the LR schedule, token budget, segment boundaries, micro-batch size or
data order.
