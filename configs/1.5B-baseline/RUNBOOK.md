# Know-Your-Sources raw-selected baselines: runbook

**Full workflow:** [`WORKFLOW_RAW_BASELINES.md`](WORKFLOW_RAW_BASELINES.md) is the single entry point for executing this grid end to end on an external cluster — what to do, in what order, and who does it. This runbook is one of its detailed references.

## Purpose
Train no-rewrite controls for four source-selection strategies so the rewriting effect can be separated from the selection effect. Each control uses the same 5B anchor and the same selected source documents as its rewritten counterpart, but keeps the documents in their original unrewritten form, subsampled at the document level to the same 5B budget. raw_random is the no-selection reference and the control for wrap_inspired. raw_rewire_inspired uses the source documents of the 5B kept by REWIRE's post-rewrite filter.

| setting | rewritten counterpart |
|---|---|
| `raw_diversity_oriented` | `diversity_oriented` |
| `raw_disagreement_aware` | `disagreement_aware` |
| `raw_random` | `wrap_inspired` |
| `raw_rewire_inspired` | `rewire_inspired` |

## What you need
- Repo: https://github.com/imHuicongZhang/nanotron, branch huicong-dev, commit: the commit recorded as `code.commit` in the data repo's manifest.json (the commit that added this runbook; a runbook cannot contain its own hash). Any later huicong-dev commit that only fills pending values in this file is equivalent.
- Data: https://huggingface.co/datasets/blab-jhu/KYS-Pre-Rewritten, folders raw_text/raw_diversity_oriented, raw_text/raw_disagreement_aware, raw_text/raw_random, raw_text/raw_rewire_inspired, each about 10B tokens, 16 parquet files. See manifest.json for token counts and checksums.
  - The data is published as **raw text**; you tokenize it (section "Tokenizing from raw_text"). Each folder is the final corpus in training order: anchor merged in, shuffled, rows in order.
  - Tokenizer: `tokenizer/` in the same dataset repo, the exact llama-2 tokenizer directory the grid was tokenized with (sha256 of each file in manifest.json, `tokenizer.sha256`). It downloads with the data; no other repo is needed.
- Init checkpoints: https://huggingface.co/wytro/Know-Your-Sources-init (model repo)
  - seed 42: `_init_1.5B_seed42/0/`, hash manifest `init_1.5B_seed42.hash.json`, rolling sha256 `2ede6612b2e48d7529f867f0e74ca0a7d9ba79cd635d4f803f8022d9b0113aba`
  - seed 43: `_init_1.5B_seed43/0/`, hash manifest `init_1.5B_seed43.hash.json`, rolling sha256 `78ab44e2b2ac954ee441ea340e35969c82cf78bf3f2266f3c8cd0590ce3b8aa3`
  - seed 44: `_init_1.5B_seed44/0/`, hash manifest `init_1.5B_seed44.hash.json`, rolling sha256 `a967df1cae0538c63bf1be412d6e3db0082e75936680f98b11b9a84d49642247`
  - 1,504,299,008 parameters each. Verify with `python tools/hash_init_checkpoint.py <init_root>/_init_1.5B_seed<S>/0 --check <init_root>/init_1.5B_seed<S>.hash.json`.
- Environment: follow INSTALL.md exactly, including grouped_gemm at the pinned commit. Verify with the 20-step smoke test described there (INSTALL.md §3.4, `tools/kys_raw/smoke_20steps.py`; run it after tokenizing `raw_diversity_oriented`).
  - If your compute nodes have no Python development headers, set `python_include` in the cluster entry (INSTALL.md, "Python headers on compute nodes"); Triton and nanotron's dataset helper compile against `Python.h` at run time.

## Order of work
1. Run seed 42 for all four settings first. Start seeds 43 and 44 only after seed 42 is complete and its checkpoints are uploaded.
2. Within a seed, each setting has 6 segments: trunk1, trunk2, trunk3 (stable phase, chained) and ep1, ep2, ep3 (decay branches, each starting from the end of the corresponding trunk). Dependencies: trunk2 after trunk1, trunk3 after trunk2, ep1 after trunk1, ep2 after trunk2, ep3 after trunk3.
3. Final steps: trunk1/2/3 end at 4292 / 8583 / 12875, ep1/2/3 end at 4768 / 9537 / 14305.
   - ep1 resumes from trunk step 4292 and decays over 476 steps, ep2 from 8583 over 954, ep3 from 12875 over 1430 (linear to 0). The trunks carry the ep3 decay parameters and stop at or before step 12875, so their LR is constant after the 500-step warmup.

## Tokenizing from raw_text
Do this once per setting, before filling configs. It needs the environment from INSTALL.md (datatrove 0.5.0) and CPU only.

1. Download the data; the tokenizer comes with it, in `<data_root>/tokenizer/`:
   ```python
   from huggingface_hub import snapshot_download
   data_root = snapshot_download("blab-jhu/KYS-Pre-Rewritten", repo_type="dataset", local_dir="<data_root>")
   ```
2. Tokenize each of the four settings into 16 shards, preserving document order. The tokenizer directory defaults to `<data_root>/tokenizer`; pass it as a third argument only if you keep it elsewhere.
   ```bash
   tools/kys_raw/tokenize_raw_text.sh <data_root> raw_diversity_oriented
   tools/kys_raw/tokenize_raw_text.sh <data_root> raw_disagreement_aware
   tools/kys_raw/tokenize_raw_text.sh <data_root> raw_random
   tools/kys_raw/tokenize_raw_text.sh <data_root> raw_rewire_inspired
   ```
   For each setting the script:
   - checks the tokenizer files against `tokenizer.sha256` and the 16 `raw_text/<setting>/part-000NN.parquet` files against the sha256 in manifest.json;
   - runs `python tools/preprocess_data_parquet.py --tokenizer-name-or-path <data_root>/tokenizer/tokenizer.json --eos-token "</s>" --output-folder <data_root>/<setting>/tokenized --logging-dir <data_root>/<setting>/tokenize_logs --n-tasks 16 parquet --dataset <data_root>/raw_text/<setting> --column text --glob-pattern "part-*.parquet"`. With 16 files and 16 tasks, task i reads exactly file i and nothing is shuffled, so shards `00000`…`00015` concatenate to the file order;
   - runs `python tools/fix_ds_metadata.py --output-folder <data_root>/<setting>/tokenized --tokenizer-dir <data_root>/tokenizer`;
   - verifies 16 shards and that the summed `.ds.metadata` token count equals `settings.<setting>.expected_total_tokens` in manifest.json, exactly.
3. **No merge or shuffle is needed on your side.** The anchor is already merged into every folder and the documents are already shuffled (seed 42). Do not pass `--shuffle`, change `--n-tasks`, or re-split the parquet files: any of these changes the data order.
4. Use `<data_root>` as `data_root` and `<data_root>/tokenizer` as `tokenizer_path` in the cluster entry. `tools/assert_invariants.py` reads the expected totals from `<data_root>/manifest.json`.

## Setup steps
1. Download the four data folders and the init checkpoints. Verify sha256 against manifest.json.
   - Data: `tokenize_raw_text.sh` checks every parquet file's sha256 against manifest.json before tokenizing.
   - Init: `tools/hash_init_checkpoint.py --check` as above. That check covers the model weights only, so also confirm each `_init_1.5B_seed<S>/0/` holds all 180 files (a complete download; `plan_submit.py`'s seeding step refuses an incomplete init). trunk1 loads the init as **weights only** (`load_optimizer: false`, `load_lr_scheduler: false` in the configs): the init's optimizer file holds no Adam state, only fp32 copies bit-identical to the weights, and nanotron refuses to load an empty optimizer state. Do not switch those two flags on for trunk1; trunk2, trunk3 and the ep branches correctly load full optimizer state from the trunk checkpoints.
2. Fill the placeholders {{DATA_ROOT}}, {{CKPT_ROOT}}, {{CLUSTER}} in configs/1.5B-baseline-seed<S>/ and the marc-cluster entry in deploy/clusters.yaml.
   - The shipped configs also mark {{TOKENIZER_PATH}}, {{WANDB_ENTITY}}, {{WANDB_DIR}} and {{RECOMPUTE_LAYER}}. Do not edit the 24 configs by hand: fill every field of `marc-cluster` in `deploy/clusters.yaml` (each field is commented there), then run `python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed <S>`. It re-renders the 24 templates in `configs/1.5B-baseline-seed<S>/templates/` with your values into `configs/1.5B-baseline-seed<S>/filled/`, deriving accum from your mbs and dp and checking the memory fit.
3. Parallelism: keep the global batch at 1024 sequences x 2048 tokens. The original grid used dp 4, mbs 32, accum 8. Use mbs 32 if it fits your GPUs, otherwise the largest mbs that fits and adjust accum to keep the global batch. Record the mbs you used. Our probe on 80GB H100 found mbs 32 with full layer recomputation (`recompute_layer: true`) and accum 8 fits: 49.6 GiB peak allocated on one H100, 23% slower than mbs 4 without recomputation (4.02 s vs 3.26 s per 64-sequence step). mbs 32 without recomputation needs about 191 GiB. Recomputation changes speed only, not the math, so mbs 32 + recompute reproduces the grid exactly. 4 x H100 s/it for the full 1024-sequence step: pending (sanity run queued).
   - If you must use another mbs, pass `--expected-mbs <N>` to `fill_placeholders.py` and `plan_submit.py`; a different mbs slightly changes the loss weighting (`masked_mean`, SOP.md §1), so report it.
4. Run tools/assert_invariants.py on every config. It must pass and must report no remaining placeholders.
   - `fill_placeholders.py` already runs it on all 24 (batch, placeholders). Before training, run it fully per config — `python tools/assert_invariants.py --config <filled cfg>` after sourcing the matching `.env` — so the corpus token count (from manifest.json) and the tokenizer path are checked against the data on disk. The launcher repeats this with `--check-resume` before every segment.
5. Use the submission planner (tools/kys_raw/plan_submit.py) to generate SLURM scripts with the dependency chain above. Jobs must resume from the latest checkpoint on requeue or preemption.
   - `python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed <S>` writes `configs/1.5B-baseline-seed<S>/filled/submit_seed<S>.sh`: it seeds each trunk directory with the seed's init checkpoint, then submits the 24 segments of `deploy/slurm/kys_segment.sbatch` with `afterok` dependencies. Run the script (or add `--submit`).
   - `kys_segment.sbatch` is submitted with `--requeue`. On restart a trunk resumes from its newest checkpoint through `latest.txt` (checkpoint_interval 1500), a branch restarts from its branch point, and a segment whose final checkpoint already exists exits 0 without training.

## Logging
Log to your own wandb project and send us the exported loss CSVs per segment. Run names: raw_<setting>_seed<S>_<segment>.

Set `wandb.project`, `wandb.entity` and `wandb.dir` in `marc-cluster` to your own (or `mode: offline`). The configs set `general.run` to exactly that name, and the rendered `.env` sets `WANDB_RUN_GROUP=raw_<setting>_seed<S>`, `WANDB_JOB_TYPE=<segment>` and tags carrying mbs, dp, accum and cluster. Note: nanotron passes `general.project` (`zhc-1p5b-10b-wsd`) to `wandb.init`; keep that project name in your entity, or change `PROJECT` in `tools/generate_configs.py` and regenerate the templates.

## Deliverables per seed
- For each setting, the final checkpoint of ep1, ep2 and ep3 in nanotron format and as an HF export, produced with tools/kys_eval/convert_to_hf.py. Upload to blab-jhu/KYS-1.5B-Raw-Selected-Baselines, layout rewrite-1p5b/seed<S>/<setting>/ep<N>/.
  - Final checkpoints: ep1 step 4768, ep2 step 9537, ep3 step 14305, under `<ckpt_root>/seed<S>/<setting>/ep<N>/<setting>/seed<S>/<step>/`.
  - Export: `python tools/kys_eval/convert_to_hf.py --checkpoint_path <step dir> --save_path <out> --tokenizer <tokenizer dir>` (CPU only; torch, safetensors, transformers).
  - `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` does not exist yet; we will create it (model repo) and grant your uploader write access before seed 42 finishes. Upload the nanotron step directory and the HF export side by side: `rewrite-1p5b/seed<S>/<setting>/ep<N>/nanotron/` and `rewrite-1p5b/seed<S>/<setting>/ep<N>/hf/`.
- Loss logs per segment as CSV.
- The filled configs and clusters.yaml entry as used, committed to a branch named marc-runs.
- A short note with the mbs used, s/it, and any deviation from this runbook.

## Evaluation
Run evaluation on each ep checkpoint. The LightEval configuration and task list will be added to this runbook in a follow-up commit; until then, upload checkpoints and loss logs as described above.

## Contact
Questions go to Huicong Zhang. Do not change the LR schedule, token budget, segment boundaries or data order.
