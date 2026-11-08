# Rebuttal consistency audit — code and data vs. the ICLR rebuttal goals (2026-11-08)

Second, independent check that the training code, evaluation code and published data produce what the rebuttal
needs (two controls, nine chains, 27 annealed checkpoints, Appendix-F-style paired comparisons). No new experiments
are proposed. CPU-only allocation: no GPU step is run or claimed.

Status legend for sections: **DONE** (written with evidence), **IN PROGRESS**, **OPEN** (not started).
This file is appended section by section; if the allocation ends, everything above the last DONE marker is final.

## 0. Starting state — DONE

| item | value |
|---|---|
| date | 2026-11-08 |
| node | CPU only (17 cores, 503 GiB RAM, no `nvidia-smi`) |
| nanotron worktree | `/weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys-lambda0`, `huicong-dev` @ `6fe10b9b130a22338bfc1c2d199761131b711a88` = origin, clean |
| kys-eval | `/weka/scratch/jhu/bvandur1/zhuicon1/projects/kys-eval`, `main` @ `1c191acfba7f035258f541aa389ffa186898908d` = origin, clean. **Note:** the brief says HEAD `216c1ba`; `1c191acf` is one README-only commit after it (the v2 note at README line 10). Code is identical to `216c1ba`. |
| main clone `nanotron-kys` | `skipjack-1p5b-baselines` @ `bc131392`, not audited, not modified (queued jobs) |
| paper | the brief's "attached" PDF is not in this session. Used the local `projects/rewrite-3B/00_paper/Know_Your_Sources.pdf` (28 pages, "Under review … ICLR 2027", created 2026-09-26, sha256 `a7aed70a…b5`). Text extract: `prelaunch_audit_2026-11-08/evidence/paper_text.txt` (kept on shared storage only, **not committed**: the repo is public and the paper is under double-blind review). If the submitted PDF differs, paper-side findings must be re-checked. |

Evidence folder: `docs/prelaunch_audit_2026-11-08/` (`scripts/`, `logs/`, `evidence/`).

### Scope and planned checks

1. **Goal mapping (§1).** For each control, the chain from registry → templates → rendered configs → data → launcher
   → checkpoints → HF conversion/upload → eval → aggregate, naming the file that does each step.
2. **Data verification (§2).** Local raw parquet for `raw_diversity_oriented`, `raw_disagreement_aware` against
   `manifest.json` sha256 / row counts; λ0 tokenized shards against `SHA256SUMS` and the registry total; tokenizer.
3. **Protocol identity (§3).** The three settings' configs vs the published v2 configs (schedule, batch, seed, init,
   tokenizer, data convention).
4. **Output side (§4).** `results.json` schema and `aggregate.py` outputs: per-seed per-task values for the six
   tasks, Mean6, MMLU macro over 57 subjects and four categories; where anything is aggregated too early; whether
   the outputs can be paired within seed against the v2 results consumed by `paired_analysis.py`.
5. **Pairing and layout (§5).** Comparator map (λ0→disagreement_aware, raw_X→X, quality_base), HF upload layout
   vs kys-eval `hf_subfolder`, `reference_check.py` inputs and pass criteria, `paired_analysis.py` input layout.
6. **CPU tests (§6).** kys-eval selftest; synthetic end-to-end: fake 27-cell results → `aggregate.py` →
   paired analysis; config invariants.
7. **Gap table (§7)**, split into number-changing vs wording-only.
8. **Known deviations (§8):** documented adequately or need a sentence.
9. **Changes made (§9)**, final revisions, handoff prompt (§10).

## 2. Data verification — DONE

Script `prelaunch_audit_2026-11-08/scripts/verify_data.py` and `verify_small_files.py`; logs in `logs/`, results in
`evidence/verify_data.json` and `evidence/verify_small_files.json`. Hub access used the cached login only; nothing
was uploaded.

| check | result |
|---|---|
| Hub `blab-jhu/KYS-Pre-Rewritten` revision | `afd1f177` (2026-11-07, README-only after `b264b0ce`; data unchanged since `8a394d84`). WORKFLOW's "`b264b0ce` or later" holds. |
| local `manifest.json` vs Hub `manifest.json` @ `afd1f177` | byte-identical |
| `raw_text/raw_diversity_oriented/` local | 16/16 parquet: sha256, bytes and row counts = manifest; Hub LFS sha256 = manifest; 7,138,615 rows = `final_merged_docs` |
| `raw_text/raw_disagreement_aware/` local | 16/16 parquet, same checks pass; 7,018,928 rows |
| local tokenized `raw_diversity_oriented` (made from that raw text earlier) | 16 shards; Σ `.ds` bytes / 2 = Σ metadata = **10,000,002,885** = registry |
| local tokenized `raw_disagreement_aware` | 16 shards; both sums = **10,000,005,069** = registry |
| λ0 tokenized, local `tokenized.SHA256SUMS` | 78/78 local files match; local SUMS = Hub `SHA256SUMS` (same 78 names and hashes) |
| λ0 on the Hub | 79 files; 52 LFS sha256 = SUMS; the 26 `.ds.metadata` downloaded: 26/26 = SUMS |
| λ0 totals | 26 shards; Σ bytes / 2 = Σ metadata = **10,000,002,556** = registry |
| tokenizer | local and Hub `tokenizer.json`, `tokenizer_config.json`, `tokenizer_report.json` sha256 = manifest |

Order of λ0 steps is consistent: the SUMS cover the `.ds.metadata` files, and `fix_ds_metadata.py` rewrites
metadata line 1 only, so `sha256sum -c` must run before it (WORKFLOW §5 line ~446 and RUNBOOK line ~151 do this).
Not re-checked here: the raw-text → tokens step on the collaborator's side (it is checked there by
`tokenize_raw_text.sh`'s exact-total assert); the published init weights (checked 2026-11-07, audit F6).

## 4. Output side — DONE

Script `scripts/output_side.py`, log `logs/output_side.log`, outputs `evidence/output_side/`. It takes the 54 real
v2 LightEval JSONs (the files `paired_analysis.py` uses) and runs them through kys-eval's own code:
`eval_checkpoint.package_existing` → `common.build_results` (the same function `evaluate()` calls after LightEval)
→ `results.json` → `aggregate.py`.

| # | check | result |
|---|---|---|
| 1 | kys-eval MMLU subject→category map vs `08_evaluation/02_code/mmlu_categories.py` (the paper's) | identical, 57 subjects, 18/13/12/14 |
| 2 | kys-eval scores from the raw JSONs vs `paired_analysis` `per_checkpoint_scores.csv`, 54 × 12 values | max diff 5.0e-5 points (= the CSV's 4-decimal rounding) |
| 3 | `kys-eval/reference/rewritten_v2_54_per_task.csv` (the comparator `aggregate.py` uses) vs the v2 JSONs | 3,402 task cells, acc and acc_norm, max diff 5e-11; steps match |
| 4 | identity: each of the 27 rebuttal cells built from its comparator's v2 JSON → `aggregate.py` | Δ = 0 with `n_paired` = 3 in all 108 rows (3 pairs × 3 epochs × 12 benchmarks); no non-rebuttal pairs |
| 5 | non-trivial: λ0 cells built from `quality_base`'s JSONs → `aggregate.py` Δ (disagreement_aware − λ0) vs `paired_analysis` DA − QB | Mean6 and MMLU, 3 epochs: max diff 0.0032 points (reference has 2 decimals), seed counts equal; per-task ep3 (6 tasks + 4 categories): max 0.0047 |

**Answers to brief item 2.**
- `results.json` keeps `acc`, `acc_norm` and their stderr for all 63 tasks (`tasks`), plus `scores` with the six
  tasks, Mean6, MMLU macro over the 57 subjects (not LightEval's `_average`) and the four category means. Every
  aggregate is computed per checkpoint from full-precision subject values (`common.benchmark_scores`).
- Nothing is aggregated too early. `aggregate.py` pairs per seed, then takes mean ± SD (ddof 1) of the three
  differences and counts positive seeds — the Appendix F definition. The CSVs write 6 decimals of a fraction
  (= 4 decimals in points); the paper prints 2.
- `raw_selected_per_checkpoint.csv` has one row per (setting, seed, epoch) with the 12 values, which is the
  shape of `paired_analysis`'s `per_checkpoint_scores.csv` (fractions instead of points).
- Gaps found here are in §7: G-O1 (no pair against `quality_base` for the new settings), G-O2 (sign
  convention is comparator − new, the opposite of Appendix F's strategy − baseline), G-O3 (`paired_analysis.py`
  cannot read kys-eval's output layout as is).

## 3. Protocol identity vs the released v2 configs — DONE

Script `scripts/protocol_identity.py`, log `logs/protocol_identity.log`, table `evidence/protocol_identity.md`,
raw diffs `evidence/protocol_identity.json`; the 27 v2 `config.yaml` + `checkpoint_metadata.json` downloaded to
`evidence/v2_configs/` from `wytro/KYS-1.5B-Rewritten-v2@b22c8733`.

For every seed × epoch, each new setting's committed ep config (huicong-dev `6fe10b9b`) was flattened (71 keys) and
compared with the v2 config of its comparator and of `quality_base`, excluding the five run/path keys (run name,
checkpoint paths, dataset folder, tokenizer path):

| comparison | differing keys |
|---|---|
| `raw_diversity_oriented` vs v2 `diversity_oriented`; `raw_disagreement_aware` and λ0 vs v2 `disagreement_aware` (27 rows) | only `parallelism.recompute_layer` (placeholder, filled per cluster; v2 had false). Speed only, no math change. |
| each new setting vs v2 `quality_base`, seeds 43 and 44 | same: only `recompute_layer` |
| each new setting vs v2 `quality_base`, seed 42 | also `micro_batch_size` 32 vs 16 and `batch_accumulation_per_replica` 8 vs 16 (known deviation 1) |

Model, optimizer, LR schedule (warmup, WSD decay, steps), seed, global batch 1024 × 2048, dp 4, tokenizer and
sequence length are equal. The 34 keys present only in the v2 dumps are defaults nanotron writes when it serializes
a config (`dataset_read_path`, `shuffle_files`, …); that the loader fills them with the v2 values was checked on
2026-11-07 (`reports/CONFIG_COMPARISON.md` part C, audit 2026-11-07 Part B). v2 finals record
`consumed_tokens_total` = step × 2,097,152 (e.g. 9,999,220,736 at 4768), the same tokens per step as the registry.

Not covered by a config diff: λ0 seed 44 at dp 8 / accum 4 (WORKFLOW §1b wave 2), which keeps the global batch and
mbs but changes the reduction order; see §8.

## 1. Goal mapping — DONE

Each step names the file that does it (paths relative to nanotron `huicong-dev` unless prefixed `kys-eval/`).

| step | goal 1: `raw_diversity_oriented`, `raw_disagreement_aware` | goal 2: `signal-disagreement-lambda0` |
|---|---|---|
| definition | `tools/kys_raw/registry.py` (family `strategy_linked`, comparator = the rewritten arm, total from `manifest.json`) | `registry.py` (family `lambda_grid`, comparator `disagreement_aware`, `expected_corpus` = 10,000,002,556 / 26) |
| data | Hub `raw_text/<s>/` 16 parquet (§2 verified) → `tools/kys_raw/tokenize_raw_text.sh` (manifest sha256 check, datatrove, `</s>` per doc, 16 tasks, exact-total assert) → `tools/fix_ds_metadata.py` | Hub `signal-disagreement-lambda0/tokenized/` 26 shards (§2 verified) → `sha256sum -c SHA256SUMS` → `tools/fix_ds_metadata.py` |
| source selection | KYS-Pre-Rewritten `manifest.json` `source_selection` / `subsampling` (seed-42 uniform 5B of the counterpart's successful-rewrite sources) + shared anchor; audited 2026-09-28 (`reports/strategy_linked_audit/`) | `06_lambda_grid/select_lambda_grid.py` re-ranks the fixed feasible domain U_τ (floor Q30(q), cap Q90(v)) with u = q + 0·√v, gated byte-exact against the saved λ ∈ {0, 0.5, 1} ids. **Checked today (below).** |
| configs | `configs/1.5B-baseline-seed<S>/<s>_seed<S>_<kind>.yaml` (6 per seed); `tools/kys_raw/fill_placeholders.py --settings` renders per cluster | same files for λ0 (identical to `raw_random`'s after renaming, 18/18) |
| invariants | `tools/assert_invariants.py` (corpus total, tokens/step, batch, placeholders; `--check-resume`) | same, with the registry total |
| launch | `tools/kys_raw/plan_submit.py --settings` → `deploy/slurm/kys_segment.sbatch` (B1 restart copy via `tools/kys_raw/trunk_restart_config.py`) | same; seed 44 optionally dp 8 (`marc-cluster-8gpu`, `filled-dp8/`) |
| finals | `<ckpt_root>/seed<S>/<s>/ep<N>/<s>/seed<S>/{4768,9537,14305}` | same |
| HF export | `tools/kys_eval/convert_to_hf.py` (byte-identical to the v2 converter `08_evaluation/02_code/convert_to_hf.py` except its usage line; **round trip checked today, §5**) | same |
| upload | WORKFLOW §11: `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` `rewrite-1p5b/seed<S>/<s>/ep<N>/{hf,nanotron}/` | same |
| scoring | `kys-eval/kys_eval/run_grid.py` + `slurm/eval_grid_array.sbatch` (cell = `hf://…/rewrite-1p5b/seed<S>/<s>/ep<N>/hf`), after `reference_check.py` | same |
| pairing | `kys-eval/kys_eval/aggregate.py` → `raw_vs_rewritten.{md,csv}`, Δ = comparator − new, per seed; comparator values from `reference/rewritten_v2_54_per_task.csv` (= the v2 JSONs, §4) | λ0 ↔ `disagreement_aware` (`config.py` `LAMBDA_GRID_TO_REWRITTEN`) |
| paper format | the user's `docs/rebuttal_q1_paired_analysis/paired_analysis.py` (reads v2 LightEval JSONs; must be extended, G-O3) | same |

**Verdict per goal.**
- **Goal 1:** the code and data produce the two raw controls exactly as specified (equal-token-budget, same anchor,
  same protocol as v2; §2, §3). The pairing raw_X ↔ X is implemented. What is not produced automatically: the
  pair against `quality_base` (G-O1) and the Appendix-F-format table (G-O3).
- **Goal 2:** produced as specified. New evidence today (`scripts/lambda0_sources.py`, `evidence/lambda0_sources.json`):
  every one of the 5,532,945 distinct source documents of the λ0 rewritten half (8,446,062 rows: 5,527,293
  wikipedia + 2,918,769 distill) lies in the λ = 0 selection over U_τ (10,435,667 docs; selection = 5,542,662 docs,
  99.82% covered, the rest failed both passes); the stored `u_score` equals q exactly; min q = 0.8146 ≥ τ_q =
  0.6957 and max v = 0.0659 ≤ τ_v = 0.1036; 4,955,869 of the sources (89.6%) are also in the λ = 0.5 selection;
  `shuffled/` = the 4,120,164 anchor rows + exactly these rewritten rows. Keyed by the scored-pool `doc_id`
  (`orig_doc_id` is the 100M-reservoir position; a first run keyed by it is kept as `logs/lambda0_sources_origid.log`).
  The rewrite logs show the same rewriter and settings for λ0 (job 1734978, 2026-07-10) and λ0.5 (job 1519364,
  2026-06-06): Qwen2.5-7B-Instruct, vLLM 0.22.0, `mode=grounded`, same engine args, `temperature=0` in
  `07_rewrite/rewrite_worker.py:296`. The prompt file's content at each date is not recorded (the launcher points
  at one path, `00_TMP/wikipedia_style_rephrasing_grounded.md`).

## 5. Pairing and HF layout — DONE

**Comparator map** (`kys-eval/kys_eval/config.py:221-235`): `raw_diversity_oriented` → `diversity_oriented`,
`raw_disagreement_aware` → `disagreement_aware`, `signal-disagreement-lambda0` → `disagreement_aware` (v2 run stem
`signal-disagreement-lambda05`). Same as nanotron `registry.py`. `quality_base` is a comparator only for the three
global Top-10B settings, not for the rebuttal settings (G-O1).

**Upload layout vs what kys-eval reads.** WORKFLOW §11 uploads the export to
`rewrite-1p5b/seed<S>/<setting>/ep<N>/hf/`; kys-eval's `config.hf_subfolder` reads exactly
`rewrite-1p5b/seed<S>/<setting>/ep<N>/hf` from `HF_REPO = blab-jhu/KYS-1.5B-Raw-Selected-Baselines`. The v2 repo
uses the same layout with the same five files.

**Conversion round trip** (`scripts/convert_roundtrip.py`, `scripts/convert_env_check.py`, evidence
`convert_roundtrip.json`, `convert_env_check.json`). The released v2 nanotron checkpoint seed 42 /
`diversity_oriented` / ep3 was converted on CPU with `tools/kys_eval/convert_to_hf.py` and the KYS-Pre-Rewritten
tokenizer:
- in the kys-eval venv (transformers 5.6.2, the version the v2 exports record): `model.safetensors` sha256
  `733262ae…3f45` = the released file = the reference check's expected hash; all 254 tensors equal; `config.json`,
  `generation_config.json`, `tokenizer.json` byte-identical; `tokenizer_config.json` differs only in
  `local_files_only` (true in v2, false here), a load flag that `save_pretrained` records; `validate_model_dir` clean.
- in the nanotron training env (`envs/pretrain`, transformers 4.46.3, the INSTALL.md pin): weights byte-identical
  again; `config.json`, `generation_config.json`, `tokenizer_config.json` differ in bytes and an extra
  `special_tokens_map.json` is written, but loaded with transformers 5.6.2 (the scoring env) the config is equal
  key by key, the tokenizer class, special tokens, `add_bos_token`/`add_eos_token`, padding side and max length are
  equal, and 22 test strings (×2 special-token modes, plus batch padding) encode identically.
- So the export step reproduces the v2 artefact regardless of which of the two environments is used. The docs do
  not say which environment to convert in (G-D4, wording).

**Reference check** (GPU; **not run here**). Command (WORKFLOW §14 / audit F12):
`python -m kys_eval.reference_check --model hf://wytro/KYS-1.5B-Rewritten-v2/rewrite-1p5b/seed42/diversity_oriented/ep3/hf@b22c87333783945e1c655ae46886b7e842ee4203`.
Inputs: that public export (verified: its `model.safetensors` = `733262ae…`, the hash in
`reference/expected_diversity_oriented_seed42_ep3.json`). Pass criteria (`reference_check.py:1-20`): weight hash
equal; for all 64 tasks the four LightEval hashes (examples, full prompts, input tokens, continuation tokens) equal;
per-task acc/acc_norm within 0.005 and Mean6/MMLU within 0.001 (exact expected on H100). It validates the eval
stack on the collaborator's GPU, not the new checkpoints; the round trip above covers the export step that the
reference check skips.

**Hub access.** `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` (1 file, `.gitattributes`, sha `10a1133b`) and
`wytro/Know-Your-Sources-init` are **gated** (auto-approval); an anonymous download of each fails with
`GatedRepoError` (tested with no token). `blab-jhu/KYS-Pre-Rewritten` and `wytro/KYS-1.5B-Rewritten-v2` are public,
not gated (the brief calls KYS-Pre-Rewritten gated). WORKFLOW §0C says "Reading needs none: all inputs are public",
and the kys-eval README says the raw-selected repo is public (G-D1).

## 6. CPU tests at the current revisions — DONE

| test | command / script | result |
|---|---|---|
| full `assert_invariants.py` on the 54 rebuttal configs, rendered at `6fe10b9b` with `fill_placeholders.py --settings raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0` against the local data (verified in §2) | `scripts/assert_54.sh <envs/pretrain python>`; logs `logs/assert54/` | **54/54 pass**; totals 10,000,002,885 / 10,000,005,069 / 10,000,002,556 exact; 2,097,152 tokens/step. (A first run failed 36/54 because the test data root lacked `manifest.json`: a setup error of the audit, fixed by linking the verified manifest; the failing logs were overwritten by the rerun.) |
| kys-eval selftest at `1c191acf` | `python -m kys_eval.selftest` | passed (specs/grid, E-F1/E-F2 checks, aggregation reproduces the v2 tables, reference comparison) — `logs/kys_selftest.log` |
| 27-cell list | `run_grid --list --settings raw_diversity_oriented raw_disagreement_aware signal-disagreement-lambda0` | 27 cells, `--array=0-26`, indices 0–8 / 9–17 / 18–26 = seeds 42 / 43 / 44 — `evidence/list27.txt` |
| output side end to end | `scripts/output_side.py` (§4) | 8/8 pass |
| HF export round trip, both envs | `scripts/convert_roundtrip.py`, `scripts/convert_env_check.py` (§5) | weights byte-identical to v2; loads identically |
| λ0 provenance | `scripts/lambda0_sources.py` (§1) | pass |
| GPU reference check | — | **not run** (no GPU on this allocation); command and criteria in §5 |

## 7. Gaps, inconsistencies, missing steps — DONE

### 7a. Could change a reported number

| id | location | expected | what is there | severity |
|---|---|---|---|---|
| G-N1 | WORKFLOW §1b, RUNBOOK "Order of work" | every new run at the v2 geometry (dp 4 / mbs 32 / accum 8) | λ0 **seed 44** may run at dp 8 / accum 4. Same global batch and mbs, so no objective change, but a different reduction order; WORKFLOW §1b says "no bitwise claim across dp". It enters one of the three paired differences of λ0 vs `disagreement_aware`. | Low. Choose dp 4 if the schedule allows; otherwise report it with the run (missing from the brief's deviation list). |
| G-N2 | WORKFLOW §13 | — | The 27 new checkpoints are trained on the collaborator's cluster, the v2 comparators on ours. §13 promises no bitwise reproducibility across GPU models/drivers. Paired within-seed differences therefore mix hardware with the treatment. Inherent; not in the brief's deviation list. | Low (document) |
| G-N3 | `kys-eval/kys_eval/common.py` GPU check (`REQUIRED_GPU_SUBSTRING = "H100"`) | score on the v2 device, H100 80GB HBM3 (SXM) | any GPU name containing "H100" passes silently (NVL/PCIe); a non-H100 needs `--allow-any-gpu` and is flagged. `config.py` records NVL vs SXM as 9/10 identical, 1 example flip. | Low (E-F5, open) |
| G-N4 | `kys-eval/kys_eval/eval_checkpoint.py` `identity_mismatch` | a stale result is never reused | a checkpoint **re-uploaded** to the same path and revision passes the identity check; only `--force` re-scores it. `results.json` records `weights_sha256`, so a post-hoc check is possible. | Low (operational; E-F1 residual) |
| G-N5 | λ0 rewrite provenance | λ0 rewrites made with exactly the λ0.5 pipeline | same rewriter, vLLM version, mode, engine args and `temperature=0` per the logs; the grounded prompt file is referenced by one path and its content on 2026-07-10 vs 2026-06-06 is not recorded | Low (provenance; cannot be closed from records) |

No defect was found that changes a number when the documented procedure is followed: data (§2), configs (§3),
export (§5) and scoring/aggregation (§4) reproduce the v2 path exactly where it could be tested on CPU.

### 7b. Missing outputs for the rebuttal analysis (numbers right, but not produced)

| id | location | expected | what is there | severity |
|---|---|---|---|---|
| G-O1 | `kys-eval/kys_eval/aggregate.py` `write_pair_report`, `config.RAW_COMPARATOR` | paired Δ of each new setting against `disagreement_aware`, `diversity_oriented` **and `quality_base`** (brief) | one comparator per setting: raw_DO↔DO, raw_DA↔DA, λ0↔DA. No pair against `quality_base`, none between new settings. `raw_selected_per_checkpoint.csv` + `reference/rewritten_v2_54_per_task.csv` contain everything needed. | Medium: do it in the extended `paired_analysis.py` |
| G-O2 | `aggregate.py` (Δ = comparator − new; "Seeds Δ>0" = seeds where the comparator wins) vs paper Appendix F (setting − Quality-Base, "seeds in which a setting exceeds QUALITY-BASE") | one sign convention in the rebuttal tables | opposite orientation from Appendix F. Not wrong, but a table copied from `raw_vs_rewritten.md` next to Appendix F would flip signs and seed counts. | Low (sign/wording) |
| G-O3 | `docs/rebuttal_q1_paired_analysis/paired_analysis.py` `load()` | reads the collaborator's returned results | reads raw LightEval JSONs at `08_evaluation/04_results_1p5_v2/downstream/<run>/<step>/results/**/results_*.json` and requires `model_name` to end in `/rewrite-1p5b-hf/<run>/<step>`. kys-eval writes `results/raw_selected/seed<S>/<setting>/ep<N>/results.json` (+ `lighteval_raw/**/results_*.json`, whose `model_name` is the download path `…/model_cache/blab-jhu/KYS-1.5B-Raw-Selected-Baselines/rewrite-1p5b/seed<S>/<setting>/ep<N>/hf`). The extension needs a second loader; the identity check should use `results.json` `cell` + `model.commit` + `model.weights_sha256` instead of `model_name`. | Medium (analysis-side) |
| G-O4 | WORKFLOW §14 last lines | one place for the 27 results | "Upload `results/` and `reports/` to `rewrite-1p5b/seed<S>/eval/` after each seed (for the rebuttal set, once its 27 cells are scored)": for the rebuttal set the seed folder is undefined (results/ holds all three seeds). | Low (wording) |

### 7c. Wording / documentation only

| id | location | expected | what is there | severity |
|---|---|---|---|---|
| G-D1 | WORKFLOW §0C ("Reading needs none: all inputs are public"); kys-eval `README.md:57` ("The raw-selected HF repo is public") | access requirements stated | `wytro/Know-Your-Sources-init` and `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` are **gated** (auto-approve); anonymous downloads fail (`GatedRepoError`). The collaborator must accept both and be logged in on training and eval nodes. Fails loudly, no wrong number. | Medium (operational) |
| G-D2 | WORKFLOW §1b ("always pass `--settings` … Without it, the script plans all eight settings of the seed") | current behaviour | since `28a205f2` both tools refuse to run without `--settings` | Low |
| G-D3 | `kys-eval/kys_eval/config.py:8-11` docstring; kys-eval `README.md:137`; `reference/expected_diversity_oriented_seed42_ep3.json:17`; HF `wytro/KYS-1.5B-Rewritten-v2` README (lines 35–41) | v2 = source of the ICLR tables; reference checkpoint public | still say v2 ≠ the paper's tables / the checkpoint is not on the public Hub | Low (the model card is the most visible) |
| G-D4 | WORKFLOW §11, RUNBOOK "Deliverables" | the environment for `convert_to_hf.py` | not stated; §5 shows both candidate envs give byte-identical weights and identically loading configs, so this is informational only | Note |
| G-D5 | the brief | — | `tools/kys_raw/assert_invariants.py` is `tools/assert_invariants.py`; `tools/kys_raw/fix_ds_metadata.py` is `tools/fix_ds_metadata.py`; KYS-Pre-Rewritten is public, not gated; the registry holds an expected total only for λ0 (raw totals come from `manifest.json`); kys-eval `main` is `1c191acf` (README-only after `216c1ba`) | Note |
| G-D6 | RUNBOOK λ0 section, WORKFLOW §2d | rewrite decoding stated | greedy decoding / no sampling seed is not mentioned; λ0's rewrites were generated separately (July 2026) with the same pipeline | Low |
| G-D7 | `tools/assert_invariants.py:78-79` comment (D-F1, open) | the check described = the check done | comment claims a bytes/2 cross-check the code does not do. Today Σ bytes/2 = Σ metadata again for all 58 shards of the three corpora (§2). | Low |
| G-D8 | paper §3 and Fig. 1 caption (`evidence/paper_text.txt` lines ~199, ~396: "We exclude this sample from subsequent training-data construction") | the 5M analysis sample is in every corpus at the chance rate | the paper says excluded; WORKFLOW (lines 50, 268-274) already states the truth | Low (paper; known deviation 4) |

## 8. Known deviations — DONE

| # | deviation (brief) | where documented | verdict |
|---|---|---|---|
| 1 | Quality-Base seed 42 mbs 16 | WORKFLOW §13 (with the instruction to add a seeds-43/44-only check when comparing against Quality-Base), RUNBOOK step 3, README; §3 here confirms it is the only config difference | adequate. When G-O1 adds Quality-Base pairs, carry the footnote into the rebuttal table. |
| 2 | equal-token-budget, not identical-document | RUNBOOK intro, WORKFLOW §2a (with per-setting retention 51.4% / 51.8%), README, Hub `manifest.json` `role` | adequate |
| 3 | `quality_base` as Quality-First's raw counterpart, top-ranked not random | WORKFLOW §2a ("fastText-best 5B of Quality-First's own rewriting input … slightly stronger, unconditioned raw side") | adequate; one sentence would help: "so it is not built like the two new raw controls (a random 5B of the rewritten sources)" |
| 4 | 5M analysis sample not excluded | WORKFLOW lines 50 and 268-274 state the truth (kept, 5.00% of every corpus) | adequate for the operators; the paper text needs the correction (G-D8) |
| 5 | greedy decoding, no sampling seed | not in RUNBOOK or WORKFLOW | **needs a sentence** in the λ0 section (G-D6): rewrites were produced once with `temperature=0` (vLLM 0.22.0, Qwen2.5-7B-Instruct) and are shipped as data; they are not regenerated |
| 6 | wave-1 seeds concurrent | RUNBOOK "Order of work", WORKFLOW §1 and §1b | adequate (concurrency does not affect any run's computation) |
| + | λ0 seed 44 at dp 8 (G-N1) | WORKFLOW §1b, RUNBOOK | documented for the operator; **add to the rebuttal's deviation list** if used |
| + | different training cluster / GPUs than v2 (G-N2) | WORKFLOW §13 (generic) | **add a sentence** to the rebuttal's deviation list |
| + | scoring GPU variant (G-N3) | kys-eval `config.py` comment | report the GPU per cell (`results.json` `environment.gpu`) |

## 9. Changes made, inaccessible resources, final revisions — DONE

**Code or documentation changes:** none to nanotron `configs/`, `tools/`, `deploy/` or to kys-eval. The only
commit is this report and its evidence folder (documentation only, `docs/`). **No training-side fix is awaiting
approval.** Every gap in §7 is left for the user to decide; none blocks launch.

**Not accessible / not done:**
- the "attached" submission PDF (the local `rewrite-3B/00_paper/Know_Your_Sources.pdf` was used, §0);
- the collaborator's cluster, its GPUs and its HF account (gated-repo access and write access to the destination
  cannot be checked from here);
- a GPU: the reference check and any LightEval run were not executed;
- the content of the grounded rewrite prompt file as of 2026-06-06 and 2026-07-10 (G-N5);
- the raw-text → tokens step was not re-run; the local tokenized copies made earlier from the verified raw text
  were checked instead (§2).

**Local-only artefacts (not committed):** `evidence/paper_text.txt` (submission text); `evidence/hub_small/tokenizer/` (copies of the public tokenizer, hashes recorded in `evidence/verify_small_files.json`); `evidence/output_side/*/results/` (the synthetic 27-cell results trees, re-created by `scripts/output_side.py`; the generated `reports/` are committed); and `docs/prelaunch_audit_2026-11-08/work/` (12 GB): the v2 seed-42
`diversity_oriented` ep3 nanotron model + HF export downloaded for §5, the two converted exports, and the §6 test
data root (symlinks) and 54 rendered configs. Re-creatable with the scripts; safe to delete.

**Revisions at the end of the audit:** nanotron `huicong-dev` = `6fe10b9b` + one docs-only commit adding this report (its parent is `6fe10b9b`; `git log -1 -- docs/prelaunch_audit_2026-11-08.md`);
kys-eval `main` = `1c191acf` (unchanged); main clone `nanotron-kys` untouched.

## 10. Summary and handoff — DONE

- **Goal 1** (raw controls): produced as specified — `tools/kys_raw/registry.py`, the configs, the verified Hub
  raw text, `tokenize_raw_text.sh`, `kys-eval` pairing raw_X ↔ X. **Goal 2** (λ0): produced as specified, and its
  sources were shown today to be the λ = 0 selection over the same floor/cap domain.
- **Output side:** per-seed, per-task values for the six tasks, Mean6, MMLU macro and four categories are in
  `results.json` and `raw_selected_per_checkpoint.csv`; nothing is aggregated before pairing. Tested end to end on
  the real v2 JSONs against the existing paired analysis (§4).
- **Pairing/layout:** comparator map, upload path and kys-eval's read path agree; the export step reproduces the v2
  artefact byte for byte (weights) in either environment.
- **Open items to decide** (none number-changing under the documented procedure): G-O1 Quality-Base pairs, G-O2
  sign convention, G-O3 paired-analysis loader, G-D1 gated repos documented as public, G-N1 λ0 seed 44 dp 8, and the
  wording items G-D2…G-D8. Deviations needing a sentence: greedy decoding (5), plus λ0 seed 44 dp 8 and
  training-hardware difference for the rebuttal's own list.
- **Handoff prompt for the final independent review:** `prelaunch_audit_2026-11-08/HANDOFF_GPT6.md`.

## 11. Addendum (2026-11-08, later) — analysis gaps G-O1, G-O2, G-O3 closed

| gap | change | commit |
|---|---|---|
| G-O1 (no Quality-Base pairs) | kys-eval `aggregate.py`: every setting whose primary comparator is not `quality_base` (the four strategy-linked controls and λ0) is also paired with `quality_base` (`config.APPENDIX_F_BASELINE`); `raw_vs_rewritten.csv` gains a `comparison` column (`primary` / `quality_base`) | kys-eval `main` **`668a9dd3e37caecc007a7a35629238ee4ccd5852`** (pushed) |
| G-O2 (sign convention) | same commit: Δ = setting − comparator per seed; "Seeds Δ>0" = seeds in which the setting is higher (Appendix F). `rewritten_*` CSV columns renamed `comparator_*` so old consumers fail loudly instead of reading flipped signs. README and the report text updated; selftest updated (13 pairs × 3 × 12) and passing | same |
| G-O3 (paired analysis cannot read kys-eval output) | `projects/rewrite/docs/rebuttal_q1_paired_analysis/paired_analysis.py --kys-results ROOT [--out-dir]`: loads kys-eval `results.json` files for the 27 rebuttal cells, each identified by `rewrite-1p5b/seed<S>/<setting>/ep<N>/hf` (stored cell, model spec/subfolder and file path must agree; complete, unique, all 27 required); scores computed with the v2 arithmetic; pairs each rebuttal setting − its comparator and − `quality_base` in the same CSV / LaTeX / LOTO / 2d outputs. Step 1 and ranks stay on the six v2 settings | `projects/rewrite` `master`: **`ecf7d74`** (the 2026-11-06 folder committed unchanged, so the change is a diff) and **`c639080`** (the loader). **Not pushed: the `projects/rewrite` repository has no git remote.** |

kys-eval scoring (`eval_checkpoint`, `run_grid`, the protocol) is unchanged; the collaborator's requirement
"kys-eval `216c1ba` or later" in WORKFLOW/RUNBOOK still holds and was not edited. No training-side file changed.

**Identity test against the 54 v2 files** (`prelaunch_audit_2026-11-08/analysis_gaps/test_analysis_gaps.py`, log
`test_run.log`, results `test_results.json`; work trees in `analysis_gaps/work/`, not committed). The 27 rebuttal
cells were filled with their primary comparator's v2 LightEval results through kys-eval's `package_existing` →
`build_results`, once in the `rewrite-1p5b/seed<S>/<setting>/ep<N>/hf/` layout and once in kys-eval's native
`raw_selected/` tree with the full model provenance dict. 17/17 checks pass:
- regression: `paired_analysis.py` without `--kys-results` writes all 13 outputs byte-identical to the committed
  2026-11-06 files (`analysis_gaps/regression/`, local);
- A1 every rebuttal cell's 12 scores equal the comparator's bit for bit; A2 the 15 v2 pairs' rows, the ranks and
  Step 1 are unchanged; A3 setting − primary comparator is 0.00 / 0.00 / 0/3 in all 222 added CSV rows; A4 setting −
  Quality Base equals the v2 row comparator − Quality Base field for field; A5 `tables.tex` has the 6 added pairs in a
  third block of both pairwise tables; B the native tree gives identical outputs;
- C the loader refuses (exit 1, with the reason) a missing cell, a wrong step, a wrong model subfolder, an incomplete
  file, a file under another cell's path, and a duplicate;
- D kys-eval `aggregate.py` on the same cells: primary Δ = 0 (|Δ| < 1e-9) with n = 3; Quality-Base Δ (setting − QB)
  equals `paired_analysis`'s comparator − QB in mean, SD and positive-seed count for all 108 rows; the report states
  the orientation.

**New minor finding (G-O5, Note).** In the identity test, 78 of kys-eval's 108 primary rows count a "positive" seed
although the true difference is 0: `aggregate.py` takes the comparator from `reference/rewritten_v2_54_per_task.csv`
(10 decimals) but the new cell from full-precision JSON, so exact ties appear as ±1e-11. Irrelevant for real
checkpoints (differences are ~1e-3), but it means kys-eval's seed counts are not trustworthy for ties at the 1e-10
level; `paired_analysis.py` reads the raw JSON on both sides and has no such artefact. No change made.

Revisions after this addendum: nanotron `huicong-dev` = the commit adding this addendum (docs only); kys-eval `main` =
`668a9dd3`; `projects/rewrite` `master` = `c639080` (local only).

## 12. Addendum (2026-11-08, second review round) — Issues A–D and documentation

Repair round after GPT-6's independent review. Same rules as before: read-only on data, CPU only, no training-side
change. **Correction to §11:** the identity test has **16** checks, not 17 as written there; all 16 passed then and
pass now.

### Issue A — comparator coverage: closed (no further change)

Closed by kys-eval `668a9dd3` (each setting also paired with `quality_base`; Δ = setting − comparator) and
`projects/rewrite` `c639080` (`paired_analysis.py --kys-results`). Identity test (`analysis_gaps/`): the 27 rebuttal
cells built from their comparators' v2 JSONs reproduce the comparators bit for bit, the 15 v2 pairs are unchanged,
setting − comparator is 0.00 / 0.00 / 0/3 in all 222 added rows, setting − Quality Base equals comparator − Quality
Base field for field, six malformed-input cases are refused, and `aggregate.py`'s Quality-Base pairs equal the
paired analysis's in mean, SD and seed count. Backup of the analysis code: Item D below.

### Issue B — positive-seed counts near zero: fixed (kys-eval `6da264bf`)

**Diagnosis confirmed.** In `reference/rewritten_v2_54_per_task.csv` every `acc`/`acc_norm` has exactly 10
decimals; against the 54 v2 LightEval JSONs only 642 of 3,456 task values are bit-identical and the largest error is
5.0e-11. LightEval and kys-eval (`common.write_json`) both serialize floats with Python's `json` (shortest round-trip
repr), so the new cells are full precision while the comparator was not. An exact tie became ±1e-11 and was counted as
a positive or negative seed (78 of 108 primary rows in the round-1 identity test).

**Fix** (kys-eval `main` **`6da264bf8665b3cfbbd0736a2c24259d138511ba`**, pushed):
- `reference/rewritten_v2_54_per_task.json`: the 54 cells × 64 task keys copied from the v2 JSONs without rounding,
  with each source file's sha256, GPU and LightEval sha; built by the new `reference/build_rewritten_v2_reference.py`,
  which also checks that every CSV value is the `.10f` rounding of the stored one (3,456/3,456).
- `aggregate.load_rewritten` reads only that file (schema-checked; passing the CSV is refused). **No tolerance:** both
  sides hold the same IEEE doubles and `benchmark_scores` combines them in a fixed order, so identical inputs give
  exactly 0.0. The reasoning is in the docstrings of `load_rewritten` and `config.REWRITTEN_V2_PER_TASK_JSON`.
  `--rewritten-csv` became `--rewritten-reference`; the CSV stays as a readable copy.
- Scoring untouched: no change to `common.py`, `eval_checkpoint.py`, `run_grid.py`, `reference_check.py` or the
  protocol constants, so "kys-eval `216c1ba` or later" still holds.

**Tests.**
- New `selftest.test_pairing_statistics`:
  - every reference double survives JSON round trips unchanged (the `results.json` path);
  - the 10-decimal CSV is shown to differ (the hazard) and is refused as a reference;
  - per-seed offsets on every task give the expected values. Epoch 1, +0.01 / −0.02 / +0.03 (mixed signs): mean, ddof-1 SD
    and 2/3 positive seeds. Epoch 2, all 0 (exact ties): Δ = 0.0, 0/3. Epoch 3, +0.01 / 0 / −0.01: mean 0, SD 0.01, 1/3.
  - checked on all 12 benchmarks against values computed independently from the shifted tasks.
- `test_aggregation`: primary-pair Δ and SD are exactly 0.0 with 0 positive seeds (no tolerance), over 13 pairs × 3 × 12.
- Selftest passes (`issue_b/selftest_6da264bf.log`).
- Independent audit check (`issue_b/check_reference_json.py`, log alongside): the reference JSON equals the 54 v2 JSONs bit
  for bit in all 3,456 task entries, and 54/54 recorded source sha256 match the files.
- Identity test rerun with strict checks (`analysis_gaps/test_run.log`; round 1 kept as `test_run_round1.log`):
  16/16 pass; **0 of 108 primary rows** count a positive seed, and Δ = 0.0 and SD = 0.0 exactly.

### Item D — analysis code backed up (nanotron `91ae7f16`)

`docs/rebuttal_q1_paired_analysis/` in nanotron `huicong-dev` is a `git archive` copy of `projects/rewrite` commit
`c639080` (15 files, byte-identical: `paired_analysis.py`, README, all CSVs, `tables.tex`, …), plus `SOURCE.md`
naming the source commit and noting that the script resolves its inputs relative to `projects/rewrite`. **Note:** the
nanotron repository is public; that README contains a draft of rebuttal text (§5) and the 54 v2 per-checkpoint scores.

### Documentation (nanotron `91ae7f16`, kys-eval `6da264bf`)

| file | change |
|---|---|
| WORKFLOW §0C | inputs (`KYS-Pre-Rewritten`, `Know-Your-Sources-init` — verified public today, anonymous download OK — and `KYS-1.5B-Rewritten-v2`) need no token; `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` is gated (anonymous download still fails), so evaluation nodes need a login, and uploads need a token with write access to `blab-jhu` |
| WORKFLOW §1b | "Without it, the script plans all eight settings" → both tools refuse to run without `--settings` (wrong since `28a205f2`) |
| WORKFLOW §13 | new "Known deviations of the rebuttal set": λ0 seed 44 at dp 8 / accum 4; the training-hardware difference against the v2 comparators (stated as "the authors' cluster, dp 4 per their checkpoint metadata": the v2 GPU model is not recorded in the files I read) |
| RUNBOOK λ0 section | greedy-decoding sentence: rewrites made once, July 2026, same rewriter and vLLM 0.22.0 as λ = 0.5, `temperature=0`, no sampling seed, shipped as data, not bitwise reproducible |
| kys-eval `config.py` docstring, README (note at the top, §access, reference-check section, file table), `reference/expected_diversity_oriented_seed42_ep3.json` `note` | no longer say v2 is not the source of the paper's tables or that the reference checkpoint is not public; README says the raw-selected repo is gated and uploads need write access to `blab-jhu`; the JSON reference and its builder are documented |

Left alone as instructed: the HF model card of `wytro/KYS-1.5B-Rewritten-v2` (still says the v2 checkpoints are not
behind the paper's numbers) — **pending authorization**; the manuscript; the collaborator's workflow structure.

### Issue C — λ0 vs λ0.5 prompt identity: same prompt established from the outputs

Scripts in `prelaunch_audit_2026-11-08/prompt_identity/` (`compare_rewrites.py`, `controls.py`, `length_check.py`);
logs `logs/prompt_identity*.log`; results `evidence/prompt_identity.json`, `prompt_identity_controls.json`,
`prompt_identity_length.json`, per-pair table `prompt_identity_pairs.parquet`, 200 mismatch excerpts
`prompt_identity_mismatches.jsonl`. Nothing was regenerated; inputs were read only.

**1. Records.**
- `projects/rewrite` git: the prompt files have no history. `00_TMP/wikipedia_style_rephrasing_grounded.md` is
  untracked; the only commit touching `07_rewrite/` / `09_Distill/` is `242f72a`, an archive commit.
- File mtimes: the grounded prompt (597 bytes) and the rewrite logs all show 2026-08-20, after both runs (June 6 and
  July 10). That date is consistent with a copy to the current filesystem, so mtimes carry no information.
- SLURM / vLLM worker logs: only progress bars and engine arguments; no prompt text, hash or token counts. The engine
  arguments are the same for the two runs (§1).
- `07_rewrite/progress/*_worker*.json` record `total_input_tokens` per worker (e.g. λ0 worker 0: 1,180,008,146 over
  693,284 docs). These are sums over different document sets, so they cannot be compared without re-tokenizing both
  source sets; not used.
- `input_tokens_qwen`: **not present** in the published corpus. `wytro/Know-Your-Sources@9e5ff24`
  `disagreement_aware/` has the columns `doc_id, orig_doc_id, text, source_prompt, train_tokens` (`metadata.json`
  lists the same five). The local λ0 `rewritten/` files have no such column either.
- Neither prompt file's content as of the run dates can be recovered from records.

**2. Direct data check.** All rows of published `disagreement_aware/part_00000.parquet` (λ0.5; 222,953 wikipedia +
113,379 distill rows) were matched to the local λ0 corpus on (scored-pool `doc_id`, `source_prompt`):

| comparison | rows | exact match | mismatches: first divergence (median / p10 of the text) | diverge in first 50 chars | diverge at char 0 |
|---|---:|---:|---|---:|---:|
| **λ0 vs λ0.5, wikipedia** | 197,689 | **51.9%** | 43% / 11% | 0.41% | 0.05% |
| **λ0 vs λ0.5, distill** | 95,794 | **45.9%** | 47% / 14% | 0.22% | 0.001% |
| control S: `diversity_oriented` (job 1545473) vs λ0.5 (1519364), same prompt path, wikipedia | 5,498 | 55.7% | 44% / 12% | 0.35% | 0.02% |
| control S, distill | 2,126 | 49.9% | 49% / 15% | 0.19% | 0 |
| control D: λ0 wikipedia vs λ0 distill rewrite of the same document (different prompts) | 3,811 | **0.0%** | 0.1% / 0% | 90% | 48% |

- **Exact-match rate falls with output length**, identically shaped for the question and the same-prompt control
  (λ0.5 output length bins; λ0-vs-λ0.5 / control S):
  - wikipedia: <1k chars 84.9 / 86.0%, 1–2k 66.3 / 69.6%, 2–3k 50.7 / 56.0%, 3–4k 40.9 / 47.4%, 4–6k 32.9 / 37.5%,
    ≥6k 20.1 / 18.2%;
  - distill: 80.8 / 82.3%, 54.4 / 55.0%, 37.6 / 44.8%, 25.9 / 30.2%, 16.3 / 23.8%, 5.8 / 6.7%.
- **Mismatches are tail-level, not systematic.** They share a long prefix and then diverge (the median at 43–47% of
  the text). The earliest divergences are near-synonym word choices with the rest of the opening identical ("In her
  upbringing" / "In her formative years", "born Victor" / "born as Victor"), the signature of a near-tied greedy step
  under batch-dependent bf16 numerics. A prompt difference instead gives 0% exact and divergence at the start
  (control D).
- **Uniformity.** Across λ0's 200 rewrite shards the wikipedia exact rate ranges 28.8–74.0% (distill 26.6–73.6%),
  more spread than binomial noise; the per-worker rates are flat (wikipedia 49–54%, distill 44–48%). The spread is
  consistent with batch-level nondeterminism (each shard is one `llm.generate` call) and is only weakly related to
  mean length (Pearson r = 0.24). The lowest shard is far from the 0% a different prompt gives, so no shard or
  worker of either run used another prompt.
- **Residual.** Within length bins, λ0-vs-λ0.5 is 1–7 points below control S. That could come from different nodes
  (n15 in July vs n01 in June) or from the smaller control sample. It is not a prompt effect, which would be ~100
  points.

**3. Conclusion.** The λ0 and λ0.5 rewrites were produced with the same prompts. A document rewritten in both runs
gives the identical text about half the time, at the same rate and with the same divergence pattern as two
independent runs that share the prompt path; a prompt difference gives no identical text. What cannot be
established is a byte-level match of the prompt files, because their dated contents were not archived. For the
rebuttal's deviation list: *"λ0's rewrites were generated in a separate run (July 2026) with the same rewriter,
vLLM version, prompts and greedy decoding as λ0.5; because batched greedy decoding is not bitwise reproducible,
about half of the documents rewritten in both runs have identical text and the rest diverge partway through, as in
any re-run."* Recommended to add it there; it is not a defect.

### Checks that need Marc's cluster or a GPU (not run here)

1. **Reference check** on the scoring H100, before any scoring:
   `python -m kys_eval.reference_check --model hf://wytro/KYS-1.5B-Rewritten-v2/rewrite-1p5b/seed42/diversity_oriented/ep3/hf@b22c87333783945e1c655ae46886b7e842ee4203`.
   - Pass: `model.safetensors` sha256 = `733262aec4ca52fe94b69df0acf7ef1faa7cd0fc1f42ee79b91004a022db3f45`;
     the four LightEval hashes equal for all 64 tasks; per-task acc/acc_norm within 0.005 and Mean6/MMLU within
     0.001; exact match expected on H100 80GB HBM3.
   - Also run `check_install` (must end OK) and `selftest` there.
2. **Training side:** the INSTALL.md smoke checks, s/it, full `assert_invariants.py` on the rendered configs against
   their data, `tokenize_raw_text.sh`'s exact totals, and the λ0 `sha256sum -c`.
   - The first real trunk1 restart must log step+1 (B1).
3. **Hub access:** acceptance of the gated `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` on every scoring node, and a
   write token for uploads.
4. **Evaluation:** the 27-cell array with `KYS_GRID_ARGS` set, then `aggregate.py`.
   - Then `paired_analysis.py --kys-results` on the returned `results.json` files, on our side.

### Unresolved limitations

- Prompt-file content at the run dates (Issue C) is not archived. Output equivalence is established; byte identity of
  the files is not.
- The v2 training GPU model is not recorded in the files read.
- The HF model card of `wytro/KYS-1.5B-Rewritten-v2` still carries the stale v2 statement: **pending authorization**.
- The `projects/rewrite` repository still has no remote; nanotron `docs/rebuttal_q1_paired_analysis/` is the remote
  copy and must be re-copied after any change.
- Earlier open items unchanged: G-N3 (any "H100" accepted for scoring), G-N4 (re-upload to the same path needs
  `--force`), G-D7 (`assert_invariants.py` comment), G-O4 (eval upload folder wording), D-F1.

### Revisions at the end of this round

- nanotron `huicong-dev`: `91ae7f16` + the commit adding this addendum and the new handoff (docs only). Since
  `6fe10b9b`, no file under `tools/`, `deploy/`, `src/` or the seed config folders has changed; only `docs/` and the two
  `.md` files under `configs/1.5B-baseline/`.
- kys-eval `main`: `6da264bf`. Since `216c1ba`, no change to `common.py`, `eval_checkpoint.py`, `run_grid.py`,
  `reference_check.py`, `tasks/`, `patches/` or `slurm/`.
- `projects/rewrite` `master`: `c639080` (local only).
- **No training-side change awaits approval.**

### Note (2026-11-08, after §12)

README §5 (draft rebuttal text and caveats) was removed from the public copy `docs/rebuttal_q1_paired_analysis/` in
`ac3dfa2c`. The text remains readable in this public repository's history (commit `91ae7f16`). By the user's
decision the history is left as is: no rewrite, no force-push. The canonical copy in `projects/rewrite` is unchanged.

### Note (2026-11-08, round 3) — access wording made exact

Re-verified anonymously (`logs/access_check_round3.log`): `wytro/Know-Your-Sources-init` private=False,
gated=False, anonymous download OK; `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` gated=auto, anonymous download fails
(`GatedRepoError`). WORKFLOW §0C, the RUNBOOK input list (new "Access" line) and the kys-eval README now state
exactly: the init repo is public with gating disabled and needs no token; the raw-selected repo is gated and the
collaborator needs a Hugging Face token with write access to `blab-jhu`, kept in the environment, never in a file.
kys-eval `main` = `00bbca3483b35e932e20f75e6a49c18a57a225e4` (README only). No token appears in any file.
