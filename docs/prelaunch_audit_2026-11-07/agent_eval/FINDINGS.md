# Evaluation sub-audit (kys-eval @ c8aa8ff): findings as reported

The main session saved this. KE = `/weka/scratch/jhu/bvandur1/zhuicon1/projects/kys-eval`.

**Evidence in this folder:**
- `cmp_out.txt`: reference CSV and expected JSON compared with the 54 v2 JSONs;
- `paper_out.txt`: the paper tables recomputed through kys-eval code;
- `list27.txt`: `run_grid --list` for the 27-cell subset;
- `tp.diff`: the patch applied in `third_party/lighteval`;
- `hfdl/`: the v2 model card;
- `scripts/`: the scripts that produced these.

**Table mapping (inferred from labels):**
- Table 12 = `tab:per-task-epoch3`
- Table 13 = `tab:mean6-by-epoch`
- Table 14 = `tab:mmlu-by-epoch`
- Table 15 = the Appendix F category table (`09_appendixF_mmlu_category.tex`)
- Table 3 = `08_table3_mmlu_category.tex`

## Alignment (Part E)

| requirement | code | verdict |
|---|---|---|
| six commonsense tasks | `config.py:82-89, 129-133` | match |
| Mean6 = unweighted mean of the six | `common.py:232-234` | match; every Mean6 cell, std and k/3 count reproduced |
| MMLU = unweighted macro over 57 (not the mean of the categories) | `common.py:235-237`; `config.py:125` asserts 57 | match; every `tab:mmlu-by-epoch` cell reproduced |
| MMLU categories 18 / 13 / 12 / 14 | `config.py:99-123`; `common.py:238-240` | identical to `08_evaluation/02_code/mmlu_categories.py`. Table 3 and Appendix F reproduced; one documented double-rounding exception (ep2 QB STEM 26.87 → 26.9) |
| seed std (ddof=1), paired Δ | `aggregate.py:75-79, 200-204` | match |
| 0-shot | `config.py:74`; `check_install.py:408`; `common.py:253-255` | match; v2 used 0 in all 3402 task configs |
| full splits, no `max_samples` | `eval_checkpoint.py:402-408`; `common.py:251-252` | split sizes 2376 / 10042 / 1838 / 1954 / 500 / 1221, MMLU 14042 |
| acc_norm | `rewrite_tasks.py:42`; patch adds `LoglikelihoodAcc(LogProbTokenNorm())`; `check_install.py:410-413` | token-length normalization |
| task definitions | `rewrite_tasks.py` sha256 `a87d708d…` = the original; `kys_tasks.py:281-283` aborts on mismatch | identical; `kys_tasks.py` only pins `hf_revision` |
| dataset revisions | `config.py:64-72` | 7 pins exist and equal the v2-era cache snapshot. v2 itself ran unpinned (documented) |
| LightEval 10b9104 plus patch | `config.py:23-38`; `install.sh:36-48`; `common.py:95-131`; `eval_checkpoint.py:443-449` | `third_party` diff = the patch file. The 4 patched-file sha256 match the original v2 tree. Enforced unless `--allow-env-mismatch` |
| reference expected JSON | `reference/expected_diversity_oriented_seed42_ep3.json` | identical to v2 `diversity-first-10B-1.5B-seed42/14305` JSON: 64 tasks, 508 fields, 0 differences |
| reference checkpoint on the Hub | – | public at `wytro/KYS-1.5B-Rewritten-v2@b22c8733` `rewrite-1p5b/seed42/diversity_oriented/ep3/hf`; `model.safetensors` sha256 `733262ae…` matches the expected JSON |
| `rewritten_v2_54_per_task.csv` | – | 3456 rows; all 6912 acc and acc_norm cells match the 54 v2 JSONs (max diff 5e-11); GPU column "NVIDIA H100 80GB HBM3" for all |
| generative tasks | – | none: all log-likelihood, so decoding seeds do not matter |
| λ0 → `disagreement_aware` (λ 0.5) | `config.py:232-235` | match; equals nanotron `registry.py:37-38`; selftest asserts it |
| 72-cell grid / 27-cell subset | `run_grid --list` | 27 cells, `--array=0-26`, canonical order (seed, setting, epoch) |

## Findings

| ID | location | actual | severity | fix |
|---|---|---|---|---|
| E-F1 | `eval_checkpoint.py:394-396, 438-440`; `run_grid.py:69, 82-94` | `is_complete` checks only the schema and the `complete` flag, not the cell, subfolder, repo, HF commit, weights sha256 or environment. A stale result is skipped silently and then aggregated. | Medium | compare the stored cell, subfolder and commit before skipping; for the rebuttal, use a fresh results root or `--force` after any re-upload |
| E-F2 | `slurm/eval_grid_array.sbatch:9, 23` | The default `--array=0-71` without `KYS_GRID_ARGS` runs all 72 cells; missing checkpoints fail loudly (`common.py:173-174`). `--array=0-26` *without* `KYS_GRID_ARGS` silently scores the wrong 27 cells and misses 18 rebuttal cells. | Medium (operational) | exact commands (Part F12) |
| E-F3 | `run_grid.py:52-54` | defaults to 72 cells | Harmless | none |
| E-F4 | `README.md:29, 120, 134`; `reference_check.py:7`; `slurm/reference_check.sbatch:17`; `reference/expected…json:17` | reference instructions use an old local path and say the checkpoint is "not on the public Hub"; it is public | Low | use the `hf://…@b22c87333783945e1c655ae46886b7e842ee4203` spec |
| E-F5 | `common.py:69-77`; `config.py:157-158` | non-H100 refused unless `--allow-any-gpu`; the substring check accepts H100 NVL/PCIe while v2 was SXM | Low (documented) | optional: require the exact name |
| E-F6 | `aggregate.py:116-133` | only GPU mismatches are surfaced in aggregate reports | Low | surface environment mismatch flags |
| E-F7 | `reference_check.py:10-11`; `README.md:125` | claims `hash_full_prompts` checks prompts; it is the hash of "" in LightEval 10b9104. Prompts are still covered by the example and token hashes. | Note | fix the wording |
| E-F8 | `install.sh:18-19` | the "already patched" check tests only the 4 file hashes, not HEAD | Low | also assert HEAD and the modified-file set |
| E-F9 | `config.py:144-147` | auto batch size is not recorded; same as v2 on H100 80GB | Note | none |
| E-F10 | `run_grid.py:56` | `--hf-revision` defaults to main; the resolved commit is recorded | Note | pin it after uploads |
| E-F11 | Hub `blab-jhu/KYS-1.5B-Raw-Selected-Baselines` | only `.gitattributes` (commit 10a1133b); all cells missing | blocks evaluation until upload | upload, then `--check-hub` |
| E-F12 | `aggregate.py:185` and the stale list below | stale "v2 ≠ paper" text; `aggregate.py:185` writes it into every generated report | Medium (`aggregate.py:185`) / Low | replace |
| E-F13 | `aggregate.py:224`; `rewrite_tasks.py:285` | not layout claims | Harmless | none |

## Stale "v2 ≠ the paper's printed tables" references

**HF `wytro/KYS-1.5B-Rewritten-v2` README.md** (b22c8733 = main):
- :35 "## Not the paper-era checkpoint"
- :37 "These are **not** the checkpoints behind the numbers printed in the paper's tables."
- :37-39 the paper-era steps 4770 / 9540 / 14305 and 14147, "released separately as `blab-jhu/KYS-1.5B-*`"
- :39-40 "per-cell scores differ from the paper's by up to about 1 accuracy point"
- :40-41 "Use this repo for kys-eval comparisons, and the `blab-jhu/KYS-1.5B-*` repos to reproduce the paper's tables."

**kys-eval README.md:**
- :10 "v2 numbers are not the numbers printed in the paper's tables."
- :10-13 paper-era steps; "v2 is the comparator by decision"
- :134-135 "The reference checkpoint is **not on the public Hub**…" (false)

**kys_eval/config.py:**
- :8 "Note: v2 numbers are NOT the numbers printed in the paper's tables."
- :8-11 paper-era steps; "v2 is the comparator by decision"

**Elsewhere:**
- `kys_eval/aggregate.py:185` "These are not the paper's printed table values." (emitted into reports)
- `reference/expected_diversity_oriented_seed42_ep3.json:17` "Not on the public HF Hub. NOT the same weights as … (paper-era step 14147)."

## Not verified

- **Table numbering:** inferred from labels.
- **Main-text eval statements:** not available.
- **The v2 dataset cache:** inferred.
- **`check_install` end to end:** the run timed out at 300 s; its component checks were verified directly.
- **GPU-dependent behavior:** reference-check reproduction on a new node; races on the shared `model_cache` between
  concurrent array tasks.
- **`blab-jhu/KYS-1.5B-Diversity-Oriented` weights vs v2:** not checked.
