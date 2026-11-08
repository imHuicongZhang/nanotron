# Know Your Sources — Raw-Selected Baselines — Execution Workflow (1.5B)

**For:** Marc Marone's cluster, and the agent operating it.
**Code:** `github.com/imHuicongZhang/nanotron`, branch `huicong-dev`, at or after `e5316a52` (training code and configs are unchanged since `0246ad59`).
**Data:** `huggingface.co/datasets/blab-jhu/KYS-Pre-Rewritten`, revision `b264b0ce` or later (raw-text data files identical since `a5b3ab93`; `8a394d84` added `signal-disagreement-lambda0/tokenized/`, `b264b0ce` its README entry). It holds
the raw text of the seven raw corpora, the tokenizer and, under `signal-disagreement-lambda0/tokenized/`, the
already-tokenized eighth corpus (§2d; added in revision `8a394d84`, so `4ba8df66` and earlier lack it).
**Init:** `huggingface.co/wytro/Know-Your-Sources-init`, revision `87e13356`.
**Outputs go to:** `huggingface.co/blab-jhu/KYS-1.5B-Raw-Selected-Baselines`.
**Evaluation:** `github.com/imHuicongZhang/kys-eval`, branch `main`, at or after `c8aa8ff` (the first commit that
covers `signal-disagreement-lambda0`; 72-cell grid).
**Comparators:** `huggingface.co/wytro/KYS-1.5B-Rewritten-v2`, revision `b22c8733`.

**What to run now: the rebuttal set, §1b** (nine chains: `raw_diversity_oriented`, `raw_disagreement_aware` and
`signal-disagreement-lambda0`, each at seeds 42, 43 and 44). The other five raw settings are not part of it and must
not be queued.

This document is the single entry point. It is self-contained. The other files are detailed references:
`RUNBOOK.md`, `README.md` (this directory), `INSTALL.md` and the kys-eval README.

**Reports behind the data:**
- [`reports/GLOBAL_TOP10B_SELECTION_REPORT.md`](reports/GLOBAL_TOP10B_SELECTION_REPORT.md): the three global
  Top-10B selections: universe, percentiles, cutoffs, digests, overlaps and topics. It also covers scorer-training
  data (§2b), the export check (§7b), consumer tokenization (§8) and publication (§9).
- [`reports/scorer_training_audit/`](reports/scorer_training_audit/): the scripts and JSON behind selection report
  §2b.
- [`reports/CONFIG_COMPARISON.md`](reports/CONFIG_COMPARISON.md): proof that the configs differ from the validated
  reference only in run/dataset fields.
- [`reports/strategy_linked_audit/RAW_SELECTED_BASELINES_PROVENANCE.md`](reports/strategy_linked_audit/RAW_SELECTED_BASELINES_PROVENANCE.md):
  the audit of the four strategy-linked raw corpora: exact construction, verdicts, what each comparison supports.
  Its evidence (scripts, JSON, the cited production code) is in the same folder.

---

## 0. Status and ownership

**Huicong publishes the inputs. Marc downloads, trains, evaluates and uploads the outputs.** Nothing in this
handoff trains on Huicong's cluster.

**Preparation-side checks that were executed** (on Huicong's cluster, CPU only, 2026-09-28). They cover the three
new global Top-10B settings, the shared tooling, and an audit of the four strategy-linked corpora (§2a), whose data
files are byte-unchanged.

| check | result |
|---|---|
| three new selections: second full run in-process, independent run under numpy 1.26 vs 2.0 | identical counts, boundaries and digests |
| stored `*-ranking-v2` percentiles recomputed from raw scores (all 99,949,162 rows) | 0 mismatches; consensus q and selection identical |
| existing Quality-Base rebuilt under the same conventions | doc-set sha256 `90252ee5…` = the published `quality_base/metadata.json` digest; global fastText Top-10B = Quality-Base minus 3 tail docs |
| scorer-training audit (ModernBERT head; ~5M analysis sample) | head's training docs: 0 in the pool; 5M sample kept, 5.00% of every corpus, no enrichment (selection report §2b) |
| assembly: every document re-tokenized, length vs `tokens-llama2` | 0 mismatches |
| export check (`verify_global_top10b.py`) | doc sets = selections, 0 duplicates, order = seed-42 shuffle, token totals exact, 3 × 32,000 sampled texts byte-identical to the raw pool |
| consumer tokenization (`tokenize_raw_text.sh`, unchanged) of the three new settings | datatrove totals = `expected_total_tokens` exactly |
| upload to `blab-jhu/KYS-Pre-Rewritten@a5b3ab93` | 54 new data files: Hub LFS sha256 and size = local; `manifest.json` round-trips; fresh download reads |
| upload of `signal-disagreement-lambda0/tokenized/` to `blab-jhu/KYS-Pre-Rewritten@8a394d84` (2026-11-06) | 79 files: the 52 LFS files' Hub sha256 and size = local; the 26 `.ds.metadata` and `SHA256SUMS` downloaded back and byte-identical |
| configs: `fill_placeholders.py` for seeds 42/43/44 (a dry-run cluster entry copying skipjack's H100 values) | 126 configs rendered; `assert_invariants.py` passes on all |
| `compare_configs.py` A/B/C vs `wytro/KYS-1.5B-Rewritten-v2@b22c8733` | PASS (only run/path, recompute and seed-42 mbs fields differ; §13) |
| `plan_submit.py` plan + `--test-only` for seeds 42/43/44 | 126/126 segments accepted by `sbatch --test-only`; nothing queued; unknown `--settings` and too-short `time` refused |
| strategy-linked audit: each raw strategy half rebuilt from `wytro/Know-Your-Sources@9e5ff241` alone; anchors vs the arms' anchors | all four equal the published corpora; anchors identical; no overlap or duplicates |
| consumer tokenization of the four strategy-linked corpora (files checked against `manifest.json` sha256) | datatrove totals = `expected_total_tokens` exactly; every document is text + `</s>`, no `<s>` |
| `signal-disagreement-lambda0` (2026-11-06; RUNBOOK "Additional setting"): anchor ids vs `raw_disagreement_aware`'s anchor; `pp_io.bucketed_shuffle(seed 42)` recomputed; 6,000 rows re-tokenized; 26 tokenized shards vs their parquet files; 18 configs rendered against the data | anchor identical (4,120,164); shuffle order reproduced exactly; `train_tokens` = text tokens in all rows; total 10,000,002,556; full `assert_invariants.py` passes on 18/18 |
| kys-eval `selftest` at `c8aa8ff` (8 settings, 72 cells; `signal-disagreement-lambda0` paired with `disagreement_aware`) | pass (run in the preparation Python env, not a fresh `./install.sh` venv) |
| rebuttal set (§1b, 2026-11-07): the 54 configs of its nine chains rendered against this cluster's data at dp 4; `signal-disagreement-lambda0` seed 44 also at dp 8 (mbs 32 / accum 4); a dry run of `fill_placeholders.py` + `plan_submit.py` with an 8-GPU `marc-cluster-8gpu` entry for that chain | full `assert_invariants.py` passes on 54/54 (dp 4) and 6/6 (dp 8), corpus totals exact; the dp-8 plan requests `--gres=gpu:8 --nodes=1` and leaves seed 44's dp-4 configs untouched; nothing submitted |

**Still pending, and Marc's to run** (hardware- and site-dependent): the INSTALL.md environment and smoke checks
§3.1–3.4 on his nodes; measured s/it and peak memory; `fill_placeholders.py` / `plan_submit.py --test-only` against
his own `marc-cluster` entry and partition; tokenization of the seven raw settings on his side (the script checks every
sha256 and total) and `fix_ds_metadata.py` on the downloaded `signal-disagreement-lambda0/tokenized/` (§5); the kys-eval install and reference check on his H100.

### Things Marc must supply (the complete list)

**A. `deploy/clusters.yaml`, entry `marc-cluster`.** Every field is `null` and commented in place.

| group | key | value |
|---|---|---|
| paths | `data_root` | the `local_dir` of the dataset download (§4) |
| | `tokenizer_path` | `<data_root>/tokenizer`, the **directory** |
| | `ckpt_root` | checkpoint root; size it per §12 |
| | `init_root` | the `local_dir` of the init download |
| | `repo_dir` | absolute path of this clone; `plan_submit.py` refuses if it is not the clone it runs from |
| | `env_activate` | file to `source` for the INSTALL.md environment |
| | `python_include` | optional; only if compute nodes lack `Python.h` |
| hardware | `arch` | `sm_90` for H100/H200, `sm_100` for B200/B300 (`nvidia-smi --query-gpu=compute_cap --format=csv`) |
| | `hbm_gib` | usable memory, e.g. `74.5` (80 GB H100), `131.3` (H200) |
| | `gpus_per_node`, `dp` | GPUs on one node; `dp` = that number (one node per job) |
| | `tp`, `pp` | `1`, `1` |
| | `micro_batch_size` | `32` (§13) |
| | `recompute_layer` | `true` on 80 GB cards (mbs 32 needs it; measured 49.6 GiB peak on one H100) |
| | `zero_stage` | `0` |
| logging | `wandb.mode/project/entity/dir` | `offline` is fine; `project: zhc-1p5b-10b-wsd`; `dir` on shared storage |
| SLURM | `partition`, `account`, `qos`, `gres`, `cpus_per_task`, `time` | from `sinfo -o "%P %a %l %D %G"`; `gres` must give `dp` GPUs on one node; **`time` ≥ 12 h** (§7) |

**B. kys-eval placeholders** in `slurm/eval_grid_array.sbatch` and `slurm/reference_check.sbatch`: `<PARTITION>`,
`<ACCOUNT>`, `<GPU_TYPE>`, `<TIME>` (`01:00:00` per checkpoint).

**C. Access.**
- An HF token with write access to `blab-jhu/KYS-1.5B-Raw-Selected-Baselines`. Reading needs none: all inputs are
  public.
- GitHub push to a branch `marc-runs` of `imHuicongZhang/nanotron`, or send a patch.
- Tell Huicong which usernames to grant.
- Keep tokens in the environment (`HF_TOKEN`, `huggingface-cli login`). **Never put a token in a config, a
  `clusters.yaml` entry or a commit.**

**D. Report before the first full submission:** GPU model and count per node, how many nodes can run at once, the
partition wall limit, and the smoke results of §8 (loss at steps 1/10/20, steady-state s/it, peak memory).

---

## 1. The grid, and the units used in this document

| unit | count | what it is |
|---|---:|---|
| settings | **8** | a training corpus (§2): 7 raw + `signal-disagreement-lambda0` |
| seeds | 3 | 42, 43, 44 (init checkpoint and data seed) |
| setting-seed **chains** | **24** | one WSD run: a trunk plus three annealing branches |
| scheduled **segments** | **144** | 6 per chain (`trunk1 trunk2 trunk3 ep1 ep2 ep3`), one SLURM job each; these are **not** independent training runs |
| **annealed final checkpoints** | **72** | 3 per chain: ep1 / ep2 / ep3 |
| templates per seed | **48** | 8 settings × 6 segments, in `configs/1.5B-baseline-seed<S>/templates/` |

The existing fastText **Quality-Base** (`quality_base` in `wytro/KYS-1.5B-Rewritten-v2`) is a **comparator**. It is
not a setting and is not retrained.

**Schedule** (authoritative: `tools/generate_configs.py` `TRAIN_STEPS / BRANCHES / TRUNK_SEGMENTS`; identical to the
released v2 checkpoints' `config.yaml`):

| segment | steps | resumes from | LR | tokens processed at end |
|---|---|---|---|---:|
| `trunk1` | 1 → 4292 | trunk dir via `latest.txt` (step 0 = init) | linear warmup 0 → 5e-4 over 500 steps, then flat | 9,000,976,384 |
| `trunk2` | 4293 → 8583 | trunk dir via `latest.txt` (4292) | flat 5e-4 | 17,999,855,616 |
| `trunk3` | 8584 → 12875 | trunk dir via `latest.txt` (8583) | flat 5e-4 | 27,000,832,000 |
| `ep1` | 4293 → **4768** | `<trunk>/4292` directly | 5e-4 → 0, linear over 476 | **9,999,220,736** |
| `ep2` | 8584 → **9537** | `<trunk>/8583` directly | 5e-4 → 0 over 954 | **20,000,538,624** |
| `ep3` | 12876 → **14305** | `<trunk>/12875` directly | 5e-4 → 0 over 1430 | **29,999,759,360** |

- Tokens per step: 1024 sequences × 2048 = 2,097,152.
- The finals sit at round(k × 10e9 / 2,097,152) for k = 1, 2, 3. The trunk branch points sit at round(0.9 × k × …).
- The step counts are **fixed for every setting**. A corpus that is not exactly 10.000B tokens only moves where the
  data wraps. The v2 grid did the same, including `diversity_oriented` at 9.89B.
- The released v2 Quality-Base ep3 records `consumed_tokens_total` 29,999,759,360, matching the table.

```
trunk1 (1→4292) ──→ trunk2 (→8583) ──→ trunk3 (→12875)
   └→ ep1 (→4768)      └→ ep2 (→9537)     └→ ep3 (→14305)
```

**Why the order matters.** Branches resume from a trunk **step directory**, not the trunk folder. If
`<trunk>/4292` does not exist when `ep1` starts, nanotron logs "No previous checkpoint found" at INFO level, trains
from random initialization, and exits 0. Two guards prevent this:
- `afterok` dependencies make it unschedulable;
- `assert_invariants.py --check-resume`, run inside every job before `torchrun`, refuses to start.

**Seeds run one after another — for the full grid only.** Start seed 43 only after all 24 ep finals of seed 42 exist
and are uploaded, and seed 44 likewise. Within a seed the eight chains are independent and should run in parallel.
**The rebuttal set (§1b) replaces this rule: its three seeds run concurrently.**

---

## 1b. Execution plan: the rebuttal set (nine chains) — run this, not the full grid

**What runs:** nine setting-seed chains, 54 segments, 27 annealed finals. Schedule, segments, init checkpoints and
mbs are exactly those of §1.

| setting | seeds | GPUs per chain | parallelism |
|---|---|---:|---|
| `raw_diversity_oriented` | 42, 43, 44 | 4 | dp 4 / mbs 32 / accum 8 |
| `raw_disagreement_aware` | 42, 43, 44 | 4 | dp 4 / mbs 32 / accum 8 |
| `signal-disagreement-lambda0` | 42, 43 | 4 | dp 4 / mbs 32 / accum 8 |
| `signal-disagreement-lambda0` | 44 | 8 if one node has 8 GPUs, otherwise 4 | dp 8 / mbs 32 / accum 4, otherwise dp 4 / mbs 32 / accum 8 |

**Not part of the rebuttal set, and must not be queued:** `raw_random`, `raw_rewire_inspired`,
`raw_top10b_fineweb_edu`, `raw_top10b_modernbert`, `raw_top10b_consensus`. Their templates and configs stay in the
repo; the full grid (§1) is unchanged. So:
- always pass `--settings` to `plan_submit.py`. Without it, the script plans all eight settings of the seed;
- tokenize only `raw_diversity_oriented` and `raw_disagreement_aware` (§5), and run the `signal-disagreement-lambda0`
  check and `fix_ds_metadata.py` (§5). The other five raw corpora are not needed.

**Launch order on a 32-GPU allocation: 8 job slots of 4 GPUs, one chain per slot.**

*Wave 1: eight chains at once, all at dp 4 with the `marc-cluster` entry.* That is `raw_diversity_oriented` and
`raw_disagreement_aware` at seeds 42, 43 and 44, plus `signal-disagreement-lambda0` at seeds 42 and 43.

```bash
W1=raw_diversity_oriented,raw_disagreement_aware,signal-disagreement-lambda0
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 42 --settings $W1
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 43 --settings $W1
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 44 --settings raw_diversity_oriented,raw_disagreement_aware
# for each line below: first without --submit (plan), then with --test-only (§9), then with --submit
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it <measured> --settings $W1 --submit
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 43 --s-per-it <measured> --settings $W1 --submit
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 44 --s-per-it <measured> \
    --settings raw_diversity_oriented,raw_disagreement_aware --submit
```

Each chain's six segments are separate jobs (§9). After a trunk segment finishes, its branch and the next trunk
segment both become eligible, so with 32 GPUs the branches wait for a free slot. That is expected; `afterok` keeps the
order.

*Wave 2: `signal-disagreement-lambda0` seed 44 alone.* Submit it once the GPUs it needs are free and no wave-1 segment
is still waiting for a slot, so that it does not delay wave-1 branches.
- **dp 8 (preferred).** The launcher supports it on one 8-GPU node. `deploy/slurm/kys_segment.sbatch` runs every
  segment on one node (`--nnodes 1`) with as many processes as the config's `dp`, and the renderer derives
  accum = 1024 / (32 × 8) = 4. Two 4-GPU nodes do **not** work: there is no multi-node launch. Add a cluster entry
  `marc-cluster-8gpu` to `deploy/clusters.yaml`: a copy of `marc-cluster` with `gpus_per_node: 8`, `dp: 8`,
  `slurm.gres` giving 8 GPUs and `cpus_per_task` scaled to match. Keep `micro_batch_size: 32`. Render into a
  separate folder, so that seed 44's dp-4 configs in `filled/` stay as they are:
  ```bash
  python tools/kys_raw/fill_placeholders.py --cluster marc-cluster-8gpu --seed 44 \
      --settings signal-disagreement-lambda0 --out-dir configs/1.5B-baseline-seed44/filled-dp8
  python tools/kys_raw/plan_submit.py --cluster marc-cluster-8gpu --seed 44 --settings signal-disagreement-lambda0 \
      --configs configs/1.5B-baseline-seed44/filled-dp8 --s-per-it <measured at dp 8> --submit
  ```
  Measure s/it at dp 8 with the §8 20-step smoke run on the `filled-dp8` trunk1 config first. dp 8 is not measured
  on our side. Report the run as dp 8 / accum 4: the global batch and the mbs are unchanged, so this is not an
  objective change (§13), but there is no bitwise claim across dp.
- **dp 4 (otherwise).** Use the `marc-cluster` entry like wave 1:
  ```bash
  python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 44 --settings signal-disagreement-lambda0
  python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 44 --s-per-it <measured> \
      --settings signal-disagreement-lambda0 --submit
  ```
  This rewrites `filled/submit_seed44.sh`, the script wave 1 already ran for seed 44; the configs from wave 1 are kept.

**Budget** (22.8 s/it at dp 4, §12):
- *Compute:* 9 chains × 99.7 = 897 node-hours on 4-GPU nodes (3,589 GPU-hours).
- *Wall clock:* wave 1 holds all 32 GPUs for about 100 h (8 × 99.7 slot-hours over 8 slots). Wave 2's critical path
  is 90.6 h at dp 4. At dp 8 it is about 45 h if the speed scales linearly, which is not measured. So the set takes
  about 6 days with dp 8, or about 8 days with dp 4, plus queueing.
- *Storage:* at most 9 × 303.9 GB = 2.7 TB active with nothing pruned. 27 finals uploaded = 0.65 TB.
- *Evaluation:* 27 cells (§14).

---

## 2. The eight settings

Two raw families (§2a, §2b) and one rewritten arm (§2d) answer different questions. The corpus directory, the setting name and the checkpoint path component
are the same string.

### 2a. Four strategy-linked raw controls (published earlier; audited, unchanged)

Each is the shared 5B anchor (identical to the anchor inside its rewritten counterpart) + a **5B raw strategy half**:
a seed-42 uniform random sample, whole documents, of the unique source documents of the counterpart's **final
rewritten half**, up to 5e9 TRAIN tokens. Audit: `reports/strategy_linked_audit/RAW_SELECTED_BASELINES_PROVENANCE.md`.

**What they are: equal-token-budget raw controls, not identical-document controls.**
- Rewriting roughly halves document length (0.46–0.53 rewritten tokens per source token). At the same 5B budget the
  raw half can therefore hold only about half of the rewritten half's source documents; it holds a uniform random
  half, so it has the same distribution as those sources.
- Only sources whose rewrite succeeded (and, for REWIRE, passed the filter) are eligible. For three settings this
  drops 0.15–0.29% of the strategy's input documents, mostly over-length ones.

| setting | comparator (v2) | raw strategy half | what the pair supports |
|---|---|---|---|
| `raw_diversity_oriented` | `diversity_oriented` | random 51.4% of the rewritten half's sources | rewrites vs original text for this strategy's sources, at equal tokens |
| `raw_disagreement_aware` | `disagreement_aware` | random 51.8% of them | same |
| `raw_random` | `wrap_inspired` | random 52.5% of the sources WRAP rewrote **successfully** = 52.4% of WRAP's input documents. WRAP's input was a uniform 10B sample of the pool minus validation and anchor. | WRAP-style rewriting of a uniform sample vs a uniform raw sample of the same population. Not WRAP's exact documents; "uniform-sample reference" only for its non-anchor half. |
| `raw_rewire_inspired` | `rewire_inspired` | random 45.9% of the sources whose rewrites **passed REWIRE's post-rewrite fastText filter** | **conditional**: rewritten vs original text *given* the filter's picks. Not "REWIRE beats raw data"; its membership uses rewrite outcomes. |

- **REWIRE's pipeline vs raw data** is `rewire_inspired` vs `raw_random`: REWIRE's input and `raw_random`'s input
  are uniform samples of the same population.
- **Quality-First has no raw arm.** Quality-Base's non-anchor half is the fastText-best 5B of Quality-First's own
  rewriting input (99.87% of it was rewritten into Quality-First), so `quality_first` vs `quality_base` is its
  comparison, with a slightly stronger, unconditioned raw side.
- The anchor is half of every corpus and identical everywhere, so each comparison varies only the other half.

### 2b. Three global Top-10B quality controls (new)

Each setting's **entire** ~10B corpus is one global Top-10B selection over the same eligible universe as the
original Quality-Base: the 99,949,162 scored documents minus the 50,000-doc validation holdout, with the 5M analysis
sample not excluded.

**Scorer-training data** (selection report §2b):
- The ModernBERT head was fit on 50,427 Claude-labelled documents. Every DCLM one was removed from the pool before
  scoring, so none is in any corpus.
- The ~5M analysis sample was **not** used to fit it and was **not** removed, as in the original Quality-Base.
  Each new corpus holds it at its pool rate of 5.00%.
- No new exclusion was introduced, so the universe stays comparable with Quality-Base.
- Benchmark contamination was not tested.

**Selection rule.** Score descending, with the original seeded tie-break. Whole documents are kept until
cumulative TRAIN tokens (`tokens-llama2 + 1`) first reach 1e10.

**What they do not have:** no anchor, no rewriting, no floors, quotas, variance terms or domain restrictions.

**Comparator:** the existing fastText Quality-Base, which *is* the global fastText Top-10B under these conventions
up to 3 tail documents. Selection report §2.

| setting | score |
|---|---|
| `raw_top10b_fineweb_edu` | `fineweb-edu-ranking-v2` |
| `raw_top10b_modernbert` | `modernbert-ranking-v2` |
| `raw_top10b_consensus` | q = (r_fastText + r_FineWebEdu + r_ModernBERT) / 3, from the tie-aware global percentiles (`*-ranking-v2`), float32 |

These compare *selection scores on raw data*. They do not isolate a rewriting effect.

### 2d. One rewritten λ-grid arm: `signal-disagreement-lambda0` (added 2026-11-06)

Not a raw baseline. It is the λ = 0 arm of the Disagreement-Aware rewritten grid. It is built from the same shared
5B anchor plus a 5B rewritten half whose source documents are chosen by `u = q + λ·√var3` with λ = 0, i.e. by
quality alone (q = mean of the three `*-ranking-v2` percentiles). The published `disagreement_aware` arm is
λ = 0.5. The pair `signal-disagreement-lambda0` vs `disagreement_aware` therefore isolates the disagreement term
at equal tokens and an identical anchor.

- **Composition:** 4,120,164 anchor docs (5,000,002,332 tokens) + 5,527,293 wikipedia-style rewrites +
  2,918,769 distill rewrites; 12,566,226 rows, shuffled once with `pp_io.bucketed_shuffle` seed 42.
- **Shipped tokenized** (26 shards), not as raw text, so it skips §5's tokenize script. Its expected total is
  in `tools/kys_raw/registry.py` (`expected_corpus`), not in `manifest.json`.
- **Total: 10,000,002,556 tokens.** The assembly budget was 10,000,002,558. Two rewrites have empty text, and
  datatrove writes no document for them. That is a property of the corpus, not a download error.
- Same schedule, batch, seeds and init checkpoints as every other setting. Provenance checks:
  RUNBOOK.md, section "Additional setting: `signal-disagreement-lambda0`".

### 2c. Setting table

| setting | family | dataset path | anchor | docs | TRAIN tokens (= `expected_total_tokens`) | comparator (v2) |
|---|---|---|---|---:|---:|---|
| `raw_diversity_oriented` | strategy-linked | `raw_text/raw_diversity_oriented/` | yes, 5B | 7,138,615 | 10,000,002,885 | `diversity_oriented` |
| `raw_disagreement_aware` | strategy-linked | `raw_text/raw_disagreement_aware/` | yes, 5B | 7,018,928 | 10,000,005,069 | `disagreement_aware` |
| `raw_random` | strategy-linked | `raw_text/raw_random/` | yes, 5B | 9,674,689 | 10,000,003,751 | `wrap_inspired` |
| `raw_rewire_inspired` | strategy-linked | `raw_text/raw_rewire_inspired/` | yes, 5B | 7,993,150 | 10,000,003,131 | `rewire_inspired` |
| `raw_top10b_fineweb_edu` | global Top-10B | `raw_text/raw_top10b_fineweb_edu/` | **no** | 7,534,192 | 10,000,007,488 | `quality_base` |
| `raw_top10b_modernbert` | global Top-10B | `raw_text/raw_top10b_modernbert/` | **no** | 5,918,416 | 10,000,000,657 | `quality_base` |
| `raw_top10b_consensus` | global Top-10B | `raw_text/raw_top10b_consensus/` | **no** | 5,643,567 | 10,000,003,972 | `quality_base` |
| `signal-disagreement-lambda0` | rewritten λ grid (§2d) | `signal-disagreement-lambda0/tokenized/` (tokenized) | yes, 5B | 12,566,224 | 10,000,002,556 (registry, not `manifest.json`) | `disagreement_aware` |

- The anchor is 4,120,164 docs / 5,000,002,332 tokens, identical in the four strategy-linked corpora and in
  `signal-disagreement-lambda0`.
- The global Top-10B corpora contain anchor documents only where their own score ranked them in (8–20% of tokens).
- `manifest.json` → `settings_overview` records the family, anchor presence and comparator of every setting.

---

## 3. Clone and environment

```bash
git clone -b huicong-dev https://github.com/imHuicongZhang/nanotron.git && cd nanotron
git log --oneline -1            # e5316a52 or later
```

Follow `INSTALL.md` §2 exactly: Python 3.11, torch 2.8.0 cu128, flash-attn 2.8.3, `datatrove[io]==0.5.0`,
`tokenizers>=0.20`, transformers 4.46.3, and `grouped_gemm` at the pinned commit. The fork carries source patches;
upstream nanotron will not work. H100/H200 (`sm_90`) are covered by the prebuilt wheels. For B200/B300 see
`INSTALL.md` Appendix A before using them.

---

## 4. Download and verify

```python
from huggingface_hub import snapshot_download
data_root = snapshot_download("blab-jhu/KYS-Pre-Rewritten", repo_type="dataset",
                              revision="b264b0cef6dab29fceebb50cb1c7b3fe152ba5d9", local_dir="/YOUR/PATH/kys_raw")
init_root = snapshot_download("wytro/Know-Your-Sources-init", revision="87e1335647a3105471928e0fd261d92846021d1a",
                              local_dir="/YOUR/PATH/kys_init")
```

`local_dir` **is** the root; nothing is moved afterwards. Sizes measured on the Hub:

| path | size |
|---|---:|
| raw text, four strategy-linked settings | 66.7 GB (16.6–16.7 GB each) |
| raw text, three global Top-10B settings | 50.1 GB (16.57 / 16.83 / 16.67 GB) |
| `signal-disagreement-lambda0/tokenized/` (78 files + `SHA256SUMS`) | 19 GB |
| `selection/`, `reports/`, `tokenizer/`, `manifest.json` | 0.31 GB |
| init checkpoints, three seeds | 27.1 GB (9.03 GB each) |

```
<data_root>/manifest.json                         counts, expected_total_tokens, sha256 of every file, tokenizer sha256, code commits
<data_root>/raw_text/<setting>/part-000NN.parquet 16 files per setting, in training order
<data_root>/signal-disagreement-lambda0/tokenized/  26 × 000NN_unshuffled.ds{,.index,.metadata} + SHA256SUMS (§2d)
<data_root>/tokenizer/                            Llama-2 tokenizer (vocab 32000; <s> = 1, </s> = 2)
<data_root>/selection/<new setting>/*.npy         selected doc ids of the three global Top-10B settings
<data_root>/reports/                              provenance, selection and config reports
<init_root>/_init_1.5B_seed<S>/0/                 init checkpoint (180 files), S in 42 43 44
<init_root>/init_1.5B_seed<S>.hash.json           its manifest
```

**Parquet columns.**
- Strategy-linked settings: `orig_doc_id`, `source` (`anchor` or `strategy`), `text`.
- Global Top-10B settings: `orig_doc_id`, `doc_id` (scored-pool id, the selection key), `source` (`selected`),
  `text`.
- Only `text` is tokenized.

**Verify the init checkpoints** (the data files are verified by the tokenize script in §5):

```bash
for S in 42 43 44; do
  python tools/hash_init_checkpoint.py <init_root>/_init_1.5B_seed$S/0 --check <init_root>/init_1.5B_seed$S.hash.json
  find <init_root>/_init_1.5B_seed$S/0 -type f | wc -l     # must print 180
done
```

Do not download the rewritten corpora (`wytro/Know-Your-Sources*`); they are not needed to train.

---

## 5. Tokenize the seven raw corpora — CPU only, once

Each parquet folder is **the final corpus in training order**: anchor merged where there is one, shuffled once with
seed 42, rows in order. Tokenization must preserve that order.

```bash
for s in raw_diversity_oriented raw_disagreement_aware raw_random raw_rewire_inspired \
         raw_top10b_fineweb_edu raw_top10b_modernbert raw_top10b_consensus; do
  tools/kys_raw/tokenize_raw_text.sh <data_root> $s          # tokenizer defaults to <data_root>/tokenizer
  # on a node with < ~170 GiB RAM:  KYS_TOKENIZE_WORKERS=4 tools/kys_raw/tokenize_raw_text.sh <data_root> $s
done
```

**What the script does, per setting:**
1. Checks the three tokenizer files and the 16 parquet files against `manifest.json` sha256.
2. Runs `tools/preprocess_data_parquet.py` with datatrove 0.5.0, 16 tasks, one `</s>` appended per document, no `<s>` and no
   shuffling. Task i reads exactly file i, so shards `00000…00015` concatenate to the file order.
3. Runs `tools/fix_ds_metadata.py --tokenizer-dir <data_root>/tokenizer`, which makes every `.ds.metadata` name
   the tokenizer directory that nanotron's config asserts against.
4. Requires 16 shards and a summed token count **exactly equal** to `settings.<setting>.expected_total_tokens`.

Output: `<data_root>/<setting>/tokenized/000NN_unshuffled.ds{,.index,.metadata}`, which is what the configs
reference as `{{DATA_ROOT}}/<setting>/tokenized`.

**Verified for all seven settings** on the preparation cluster: tokenized this way from files whose sha256 match
`manifest.json`, and every total matched `expected_total_tokens` exactly (selection report §8; strategy-linked
audit §R5).

**Token convention: document text + one `</s>` (id 2), no `<s>` (id 1).** The Llama-2 `tokenizer.json` would prepend
`<s>` by default; datatrove 0.5.0 replaces that post-processor with `$A </s>`. The rewritten comparators' training
streams have the same structure (checked on the bytes). Older code comments that call the per-document "+1" a BOS
(`select_10b.py`, `pp_io.py`) mislabel it: it is the appended `</s>`, and the counts are unaffected.
- **Memory:** each of the 16 datatrove tasks peaked at ~10.2 GiB RSS, so the default (all 16 at once) needs
  ~165 GiB. On a smaller node set `KYS_TOKENIZE_WORKERS=<n>` with n × 10.5 GiB fitting in memory; the output is
  identical. If a task is OOM-killed, delete `<data_root>/<setting>/` and rerun with a smaller n.
- **Time:** 16–18 min per setting with `KYS_TOKENIZE_WORKERS=4` on 17 cores, sha256 checks included.
- **Output:** 20.06 GB of `.ds` + index per setting, about 140 GB for all seven.
- Run settings in parallel if you have the cores and memory.

**Do not** pass a shuffle option, change the task count, re-split or concatenate the parquet files, or use any
other tokenizer file. Any of these changes the data order or the token stream.

`000NN_unshuffled.ds` is datatrove's default filename; the shuffle happened upstream. Use the same `--tokenizer-dir`
string for every setting, because nanotron asserts that all metadata files carry the identical tokenizer path.

**`signal-disagreement-lambda0` is downloaded already tokenized. Do not tokenize it.** It was tokenized on the
preparation cluster with the same script and tokenizer, using 26 tasks over its 26 files (shard i = file i). Its
metadata still names the preparation cluster's tokenizer path, so check it and point it at yours:

```bash
cd <data_root>/signal-disagreement-lambda0/tokenized && sha256sum -c SHA256SUMS && cd -     # 78 files OK
python tools/fix_ds_metadata.py --output-folder <data_root>/signal-disagreement-lambda0/tokenized \
    --tokenizer-dir <data_root>/tokenizer
```

`assert_invariants.py` then requires 26 shards and exactly 10,000,002,556 tokens (§2d).

---

## 6. (Human) fill `marc-cluster` in `deploy/clusters.yaml`

Fill every field listed in §0A. Then:

```bash
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 42
```

The script:
- re-renders the **48** templates of `configs/1.5B-baseline-seed42/templates/` (or the `--settings` subset) into
  `configs/1.5B-baseline-seed42/filled/` (a `.yaml` and a `.env` each);
- derives `accum = 1024 / (mbs × dp)` and refuses a non-integer result;
- checks the memory fit against `hbm_gib`;
- runs `assert_invariants.py` on every file.

`--settings a,b` restricts it to a subset. Never edit rendered configs by hand. Commit the `marc-cluster` entry and
the `filled/` configs to branch `marc-runs`, never `huicong-dev`; `.env` files are git-ignored.

---

## 7. SLURM wall time — the one sizing rule

At the measured 22.8 s/it (4 × H100, mbs 32 + recompute, §12), the longest stretch **without a checkpoint** is
9.5 h:
- a trunk persists work only every 1500 steps (9.5 h) or at its segment end;
- a branch writes nothing until its final step (ep3: 1430 steps, 9.1 h).

A job with a shorter limit requeues forever at the same step. So `slurm.time` must be at least about 12 h at that
speed. A full trunk segment is 27.2 h, so a limit of ≥ 28 h avoids trunk requeues entirely.
`plan_submit.py --s-per-it <measured>` refuses a `time` that is too short for your measured speed.

A limit between 12 h and 28 h is safe for trunk1 too, but only with the launcher at or after the B1 fix
(prelaunch audit 2026-11-07). trunk1 is rendered weights-only, because its step 0 is the init, which has no
optimizer state. Before that fix, a trunk1 restarted from its step-1500 or 3000 save reloaded the weights but
restarted the LR schedule, AdamW state and data stream from step 0, and still exited 0. Now `kys_segment.sbatch`
runs a state-loading copy of the config whenever `latest.txt` > 0 (§9), and `assert_invariants.py --check-resume`
refuses a weights-only trunk config whose `latest.txt` is not 0.

---

## 8. Smoke checks — before any full submission

Run `INSTALL.md` §3.1–3.4 on your nodes:

| check | proves |
|---|---|
| §3.1 GPU + flash-attn | torch sees the GPUs; flash-attn imports for your `arch` |
| §3.2 tokenizer | `AutoTokenizer.from_pretrained(<data_root>/tokenizer)` loads, `len == 32000` |
| §3.3 init checkpoints | `hash_init_checkpoint.py --check` matches for 42, 43, 44 |
| §3.4 20-step run | real data, real init, filled config, full global batch, one node |

```bash
python tools/kys_raw/smoke_20steps.py \
    --config configs/1.5B-baseline-seed42/filled/raw_diversity_oriented_seed42_trunk1.yaml \
    --init <init_root>/_init_1.5B_seed42/0 --workdir <scratch dir>
# then run the torchrun command it prints, on one node with dp GPUs
```

**Report to Huicong:**
- `lm_loss` at steps 1, 10 and 20;
- steady-state `time_per_iteration_ms`, and peak GPU memory.

Our reference on 4 × H100 (dp 4, mbs 32 + recompute) is 10.8 / 8.83 / 8.19, 22.8 s/it and 59.4 GiB. That reference
is on the rewritten `diversity_oriented` corpus; a raw corpus will differ slightly in loss, but should have the same
shape. **The measured s/it is the basis for every wall-time number: pass it to `plan_submit.py --s-per-it`.**

---

## 9. Plan, dry-run, submit — per seed

```bash
# 1) plan (dry run): writes configs/1.5B-baseline-seed42/filled/submit_seed42.sh, prints the job table, submits nothing
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it <measured>
# 2) let SLURM validate every segment's partition/account/qos/gres/time without queueing anything
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it <measured> --test-only
# 3) validate one chain end to end first (optional), then the whole seed
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it <measured> --settings raw_random --submit
python tools/kys_raw/plan_submit.py --cluster marc-cluster --seed 42 --s-per-it <measured> --submit
```

The generated `submit_seed<S>.sh`:
1. **Refuses to submit anything** if a job named `kys_<setting>_seed<S>_<segment>` for any of its segments is
   already pending or running for `$USER` (`squeue`).
2. Seeds each trunk directory with the init checkpoint as step `0` plus `latest.txt`. This is skipped if
   `latest.txt` exists, and refused if the init is incomplete (not 180 files).
3. Submits the 6 segments per setting with the `afterok` graph of §1 and `--requeue`.

In `--test-only` mode dependencies are omitted, because the upstream ids do not exist yet.

**Inside each job** (`deploy/slurm/kys_segment.sbatch`):
- if `<ckpt>/<final step>/model_config.json` already exists, the job exits 0;
- otherwise it sources the `.env` and picks the config to run with `tools/kys_raw/trunk_restart_config.py`. That
  is the rendered config itself, except for a trunk whose config is weights-only (trunk1) while `latest.txt` > 0.
  Then the job writes `<trunk dir>.restart/<run>_from<step>.yaml`, a copy that differs only in
  `load_optimizer: true` and `load_lr_scheduler: true`, and logs `[restart-config] … running …`;
- it runs `assert_invariants.py --config <that config> --check-resume`, then `torchrun`;
- it fails if the final checkpoint was not written, so dependents do not start.

**What is and is not safe:**
- *Requeue after preemption or node failure* is handled.
  - A trunk resumes from its newest checkpoint via `latest.txt`, losing at most 1500 steps. It restores the weights,
    AdamW state, LR schedule and step/consumed-sample counters, so the data stream continues where it stopped. For
    trunk1 this holds only because of the restart copy above. `--check-resume` refuses to start a weights-only
    trunk1 whose `latest.txt` is past 0, so a manual launch cannot silently restart it from step 0 either. A
    restarted trunk1's first logged iteration is therefore the saved step + 1 (1501 or 3001), not 1.
  - A branch reruns from its branch point.
  - The dependency chain holds because SLURM keeps the job id across a requeue. This is SLURM semantics and was not
    exercised in this preparation.
- *Resubmitting after a segment finished* is safe: the final-checkpoint check skips it.
- *Resubmitting while jobs are queued or running is not safe in general.* The completed-checkpoint skip does not
  protect an unfinished segment, and two copies of a trunk would write into the same checkpoints path. The `squeue`
  guard blocks the common case (same user, same names). It cannot see other users, renamed jobs or other clusters.
  Check `squeue` before resubmitting.

---

## 10. Setting ↔ corpus ↔ checkpoint paths

For setting `X` and seed `S` (composed by `tools/render_config.py`):

```
data        <data_root>/X/tokenized
trunk       <ckpt_root>/seed<S>/X/trunk/X/seed<S>/{0, 1500, 3000, 4292, 4500, 6000, 7500, 8583, 9000, 10500, 12000, 12875}/  latest.txt
ep1 final   <ckpt_root>/seed<S>/X/ep1/X/seed<S>/4768/
ep2 final   <ckpt_root>/seed<S>/X/ep2/X/seed<S>/9537/
ep3 final   <ckpt_root>/seed<S>/X/ep3/X/seed<S>/14305/
wandb run   X_seed<S>_<segment>     (group X_seed<S>)
```

**Model:** identical for all eight and identical to the v2 grid. 1.5B Llama, 1,504,299,008 parameters:
- 28 layers, hidden 2048, FFN 5632, 16 heads (no GQA), vocab 32000, tied embeddings, RoPE θ = 10000;
- sequence length 2048, bf16;
- AdamW β 0.9/0.95, ε 1e-8, peak LR 5e-4, weight decay 0.1, clip 1.0, fp32 gradient accumulation, `zero_stage 0`.

`reports/CONFIG_COMPARISON.md` shows that each setting's configs equal `raw_random`'s after renaming, for all 126
raw segments. The 18 `signal-disagreement-lambda0` configs equal `raw_random`'s after renaming, apart from the
comment line naming the template path (checked 2026-11-06; not yet in that report). It also shows that each branch config, parsed by nanotron, equals the released v2 Quality-Base
checkpoint's `config.yaml` in all 105 keys except run name, paths and (seed 42 only) mbs/accum.

---

## 11. Deliverables per chain — as soon as its three finals exist

1. **HF export** of each final (CPU):
   ```bash
   python tools/kys_eval/convert_to_hf.py --checkpoint_path <ckpt_root>/seed<S>/X/ep<N>/X/seed<S>/<step> \
       --save_path <out>/X_seed<S>_ep<N>/hf --tokenizer <data_root>/tokenizer
   ```
2. **Upload** to `blab-jhu/KYS-1.5B-Raw-Selected-Baselines`:
   ```
   rewrite-1p5b/seed<S>/X/ep<N>/nanotron/    the step directory as written (21.06 GB incl. optimizer state)
   rewrite-1p5b/seed<S>/X/ep<N>/hf/          the export (3.01 GB)
   rewrite-1p5b/seed<S>/X/logs/X_seed<S>_<segment>.csv    step, lm_loss, lr, tokens/s
   ```
   For example: `huggingface-cli upload blab-jhu/KYS-1.5B-Raw-Selected-Baselines <out>/X_seed<S>_ep<N>/hf rewrite-1p5b/seed<S>/X/ep<N>/hf`.
3. **A note per seed** on branch `marc-runs`: GPU type, `dp`, `mbs`, `recompute_layer`, measured s/it, any requeue
   or preemption and where, and any deviation from this document.

---

## 12. Storage and runtime (from measured sizes)

**Measured inputs:**
- a full checkpoint with optimizer state is **21.06 GB** (18.05 optimizer + 3.01 model; v2 manifest);
- the HF export is **3.01 GB**;
- the init copied to step 0 is **9.03 GB**;
- a chain writes 14 full checkpoints: 11 on the trunk (1500, 3000, 4292, 4500, 6000, 7500, 8583, 9000, 10500, 12000,
  12875) and 3 finals.

| storage | per chain | per seed (8 chains) | all 3 seeds |
|---|---:|---:|---:|
| **active**, nothing pruned (14 × 21.06 + 9.03) | 303.9 GB | 2.43 TB | — (seeds are serial) |
| **retained locally** until uploads are confirmed (branch points 4292/8583/12875 + finals; 6 × 21.06) | 126.4 GB | 1.01 TB | 3.03 TB if never deleted after upload |
| **uploaded** (3 × (21.06 + 3.01)) | 72.2 GB | 0.58 TB | 1.73 TB (24 chains) |
| inputs: raw text + tokenized, eight settings | — | — | 117 GB raw + 159 GB tokenized = 276 GB, once (+ 27 GB init) |

**How to plan for it:**
- **Peak local need** with the serial-seed plan: one seed's active set (2.43 TB) plus the previous seed's retained
  set until Huicong confirms its upload (1.01 TB), about **3.4 TB**, plus inputs.
- **Pruning.** The eight trunk restart checkpoints per chain (1500, 3000, 4500, 6000, 7500, 9000, 10500, 12000) may
  be deleted once the trunk segment after them has completed. **Never delete 4292 / 8583 / 12875 before their branch
  has finished**: a missing branch point makes that branch train from random init and exit 0.
- **Optimizer state in uploads.** Uploading the step directory with optimizer state follows the original plan.
  Excluding `optimizer/`, as the v2 comparator repo did, would cut the upload to 6.02 GB per final (0.43 TB total).
  Ask Huicong before changing this.

The table is for the full grid; the rebuttal set's numbers are in §1b.

**Runtime.** At 22.8 s/it measured on 4 × H100 (skipjack; your hardware will differ, so use your §8 number):

| | steps | wall |
|---|---:|---:|
| each trunk segment | 4292 / 4291 / 4292 | 27.2 h |
| ep1 / ep2 / ep3 | 476 / 954 / 1430 | 3.0 / 6.0 / 9.1 h |
| one chain, all 6 segments | 15,735 | **99.7 node-hours** (399 GPU-hours) |
| critical path of a chain (trunk1→2→3→ep3) | 14,305 | 90.6 h |
| whole grid (24 chains) | — | **2,392 node-hours** (9,569 GPU-hours) |

**Wall-clock scenarios:**
- With 8 nodes held (one per chain), a seed takes about 100 h and the three serial seeds about 12.5 days, plus
  queueing.
- With enough nodes for the branches to overlap the trunks (up to 16), a seed takes about 91 h (11.3 days for three).
- Evaluation: 72 checkpoints × about 9 min on one H100 (the rebuttal set: 27, §1b).

---

## 13. Do not change these — and what is and is not claimed

**Micro-batch size stays 32; scale `dp`, never `mbs`.**
- nanotron normalizes the loss **per micro-batch**: `masked_mean = (loss·mask).sum() / mask.sum()`
  (`src/nanotron/models/llama.py:984-1006`), then divides by the number of micro-batches
  (`src/nanotron/parallel/pipeline_parallel/engine.py:55`). `label_mask` removes only the label at each document
  start (`src/nanotron/data/clm_collator.py:83-94`), so regrouping the same 1024 sequences into a different mbs
  changes per-token weights slightly. The recorded measurement is 1.43e-3 relative for mbs 4 vs 16
  (`deploy/clusters.yaml`, h200 entry).
- On 80 GB cards mbs 32 needs `recompute_layer: true`. Recomputation changes speed, not the math.
- **Known deviation in the comparator:** of the 54 released v2 checkpoints, 51 used mbs 32 / accum 8; the three
  Quality-Base **seed 42** checkpoints (ep1–ep3) used mbs 16 / accum 16. Keep mbs 32 for every new run (it matches
  every rewritten comparator and Quality-Base seeds 43/44). When comparing against Quality-Base, report seed-averaged
  results with this footnote plus a seeds-43/44-only check; the effect is expected to be far below seed noise
  (Quality-Base ep3 Mean6 by seed: 0.4566 / 0.4617 / 0.4698).
- If you truly cannot fit mbs 32, pass `--expected-mbs N` to `fill_placeholders.py` and `plan_submit.py`, and report
  it.

**Global batch:** 1024 × 2048 = 2,097,152 tokens per step. The renderer derives accum; never set it by hand.

**Data:** no reshuffle, no re-sharding, no filtering, no other tokenizer, 16 tasks (`signal-disagreement-lambda0`:
use the 26 shards as downloaded). The parquet order *is* the
experiment.

**Files you do not edit:** the templates, `tools/kys_raw/`, `tools/generate_configs.py`, `tools/render_config.py`,
`tools/assert_invariants.py`, `deploy/slurm/kys_segment.sbatch`. You own the `marc-cluster` entry and how many chains
run at once.

**Claims deliberately *not* made:**
- **No bitwise reproducibility is promised** across GPU models, drivers or dp. The recorded evidence is losses that
  agree to about 0.01 at three significant figures across dp 1/4 and mbs 4/32, which is the tolerance to expect from
  the smoke check.
- **No portability guarantee beyond `sm_90`** without the INSTALL.md Appendix A steps.
- Requeue and duplicate-submission behaviour is exactly as described in §9.

---

## 14. Evaluation (kys-eval)

It uses the exact v2 protocol (LightEval commit and patch, 0-shot `acc_norm`, dataset revisions, H100):
- **Benchmarks:** Mean6 (ARC-Easy, HellaSwag, PIQA, SIQA, OpenBookQA, CommonsenseQA) and MMLU (57 subjects,
  macro-averaged, plus 4 categories).
- **Grid:** 72 cells = 8 settings × 3 seeds × 3 epochs. kys-eval covers `signal-disagreement-lambda0` as of `main`
  `c8aa8ff` (required: `c8aa8ff` or later). It pairs that setting with `disagreement_aware` (λ = 0.5, §2d), and
  labels the pair "λ grid" in `raw_vs_rewritten.md`.
- **Rebuttal set (§1b):** 27 of the 72 cells. Score them with
  `--settings raw_diversity_oriented raw_disagreement_aware signal-disagreement-lambda0`; the reports then mark
  the other five settings "—".
- **Comparisons:** each strategy-linked setting against its rewritten counterpart (read `raw_rewire_inspired`'s
  as conditional, §2a); `rewire_inspired` against `raw_random` for REWIRE's pipeline effect; each global Top-10B
  setting against `quality_base`.

```bash
git clone https://github.com/imHuicongZhang/kys-eval.git && cd kys-eval
./install.sh && source .venv/bin/activate && python -m kys_eval.check_install     # must end with OK
python -m kys_eval.selftest && python -m kys_eval.prefetch
# once per cluster, before scoring anything: must print PASS (exact match expected on H100)
python -m kys_eval.reference_check --model hf://wytro/KYS-1.5B-Rewritten-v2/rewrite-1p5b/seed42/diversity_oriented/ep3/hf
# rebuttal set: 27 cells (indices 0-26 of the filtered list; check with --list)
R="--settings raw_diversity_oriented raw_disagreement_aware signal-disagreement-lambda0"
python -m kys_eval.run_grid $R --list --check-hub
python -m kys_eval.run_grid $R --delete-weights        # skips finished cells
mkdir -p logs && sbatch --array=0-26 --export=ALL,KYS_EVAL_ROOT=$PWD,KYS_GRID_ARGS="$R" slurm/eval_grid_array.sbatch
# full grid: the SLURM array 0-71 (one seed: 24 cells, --array=0-23 with KYS_GRID_ARGS="--seeds <S>")
python -m kys_eval.run_grid --list --check-hub        # which checkpoints are uploaded / scored
python -m kys_eval.aggregate                          # reports/raw_selected_report.md, raw_vs_rewritten.md, CSVs
```

Upload `results/` and `reports/` to `rewrite-1p5b/seed<S>/eval/` after each seed (for the rebuttal set, once its 27 cells are scored).
- If you have no H100, say so before scoring. `--allow-any-gpu` records and flags it; bf16 near-ties move Mean6 by
  about 7e-4 across architectures (measured, kys-eval `config.py`).
- `eval_checkpoint` scores a single model into `--out-dir`, but only `run_grid` results are aggregated.

---

## 15. Things that look broken and are not

- **Every `.ds` is `000NN_unshuffled.ds`.** That is datatrove's default name; the shuffle happened at the parquet
  stage.
- **`.cache/huggingface/` inside `data_root`** is the download ledger. It is inert; keep it.
- **Checkpoints at 1500 / 3000 / 4500 …** are trunk restart insurance. Branch points come from a segment *ending*
  there. Branches use `checkpoint_interval: 100000` and write only their final state.
- **A segment exits 0 in seconds.** Its final checkpoint existed, so the job skipped `torchrun`.
- **A small loss step inside trunk2/trunk3 near steps 4768/9537** is where the data wraps into the next epoch. A spike
  or a persistent rise is not normal.
- **A global Top-10B corpus has no `anchor` rows.** That is by design.

## 16. If something fails

| symptom | likely cause |
|---|---|
| `ModuleNotFoundError: grouped_gemm` | not installed at the pinned commit (INSTALL.md §2) |
| `data did not match any variant of untagged enum ModelWrapper` | `tokenizers < 0.20` |
| `Tokenizer passed in config … does not match dataset's` | `fix_ds_metadata.py` not run, or a different `--tokenizer-dir` (§5) |
| token total ≠ manifest after tokenizing | files re-split, shuffled, or another tokenizer; stop and report |
| `Python.h: No such file` | set `python_include` |
| the same trunk step repeats after every requeue | `slurm.time` shorter than a checkpoint interval (§7) |
| `DependencyNeverSatisfied` | an upstream segment failed; find it in the job-id lines printed by `submit_seed<S>.sh` |
| `submit_seed<S>.sh` says a job is already queued | a previous submission is still live; do not resubmit |
| a run completes fast with a clean anneal | check that it resumed from a real checkpoint (§1) |
| NaN/inf, or loss rising for hundreds of steps | stop the chain and report; do not tune |

Anything that exits 0 but looks wrong: stop the chain and ask rather than resubmitting.

## 17. Escalation and reporting

**Ask Huicong first before:**
- any value outside `marc-cluster`;
- mbs ≠ 32, or dp not a power of two;
- proceeding past a failed hash, token total or invariant;
- deleting anything under `<ckpt_root>` other than the eight prunable restart points;
- queueing any chain outside the rebuttal set (§1b);
- for the full grid only: starting seed 43 before seed 42 is uploaded (the rebuttal set runs its seeds concurrently).

**Report without being asked:**
- the §8 smoke numbers;
- the seven tokenized totals and the `signal-disagreement-lambda0` check (`sha256sum -c`, then `assert_invariants.py`);
- peak memory and s/it of the first `trunk1`, and for `signal-disagreement-lambda0` seed 44 whether it ran at dp 8
  (with its s/it) or dp 4;
- each chain's three finals uploaded;
- each seed's completion.
