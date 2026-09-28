# Know Your Sources — Raw-Selected Baselines — Execution Workflow (1.5B)

**For:** Marc Marone's cluster, and the agent operating it.
**Code:** `github.com/imHuicongZhang/nanotron`, branch `huicong-dev`, at or after `0246ad59`.
**Data:** `huggingface.co/datasets/blab-jhu/KYS-Pre-Rewritten`, revision `ed09db2a` or later. It holds
the raw text of all seven corpora and the tokenizer.
**Init:** `huggingface.co/wytro/Know-Your-Sources-init`, revision `87e13356`.
**Outputs go to:** `huggingface.co/blab-jhu/KYS-1.5B-Raw-Selected-Baselines`.
**Evaluation:** `github.com/imHuicongZhang/kys-eval`, branch `main`, at or after `50bb288`.
**Comparators:** `huggingface.co/wytro/KYS-1.5B-Rewritten-v2`, revision `b22c8733`.

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

---

## 0. Status and ownership

**Huicong publishes the inputs. Marc downloads, trains, evaluates and uploads the outputs.** Nothing in this
handoff trains on Huicong's cluster.

**Preparation-side checks that were executed** (on Huicong's cluster, CPU only, 2026-09-28). They cover the three
new global Top-10B settings and the shared tooling. The four strategy-linked corpora were **not** re-audited or
re-tokenized; their Hub files and manifest entries are byte-unchanged.

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
| configs: `fill_placeholders.py` for seeds 42/43/44 (a dry-run cluster entry copying skipjack's H100 values) | 126 configs rendered; `assert_invariants.py` passes on all |
| `compare_configs.py` A/B/C vs `wytro/KYS-1.5B-Rewritten-v2@b22c8733` | PASS (only run/path, recompute and seed-42 mbs fields differ; §13) |
| `plan_submit.py` plan + `--test-only` for seeds 42/43/44 | 126/126 segments accepted by `sbatch --test-only`; nothing queued; unknown `--settings` and too-short `time` refused |
| kys-eval `selftest` (7 settings, 63 cells) | pass (run in the preparation Python env, not a fresh `./install.sh` venv) |

**Still pending, and Marc's to run** (hardware- and site-dependent): the INSTALL.md environment and smoke checks
§3.1–3.4 on his nodes; measured s/it and peak memory; `fill_placeholders.py` / `plan_submit.py --test-only` against
his own `marc-cluster` entry and partition; tokenization of all seven settings on his side (the script checks every
sha256 and total); the kys-eval install and reference check on his H100.

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
| settings | **7** | a training corpus (§2) |
| seeds | 3 | 42, 43, 44 (init checkpoint and data seed) |
| setting-seed **chains** | **21** | one WSD run: a trunk plus three annealing branches |
| scheduled **segments** | **126** | 6 per chain (`trunk1 trunk2 trunk3 ep1 ep2 ep3`), one SLURM job each; these are **not** independent training runs |
| **annealed final checkpoints** | **63** | 3 per chain: ep1 / ep2 / ep3 |
| templates per seed | **42** | 7 settings × 6 segments, in `configs/1.5B-baseline-seed<S>/templates/` |

The existing fastText **Quality-Base** (`quality_base` in `wytro/KYS-1.5B-Rewritten-v2`) is a **comparator**. It is
not an eighth setting and is not retrained.

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

**Seeds run one after another.** Start seed 43 only after all 21 ep finals of seed 42 exist and are uploaded, and
seed 44 likewise. Within a seed the seven chains are independent and should run in parallel.

---

## 2. The seven settings

Two families answer different questions. The corpus directory, the setting name and the checkpoint path component
are the same string.

### 2a. Four strategy-linked raw controls (published earlier, unchanged)

Each is the shared 5B anchor plus a 5B raw strategy half linked to one rewritten arm. They were built and published
before this handoff: code `tools/kys_raw/build_raw_sources.py` etc. at the commit recorded in `manifest.json`. This
handoff does not change their data, configs or schedule; it only adds the three settings of §2b.

| setting | comparator (v2) |
|---|---|
| `raw_diversity_oriented` | `diversity_oriented` |
| `raw_disagreement_aware` | `disagreement_aware` |
| `raw_random` | `wrap_inspired` |
| `raw_rewire_inspired` | `rewire_inspired` |

**Their interpretation is under separate review.** Nothing in this handoff re-audits them. Train them as specified;
do not describe them as "the same documents, unrewritten" until that review is published.

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

- The anchor is 4,120,164 docs / 5,000,002,332 tokens, identical in the four strategy-linked corpora.
- The global Top-10B corpora contain anchor documents only where their own score ranked them in (8–20% of tokens).
- `manifest.json` → `settings_overview` records the family, anchor presence and comparator of every setting.

---

## 3. Clone and environment

```bash
git clone -b huicong-dev https://github.com/imHuicongZhang/nanotron.git && cd nanotron
git log --oneline -1            # 0246ad59 or later
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
                              revision="ed09db2aa18b0319297735af181bcbe8b877e830", local_dir="/YOUR/PATH/kys_raw")
init_root = snapshot_download("wytro/Know-Your-Sources-init", revision="87e1335647a3105471928e0fd261d92846021d1a",
                              local_dir="/YOUR/PATH/kys_init")
```

`local_dir` **is** the root; nothing is moved afterwards. Sizes measured on the Hub:

| path | size |
|---|---:|
| raw text, four strategy-linked settings | 66.7 GB (16.6–16.7 GB each) |
| raw text, three global Top-10B settings | 50.1 GB (16.57 / 16.83 / 16.67 GB) |
| `selection/`, `reports/`, `tokenizer/`, `manifest.json` | 0.31 GB |
| init checkpoints, three seeds | 27.1 GB (9.03 GB each) |

```
<data_root>/manifest.json                         counts, expected_total_tokens, sha256 of every file, tokenizer sha256, code commits
<data_root>/raw_text/<setting>/part-000NN.parquet 16 files per setting, in training order
<data_root>/tokenizer/                            Llama-2 tokenizer (vocab 32000, BOS 1, EOS 2)
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

## 5. Tokenize the seven corpora — CPU only, once

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
2. Runs `tools/preprocess_data_parquet.py` with datatrove 0.5.0, 16 tasks, `</s>` per document, no BOS and no
   shuffling. Task i reads exactly file i, so shards `00000…00015` concatenate to the file order.
3. Runs `tools/fix_ds_metadata.py --tokenizer-dir <data_root>/tokenizer`, which makes every `.ds.metadata` name
   the tokenizer directory that nanotron's config asserts against.
4. Requires 16 shards and a summed token count **exactly equal** to `settings.<setting>.expected_total_tokens`.

Output: `<data_root>/<setting>/tokenized/000NN_unshuffled.ds{,.index,.metadata}`, which is what the configs
reference as `{{DATA_ROOT}}/<setting>/tokenized`.

**Verified before publication, for the three global Top-10B settings:** tokenized this way from the files that
were uploaded, and every total matched (selection report §8). The four earlier settings were not re-tokenized in
this preparation; the script applies the same total check to them on your side.
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

---

## 6. (Human) fill `marc-cluster` in `deploy/clusters.yaml`

Fill every field listed in §0A. Then:

```bash
python tools/kys_raw/fill_placeholders.py --cluster marc-cluster --seed 42
```

The script:
- re-renders the **42** templates of `configs/1.5B-baseline-seed42/templates/` into
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
- otherwise it sources the `.env`, runs `assert_invariants.py --config <cfg> --check-resume`, then `torchrun`;
- it fails if the final checkpoint was not written, so dependents do not start.

**What is and is not safe:**
- *Requeue after preemption or node failure* is handled.
  - A trunk resumes from its newest checkpoint via `latest.txt`, losing at most 1500 steps.
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

**Model:** identical for all seven and identical to the v2 grid. 1.5B Llama, 1,504,299,008 parameters:
- 28 layers, hidden 2048, FFN 5632, 16 heads (no GQA), vocab 32000, tied embeddings, RoPE θ = 10000;
- sequence length 2048, bf16;
- AdamW β 0.9/0.95, ε 1e-8, peak LR 5e-4, weight decay 0.1, clip 1.0, fp32 gradient accumulation, `zero_stage 0`.

`reports/CONFIG_COMPARISON.md` shows that each setting's configs equal `raw_random`'s after renaming, for all 126
segments. It also shows that each branch config, parsed by nanotron, equals the released v2 Quality-Base
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

| storage | per chain | per seed (7 chains) | all 3 seeds |
|---|---:|---:|---:|
| **active**, nothing pruned (14 × 21.06 + 9.03) | 303.9 GB | 2.13 TB | — (seeds are serial) |
| **retained locally** until uploads are confirmed (branch points 4292/8583/12875 + finals; 6 × 21.06) | 126.4 GB | 0.88 TB | 2.65 TB if never deleted after upload |
| **uploaded** (3 × (21.06 + 3.01)) | 72.2 GB | 0.51 TB | 1.52 TB (21 chains) |
| inputs: raw text + tokenized, seven settings | — | — | 117 GB raw + 140 GB tokenized = 257 GB, once (+ 27 GB init) |

**How to plan for it:**
- **Peak local need** with the serial-seed plan: one seed's active set (2.13 TB) plus the previous seed's retained
  set until Huicong confirms its upload (0.88 TB), about **3.0 TB**, plus inputs.
- **Pruning.** The eight trunk restart checkpoints per chain (1500, 3000, 4500, 6000, 7500, 9000, 10500, 12000) may
  be deleted once the trunk segment after them has completed. **Never delete 4292 / 8583 / 12875 before their branch
  has finished**: a missing branch point makes that branch train from random init and exit 0.
- **Optimizer state in uploads.** Uploading the step directory with optimizer state follows the original plan.
  Excluding `optimizer/`, as the v2 comparator repo did, would cut the upload to 6.02 GB per final (0.38 TB total).
  Ask Huicong before changing this.

**Runtime.** At 22.8 s/it measured on 4 × H100 (skipjack; your hardware will differ, so use your §8 number):

| | steps | wall |
|---|---:|---:|
| each trunk segment | 4292 / 4291 / 4292 | 27.2 h |
| ep1 / ep2 / ep3 | 476 / 954 / 1430 | 3.0 / 6.0 / 9.1 h |
| one chain, all 6 segments | 15,735 | **99.7 node-hours** (399 GPU-hours) |
| critical path of a chain (trunk1→2→3→ep3) | 14,305 | 90.6 h |
| whole grid (21 chains) | — | **2,093 node-hours** (8,373 GPU-hours) |

**Wall-clock scenarios:**
- With 7 nodes held (one per chain), a seed takes about 100 h and the three serial seeds about 12.5 days, plus
  queueing.
- With enough nodes for the branches to overlap the trunks (up to 14), a seed takes about 91 h (11.3 days for three).
- Evaluation: 63 checkpoints × about 9 min on one H100.

---

## 13. Do not change these — and what is and is not claimed

**Micro-batch size stays 32; scale `dp`, never `mbs`.**
- nanotron normalizes the loss **per micro-batch**: `masked_mean = (loss·mask).sum() / mask.sum()`
  (`src/nanotron/models/llama.py:984-1006`). `label_mask` drops the token before each document boundary, so
  regrouping the same 1024 sequences into a different mbs changes per-token weights. The recorded measurement is
  1.43e-3 relative for mbs 4 vs 16 (`deploy/clusters.yaml`, h200 entry).
- On 80 GB cards mbs 32 needs `recompute_layer: true`. Recomputation changes speed, not the math.
- **Known deviation in the comparator:** the v2 Quality-Base **seed 42** ran mbs 16 / accum 16; seeds 43/44 and all
  other v2 runs used mbs 32 / accum 8. Keep that in mind when comparing seed-42 cells against Quality-Base.
- If you truly cannot fit mbs 32, pass `--expected-mbs N` to `fill_placeholders.py` and `plan_submit.py`, and report
  it.

**Global batch:** 1024 × 2048 = 2,097,152 tokens per step. The renderer derives accum; never set it by hand.

**Data:** no reshuffle, no re-sharding, no filtering, no other tokenizer, 16 tasks. The parquet order *is* the
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
- **Grid:** 63 cells = 7 settings × 3 seeds × 3 epochs.
- **Comparisons:** each strategy-linked setting against its rewritten counterpart; each global Top-10B setting
  against `quality_base`.

```bash
git clone https://github.com/imHuicongZhang/kys-eval.git && cd kys-eval
./install.sh && source .venv/bin/activate && python -m kys_eval.check_install     # must end with OK
python -m kys_eval.selftest && python -m kys_eval.prefetch
# once per cluster, before scoring anything: must print PASS (exact match expected on H100)
python -m kys_eval.reference_check --model hf://wytro/KYS-1.5B-Rewritten-v2/rewrite-1p5b/seed42/diversity_oriented/ep3/hf
# score (skips finished cells); or the SLURM array 0-62 (one seed: 21 cells, --array=0-20 with KYS_GRID_ARGS="--seeds <S>")
python -m kys_eval.run_grid --seeds 42 --delete-weights
mkdir -p logs && sbatch --export=ALL,KYS_EVAL_ROOT=$PWD slurm/eval_grid_array.sbatch
python -m kys_eval.run_grid --list --check-hub        # which checkpoints are uploaded / scored
python -m kys_eval.aggregate                          # reports/raw_selected_report.md, raw_vs_rewritten.md, CSVs
```

Upload `results/` and `reports/` to `rewrite-1p5b/seed<S>/eval/` after each seed.
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
- starting seed 43 before seed 42 is uploaded.

**Report without being asked:**
- the §8 smoke numbers;
- the seven tokenized totals;
- peak memory and s/it of the first `trunk1`;
- each chain's three finals uploaded;
- each seed's completion.
