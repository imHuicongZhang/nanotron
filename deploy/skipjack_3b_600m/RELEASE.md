# KYS 3B four-setting release (code)

Branch `release/kys-3b-600m` of `imHuicongZhang/nanotron`. The data is in the private HF dataset
`blab-jhu/KYS-3B-pretraining-data`; its `HANDOFF_RELEASE.json` names the code commit it was verified against.
Trained models go to `blab-jhu/KYS-3B-pretraining-models` (nothing is uploaded there yet).

## Grid (unchanged scientific definition; single source: `registry_3b.py`)

| | |
|---|---|
| settings | quality-first, diversity-oriented, disagreement-aware (rewritten), quality-base (raw shared base + next 10B by the same fastText order) |
| model | the 1.5B llama architecture with only `num_hidden_layers` 28 -> 57 (2,994,444,288 params, tied embeddings), bf16 |
| tokenizer | the 1.5B llama-2 tokenizer, sha256-pinned in `TOKENIZER_SHA256` |
| seed / init | 42; one shared init for all settings (std 0.02, scaled output projections), see below |
| batch | 1024 x 2048 = 2,097,152 tokens/step; layout A = DP 4, ZeRO-1, mbs 16, accum 16, recompute on |
| LR (WSD) | AdamW (0.9, 0.95, 1e-8), wd 0.1, clip 1.0; warmup 1000 linear to 4e-4; constant; linear decay to 0 |
| horizons | 20B / 40B / 60B = finals 9537 / 19073 / 28610, decay from branch points 8583 / 17166 / 25749 |
| segments | trunk1..trunk6 (ends 4292 ... 25749; trunk1 weights-only from the init, later segments resume optimizer + LR), ep1..ep3 branch from the trunk step dirs and decay in their own namespace |
| grid | 4 settings x 9 segments = 36 configs, 12 annealed finals |

The LR-resume fix (`src/nanotron/serialize/optimizer.py`, off-by-one on resume) is included; regression test
`test_lr_resume_3b.py` (replays the scheduler across every segment boundary against an uninterrupted run).

## Portable vs Skipjack-specific

* Portable: the rendered YAML configs' scientific content, `registry_3b.py` constants, `generate_configs_3b.py`,
  `assert_invariants_3b.py`, `epoch_stats_3b.py`, `portable_configs_3b.py`, `tools/fix_ds_metadata.py`,
  `tools/kys_eval/convert_to_hf.py`.
* Skipjack only: `registry_3b.SITE`, `CPU_SLURM`, every `*.sbatch` header (partition/account/QoS/GPU limits), the
  submit/monitor/lane tools (`train8_3b.py`, `capped_submit_3b.py`, `train8_monitor.*`, `handoff_*`), the hard-coded
  worktree path in `init_3b.sh` and the sbatch files, and the committed configs' absolute paths.

## Running elsewhere

    pip install -r deploy/skipjack_3b_600m/requirements-3b.lock.txt && pip install -e .
    hf download blab-jhu/KYS-3B-pretraining-data --repo-type dataset --revision <rev> --local-dir $DATA \
        --include 'tokenized/*' 'tokenizer/*' 'RELEASE_MANIFEST.json'
    for s in quality-first diversity-oriented disagreement-aware quality-base; do
        python tools/fix_ds_metadata.py --output-folder $DATA/tokenized/$s --tokenizer-dir $DATA/tokenizer; done
    wandb login                                                                                # your own key
    python deploy/skipjack_3b_600m/portable_configs_3b.py --data $DATA --ckpt $CKPT --out $CFG \
        --wandb-entity $YOUR_ENTITY --wandb-project $YOUR_PROJECT                              # prints the KYS3B_* exports
    export KYS3B_...=...                                                                       # as printed
    python deploy/skipjack_3b_600m/assert_invariants_3b.py --dir $CFG --corpus                 # 36 configs + exact token totals

W&B: runs log online to the entity and project you pass. Both are required, and the Skipjack owner's entity is refused.
They are written to every `.env` (`WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_MODE=online`) and to `general.project`. The
committed Skipjack `.env` files name the Skipjack owner's entity, so don't source them elsewhere. No W&B credential is
part of this branch or the dataset.

`fix_ds_metadata.py` is required: nanotron loads the tokenizer named in line 1 of every `.ds.metadata`, which records
the Skipjack path the shards were written under (`.../training/tokenizer|2`). It rewrites only that text line; the
`.ds` / `.index` binaries are untouched.

Each segment is `torchrun --nproc_per_node 4 run_train.py --config-file <cfg>` with the segment's `.env` sourced
(W&B variables and the layout check). Order: trunk1 -> trunk2 -> ... -> trunk6; ep1 after trunk2, ep2 after trunk4,
ep3 after trunk6 (`registry_3b.DEPENDS`). On Skipjack `segment_3b.sbatch` adds the resume-config rewrite
(`startup_3b.py resume-config`) and the checks; reuse that logic.

## Initialization

No init checkpoint has been created yet (it needs a GPU). `init_3b.sh` creates it once, before trunk1 of the first
setting: `startup_3b.py derive-init` writes a 1-step config from trunk1 with `save_initial_state`, nanotron saves
step 0, and the result plus `init_manifest.json` (`capped_submit_3b.init_digest`: per-file sha256) is moved to
`<CKPT_BASE>/init/_init_3B_seed42/`. Every setting's trunk1 starts weights-only from that same directory
(`startup_3b.py seed-trunk`), so all four settings share one init. `tools/hash_init_checkpoint.py` re-hashes it.

## Evaluation and conversion

* nanotron -> HF: `tools/kys_eval/convert_to_hf.py` (CPU; handles tied embeddings).
* evaluation harness (not in this repo): `imHuicongZhang/kys-eval` @ `50bb288394c16dc23b36ac1a647a9a4def0e7f87`
  (on `origin/main`); `imHuicongZhang/lighteval` @ `0228edcd06f0f3bcf88baf5948e7746577b2af71` (on
  `origin/huicong-dev`; the Skipjack checkout has 3 further uncommitted file changes that are not part of this pin).

## Known issues

* The Skipjack train8 lanes stopped at their first GPU job (startup 1187792, DIV trunk1 1187793): the nanotron dataset
  helpers (`src/nanotron/data/nemo_dataset/Makefile`) are JIT-built with `g++`, whose include path comes from
  `python3 -m pybind11 --includes` (`/usr/include/python3.11`, no `Python.h` on the compute nodes) and whose output
  suffix comes from `python3-config` on PATH (resolved to a CPython 3.14 one). `segment_3b.sbatch` exports
  `C_INCLUDE_PATH`, which `g++` does not read for C++. Elsewhere: use an environment whose `python3`/`python3-config`
  are the training Python 3.11 with its headers installed (or set `CPLUS_INCLUDE_PATH`), or pre-build the helpers.
* `test_train8_3b.py` and `test_capped_submit_3b.py` fail to collect in one pytest invocation (module name clash,
  pre-existing); run them separately.
* quality-base is a prepared setting (`registry_3b.PREPARED_SETTINGS`, configs in `seed42_prepared/`) so the live
  three-setting Skipjack lanes and their 27-config gate stay unchanged; `--grid` / `--dir` check all four.
* diversity-oriented has 19,834,314,433 train tokens (accepted 165,685,567 shortfall vs 20B); its epoch statistics are
  in the dataset repo.
