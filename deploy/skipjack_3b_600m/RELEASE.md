# KYS 3B four-setting release (code)

Branch `release/kys-3b-600m` of `imHuicongZhang/nanotron`. The data is in the **public** HF dataset
`blab-jhu/KYS-3B-pretraining-data`; its `HANDOFF_RELEASE.json` names the code commit it was verified against.
Trained models go to `blab-jhu/KYS-3B-pretraining-models` (nothing is uploaded there yet).

**Running the grid on another cluster: `deploy/portable_3b/README.md`** (install, operator config, command
sequence). Everything else in `deploy/skipjack_3b_600m/` is the historical Skipjack deployment.

## Grid (scientific definition; single source: `deploy/portable_3b/science_3b.py`)

| | |
|---|---|
| settings | quality-first, diversity-oriented, disagreement-aware (rewritten), quality-base (raw shared base + next 10B by the same fastText order) |
| model | the 1.5B llama architecture with only `num_hidden_layers` 28 -> 57 (2,994,444,288 params, tied embeddings, vocab 32000, hidden 2048, 16 attention / KV heads, FFN 5632), bf16 |
| tokenizer | the 1.5B llama-2 tokenizer, sha256-pinned in `TOKENIZER_SHA256` |
| seed / init | 42; one shared init for all four settings (std 0.02, scaled output projections) |
| batch | 1024 x 2048 = 2,097,152 tokens per update = micro_batch_size x accumulation x data-parallel size x 2048; tp = pp = 1 |
| LR (WSD) | AdamW (0.9, 0.95, 1e-8), wd 0.1, clip 1.0; warmup 1000 linear to 4e-4; constant; linear decay to 0 |
| horizons | 20B / 40B / 60B = finals 9537 / 19073 / 28610 (20,000,538,624 / 39,998,980,096 / 59,999,518,720 tokens), decay from branch points 8583 / 17166 / 25749 |
| segments | trunk1..trunk6 (ends 4292 ... 25749; trunk1 weights-only from the init, later segments and trunk1 past step 0 resume optimizer + LR + data position), ep1..ep3 branch from the fixed trunk step dirs and decay in their own namespace (never written back into the trunk) |
| grid | 4 settings x 9 segments = 36 configs, 12 annealed finals |

The LR-resume fix (`src/nanotron/serialize/optimizer.py`, off-by-one on resume) is included; regression tests
`deploy/skipjack_3b_600m/test_lr_resume_3b.py` (committed configs) and `deploy/portable_3b/test_portable_3b_lr.py`
(portable configs of all four settings, with a negative control on the unfixed resume).

## Portable vs Skipjack-only

* **Portable** (`deploy/portable_3b/`, never imports `deploy/skipjack_3b_600m/`): `science_3b.py` (the contract),
  `kys3b.py` (render / check / init / prepare / finish from one operator config), `operator.template.yaml`,
  `startup.py`, `initlib.py`, `lrsched.py`, `run_segment.sh`, `launch_torchrun.sh`, `create_init.sh`,
  `install_env.sh` + `requirements/`, `check_env.py`; plus `tools/fix_ds_metadata.py`, `tools/hash_init_checkpoint.py`,
  `tools/kys_eval/convert_to_hf.py`.
* **Skipjack-only** (`deploy/skipjack_3b_600m/`): `registry_3b.py` (Skipjack paths, Slurm site, 4-GPU layouts, W&B
  owner, live 3-setting lanes; science imported from `science_3b.py`), the committed configs under
  `configs/3B-rewrite-600m/` (Skipjack absolute paths), every `*.sbatch`, `init_3b.sh` (hard-coded Skipjack worktree),
  `train8_*`, `capped_submit_3b.py`, `handoff_*`, `smoke_*`, `release_3b.py`, `startup_3b.py` (a Skipjack CLI over
  `portable_3b/startup.py`), `generate_configs_3b.py`, `assert_invariants_3b.py`, `epoch_stats_3b.py`,
  `requirements-3b.lock.txt` (the frozen env; install from `portable_3b/requirements/` instead).
The committed Skipjack configs are byte-identical to what `generate_configs_3b.py` renders through `science_3b`, and
`test_portable_3b_render.py` asserts that every portable config has the same science as them.

## Initialization

Not created yet (it needs the GPUs of one segment). Portable procedure: `deploy/portable_3b/create_init.sh` runs the
1-step init config `<CONFIG_ROOT>/init/init.yaml`, which `kys3b.py render` derives from the rendered quality-first
trunk1 config (`startup.derive_init`: only run name, train_steps 1, the init checkpoint folder, load flags off,
`save_initial_state` on); `kys3b.py finalize-init` hashes step 0 (`initlib.init_digest`) and moves it with
`init_manifest.json` to `<INIT_ROOT>/_init_3B_seed42/` (never overwritten). Every setting's trunk1 is seeded from that
directory by `kys3b.py prepare` (digest-checked copy to `trunk/0` + `latest.txt`, never over progress).
`tools/hash_init_checkpoint.py` gives a layout-independent hash for comparing inits across sites.
On Skipjack the same functions run through `init_3b.sh` / `startup_3b.py derive-init --config <trunk1.yaml>`.

## Evaluation and conversion

* nanotron -> HF: `tools/kys_eval/convert_to_hf.py` (CPU; handles tied embeddings).
* evaluation harness (not in this repo): `imHuicongZhang/kys-eval` @ `50bb288394c16dc23b36ac1a647a9a4def0e7f87`
  (on `origin/main`); `imHuicongZhang/lighteval` @ `0228edcd06f0f3bcf88baf5948e7746577b2af71` (on
  `origin/huicong-dev`; the Skipjack checkout has 3 further uncommitted file changes that are not part of this pin).

## Known issues

* Fixed in this release, not yet exercised on a GPU node: the nanotron dataset helper (`src/nanotron/data/nemo_dataset`)
  was JIT-built with whatever `python3` / `python3-config` were first on PATH, which on Skipjack picked a CPython 3.14
  suffix and a header-less include dir and stopped the train8 lanes at their first GPU job (startup 1187792, DIV
  trunk1 1187793). The Makefile now takes `PYTHON` (compile_helper passes `sys.executable`) and `install_env.sh`
  pre-builds it with the env's interpreter. The interpreter must ship `Python.h` (the Skipjack pretrain env does not).
* `test_train8_3b.py` and `test_capped_submit_3b.py` interfere when run in one pytest invocation (pre-existing, also at
  08ebf6c7); run them separately, as the release gate does.
* quality-base is a prepared setting in the Skipjack registry (`PREPARED_SETTINGS`, configs in `seed42_prepared/`) so
  the live three-setting Skipjack lanes and their 27-config gate stay unchanged; the portable grid trains all four.
* diversity-oriented has 19,834,314,433 train tokens (accepted 165,685,567 shortfall vs 20B); its epoch statistics are
  in the dataset repo.
