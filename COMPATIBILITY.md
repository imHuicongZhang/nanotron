# Compatibility check — new clone vs. the JHU install

Run 2026-08-09, before anything was built on this clone. Verdict: **fully compatible.
Nothing changed. No config edits required.**

## Why the result is this strong

Upstream `huggingface/nanotron` `main` had **not advanced** since 2026-04-07. Its tip
was `2411b022`, which is the exact commit the JHU install descends from. So there is no
upstream drift, no refactor, and no schema change to reconcile — the "new upstream" and
"the version my runs used" are the same commit. The whole patch series applied with
**zero conflicts**.

## 1. Source-tree equivalence

```
diff -rq nanotron-kys/src/  nanotron/src/
```
Only differences are build artifacts (`__pycache__/`, `nanotron.egg-info/`,
`nemo_dataset/helpers.cpython-311-x86_64-linux-gnu.so`). **Every `.py` is byte-identical.**

## 2. Config schema — all legacy YAMLs parse

`get_config_from_file(..., skip_unused_config_keys=False, skip_null_keys=False)` over all
91 configs in `configs/legacy-arr/`:

| source | parsed | failed |
|---|---|---|
| nanotron-kys (new) | 87 | 4 |
| live JHU install (control) | 87 | 4 |

**Identical, same 4 files.** The 4 are `config_S0_1.1B{,_resume}.yaml` and
`config_S0_1.5B{,_resume}.yaml`, failing with

```
AssertionError: Model's vocab_size (32000) does not match dataset's
(['.../shared-top-5B/tokenized', '.../2nd-top-5B/tokenized']) vocab_size (None)
```

This is **pre-existing and data-side**, not a schema regression: those two old corpora have
`.ds.metadata` without a vocab_size (see `tools/fix_ds_metadata.py`). None of the six ARR
settings are affected. No field was renamed, removed, or had its default changed.

Fields used by the configs, confirmed present with unchanged meaning in
`src/nanotron/config/config.py`: `LRSchedulerArgs.{learning_rate, lr_warmup_steps,
lr_warmup_style, lr_decay_style, lr_decay_steps, lr_decay_starting_step, min_decay_lr}`,
`CheckpointsArgs.{checkpoints_path, resume_checkpoint_path, load_optimizer,
load_lr_scheduler, checkpoint_interval, save_initial_state, save_final_state}`,
`TokensArgs.*`, `ParallelismArgs.*`, `OptimizerArgs.*`, `GeneralArgs.seed`,
`DataArgs.seed`.

## 3. `.ds` dataset format — reads correctly

`TokenizedBytesFolderDataset` under datatrove 0.5.0, all six ARR corpora:

| setting | sequences | tokens | `folder_path` is `str` | `ds[len(ds)] == ds[0]` |
|---|---:|---:|:--:|:--:|
| quality_base | 4,894,576 | 10.024 B | yes | yes |
| quality_first | 4,908,594 | 10.053 B | yes | yes |
| diversity_oriented | 4,850,875 | 9.935 B | yes | yes |
| wrap | 4,913,882 | 10.064 B | yes | yes |
| rewire | 4,912,456 | 10.061 B | yes | yes |
| disagreement_aware_0p5 | 4,904,946 | 10.045 B | yes | yes |

The last two columns exercise patch #3 (`folder_path` restored to `str`) and patch #5
(multi-epoch modulo wrap) respectively.

## 4. Bit-identical reads (A/B)

Rolling SHA-256 over samples `{0, 1, 12345, len-1}` from each of the six corpora plus each
dataset length, computed once against each source tree:

```
OLD (live install)   1dc9ac16101c7b275c7ecbe3838ef30c0c86a6a112fa44faf34bb5ca568d4e05
NEW (nanotron-kys)   1dc9ac16101c7b275c7ecbe3838ef30c0c86a6a112fa44faf34bb5ca568d4e05
```

**Identical.** The clone consumes the existing tokenized data byte-for-byte as the live
install does.

## Two operational notes

1. **The compiled Nanoset helper is not in this clone.** The live install has a built
   `src/nanotron/data/nemo_dataset/helpers.cpython-311-x86_64-linux-gnu.so`; a fresh clone
   compiles it at first use via that directory's `Makefile`, which needs `pybind11`
   installed (it is, in `nanotron-train`). The `TokenizedBytes` path used by all six ARR
   settings does not need it — verified above without it present.
2. **This clone is not `pip install -e`'d, deliberately.** Both conda envs still point at
   `/scratch/.../projects/nanotron`. To run from this clone, either
   `pip install -e /scratch/bvandur1/zhuicon1/projects/nanotron-kys` in a *new* env, or
   prepend `nanotron-kys/src` to `PYTHONPATH`. Do not re-point the existing envs without
   deciding you want to.
