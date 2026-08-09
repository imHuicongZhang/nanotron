# PATCH_NOTES — datatrove 0.5.0 public-signature adaptation (local-disk Nanoset)

**Date:** 2026-06-01. **Why:** nanotron 0.4's "new dataloader" (upstream `1dfee7f`, SmolLM3)
was written against a HuggingFace-internal **datatrove fork** whose `DatatroveFolderDataset`
had extra kwargs (`folder_path`, `max_tokens`, `eos_token_id`, `read_path`, `matched_files`,
`file_sizes`). **No public datatrove release or `main` provides that signature** — verified by
inspecting 0.3.0/0.4.0/0.5.0/0.9.0 and `main`. We adapt nanotron to the **public** datatrove
signature instead of chasing the unidentifiable fork. Validated end-to-end (tokenize → train)
with the §5.1 smoke.

## Environment pins (apply to the `nanotron-train` conda env)
```
pip install "datatrove[io]==0.5.0" "numpy==2.0.2" "huggingface_hub<1.0" pybind11
# torch stays 2.4.1+cu124; transformers 4.46.3; tokenizers 0.20.3
```
- **datatrove 0.5.0, NOT main.** datatrove `main` (0.9.0) requires `huggingface-hub>=1.5.0`,
  which breaks `transformers==4.46.3` + `tokenizers==0.20.3` (both need hub `<1.0`); we can't
  bump transformers because 5.x needs torch≥2.5. 0.5.0 is the newest datatrove with the public
  `return_positions`/`positions_from_eos_token_id` API that still allows hub `<1.0`.
- **numpy 2.0.2** (every datatrove ≥0.4 requires `numpy>=2.0.0`; kept `<2.1` for numba 0.60).
  Verified: numpy 1→2 causes **no** nanotron/torch/flash-attn/grouped_gemm/numba import regression.
  (nanotron's `pyproject` still pins `numpy<2`; that pin is stale/inconsistent with its own
  `nanosets` extra and is ignored.)
- **pybind11** — Nanoset compiles a C++ index helper at runtime via `python3 -m pybind11 --includes`
  (`src/nanotron/data/nemo_dataset/Makefile`); without it: `fatal error: pybind11/pybind11.h`.

## Source edits (5 sites; each carries an inline WHY comment)
1. **`src/nanotron/data/nanoset.py`** (~line 75): `eos_token_id=` → `positions_from_eos_token_id=`.
2. **`src/nanotron/data/tokenized_bytes.py`** `TokenizedBytesFolderDataset.super().__init__` (~420):
   `eos_token_id`→`positions_from_eos_token_id`; **drop** `max_tokens`, `read_path`,
   `matched_files`, `file_sizes` (HF-fork-only S3-offload / per-dataset cap / cached file-list).
3. **`src/nanotron/data/tokenized_bytes.py`** (just after that `super().__init__`): add
   `self.folder_path = folder_path` — public datatrove sets `self.folder_path` to a `DataFolder`
   object, but nanotron's consumption/offset accounting compares it as a string.
4. **`src/nanotron/data/nemo_dataset/blendable_dataset.py`** `get_consumption_stats` (~188):
   remove the `assert "s3" in dataset.folder_path` ("Only S3 paths…", already `# TODO: remove this`)
   so local-disk dataset folders work.
5. **`src/nanotron/data/tokenized_bytes.py`** `TokenizedBytesFolderDataset.__getitem__` (~456,
   added 2026-06-03): override to `return super().__getitem__(item % len(self))`. The public
   datatrove `DatatroveFolderDataset.__getitem__` (`datatrove/utils/dataset.py:334`) indexes files
   directly with **no wrap**, so `item >= len(self)` (one epoch) makes `bisect` walk off the file
   list → `IndexError: list index out of range`. nanotron's own datasets already wrap with
   `item % len(self)`; this subclass lost that when re-pointed at the public class. See the
   "Multi-epoch boundary crash" section below.

## Preprocessing script
- **`tools/preprocess_data_parquet.py`**: `DocumentTokenizer(shuffle=False)` →
  `shuffle_documents=False` (datatrove 0.5.0 renamed the kwarg).
- (Companion: `tools/fix_ds_metadata.py` rewrites `.ds.metadata` line 1 to the tokenizer
  **directory**; see `data_preprocessing_guide.md` §3.3/§4b.)

## Semantics confirmed
- `positions_from_eos_token_id` == old `eos_token_id` (datatrove docstring: "Token ID … marking
  the end of sequences", used to compute per-document positions). Clean 1:1 map.
- `max_tokens` is **only** datatrove's per-dataset on-disk token cap (`None` in our configs ⇒ no
  effect); it is NOT used by epoch-wrapping (driven by `train_steps` × modulo-wrap over on-disk
  tokens — but note that modulo-wrap was **not actually present** on the public folder dataset
  until source edit #5; see "Multi-epoch boundary crash" below), per-dataset consumption (keyed by
  index/folder-path), or blend ratios
  (`dataset_weights`/`dataset_lengths`). The `max_tokens` `_len`-cap at
  `tokenized_bytes.py:114` lives in `TokenizedBytesFileDataset`, used only by
  `OldTokenizedBytesFolderDataset` (`use_old_brrr_dataloader=True`, inactive).
  **Limitation:** per-dataset token capping via `dataset_max_tokens` is no longer enforced — keep
  it `None`.

## Multi-epoch boundary crash (added 2026-06-03)
**Symptom:** training dies with `IndexError: list index out of range` at
`datatrove/utils/dataset.py:334` (via `blendable_dataset.py:146`), at **exactly the 1-epoch
boundary** — for the S0 top-10B data that is **step ~4768, `consumed_tokens` 10.0B** (dataset =
shared-top-5B 5.0B + 2nd-top-5B 5.0B = 10,000,003,137 tok ÷ 2,097,152 tok/step ≈ 4768). Hit the
1.1B run 3× (jobs 1509221/1509222/1509223) and would hit the 1.5B (1509184) at the same step.

**Not a path/rename bug** — it ran 4766 steps cleanly first; a wrong folder fails at startup.

**Cause:** with `train_steps=14305` (~3 epochs / 30B over ~10B on-disk tokens), `BlendableDataset`
maps global indices to per-sub-dataset `sample_idx` values that exceed one epoch. Public datatrove's
`DatatroveFolderDataset.__getitem__` indexes directly (no wrap), so once `sample_idx >= len(dataset)`
the `bisect(self.lens, item) - 1` lands past the last file → `self.files[...]` IndexError. A bare
resume re-crashes at the same step every time.

**Fix:** source edit #5 above (modulo-wrap in `TokenizedBytesFolderDataset.__getitem__`). Restores
the multi-epoch behavior nanotron's own loaders (`TokenizedBytesFileDataset`,
`OldTokenizedBytesFolderDataset`) already have. Epochs 2/3 re-read the same token order (`shuffle`
off in the S0 configs), which is the intended 3-epoch S0 baseline. Resume jobs import the patched
source fresh and sail past step 4768; an already-running buggy process still crashes at its boundary
and must be resumed from its last checkpoint.

## Re-apply if the env is rebuilt
Re-run the pip line above, re-apply the 5 source edits (they're committed in-tree with WHY
comments), and confirm with the §5.1 end-to-end smoke (`data_preprocessing_guide.md` §5.1).
**Multi-epoch check:** if `train_steps` spans >1 epoch, confirm a run crosses the 1-epoch boundary
(S0: step ~4768 / 10B tok) without `IndexError` — that exercises source edit #5.
