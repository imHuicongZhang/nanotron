# Data Preprocessing Guide — Nanotron Nanosets

**Goal:** turn raw document folders (a `text` column) into the tokenized, packed,
memory-mapped binary format Nanotron's `Nanoset` dataset reads during pretraining.

Everything below is grounded in the code in *this* checkout. File paths and argument
names are real and were read directly from the repo / installed packages. Where I am
not 100% certain about runtime behavior, it is marked **⚠️ VERIFY** with the exact file
to check — do not treat those as settled.

---

## 0. What your inputs actually are (verified)

| Block | Path | Format | Files | Size | Tokens (per `_manifest.json`) |
|---|---|---|---|---|---|
| shared-top-5B | `/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/5B/shared-top-5B` | **Parquet** | 200 × `part_*.parquet` + `_manifest.json` | ~8.7 GB | 5,000,002,332 |
| quality-first | `/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/5B/quality-first` | **Parquet** | 200 × `part_*.parquet` + `_manifest.json` | ~8.5 GB | (see its manifest) |

Parquet schema (verified from `part_00000.parquet`): the text lives in the **`text`**
column. (Other columns like `tokens-llama2`, `url`, `topic`, … are ignored by tokenization.)

> **This matters:** the bundled preprocessing script supports only Hugging Face datasets
> and `.jsonl` — **not Parquet**. See §3.1 for how to handle Parquet. Do not assume the
> `jsonl` subcommand will read your `.parquet` files; it will not.

---

## 1. Which Nanotron scripts/components handle preprocessing

The preprocessing is done by **`datatrove`** (HuggingFace's data pipeline library), driven
by a thin Nanotron wrapper. The relevant pieces:

| Component | Path | Role |
|---|---|---|
| Preprocessing CLI | `tools/preprocess_data.py` | Wires a datatrove **reader** → **tokenizer** and runs it locally. |
| datatrove tokenizer | `datatrove.pipeline.tokens.DocumentTokenizer` | Tokenizes each document, packs token IDs, writes the binary `.ds` files + index + metadata. |
| datatrove readers | `datatrove.pipeline.readers.{HuggingFaceDatasetReader, JsonlReader, ParquetReader, …}` | Stream raw documents from disk/hub. `tools/preprocess_data.py` only exposes `hf` and `jsonl`. |
| tokenizer loader | `datatrove.utils.tokenization.load_tokenizer` | `Tokenizer.from_file(path)` if the path exists locally, else `Tokenizer.from_pretrained(name)`. Uses the `tokenizers` (Rust) library directly. |
| Train-time dataset | `src/nanotron/data/nanoset.py` (`Nanoset`) + `src/nanotron/data/tokenized_bytes.py` (`DatatroveFolderDataset`) | Reads the `.ds` files at training time, builds `seq_len + 1` token samples, mixes multiple folders by weight. |
| Config dataclass | `src/nanotron/config/config.py` → `NanosetDatasetsArgs` | The `data_stages[].data.dataset` block in your YAML. |
| Docs | `docs/nanoset.md` | Official description of the format and the 3 ways to specify `dataset_folder`. |

There is **no** standalone "pack" or "index" script — packing and indexing happen *inside*
`DocumentTokenizer` in a single pass.

---

## 2. What the pipeline does, stage by stage

```
raw docs (parquet/jsonl/hf)              tokenized binary (memory-mapped)
        │                                          │
   [Reader] ── stream {text, id, metadata} ──▶ [DocumentTokenizer] ──▶ output_folder/
   JsonlReader / ParquetReader /                    │                    ├── NNNNN_<name>.ds
   HuggingFaceDatasetReader                          │                    ├── NNNNN_<name>.ds.index
                                                     │                    └── NNNNN_<name>.ds.metadata
```

1. **Read** — the reader yields one `Document` per row, pulling the text from `text_key`
   (default `"text"`). `datatrove.utils.tokenization.load_tokenizer` is *not* used by the
   reader; it's used by the tokenizer step.

2. **Tokenize** — `DocumentTokenizer.tokenizer.encode_batch(...)` encodes a batch of
   documents into token IDs using the **`tokenizers`** Rust library. If `--eos-token` is
   given, a `TemplateProcessing` post-processor appends that EOS id after each document
   (`datatrove/utils/tokenization.py`, `tokenizer` property). **If you don't pass an EOS
   token, documents are concatenated with no separator** — for pretraining you almost
   always want an EOS between documents.

3. **Pack + size** — token IDs are written as a contiguous little-endian array.
   The per-token width is chosen automatically (`PipelineStepWithTokenizer.token_size`):
   - `vocab_size ≤ 65535` → **`uint16`, 2 bytes/token** (our case: vocab 32000 → 2 bytes)
   - else → `uint32`, 4 bytes/token

   So expected output size ≈ `num_tokens × 2 bytes`. For a 5B-token block that's **~10 GB**
   of `.ds` data (plus small index/metadata).

4. **Index** — for each `.ds` file a `.ds.index` records document boundaries (start/end
   offsets), so the dataset can recover individual documents. (Written when `shuffle=False`,
   which is what `tools/preprocess_data.py` sets.)

5. **Metadata** — a `.ds.metadata` file stores `tokenizer_name|token_size_in_bytes` on its
   first line, plus token counts. **Nanotron reads this at train time** to (a) recover the
   tokenizer name and (b) derive `vocab_size` — see `NanosetDatasetsArgs.__post_init__`
   (`src/nanotron/config/config.py`, it does
   `vocab_size = len(AutoTokenizer.from_pretrained(tokenizer_name).get_vocab())`).

**Output file naming:** each of the `--n-tasks` workers writes its own shard
(`00000_<name>.ds`, `00001_<name>.ds`, …). `tools/preprocess_data.py` sets
`max_tokens_per_file=1e9`, so a worker that produces >1B tokens will roll over into
multiple numbered files. All shards living in one `output_folder` form one logical dataset
that you point `dataset_folder` at.

---

## 3. Prerequisites you must resolve BEFORE running

### 3.1 ⚠️ Parquet is not supported by the bundled script

`tools/preprocess_data.py` defines only two subcommands (read it — the `sp.add_parser`
calls): `hf` and `jsonl`. Your data is Parquet. Three options, in order of preference:

**Option A (recommended): add a `parquet` subcommand.** datatrove already ships
`ParquetReader` (`datatrove/pipeline/readers/parquet.py`, signature verified:
`ParquetReader(data_folder, text_key="text", glob_pattern=None, ...)` — same shape as
`JsonlReader`). Copy `tools/preprocess_data.py` to e.g. `tools/preprocess_data_parquet.py`
and make two changes:

```python
# add to the imports
from datatrove.pipeline.readers import ParquetReader

# add a third subparser next to p1 ("hf") and p2 ("jsonl"):
p3 = sp.add_parser(name="parquet")
p3.add_argument("--dataset", type=str, required=True,
                help="Folder containing .parquet files")
p3.add_argument("--column", type=str, default="text")
p3.add_argument("--glob-pattern", type=str, default=None)

# and in main(), add the branch:
elif args.readers == "parquet":
    datatrove_reader = ParquetReader(
        data_folder=args.dataset, text_key=args.column, glob_pattern=args.glob_pattern,
    )
```

I am **describing** this change, not making it — you write the file and **⚠️ VERIFY** the
`ParquetReader` kwargs against your installed datatrove version
(`…/site-packages/datatrove/pipeline/readers/parquet.py` and `base.py`).

**Option B:** convert Parquet → JSONL first (e.g. with `pyarrow`/`pandas`), then use the
existing `jsonl` subcommand. Wastes ~10s of GB of scratch and time; not recommended at 5B scale.

**Option C:** try the existing `hf` subcommand pointing at the folder (it calls
`datasets.load_dataset(<path>, split="train")`). Whether `load_dataset` auto-detects a bare
folder of `part_*.parquet` is version-dependent — **⚠️ VERIFY** on a 1-file copy before
trusting it. Option A is cleaner.

### 3.2 ✅ RESOLVED — use the `nanotron-train` env (transformers 4.46.3 / tokenizers 0.20.3)

Background (verified):
- The tokenizer at `/scratch/bvandur1/zhuicon1/tokenizers/llama2-unsloth-tokenizer`
  (source `unsloth/llama-2-7b`, **vocab = 32000**) has a `tokenizer.json` that
  `tokenizers==0.19.1` **fails** to parse:
  `Exception: data did not match any variant of untagged enum ModelWrapper`.
- `tokenizers==0.20.3` loads it fine (`get_vocab_size() == 32000`).

**Resolution (tested):** we use a dedicated env **`nanotron-train`** =
`/scratch/bvandur1/zhuicon1/basic/miniconda3/envs/nanotron-train`. It was built by cloning the
original `…/envs/nanotron` env and upgrading it to **`transformers==4.46.3` +
`tokenizers==0.20.3`** with **torch held at 2.4.1**. Compatibility results (all PASS):
`import torch` → 2.4.1+cu124;
`import nanotron` + `nanotron.trainer`/`models.llama`/`models.qwen` import cleanly;
`AutoTokenizer.from_pretrained(<dir>)` → 32000; `Tokenizer.from_file(<json>)` → 32000.
`pip check` clean. The original `…/envs/nanotron` (transformers 4.44.2 / tokenizers 0.19.1)
is kept **unchanged as backup**. **This single `nanotron-train` env is used for BOTH
preprocessing and training.** Activate it (not the old env) everywhere below.

### 3.3 ✅ CONFIRMED — tokenizer path conflict (file vs directory) + the metadata-rewrite fix

**This is a hard conflict; it was tested in `nanotron-train` (tokenizers 0.20.3 / transformers
4.46.3). No single local path string works for both stages.**

Two loaders want different things:

| Stage | Code | Loader | DIR | `tokenizer.json` FILE |
|---|---|---|---|---|
| **Preprocessing** | `datatrove/utils/tokenization.py::load_tokenizer` → `Tokenizer.from_file` | tokenizers (Rust) | ❌ `Is a directory (os error 21)` | ✅ vocab 32000 |
| **Training** | `src/nanotron/config/config.py:192` → `AutoTokenizer.from_pretrained` | transformers | ✅ vocab 32000 | ❌ `OSError: Incorrect path_or_model_id` |

```python
# datatrove/utils/tokenization.py
def load_tokenizer(name_or_path):
    if os.path.exists(name_or_path):
        return Tokenizer.from_file(name_or_path)   # needs the .json FILE (a dir errors)
    return Tokenizer.from_pretrained(name_or_path)  # hub id
```

**The trap (verified path):** datatrove writes the string you pass to
`--tokenizer-name-or-path` *verbatim* into the first line of each `.ds.metadata` as
`<tokenizer_string>|<token_size_bytes>` (datatrove `TokenizedFile.write_final_metadata`).
At train time `NanosetDatasetsArgs.__post_init__` reads that line, `split("|")`, and runs:
```python
# src/nanotron/config/config.py  (lines 188–199)
tokenizer_name, token_size_in_bytes = first_line.split("|")
if self.tokenizer_name is None:
    self.tokenizer_name = tokenizer_name
    self.token_size_in_bytes = int(token_size_in_bytes)
    self.vocab_size = len(AutoTokenizer.from_pretrained(tokenizer_name).get_vocab())   # ← line 192
else:
    assert self.tokenizer_name == tokenizer_name, "Tokenizer name mismatch ..."         # ← lines 194–196
    assert self.token_size_in_bytes == int(token_size_in_bytes), "Token size mismatch ..."  # ← 197–199
```
So whatever string datatrove stored is fed to `AutoTokenizer.from_pretrained` (line 192), and
**lines 194–196 assert that every `.ds.metadata` across all `dataset_folder`s carries the
identical tokenizer string** (and 197–199 the identical token size). If you preprocess with
the only string datatrove accepts (the `.json` file), line 192 crashes at config load.

> We deliberately **reject** the hub-id alternative (`unsloth/llama-2-7b`): it needs internet
> at both stages and risks the local `tokenizer.json` differing byte-wise from the hub copy.

#### The fix (3 steps) — preprocess with `.json`, rewrite metadata to the DIR, train with the DIR
1. **Preprocess** with the **`tokenizer.json` FILE** (the only value datatrove can load):
   `--tokenizer-name-or-path /scratch/bvandur1/zhuicon1/tokenizers/llama2-unsloth-tokenizer/tokenizer.json`
   → metadata line 1 becomes `…/llama2-unsloth-tokenizer/tokenizer.json|2`.
2. **Rewrite** line 1 of every `*.ds.metadata` to the **DIRECTORY** string, preserving the
   `|2` suffix, using `tools/fix_ds_metadata.py` (see §4b). Apply the **same fixed directory
   string to every block folder** so the cross-folder assert (lines 194–196) holds:
   → metadata line 1 becomes `/scratch/bvandur1/zhuicon1/tokenizers/llama2-unsloth-tokenizer|2`.
3. **Train** with YAML `tokenizer.tokenizer_name_or_path:
   /scratch/bvandur1/zhuicon1/tokenizers/llama2-unsloth-tokenizer` (the **directory**), which
   matches the rewritten metadata (passes the Config-level assert) and loads via
   `AutoTokenizer.from_pretrained` (verified PASS). `token_size_in_bytes` stays `2`.

---

## 4. The reusable preprocessing command (parameterized)

Define variables so the **same** commands work for every block now and later (other 5B
blocks, rewritten data). Nothing here is hardcoded to shared-top-5B / quality-first.

```bash
# --- activate the env (nanotron-train: used for BOTH preprocessing and training; see §3.2) ---
source /scratch/bvandur1/zhuicon1/basic/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/bvandur1/zhuicon1/basic/miniconda3/envs/nanotron-train

# --- parameters: change ONLY these per block ---
export INPUT_DIR="/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/5B/shared-top-5B"
export OUT_ROOT="/scratch/bvandur1/zhuicon1/<YOUR_OUTPUT_ROOT>"      # <-- you choose; must NOT be a source folder
export BLOCK_NAME="$(basename "$INPUT_DIR")"                          # -> shared-top-5B
export OUT_DIR="${OUT_ROOT}/${BLOCK_NAME}/tokenized"
export LOG_DIR="${OUT_ROOT}/${BLOCK_NAME}/logs"

# tokenizer + tokenization settings
export TOKENIZER="/scratch/bvandur1/zhuicon1/tokenizers/llama2-unsloth-tokenizer/tokenizer.json"  # the .json FILE for datatrove (§3.3); training uses the DIR after §4b
export EOS_TOKEN="</s>"     # verified eos for this tokenizer (id 2); appended between docs
export N_TASKS=32           # parallel workers; tune to your CPU allocation (you have 108 cores on cpu002)

mkdir -p "$OUT_DIR" "$LOG_DIR"
```

**Run (Parquet, using the `parquet` subcommand you added in §3.1 Option A):**
```bash
cd /scratch/bvandur1/zhuicon1/projects/nanotron
python3 tools/preprocess_data_parquet.py \
    --tokenizer-name-or-path "$TOKENIZER" \
    --eos-token "$EOS_TOKEN" \
    --output-folder "$OUT_DIR" \
    --logging-dir "$LOG_DIR" \
    --n-tasks "$N_TASKS" \
    parquet \
    --dataset "$INPUT_DIR" \
    --column text \
    --glob-pattern "*.parquet"
```

**Real argument names** (verified in `tools/preprocess_data.py::get_args`):
`--tokenizer-name-or-path` (required), `--eos-token` (default `None`),
`--output-folder` (required), `--logging-dir` (default `None`),
`--n-tasks` (default 8). The reader subcommand (`hf` / `jsonl` / your added `parquet`) is
**required** and comes after the global flags. Per-reader flags: `--dataset` (required),
`--column` (default `text`), and `--glob-pattern` (jsonl/parquet only).

**To run the second block, change one line and re-run the same command:**
```bash
export INPUT_DIR="/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/5B/quality-first"
# (OUT_DIR/LOG_DIR/BLOCK_NAME recompute from INPUT_DIR; re-export them, then re-run the python command)
```

**Source folders are never written to** — the script only reads `--dataset` and writes to
`--output-folder`. Keep `OUT_ROOT` on scratch and distinct from the data_rewrite tree.

> **Run it on a CPU allocation, not the login node.** This is CPU-bound and long. You are
> already inside a CPU `srun` (cpu002, 108 cores) — good. For a fresh allocation use your
> `init_compute_node.sh cpu <N>`. No GPU needed for tokenization.

## 4b. Rewrite metadata to the tokenizer DIRECTORY (`tools/fix_ds_metadata.py`)

After tokenizing a block (§4, which used the `.json` path), run the fixer to make the
`.ds.metadata` files point at the **directory** so training can load them (§3.3 step 2).
The fixed directory string is a **parameter**, so apply the identical string to every block:

```bash
export TOK_DIR="/scratch/bvandur1/zhuicon1/tokenizers/llama2-unsloth-tokenizer"   # NO trailing slash, NO /tokenizer.json

python3 tools/fix_ds_metadata.py --output-folder "$OUT_DIR" --tokenizer-dir "$TOK_DIR"
# repeat for each block's OUT_DIR using the SAME --tokenizer-dir
```

What it does (and asserts): for every `*.ds.metadata` under `--output-folder`, it replaces the
path part of line 1 with `--tokenizer-dir`, **preserves the `|<token_size>` suffix** (e.g. `|2`),
then re-reads all files and asserts (a) line 1 is **byte-identical** across every metadata file
in the folder, and (b) the original `|<token_size>` suffix survived. It refuses to run if a
metadata file's line 1 has no `|`. Idempotent: re-running on already-fixed files is a no-op.
**You must pass the SAME `--tokenizer-dir` to every block** or the cross-folder assert at
`config.py:194–196` will fire when you blend folders.

---

## 5. Verify before you commit hours

### 5.1 END-TO-END smoke gate (10 docs → tokenize → fix metadata → train a few steps)

**Do not tokenize the full 5B blocks until this whole sequence passes.** It exercises every
moving part: the parquet reader, the `.json`-path tokenization, `fix_ds_metadata.py`, the
`config.py:194-199` tokenizer/size asserts, `AutoTokenizer.from_pretrained(<dir>)`, and the
Nanoset dataloader actually reading tokens.

**Step 1 — make a 10-document parquet subset** (does not touch the source folder):
```bash
export SMOKE_ROOT="${OUT_ROOT}/_smoke"
export SMOKE_SRC="${SMOKE_ROOT}/src"; export SMOKE_OUT="${SMOKE_ROOT}/tokenized"; export SMOKE_LOG="${SMOKE_ROOT}/logs"
mkdir -p "$SMOKE_SRC" "$SMOKE_OUT" "$SMOKE_LOG"
python3 - "$INPUT_DIR" "$SMOKE_SRC" <<'PY'
import sys, glob, os, pyarrow.parquet as pq, pyarrow as pa
src_glob = sorted(glob.glob(os.path.join(sys.argv[1], "*.parquet")))[0]
t = pq.read_table(src_glob, columns=["text"]).slice(0, 10)     # first 10 docs, text column only
pq.write_table(t, os.path.join(sys.argv[2], "part_smoke.parquet"))
print("wrote", t.num_rows, "docs to", sys.argv[2])
PY
```

**Step 2 — tokenize the 10 docs with the `.json` path** (single task):
```bash
python3 tools/preprocess_data_parquet.py \
    --tokenizer-name-or-path "$TOKENIZER" \
    --eos-token "$EOS_TOKEN" \
    --output-folder "$SMOKE_OUT" --logging-dir "$SMOKE_LOG" --n-tasks 1 \
    parquet --dataset "$SMOKE_SRC" --column text --glob-pattern "*.parquet"
head -1 "$SMOKE_OUT"/*.ds.metadata    # expect: <…>/tokenizer.json|2
```

**Step 3 — rewrite metadata to the tokenizer DIRECTORY** (§4b):
```bash
python3 tools/fix_ds_metadata.py --output-folder "$SMOKE_OUT" --tokenizer-dir "$TOK_DIR"
head -1 "$SMOKE_OUT"/*.ds.metadata    # expect: <…>/llama2-unsloth-tokenizer|2  (byte-identical across files)
```

**Step 4 — train a few steps on the 10 docs** with a minimal Nanoset config. Save this as
`examples/config_smoke_nanoset.yaml` and set `dataset_folder`/paths to your `$SMOKE_OUT`:
```yaml
checkpoints: {checkpoint_interval: 1000, checkpoints_path: "<SMOKE_ROOT>/ckpt", save_initial_state: false}
data_stages:
- name: smoke
  start_training_step: 1
  data:
    dataset: {dataset_folder: "<SMOKE_ROOT>/tokenized"}   # <- your $SMOKE_OUT
    num_loading_workers: 1
    seed: 42
general: {project: smoke, run: nanoset_smoke, seed: 42, ignore_sanity_checks: true}
logging: {iteration_step_info_interval: 1, log_level: info, log_level_replica: info}
model:
  dtype: bfloat16
  make_vocab_size_divisible_by: 1
  init_method: {std: 0.02}
  model_config:
    is_llama_config: true
    hidden_size: 256
    num_hidden_layers: 2
    num_attention_heads: 4
    num_key_value_heads: 4
    intermediate_size: 512
    max_position_embeddings: 128
    vocab_size: 32000              # MUST equal tokenizer vocab; Nanoset asserts this
    tie_word_embeddings: false
    rms_norm_eps: 1.0e-5
    hidden_act: silu
    bos_token_id: 1
    eos_token_id: 2
optimizer:
  zero_stage: 0
  weight_decay: 0.01
  clip_grad: 1.0
  accumulate_grad_in_fp32: true
  learning_rate_scheduler: {learning_rate: 0.0003, lr_warmup_steps: 1, lr_warmup_style: linear, lr_decay_style: cosine, min_decay_lr: 1.0e-5}
  optimizer_factory: {name: adamW, adam_beta1: 0.9, adam_beta2: 0.95, adam_eps: 1.0e-8, torch_adam_is_fused: true}
parallelism: {dp: 1, pp: 1, tp: 1, expert_parallel_size: 1, pp_engine: 1f1b, tp_mode: REDUCE_SCATTER, tp_linear_async_communication: true}
tokenizer:
  tokenizer_name_or_path: /scratch/bvandur1/zhuicon1/tokenizers/llama2-unsloth-tokenizer   # the DIRECTORY (matches fixed metadata)
tokens: {sequence_length: 128, micro_batch_size: 1, batch_accumulation_per_replica: 1, train_steps: 5, val_check_interval: -1, limit_val_batches: 0, limit_test_batches: 0}
```
Launch on **one GPU** with the **`nanotron-train`** env (submit via `sbatch` — this session is
inside a CPU allocation, so a nested `srun` cannot get a GPU; see how the training smoke was run):
```bash
source /scratch/bvandur1/zhuicon1/basic/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/bvandur1/zhuicon1/basic/miniconda3/envs/nanotron-train
export CUDA_HOME="$CONDA_PREFIX" CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_MODE=disabled
cd /scratch/bvandur1/zhuicon1/projects/nanotron
python -u -m torch.distributed.run --nproc_per_node 1 --nnodes 1 --rdzv_backend c10d --max_restarts 0 \
    run_train.py --config-file examples/config_smoke_nanoset.yaml
```

**PASS criteria — all must hold:**
- No `Tokenizer name mismatch` / `Token size mismatch` assertion (config.py:194-199 satisfied);
- the log shows the tokenizer loading via `AutoTokenizer.from_pretrained` with no `path_or_model_id` error, and the model builds with `vocab_size 32000`;
- training reaches `iteration: 1 / 5 … lm_loss: …` and runs all 5 steps (proves the Nanoset read real tokens).

If the dataloader complains there are too few samples, either lower `sequence_length` (e.g. 64)
or put more than 10 docs in the subset (Step 1 `slice(0, N)`).

**Gate:** only after this passes do you run §4 on the full `shared-top-5B` and `quality-first`
blocks (then §4b on each, with the **same** `--tokenizer-dir`).

### 5.2 Sanity-check token counts vs the manifest
Each block's `_manifest.json` states `train_tokens_sum` (shared-top-5B: 5,000,002,332).
The `.ds.metadata` files record per-shard token counts; summing them should land near the
manifest's value (EOS tokens add ~1 per document, so expect a small positive delta of about
`num_docs` — shared-top-5B has 4,120,164 docs). A large mismatch means a wrong `--column`,
truncation, or a reader problem.

---

## 6. Pointing training at the output

In the **training YAML**, the dataset is configured by a single field, `dataset_folder`,
inside `data_stages[].data.dataset` (a `NanosetDatasetsArgs`). Three forms (from `docs/nanoset.md`):

```yaml
# Single block:
data_stages:
- name: Stable Training Stage
  start_training_step: 1
  data:
    dataset:
      dataset_folder: /scratch/.../OUT_ROOT/shared-top-5B/tokenized
    num_loading_workers: 1
    seed: 42

# Multiple blocks, each sample consumed once per epoch (list form):
    dataset:
      dataset_folder:
        - /scratch/.../OUT_ROOT/shared-top-5B/tokenized
        - /scratch/.../OUT_ROOT/quality-first/tokenized

# Weighted blend (dict form: folder -> weight):
    dataset:
      dataset_folder:
        /scratch/.../OUT_ROOT/shared-top-5B/tokenized: 0.5
        /scratch/.../OUT_ROOT/quality-first/tokenized: 0.5
```

You must also set, in the same YAML (enforced by asserts in `src/nanotron/config/config.py`):
- `tokenizer.tokenizer_name_or_path` = the **same** string stored in `.ds.metadata` (§3.3);
- `model.model_config.vocab_size` = **32000** (must equal the tokenizer's vocab — see the
  model config guide). It is read/validated against the dataset metadata.

A complete reference YAML lives at `examples/config_nanoset.yaml`.

---

## 7. Quick reference — real names only

- Script: `tools/preprocess_data.py` (you add `tools/preprocess_data_parquet.py` for Parquet).
- Flags: `--tokenizer-name-or-path`, `--eos-token`, `--output-folder`, `--logging-dir`,
  `--n-tasks`; subcommands `hf|jsonl|parquet`; per-reader `--dataset`, `--column`, `--glob-pattern`.
- Output triplet: `*.ds`, `*.ds.index`, `*.ds.metadata` (token width 2 bytes here).
- Train-time dataset class: `Nanoset` (`src/nanotron/data/nanoset.py`); folder reader
  `DatatroveFolderDataset` (`src/nanotron/data/tokenized_bytes.py`).
- Config: `NanosetDatasetsArgs.dataset_folder` (str | list | dict-with-weights).
- **⚠️ VERIFY** items: Parquet reader kwargs (§3.1); tokenizers≥0.20.3 + transformers bump
  (§3.2); tokenizer path file-vs-dir for both loaders (§3.3). Smoke-test first (§5).

---

# Appendix A — Token behavioral confirmation (special tokens, packing, doc boundaries)

> Source of truth: nanotron checkout `src/...` + **installed datatrove 0.5.0** in `nanotron-train`
> (`.../envs/nanotron-train/lib/python3.11/site-packages/datatrove/...`). Line numbers are from
> these files as of 2026-06-01. Confirmed by reading source, not from memory.

## A.1 Special tokens at preprocess time
**One trailing EOS per document; NO BOS.**
- `datatrove/utils/tokenization.py:78-99` — the tokenizer's post-processor is set to
  `TemplateProcessing(single="$A <EOS>", special_tokens=[("<EOS>", token_to_id(eos_token))], pair=None)`
  when `--eos-token` is given. `$A <EOS>` = document tokens then EOS; **no leading BOS**. This line
  **overwrites** any BOS-adding post-processor baked into `tokenizer.json`, and in the `tokenizers`
  Rust lib special tokens are inserted only via the post-processor ⇒ no BOS.
- Applied at encode: `datatrove/pipeline/tokens/tokenizer.py:403`
  (`self.tokenizer.encode_batch([document.text ...])`).
- Our `tools/preprocess_data_parquet.py` passes `eos_token="</s>"`, no `post_processor`, and has no
  `--add-bos` flag ⇒ EOS-only.
- **`tokens-llama2` (computed with `add_special_tokens=False`) + 1 is correct**: the `+1` is exactly
  the one EOS datatrove appends; there is no BOS, so it is **+1, not +2**. Matches the
  `(tokens-llama2 + 1)` budget in `SELECTION_REPORT.md`.
- Token width: 2 bytes / uint16 (vocab 32000 ≤ 65535) — `tokenization.py:60-67`; matches `|2` in `.ds.metadata`.

## A.2 Packing / truncation (concatenate-then-chunk)
Each `.ds` file is one contiguous token stream (per-document EOS-terminated streams written back-to-back),
sliced into consecutive non-overlapping windows of **`seq_len + 1`** tokens. `datatrove/utils/dataset.py`:
- `dataset.py:62-63` — `num_tokens = fsize // token_size; self._len = num_tokens // (seq_len + 1)` (floor).
- `dataset.py:168-175` — `chunk_size = token_size * (seq_len + 1); seek(item*chunk_size); read(chunk_size)`.
- Sample length **`seq_len+1`** (the extra token is for shifted LM labels; the collator drops last input /
  first label — `clm_collator.py:142-146`).

**No padding.** **BUT the per-file trailing remainder IS discarded** (`// (seq_len+1)` floors), i.e. the last
`num_tokens mod (seq_len+1)` tokens of each `.ds` file (each block is one file) never form a sample
(≤ 2048 tokens for seq_len=2048 — negligible). Documents are **not** individually discarded — they span samples.

Example (seq_len=2048 ⇒ window=2049): doc A=15 tok, doc B=2600 tok (each incl. EOS):
- sample 0 = stream `[0:2049)` = A(15) + B's first 2034 tokens;
- sample 1 = stream `[2049:4098)` = B's remaining 566 tokens + start of next doc.
B spans both samples; no pad inserted; remainder not dropped.

## A.3 Document boundaries (LLaMA model — what you are training)
**Attention does NOT respect document boundaries and RoPE does NOT reset per document.** The
`return_positions`/`positions_from_eos_token_id`/`use_doc_masking` machinery is wired through datatrove +
the collator, but the **LLaMA training forward never consumes the per-document `position_ids`** — the only
boundary effect reaching the model is **loss masking**.
- Produced: `return_positions` default **True** (`config/config.py:157-159`); datatrove resets positions at
  the EOS id (`dataset.py:129-151`); collator `DataCollatorForCLMWithPositionIds`
  (`clm_collator.py:138`, `use_doc_masking=True` line 151; chosen in `dataloader_builder.py:16-21`) builds a
  **label_mask** that drops the loss on each document's first token (`clm_collator.py:229-244`).
- NOT done by Llama (`models/llama.py`): `_forward_training` (675-717) applies RoPE via
  `flash_rotary_embedding(query_states, kv=...)` (line 683) **without** position_ids ⇒ sequential RoPE across
  the whole window; attention `cu_seqlens` is built from the **padding** `sequence_mask` (lines 278-281) ⇒
  one segment per batch row (full `seq_len`), so all docs in a window attend to each other causally.
  `position_ids` is not referenced anywhere in the Llama model forward; `models_config.py:156`
  `_use_doc_masking` is defined but never read; `llama.py:413` carries `TODO ... position_ids not supported yet`.
- **Net:** packed window = one sequence for attention (crosses EOS), RoPE continuous; document boundaries
  affect **only the loss** (boundary token masked).
- **Not in scope / different model:** true varlen per-document attention exists only in `models/qwen.py`
  (`flash_attn_varlen_kvpacked_func` + llama3 ring-attention), NOT in the `is_llama_config` path. Verify in
  `qwen.py` before relying on it.

### Summary
| item | result |
|---|---|
| EOS per doc | yes (1, trailing) | 
| BOS per doc | no |
| `tokens-llama2 + 1` | correct (EOS; not +2) |
| packing | concatenate-then-chunk, window = seq_len+1 |
| padding | none |
| remainder | per-file trailing `<(seq_len+1)` tokens dropped; docs span samples |
| doc-boundary attention (Llama) | none (full-window causal) |
| RoPE reset per doc (Llama train) | none (sequential) |
| doc-boundary effect (Llama) | loss masking only (label_mask drops each doc's first-token loss) |

---

# Caveats

Reproducible-but-surprising gotchas hit during setup. All confirmed against this checkout / the
`nanotron-train` env (datatrove 0.5.0).

## Caveat 1 — datatrove silently SKIPS re-runs when `--logging-dir` has completion markers
datatrove writes per-task completion markers into `--logging-dir`. On a re-run with the **same**
`--logging-dir`, it sees the markers, skips all work, prints `Not doing anything as all tasks completed`,
and **exits 0 with no output** — silently, no error.

Consequences:
- Re-tokenizing a **different** block while reusing a logging-dir → the new block is **not** tokenized
  (output empty/incomplete), yet the command "succeeds". You may only notice when training can't find data.
- Re-tokenizing the **same** block after a partial/failed run, reusing the logging-dir → it skips the
  parts marked done and produces **incomplete** output.

Rules:
1. Use a **fresh per-block** `--logging-dir` every run, e.g. `logs/tokenize_<block_name>`
   (`logs/tokenize_shared-top-5B`, `logs/tokenize_2nd-top-5B`, …). Never reuse across blocks.
2. To re-tokenize a block, **delete or rename** its old `--logging-dir` first.
3. After each run, **verify output** — sum the `.ds.metadata` token counts and confirm it is non-zero and
   roughly matches the block's expected tokens. **Do not trust exit code 0 alone.**

## Caveat 2 — Llama training does NOT respect document boundaries (cross-doc attention + non-resetting RoPE)
With concatenate-then-chunk packing (Appendix A.2), one `seq_len` window holds several unrelated documents
separated by EOS. In the Llama training path (`models/llama.py`), the per-document
`position_ids`/doc-masking machinery (`return_positions`, `positions_from_eos_token_id`, `use_doc_masking`)
is produced by datatrove + the collator but is **not consumed** by the model forward
(`llama.py` carries `TODO: position_ids not supported yet`). Concretely:
- **Attention** treats the whole window as one continuous sequence; EOS is just a token, not a barrier — a
  token in document B can attend across the EOS to tokens in unrelated document A (full-window causal).
- **RoPE** positions do **not** reset at EOS; they increase continuously across the window, so document B's
  first token gets a non-zero position instead of restarting at 0.
- The **only** doc-boundary effect reaching the Llama model is **loss masking**: the collator's `label_mask`
  drops the loss on each document's first token (`clm_collator.py`). True per-document varlen attention
  exists only in `models/qwen.py`, **not** in the `is_llama_config` path.

Implications for experiments:
- This is a known **limitation** of this Llama implementation (the TODO), not a bug, and is standard for much
  pretraining. It does **not** break cross-setting comparability — every setting (S0/S1/S2/S5/diversity) sees
  the same full-window attention, so **relative** conclusions (which setting is better) remain valid.
- It becomes a **confound only if a setting changes the document-LENGTH distribution.** Our docs are short
  (median ~526 tokens), so a ~2048 window packs ~4 docs and cross-doc contamination is heavy. If a rewrite
  makes docs longer/shorter, that setting packs fewer/more docs per window → lighter/heavier contamination,
  so an eval gap could partly reflect contamination level rather than data quality. When interpreting
  rewrite-setting results, check whether the rewrite materially shifted the doc-length distribution; if so,
  treat cross-doc contamination as a variable to control.
- **S0** (original, un-rewritten data) is unaffected by the rewrite-length concern and can be run as-is.

## Caveat 3 — Nanoset compiles a C++ index helper at runtime → needs `pybind11`
On first use of the Nanoset/TokenizedBytes dataloader, nanotron runs `make` to build a C++ helper
(`src/nanotron/data/nemo_dataset/Makefile`, via `python3 -m pybind11 --includes`). Without `pybind11`
installed it fails at training start with `fatal error: pybind11/pybind11.h: No such file or directory`
(`Making C++ dataset helpers module failed, exiting.`). Fix: `pip install pybind11` in the env (already done
in `nanotron-train`; see `PATCH_NOTES.md`). Also requires `python3-config` (present in the conda env).

## Caveat 4 — datatrove 0.5.0 renamed `DocumentTokenizer(shuffle=…)` → `shuffle_documents=…`
`tools/preprocess_data_parquet.py` uses `DocumentTokenizer(shuffle_documents=False, …)`. On older datatrove
(e.g. 0.3.0) the kwarg was `shuffle`; passing `shuffle=` to 0.5.0 raises
`TypeError: DocumentTokenizer.__init__() got an unexpected keyword argument 'shuffle'`. We pin datatrove
**0.5.0** (see `PATCH_NOTES.md` / §3.2), so use `shuffle_documents`. (More datatrove-version-specific
adaptations — the `DatatroveFolderDataset` signature, `numpy 2`, `huggingface_hub<1.0` — are in `PATCH_NOTES.md`.)

## Caveat 5 — `--n-tasks 32` deadlocked one datatrove worker; use 16
On the full 5B blocks, `--n-tasks 32` (`LocalPipelineExecutor` with 96 CPU / 256 G) hung: 31/32 tasks
finished in ~2 min but **one worker (rank 12) stalled mid-stream** (wrote its `.ds` but never its
`.ds.metadata`), and the executor blocked on it **indefinitely** (3 h, no error — a stuck worker, not a
crash). No `DONE`, no block 2. Re-running with **`--n-tasks 16`** completed cleanly in **3m42s** with
**exact** token totals. Rules: (a) use `--n-tasks 16` for these blocks; (b) always run with **hang
detection** — a healthy run finishes in minutes, so if the log mtime stalls for >5 min, `scancel` and
re-run rather than waiting (the per-task completion markers mean a same-`--logging-dir` resume would re-run
only the stuck task, but a clean re-run at 16 tasks is simplest). Symptom to check: `#.ds` files >
`#.ds.metadata` files ⇒ a worker stalled.

---

# Plan Example — Tokenize the two S0 blocks (shared-top-5B + 2nd-top-5B)

A concrete, reproducible application of this guide: tokenize both S0 blocks for training. Validated
pre-flight (tokenizer `</s>`=2 / `<s>`=1, one trailing EOS, no BOS — both `AutoTokenizer` and
`Tokenizer.from_file` agree) and follows §3.2 (env), §3.3/§4b (tokenizer file→dir + metadata rewrite),
§4 (command), §5.2 (verify), and `# Caveats` 1 (fresh logging-dir) / 3 (pybind11) / 4 (shuffle_documents).

### Exact paths (the output dirs become `dataset_folder` in the training YAML)
| block | source (READ-ONLY) | tokenized output `dataset_folder` |
|---|---|---|
| shared-top-5B | `/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/5B/shared-top-5B` | `/scratch/bvandur1/zhuicon1/nanotron_tokenized/shared-top-5B/tokenized` |
| 2nd-top-5B | `/scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/5B/2nd-top-5B` | `/scratch/bvandur1/zhuicon1/nanotron_tokenized/2nd-top-5B/tokenized` |

Outputs live under a new `…/nanotron_tokenized/` tree, separate from the source `…/data_rewrite/…`
folders (the job only reads source, writes output). Env: `nanotron-train`. Tokenizer (datatrove): the
`tokenizer.json` **file** + `--eos-token "</s>"`. `fix_ds_metadata` rewrites metadata to the tokenizer
**directory** — the **same** `--tokenizer-dir` for both blocks (nanotron asserts one shared tokenizer string).

### sbatch script (`/scratch/bvandur1/zhuicon1/basic/tokenize_s0.sh`)
```bash
#!/usr/bin/env bash
#SBATCH --job-name=tok_s0
#SBATCH --partition=cpu
#SBATCH --account=bvandur1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/bvandur1/zhuicon1/basic/tokenize_s0.%j.log
set -euo pipefail
source /scratch/bvandur1/zhuicon1/basic/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/bvandur1/zhuicon1/basic/miniconda3/envs/nanotron-train

REPO=/scratch/bvandur1/zhuicon1/projects/nanotron
TOKENIZER=/scratch/bvandur1/zhuicon1/tokenizers/llama2-unsloth-tokenizer/tokenizer.json
TOK_DIR=/scratch/bvandur1/zhuicon1/tokenizers/llama2-unsloth-tokenizer
EOS_TOKEN="</s>"
OUT_ROOT=/scratch/bvandur1/zhuicon1/nanotron_tokenized
N_TASKS=16   # 16 works reliably; 32 deadlocked one worker (see Caveat 5). Both 5B blocks tokenize in ~4 min.
cd "$REPO"

tokenize_block () {
  local INPUT_DIR="$1"
  local BLOCK; BLOCK="$(basename "$INPUT_DIR")"
  local OUT_DIR="$OUT_ROOT/$BLOCK/tokenized"
  local LOG_DIR="$OUT_ROOT/$BLOCK/logs/tokenize_$BLOCK"
  echo "=================== BLOCK: $BLOCK ==================="
  rm -rf "$OUT_DIR" "$LOG_DIR"            # Caveat 1: fresh logging-dir (no stale completion markers)
  mkdir -p "$OUT_DIR" "$LOG_DIR"
  python3 tools/preprocess_data_parquet.py \
      --tokenizer-name-or-path "$TOKENIZER" --eos-token "$EOS_TOKEN" \
      --output-folder "$OUT_DIR" --logging-dir "$LOG_DIR" --n-tasks "$N_TASKS" \
      parquet --dataset "$INPUT_DIR" --column text --glob-pattern "*.parquet"
  python3 tools/fix_ds_metadata.py --output-folder "$OUT_DIR" --tokenizer-dir "$TOK_DIR"
  python3 - "$OUT_DIR" <<'PY'
import sys, glob, os
metas = sorted(glob.glob(os.path.join(sys.argv[1], "*.ds.metadata")))
total = 0
for m in metas:
    with open(m) as f:
        f.readline(); total += int(f.readline().strip())   # line2 = shard token count
print(f"[VERIFY] {os.path.basename(os.path.dirname(sys.argv[1]))}: shards={len(metas)} total_tokens={total:,}")
assert total > 0, "ZERO tokens — datatrove skipped (Caveat 1) or read failed"
PY
}

tokenize_block /scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/5B/shared-top-5B
tokenize_block /scratch/bvandur1/zhuicon1/data_rewrite/experiments/train/5B/2nd-top-5B
echo "=== cross-block metadata line-1 (must be identical) ==="
head -1 "$OUT_ROOT"/shared-top-5B/tokenized/*.ds.metadata | sort -u
head -1 "$OUT_ROOT"/2nd-top-5B/tokenized/*.ds.metadata | sort -u
echo "DONE"
```

### Submit (from inside the current CPU allocation; strip inherited job vars)
```bash
cd /scratch/bvandur1/zhuicon1/basic
env -u SLURM_JOB_ID -u SLURM_JOBID sbatch tokenize_s0.sh
```

### Expected (verify; don't trust exit 0)
On-disk `.ds` token totals ≈ each block's `train_tokens_sum` (which already includes the +1 EOS/doc):
`shared-top-5B ≈ 5,000,002,332`, `2nd-top-5B ≈ 5,000,000,805`. Both blocks' `.ds.metadata` line-1 must be
identical: `/scratch/bvandur1/zhuicon1/tokenizers/llama2-unsloth-tokenizer|2`. (Note: this is the budget
total, NOT `train_tokens_sum + N_docs` — the EOS is already counted.)
</content>

# Plan Example — Re-tokenize arms, add a new arm, relaunch exp-0616 (2026-06-20)

Trigger: raw rewrite data was reorganized and a new arm `rewrite` was added; relaunch all arms except quality-10B-base.
Experimental contract: every arm's config is **identical except (a) 3-epoch step counts and (b) data-source fields**
(`general.run`, `data_stages[].data.dataset.dataset_folder`, `checkpoints.checkpoints_path`, wandb id). Any other drift
is a confound — verify parity before launching.

1. **Pre-flight** — validate each arm's `shuffled/*.parquet` (row count vs `_pretrain_manifest.json:total_docs_in_shuffled`,
   no zero-byte/unreadable files, `text` column present). A data-prep OOM can leave a truncated arm.
2. **Re-tokenize** the arms that have a `shuffled/` corpus only (base arms are flat + already tokenized):
   `for a in <arms>; do sbatch --job-name=tok_$a --output=…/tok_$a.%j.log /scratch/.../basic/tokenize_arm.sh $a; done`
   (cpu partition, `--n-tasks 16` — 32 deadlocks; it `rm -rf`'s OUT+LOG so no stale-skip; runs fix_ds_metadata + VERIFY).
3. **Recompute steps from ACTUAL `.ds` counts** (sum line-2 of `*.ds.metadata`): `train_steps = round(3*tok/2_097_152)`;
   `lr_decay_steps = round(0.1*train_steps)`; `lr_decay_starting_step = train_steps - lr_decay_steps`. Verify `ts*2_097_152/tok ≈ 3.000`.
4. **Configs** — generate the new arm from a canonical template (copy quality-first, change only the data-source fields +
   step counts); update existing arm step counts only if the `.ds` recompute differs. `diff`-verify parity (normalize the
   allowed fields, everything else must be identical; resume vs base differs only in `resume_checkpoint_path`+`load_*`).
5. **Launch** each arm base + a chained `--dependency=afterany` resume backup, **always `--exclude=h06`** (bad GPU node).
   6×4 GPU = 24 ≤ 32 QOS. A 15B arm (~82h) exceeds the 72h walltime → it relies on its resume leg to finish.
   Verify: `squeue` shows base(None)+resume(Dependency); `scontrol show job | grep ExcNodeList` == h06; first arm clears step 0.

# Plan Example — Finish the signal-disagreement λ sweep: λ1.5 + λ2 × seeds 42/43/44 (2026-07-23)

**Situation.** The λ sweep (`λ ∈ {0, 0.5, 1, 1.5, 2}` × seeds 42/43/44, 1.5B / 10B-token arm, 14 305 steps
≈ 30B tok / 3 epochs) was half finished: λ0 and λ0.5 complete, λ1 complete except seed43 (finishing),
**λ1.5 and λ2 had no usable checkpoints at all** except a stranded λ1.5-seed42 at step 4293.

**Two root causes, both worth remembering.**

1. **Dead node h06 ate the whole sweep.** Jobs 1804229–1804240 each died in ~20 s with
   `RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at CUDAContext.cpp:49`,
   every one of them on **h06** (3 GPUs in Xid error). `logs/slurm_scripts/launch_nvl_generic.sbatch`
   carries no `--exclude`, so SLURM kept re-placing the `afterany` retries on the same dead node.
2. **A 2-deep chain is too short.** At the measured **19.0 s/iter** on nvl dp4 (110K tok/s, ~288 TFLOPs/GPU),
   14 305 steps = **~75.5 h** against a `--time=1-00:00:00` wall ⇒ **4 productive links**. The old
   base+one-resume pattern silently stopped mid-run — that is exactly how λ1.5-seed42 ended at step 4535.

**⚠️ `c001` is decommissioned.** The a100 partition is now `c[002-003,007]`. Leaving `c001` in an
`--exclude` list does not get ignored — sbatch rejects the whole submission with
`Batch job submission failed: Invalid node name specified`. Verify names with `sinfo -o "%.12P %N" -h`
before reusing any exclude list. Current good list for 1.5B dp4: `--exclude=h06,n02,n06,n08`.

## Runbook

1. **Trim any pending eval array that references not-yet-existing checkpoints.**
   `scancel 1804242_[18-35]` — tasks 0–17 (λ0/λ1) still produce real results; 18–35 would have failed
   on missing λ1.5/λ2 checkpoints.

2. **Use the chain launcher, never one-shot sbatch, for runs longer than the walltime.**
   `/scratch/bvandur1/zhuicon1/projects/nanotron/logs/slurm_scripts/launch_nvl_chain.sbatch` —
   same body as `launch_nvl_generic.sbatch` plus `#SBATCH --exclude=h06,n02,n06,n08` and an
   **already-done guard**: a 5th arg `checkpoints_path` (+ optional 6th, target step, default 14305);
   the script exits 0 the moment `latest.txt >= target`, so surplus chain links cost nothing.
   Args: `<config-relpath> <wandb-run-id> <wandb-mode> <wandb-project> <checkpoints_path> [target-step]`.

3. **Submit N-deep `afterany` chains** with
   `/scratch/bvandur1/zhuicon1/projects/nanotron/logs/slurm_scripts/chain_lambda_sweep.sh`
   (`submit_chain <jobname> <link1-cfg> <resume-cfg> <run-name> <wandb-project> <depth>`).
   `DRY=1 bash chain_lambda_sweep.sh` prints the sbatch commands without submitting — always dry-run first
   and confirm every referenced config exists. Submission uses `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch`.
   Depth = ceil(75.5h / 24h) + 1 spare = **5** for a fresh 14 305-step run; **4** for λ1.5-seed42's resume.

   | job | link 1 config (in `examples/`) | links 2..N | wandb project | depth |
   |---|---|---|---|---|
   | `sigL15_s42` | `config_signal-disagreement-lambda15_10B_1.5B_resume.yaml` | same | `exp-0715-lambda-sweep` | 4 |
   | `sigL15_s43` | `config_signal-disagreement-lambda15_10B_1.5B_seed43.yaml` | `…_resume_seed43.yaml` | `…-seed43` | 5 |
   | `sigL15_s44` | `config_signal-disagreement-lambda15_10B_1.5B_seed44.yaml` | `…_resume_seed44.yaml` | `…-seed44` | 5 |
   | `sigL2_s42` | `config_signal-disagreement-lambda2_10B_1.5B.yaml` | `…_resume.yaml` | `exp-0715-lambda-sweep` | 5 |
   | `sigL2_s43` | `config_signal-disagreement-lambda2_10B_1.5B_seed43.yaml` | `…_resume_seed43.yaml` | `…-seed43` | 5 |
   | `sigL2_s44` | `config_signal-disagreement-lambda2_10B_1.5B_seed44.yaml` | `…_resume_seed44.yaml` | `…-seed44` | 5 |

   Seed-42 configs are the **un-suffixed** filenames and use the un-suffixed wandb project.
   **Link 1 must be the `_resume` config whenever a checkpoint already exists** (λ1.5-seed42): the fresh
   config would reload the shared init at
   `/scratch/bvandur1/zhuicon1/checkpoints/_init_1.5B_seed42/0` and silently discard the partial run.
   Checkpoint dirs: `/scratch/bvandur1/zhuicon1/checkpoints/signal-disagreement-<lambdaX>-10B-1.5B-seed4Y`.
   Data: `/scratch/bvandur1/zhuicon1/nanotron_tokenized/signal-disagreement-<lambdaX>/tokenized` (19 G each).

   Budget: 6 × 4 = 24 GPUs ≤ 32 QOS cap, ~3.2 days wall clock, ~1 700 GPU-h.

4. **Prune checkpoints as they land.** `checkpoint_interval=477` ⇒ 30 ckpts × ~20 GB × 6 runs ≈ 3.6 TB
   against a scratch filesystem already at 85 %. Keep steps `1431·k` plus epoch boundaries
   4770 / 9540 / 14305; never delete the step named by `latest.txt` while a chain link is still pending.

5. **Re-run the λ1.5/λ2 evals** once all six reach 14305 — rows 19–36 of `ckpt_list_lambda_sweep.txt`:
   ```bash
   cd /scratch/bvandur1/zhuicon1/projects/rewrite/08_evaluation/02_code
   env -u SLURM_JOB_ID -u SLURM_JOBID sbatch \
     --export=ALL,CKPT_LIST=$PWD/ckpt_list_lambda_sweep.txt \
     --array=18-35%16 --exclude=h06,n02,n06,n08,n03 run_eval.slurm
   ```
   Then `build_summary.py` and extend `08_evaluation/03_results/RESULTS_*.md` with the full λ curve.

## Verification

- **Guard:** `bash launch_nvl_chain.sbatch <any-cfg> x disabled x /scratch/.../signal-disagreement-lambda0-10B-1.5B-seed42`
  must print `already at step 14305 >= 14305 — nothing to do` and exit 0 without allocating.
- **Exclusion:** `LD_LIBRARY_PATH=/tmp/rl7shim scontrol show job <id> | grep ExcNodeList` → `h06,n[02,06,08]`.
  (`scontrol`/`sacctmgr` need the readline-8→7 shim; see the cluster QOS notes.)
- **Resume, not restart:** λ1.5-seed42's link-1 log must report `~10012 remaining training steps` and a first
  `iteration:` near **4294**. If it says 14305 remaining / iteration 1, kill it — wrong config was used.
- **Steady state:** ~19 s/iter, ~110K tokens/sec, ~288 model_tflops_per_gpu, lm_loss ~2.4 early.
  Logs: `/scratch/bvandur1/zhuicon1/projects/nanotron/logs/slurm_logs/<jobname>.<jobid>.log`.
- **Completion:** all 6 dirs report `latest.txt` = 14305 and contain 4770 / 9540 / 14305 — re-run the
  existence check over `ckpt_list_lambda_sweep.txt` before submitting the eval array.
