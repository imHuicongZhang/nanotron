# KYS grid — handover package

Everything needed to run the 72-run per-epoch-WSD grid on a fresh cluster. Sizes measured
2026-08-09.

**The data, tokenizer and init checkpoints are all on HuggingFace — see §9 for how to get
them.** §2–§7 describe what those artifacts are and how much disk they need; §9 is the
procedure.

## 1. Code — ~32 MB

| item | size | note |
|---|---:|---|
| `nanotron-kys` worktree | 14.4 MB | branch `huicong-dev` |
| `.git` | 17.3 MB | carries tags `upstream-pin-2411b022`, `kys-patched-base` |

Ship as a git bundle so history and tags survive:

```bash
git bundle create nanotron-kys.bundle --all
# receiving side:
git clone nanotron-kys.bundle nanotron-kys && cd nanotron-kys && git checkout huicong-dev
```

Pinned upstream commit `2411b022a75fb7f7561a1bb4166706da5e1b76de` (2026-04-07), which is
**still upstream `main` HEAD** as of 2026-08-09 — a fresh clone of `huggingface/nanotron`
gives byte-identical source. 9 patches on top; see `INSTALL.md` §4 for which and why.

Includes:
- `INSTALL.md` — install instructions (H200 primary, Blackwell appendix)
- `tools/probe_blackwell.py` — 30-second on-hardware go/no-go check
- `tools/generate_configs.py` — emits the 108 experiment templates
- `tools/render_config.py` + `deploy/clusters.yaml` — per-cluster deployment stamping
- `tools/hash_init_checkpoint.py` — init-checkpoint verification
- `tools/assert_invariants.py` — preflight + step-200 batch-invariant assertion
- `SOP.md` — launch procedure, the mbs rationale, and the full wandb setup
- `PATCH_NOTES.md`, `UPSTREAM_PIN.md`, `COMPATIBILITY.md`

## 2. Configs — <2 MB

108 templates = 18 trunks × 3 segments + 54 branches, covering 72 logical runs
(6 settings × 3 seeds × (1 trunk + 3 cooldown branches)).

Templates deliberately do **not** parse on their own: `parallelism`, `micro_batch_size`,
`batch_accumulation_per_replica`, `zero_stage` and `sequence_length` are stamped at launch by
`render_config.py`, which derives `accum = 1024 / (mbs × dp)` and refuses non-integers, so the
2,097,152-token global batch is invariant across every run and every cluster.

## 3. Tokenizer — 3.6 MB

llama-2 32000 vocab, token_size 2 bytes. Ships inside the corpora repo as `tokenizer/`, so
`tokenizer_path` is normally `<data_root>/tokenizer` — see §9. (It was tokenized under the
JHU path `…/tokenizers/llama2-unsloth-tokenizer`, which is still recorded in every
`.ds.metadata`; §9.2 repoints those at your copy.)

## 4. Tokenized corpora — ~120.4 GB

| setting | folder | size |
|---|---|---:|
| quality_base | `quality_base/tokenized` | ~20.06 GB |
| quality_first | `quality_first/tokenized` | 20.12 GB |
| diversity_oriented | `diversity_oriented/tokenized` | 19.88 GB |
| wrap_inspired | `wrap_inspired/tokenized` | 20.14 GB |
| rewire_inspired | `rewire_inspired/tokenized` | 20.13 GB |
| disagreement_aware | `disagreement_aware/tokenized` | 20.10 GB |

16 `.ds` shards each plus `.ds.index` / `.ds.metadata`. All three are in the published repo,
so a `snapshot_download` (§9) gets them; if you ever copy a corpus by hand, **the
`.ds.metadata` files must come with it** — nanotron's config validator reads `vocab_size` from
them and refuses to start without it (`Model's vocab_size (32000) does not match dataset's
(None)`). One shard of `quality_base` is legitimately zero bytes; see §9.4 before assuming a
bad transfer.

`quality_base` is the re-shuffled corpus (seed 42, matching the other five). It replaced an
earlier unshuffled build, which was never published — the only `quality_base` you can download
is the correct one. `assert_invariants.py` rejects the old directory name outright.

Directory names above are the unified names, as published in
`wytro/Know-Your-Sources-tokenized`. The original JHU `/scratch` tree used the pre-upload names
(`10B-base-shuf42`, `quality-first`, `diversity-first`, `wrap`, `rewrite`,
`signal-disagreement-lambda05`), which were remapped at upload time and will not resolve
against the current `SETTING_CORPUS`.

## 5. Init checkpoints — 27.1 GB

`_init_1.5B_seed{42,43,44}/0/` at 9.03 GB each. Each trunk directory must be pre-seeded with
its seed's init as step 0 plus a `latest.txt` containing `0`, so the trunk's latest.txt
auto-resume works for both first launch and crash-restart.

Verify after downloading **and** after nanotron loads them. The `.hash.json` manifests ship at
the root of the init repo, so `--check` needs no separate artifact:

```bash
python tools/hash_init_checkpoint.py <init_root>/_init_1.5B_seedNN/0 \
    --check <init_root>/init_1.5B_seedNN.hash.json
python tools/hash_init_checkpoint.py <init_root>/_init_1.5B_seedNN/0 \
    --check <init_root>/init_1.5B_seedNN.hash.json --mode loaded
```

| checkpoint | parameters | rolling sha256 |
|---|---:|---|
| `_init_1.5B_seed42` | 1,504,299,008 | `2ede6612b2e48d7529f867f0e74ca0a7d9ba79cd635d4f803f8022d9b0113aba` |
| `_init_1.5B_seed43` | 1,504,299,008 | `78ab44e2b2ac954ee441ea340e35969c82cf78bf3f2266f3c8cd0590ce3b8aa3` |
| `_init_1.5B_seed44` | 1,504,299,008 | `a967df1cae0538c63bf1be412d6e3db0082e75936680f98b11b9a84d49642247` |

The hash is layout-independent: it keys on the logical parameter path with the
`_pp-rank-N-of-M_tp-rank-N-of-M` suffix stripped, and hashes raw storage bytes via a uint8
reinterpret (numpy has no bfloat16). It is therefore unaffected by `dp`, which is correct —
with `tp=1, pp=1, zero_stage=0` the `model/`, `optimizer/` and `lr_scheduler/` payloads carry
no `dp` in their filenames and `load_random_states()` is never called, so a checkpoint written
at dp=4 resumes at any dp.

The init optimizer state is empty (`state_dict["state"] == {}` — Adam has never stepped), so
`load_optimizer: true` on the first trunk segment is equivalent to a fresh optimizer.

## 6. Total to download

| | |
|---|---:|
| code + configs (git) | ~36 MB |
| tokenized corpora + tokenizer (`…-tokenized`) | ~120.4 GB |
| init checkpoints (`…-init`) | 27.1 GB |
| **total** | **~148 GB** |

Excludes the 104 GB parquet repo, which training does not need (§9). Unchanged by the stack
move — the Blackwell upgrade adds no bytes, only different pip pins.

## 7. Disk to reserve for checkpoints

| | per (setting,seed) | grid (×18) |
|---|---:|---:|
| checkpoints, peak | 295 GB | 5.31 TB |
| after pruning trunk restart points | 126 GB | 2.27 TB |

14 checkpoints per chain at 21.06 GB each (optimizer 18.05 + model 3.01): 6 must be kept
(3 branch points 4292/8583/12875 + 3 annealed finals 4768/9537/14305), 8 are trunk restart
insurance and can be deleted once their segment completes. Stripping optimizer state from the
3 annealed finals — the only ones eval needs — takes each from 21.06 GB to 3.01 GB.

## 8. What is and is not comparable across layouts

Measured 2026-08-09 against the real corpora and the real sampler.

**`dp` is free.** `MegatronPretrainingSampler.__iter__` accumulates `mbs × dp` consecutive
sample indices and hands rank *d* the slice `[d*mbs : (d+1)*mbs]`, so the partition of the
1024-sequence global batch into micro-batches is **consecutive blocks of size `mbs`,
regardless of `dp`**. The accumulated gradient is
`(1/dp)·(1/accum)·Σ_j loss_j` = the mean over all `1024/mbs` micro-batches, which has no `dp`
in it. Verified numerically on 1024 real sequences: per-token weights for
`(mbs=4, dp=4, accum=64)` vs `(mbs=4, dp=8, accum=32)` differ by **0.000e+00**. Only
floating-point rounding (the all-reduce tree) differs.

**`micro_batch_size` is not free.** `Loss.forward` computes
`masked_mean(loss, label_mask) = (loss*mask).sum() / mask.sum()`, and `label_mask` drops the
token before every document boundary (`return_positions` defaults True; datatrove supplies
`positions`; the collator masks `position_ids == 0`). Measured on quality-first: **2.8 masked
tokens per 2048-token sequence, range 0–10**. So `masked_mean`'s denominator `n_j` varies
between micro-batches — relative std 6.2e-4 at mbs=4, 3.9e-4 at mbs=16 — and the effective
per-token weight is `1/(n_micro · n_j)`. Regrouping the same 1024 sequences into
different-sized micro-batches therefore changes the objective:

| comparison | max relative per-token weight difference |
|---|---:|
| mbs=4 dp=4 accum=64 **vs** mbs=4 dp=8 accum=32 | **0.000e+00** (identical) |
| mbs=4 **vs** mbs=8 | 1.04e-3 |
| mbs=4 **vs** mbs=16 | 1.43e-3 |
| mbs=8 **vs** mbs=16 | 7.04e-4 |

That is ~4 orders of magnitude above fp32 rounding — a genuine difference in the optimization
objective, not a rounding artefact.

**Consequence.** `micro_batch_size` must be identical across all 72 runs;
`dp` may differ per cluster. `render_config.py` enforces both: it refuses to render when the
clusters named in `seed_assignment` disagree about `micro_batch_size`, and refuses an `mbs`
that will not fit a cluster's HBM. H100 (80 GB) caps `mbs` at 4; H200 (141 GB) allows 16 —
so a grid spanning both must run mbs=4 on both sides.

---

## 9. Obtaining the artifacts — three HuggingFace repos

Everything above is published. Nothing has to be shipped by hand. Listings below verified
against the live repos 2026-08-12.

| repo | type | size | what it is |
|---|---|---:|---|
| `wytro/Know-Your-Sources` | dataset | 104 GB | raw parquet, six configs. The paper artifact — **not needed to train** |
| `wytro/Know-Your-Sources-tokenized` | dataset | 120 GB | the `.ds` corpora (§4) **and** the tokenizer (§3). This is `data_root` |
| `wytro/Know-Your-Sources-init` | model | 27 GB | the three init checkpoints (§5) plus their hash manifests. This is `init_root` |

### 9.1 The two downloads training needs

```python
from huggingface_hub import snapshot_download

# data_root AND tokenizer_path come from this one repo — 120 GB
data_root = snapshot_download(
    repo_id="wytro/Know-Your-Sources-tokenized",
    repo_type="dataset",
    local_dir="/shared/scratch/kys/nanotron_tokenized",
)

# init_root — 27 GB. A model repo, so no repo_type argument.
init_root = snapshot_download(
    repo_id="wytro/Know-Your-Sources-init",
    local_dir="/shared/scratch/kys/init",
)
```

**`local_dir` *is* the root.** Nothing is moved or renamed afterwards. The remote layout is
already what the renderer expects:

```
<data_root>/<setting>/tokenized/*.ds        <- exactly SETTING_CORPUS[setting] + CORPUS_LEAF
<data_root>/tokenizer/                      <- tokenizer_path (the DIRECTORY, per §3)
<init_root>/_init_1.5B_seed<S>/0/           <- exactly the `cp -al` source in SOP.md §3
<init_root>/init_1.5B_seed<S>.hash.json     <- the --check manifest §5 asks for
```

The setting directories are the six unified names, identical to `SETTING_CORPUS` keys — the
legacy on-disk names were remapped at upload time, so there is nothing to reconcile. (If you
are holding an old JHU `/scratch` tree instead of a download, see the note at the end of §4.)

`snapshot_download` also writes a **`.cache/huggingface/`** directory inside `local_dir`. That
is expected, not a failed download: it holds the per-file etags that make a re-run resume
instead of re-fetching. **Leave it.** Nothing reads `data_root` by listing it —
`render_config.py` composes explicit `<data_root>/<setting>/tokenized` paths — so it is inert.
Deleting it costs you a full 120 GB re-download next time.

### 9.2 One required fixup: repoint the metadata at your tokenizer

**Do this before rendering anything.** Line 1 of every `*.ds.metadata` records the tokenizer
path the corpus was tokenized under, which is the original JHU path and does not exist on your
machine. nanotron does not treat that as cosmetic:

- `config.py:192` calls `AutoTokenizer.from_pretrained(<that string>)` to derive `vocab_size`;
- `config.py:521` asserts your `tokenizer_path` **equals** that string exactly.

So a freshly downloaded tree fails preflight no matter what you put in `tokenizer_path`.
`tools/fix_ds_metadata.py` rewrites line 1 in place, preserving the `|<token_size>` suffix. It
is stdlib-only and idempotent, so re-running it is harmless:

```bash
for s in quality_base quality_first diversity_oriented \
         wrap_inspired rewire_inspired disagreement_aware; do
    python tools/fix_ds_metadata.py \
        --output-folder <data_root>/$s/tokenized \
        --tokenizer-dir  <data_root>/tokenizer
done
```

Pass the **same** `--tokenizer-dir` to all six — nanotron also asserts every metadata file
across every dataset folder carries an identical tokenizer string (`config.py:194-199`).

Verified end to end: before the rewrite, loading a rendered config raises
`AssertionError: Tokenizer passed in config (…) does not match dataset's (…) tokenizer (…)`;
after it, the same config loads and reports `vocab_size 32000`.

**If you skip this step you will be told, not left to find out.** `assert_invariants.py`
checks it as part of its normal run (§9.6) — no extra flag — and on mismatch prints the exact
`fix_ds_metadata.py` command for your paths, ready to paste. It also catches the rarer case of
individual shards disagreeing with each other, which nanotron asserts separately at
`config.py:194-199`. Without that check the failure would surface inside nanotron's config
parsing, after SLURM had already allocated the job.

### 9.3 What goes in `deploy/clusters.yaml`

Three of the four paths come straight out of the two calls above:

```yaml
data_root:      /shared/scratch/kys/nanotron_tokenized            # local_dir of the tokenized repo
tokenizer_path: /shared/scratch/kys/nanotron_tokenized/tokenizer  # its tokenizer/ subdirectory
ckpt_root:      /shared/scratch/kys/checkpoints                   # you choose; reserve ~5.3 TB (§7)
wandb:
  dir:          /shared/scratch/kys/wandb                         # shared storage, see SOP.md §4.2
```

`init_root` is not a `clusters.yaml` field — it is only used once, by the `cp -al` trunk-seeding
command in `SOP.md` §3.

### 9.4 Two things that look broken and are not

**A zero-byte shard in `quality_base`.** `quality_base/tokenized/00015_unshuffled.ds` is
**exactly 0 bytes**, and its `.ds.metadata` records `0` tokens. This is correct, not a failed
upload: datatrove ran with 16 ranks and rank 15 received no input documents. It survives the
round trip intact and must be kept —

- `assert_invariants.py` expects **16** shards and would fail on 15;
- the loader indexes with `bisect` over cumulative shard lengths
  (`data/tokenized_bytes.py:289`), and a zero-length file adds a duplicate boundary that
  `bisect_right` steps past, so no sample can ever land in it.

The token counts still add up exactly: the sixteen `.ds.metadata` files sum to
**10,000,003,137**, matching `EXPECTED_CORPUS['quality_base']`.

**Every `.ds` is named `000NN_unshuffled.ds`, including in the shuffled corpora.** All six
arms use that suffix. It is a datatrove output-filename default and says nothing about the
contents — the shuffle happened upstream, at the parquet stage. **The directory name is the
authority, not the filename.** Do not go looking for a `_shuffled` variant; there isn't one.

### 9.5 Fetching one corpus instead of all six

`allow_patterns` takes glob patterns matched against repo-relative paths, so a single arm plus
the tokenizer is 20.1 GB rather than 120 GB:

```python
snapshot_download(
    repo_id="wytro/Know-Your-Sources-tokenized",
    repo_type="dataset",
    local_dir="/shared/scratch/kys/nanotron_tokenized",
    allow_patterns=["quality_base/*", "tokenizer/*"],
)
```

That pattern selects 51 of the repo's 293 files (48 corpus + 3 tokenizer). Re-running with a
different arm added to the list tops the same `local_dir` up in place. Note that a partial
`data_root` renders and trains fine for the arms present, but only those — the other five
settings will fail preflight on a missing directory.

### 9.6 Verify after downloading

```bash
# corpus: shard count and exact token total, per rendered config
python tools/assert_invariants.py --config rendered/<name>.yaml

# init checkpoints: on-disk bytes, then again as nanotron actually loads them
python tools/hash_init_checkpoint.py <init_root>/_init_1.5B_seed42/0 \
    --check <init_root>/init_1.5B_seed42.hash.json
python tools/hash_init_checkpoint.py <init_root>/_init_1.5B_seed42/0 \
    --check <init_root>/init_1.5B_seed42.hash.json --mode loaded
```

The corpus check is the one that matters: a `data_root` that exists but holds the wrong data
is the failure this whole package is built to prevent, and it is caught here rather than 27
hours into a run. An unknown corpus directory name is rejected outright.
