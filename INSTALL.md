# Installing the KYS training stack

Target: **H200 (Hopper, sm_90)**. Blackwell/B300 notes are in Appendix A in case we move
later. Verified 2026-08-09 by dumping the actual wheels with `cuobjdump`, not from release
notes.

> **`pip install` alone is not enough.** nanotron needs 9 source patches to run this grid,
> and none of them are upstream. Install from **this fork**, not `huggingface/nanotron`.
> See §4.

---

## 1. Good news: H200 needs nothing special

H200 SXM is the same GH100 die as H100 — compute capability **9.0**, `sm_90`. Both prebuilt
wheels carry sm_90 SASS, so there is no probe step and no source build:

| binary | SASS architectures | sm_90? |
|---|---|:--:|
| torch 2.8.0+cu128 `libtorch_cuda.so` | sm_70, 75, 80, 86, 89, **90**, 90a, 100, 100a, 120, 120a | yes |
| flash-attn 2.8.3 prebuilt (cu12/torch2.8) | sm_80, **90**, 100, 120 | yes |

The difference from H100 is memory: 141 GB HBM3e vs 80 GB, which is what lets us run
`micro_batch_size: 16` (≈109.7 GiB peak) instead of 4.

---

## 2. Install

Python 3.11. Order matters — install torch first so flash-attn links against it.

```bash
conda create -n kys python=3.11 -y && conda activate kys

# 1. torch
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 2. flash-attn (prebuilt wheel; contains sm_90)
pip install flash-attn==2.8.3 --no-build-isolation

# 3. data stack — these pins are load-bearing, see below
pip install "datatrove[io]==0.5.0" "numpy==2.0.2" "huggingface_hub<1.0" \
            "transformers==4.46.3" "tokenizers==0.20.3" numba==0.60.0 pybind11

# 4. nanotron FROM THIS FORK (NOT huggingface/nanotron — see §4)
pip install -e /path/to/nanotron-kys
```

**Why those pins.** `transformers==4.46.3` and `tokenizers==0.20.3` both require
`huggingface_hub<1.0`; datatrove `main` (0.9.0) requires `huggingface-hub>=1.5.0` and would
break both. `datatrove==0.5.0` is the newest release with the public `return_positions` /
`positions_from_eos_token_id` API that still allows hub `<1.0` — the patch series targets
exactly that API. `numpy==2.0.2`: datatrove ≥0.4 needs numpy≥2, numba 0.60 needs <2.1.
`pybind11` is needed because nanotron compiles a C++ index helper at runtime.

The stale `torch>=1.13.1` / `numpy<2` / `flash-attn<2.7.0` pins in upstream's
`pyproject.toml` are corrected in this fork (patch #7), so step 4 does not fight steps 1–3.

---

## 3. Smoke check

```bash
python - <<'PY'
import torch, flash_attn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
print("torch", torch.__version__, "| flash_attn", flash_attn.__version__)
print("arch_list", torch.cuda.get_arch_list())
q=k=v=torch.randn(256,16,128,device="cuda",dtype=torch.bfloat16)
cu=torch.tensor([0,256],dtype=torch.int32,device="cuda")
o=flash_attn_varlen_func(q=q,k=k,v=v,cu_seqlens_q=cu,cu_seqlens_k=cu,
                         max_seqlen_q=256,max_seqlen_k=256,dropout_p=0.0,
                         softmax_scale=None,causal=True)
torch.cuda.synchronize(); print("flash-attn OK", tuple(o.shape))
PY
```

Expect `(9, 0)` and `sm_90` in `arch_list`. Then verify the init checkpoints:

```bash
python tools/hash_init_checkpoint.py <ckpt>/0 --check init_1.5B_seedNN.hash.json
```

Please report back: **GPUs per node, how many nodes we can hold concurrently, and the SLURM
wall-clock limit.** Those are the last blanks in `deploy/clusters.yaml`.

---

## 4. The 9 patches — all still required

Upstream `huggingface/nanotron` `main` is **still at `2411b022`, dated 2026-04-07** — verified
2026-08-09 via `git ls-remote` and the GitHub API (HEAD sha
`2411b022a75fb7f7561a1bb4166706da5e1b76de`, latest tag `v0.5`, not archived). A fresh clone
today is byte-identical to our pin. **Nothing has been fixed upstream.**

These are API- and dataloader-level fixes, **independent of GPU architecture** — they are just
as necessary on H200 as they would be on Blackwell.

| # | file | what | why |
|---|---|---|---|
| 0 | `scaling/parametrization.py` | `config.model.X` → `config.X` (4 attrs) | **upstream bug**: `llama.py:1102` constructs the parametrizator with `config=config.model` (a `ModelArgs`), so upstream's `config.model.init_method.std` raises `AttributeError` at model init. Predates this grid (fork commit `75cb1f1c`) |
| 1 | `data/nanoset.py` | `eos_token_id` → `positions_from_eos_token_id` | public datatrove 0.5.0 signature |
| 2 | `data/tokenized_bytes.py` | same rename; drop HF-fork-only kwargs | ditto |
| 3 | `data/tokenized_bytes.py` | restore `self.folder_path` as `str` | consumption accounting compares it as a string |
| 4 | `nemo_dataset/blendable_dataset.py` | drop `assert "s3" in folder_path` | local-disk data |
| 5 | `data/tokenized_bytes.py` | `__getitem__` modulo wrap | **without it every run dies at the 1-epoch boundary**, step ~4768 |
| — | `tools/preprocess_data_parquet.py` | `shuffle` → `shuffle_documents` | datatrove 0.5.0 rename |
| 6 | `models/llama.py` | `unpad_input(...)[:4]` at 3 sites | 2.6.x returns a 4-tuple, ≥2.7.x a 5-tuple |
| 7 | `pyproject.toml` | `torch>=2.7.0`, `numpy>=2.0,<2.1`, `flash-attn>=2.8.0` | old pins block the stack above |
| 8 | `trainer.py` | wandb run name = `general.run` verbatim | upstream prepends a `dd/mm/YYYY_HH:MM:SS_` timestamp; breaks run selection across 108 runs |

Patches #0 and #5 are the ones that bite if skipped — #0 fails immediately at model init, #5 fails 4768 steps in. Patch #5 in particular: it fails 4768 steps into a run, not at
startup. Patch #6 only affects the generation / kv-cache path (`run_generate.py`); the
training path calls `flash_attn_varlen_func` with keyword arguments whose signature is
unchanged between 2.6 and 2.8.

Full rationale in `PATCH_NOTES.md`; the pin in `UPSTREAM_PIN.md`.

---

## 5. Launching

Templates in `configs/know-your-sources/` are **deliberately not runnable as-is** — they carry no
`parallelism`, `micro_batch_size`, `batch_accumulation_per_replica`, `zero_stage` or
`sequence_length`. Stamp them per cluster:

```bash
python tools/render_config.py \
    --template configs/know-your-sources/quality_first_seed43_trunk1.yaml \
    --cluster h200 --seed 43 --out rendered/quality_first_seed43_trunk1.yaml
torchrun --nproc_per_node=8 run_train.py --config-file rendered/quality_first_seed43_trunk1.yaml
```

The renderer derives `accum = 1024 / (mbs × dp)` so every run takes exactly
**2,097,152 tokens/step**, and refuses to emit anything that would not. It also refuses an
`mbs` that will not fit the cluster's HBM, and refuses to render a seed onto a cluster it is
not assigned to.

Order per (setting, seed): `trunk1 → trunk2 → trunk3`, then `ep1`, `ep2`, `ep3` (each branch
can start as soon as its trunk segment has written its final checkpoint). Before the first
trunk segment, seed the trunk directory:

```bash
T=<ckpt_root>/<setting>_seed<S>_trunk
mkdir -p $T && cp -al <init_root>/_init_1.5B_seed<S>/0 $T/0 && echo 0 > $T/latest.txt
```

The templates contain **no absolute paths at all**. `data_root`, `tokenizer_path`, `ckpt_root`
and `wandb.dir` in `deploy/clusters.yaml` are the only locations anyone sets; see `SOP.md` §2.

---

## Appendix A — if we ever move to B300 (Blackwell Ultra)

Not currently in use. The blockers are real and were measured:

- Both prebuilt wheels stop at **sm_100** and carry **no PTX**, so there is no JIT fallback.
- CUDA 12.9 exposes `compute_100/100a/100f` **and** `compute_103/103a/103f` as distinct
  targets (from `ptxas --help` in `nvidia-cuda-nvcc-cu12==12.9.86`), so 10.3 is genuinely a
  separate architecture, not an alias of 10.0.
- Whether an `sm_100` cubin loads on a cc-10.3 device depends on CUDA's minor-version binary
  compatibility rule. We could not test it without the hardware.

`tools/probe_blackwell.py` settles it in ~30 seconds on a real B300: it prints the device
capability and arch list, then runs a bf16 matmul and a forward+backward through
`flash_attn_varlen_func`. Exit 0 = prebuilt stack fine; exit 3 = rebuild from source.

Build-from-source path, if needed:

```bash
export TORCH_CUDA_ARCH_LIST="10.3+PTX"        # for torch
export FLASH_ATTN_CUDA_ARCHS="103"            # flash-attn IGNORES TORCH_CUDA_ARCH_LIST
pip install flash-attn==2.8.3 --no-build-isolation --no-binary flash-attn
```

If `103` is rejected by the 2.8.3 setup.py (it only knows `80;90;100;120`), build from
flash-attn `main`, which emits the **family** target `compute_100f` under CUDA ≥12.9 —
`sm_100f` is the target NVIDIA introduced so one binary covers the whole sm_10x family:

```bash
git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention
export FLASH_ATTN_CUDA_ARCHS="100"
MAX_JOBS=8 NVCC_THREADS=2 pip install . --no-build-isolation
```

Budget several hours; flash-attn is slow to compile.

Note that moving to Blackwell would also break the "one architecture for the whole grid" rule
unless the *entire* grid moves. H100 and H200 are both sm_90 and can be mixed freely (see
`HANDOVER.md` §5 for what is and is not comparable across layouts); Blackwell cannot be mixed
with either.
