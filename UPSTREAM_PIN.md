# Upstream pin

This fork is pinned to an exact upstream commit. Do **not** merge a moving `main`;
pull upstream deliberately and bump this file in the same commit.

| | |
|---|---|
| Upstream repo | `https://github.com/huggingface/nanotron` |
| **Pinned commit** | **`2411b022a75fb7f7561a1bb4166706da5e1b76de`** |
| Short SHA | `2411b022` |
| Commit subject | `🔒 Pin GitHub Actions to commit SHAs` |
| Commit date | 2026-04-07 15:25:09 +0200 |
| nanotron version at pin | `0.4` (`setup.py` / `pyproject.toml`) |
| Git tag for this point | `upstream-pin-2411b022` |
| Work branch | `huicong-dev` |
| Recorded | 2026-08-09 |

## Citation string

> Experiments use [nanotron](https://github.com/huggingface/nanotron) at commit
> `2411b022a75fb7f7561a1bb4166706da5e1b76de` (2026-04-07), with the local patch
> series in `patches/` (see `PATCH_NOTES.md`).

## Why this commit

It is the exact commit the JHU install at `/scratch/bvandur1/zhuicon1/projects/nanotron`
descends from, so this clone reproduces the ARR-submission runs bit-for-bit at the
source level.

At the time of pinning it was **also** the tip of upstream `main` — upstream had not
advanced since 2026-04-07, so "the commit my runs used" and "latest upstream" were the
same commit. There is therefore **no** upstream drift to reconcile and the patch series
applied with zero conflicts.

## Environment pins (not captured by git)

The source pin alone is not sufficient for reproducibility; the patch series targets
these library versions specifically. See `PATCH_NOTES.md`.

```
datatrove[io]==0.5.0     # NOT main; 0.9.0 needs huggingface-hub>=1.5.0
numpy==2.0.2             # datatrove >=0.4 needs numpy>=2; <2.1 for numba 0.60
huggingface_hub<1.0      # required by transformers 4.46.3 / tokenizers 0.20.3
torch==2.4.1+cu124
transformers==4.46.3
tokenizers==0.20.3
pybind11                 # Nanoset compiles a C++ index helper at runtime
```

## Refreshing the pin

```bash
git fetch upstream
git log --oneline 2411b022..upstream/main     # review what actually changed
# then, deliberately:
git rebase --onto <new-sha> 2411b022 huicong-dev
# re-run the compatibility check in COMPATIBILITY.md before trusting any run
```
