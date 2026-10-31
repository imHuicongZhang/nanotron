# KYS 3B grid — portable release

Train the four-setting KYS 3B grid (seed 42; quality-first, diversity-oriented, disagreement-aware, quality-base;
trunk + three annealed branches each = 12 final models) on your own cluster from the public dataset
`blab-jhu/KYS-3B-pretraining-data`. Everything site-specific comes from ONE operator config that you fill in; nothing
here defaults to the publisher's paths, W&B account, scheduler, GPU model or GPU count.

| fixed (science_3b.py; not configurable) | yours (operator.template.yaml) |
|---|---|
| 57-layer llama, hidden 2048, 16 attention/KV heads, FFN 5632, tied embeddings, vocab 32000, bf16 | all paths: code, data, configs, init, checkpoints, logs, reports, W&B dir |
| the published llama-2 tokenizer (sha256-pinned) and data, in published order and shard boundaries | Python env (built by `install_env.sh`) |
| seed 42, one shared initialization for all four settings | W&B entity + project (online) |
| 1024 sequences x 2048 tokens = 2,097,152 tokens per update | data-parallel size, micro-batch size, ZeRO stage (0/1), activation recompute |
| AdamW (0.9, 0.95, 1e-8), wd 0.1, clip 1.0; warmup 1000 to 4e-4; constant; linear decay to 0 | nodes, GPUs per node, rendezvous |
| branch points 8583 / 17166 / 25749; finals 9537 / 19073 / 28610 | scheduler (Slurm account/partition/time/... or none) |
| tp = pp = 1 | |

The global batch is fixed: `micro_batch_size x accumulation x data_parallel_size = 1024`. You choose
`data_parallel_size` (= nnodes x GPUs per node) and `micro_batch_size`; accumulation is derived and any combination
that does not divide 1024 exactly is refused. Use ONE deployment for all four settings and every segment: the loss is
averaged per micro-batch and checkpoints carry their data-parallel layout, so `prepare` refuses a change once training
started (`CHECKPOINT_ROOT/kys3b_deployment.json`).

## 1. Install (CPU host with network; no GPU or CUDA toolkit needed)

    git clone --branch release/kys-3b-600m https://github.com/imHuicongZhang/nanotron.git
    cd nanotron
    bash deploy/portable_3b/install_env.sh <CPython 3.11 with headers> <new env dir>

`install_env.sh`, in this order (every version pinned; `requirements/` equals the frozen env of the reference run):
1. a venv from your CPython **3.11**; it must ship `Python.h` (conda / uv / pyenv builds do; a distro Python needs
   `python3.11-devel`). The binary wheels below are cp311.
2. `torch==2.8.0+cu128` from the official PyTorch index `https://download.pytorch.org/whl/cu128`
   (`requirements/1-torch-cu128.txt`; that build is not on PyPI, so a plain `pip install` of the old lock fails).
3. the rest from PyPI (`requirements/2-stack.txt`), including two prebuilt wheels pinned by URL + sha256:
   flash-attn 2.8.3 (CUDA 12 / torch 2.8 / cp311) and nv-grouped-gemm 1.1.4.post8 (nanotron imports it when the model
   code is imported, although the 3B model is dense).
4. this checkout, `pip install --no-deps -e .`
5. nanotron's C++ dataset helper, compiled with the env's interpreter (`make -C src/nanotron/data/nemo_dataset
   PYTHON=<env python>`; needs g++ and make). Without this, rank 0 compiles it at the first step.
6. `pip check` and `check_env.py`: exact versions, CPU imports, the helper. GPU-dependent imports (flash_attn kernels,
   nanotron.models.llama, CUDA) are only reported — they are checked for real on the GPUs (sections 5 and 6).

GPU hosts additionally need an NVIDIA driver supporting CUDA 12.8, plus a C compiler and the same interpreter's
headers at run time (Triton JIT-compiles small launcher stubs). `pip install -r deploy/portable_3b/requirements/test.txt`
adds pytest for the CPU tests.

## 2. Data

    hf download blab-jhu/KYS-3B-pretraining-data --repo-type dataset --revision <revision named in HANDOFF/receipt> \
        --local-dir $DATA_ROOT --include 'tokenized/*' 'tokenizer/*' 'RELEASE_MANIFEST.json' 'HANDOFF_RELEASE.json'

(~160 GB; add `'corpora/*'` for the parquet text, ~136 GB more.) Do not reshard or reorder anything: shard i is
shuffled part i, and the shard boundaries determine the token-window tails.

## 3. Operator config, render, check

    cp deploy/portable_3b/operator.template.yaml $HOME/kys3b.operator.yaml     # fill in every <<...>>
    P=<env>/bin/python
    $P deploy/portable_3b/kys3b.py validate          --operator $HOME/kys3b.operator.yaml
    $P deploy/portable_3b/kys3b.py render            --operator $HOME/kys3b.operator.yaml
    $P deploy/portable_3b/kys3b.py relocate-metadata --operator $HOME/kys3b.operator.yaml
    $P deploy/portable_3b/kys3b.py check             --operator $HOME/kys3b.operator.yaml --corpus [--hash]

* `validate` / `render` name every placeholder left, missing/unknown/wrong-typed key, publisher path/entity/account and
  every invalid batch combination. `render` writes `CONFIG_ROOT/{grid/*.yaml,*.env, init/, site.env, plan.json,
  operator.resolved.yaml, render_manifest.json}` and, with `scheduler.type: slurm`, `slurm/*.sbatch` + `submit_grid.sh`.
  A second render of a different operator config into the same CONFIG_ROOT needs `--force` (and is refused after
  training recorded another deployment).
* `relocate-metadata` rewrites line 1 of every `.ds.metadata` to your `DATA_ROOT/tokenizer` (nanotron loads the
  tokenizer named there and requires one string across folders); the `.ds` / `.index` binaries are not touched.
* `check` re-renders and compares every file, re-checks the architecture against the 1.5B config, the exact batch,
  LR schedule, branch sources and 16 separate output folders; `--corpus` checks the tokenizer hashes, shard counts,
  sizes vs `RELEASE_MANIFEST.json`, exact token totals and the relocated metadata; `--hash` also sha256-hashes every
  `.ds` / `.index` file (slow, ~160 GB read).

W&B: runs log **online** to `wandb.WANDB_ENTITY / WANDB_PROJECT`. Run `wandb login` with your own key on the training
hosts (`kys3b.py wandb-check` tests for it; `prepare` refuses without it). No credential is part of this release. To
give Huicong access, add Huicong as a member of that project or team in W&B (project / team settings) — never use
Huicong's account or key.

## 4. Plan (all four settings)

`CONFIG_ROOT/plan.json`: `init`, then per setting `trunk1 -> trunk2 -> ... -> trunk6` (each resumes the previous one),
`ep1` after trunk2 (branches from step 8583), `ep2` after trunk4 (17166), `ep3` after trunk6 (25749). The four
settings are independent chains; run them in parallel or one after another. Final models:
`CHECKPOINT_ROOT/seed42/<setting>/ep{1,2,3}/<setting>/seed42/{9537,19073,28610}` (20,000,538,624 / 39,998,980,096 /
59,999,518,720 tokens).

## 5. Shared initialization (once, on the GPUs of one segment)

    bash deploy/portable_3b/create_init.sh $CONFIG_ROOT/site.env      # Slurm: sbatch $CONFIG_ROOT/slurm/init.sbatch

Runs the 1-step init config `init/init.yaml` (derived from the rendered quality-first trunk1: your tokenizer, data,
deployment and W&B project; only the run name, train_steps 1, the init folder, load flags off and
`save_initial_state` differ) and then `kys3b.py finalize-init`, which hashes step 0 and moves it to
`INIT_ROOT/_init_3B_seed42/{0,init_manifest.json}`. An existing init is reused and never overwritten. trunk1 of every
setting is seeded from it by `prepare` (digest-checked copy to `trunk/0`, only when the trunk has no `latest.txt`).

## 6. Recommended first GPU check: bounded startup validation (quality-first trunk1, steps 0 -> 30)

    $P deploy/portable_3b/kys3b.py seed --operator $OP --config $CONFIG_ROOT/grid/3b_quality_first_seed42_trunk1.yaml
    $P deploy/portable_3b/kys3b.py startup-derive --operator $OP --workdir $W
    $P deploy/portable_3b/kys3b.py seed --operator $OP --config $W/cfg/s_straight.yaml
    for ph in s_trunk s_resume s_straight; do       # on every node; nvidia-smi samples to $W/logs/$ph.nvsmi.csv are optional
      bash deploy/portable_3b/launch_torchrun.sh $CONFIG_ROOT/site.env \
          $CONFIG_ROOT/grid/3b_quality_first_seed42_trunk1.env $W/cfg/$ph.yaml $W/logs/$ph.log; done
    rm -rf $W/ckpt/straight
    $P deploy/portable_3b/kys3b.py startup-summarize --operator $OP --workdir $W [--mem-budget-gib <your GPU limit>]

It trains 20 steps into the real quality-first trunk folder, resumes 20 -> 30 with optimizer + LR state, and compares
with 30 uninterrupted steps: finite loss, logged LR = scheduler, consumed tokens = step x 2,097,152, resume within
tolerance of the uninterrupted run. Production trunk1 then continues from step 30 with full state.

## 7. Train

    Slurm:   bash $CONFIG_ROOT/slurm/submit_grid.sh         # init + 4 x 9 jobs with afterok dependencies; review it first
    other:   bash deploy/portable_3b/run_segment.sh $CONFIG_ROOT/site.env <run>    # one node per segment

`run_segment.sh` = `kys3b.py prepare` (checks; seeds trunk1; trunk1 past step 0 gets optimizer + LR state; refuses a
missing resume source, a resume source written at another data-parallel size, a changed deployment, a drifted config,
unrelocated data or a missing W&B login) -> `launch_torchrun.sh` on every node (srun inside a multi-node Slurm
allocation) -> `kys3b.py finish` (the final checkpoint must exist, so dependents never start on a failed segment). It
is requeue-safe: a finished segment exits 0 at once, a trunk segment continues from `latest.txt`, a branch restarts
from its fixed source step.

**Multi-node without Slurm:** run `kys3b.py prepare --operator $OP --run <run>` once (it prints the config to run),
start `launch_torchrun.sh $CONFIG_ROOT/site.env $CONFIG_ROOT/grid/<run>.env <printed config> <log>` on every node with
the rendezvous values resolved (e.g. export MASTER_ADDR), then `kys3b.py finish --operator $OP --run <run>`.

## 8. What was verified, and what was not

On CPU, in a clean environment installed with `install_env.sh` (no CUDA toolkit), with temporary operator configs
whose paths and W&B values exist nowhere else: `pip check`, every pin, CPU imports, the compiled helper; operator
validation; rendering of all 36 configs for several deployments with the science identical to the reviewed configs;
batch accounting; branch sources and output namespaces; W&B routing; the init config derived from the supplied config
(the CLI itself); init finalize/seed/continuation rules; LR continuity over resumes, requeues and branches for all four
settings (`deploy/portable_3b/test_*.py`).

NOT verified (needs your hardware): CUDA/NCCL/flash-attn at run time, multi-node rendezvous, the rendered sbatch files
on your Slurm, memory per GPU for your micro-batch size, throughput, and the actual init + training runs. Section 6 is
the bounded first check of those.
