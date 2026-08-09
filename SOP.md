# KYS grid — standard operating procedure

72 logical runs: 6 settings × 3 seeds × (1 trunk + 3 cooldown branches), emitted as 108
config templates (each trunk is 3 chained segments). H200, `dp=8 tp=1 pp=1`, single node.

---

## 1. The one thing not to change: `micro_batch_size = 16`

**It is pinned for numerical comparability, not tuned for throughput.** 109.7 GiB of 141 GB
is 84%, and the remaining headroom is deliberately unused.

`Loss.forward` (`src/nanotron/models/llama.py`) computes

```python
loss = masked_mean(loss, label_mask, dtype=torch.float)   # (loss*mask).sum() / mask.sum()
```

and `label_mask` drops the token before **every document boundary** — `return_positions`
defaults True, datatrove supplies per-token `positions`, and the collator masks where
`position_ids == 0`. Measured on the real corpus: **2.8 masked tokens per 2048-token
sequence, range 0–10**. So `masked_mean`'s denominator `n_j` differs from micro-batch to
micro-batch, and after `loss_j / accum` the effective per-token weight is `1/(n_micro · n_j)`.

Regrouping the same 1024 sequences into a different micro-batch size therefore changes the
objective:

| comparison | max relative per-token weight difference |
|---|---:|
| mbs=4 dp=4 accum=64 **vs** mbs=4 dp=8 accum=32 | **0.000e+00 — identical** |
| mbs=4 vs mbs=8 | 1.04e-3 |
| mbs=4 vs mbs=16 | 1.43e-3 |
| mbs=8 vs mbs=16 | 7.04e-4 |

1.43e-3 is roughly **four orders of magnitude above fp32 rounding**. A run at a different
`mbs` is measuring a slightly different objective from the rest of the grid, and *nothing in
the loss curve will look wrong* — which is precisely why this is written down rather than
left as a convention.

Note the first row: **`dp` is mathematically free.** `MegatronPretrainingSampler` fills a
batch of `mbs × dp` consecutive samples and hands rank *d* the slice `[d*mbs : (d+1)*mbs]`,
so the partition into micro-batches is consecutive blocks of size `mbs` regardless of `dp`.
Scale `dp` to fit the cluster; never touch `mbs`.

`tokens/step` staying at 2,097,152 is **not** sufficient evidence that nothing changed —
`mbs=32, accum=4, dp=8` also gives 2,097,152 and is *not* comparable. Check `mbs` itself.

Three guards enforce this:

1. `deploy/clusters.yaml` carries the reasoning inline at the `micro_batch_size` field.
2. `tools/render_config.py` refuses to render when the clusters named in `seed_assignment`
   disagree about `micro_batch_size`, and refuses an `mbs` that will not fit the cluster's HBM.
3. `tools/assert_invariants.py` — run it as preflight, and again against the log at step 200.

```bash
# preflight, before any GPU time
python tools/assert_invariants.py --config rendered/<name>.yaml
# smoke test, after ~200 steps
python tools/assert_invariants.py --config rendered/<name>.yaml --log train.log --at-step 200
```

---

## 2. Launching

Templates in `configs/know-your-sources/` are deliberately **not runnable as-is** — no
`parallelism`, `micro_batch_size`, `batch_accumulation_per_replica`, `zero_stage` or
`sequence_length`. Stamp them:

```bash
python tools/render_config.py \
    --template configs/know-your-sources/quality-first_seed43_trunk1.yaml \
    --cluster h200 --seed 43 \
    --out rendered/quality-first_seed43_trunk1.yaml
```

This writes two files: the config, and a companion `.env` with the wandb wiring. Then:

```bash
set -a; source rendered/quality-first_seed43_trunk1.env; set +a
python tools/assert_invariants.py --config rendered/quality-first_seed43_trunk1.yaml
torchrun --nproc_per_node=8 run_train.py --config-file rendered/quality-first_seed43_trunk1.yaml
```

Order per (setting, seed): `trunk1 → trunk2 → trunk3`, then `ep1`, `ep2`, `ep3`. Each branch
may start as soon as its trunk segment has written its final checkpoint (4292 / 8583 / 12875);
`ep1` and `ep2` therefore overlap with later trunk segments.

Before the first trunk segment of each chain, seed the trunk directory:

```bash
T=$CKPT/kys/<setting>-seed<S>-trunk
mkdir -p $T && cp -al $CKPT/_init_1.5B_seed<S>/0 $T/0 && echo 0 > $T/latest.txt
```

---

## 3. wandb

### 3.1 What nanotron actually does

Verified by reading `src/nanotron/trainer.py` and by running wandb 0.27.0 locally — not from
general wandb knowledge, because the wiring is unusually thin:

| behaviour | finding |
|---|---|
| import | `try: import wandb / except ImportError: wandb = None`. **If wandb is not installed, all logging is silently skipped with no error.** |
| `wandb.init` args | `project=config.general.project`, `name=run_name`, `config={"nanotron_config": ...}`, `settings=...` |
| `entity` | **never passed.** Not a nanotron config field at all. |
| `tags`, `group`, `job_type` | **never passed.** |
| `id` / `resume` | **never passed** — so every process invocation is a new wandb run. |
| who logs | only `world_rank == logger_ranks[0]` (rank 0 at tp=1). |
| step axis | `wandb.log(..., step=self.iteration_step)` — the **absolute** nanotron step. |
| config upload | the full resolved config is uploaded as `config.nanotron_config`, plus the YAML as a file artifact. |
| run name | upstream forces `{dd/mm/YYYY_HH:MM:SS}_{general.run}`. **Patch #8** in this fork uses `general.run` verbatim (set `KYS_WANDB_TIMESTAMP_PREFIX=1` to restore upstream). |

Because `entity`/`tags`/`group` are never passed, they must come from the environment.
Verified that wandb picks all of them up from `WANDB_ENTITY`, `WANDB_TAGS`,
`WANDB_RUN_GROUP`, `WANDB_JOB_TYPE` — `render_config.py` writes exactly those into the
companion `.env`.

### 3.2 Can his runs log into your entity under his own account?

**Yes, and this is the recommended path.** No API key is shared.

*On your side (once):*
1. wandb → your **Team** (org/team entity). If you only have a personal entity, create a Team —
   personal entities cannot take collaborators.
2. Invite his wandb account to that team (Team Settings → Members → Invite). Member role is
   enough; it can create runs in team projects.
3. Create the project `kys-epoch-wsd` inside the team, or let the first run create it.
4. Give him the exact strings for `entity` and `project`. Nothing else.

*On his side (once):*
```bash
pip install wandb            # nanotron logs nothing at all if this is missing
wandb login                  # HIS OWN key, stored in his ~/.netrc
```
Then he only ever sources the generated `.env`; he never types an entity or project.

Runs land in **your** project, attributed to **his** account. You keep ownership; he keeps his
credential.

### 3.3 If that does not work — alternatives, and what each costs

| option | cost |
|---|---|
| **Team invite (above)** | none beyond an invite. **Do this.** |
| **Service-account key shipped with the repo** | A long-lived credential that can write to *any* project in your entity, sitting in a file that gets copied between clusters, into `.env`s, into shell history and into SLURM job environments (`scontrol show job` exposes them). It cannot be scoped per project and revoking it kills every run using it. Only acceptable as a same-day stopgap, never committed — and if it is ever used, rotate it the moment the grid finishes. |
| **He owns the runs, you use a wandb Report / `api.runs()` to pull them** | No credential moves, but the runs live in his entity — if his account or funding lapses you lose the record, and you cannot administer the project. Acceptable as a fallback, worse for a paper. |
| **Offline + ship the run directories, you sync them** | See 3.4. Zero credential exchange, and the runs end up owned by you. Slightly more manual. |

### 3.4 If his compute nodes have no outbound internet

This is common, and nanotron handles it **only** because it does nothing clever: it calls
plain `wandb.init()`, so all standard wandb env vars apply.

Verified locally with `WANDB_API_KEY` unset and `WANDB_MODE=offline`: `wandb.init` succeeds,
the run is written to `$WANDB_DIR/wandb/offline-run-<ts>-<id>/`, metrics log normally, and
wandb prints the resync command. **No API key is needed on the compute node at all.**

Set `wandb.mode: offline` in `deploy/clusters.yaml`; the renderer propagates it to
`WANDB_MODE` in every `.env`. Then, from a login node that does have egress:

```bash
wandb login                                  # once, whoever owns the destination entity
export WANDB_ENTITY=<entity> WANDB_PROJECT=<project>
wandb sync --sync-all                        # or: wandb sync path/to/offline-run-*
```

Two things to get right, or this bites later:
- Point `WANDB_DIR` at shared storage the login node can also see, not node-local `/tmp`.
- Sync with the credential of whoever should **own** the runs. Offline runs carry no
  ownership until sync, so this is the cleanest way for his compute to produce runs that end
  up in your account with no key ever leaving your machine.

Offline runs keep their name, tags, group, config and full step history; the only loss is
live monitoring.

### 3.5 Naming, tags, grouping

`entity` and `project` are **never defaulted**. `render_config.py` aborts while either is
null in `deploy/clusters.yaml`, and also aborts if `wandb.project` disagrees with the
template's `general.project`. This exists because wandb silently falls back to whatever
account holds the cached credential on the machine — which is not hypothetical: during this
investigation an online `wandb.init()` with no `WANDB_API_KEY` set did **not** fail, it
created a run under the cached local login.

Run names (patch #8 makes these exact, no timestamp prefix):

```
{setting}_seed{n}_trunk1   {setting}_seed{n}_trunk2   {setting}_seed{n}_trunk3
{setting}_seed{n}_ep1      {setting}_seed{n}_ep2      {setting}_seed{n}_ep3
```

`{setting}` ∈ `quality-base, quality-first, diversity-first, wrap, rewrite,
signal-disagreement-lambda05`. Config filename == wandb run name == checkpoint dir stem.

Tags (auto-generated into the `.env`):

| tag | purpose |
|---|---|
| `setting:<name>` | filter one arm across seeds |
| `seed:<n>` | filter one seed across arms |
| `kind:<trunk1..3\|ep1..3>` | exact segment |
| `phase:trunk` / `phase:endpoint` | **the main split** — 3 annealed endpoints vs trunk segments |
| `cluster:<h200\|h100>` | which side it ran on |
| `mbs:16`, `dp:8`, `accum:8`, `tok_per_step:2097152` | provenance — audit after the fact which layout a run actually used, without opening 108 configs |

Also set: `WANDB_RUN_GROUP={setting}_seed{n}` (groups the 6 runs of one chain) and
`WANDB_JOB_TYPE={kind}`.

The authoritative record is still `config.nanotron_config.tokens.*`, which nanotron uploads
in full; the tags exist so a wrong `mbs` is visible at a glance in the runs table.

### 3.6 Trunk segments: three runs or one?

**Three separate wandb runs**, and that is the recommendation.

nanotron never passes `id=` or `resume=` to `wandb.init`, so making the three segments one
resumed run would need a further patch. It is not worth it, because the step axis already
works out: `wandb.log(..., step=self.iteration_step)` uses the **absolute** nanotron step, and
the segments cover disjoint, consecutive ranges:

```
trunk1  steps     1 – 4292
trunk2  steps  4293 – 8583
trunk3  steps  8584 – 12875
ep1     steps  4293 – 4768      (branches from trunk1's final checkpoint)
ep2     steps  8584 – 9537
ep3     steps 12876 – 14305
```

Verified that wandb accepts a run whose first logged step is 4293 — there is no requirement
that a run start at 0, only that steps be non-decreasing within a run. So plotting the three
trunk segments together gives one continuous, non-overlapping loss curve; `WANDB_RUN_GROUP`
puts them under one group in the UI, and the branch runs overlay on the same axis at their
correct absolute positions.

---

## 4. What you need before Tianjian starts

**You:** create/choose the Team entity, invite his wandb account, decide the project name,
and send him the two strings. Fill `wandb.entity` / `wandb.project` in
`deploy/clusters.yaml` — nothing renders until you do.

**Him:** `pip install wandb`, `wandb login` with his own key, fill the `slurm:` block and
confirm `gpus_per_node`/`dp` in `deploy/clusters.yaml`, then render and launch. If his
compute nodes have no egress, set `wandb.mode: offline` and sync from a login node per 3.4.
