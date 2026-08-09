# KYS grid — standard operating procedure

72 logical runs: 6 settings × 3 seeds × (1 trunk + 3 cooldown branches), emitted as 108
config templates (each trunk is 3 chained segments). H200, `dp=8 tp=1 pp=1`, single node.

---

## READ FIRST — five ways this pipeline fails silently

Every one of these produces a run that **exits 0 and looks fine**. None shows up in a loss
curve. Each has a guard; the guards only help if they are actually run.

| # | failure | what you see | guard |
|---|---|---|---|
| 1 | **`wandb sync` reports success but uploads nothing** | "done", then you count runs later and find 40 instead of 108 | §4.6 — a *failed* sync still writes `.synced`; the retry needs `--include-synced` |
| 2 | **`wandb` package not installed** | training completes with zero metrics and no error at all | `assert_invariants.py` env preflight |
| 3 | **`resume_checkpoint_path` doesn't resolve** | a cooldown branch trains from **random init**, exits 0, writes a checkpoint | `assert_invariants.py --check-resume` |
| 4 | **right path, wrong corpus** | trains to completion on the wrong data | `assert_invariants.py` token-count check |
| 5 | **`micro_batch_size` raised to "use the headroom"** | loss curve looks perfect; the run is no longer comparable to the other 71 | §1, and `--log … --at-step 200` |

After rendering, and again immediately before each launch:

```bash
python tools/assert_invariants.py --config rendered/<name>.yaml --check-resume
```

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

## 2. Paths: four values, and nothing else

**The 108 templates contain no absolute paths at all** — verified, `grep -rlE "/(scratch|weka|shared)/" configs/know-your-sources/` returns 0. Every location is composed at render time from `deploy/clusters.yaml`:

| field | what it is |
|---|---|
| `data_root` | directory holding the six tokenized corpora |
| `tokenizer_path` | the llama2-unsloth tokenizer **directory** (not `tokenizer.json`) |
| `ckpt_root` | where checkpoints are written (reserve ~5.3 TB, 2.3 TB after pruning) |
| `wandb.dir` | where offline run directories go (shared storage — §4.2) |

That is the complete set. **Do not edit paths in the config templates**; `render_config.py` refuses to render a template that sets any of them.

### Why the setting → corpus mapping is not configurable

It lives hardcoded in `render_config.py` because three naming schemes are in play for the same six arms and they do not line up:

| paper setting | repo folder | internal run name | corpus dir |
|---|---|---|---|
| QUALITY-BASE | `quality_base` | `quality_base` | **`10B-base-shuf42`** |
| QUALITY-FIRST | `quality_first` | `quality_first` | `quality-first` |
| DIVERSITY-ORIENTED | `diversity_oriented` | `diversity_oriented` | **`diversity-first`** |
| WRAP-INSPIRED | `wrap_inspired` | `wrap` | `wrap` |
| REWIRE-INSPIRED | `rewire_inspired` | `rewire` | **`rewrite`** |
| DISAGREEMENT-AWARE | `disagreement_aware` | `disagreement_aware_0p5` | **`signal-disagreement-lambda05`** |

Wiring these by hand gets at least one wrong, and **a wrong-but-existing path does not crash**: nanotron reads whatever corpus is there, trains to completion, and the numbers are meaningless. So nobody retypes them — `data_root` is set once and the table does the rest.

### The two silent failures this prevents, and how they are caught

| failure | what nanotron does | caught by |
|---|---|---|
| right path, **wrong corpus** | trains normally on the wrong data | `assert_invariants.py` token-count check |
| `resume_checkpoint_path` doesn't resolve | `serialize/main.py:231` logs *"No previous checkpoint found"* at **INFO** and returns `None` → **starts from random init**. A cooldown branch runs its 476 steps, exits 0, and has annealed noise | `assert_invariants.py --check-resume` |

```bash
python tools/assert_invariants.py --config rendered/<name>.yaml                  # after render
python tools/assert_invariants.py --config rendered/<name>.yaml --check-resume   # before launch
```

The corpus check compares the summed `.ds.metadata` token counts against the recorded value for that corpus (cross-checked against raw `.ds` bytes/2; they match exactly for all six). It caught a deliberately mis-pointed `wrap` → `rewrite` corpus on a 264-token difference. Note `diversity-first` is legitimately ~1.1% short of 10B — a property of that corpus, not an error.

## 3. Launching

Templates in `configs/know-your-sources/` are deliberately **not runnable as-is** — no
`parallelism`, `micro_batch_size`, `batch_accumulation_per_replica`, `zero_stage` or
`sequence_length`. Stamp them:

```bash
python tools/render_config.py \
    --template configs/know-your-sources/quality_first_seed43_trunk1.yaml \
    --cluster h200 --seed 43 \
    --out rendered/quality_first_seed43_trunk1.yaml
```

This writes two files: the config, and a companion `.env` with the wandb wiring. Then:

```bash
set -a; source rendered/quality_first_seed43_trunk1.env; set +a
python tools/assert_invariants.py --config rendered/quality_first_seed43_trunk1.yaml --check-resume
torchrun --nproc_per_node=8 run_train.py --config-file rendered/quality_first_seed43_trunk1.yaml
```

Order per (setting, seed): `trunk1 → trunk2 → trunk3`, then `ep1`, `ep2`, `ep3`. Each branch
may start as soon as its trunk segment has written its final checkpoint (4292 / 8583 / 12875);
`ep1` and `ep2` therefore overlap with later trunk segments.

Before the first trunk segment of each chain, seed the trunk directory:

```bash
T=<ckpt_root>/<setting>_seed<S>_trunk          # note underscores: dir stem == run name
mkdir -p $T && cp -al <init_root>/_init_1.5B_seed<S>/0 $T/0 && echo 0 > $T/latest.txt
```

`cp -al` hardlinks, so seeding all 18 trunks costs no extra disk. Verify afterwards with
`tools/hash_init_checkpoint.py $T/0 --check init_1.5B_seed<S>.hash.json`.

---

## 4. wandb

### 4.1 What nanotron actually does

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

### 4.2 The chosen flow: offline on his side, synced by you

**The grid runs `WANDB_MODE=offline`. Tianjian never syncs. You sync, and that makes the runs
yours.**

Why this beats a team invite: an offline run carries **no ownership until it is synced** —
verified, `run.entity` is the empty string when `WANDB_ENTITY` is unset at init, and nothing
about a destination is written into the run directory. The entity is chosen entirely at sync
time. Consequences:

- your API key never leaves your machine;
- he needs **no wandb account at all** — only `pip install wandb`;
- it does not matter whether his compute nodes have outbound internet;
- the runs end up owned and administered by you, which is what you want for a paper.

`deploy/clusters.yaml` therefore has `mode: offline`, `project: zhc-1p5b-10b-wsd`, and
**`entity: null` on purpose**. `render_config.py` refuses to render if `entity` is set while
`mode: offline`, and never writes `WANDB_ENTITY` into the `.env`.

**His side — once:**
```bash
pip install wandb          # nanotron silently logs NOTHING if this is missing
```
No `wandb login`. No key. He fills `wandb.dir` in `deploy/clusters.yaml` (see below), sources
each generated `.env`, and launches. That is the whole of his involvement.

> **`wandb.dir` must be shared storage.** It becomes `WANDB_DIR`, where every offline run
> directory is written. If it is unset, wandb falls back to the process working directory —
> on a compute node that is the standard way to lose an entire grid's logs. It must (a)
> outlive the job, (b) be readable from wherever the sync happens. `render_config.py` refuses
> to render for offline mode while `wandb.dir` is null.

**He does NOT run `wandb sync`.** He leaves the `offline-run-*` directories in place and tells
you the path. The generated `.env` says so in a comment, for whoever reads it at 3am.

**Your side — after the grid finishes:**
```bash
wandb login                                   # your key, your machine
wandb sync --entity <YOUR_ENTITY> --project zhc-1p5b-10b-wsd --sync-all <path>/wandb
```
`wandb sync` takes `-e/--entity` and `-p/--project` explicitly (confirmed in `wandb sync
--help` for wandb 0.27.0), so the destination is stated at sync time rather than inherited
from whatever environment happens to be loaded.

**Both directions of the entity question were tested end to end (2026-08-09):**

| test | result |
|---|---|
| offline `wandb.init`, `WANDB_ENTITY` unset | `run.entity == ''` — no destination baked in |
| `wandb sync` with `WANDB_ENTITY` = a **bogus** entity | `ERROR ... entity ... not found (404)`. **Nothing was created** — verified the project did not appear under the cached-credential account. It does *not* silently fall back to `~/.netrc`. |
| `wandb sync` with `WANDB_ENTITY` = the real entity | run lands in exactly that entity, name and history intact |

So supplying the entity at sync time is reliable: a wrong value is a loud 404, not a
misfiled run.

> **Gotcha, and this one will bite.** A *failed* sync still writes a `.synced` marker into the
> offline run directory. `wandb sync --sync-all` then **skips** that run on retry and reports
> success having uploaded nothing. After any failed or partial sync, retry with
> `--include-synced`:
>
> ```bash
> wandb sync --include-synced --entity <YOUR_ENTITY> --project zhc-1p5b-10b-wsd <path>/wandb
> ```
>
> Always reconcile the count afterwards: **108 runs** expected in the project.

Offline runs keep their name, tags, group, config and full step history; the only thing lost
is live monitoring during the run.

### 4.3 Alternatives that were considered, and what each costs

| option | cost |
|---|---|
| **Offline + you sync (chosen)** | No credential exchange, no account needed on his side, runs owned by you. Cost: no live monitoring, and one manual sync step at the end. |
| **Team invite — he logs into your entity with his own key** | Works (nanotron reads `WANDB_ENTITY` from the environment; verified). Cost: he needs a wandb account, an invite, and his compute nodes need egress. Strictly more moving parts than offline for no gain here. |
| **Service-account key shipped with the repo** | A long-lived credential that can write to *any* project in your entity, sitting in a file copied between clusters, into `.env`s, shell history, and SLURM job environments (`scontrol show job` exposes them). Cannot be scoped per project; revoking it kills every run using it. Same-day stopgap at best, never committed, rotate immediately after. |
| **He owns the runs, you pull with `api.runs()`** | No credential moves, but the record lives in his account — if it lapses you lose it and cannot administer the project. Worse for a paper. |

### 4.4 Naming, tags, grouping

`project` is **never defaulted**: `render_config.py` aborts while `wandb.project` is null, and
aborts again if it disagrees with the template's `general.project`. `entity` is not set at all
in offline mode (§4.2) — it is supplied by `wandb sync --entity`.

These guards exist because wandb's fallback is silent, not loud. During this investigation an
online `wandb.init()` with **no `WANDB_API_KEY` set** did *not* fail — it found a cached
credential in `~/.netrc` and created a real run under that account. "No key" does not mean
"no upload".

Run names (patch #8 makes these exact, no timestamp prefix):

```
{setting}_seed{n}_trunk1   {setting}_seed{n}_trunk2   {setting}_seed{n}_trunk3
{setting}_seed{n}_ep1      {setting}_seed{n}_ep2      {setting}_seed{n}_ep3
```

`{setting}` ∈ `quality_base, quality_first, diversity_oriented, wrap, rewire,
disagreement_aware_0p5`. Config filename == wandb run name == checkpoint dir stem.

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

### 4.5 Trunk segments: three runs or one?

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

## 5. Division of labour

**You — before he starts:** nothing blocking. `wandb.project` is already `zhc-1p5b-10b-wsd` and
`mode: offline`; no entity is needed until sync. Optionally run the 30-second sync probe in
§4.2 so the destination is confirmed before there is anything valuable to lose.

**You — after the grid finishes:** `wandb login`, then
`wandb sync --entity <YOUR_ENTITY> --project zhc-1p5b-10b-wsd --sync-all <his path>/wandb`.
That single command is what makes the 108 runs yours.

**Him — once:** `pip install wandb` (no login, no key, no account). Fill in exactly five
values in `deploy/clusters.yaml` and nothing else:

| field | value |
|---|---|
| `data_root` | where the six tokenized corpora were unpacked |
| `tokenizer_path` | the tokenizer **directory** |
| `ckpt_root` | checkpoint destination (~5.3 TB) |
| `wandb.dir` | shared storage for offline runs (§4.2) |
| `clusters.h200.slurm.*` | partition / gres / time, and confirm `gpus_per_node` / `dp` |

He does **not** touch dataset paths, the setting→corpus mapping, `micro_batch_size`, or any
of the 108 templates. `render_config.py` refuses to render while any of the four paths is
null, so there is no silent-default path.

**Him — per run:** render, source the `.env`, run the preflight assert, launch. He does
**not** run `wandb sync`; he leaves the `offline-run-*` directories where they are and tells
you the path.

### 4.6 Syncing — the trap that loses runs silently

**A failed `wandb sync` still marks the run directory as synced.** Verified on wandb 0.27.0:
a sync that died with `ERROR ... entity ... not found (404)` and uploaded nothing still wrote
`run-<id>.wandb.synced` into the offline run directory. `wandb sync --sync-all` then **skips
that run on every subsequent attempt and reports success**.

This is the failure mode where you come back later expecting 108 runs and find 40, with no
error anywhere to explain the other 68.

**Always, after any sync:**

```bash
# 1. sync, naming the destination explicitly
wandb sync --entity <YOUR_ENTITY> --project zhc-1p5b-10b-wsd --sync-all <path>/wandb

# 2. RECONCILE. This is not optional.
python - <<'PY'
import wandb
runs = list(wandb.Api().runs("<YOUR_ENTITY>/zhc-1p5b-10b-wsd"))
print(f"{len(runs)} runs in project (expected 108)")
missing = {f"{s}_seed{d}_{k}"
           for s in ["quality_base","quality_first","diversity_oriented","wrap","rewire",
                     "disagreement_aware_0p5"]
           for d in (42,43,44)
           for k in ("trunk1","trunk2","trunk3","ep1","ep2","ep3")} - {r.name for r in runs}
print(f"missing ({len(missing)}):", sorted(missing)[:10])
PY

# 3. if anything is missing, retry with --include-synced or the .synced marker will skip it
wandb sync --include-synced --entity <YOUR_ENTITY> --project zhc-1p5b-10b-wsd <path>/wandb
```

`--include-synced` is the only way past the marker short of deleting the `.synced` files by
hand. Re-syncing an already-uploaded run is idempotent (same run id), so it is safe to pass
it whenever the count is short.
