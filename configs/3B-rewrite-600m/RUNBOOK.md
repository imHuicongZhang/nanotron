# 3B rewrite-600m grid: resumable runbook

**Status (2026-10-17, owner authorization of the same day):**
- Tokenization: CPU jobs 1126253 (tokenized all three corpora; its exact-total check failed by the empty-text documents,
  see Stage 1) and 1126433 (verify-only reuse of that output with the corrected accounting).
- LR resume off-by-one fixed in this worktree (`src/nanotron/serialize/optimizer.py`, regression test
  `deploy/skipjack_3b_600m/test_lr_resume_3b.py`; report `training/reports/lr_resume_fix_20261017/`).
- Durable waiter `rw3b_handoff_waiter` running (Stage 2). It may submit only the smoke test, after the handoff.
- `training/reports/SMOKE_AUTHORIZED` records the owner's smoke-only authorization. LAUNCH_APPROVED and SMOKE_ACCEPTED
  are NOT written: full training stays owner-gated.
- Recovery after a lost session: `training/reports/RECOVERY.md`.

Every command below runs from the 3B worktree with the shared training environment, used read-only:

    cd /weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys-3b600m
    P=/weka/scratch/jhu/bvandur1/zhuicon1/envs/pretrain/bin/python

Never run any of this from `projects/nanotron-kys`. Never switch that checkout or edit its files: its pending
baseline jobs `cd` into it at start time.

## Namespaces (all new)
| what | path |
|---|---|
| code | worktree above, branch `rewrite3b-600m-train` (base bc131392) |
| configs | `configs/3B-rewrite-600m/seed42/3b_<setting>_seed42_<trunk1..6/ep1..3>.yaml` (+ `.env`) |
| tokenized data, tokenizer | `/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m/training/{tokenized/<setting>,tokenizer}` |
| reports, state, logs, W&B | `…/rewrite-3b-600m/training/{reports,reports/capped_submit/seed42,logs/slurm,wandb}` |
| init | `/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite3b_600m/init/_init_3B_seed42/{0,init_manifest.json}` |
| checkpoints | `/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite3b_600m/runs/seed42/<setting>/{trunk,ep1,ep2,ep3}/<setting>/seed42/<step>` |
| W&B project | `zhc-3b-600m-20b-wsd` (entity imwytro-johns-hopkins-university) |
| Slurm names | GPU jobs `kys_3b_*` (counted by both submitters against the shared 20); CPU jobs `rw3b_*` |

## Stage 0: preflight (CPU, safe any time; no job is queued)
    $P deploy/skipjack_3b_600m/generate_configs_3b.py               # refuses once a submission state exists
    $P deploy/skipjack_3b_600m/assert_invariants_3b.py --all
    $P deploy/skipjack_3b_600m/capped_submit_3b.py plan --s-per-it 31
    $P deploy/skipjack_3b_600m/capped_submit_3b.py test-only --s-per-it 31
    $P deploy/skipjack_3b_600m/handoff_gate.py check --require-approval
    PYTHONPATH=<dir with pytest> $P -m pytest -q -p no:cacheprovider deploy/skipjack_3b_600m/test_capped_submit_3b.py

## Stage 1: tokenization (CPU only; needs owner authorization; doesn't depend on the handoff)
    sbatch --output=/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m/training/logs/slurm/%x.%A_%a.log \
           deploy/skipjack_3b_600m/tokenize_3b.sbatch

This is an array of 3 jobs (64 CPU / 320G / 12 h each, partition med) that reads `05_final/<s>/shuffled` only. Each job:
1. checks the tokenizer sha256 and each part's sha256 against `validate_assembly_scope.json`;
2. runs one datatrove task per part (65 / 54 / 54 shards), using a copy of the exact 1.5B tokenizer directory with all three files sha256-checked;
3. rewrites `.ds.metadata` to the tokenizer directory;
4. requires the summed tokens to equal corpus_train_tokens minus the empty-text documents exactly
   (20,000,000,270 − 27 / 19,834,314,433 − 2 / 20,000,000,227 − 8; `registry_3b.TOKENIZED_TOKENS`). Stage 04's strip
   left 27/2/8 retained rows with empty text; the manifest counts one EOS for each, datatrove skips them. datatrove's
   documents-read count must equal rows − empty;
5. writes `reports/tokenize/<s>/epoch_stats.json` (exact windows per pass and passes at each horizon).

A complete earlier run (all datatrove tasks completed, one .ds per part) is reused and only re-verified; any other
non-empty output folder is refused. On failure, inspect the log, move the partial folder aside (don't delete before
review), and resubmit with `--array=<i>`.

## Stage 2: handoff (wait for the full baseline workflow)
The gate (`handoff_gate.py`) passes only when **all** of these hold:
- `seed{42,43,44}/SEED_COMPLETE` exists under `data/1p5_baseline/pretraining/reports/capped_submit/`. Each is written
  by hand after that seed's finals are verified, uploaded and evaluated and its GPUs are released.
- All 63 baseline branch finals are on disk.
- No live `kys_` job other than `kys_3b_*` exists.
- The seed list is re-read from the active `deploy/clusters.yaml` on every check.
- The owner has written `…/training/reports/LAUNCH_APPROVED`.

- For the smoke test only (`--scope smoke`), the owner's `SMOKE_AUTHORIZED` substitutes for LAUNCH_APPROVED.
- GPU evaluation is evidenced by SEED_COMPLETE (the baseline's own "verified, uploaded, evaluated, GPUs released"
  record) plus no live `kys_` job; the baseline tools define no machine-readable eval output.

An empty queue or one finished seed never passes. Durable waiter (`handoff_waiter.py`; CPU, one tick every 2 h):
- it queues its successor first (`--dependency=singleton`), so a crash never ends the chain;
- the gate must pass on 2 consecutive ticks;
- then `capped_submit_3b.py smoke`, which re-checks the gate, the 20-GPU cap, tokenization (`--corpus`,
  epoch_stats) and the LR regression test;
- it then follows the smoke job, writes `reports/handoff/smoke_finished.json` and stops. It never submits the grid.

    sbatch --output=/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m/training/logs/slurm/rw3b_handoff_waiter.log \
           deploy/skipjack_3b_600m/handoff_waiter.sbatch          # start (a second chain refuses to run)
    $P deploy/skipjack_3b_600m/handoff_waiter.py status            # last tick, smoke_submitted / smoke_finished
    touch /weka/projects/bvandur1/zhuicon1/rewrite-3b-600m/training/reports/handoff/STOP   # stop the chain

## Stage 3: smoke test (4 H100, ~2 h, limit 4 h)
    $P deploy/skipjack_3b_600m/capped_submit_3b.py smoke             # gate + cap; job kys_3b_smoke

Phases are in `smoke_3b.py`:
- creates the shared init if absent and pins it in `init_manifest.json`;
- layout A: 20 steps, then resume to 30, then a branch from 20 that decays to 0 at 30, then 30 uninterrupted steps
  (`a_straight`) as the reference for the resume;
- every logged lr is checked against nanotron's scheduler (the shifted-resume signature fails the branch check), and
  consumed_tokens against step × 2,097,152;
- resume vs uninterrupted: loss difference at steps 21-30 within max(0.02, display rounding, 3 × the 1-20 noise);
- 3 steps each on DIV and DA;
- layout B: 10 steps;
- optional layout C (ZeRO-1, mbs 32, the 1.5B micro-batch): 10 steps, run last. An out-of-memory there is recorded, not a smoke failure.

`smoke_report.json` lists s/it, peak GiB, finite loss and grad norm, resume and branch checks, and any problems.
After reviewing it, the owner writes `…/training/reports/SMOKE_ACCEPTED`:

    {"layout": "A", "s_per_it": <median s/it of the chosen layout>, "smoke_job": <id>, "accepted_by": "...", "date": "..."}

If layout B or C is chosen: `$P deploy/skipjack_3b_600m/generate_configs_3b.py --layout B` (or `--layout C`), then rerun Stage 0's
`plan` / `test-only`. A smoke test shows stability, memory, throughput and mechanics. It doesn't validate the LR choice.

## Stage 4: training grid (27 segments, ≤ 20 GPUs)
    $P deploy/skipjack_3b_600m/capped_submit_3b.py submit --s-per-it <SMOKE_ACCEPTED s_per_it>
    $P deploy/skipjack_3b_600m/capped_submit_3b.py status

The submitter refuses unless all of these hold:
- the handoff and approval;
- SMOKE_ACCEPTED matches the rendered layout;
- `assert_invariants_3b --all --corpus` passes;
- the init matches its manifest;
- no non-3B `kys_` job is live.

How it places work:
- Trunks are seeded from the init (identical weights).
- There are 5 serial `afterok` lanes: lanes 0-2 hold the three trunks (6 segments each) followed by their ep3; lanes 3-4
  hold ep1 (after trunk2) and ep2 (after trunk4).
- Peak is 20 GPUs, enforced by Slurm dependencies.

**Resume and retry.**
- Rerunning `submit` is always safe: it reconciles squeue, sacct and the checkpoints on disk, and submits only what is
  missing.
- A failed segment halts its lane. Inspect its log in `training/logs/slurm/`, then
  `submit --s-per-it <same> --retry <segment>`.
- Requeued jobs resume from `latest.txt` (trunk) or re-run from the branch point (branch).

## Stage 5: evaluation (after all 27 training jobs have ended)
Before each evaluation job: `$P deploy/skipjack_3b_600m/capped_submit_3b.py guard --need-gpus <n>`, and name the job
`kys_3b_eval_*`. Use the kys-eval v2 protocol (LightEval 0-shot acc_norm, 6 commonsense tasks + MMLU) on
`…/runs/seed42/<setting>/ep{1,2,3}/<setting>/seed42/{9537,19073,28610}` after `tools/kys_eval/convert_to_hf.py`.
No upload is authorized by this runbook.

## Recovery quick reference
| symptom | action |
|---|---|
| tokenize task FAILED / OOM | read `training/logs/slurm/rw3b_tokenize.*`; `KYS3B_TOKENIZE_WORKERS=8 sbatch --array=<i> …` after moving the partial folder aside |
| smoke peak > 74.5 GiB on A | accept B (or mbs 8 with ZeRO-1: add a layout to registry_3b, re-render, re-smoke) |
| s/it > 38 | split the trunk further (`registry_3b.TRUNK_ENDS`; branch points must remain segment ends), re-render, re-plan (before any submit) |
| `submit` refuses: other kys_ jobs | a baseline job is live; the handoff isn't complete; wait |
| lane halted | inspect, then `--retry <segment>`; never resubmit by hand with sbatch |
