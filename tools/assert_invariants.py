#!/usr/bin/env python
"""Fail fast if the batch invariants have drifted.

Two invariants hold across all 72 runs of the grid, and both must be true or the runs are
not comparable to each other:

    micro_batch_size == 16                  (pinned for NUMERICAL comparability, not speed)
    mbs * accum * dp * seq_len == 2,097,152 tokens/step

The second is what most people check. The first is the one that actually gets broken,
because 109.7 GiB of 141 GB looks like headroom. It is not: `Loss.forward` computes
`masked_mean(loss, label_mask)` and `label_mask` drops the token before every document
boundary, so masked_mean's denominator varies per micro-batch and the effective per-token
weight is `1/(n_micro * n_j)`. Regrouping into a different mbs moves those weights by
1.43e-3 (mbs=4 vs 16) — four orders of magnitude above fp32 rounding. Nothing in a loss
curve will look wrong; the runs are simply no longer measuring the same objective.

Run it two ways:

  preflight (before any GPU time — this is the cheap one):
      python tools/assert_invariants.py --config rendered/quality_base_seed42_trunk1.yaml

  smoke test (after ~200 steps of a real run, catches an edited config or a stale env):
      python tools/assert_invariants.py --config rendered/x.yaml --log train.log --at-step 200

Exit 0 = invariants hold. Exit 1 = they do not; do not start the grid.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

EXPECTED_MBS = 16
EXPECTED_TOK_PER_STEP = 2_097_152

# Corpus identity, keyed by the corpus subdirectory name. Token counts are the sum of line 2
# of every *.ds.metadata in the folder, cross-checked against raw .ds bytes/2 (they matched
# exactly for all six on 2026-08-09). Source: tmp/kys/manifest/provenance.tsv, Gate 4.
#
# This is the check that catches a path which EXISTS but holds the wrong corpus. That failure
# is otherwise silent: nanotron starts, trains for 27 hours, and produces numbers that mean
# nothing. Note diversity-first is legitimately ~1.1% short of 10B — that is a property of the
# corpus, not an error.
EXPECTED_CORPUS = {
    '10B-base-shuf42':              (10_000_003_137, 16),
    '10B-base':                     (10_000_003_137, 16),   # pre-shuffle; should NOT be used
    'quality-first':                (10_000_002_634, 16),
    'diversity-first':              ( 9_889_637_833, 16),
    'wrap':                         (10_000_002_419, 16),
    'rewrite':                      (10_000_002_683, 16),
    'signal-disagreement-lambda05': (10_000_002_333, 16),
}
DEPRECATED_CORPUS = {
    '10B-base': 'the UNSHUFFLED quality_base corpus. Its .ds stream is a 16-period sawtooth of '
                'pure-upper / pure-lower quality strata, so every optimizer step draws its whole '
                'batch from one stratum. Use 10B-base-shuf42.',
}

# nanotron/trainer.py logs this banner once at startup:
#   mbs: 16 | grad_accum: 8 | cp: 1 | sequence_length: 2048 | global_batch_size: 1024 | ...
BANNER = re.compile(
    r"mbs:\s*(\d+)\s*\|\s*grad_accum:\s*(\d+)\s*\|\s*cp:\s*(\d+)\s*\|\s*"
    r"sequence_length:\s*(\d+)\s*\|\s*global_batch_size:\s*(\d+)"
)
# and per iteration:  iteration: 200 / 4292 | consumed_tokens: 419M | ...
ITER = re.compile(r"iteration:\s*(\d+)\s*/\s*\d+.*?consumed_tokens:\s*([\d.]+)([KMBT]?)")
SUFFIX = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def fail(msgs):
    print("INVARIANT VIOLATION", file=sys.stderr)
    for m in msgs:
        print(f"  - {m}", file=sys.stderr)
    print("\n  mbs is pinned for numerical comparability across the 72 runs, not for speed.", file=sys.stderr)
    print("  See deploy/clusters.yaml and SOP.md before changing anything.", file=sys.stderr)
    sys.exit(1)


def check_config(path: Path):
    cfg = yaml.safe_load(path.read_text())
    errs = []
    try:
        mbs = cfg["tokens"]["micro_batch_size"]
        accum = cfg["tokens"]["batch_accumulation_per_replica"]
        seq = cfg["tokens"]["sequence_length"]
        dp = cfg["parallelism"]["dp"]
    except KeyError as e:
        fail([f"{path}: missing {e} — is this a RENDERED config, or a raw template? "
              f"Templates carry no parallelism/tokens deployment keys by design."])
    tok = mbs * accum * dp * seq
    print(f"config  : {path}")
    print(f"          mbs={mbs} accum={accum} dp={dp} seq={seq} -> {tok:,} tokens/step")
    if mbs != EXPECTED_MBS:
        errs.append(f"micro_batch_size is {mbs}, expected {EXPECTED_MBS}")
    if tok != EXPECTED_TOK_PER_STEP:
        errs.append(f"tokens/step is {tok:,}, expected {EXPECTED_TOK_PER_STEP:,}")
    if mbs * accum * dp != cfg["tokens"].get("_gbs", mbs * accum * dp):
        errs.append("global batch inconsistent")
    return errs, (mbs, accum, dp, seq)


def check_env(cfg_path: Path):
    """Preflight the logging environment.

    The headline case: nanotron does `try: import wandb / except ImportError: wandb = None`
    and then guards every logging call with `if wandb is not None`. So a MISSING wandb
    package produces a training run with **zero logs and no error at all** — 72 runs later
    you have checkpoints and no loss curves. Nothing else in the pipeline notices.
    """
    import os
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    errs = []

    try:
        import wandb
        print(f"wandb   : {wandb.__version__}")
    except ImportError:
        return ["wandb is NOT installed in this environment. nanotron wraps `import wandb` in "
                "try/except ImportError and silently skips ALL logging when it fails — the run "
                "trains to completion with no metrics and no warning. `pip install wandb`."]

    mode = os.environ.get("WANDB_MODE")
    proj = os.environ.get("WANDB_PROJECT")
    ent = os.environ.get("WANDB_ENTITY")
    wdir = os.environ.get("WANDB_DIR")
    cfg_proj = cfg.get("general", {}).get("project")
    print(f"          WANDB_MODE={mode} WANDB_PROJECT={proj} WANDB_DIR={wdir}")

    if mode is None:
        errs.append("WANDB_MODE is unset — did you `source` the generated .env? Without it "
                    "wandb defaults to online and will use whatever cached credential exists "
                    "on this machine.")
    elif mode == "offline":
        if ent:
            errs.append(f"WANDB_ENTITY is set ({ent}) but mode is offline. Offline runs must "
                        f"stay unattributed; the owner is chosen by `wandb sync --entity`.")
        if not wdir:
            errs.append("WANDB_DIR is unset in offline mode — wandb writes run directories to "
                        "the working directory, which on a compute node is how a whole grid's "
                        "logs get lost.")
        else:
            p = Path(wdir)
            if not p.is_dir():
                errs.append(f"WANDB_DIR {p} does not exist")
            elif not os.access(p, os.W_OK):
                errs.append(f"WANDB_DIR {p} is not writable")
            if str(p).startswith(("/tmp", "/var/tmp", "/dev/shm", "/scratch/local")):
                errs.append(f"WANDB_DIR {p} looks node-local. Offline runs must land on shared "
                            f"storage that outlives the job and is readable from the sync host.")
    if proj and cfg_proj and proj != cfg_proj:
        errs.append(f"WANDB_PROJECT={proj} but config general.project={cfg_proj}; nanotron "
                    f"passes the CONFIG value to wandb.init, so the env var would be ignored "
                    f"and the run would land in {cfg_proj}.")
    return errs


def check_corpus(cfg_path: Path):
    """Verify the rendered dataset_folder points at the corpus it claims to.

    Four ways this goes wrong, all caught here rather than 27 hours in:
      - data_root typo               -> folder missing
      - corpus not downloaded yet    -> folder present but no .ds
      - .ds.metadata not shipped     -> nanotron's own vocab_size assert would fire at startup
      - RIGHT path, WRONG corpus     -> token count disagrees. This is the silent one.
    """
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    errs = []
    try:
        folders = cfg["data_stages"][0]["data"]["dataset"]["dataset_folder"]
    except (KeyError, TypeError):
        return ["rendered config has no data_stages[0].data.dataset.dataset_folder — "
                "did render_config.py fail to compose it from data_root?"]
    if len(folders) != 1:
        errs.append(f"expected exactly 1 dataset_folder, got {len(folders)}: {folders}")
    for f in folders:
        p = Path(f)
        corpus = p.parent.name                      # <data_root>/<corpus>/tokenized
        print(f"corpus  : {corpus}  ({p})")
        if corpus in DEPRECATED_CORPUS:
            errs.append(f"{corpus} is deprecated: {DEPRECATED_CORPUS[corpus]}")
        if not p.is_dir():
            errs.append(f"{p} does not exist or is not a directory")
            continue
        ds = sorted(p.glob("*.ds"))
        meta = sorted(p.glob("*.ds.metadata"))
        if not ds:
            errs.append(f"{p} contains no *.ds shards")
            continue
        if len(meta) != len(ds):
            errs.append(f"{p}: {len(ds)} *.ds but {len(meta)} *.ds.metadata — nanotron reads "
                        f"vocab_size from .ds.metadata and will refuse to start without it")
        total = 0
        for m in meta:
            try:
                with open(m) as fh:
                    fh.readline()                    # line 1: <tokenizer dir>|<token size>
                    total += int(fh.readline().strip())
            except Exception as e:
                errs.append(f"{m}: unreadable ({e!r})")
        exp = EXPECTED_CORPUS.get(corpus)
        if exp is None:
            errs.append(f"{corpus}: no expected token count on record — unknown corpus name")
            continue
        exp_tok, exp_shards = exp
        print(f"          {len(ds)} shards, {total:,} tokens (expected {exp_tok:,})")
        if len(ds) != exp_shards:
            errs.append(f"{corpus}: {len(ds)} shards, expected {exp_shards}")
        if total != exp_tok:
            errs.append(f"{corpus}: {total:,} tokens but this corpus should have "
                        f"{exp_tok:,} ({total - exp_tok:+,}). The path exists but does not hold "
                        f"the corpus it claims to — check data_root and the download.")
    return errs


def check_resume(cfg_path: Path):
    """Verify this run will actually resume from where it claims to.

    Run this immediately before launching each job. It is the only guard against the worst
    silent failure in the pipeline: `serialize/main.py:231` logs "No previous checkpoint
    found" at INFO level and returns None when `resume_checkpoint_path` does not resolve, so
    nanotron proceeds from RANDOM INIT. A cooldown branch would run its 476/954/1430 steps,
    exit 0, write a checkpoint, and be entirely meaningless.
    """
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    errs = []
    ck = cfg.get("checkpoints", {})
    resume = Path(ck.get("resume_checkpoint_path", ""))
    run = cfg.get("general", {}).get("run", "?")
    kind = run.rsplit("_", 1)[-1]
    print(f"resume  : {resume}")
    if not resume.is_dir():
        errs.append(f"resume_checkpoint_path {resume} does not exist. nanotron would log 'No "
                    f"previous checkpoint found' at INFO and START FROM RANDOM INIT.")
        return errs
    if kind.startswith("trunk"):
        # folder form: resolved through latest.txt
        latest = resume / "latest.txt"
        if not latest.is_file():
            errs.append(f"{latest} missing — the trunk directory must be pre-seeded with the "
                        f"init as step 0 and a latest.txt containing 0, or the first segment "
                        f"starts from random init.")
        else:
            step = latest.read_text().strip()
            tgt = resume / step
            print(f"          latest.txt -> step {step}")
            if not (tgt / "model_config.json").is_file():
                errs.append(f"latest.txt points at step {step} but {tgt}/model_config.json "
                            f"is missing")
    else:
        # direct step-dir form: must BE a checkpoint, and must be the right step
        want = str(cfg["optimizer"]["learning_rate_scheduler"]["lr_decay_starting_step"])
        if resume.name != want:
            errs.append(f"branch {kind} resumes from step dir {resume.name} but its "
                        f"lr_decay_starting_step is {want} — the branch point and the decay "
                        f"start must be the same step or the cooldown is not equivalent to an "
                        f"independent run")
        if not (resume / "model_config.json").is_file():
            errs.append(f"{resume}/model_config.json missing — parse_ckpt_path only treats a "
                        f"directory as a checkpoint if that file is present; without it this "
                        f"branch starts from random init")
        if (resume / "latest.txt").is_file():
            errs.append(f"{resume} contains latest.txt — it is being treated as a run folder, "
                        f"not a step directory; the branch would resume from the trunk's LATEST "
                        f"step instead of its own branch point")
    return errs


def check_log(path: Path, at_step: int, expect):
    mbs, accum, dp, seq = expect
    errs = []
    banner = None
    iter_tok = None
    for line in path.read_text(errors="ignore").splitlines():
        m = BANNER.search(line)
        if m and banner is None:
            banner = tuple(int(x) for x in m.groups())
        m = ITER.search(line)
        if m and int(m.group(1)) == at_step:
            iter_tok = float(m.group(2)) * SUFFIX[m.group(3)]
    if banner is None:
        errs.append(f"{path}: no 'mbs: ... | grad_accum: ...' banner found — did the run start?")
    else:
        b_mbs, b_accum, b_cp, b_seq, b_gbs = banner
        print(f"log     : mbs={b_mbs} grad_accum={b_accum} cp={b_cp} seq={b_seq} gbs={b_gbs}")
        if b_mbs != EXPECTED_MBS:
            errs.append(f"log reports micro_batch_size {b_mbs}, expected {EXPECTED_MBS}")
        if (b_mbs, b_accum, b_seq) != (mbs, accum, seq):
            errs.append(f"log banner {(b_mbs, b_accum, b_seq)} disagrees with config {(mbs, accum, seq)}")
        if b_gbs * b_seq != EXPECTED_TOK_PER_STEP:
            errs.append(f"log global_batch_size {b_gbs} x seq {b_seq} = {b_gbs*b_seq:,} tokens/step, "
                        f"expected {EXPECTED_TOK_PER_STEP:,}")
    if iter_tok is None:
        errs.append(f"{path}: no 'iteration: {at_step} / ...' line — run has not reached step {at_step}")
    else:
        want = at_step * EXPECTED_TOK_PER_STEP
        # nanotron prints consumed_tokens with 3 significant figures, so compare loosely
        if abs(iter_tok - want) / want > 0.01:
            errs.append(f"consumed_tokens at step {at_step} is {iter_tok:,.0f}, expected ~{want:,} "
                        f"({at_step} x {EXPECTED_TOK_PER_STEP:,})")
        else:
            print(f"          consumed_tokens@{at_step} = {iter_tok:,.0f} (expected ~{want:,}) OK")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="a RENDERED config (not a template)")
    ap.add_argument("--log", type=Path, help="training log, for the post-step-200 check")
    ap.add_argument("--at-step", type=int, default=200)
    ap.add_argument("--skip-corpus", action="store_true",
                    help="skip the dataset checks (use only when the corpora are not on this host)")
    ap.add_argument("--check-resume", action="store_true",
                    help="also verify resume_checkpoint_path resolves; run this immediately "
                         "before launching each job, once its predecessor has finished")
    ap.add_argument("--skip-env", action="store_true",
                    help="skip the wandb/environment preflight (use when rendering on a host "
                         "that will not run the training)")
    args = ap.parse_args()

    errs, expect = check_config(args.config)
    if not args.skip_env:
        errs += check_env(args.config)
    if not args.skip_corpus:
        errs += check_corpus(args.config)
    if args.check_resume:
        errs += check_resume(args.config)
    if args.log:
        errs += check_log(args.log, args.at_step, expect)
    if errs:
        fail(errs)
    extra = [] if args.skip_corpus else ["corpus verified by token count"]
    if args.check_resume:
        extra.append("resume path verified")
    print(f"OK: micro_batch_size={EXPECTED_MBS}, {EXPECTED_TOK_PER_STEP:,} tokens/step"
          + (", " + ", ".join(extra) if extra else ""))


if __name__ == "__main__":
    main()
