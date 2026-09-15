#!/usr/bin/env python
"""Fail fast if the batch invariants have drifted.

Two invariants hold across all 72 runs of the grid, and both must be true or the runs are
not comparable to each other:

    micro_batch_size == 32                  (pinned for NUMERICAL comparability, not speed;
                                             the value in 53 of the 54 released checkpoints)
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

# What the grid actually ran with, from the config.yaml of all 54 released checkpoints
# (2026-09-15): mbs 32 in 53 of 54 (seed42/quality_base ran mbs 16). The 16 previously pinned
# here was never the grid's value. See deploy/clusters.yaml `kys_grid_1p5b`.
EXPECTED_MBS = 32
# Raw-selected baselines (configs/1.5B-baseline-seed*) use the grid's mbs too; on 80 GB cards that
# needs full layer recomputation (probe job 424229, configs/1.5B-baseline/README.md). A cluster that
# cannot fit mbs 32 passes --expected-mbs N and must report the deviation (RUNBOOK.md).
RAW_SETTINGS = {'raw_diversity_oriented', 'raw_disagreement_aware', 'raw_random', 'raw_rewire_inspired'}
EXPECTED_MBS_RAW = 32
EXPECTED_TOK_PER_STEP = 2_097_152
MBS_OVERRIDE = None     # set by --expected-mbs

# Site-specific values in the shipped configs are {{NAME}} markers (tools/kys_raw/render_placeholders.py).
PLACEHOLDER = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")


def expected_mbs(cfg):
    if MBS_OVERRIDE is not None:
        return MBS_OVERRIDE
    setting = cfg.get("general", {}).get("run", "").rsplit("_", 2)[0]
    return EXPECTED_MBS_RAW if setting in RAW_SETTINGS else EXPECTED_MBS


def check_placeholders(path: Path):
    """Refuse any config (or its companion .env) that still carries an unfilled {{MARKER}}."""
    errs = []
    for p in (path, path.with_suffix(".env")):
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            for m in PLACEHOLDER.findall(line):
                errs.append(f"{p.name}:{i} still contains placeholder {m}")
    print(f"placeholders: {'none' if not errs else f'{len(errs)} remaining'}")
    return errs

# Corpus identity, keyed by the corpus subdirectory name — the unified names, which are what
# `data_root` holds when it is a snapshot of wytro/Know-Your-Sources-tokenized. See the
# legacy-name remap table in tools/render_config.py if you are looking at an old /scratch tree.
# Token counts are the sum of line 2 of every *.ds.metadata in the folder, cross-checked
# against raw .ds bytes/2 (they matched exactly for all six on 2026-08-09). The counts are
# properties of the data and did not change when the directories were renamed.
# Source: tmp/kys/manifest/provenance.tsv, Gate 4.
#
# This is the check that catches a path which EXISTS but holds the wrong corpus. That failure
# is otherwise silent: nanotron starts, trains for 27 hours, and produces numbers that mean
# nothing. Note diversity_oriented is legitimately ~1.1% short of 10B — that is a property of
# the corpus, not an error.
#
# Any corpus directory not listed here is REJECTED as unknown. That is deliberate: it is how
# the retired pre-shuffle `10B-base` sawtooth corpus (and any other stray tree) is refused
# rather than silently accepted.
EXPECTED_CORPUS = {
    'quality_base':       (10_000_003_137, 16),
    'quality_first':      (10_000_002_634, 16),
    'diversity_oriented': ( 9_889_637_833, 16),
    'wrap_inspired':      (10_000_002_419, 16),
    'rewire_inspired':    (10_000_002_683, 16),
    'disagreement_aware': (10_000_002_333, 16),
    # Raw-selected baselines, built locally. None until tokenized and counted: refuses to pass.
    'raw_diversity_oriented': None,
    'raw_disagreement_aware': None,
    'raw_random':             None,
    'raw_rewire_inspired':    None,
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
    want_mbs = expected_mbs(cfg)
    if want_mbs is None:
        errs.append("micro_batch_size for the raw baselines has not been chosen yet "
                    "(EXPECTED_MBS_RAW is None)")
    elif mbs != want_mbs:
        errs.append(f"micro_batch_size is {mbs}, expected {want_mbs}")
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

    Five ways this goes wrong, all caught here rather than 27 hours in:
      - data_root typo               -> folder missing
      - corpus not downloaded yet    -> folder present but no .ds
      - .ds.metadata not shipped     -> nanotron's own vocab_size assert would fire at startup
      - RIGHT path, WRONG corpus     -> token count disagrees. This is the silent one.
      - metadata names a foreign     -> nanotron's config.py:521 assert would fire after SLURM
        tokenizer path                  has allocated. See check_tokenizer_metadata().
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
    cfg_tok = (cfg.get("tokenizer") or {}).get("tokenizer_name_or_path")
    tok_by_folder = {}          # folder -> {tokenizer string: [shard filenames]}
    for f in folders:
        p = Path(f)
        corpus = p.parent.name                      # <data_root>/<corpus>/tokenized
        print(f"corpus  : {corpus}  ({p})")
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
        seen = tok_by_folder.setdefault(str(p), {})
        for m in meta:
            try:
                with open(m) as fh:
                    line1 = fh.readline().strip()    # line 1: <tokenizer dir>|<token size>
                    total += int(fh.readline().strip())
            except Exception as e:
                errs.append(f"{m}: unreadable ({e!r})")
                continue
            # Line 1 is what nanotron feeds to AutoTokenizer (config.py:192) and asserts
            # against tokenizer_name_or_path (config.py:521). Split on the LAST '|' so a
            # tokenizer path containing '|' survives — same rule fix_ds_metadata.py uses.
            if "|" not in line1:
                errs.append(f"{m}: metadata line 1 has no '|' separator: {line1!r}")
                continue
            seen.setdefault(line1.rsplit("|", 1)[0], []).append(m.name)
        if corpus not in EXPECTED_CORPUS:
            errs.append(f"{corpus}: no expected token count on record — unknown corpus name")
            continue
        exp = EXPECTED_CORPUS[corpus]
        if exp is None:
            # Raw-selected corpora are tokenized by the consumer from blab-jhu/KYS-Pre-Rewritten; the
            # expected total ships in that repo's manifest.json, at <data_root>/manifest.json.
            man = p.parent.parent / "manifest.json"
            try:
                rec = __import__("json").loads(man.read_text())["settings"][corpus]
                exp = (int(rec["expected_total_tokens"]), int(rec["shards_after_tokenization"]))
                print(f"          expected token total from {man}")
            except Exception:
                errs.append(f"{corpus}: no recorded token count — expected {man} (blab-jhu/KYS-Pre-Rewritten "
                            f"manifest.json) with settings.{corpus}.expected_total_tokens")
                continue
        exp_tok, exp_shards = exp
        print(f"          {len(ds)} shards, {total:,} tokens (expected {exp_tok:,})")
        if len(ds) != exp_shards:
            errs.append(f"{corpus}: {len(ds)} shards, expected {exp_shards}")
        if total != exp_tok:
            errs.append(f"{corpus}: {total:,} tokens but this corpus should have "
                        f"{exp_tok:,} ({total - exp_tok:+,}). The path exists but does not hold "
                        f"the corpus it claims to — check data_root and the download.")
    errs += check_tokenizer_metadata(tok_by_folder, cfg_tok)
    return errs


def check_tokenizer_metadata(tok_by_folder, cfg_tok):
    """Replicate nanotron's tokenizer-vs-metadata asserts, at preflight instead of at torchrun.

    Line 1 of every *.ds.metadata is `<tokenizer path>|<token size>`. nanotron:
      - config.py:192 feeds that path straight to AutoTokenizer.from_pretrained to derive
        vocab_size, so it must resolve ON THIS HOST;
      - config.py:194-199 asserts it is identical across every metadata file in every
        dataset folder;
      - config.py:521 asserts it equals tokenizer.tokenizer_name_or_path EXACTLY.

    A freshly downloaded corpus fails all three: the published metadata still names the
    absolute path the corpus was tokenized under. Without this check that surfaces as an
    AssertionError inside config parsing, after SLURM has allocated. tools/fix_ds_metadata.py
    rewrites line 1 in place and is idempotent.
    """
    errs = []
    if not tok_by_folder:
        return errs
    distinct = sorted({t for seen in tok_by_folder.values() for t in seen})
    if not distinct:
        return errs

    # (1) shards must agree with each other — report the split, never average it away.
    if len(distinct) > 1:
        lines = []
        for folder, seen in sorted(tok_by_folder.items()):
            for tok, shards in sorted(seen.items()):
                shown = ", ".join(shards[:3]) + (f", +{len(shards) - 3} more" if len(shards) > 3 else "")
                lines.append(f"    {tok!r}  <- {folder}: {shown}")
        errs.append("*.ds.metadata files disagree about the tokenizer path; nanotron asserts "
                    "they are identical across all dataset folders (config.py:194-199):\n"
                    + "\n".join(lines)
                    + "\n  Re-run tools/fix_ds_metadata.py over EVERY corpus folder with the same "
                      "--tokenizer-dir.")
        return errs

    meta_tok = distinct[0]
    print(f"tokenizer: {meta_tok}  (from .ds.metadata line 1)")

    # (2) it has to resolve on this host — config.py:192 loads it, and a non-local string
    #     sends AutoTokenizer to the Hub, which compute nodes may not reach.
    if not Path(meta_tok).is_dir():
        errs.append(f"the tokenizer path recorded in *.ds.metadata does not exist on this host: "
                    f"{meta_tok}\n  config.py:192 calls AutoTokenizer.from_pretrained() on that "
                    f"string; a non-local value falls through to the HuggingFace Hub.")

    # (3) and it has to equal the config's tokenizer_name_or_path — config.py:521.
    if cfg_tok is None:
        errs.append("rendered config sets no tokenizer.tokenizer_name_or_path, so nanotron will "
                    f"adopt the metadata value ({meta_tok}) verbatim. Set tokenizer_path in "
                    "deploy/clusters.yaml and re-render.")
    elif str(cfg_tok) != meta_tok:
        cmd = "\n".join(
            f"    python tools/fix_ds_metadata.py --output-folder {folder} --tokenizer-dir {cfg_tok}"
            for folder in sorted(tok_by_folder)
        )
        errs.append(
            "tokenizer path mismatch — nanotron will refuse to start (config.py:521):\n"
            f"    config tokenizer_name_or_path : {cfg_tok}\n"
            f"    *.ds.metadata line 1          : {meta_tok}\n"
            "  The published corpora record the path they were tokenized under, which is not "
            "yours.\n  Fix (idempotent, no network, run once per corpus folder):\n" + cmd
            + "\n  Pass the SAME --tokenizer-dir to every corpus folder. See HANDOVER.md §9.2."
        )
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


def check_log(path: Path, at_step: int, expect, want_mbs):
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
        if b_mbs != want_mbs:
            errs.append(f"log reports micro_batch_size {b_mbs}, expected {want_mbs}")
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
    ap.add_argument("--expected-mbs", type=int,
                    help="accept this micro_batch_size instead of the grid's 32 — only for a cluster "
                         "that cannot fit 32; the deviation changes masked_mean weighting and must "
                         "be reported")
    args = ap.parse_args()

    placeholder_errs = check_placeholders(args.config)
    if placeholder_errs:
        fail(placeholder_errs + ["fill them with tools/kys_raw/fill_placeholders.py (RUNBOOK.md) before running"])
    global MBS_OVERRIDE
    if args.expected_mbs is not None:
        MBS_OVERRIDE = args.expected_mbs
        if args.expected_mbs != EXPECTED_MBS:
            print(f"WARNING : --expected-mbs {args.expected_mbs} deviates from the grid's micro_batch_size "
                  f"{EXPECTED_MBS}; this changes masked_mean per-token weighting (SOP.md §1). Report it.")

    errs, expect = check_config(args.config)
    if not args.skip_env:
        errs += check_env(args.config)
    if not args.skip_corpus:
        errs += check_corpus(args.config)
    if args.check_resume:
        errs += check_resume(args.config)
    want_mbs = expected_mbs(yaml.safe_load(args.config.read_text()))
    if args.log:
        errs += check_log(args.log, args.at_step, expect, want_mbs)
    if errs:
        fail(errs)
    extra = [] if args.skip_corpus else ["corpus verified by token count", "tokenizer path matches metadata"]
    if args.check_resume:
        extra.append("resume path verified")
    print(f"OK: micro_batch_size={want_mbs}, {EXPECTED_TOK_PER_STEP:,} tokens/step"
          + (", " + ", ".join(extra) if extra else ""))


if __name__ == "__main__":
    main()
