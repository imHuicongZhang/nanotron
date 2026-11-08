#!/usr/bin/env python
"""Pick the config a trunk segment must actually run with, given its trunk directory's latest.txt.

    python tools/kys_raw/trunk_restart_config.py <rendered config.yaml>    # prints the config path to run

Why (prelaunch audit 2026-11-07, finding B1 / T-F1). trunk1 is rendered with
`load_optimizer: false` and `load_lr_scheduler: false`, because it starts from the init checkpoint
(step 0), whose optimizer file holds no Adam state and which nanotron refuses to load
(tools/generate_configs.py). If trunk1 is restarted after its step-1500 or 3000 save, latest.txt
points at that step: nanotron then loads the trained *weights* but, with load_lr_scheduler false,
builds fresh metadata (step 0, consumed samples 0; src/nanotron/trainer.py, "Define iteration start
state"). Warmup, AdamW state and the data stream would restart from sample 0 on trained weights,
and the run would still end with a normal 4292 checkpoint and exit 0.

Rule:
  * latest.txt = 0, or the config already loads optimizer and LR state -> print the config unchanged;
  * latest.txt > 0 and either flag is false -> write a copy whose ONLY change is
    load_optimizer: true and load_lr_scheduler: true, and print its path. Optimizer, LR scheduler and
    training metadata (step, consumed samples, hence the data position) are then restored from that
    checkpoint, exactly as trunk2 restores them from trunk1's 4292 save.
The copy goes to `<trunk dir>.restart/<run>_from<step>.yaml`, a sibling of the trunk directory, so it
neither lands among the rendered configs nor among the step directories.

Anything unexpected (not a trunk, latest.txt missing or not an integer, no checkpoint at that step)
exits non-zero; `assert_invariants.py --check-resume` reports the details and also refuses a trunk config
that would restart without its state.
"""
import sys
from pathlib import Path

import yaml

FLAGS = ("load_optimizer", "load_lr_scheduler")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.splitlines()[2].strip(), file=sys.stderr)
        return 2
    cfg_path = Path(sys.argv[1])
    cfg = yaml.safe_load(cfg_path.read_text())
    run = cfg["general"]["run"]
    kind = run.rsplit("_", 1)[-1]
    ck = cfg["checkpoints"]
    if not kind.startswith("trunk") or all(ck.get(f) for f in FLAGS):
        print(cfg_path)
        return 0

    trunk_dir = Path(ck["resume_checkpoint_path"])
    latest = trunk_dir / "latest.txt"
    if not latest.is_file():
        print(f"[restart-config] {latest} missing", file=sys.stderr)
        return 1
    step_s = latest.read_text().strip()
    if not step_s.isdigit():
        print(f"[restart-config] {latest} holds {step_s!r}, not a step number", file=sys.stderr)
        return 1
    step = int(step_s)
    if step == 0:
        print(cfg_path)
        return 0
    if not (trunk_dir / step_s / "model_config.json").is_file():
        print(f"[restart-config] latest.txt -> {step}, but {trunk_dir / step_s}/model_config.json is missing",
              file=sys.stderr)
        return 1
    if step >= cfg["tokens"]["train_steps"]:
        print(f"[restart-config] latest.txt -> {step} >= train_steps {cfg['tokens']['train_steps']}",
              file=sys.stderr)
        return 1

    new = yaml.safe_load(cfg_path.read_text())
    for f in FLAGS:
        new["checkpoints"][f] = True
    # The copy must differ from the rendered config in exactly these two keys.
    def flat(d, p=""):
        out = {}
        for k, v in d.items():
            key = f"{p}.{k}" if p else str(k)
            out.update(flat(v, key) if isinstance(v, dict) else {key: v})
        return out
    a, b = flat(cfg), flat(new)
    diff = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    assert set(diff) <= {f"checkpoints.{f}" for f in FLAGS}, diff

    out_dir = trunk_dir.parent / f"{trunk_dir.name}.restart"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run}_from{step}.yaml"
    header = (f"# Restart config written by tools/kys_raw/trunk_restart_config.py from {cfg_path.resolve()}\n"
              f"# latest.txt = {step}: identical except {', '.join(f + ': true' for f in FLAGS)} (audit B1).\n")
    out.write_text(header + yaml.safe_dump(new, sort_keys=False))
    print(f"[restart-config] {run}: latest.txt = {step} > 0 -> loading optimizer, LR and metadata; "
          f"running {out}", file=sys.stderr)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
