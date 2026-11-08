#!/usr/bin/env python
"""§3 protocol identity: the three rebuttal settings' ep1/ep2/ep3 configs vs the released v2 nanotron configs.

For seed S, epoch N, new setting X with comparator W (raw_diversity_oriented -> diversity_oriented,
raw_disagreement_aware and signal-disagreement-lambda0 -> disagreement_aware; plus quality_base for every X), flatten
  configs/1.5B-baseline-seed<S>/<X>_seed<S>_ep<N>.yaml      (committed placeholder config, huicong-dev)
  wytro/KYS-1.5B-Rewritten-v2@b22c8733 rewrite-1p5b/seed<S>/<W>/ep<N>/nanotron/config.yaml
and list every key that differs or exists on one side only. Also records the v2 checkpoint_metadata.json
(step and consumed samples at the final). Output: ../evidence/protocol_identity.{json,md}
"""
import json
import sys
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EV = HERE.parent / "evidence"
V2 = "wytro/KYS-1.5B-Rewritten-v2"
REV = "b22c87333783945e1c655ae46886b7e842ee4203"
PAIRS = {"raw_diversity_oriented": "diversity_oriented", "raw_disagreement_aware": "disagreement_aware",
         "signal-disagreement-lambda0": "disagreement_aware"}
RUN_FIELDS = {"general.run", "checkpoints.checkpoints_path", "checkpoints.resume_checkpoint_path",
              "data_stages.0.data.dataset.dataset_folder", "tokenizer.tokenizer_name_or_path"}


def flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{pre}{k}."))
    elif isinstance(d, list) and d and isinstance(d[0], dict):
        for i, v in enumerate(d):
            out.update(flat(v, f"{pre}{i}."))
    else:
        out[pre[:-1]] = d
    return out


def main():
    res, md = {}, ["| seed | ep | new setting | v2 config | keys in new | equal | differing (non-run) | only in v2 (count) |",
                   "|---|---|---|---|---:|---:|---|---:|"]
    all_diff_keys = {}
    for s in (42, 43, 44):
        for n in (1, 2, 3):
            v2 = {}
            for w in sorted(set(PAIRS.values()) | {"quality_base"}):
                p = hf_hub_download(V2, f"rewrite-1p5b/seed{s}/{w}/ep{n}/nanotron/config.yaml", revision=REV,
                                    local_dir=EV / "v2_configs")
                m = hf_hub_download(V2, f"rewrite-1p5b/seed{s}/{w}/ep{n}/nanotron/checkpoint_metadata.json", revision=REV,
                                    local_dir=EV / "v2_configs")
                v2[w] = (flat(yaml.safe_load(open(p))), json.load(open(m)))
            for x, w in PAIRS.items():
                new = flat(yaml.safe_load(open(REPO / f"configs/1.5B-baseline-seed{s}/{x}_seed{s}_ep{n}.yaml")))
                for ref in (w, "quality_base"):
                    ref_flat, meta = v2[ref]
                    diff = {k: (new[k], ref_flat.get(k, "<absent>")) for k in new
                            if k not in RUN_FIELDS and new[k] != ref_flat.get(k, "<absent>")}
                    only_v2 = sorted(set(ref_flat) - set(new))
                    eq = sum(1 for k in new if new[k] == ref_flat.get(k, object()))
                    res[f"{s}/ep{n}/{x}/vs/{ref}"] = {"diff": diff, "only_in_v2": only_v2, "equal": eq, "n_new": len(new),
                                                     "run_fields": {k: (new.get(k), ref_flat.get(k)) for k in RUN_FIELDS},
                                                     "v2_checkpoint_metadata": meta}
                    for k in diff:
                        all_diff_keys.setdefault(k, set()).add(json.dumps(diff[k], default=str))
                    md.append(f"| {s} | ep{n} | `{x}` | `{ref}` | {len(new)} | {eq} | "
                              + (", ".join(f"`{k}` {a!r} vs {b!r}" for k, (a, b) in diff.items()) or "none")
                              + f" | {len(only_v2)} |")
    summary = {k: sorted(v) for k, v in all_diff_keys.items()}
    (EV / "protocol_identity.json").write_text(json.dumps({"rows": res, "differing_keys": summary}, indent=1, default=str))
    (EV / "protocol_identity.md").write_text("\n".join(md) + "\n\nDiffering keys across all rows:\n\n"
                                             + "\n".join(f"- `{k}`: {v}" for k, v in summary.items()) + "\n")
    print("\n".join(md))
    print("differing keys:", json.dumps(summary, indent=1))
    only = sorted({k for r in res.values() for k in r["only_in_v2"]})
    print("keys only in v2 dumps:", len(only), only[:60])
    metas = {k.split("/vs/")[1] + " " + k.split("/")[0] + " " + k.split("/")[1]: r["v2_checkpoint_metadata"] for k, r in res.items()}
    print("v2 checkpoint_metadata examples:", json.dumps(dict(list(metas.items())[:3]), default=str)[:800])


if __name__ == "__main__":
    sys.exit(main())
