#!/usr/bin/env python
"""§5 CPU round trip of the HF export step the collaborator runs (tools/kys_eval/convert_to_hf.py).

Downloads the released v2 nanotron checkpoint wytro/KYS-1.5B-Rewritten-v2@b22c8733
rewrite-1p5b/seed42/diversity_oriented/ep3/nanotron/ (model weights + model_config.json only, no optimizer state)
and its released HF export .../ep3/hf/, converts the former with the huicong-dev converter and the KYS-Pre-Rewritten
tokenizer, then compares:
  - every tensor (names, shapes, dtypes, exact values) of model.safetensors
  - config.json (all keys), tokenizer.json / tokenizer_config.json sha256
  - kys_eval.common.validate_model_dir and weights_sha256 on the converted dir (what eval_checkpoint records)
Large downloads go to ../work/ (git-ignored by the audit commit; delete after). Output: ../evidence/convert_roundtrip.json
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
NANO = HERE.parents[2]
KYS = Path("/weka/scratch/jhu/bvandur1/zhuicon1/projects/kys-eval")
WORK = HERE.parent / "work"
EV = HERE.parent / "evidence"
V2, REV = "wytro/KYS-1.5B-Rewritten-v2", "b22c87333783945e1c655ae46886b7e842ee4203"
CELL = "rewrite-1p5b/seed42/diversity_oriented/ep3"
TOK = Path("/home/jhu/zhuicon1/projects_bvandur1/zhuicon1/data/1p5_baseline/pretraining/raw/tokenizer")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main():
    sys.path.insert(0, str(KYS))
    from kys_eval import common
    from safetensors import safe_open

    local = Path(snapshot_download(V2, revision=REV, local_dir=WORK / "v2",
                                   allow_patterns=[f"{CELL}/nanotron/model/*", f"{CELL}/nanotron/model/**",
                                                   f"{CELL}/nanotron/model_config.json", f"{CELL}/nanotron/config.yaml",
                                                   f"{CELL}/hf/*"]))
    step_dir, ref_hf, out = local / CELL / "nanotron", local / CELL / "hf", WORK / "converted_hf"
    r = subprocess.run([sys.executable, NANO / "tools/kys_eval/convert_to_hf.py", "--checkpoint_path", step_dir,
                        "--save_path", out, "--tokenizer", TOK], capture_output=True, text=True)
    res = {"convert_rc": r.returncode, "convert_tail": (r.stdout + r.stderr)[-1500:]}
    if r.returncode:
        print(res["convert_tail"]); (EV / "convert_roundtrip.json").write_text(json.dumps(res, indent=1)); return 1
    a, b = safe_open(out / "model.safetensors", "pt"), safe_open(ref_hf / "model.safetensors", "pt")
    ka, kb = set(a.keys()), set(b.keys())
    diffs = []
    for k in sorted(ka & kb):
        x, y = a.get_tensor(k), b.get_tensor(k)
        if x.shape != y.shape or x.dtype != y.dtype or not bool((x == y).all()):
            diffs.append(k)
    res["tensors"] = {"converted": len(ka), "released": len(kb), "only_converted": sorted(ka - kb),
                      "only_released": sorted(kb - ka), "value_or_shape_diffs": diffs}
    ca, cb = json.loads((out / "config.json").read_text()), json.loads((ref_hf / "config.json").read_text())
    res["config_json_diff"] = {k: (ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)}
    res["files"] = {"converted": sorted(p.name for p in out.iterdir()), "released": sorted(p.name for p in ref_hf.iterdir())}
    res["tokenizer_sha_equal"] = {n: sha(out / n) == sha(ref_hf / n) for n in ("tokenizer.json", "tokenizer_config.json")
                                  if (out / n).exists() and (ref_hf / n).exists()}
    res["model_safetensors_sha256"] = {"converted": sha(out / "model.safetensors"), "released": sha(ref_hf / "model.safetensors")}
    problems, warnings = common.validate_model_dir(out)
    res["kys_validate_model_dir"] = {"problems": problems, "warnings": warnings}
    print(json.dumps(res, indent=1)[:4000])
    (EV / "convert_roundtrip.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
