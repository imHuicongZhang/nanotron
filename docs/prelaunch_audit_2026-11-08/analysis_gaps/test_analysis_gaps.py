#!/usr/bin/env python
"""Identity test of the two analysis changes against the 54 v2 LightEval files (CPU, no network).

The 27 rebuttal cells are filled with the v2 results of their primary comparator (raw_diversity_oriented <-
diversity_oriented, raw_disagreement_aware and signal-disagreement-lambda0 <- disagreement_aware), packaged into
kys-eval results.json files with kys-eval's own code (eval_checkpoint.package_existing -> common.build_results, the
function evaluate() uses after LightEval). Then:

  A. paired_analysis.py --kys-results <tree in the rewrite-1p5b/seed<S>/<setting>/ep<N>/hf layout>
     A1 every rebuttal cell's 12 scores == its comparator's, bit for bit (module-level check)
     A2 rows of the 15 v2 pairs in every CSV, and ranks / Step 1 files, are identical to the v2-only run
     A3 new - primary comparator rows: mean, SD and the three per-seed differences are all 0.00, 0/3 seeds
     A4 new - quality_base rows equal the v2 rows comparator - quality_base field for field
     A5 tables.tex holds the 6 added pairs in a separate block
  B. the same with kys-eval's native results/raw_selected/... tree (model dict as evaluate() writes it): same outputs
  C. the loader refuses: a missing cell, a wrong step, a file under another cell's path, a duplicate, an incomplete file
  D. kys-eval aggregate.py on the native tree: primary Δ == 0 exactly with 0 positive seeds; quality_base Δ (setting −
     quality_base) == paired_analysis's comparator − quality_base computed from the same data (mean, SD, seed count)

Usage (kys-eval venv):  python test_analysis_gaps.py
"""
import csv
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KYS = Path("/weka/scratch/jhu/bvandur1/zhuicon1/projects/kys-eval")
PA_DIR = Path("/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite/docs/rebuttal_q1_paired_analysis")
sys.path.insert(0, str(KYS))
from kys_eval import aggregate, common  # noqa: E402
from kys_eval import config as C  # noqa: E402
from kys_eval.eval_checkpoint import package_existing  # noqa: E402

spec = importlib.util.spec_from_file_location("paired_analysis", PA_DIR / "paired_analysis.py")
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)

W = HERE / "work"
NEW = {"raw_diversity_oriented": "diversity_oriented", "raw_disagreement_aware": "disagreement_aware",
       "signal-disagreement-lambda0": "disagreement_aware"}
EP = {1: "ep1", 2: "ep2", 3: "ep3"}
results, ok_all = {}, True


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    results[name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def build(layout, full_model):
    """Write the 27 cells; layout 'hf' -> <root>/rewrite-1p5b/seed<S>/<s>/ep<N>/hf/results.json,
    'native' -> <root>/raw_selected/seed<S>/<s>/ep<N>/results.json (kys-eval run_grid layout)."""
    root = W / f"tree_{layout}"
    shutil.rmtree(root, ignore_errors=True)
    v2 = {(r["setting"], int(r["seed"]), int(r["epoch"])): r for r in csv.DictReader(open(PA_DIR / "result_files_used.csv"))}
    for s, comp in NEW.items():
        for seed in C.SEEDS:
            for ep in (1, 2, 3):
                src = v2[(comp, seed, ep)]
                sub = C.hf_subfolder(seed, s, EP[ep])
                cell = {"setting": s, "seed": seed, "epoch": EP[ep], "step": C.EPOCH_STEPS[EP[ep]],
                        "hf_subfolder": sub, "hf_repo": C.HF_REPO}
                d = root / (sub if layout == "hf" else f"raw_selected/seed{seed}/{s}/{EP[ep]}")
                d.mkdir(parents=True, exist_ok=True)
                res = package_existing(src["result_json"], d, label=f"identity {s} {seed} {EP[ep]}",
                                       model=f"hf://{C.HF_REPO}/{sub}", gpu=src["gpu"], cell=cell)
                if full_model:  # the provenance dict evaluate() writes
                    res["model"] = {"spec": f"hf://{C.HF_REPO}/{sub}", **common.parse_model_spec(f"hf://{C.HF_REPO}/{sub}"),
                                    "commit": "0" * 40, "local_path": f"/model_cache/{C.HF_REPO}/{sub}",
                                    "weights_sha256": {"model.safetensors": "test"}}
                    common.write_json(d / "results.json", res)
    return root


def run_pa(out, kys=None):
    cmd = [sys.executable, str(PA_DIR / "paired_analysis.py"), "--out-dir", str(out)] + (["--kys-results", str(kys)] if kys else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out / "run_output.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "run_output.txt").write_text(r.stdout + r.stderr)
    return r.returncode, r.stdout + r.stderr


def rows(path):
    return list(csv.DictReader(open(path)))


def main():
    shutil.rmtree(W, ignore_errors=True)
    W.mkdir(parents=True)
    base_out = W / "out_v2_only"
    rc, _ = run_pa(base_out)
    check("0 v2-only run", rc == 0)

    tree_hf = build("hf", full_model=False)
    tree_nat = build("native", full_model=True)

    # A1 bitwise scores
    v2data, _, miss = pa.load()
    kdata, kfiles, kprob = pa.load_kys(str(tree_hf))
    bit = all(kdata[(s, seed, ep)] == v2data[(NEW[s], seed, ep)] for s in NEW for seed in C.SEEDS for ep in (1, 2, 3))
    check("A1 rebuttal cell scores == comparator scores, bit for bit (27 x 12)", not miss and not kprob and len(kfiles) == 27 and bit,
          f"{len(kfiles)} files, problems {len(kprob)}")

    out_hf = W / "out_with_kys_hf"
    rc, log = run_pa(out_hf, tree_hf)
    check("A0 run with --kys-results (hf layout)", rc == 0, log.splitlines()[0] if log else "")
    v2keys = set(pa.KEYS)
    same_v2 = True
    for name in ("pairwise_mean6_mmlu.csv", "pairwise_per_task_ep3.csv", "loto_mean6.csv"):
        a = [r for r in rows(base_out / name)]
        b = [r for r in rows(out_hf / name) if r["setting_a"] in v2keys and r["setting_b"] in v2keys]
        same_v2 &= a == b
    for name in ("ranks_per_seed.csv", "step1_verification.csv", "step1_verification.md"):
        same_v2 &= (base_out / name).read_bytes() == (out_hf / name).read_bytes()
    pcs_a = rows(base_out / "per_checkpoint_scores.csv")
    pcs_b = rows(out_hf / "per_checkpoint_scores.csv")
    same_v2 &= pcs_a == [r for r in pcs_b if r["setting"] in v2keys]
    new_pcs = {(r["setting"], r["seed"], r["epoch"]): r for r in pcs_b if r["setting"] in NEW}
    by = {(r["setting"], r["seed"], r["epoch"]): r for r in pcs_a}
    cols = pa.TASK_COLS + ["mean6", "mmlu"] + pa.CAT_COLS
    pcs_eq = len(new_pcs) == 27 and all(all(new_pcs[k][c] == by[(NEW[k[0]], k[1], k[2])][c] for c in cols) for k in new_pcs)
    check("A2 v2-pair rows, ranks and Step 1 unchanged; new per-checkpoint rows == comparator rows", same_v2 and pcs_eq)

    zero_ok, qb_ok, n_new = True, True, 0
    for name in ("pairwise_mean6_mmlu.csv", "pairwise_per_task_ep3.csv", "loto_mean6.csv"):
        all_rows = rows(out_hf / name)
        v2rows = {(r["setting_a"], r["setting_b"], r["epoch"], r.get("metric") or r.get("dropped_task")): r
                  for r in all_rows if r["setting_a"] in v2keys}
        for r in all_rows:
            if r["setting_a"] not in NEW:
                continue
            n_new += 1
            tag = r.get("metric") or r.get("dropped_task")
            vals = [r[c] for c in r if c.startswith("diff_seed")] + [r.get("mean_diff_a_minus_b") or r.get("mean_diff"), r["sd_diff"]]
            if r["setting_b"] == NEW[r["setting_a"]]:
                zero_ok &= all(v in ("0.00", "-0.00") for v in vals) and r["seeds_a_gt_b"] == "0"
            else:
                ref = v2rows[(NEW[r["setting_a"]], "quality_base", r["epoch"], tag)]
                drop = {"setting_a", "label_a", "requested_pair", "criterion_winner"}
                qb_ok &= {k: v for k, v in r.items() if k not in drop} == {k: v for k, v in ref.items() if k not in drop}
                if "criterion_winner" in r:
                    qb_ok &= (r["criterion_winner"] or "") == ("" if not ref["criterion_winner"] else
                                                               (r["setting_a"] if ref["criterion_winner"] == ref["setting_a"] else ref["criterion_winner"]))
    check("A3 new − primary comparator: all 0.00, 0/3 seeds", zero_ok, f"{n_new} rebuttal rows in the three CSVs")
    check("A4 new − quality_base rows == v2 comparator − quality_base rows", qb_ok)
    tex = (out_hf / "tables.tex").read_text()
    labels = {"raw_diversity_oriented": "Raw Diversity Oriented", "raw_disagreement_aware": "Raw Disagreement Aware",
              "signal-disagreement-lambda0": "Disagreement Aware ($\\lambda=0$)"}
    n_tex = sum(tex.count(f"\\textsc{{{labels[k]}}} $-$") for k in NEW)
    check("A5 tables.tex: 6 added pairs in each of the two pairwise tables", n_tex == 12 and "in the last block" in tex, f"{n_tex} rows")

    # B native layout
    out_nat = W / "out_with_kys_native"
    rc, _ = run_pa(out_nat, tree_nat)
    same = rc == 0 and all((out_hf / n).read_bytes() == (out_nat / n).read_bytes() for n in
                           ("pairwise_mean6_mmlu.csv", "pairwise_per_task_ep3.csv", "loto_mean6.csv", "per_checkpoint_scores.csv",
                            "tables.tex", "summary_supported_3of3_mean_gt_sd.csv", "summary_consistent_3of3_mean_le_sd.csv"))
    check("B native kys-eval tree gives the same outputs", same)

    # C negative tests: each mutation must make the run exit 1 with the expected reason
    one = "rewrite-1p5b/seed43/raw_disagreement_aware/ep2/hf/results.json"

    def edit(f):
        def m(t):
            p = t / one
            d = json.loads(p.read_text())
            f(d)
            p.write_text(json.dumps(d))
        return m

    def copy_to(rel):
        def m(t):
            (t / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(t / one, t / rel)
        return m

    tests = {
        "missing_cell": (lambda t: (t / one).unlink(), "no valid results.json"),
        "wrong_step": (edit(lambda d: d["cell"].update(step=9536)), "cell step 9536"),
        "wrong_model_subfolder": (edit(lambda d: d["model"].update(
            spec=f"hf://{C.HF_REPO}/rewrite-1p5b/seed43/raw_disagreement_aware/ep1/hf")), "model subfolder"),
        "incomplete": (edit(lambda d: d.update(complete=False, missing_tasks=["rw_piqa"])), "incomplete"),
        "misplaced_path": (copy_to("rewrite-1p5b/seed44/raw_disagreement_aware/ep2/hf/extra/results.json"), "file path names"),
        "duplicate": (copy_to("copies/results.json"), "duplicate of"),
    }
    for name, (mutate, reason) in tests.items():
        t = W / f"neg_{name}"
        shutil.rmtree(t, ignore_errors=True)
        shutil.copytree(tree_hf, t)
        mutate(t)
        rc, log = run_pa(W / f"out_neg_{name}", t)
        hit = [l.strip() for l in log.splitlines() if reason in l]
        check(f"C refuses {name}", rc == 1 and "MISSING / INVALID kys-eval" in log and hit, (hit or ["(reason not found)"])[0][:200])

    # D kys-eval aggregate on the native tree
    rep = W / "kys_reports"
    aggregate.main(["--results-root", str(tree_nat), "--out-dir", str(rep)])
    agg = rows(rep / "raw_vs_rewritten.csv")
    d_ok, n_d, tie_noise = True, 0, []
    for r in agg:
        if r["raw_setting"] not in NEW:
            continue
        n_d += 1
        ep = int(r["epoch"][2:])
        b = r["benchmark"]
        comp = r["comparator_setting"]
        if r["comparison"] == "primary":
            # aggregate.py reads the comparator from reference/rewritten_v2_54_per_task.csv (10 decimals) and the new
            # cell from the full-precision JSON, so an exact tie shows as |Δ| ~ 1e-11; n_delta_positive is then noise
            d_ok &= comp == NEW[r["raw_setting"]] and abs(float(r["delta_mean"])) < 1e-9 and abs(float(r["delta_std"])) < 1e-9 \
                and r["n_paired"] == "3"
            tie_noise.append(int(r["n_delta_positive"]))
        else:
            d = [v2data[(NEW[r["raw_setting"]], s, ep)][b] - v2data[("quality_base", s, ep)][b] for s in C.SEEDS]
            d_ok &= comp == "quality_base" and abs(float(r["delta_mean"]) * 100 - statistics.fmean(d)) < 1e-4 \
                and abs(float(r["delta_std"]) * 100 - statistics.stdev(d)) < 1e-4 and int(r["n_delta_positive"]) == sum(x > 0 for x in d)
    check("D kys-eval aggregate: primary Δ = 0; quality_base Δ = setting − QB = paired_analysis comparator − QB", d_ok and n_d == 3 * 2 * 3 * 12,
          f"{n_d} rows (3 settings x 2 comparators x 3 epochs x 12 benchmarks); primary rows with a 'positive' seed from "
          f"CSV rounding: {sum(1 for x in tie_noise if x)} of {len(tie_noise)}")
    md = (rep / "raw_vs_rewritten.md").read_text()
    check("D2 report states the Appendix F orientation", "Δ = setting − comparator" in md and "Also paired with" in md)
    (HERE / "test_results.json").write_text(json.dumps(results, indent=1))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
