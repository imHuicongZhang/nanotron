#!/usr/bin/env python
"""§4/§6 output-side checks for the 2026-11-08 rebuttal audit (CPU only, no network).

Uses the 54 real v2 LightEval JSONs (paths from rebuttal_q1_paired_analysis/result_files_used.csv) and pushes
them through kys-eval's own code path, so the test exercises exactly what the collaborator's run will produce:

  1. MMLU category map: kys-eval config == 08_evaluation/02_code/mmlu_categories.py (the paper's tables)
  2. kys-eval benchmark_scores(v2 JSON) == paired_analysis per_checkpoint_scores.csv (12 values x 54)
  3. kys-eval reference/rewritten_v2_54_per_task.csv == the v2 JSONs (acc, acc_norm, 63 tasks x 54)
  4. end to end, identity: each rebuttal cell's results.json is built with eval_checkpoint.package_existing (the
     same build_results that evaluate() uses) from its comparator's v2 JSON; aggregate.py must report Δ = 0
     with n_paired = 3 for all 3 pairs x 3 epochs x 12 benchmarks
  5. end to end, non-trivial: λ0 cells are filled with quality_base's v2 JSONs; aggregate's Δ (disagreement_aware
     − λ0) must equal paired_analysis's DA − QB (pairwise_mean6_mmlu.csv, pairwise_per_task_ep3.csv)
Run with the kys-eval checkout on sys.path:  python output_side.py  (writes ../evidence/output_side/)
"""
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV = HERE.parent / "evidence" / "output_side"
KYS = Path("/weka/scratch/jhu/bvandur1/zhuicon1/projects/kys-eval")
PA = Path("/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite/docs/rebuttal_q1_paired_analysis")
CATS = Path("/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite/08_evaluation/02_code/mmlu_categories.py")
sys.path.insert(0, str(KYS))
from kys_eval import aggregate, common  # noqa: E402
from kys_eval import config as C  # noqa: E402
from kys_eval.eval_checkpoint import package_existing  # noqa: E402

EPOCH = {"1": "ep1", "2": "ep2", "3": "ep3"}
BENCH = ["arc_easy", "hellaswag", "piqa", "siqa", "openbookqa", "commonsense_qa", "mean6", "mmlu",
         "mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other"]
REBUTTAL = {"raw_diversity_oriented": "diversity_oriented", "raw_disagreement_aware": "disagreement_aware",
            "signal-disagreement-lambda0": "disagreement_aware"}
out = {}


def check(name, ok, detail=""):
    out[name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def main():
    EV.mkdir(parents=True, exist_ok=True)
    # 1
    spec = importlib.util.spec_from_file_location("mmlu_categories", CATS)
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    check("1 MMLU category map identical", mc.SUBJECT_CATEGORY == C.MMLU_SUBJECT_CATEGORY,
          f"{len(C.MMLU_SUBJECT_CATEGORY)} subjects; categories {sorted(set(C.MMLU_SUBJECT_CATEGORY.values()))}")
    keys = {k: common.category_key(c) for k, c in [("mmlu_stem", "STEM"), ("mmlu_humanities", "Humanities"),
            ("mmlu_social_sciences", "Social Sciences"), ("mmlu_other", "Other")]}
    check("1b category keys", all(k == v for k, v in keys.items()), str(keys))

    # 2
    files = list(csv.DictReader(open(PA / "result_files_used.csv")))
    pcs = {(r["setting"], int(r["seed"]), EPOCH[r["epoch"]]): r for r in csv.DictReader(open(PA / "per_checkpoint_scores.csv"))}
    raws, worst = {}, 0.0
    for r in files:
        key = (r["setting"], int(r["seed"]), EPOCH[r["epoch"]])
        raw = json.loads(Path(r["result_json"]).read_text())
        raws[key] = (raw, r)
        tasks = common.parse_lighteval_results(raw)
        s = common.benchmark_scores({t: e["acc_norm"] for t, e in tasks.items() if "acc_norm" in e})
        for b in BENCH:
            worst = max(worst, abs(s[b] * 100 - float(pcs[key][b])))
    check("2 kys-eval scores == paired_analysis per_checkpoint_scores (54 x 12)", len(raws) == 54 and worst <= 5.1e-5,
          f"{len(raws)} checkpoints, max |diff| {worst:.2e} points (CSV has 4 decimals)")

    # 3
    ref = {}
    for row in csv.DictReader(open(C.REWRITTEN_V2_PER_TASK_CSV)):
        ref[(row["setting"], int(row["seed"]), row["epoch"], row["task"])] = row
    worst3, n3, steps_ok = 0.0, 0, True
    for key, (raw, r) in raws.items():
        tasks = common.parse_lighteval_results(raw)
        for t in C.EXPECTED_TASKS:
            row = ref.get((*key, t))
            if row is None:
                worst3 = float("inf")
                continue
            n3 += 1
            worst3 = max(worst3, abs(float(row["acc_norm"]) - tasks[t]["acc_norm"]), abs(float(row["acc"]) - tasks[t]["acc"]))
            steps_ok &= int(row["step"]) == int(r["step"])
    check("3 reference CSV == v2 JSONs (acc, acc_norm)", n3 == 54 * 63 and worst3 < 1e-9 and steps_ok,
          f"{n3} task cells, max |diff| {worst3:.1e}, steps match {steps_ok}")

    # 4 and 5
    def build(tag, fill):
        root = EV / tag / "results"
        shutil.rmtree(EV / tag, ignore_errors=True)
        for raw_setting, comp in REBUTTAL.items():
            for seed in C.SEEDS:
                for ep in C.EPOCHS:
                    src_setting = fill.get(raw_setting, comp)
                    raw, r = raws[(src_setting, seed, ep)]
                    cell = {"setting": raw_setting, "seed": seed, "epoch": ep, "step": C.EPOCH_STEPS[ep],
                            "hf_subfolder": C.hf_subfolder(seed, raw_setting, ep), "hf_repo": C.HF_REPO}
                    d = root / "raw_selected" / f"seed{seed}" / raw_setting / ep
                    tmpraw = d.parent / f"_src_{ep}.json"
                    d.mkdir(parents=True, exist_ok=True)
                    tmpraw.write_text(json.dumps(raw))
                    package_existing(tmpraw, d, label=f"{tag} {raw_setting} {seed} {ep}", model=r["model_name"],
                                     gpu=r["gpu"], cell=cell)
        rep = EV / tag / "reports"
        aggregate.main(["--results-root", str(root), "--out-dir", str(rep)])
        return {(x["raw_setting"], x["epoch"], x["benchmark"]): x for x in csv.DictReader(open(rep / "raw_vs_rewritten.csv"))}

    ident = build("identity", {})
    bad = [(k, v["delta_mean"], v["n_paired"]) for k, v in ident.items() if k[0] in REBUTTAL
           and (abs(float(v["delta_mean"] or "nan")) > 1e-12 or v["n_paired"] != "3")]
    n = sum(1 for k in ident if k[0] in REBUTTAL)
    check("4 identity: Δ = 0, n_paired = 3 for the rebuttal pairs", n == 3 * 3 * 12 and not bad, f"{n} rows, {len(bad)} bad")
    others = {k for k in ident if k[0] not in REBUTTAL and ident[k]["n_paired"] not in ("0", "")}
    check("4b non-rebuttal settings have no pairs", not others, f"{len(others)} rows with data")

    shift = build("lambda0_as_quality_base", {"signal-disagreement-lambda0": "quality_base"})
    pw = {(r["metric"], r["epoch"], r["setting_a"], r["setting_b"]): r for r in csv.DictReader(open(PA / "pairwise_mean6_mmlu.csv"))}
    pt = list(csv.DictReader(open(PA / "pairwise_per_task_ep3.csv")))
    worst5, n5, seeds_ok = 0.0, 0, True
    for metric in ("mean6", "mmlu"):
        for e, ep in EPOCH.items():
            a = shift[("signal-disagreement-lambda0", ep, metric)]
            ref_row = pw.get((metric, e, "disagreement_aware", "quality_base"))
            sign = 1
            if ref_row is None:
                ref_row, sign = pw[(metric, e, "quality_base", "disagreement_aware")], -1
            worst5 = max(worst5, abs(float(a["delta_mean"]) * 100 - sign * float(ref_row["mean_diff_a_minus_b"])),
                         abs(float(a["delta_std"]) * 100 - float(ref_row["sd_diff"])))
            want_pos = int(ref_row["seeds_a_gt_b"]) if sign == 1 else 3 - int(ref_row["seeds_a_gt_b"])
            seeds_ok &= int(a["n_delta_positive"]) == want_pos
            n5 += 1
    pt_cols = set(pt[0]) if pt else set()
    check("5 aggregate Δ(DA − λ0:=QB) == paired_analysis DA − QB (Mean6, MMLU, 3 epochs)",
          n5 == 6 and worst5 <= 0.0051 and seeds_ok, f"max |diff| {worst5:.4f} points (reference has 2 decimals), seed counts {seeds_ok}")
    (EV / "pairwise_per_task_ep3_columns.txt").write_text("\n".join(sorted(pt_cols)))
    worst5b, n5b = 0.0, 0
    for r in pt:
        if {r.get("setting_a"), r.get("setting_b")} != {"disagreement_aware", "quality_base"}:
            continue
        sign = 1 if r["setting_a"] == "disagreement_aware" else -1
        b = r.get("task") or r.get("metric") or r.get("benchmark")
        row = shift.get(("signal-disagreement-lambda0", "ep3", b))
        if row is None:
            continue
        worst5b = max(worst5b, abs(float(row["delta_mean"]) * 100 - sign * float(r["mean_diff_a_minus_b"])))
        n5b += 1
    check("5b per-task epoch 3 Δ == paired_analysis per-task DA − QB", n5b >= 10 and worst5b <= 0.0051,
          f"{n5b} rows, max |diff| {worst5b:.4f}")
    (EV / "summary.json").write_text(json.dumps(out, indent=1))
    return 0 if all(v["ok"] for v in out.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
