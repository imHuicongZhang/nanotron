#!/usr/bin/env python3
"""
ICLR rebuttal Q1: paired, within-seed comparisons among all six 1.5B settings.

Read-only. Inputs are the 54 raw LightEval result JSONs of the v2 (trunk-and-branch,
re-released) evaluation:

    08_evaluation/04_results_1p5_v2/downstream/<run>/<step>/results/**/results_*.json

Every number written by this script is computed from those JSONs. The only literals are
(i) the settings / tasks / steps that define the grid, (ii) the 57 -> 4 MMLU taxonomy, imported
from 08_evaluation/02_code/mmlu_categories.py, and (iii) the Step-1 reference values, which
are read from the paper's appendix LaTeX (plus the few values the user quoted that are not
in that LaTeX, kept in USER_REFERENCE below and used only for the check).

Run:  python3 paired_analysis.py            (writes CSV / LaTeX / verification.md next to it)
Exit status is 1 if a result file is missing or Step 1 does not reproduce the paper.

Rebuttal settings (optional). With --kys-results ROOT, the script also reads kys-eval results.json files
(schema "kys-eval/results/v1", github.com/imHuicongZhang/kys-eval) for the three rebuttal settings
raw_diversity_oriented, raw_disagreement_aware and signal-disagreement-lambda0. Each file is identified by its
checkpoint, rewrite-1p5b/seed<S>/<setting>/ep<N>/hf in blab-jhu/KYS-1.5B-Raw-Selected-Baselines: the stored
`cell` (setting, seed, epoch, step, hf_subfolder) and the stored model spec / subfolder must all name the same
checkpoint, and so must the file's own path if it contains a rewrite-1p5b/seed<S>/<setting>/ep<N> or
raw_selected/seed<S>/<setting>/ep<N> segment. Files may sit anywhere below ROOT (kys-eval's
results/raw_selected/... tree, or a tree laid out as rewrite-1p5b/seed<S>/<setting>/ep<N>/hf/results.json). All 27
cells must be present, complete (63 tasks, no protocol violation) and unique. Scores are computed from the stored
per-task acc_norm exactly as for the v2 files. Each rebuttal setting is then paired within seed, setting minus
comparator, against its v2 comparator (raw_diversity_oriented - diversity_oriented, raw_disagreement_aware -
disagreement_aware, signal-disagreement-lambda0 - disagreement_aware) and against quality_base, in the same CSV and
LaTeX tables as the 15 v2 pairs (an added block of six pairs), in the LOTO and 2d outputs, and in
per_checkpoint_scores.csv / result_files_used.csv. Step 1 and the per-seed ranks use the six v2 settings only.
Without --kys-results the outputs are byte-identical to the v2-only version.

      python3 paired_analysis.py --kys-results <dir> [--out-dir <dir>]
"""

import argparse
import csv
import glob
import itertools
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REWRITE = os.path.dirname(os.path.dirname(HERE))  # .../projects/rewrite
EVAL = os.path.join(REWRITE, "08_evaluation")
DOWN = os.path.join(EVAL, "04_results_1p5_v2", "downstream")
CKPT_LIST = os.path.join(EVAL, "02_code", "ckpt_list_v2.txt")
PAPER_TEX = "/weka/scratch/jhu/bvandur1/zhuicon1/paper-writing/06_iclr/appendix_training_and_per_task.tex"
PAPER_MMLU_CAT_TEX = "/weka/scratch/jhu/bvandur1/zhuicon1/paper-writing/06_iclr/09_appendixF_mmlu_category.tex"

sys.path.insert(0, os.path.join(EVAL, "02_code"))
from mmlu_categories import CATEGORIES, SUBJECT_CATEGORY  # noqa: E402

# ---------------------------------------------------------------- grid definition
# (key, run-directory stem, paper label). Order = the paper's table order.
SETTINGS = [
    ("quality_base", "quality-base", "Quality Base"),
    ("quality_first", "quality-first", "Quality First"),
    ("diversity_oriented", "diversity-first", "Diversity Oriented"),
    ("wrap_inspired", "wrap", "WRAP Inspired"),
    ("rewire_inspired", "rewrite", "REWIRE Inspired"),
    ("disagreement_aware", "signal-disagreement-lambda05", "Disagreement Aware"),
]
KEYS = [s[0] for s in SETTINGS]
STEM = {s[0]: s[1] for s in SETTINGS}
LABEL = {s[0]: s[2] for s in SETTINGS}
SHORT = {"quality_base": "QB", "quality_first": "QF", "diversity_oriented": "DO",
         "wrap_inspired": "WRAP", "rewire_inspired": "REWIRE", "disagreement_aware": "DA"}
BASE = "quality_base"
STRATEGIES = [k for k in KEYS if k != BASE]
SEEDS = [42, 43, 44]
EPOCHS = [(1, 4768), (2, 9537), (3, 14305)]
# (lighteval task, column, display) in the paper's column order
TASKS6 = [("rw_arc:easy", "arc_easy", "ARC-e"), ("rw_hellaswag", "hellaswag", "HellaSwag"),
          ("rw_piqa", "piqa", "PIQA"), ("rw_siqa", "siqa", "SIQA"),
          ("rw_openbookqa", "openbookqa", "OBQA"), ("rw_commonsense_qa", "commonsense_qa", "CSQA")]
TASK_COLS = [t[1] for t in TASKS6]
TASK_DISP = {t[1]: t[2] for t in TASKS6}
CAT_COLS = ["mmlu_" + c.lower().replace(" ", "_") for c in CATEGORIES]
CAT_DISP = dict(zip(CAT_COLS, ["STEM", "Humanities", "Social Sci.", "Other"]))
# Pairs the user asked for explicitly, in the requested orientation (first minus second).
REQUESTED = [("diversity_oriented", "disagreement_aware"), ("diversity_oriented", "quality_first"),
             ("disagreement_aware", "quality_first"), ("quality_first", "rewire_inspired")]

# Values quoted in the request that do not appear in the appendix LaTeX (Step 1 check only).
USER_REFERENCE = {
    "loto_do_ep3_min": -0.14, "loto_do_ep3_max": +0.42,
    "loto_do_ep3_negative_when_dropped": {"hellaswag", "piqa"},
}

TOL = 0.005 + 1e-9  # half a unit in the 2nd decimal: anything larger is more than rounding

# Rebuttal settings scored with kys-eval: (key, paper-style label, short name, primary v2 comparator).
NEW_SETTINGS = [
    ("raw_diversity_oriented", "Raw Diversity Oriented", "rawDO", "diversity_oriented"),
    ("raw_disagreement_aware", "Raw Disagreement Aware", "rawDA", "disagreement_aware"),
    ("signal-disagreement-lambda0", "Disagreement Aware ($\\lambda=0$)", "DA0", "disagreement_aware"),
]
NEW_KEYS = [s[0] for s in NEW_SETTINGS]
NEW_COMPARATOR = {s[0]: s[3] for s in NEW_SETTINGS}
KYS_SCHEMA = "kys-eval/results/v1"
KYS_REPO = "blab-jhu/KYS-1.5B-Raw-Selected-Baselines"
OUT = HERE  # output directory (--out-dir)
ALL_KEYS = list(KEYS)  # KEYS plus the rebuttal settings once they are loaded


def hf_subfolder(seed, setting, ep):
    return f"rewrite-1p5b/seed{seed}/{setting}/ep{ep}/hf"


def pts(x):
    return 100.0 * x


def sd(v):
    return statistics.stdev(v)  # sample SD, ddof=1 (the paper's convention)


def mean(v):
    return sum(v) / len(v)


def f2(x):
    """Two decimals, with -0.00 printed as 0.00."""
    s = f"{x:.2f}"
    return "0.00" if s in ("-0.00",) else s


def s2(x):
    s = f"{x:+.2f}"
    return "+0.00" if s in ("-0.00", "+0.00") else s


# ---------------------------------------------------------------- load
def load():
    srcmap = {}
    if os.path.exists(CKPT_LIST):
        for line in open(CKPT_LIST):
            p = line.split()
            if p:
                srcmap[(p[0], p[1])] = p[3]
    data, files, missing = {}, [], []
    for k in KEYS:
        for seed in SEEDS:
            run = f"{STEM[k]}-10B-1.5B-seed{seed}"
            for ep, step in EPOCHS:
                hits = sorted(glob.glob(os.path.join(DOWN, run, str(step), "results", "**",
                                                    "results_*.json"), recursive=True))
                if len(hits) != 1:
                    missing.append((k, seed, ep, step, f"{len(hits)} result files"))
                    continue
                path = hits[0]
                d = json.load(open(path))
                res = d["results"]
                model_name = d["config_general"]["model_name"]
                expect = f"/rewrite-1p5b-hf/{run}/{step}"
                if not model_name.endswith(expect):
                    missing.append((k, seed, ep, step, f"model_name {model_name} != *{expect}"))
                    continue
                row = {}
                for task, col, _ in TASKS6:
                    row[col] = pts(res[f"{task}|0"]["acc_norm"])
                subj = {kk.split("|")[0].split(":", 1)[1]: m["acc_norm"] for kk, m in res.items()
                        if kk.startswith("rw_mmlu:") and not kk.startswith("rw_mmlu:_average")}
                if set(subj) != set(SUBJECT_CATEGORY):
                    missing.append((k, seed, ep, step, f"{len(subj)} MMLU subjects"))
                    continue
                row["mean6"] = mean([row[c] for c in TASK_COLS])
                row["mmlu"] = pts(mean(list(subj.values())))
                for cat, col in zip(CATEGORIES, CAT_COLS):
                    row[col] = pts(mean([v for s, v in subj.items() if SUBJECT_CATEGORY[s] == cat]))
                gpu_f = os.path.join(DOWN, run, str(step), "gpu.txt")
                data[(k, seed, ep)] = row
                files.append({"setting": k, "paper_label": LABEL[k], "seed": seed, "epoch": ep,
                              "step": step, "result_json": path, "model_name": model_name,
                              "nanotron_source": srcmap.get((run, str(step)), ""),
                              "lighteval_sha": d["config_general"].get("lighteval_sha", ""),
                              "gpu": open(gpu_f).read().strip() if os.path.exists(gpu_f) else ""})
    return data, files, missing


def scores_from_acc_norm(acc):
    """{lighteval task id without '|0': acc_norm} -> the row dict, with exactly the arithmetic of load()."""
    row = {}
    for task, col, _ in TASKS6:
        row[col] = pts(acc[task])
    subj = {kk.split(":", 1)[1]: v for kk, v in acc.items()
            if kk.startswith("rw_mmlu:") and not kk.startswith("rw_mmlu:_average")}
    if set(subj) != set(SUBJECT_CATEGORY):
        return None, f"{len(subj)} MMLU subjects"
    row["mean6"] = mean([row[c] for c in TASK_COLS])
    row["mmlu"] = pts(mean(list(subj.values())))
    for cat, col in zip(CATEGORIES, CAT_COLS):
        row[col] = pts(mean([v for s, v in subj.items() if SUBJECT_CATEGORY[s] == cat]))
    return row, None


PATH_CELL = re.compile(r"(?:rewrite-1p5b|raw_selected)/seed(\d+)/([^/]+)/ep(\d)(?:/|$)")


def load_kys(root):
    """kys-eval results.json files for the 27 rebuttal cells below root. Returns (data, files, problems)."""
    data, files, problems, seen = {}, [], [], {}
    steps = dict(EPOCHS)
    for path in sorted(glob.glob(os.path.join(root, "**", "results.json"), recursive=True)):
        try:
            d = json.load(open(path))
        except (OSError, ValueError) as exc:
            problems.append((path, f"unreadable: {exc}"))
            continue
        if d.get("schema") != KYS_SCHEMA:
            continue  # not a kys-eval results file
        cell, model = d.get("cell") or {}, d.get("model") or {}
        setting = cell.get("setting")
        if setting not in NEW_KEYS:
            continue  # another kys-eval cell (e.g. one of the five non-rebuttal settings)
        try:
            seed, ep = int(cell["seed"]), int(str(cell["epoch"]).replace("ep", ""))
        except (KeyError, ValueError):
            problems.append((path, f"bad cell {cell}"))
            continue
        want_sub = hf_subfolder(seed, setting, ep)
        why = []
        if seed not in SEEDS or ep not in steps:
            why.append(f"seed/epoch {seed}/{ep} outside the grid")
        elif int(cell.get("step", -1)) != steps[ep]:
            why.append(f"cell step {cell.get('step')} != {steps[ep]}")
        if cell.get("hf_subfolder") != want_sub:
            why.append(f"cell hf_subfolder {cell.get('hf_subfolder')!r} != {want_sub!r}")
        spec = model.get("spec") or ""
        sub_m = model.get("subfolder")
        if sub_m is None and spec.startswith("hf://"):
            sub_m = "/".join([x for x in spec[5:].split("@")[0].split("/") if x][2:])
        if sub_m is not None:
            if sub_m != want_sub:
                why.append(f"model subfolder {sub_m!r} != {want_sub!r}")
        elif not spec.rstrip("/").endswith(want_sub):
            why.append(f"model spec {spec!r} does not name {want_sub}")
        for m in PATH_CELL.finditer(os.path.relpath(path, root).replace(os.sep, "/")):
            if (int(m.group(1)), m.group(2), int(m.group(3))) != (seed, setting, ep):
                why.append(f"file path names seed{m.group(1)}/{m.group(2)}/ep{m.group(3)}")
        if not d.get("complete") or d.get("missing_tasks") or d.get("protocol_violations"):
            why.append(f"incomplete: missing {len(d.get('missing_tasks') or [])} tasks, "
                       f"violations {d.get('protocol_violations')}")
        acc = {t: e["acc_norm"] for t, e in (d.get("tasks") or {}).items() if "acc_norm" in e}
        row, err = (None, None)
        if not why:
            row, err = scores_from_acc_norm(acc) if all(t in acc for t, _, _ in TASKS6) else (None, "commonsense task missing")
            if err:
                why.append(err)
        if why:
            problems.append((path, "; ".join(why)))
            continue
        key = (setting, seed, ep)
        if key in seen:
            problems.append((path, f"duplicate of {seen[key]}"))
            continue
        seen[key] = path
        data[key] = row
        env = d.get("environment") or {}
        files.append({"setting": setting, "paper_label": dict((s[0], s[1]) for s in NEW_SETTINGS)[setting],
                      "seed": seed, "epoch": ep, "step": steps[ep], "result_json": os.path.abspath(path),
                      "model_name": spec, "nanotron_source": model.get("commit", ""),
                      "lighteval_sha": env.get("lighteval_sha_recorded", ""), "gpu": env.get("gpu", "")})
    for k in NEW_KEYS:
        for seed in SEEDS:
            for ep, step in EPOCHS:
                if (k, seed, ep) not in data:
                    problems.append((hf_subfolder(seed, k, ep), "no valid results.json"))
    return data, files, problems


def rebuttal_pairs():
    """Each rebuttal setting minus its v2 comparator, then minus Quality Base (Appendix F orientation)."""
    out = []
    for k in NEW_KEYS:
        out.append((k, NEW_COMPARATOR[k]))
        if NEW_COMPARATOR[k] != BASE:
            out.append((k, BASE))
    return out


# ---------------------------------------------------------------- paired statistics
def paired(data, a, b, ep, metric):
    diffs = [data[(a, s, ep)][metric] - data[(b, s, ep)][metric] for s in SEEDS]
    return {"diffs": diffs, "mean": mean(diffs), "sd": sd(diffs),
            "n_pos": sum(d > 0 for d in diffs), "n_neg": sum(d < 0 for d in diffs)}


def absolute(data, k, ep, metric):
    v = [data[(k, s, ep)][metric] for s in SEEDS]
    return mean(v), sd(v)


def loto_metric(row, drop):
    return mean([row[c] for c in TASK_COLS if c != drop])


def pairs15():
    out = []
    order = STRATEGIES + [BASE]  # so every QB pair reads strategy minus Quality Base
    for a, b in itertools.combinations(order, 2):
        if (b, a) in REQUESTED:
            a, b = b, a
        out.append((a, b))
    # strategy-vs-strategy pairs first, then the five strategy-vs-Quality-Base pairs
    return [p for p in out if BASE not in p] + [p for p in out if BASE in p]


# ---------------------------------------------------------------- Step 1: paper check
NUM = r"([+\-]?\d+\.\d+)"


def parse_paper():
    """Return {(table, setting, epoch_or_col): (mean, sd, n_pos)} from the appendix LaTeX."""
    tex = open(PAPER_TEX).read()
    ref = {}
    lab_to_key = {v: k for k, v in LABEL.items()}

    def tbl(label):
        i = tex.index(f"\\label{{{label}}}")
        return tex[i:tex.index("\\end{tabular}", i)]

    # per-task epoch 3
    for line in tbl("tab:per-task-epoch3").splitlines():
        m = re.match(r"\\textsc\{(.+?)\}\s*&(.*)\\\\", line.strip())
        if m and m.group(1) in lab_to_key:
            cells = re.findall(r"\$(?:\\mathbf\{)?" + NUM + r"\s*\\pm\s*" + NUM, m.group(2))
            for col, (mu, sdv) in zip(TASK_COLS, cells):
                ref[("pertask_ep3", lab_to_key[m.group(1)], col)] = (float(mu), float(sdv), None)
    # Mean6 / MMLU by epoch
    for label, tag in (("tab:mean6-by-epoch", "mean6"), ("tab:mmlu-by-epoch", "mmlu")):
        cur = None
        for line in tbl(label).splitlines():
            m = re.search(r"\\multicolumn\{4\}\{l\}\{\\textsc\{(.+?)\}\}", line)
            if m:
                cur = lab_to_key[m.group(1)]
                continue
            m = re.match(r"\\quad\s*(\d)\s*&(.*)\\\\", line.strip())
            if m and cur:
                ep = int(m.group(1))
                cells = re.findall(NUM + r"\s*\\pm\s*" + NUM, m.group(2))
                ref[(tag + "_abs", cur, ep)] = (float(cells[0][0]), float(cells[0][1]), None)
                if cur != BASE:
                    k = re.search(r"(\d)/3", m.group(2))
                    ref[(tag + "_delta", cur, ep)] = (float(cells[1][0]), float(cells[1][1]),
                                                      int(k.group(1)))
    # MMLU by category, epoch 3 (deltas with seed counts)
    if os.path.exists(PAPER_MMLU_CAT_TEX):
        for line in open(PAPER_MMLU_CAT_TEX):
            m = re.match(r"\\textsc\{(.+?)\}\s*&(.*)\\\\", line.strip())
            if not m or m.group(1) not in lab_to_key:
                continue
            k = lab_to_key[m.group(1)]
            if k == BASE:
                cells = re.findall(NUM + r"\s*\\pm\s*" + NUM, m.group(2))
                for col, (mu, sdv) in zip(CAT_COLS + ["mmlu"], cells):
                    ref[("mmlucat_abs", k, col)] = (float(mu), float(sdv), None)
            else:
                cells = re.findall(r"\$(?:\\mathbf\{)?" + NUM + r"\}?\$\s*\[\$(?:\\mathbf\{)?(\d)",
                                   m.group(2))
                for col, (mu, n) in zip(CAT_COLS + ["mmlu"], cells):
                    ref[("mmlucat_delta", k, col)] = (float(mu), None, int(n))
    return ref


def verify(data):
    ref = parse_paper()
    rows, worst = [], 0.0
    for (tbl, k, x), (mu, sdv, n) in sorted(ref.items(), key=str):
        if tbl == "pertask_ep3":
            got = absolute(data, k, 3, x) if k == BASE else \
                (lambda p: (p["mean"], p["sd"], p["n_pos"]))(paired(data, k, BASE, 3, x))
        elif tbl.endswith("_abs") and tbl != "mmlucat_abs":
            got = absolute(data, k, x, tbl[:-4])
        elif tbl.endswith("_delta") and tbl != "mmlucat_delta":
            p = paired(data, k, BASE, x, tbl[:-6])
            got = (p["mean"], p["sd"], p["n_pos"])
        elif tbl == "mmlucat_abs":
            got = absolute(data, k, 3, x)
        else:
            p = paired(data, k, BASE, 3, x)
            got = (p["mean"], p["sd"], p["n_pos"])
        got = tuple(got) + (None,) * (3 - len(got))
        e_mu = abs(got[0] - mu)
        e_sd = abs(got[1] - sdv) if sdv is not None else 0.0
        n_ok = (n is None) or (got[2] == n)
        worst = max(worst, e_mu, e_sd)
        rows.append({"source": "appendix_tex", "table": tbl, "setting": k, "cell": x,
                     "paper_mean": mu, "computed_mean": round(got[0], 4),
                     "paper_sd": "" if sdv is None else sdv,
                     "computed_sd": "" if sdv is None else round(got[1], 4),
                     "paper_seeds_pos": "" if n is None else n,
                     "computed_seeds_pos": "" if n is None else got[2],
                     "ok": e_mu <= TOL and e_sd <= TOL and n_ok})
    # user-quoted LOTO reference (not in the LaTeX)
    loto = {c: paired_loto(data, "diversity_oriented", BASE, 3, c)["mean"] for c in TASK_COLS}
    lo, hi = min(loto.values()), max(loto.values())
    neg = {c for c, v in loto.items() if v < 0}
    for name, paper, got, ok in (
            ("loto_do_ep3_min", USER_REFERENCE["loto_do_ep3_min"], lo,
             abs(lo - USER_REFERENCE["loto_do_ep3_min"]) <= TOL),
            ("loto_do_ep3_max", USER_REFERENCE["loto_do_ep3_max"], hi,
             abs(hi - USER_REFERENCE["loto_do_ep3_max"]) <= TOL),
            ("loto_do_ep3_negative_when_dropped",
             "|".join(sorted(USER_REFERENCE["loto_do_ep3_negative_when_dropped"])),
             "|".join(sorted(neg)), neg == USER_REFERENCE["loto_do_ep3_negative_when_dropped"])):
        rows.append({"source": "user_quote", "table": "loto", "setting": "diversity_oriented",
                     "cell": name, "paper_mean": paper,
                     "computed_mean": round(got, 4) if isinstance(got, float) else got,
                     "paper_sd": "", "computed_sd": "", "paper_seeds_pos": "",
                     "computed_seeds_pos": "", "ok": ok})
    return rows, worst


def paired_loto(data, a, b, ep, drop):
    diffs = [loto_metric(data[(a, s, ep)], drop) - loto_metric(data[(b, s, ep)], drop)
             for s in SEEDS]
    return {"diffs": diffs, "mean": mean(diffs), "sd": sd(diffs),
            "n_pos": sum(d > 0 for d in diffs)}


def criterion(p, a, b):
    """The 2d 'supported' rule: 3/3 seeds in one direction and |mean| > SD. Returns (met, winner)."""
    if p["n_pos"] == 3 and p["mean"] > p["sd"]:
        return True, a
    if sum(d < 0 for d in p["diffs"]) == 3 and -p["mean"] > p["sd"]:
        return True, b
    return False, None


FOCUS = [("diversity_oriented", k) for k in ("quality_first", "wrap_inspired", "rewire_inspired",
                                             "disagreement_aware", "quality_base")]


def write_loto_summary(cells):
    """loto_summary.md: every number below is computed from `cells`."""
    def oriented(a, b, p, win):
        """Mean and per-seed diffs expressed as winner minus loser."""
        s = 1 if win == a else -1
        return s * p["mean"], [s * d for d in p["diffs"]]

    def cell_str(p, a, b, win):
        mu, ds = oriented(a, b, p, win)
        k = sum(d > 0 for d in ds)
        return f"{s2(mu)} ± {f2(p['sd'])} ({k}/3)"

    def label(a, b, win):
        lose = b if win == a else a
        return f"{SHORT[win]} > {SHORT[lose]}"

    out = ["# Leave-one-task-out (LOTO) robustness of the Mean6 pairwise comparisons", "",
           "Generated by `paired_analysis.py`; source rows are in `loto_mean6.csv`. For each of "
           f"the {len(cells) // len(EPOCHS)} pairs at each epoch, the commonsense mean is recomputed six times, each time "
           "without one of ARC-e, HellaSwag, PIQA, SIQA, OBQA, CSQA (an unweighted five-task "
           "mean). Each variant is paired within seed exactly as in the full analysis. "
           "**Supported** is the criterion used throughout this directory: all three per-seed "
           "differences have the same sign *and* |mean| > SD. A variant *keeps* a supported "
           "comparison if it meets that criterion in the same direction. All values are accuracy "
           "points, written as winner minus loser, mean ± SD (seeds > 0).", ""]

    # ---- focus table
    out += ["## Focus: Diversity-Oriented vs the other settings, epochs 2 and 3", "",
            "| Comparison | Ep | Full Mean6 | Supported? | LOTO keeps (of 6) | LOTO range of Δ | "
            "Variants that break it |", "|---|---|---|---|---|---|---|"]
    for a, b, ep, full, fc, fw, var in cells:
        pair = (a, b) if (a, b) in FOCUS else ((b, a) if (b, a) in FOCUS else None)
        if pair is None or ep == 1:
            continue
        do, other = pair
        win = fw if fc else do  # orient on DO when unsupported
        mus = [oriented(a, b, v[0], win)[0] for v in var.values()]
        if fc:
            broken = [f"−{TASK_DISP[c]}: {cell_str(v[0], a, b, win)}"
                      for c, v in var.items() if not v[4]]
            keep = f"{sum(v[4] for v in var.values())}/6"
        else:
            broken = [f"−{TASK_DISP[c]}: {cell_str(v[0], a, b, win)}"
                      for c, v in var.items() if v[1]]
            broken = (["sign flips: " + "; ".join(broken)] if broken else [])
            keep = "n/a"
        out.append(f"| {label(a, b, win)} | {ep} | {cell_str(full, a, b, win)} | "
                   f"{'yes' if fc else 'no'} | {keep} | {s2(min(mus))} to {s2(max(mus))} | "
                   f"{'; '.join(broken) if broken else '—'} |")
    out.append("")

    # ---- all supported comparisons
    robust, fragile, unsup_flip, emergent = [], [], [], []
    for a, b, ep, full, fc, fw, var in cells:
        if fc:
            n_keep = sum(v[4] for v in var.values())
            breaks = [c for c, v in var.items() if not v[4]]
            flips = [c for c, v in var.items() if v[1]]
            rec = (ep, a, b, full, fw, var, n_keep, breaks, flips)
            (robust if n_keep == 6 else fragile).append(rec)
        else:
            flips = [c for c, v in var.items() if v[1]]
            if flips:
                unsup_flip.append((ep, a, b, full, flips, var))
            for c, v in var.items():
                if v[2]:
                    emergent.append((ep, a, b, full, c, v))

    out += ["## Every pair × epoch: how many LOTO variants keep the supported criterion", "",
            "| Ep | Comparison (a − b) | Full Mean6 Δ (a − b) | Seeds a > b | Full supported? | "
            "LOTO keeps (of 6) | Sign flips (of 6) |", "|---|---|---|---|---|---|---|"]
    for a, b, ep, full, fc, fw, var in sorted(cells, key=lambda c: (c[2],)):
        n_keep = sum(v[4] for v in var.values())
        out.append(f"| {ep} | {SHORT[a]} − {SHORT[b]} | {s2(full['mean'])} ± {f2(full['sd'])} | "
                   f"{full['n_pos']}/3 | {'yes (' + SHORT[fw] + ')' if fc else 'no'} | "
                   f"{str(n_keep) + '/6' if fc else '—'} | "
                   f"{sum(v[1] for v in var.values())} |")
    out.append("")

    out += [f"## Supported on the full Mean6 and in all six LOTO variants ({len(robust)})", "",
            "| Ep | Comparison | Full Mean6 | Weakest LOTO variant |", "|---|---|---|---|"]
    for ep, a, b, full, fw, var, n_keep, breaks, flips in sorted(robust, key=lambda r: r[0]):
        wc = min(var, key=lambda c: oriented(a, b, var[c][0], fw)[0])
        out.append(f"| {ep} | {label(a, b, fw)} | {cell_str(full, a, b, fw)} | "
                   f"−{TASK_DISP[wc]}: {cell_str(var[wc][0], a, b, fw)} |")
    out += ["", f"## Supported on the full Mean6 but not under every LOTO variant ({len(fragile)})",
            "", "| Ep | Comparison | Full Mean6 | Keeps | Variants that break it | Sign flips |",
            "|---|---|---|---|---|---|"]
    for ep, a, b, full, fw, var, n_keep, breaks, flips in sorted(fragile, key=lambda r: r[0]):
        br = "; ".join(f"−{TASK_DISP[c]}: {cell_str(var[c][0], a, b, fw)}" for c in breaks)
        out.append(f"| {ep} | {label(a, b, fw)} | {cell_str(full, a, b, fw)} | {n_keep}/6 | "
                   f"{br} | {', '.join(TASK_DISP[c] for c in flips) or 'none'} |")
    out += ["", f"## Not supported on the full Mean6, with a LOTO sign flip ({len(unsup_flip)})", "",
            "| Ep | Comparison (a − b) | Full Mean6 Δ | Dropped task → Δ |", "|---|---|---|---|"]
    for ep, a, b, full, flips, var in sorted(unsup_flip, key=lambda r: r[0]):
        fl = "; ".join(f"−{TASK_DISP[c]}: {s2(var[c][0]['mean'])} ± {f2(var[c][0]['sd'])} "
                       f"({var[c][0]['n_pos']}/3)" for c in flips)
        out.append(f"| {ep} | {SHORT[a]} − {SHORT[b]} | {s2(full['mean'])} ± {f2(full['sd'])} "
                   f"({full['n_pos']}/3) | {fl} |")
    out += ["", "## Not supported on the full Mean6, but one LOTO variant meets the criterion "
            f"({len(emergent)})", "",
            "These are not evidence for the comparison; they show how much one task can move a "
            "three-seed result.", "", "| Ep | Comparison | Full Mean6 | Dropped | LOTO Δ |",
            "|---|---|---|---|---|"]
    for ep, a, b, full, c, v in sorted(emergent, key=lambda r: r[0]):
        out.append(f"| {ep} | {label(a, b, v[3])} | {cell_str(full, a, b, v[3])} | "
                   f"{TASK_DISP[c]} | {cell_str(v[0], a, b, v[3])} |")
    out.append("")
    with open(os.path.join(OUT, "loto_summary.md"), "w") as f:
        f.write("\n".join(out))
    print(f"[loto] {len(cells)} pair-epoch cells; supported {len(robust) + len(fragile)} "
          f"(robust to all 6 LOTO: {len(robust)}, benchmark-dependent: {len(fragile)}); "
          f"unsupported with sign flip: {len(unsup_flip)}; emergent LOTO-supported variants: "
          f"{len(emergent)}")


# ---------------------------------------------------------------- output helpers
def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def tex_pair(a, b):
    return f"\\textsc{{{LABEL[a]}}} $-$ \\textsc{{{LABEL[b]}}}"


def tex_cell(p):
    k = f"{p['n_pos']}/3"
    k = f"$\\mathbf{{{k}}}$" if p["n_pos"] == 3 else f"${k}$"
    return f"${s2(p['mean'])} \\pm {f2(p['sd'])}$ & {k}"


# ---------------------------------------------------------------- main
def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kys-results", metavar="ROOT", help="directory holding kys-eval results.json files for the "
                    "27 rebuttal cells (see the module docstring)")
    ap.add_argument("--out-dir", default=HERE, help="where to write the outputs (default: next to this script)")
    args = ap.parse_args(argv)
    OUT = os.path.abspath(args.out_dir)
    os.makedirs(OUT, exist_ok=True)

    data, files, missing = load()
    new_pairs = []
    if args.kys_results:
        kdata, kfiles, kproblems = load_kys(args.kys_results)
        if kproblems:
            print("MISSING / INVALID kys-eval result sets:")
            for m in kproblems:
                print("   ", m)
            sys.exit(1)
        assert len(kfiles) == 27
        data.update(kdata)
        files += kfiles
        ALL_KEYS.extend(NEW_KEYS)
        LABEL.update({s[0]: s[1] for s in NEW_SETTINGS})
        SHORT.update({s[0]: s[2] for s in NEW_SETTINGS})
        new_pairs = rebuttal_pairs()
        print(f"[kys] {len(kfiles)} rebuttal cells from {os.path.abspath(args.kys_results)}; "
              f"{len(new_pairs)} added pairs: " + ", ".join(f"{SHORT[a]}-{SHORT[b]}" for a, b in new_pairs))
    write_csv("result_files_used.csv", list(files[0].keys()) if files else ["none"],
              [list(r.values()) for r in files])
    if missing:
        print("MISSING / INVALID result sets:")
        for m in missing:
            print("   ", m)
        sys.exit(1)
    assert len(files) == 54 + (27 if new_pairs else 0)

    # per-checkpoint values (the base table everything else is computed from)
    cols = TASK_COLS + ["mean6", "mmlu"] + CAT_COLS
    write_csv("per_checkpoint_scores.csv", ["setting", "paper_label", "seed", "epoch", "step"] + cols,
              [[k, LABEL[k], s, ep, step] + [f"{data[(k, s, ep)][c]:.4f}" for c in cols]
               for k in ALL_KEYS for ep, step in EPOCHS for s in SEEDS])

    # ---------------- Step 1
    vrows, worst = verify(data)
    write_csv("step1_verification.csv", list(vrows[0].keys()), [list(r.values()) for r in vrows])
    n_bad = sum(not r["ok"] for r in vrows)
    with open(os.path.join(OUT, "step1_verification.md"), "w") as f:
        f.write(f"# Step 1 verification\n\nChecked {len(vrows)} reference cells "
                f"({sum(r['source'] == 'appendix_tex' for r in vrows)} parsed from `{PAPER_TEX}` and "
                f"`{PAPER_MMLU_CAT_TEX}`, {sum(r['source'] == 'user_quote' for r in vrows)} quoted "
                f"in the request). Tolerance: |computed - printed| <= 0.005 points on means and SDs; "
                f"seed counts must match exactly.\n\n"
                f"Largest absolute deviation on any mean or SD: {worst:.4f} points. "
                f"Mismatches: {n_bad}.\n\n")
        for r in vrows:
            if not r["ok"]:
                f.write(f"- MISMATCH {r}\n")
    print(f"[step1] {len(vrows)} cells, worst |dev| = {worst:.4f} pts, mismatches = {n_bad}")
    if n_bad:
        sys.exit(1)

    # ---------------- Step 2a: pairwise, Mean6 + MMLU, all epochs
    P = pairs15() + new_pairs
    comps = []  # every comparison, for 2d
    rows = []
    for a, b in P:
        for ep, _ in EPOCHS:
            for metric in ("mean6", "mmlu"):
                p = paired(data, a, b, ep, metric)
                fam = "Commonsense Mean6" if metric == "mean6" else "MMLU macro"
                comps.append((fam, metric, ep, a, b, p))
                rows.append([metric, ep, a, b, LABEL[a], LABEL[b], f2(p["mean"]), f2(p["sd"]),
                             p["n_pos"], *[f2(d) for d in p["diffs"]],
                             int((a, b) in REQUESTED)])
    hdr = ["metric", "epoch", "setting_a", "setting_b", "label_a", "label_b", "mean_diff_a_minus_b",
           "sd_diff", "seeds_a_gt_b", "diff_seed42", "diff_seed43", "diff_seed44", "requested_pair"]
    write_csv("pairwise_mean6_mmlu.csv", hdr, rows)

    # Step 2a: per task / per MMLU category at epoch 3, all 15 pairs (the 4 requested are flagged)
    rows = []
    for a, b in P:
        for metric in TASK_COLS + CAT_COLS:
            p = paired(data, a, b, 3, metric)
            fam = "Commonsense task (ep3)" if metric in TASK_COLS else "MMLU category (ep3)"
            comps.append((fam, metric, 3, a, b, p))
            rows.append([metric, 3, a, b, LABEL[a], LABEL[b], f2(p["mean"]), f2(p["sd"]),
                         p["n_pos"], *[f2(d) for d in p["diffs"]], int((a, b) in REQUESTED)])
    write_csv("pairwise_per_task_ep3.csv", hdr, rows)

    # ---------------- Step 2b: per-seed ranks
    rank_rows, mean_rank_rows, ties = [], [], []
    ranks = {}
    for metric in ("mean6", "mmlu"):
        for ep, _ in EPOCHS:
            for s in SEEDS:
                vals = {k: data[(k, s, ep)][metric] for k in KEYS}
                for k in KEYS:
                    better = sum(v > vals[k] for v in vals.values())
                    equal = sum(v == vals[k] for v in vals.values())
                    ranks[(metric, ep, s, k)] = better + (equal + 1) / 2  # average rank on ties
                    if equal > 1:
                        ties.append((metric, ep, s, k))
            for k in KEYS:
                r = [ranks[(metric, ep, s, k)] for s in SEEDS]
                rank_rows.append([metric, ep, k, LABEL[k]] + [f"{x:g}" for x in r]
                                 + [f2(mean(r)), int(len(set(r)) == 1)])
    write_csv("ranks_per_seed.csv", ["metric", "epoch", "setting", "label", "rank_seed42",
                                     "rank_seed43", "rank_seed44", "mean_rank",
                                     "same_rank_all_seeds"], rank_rows)

    # ---------------- Step 2c: leave-one-task-out
    # All 15 pairs (same orientation as 2a). "criterion" = the 2d supported rule: all three
    # per-seed differences share one sign and |mean| > SD; "winner" is the setting it favours.
    loto_rows, loto_cells = [], []
    for a, b in P:
        for ep, _ in EPOCHS:
            full = paired(data, a, b, ep, "mean6")
            f_crit, f_win = criterion(full, a, b)
            loto_rows.append([a, b, ep, "none", f2(full["mean"]), f2(full["sd"]), full["n_pos"],
                              *[f2(d) for d in full["diffs"]], 0, int(f_crit),
                              f_win if f_crit else "", ""])
            variants = {}
            for c in TASK_COLS:
                p = paired_loto(data, a, b, ep, c)
                crit, win = criterion(p, a, b)
                flip = int((p["mean"] > 0) != (full["mean"] > 0))
                keeps = int(f_crit and crit and win == f_win)
                variants[c] = (p, flip, crit, win, keeps)
                loto_rows.append([a, b, ep, c, f2(p["mean"]), f2(p["sd"]), p["n_pos"],
                                  *[f2(d) for d in p["diffs"]], flip, int(crit),
                                  win if crit else "", keeps if f_crit else ""])
            loto_cells.append((a, b, ep, full, f_crit, f_win, variants))
    write_csv("loto_mean6.csv", ["setting_a", "setting_b", "epoch", "dropped_task", "mean_diff",
                                 "sd_diff", "seeds_a_gt_b", "diff_seed42", "diff_seed43",
                                 "diff_seed44", "sign_differs_from_full_mean6",
                                 "meets_supported_criterion", "criterion_winner",
                                 "keeps_full_mean6_supported_status"], loto_rows)
    write_loto_summary(loto_cells)

    # ---------------- Step 2d: classification (winner listed first; 0/3 is a 3/3 reversal)
    strong, weak = [], []
    fam_order = ["Commonsense Mean6", "MMLU macro", "Commonsense task (ep3)", "MMLU category (ep3)"]
    for fam, metric, ep, a, b, p in comps:
        if p["n_pos"] == 3:
            w, l, mu = a, b, p["mean"]
        elif p["n_neg"] == 3:
            w, l, mu = b, a, -p["mean"]
        else:
            continue
        disp = {"mean6": "Mean6", "mmlu": "MMLU macro"}.get(metric) or TASK_DISP.get(metric) \
            or CAT_DISP[metric]
        diffs = [d if w == a else -d for d in p["diffs"]]
        rec = [fam, ep, disp, w, l, LABEL[w], LABEL[l], f2(mu), f2(p["sd"]),
               *[f2(d) for d in diffs], f"{min(diffs):.4f}"]
        (strong if mu > p["sd"] else weak).append(rec)
    key = lambda r: (fam_order.index(r[0]), r[1], r[2], ALL_KEYS.index(r[3]), ALL_KEYS.index(r[4]))
    strong.sort(key=key)
    weak.sort(key=key)
    chdr = ["family", "epoch", "benchmark", "winner", "loser", "winner_label", "loser_label",
            "mean_diff", "sd_diff", "diff_seed42", "diff_seed43", "diff_seed44",
            "smallest_seed_margin_4dp"]
    write_csv("summary_supported_3of3_mean_gt_sd.csv", chdr, strong)
    write_csv("summary_consistent_3of3_mean_le_sd.csv", chdr, weak)
    n_total = len(comps)

    # ---------------- LaTeX
    with open(os.path.join(OUT, "tables.tex"), "w") as f:
        f.write("% Generated by docs/rebuttal_q1_paired_analysis/paired_analysis.py from the 54 v2\n"
                "% LightEval JSONs. Do not hand-edit -- rerun the script.\n\n")
        for metric, lab, name in (("mean6", "tab:pairwise-mean6", "six-task commonsense mean"),
                                  ("mmlu", "tab:pairwise-mmlu", "MMLU macro-average")):
            f.write("\\begin{table*}[t]\n\\centering\n")
            f.write(f"\\caption{{Pairwise paired differences on the {name} at each epoch "
                    "boundary, for all $15$ pairs of settings"
                    + (" and, in the last block, each rebuttal setting against its comparator and "
                       "against \\textsc{Quality Base}" if new_pairs else "")
                    + ". Each cell is the mean over seeds "
                    "$42$, $43$ and $44$ of the per-seed difference (first setting minus second) "
                    "in accuracy points, with the standard deviation over the three per-seed "
                    "differences, followed by the number of seeds in which the first setting is "
                    "higher. The six settings at a given seed share initialization, so "
                    "differences are paired within seed. $0/3$ means the second setting is "
                    "higher in every seed.}\n")
            f.write(f"\\label{{{lab}}}\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{3.5pt}}\n"
                    "\\renewcommand{\\arraystretch}{1.15}\n\n\\begin{tabular}{lcccccc}\n\\toprule\n"
                    "& \\multicolumn{2}{c}{\\textbf{Epoch 1}} & \\multicolumn{2}{c}{\\textbf{Epoch 2}}"
                    " & \\multicolumn{2}{c}{\\textbf{Epoch 3}} \\\\\n"
                    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
                    "\\textbf{Comparison} & \\textbf{$\\Delta$} & \\textbf{Seeds $>0$} & "
                    "\\textbf{$\\Delta$} & \\textbf{Seeds $>0$} & \\textbf{$\\Delta$} & "
                    "\\textbf{Seeds $>0$} \\\\\n\\midrule\n")
            prev = None
            for a, b in P:
                grp = "new" if a in NEW_KEYS else ("base" if b == BASE else "strat")
                if prev and grp != prev:
                    f.write("\\midrule\n")
                prev = grp
                cells = " & ".join(tex_cell(paired(data, a, b, ep, metric)) for ep, _ in EPOCHS)
                f.write(f"{tex_pair(a, b)} & {cells} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")

        for metric, lab, name in (("mean6", "tab:ranks-mean6", "six-task commonsense mean"),
                                  ("mmlu", "tab:ranks-mmlu", "MMLU macro-average")):
            f.write("\\begin{table*}[t]\n\\centering\n")
            f.write(f"\\caption{{Per-seed rank of each setting on the {name} ($1$ is best of "
                    "six), at each epoch boundary, with the mean rank over the three seeds. "
                    "Bold marks a setting that holds the same rank in all three seeds.}\n")
            f.write(f"\\label{{{lab}}}\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{3.5pt}}\n"
                    "\\renewcommand{\\arraystretch}{1.15}\n\n\\begin{tabular}{lcccccccccccc}\n"
                    "\\toprule\n& \\multicolumn{4}{c}{\\textbf{Epoch 1}} & "
                    "\\multicolumn{4}{c}{\\textbf{Epoch 2}} & \\multicolumn{4}{c}{\\textbf{Epoch 3}}"
                    " \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\n"
                    "\\textbf{Setting}" + " & $42$ & $43$ & $44$ & \\textbf{Mean}" * 3
                    + " \\\\\n\\midrule\n")
            for k in KEYS:
                cells = []
                for ep, _ in EPOCHS:
                    r = [ranks[(metric, ep, s, k)] for s in SEEDS]
                    same = len(set(r)) == 1
                    cells += [f"$\\mathbf{{{x:g}}}$" if same else f"${x:g}$" for x in r]
                    cells.append(f"${f2(mean(r))}$")
                f.write(f"\\textsc{{{LABEL[k]}}} & " + " & ".join(cells) + " \\\\\n")
                if k == BASE:
                    f.write("\\midrule\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")

    # ---------------- console summary
    print(f"[step2] {n_total} comparisons; {len(strong)} supported (3/3, |mean|>SD), "
          f"{len(weak)} consistent but |mean|<=SD; ties in ranks: {len(ties)}")
    for title, tab in (("SUPPORTED: 3/3 and mean > SD", strong),
                       ("CONSISTENT ONLY: 3/3 and mean <= SD", weak)):
        print(f"\n== {title}")
        for r in tab:
            print(f"  {r[0]:<24} ep{r[1]} {r[2]:<12} {SHORT[r[3]]:>6} > {SHORT[r[4]]:<6} "
                  f"{r[7]:>6} ± {r[8]:<5} [{', '.join(r[9:12])}]" + ("  (min seed margin "
                  + r[12] + ")" if float(r[12]) < 0.05 else ""))


if __name__ == "__main__":
    main()
