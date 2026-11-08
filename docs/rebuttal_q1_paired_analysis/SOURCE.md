# Source of this folder

A copy, for remote versioning, of `docs/rebuttal_q1_paired_analysis/` in the local-only repository
`projects/rewrite` (no git remote), taken with `git archive` at commit **`c639080`** ("paired_analysis.py: load
kys-eval results for the three rebuttal settings"; its parent `ecf7d74` holds the 2026-11-06 version). The 15
files are byte-identical to that commit. Copied 2026-11-08.

The canonical copy stays in `projects/rewrite`; edit there and re-copy. `paired_analysis.py` resolves its inputs
relative to that location (`08_evaluation/04_results_1p5_v2/downstream`, `08_evaluation/02_code/mmlu_categories.py`)
and reads the paper's appendix LaTeX from an absolute path on the originating cluster, so it runs only there.
Rebuttal settings: `python3 paired_analysis.py --kys-results <dir> --out-dir <dir>` (README §6). Identity test:
`docs/prelaunch_audit_2026-11-08/analysis_gaps/`.
