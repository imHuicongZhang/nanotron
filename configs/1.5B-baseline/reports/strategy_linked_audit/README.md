# Strategy-linked raw baselines: provenance audit and evidence

- [`RAW_SELECTED_BASELINES_PROVENANCE.md`](RAW_SELECTED_BASELINES_PROVENANCE.md): the audit (verdicts, exact
  construction, what each raw/rewritten comparison supports). Byte-identical to the canonical copy
  `docs/RAW_SELECTED_BASELINES_PROVENANCE.md` in the selection project (`projects/rewrite`, commit `70195a0`;
  that repository has no remote, so it is published here).
- `15_kys_raw_audit/`: the audit scripts and their JSON outputs, as committed in the selection project;
  `run_records_2026-09-28/` holds the re-verification run records (audit re-run JSON, logs, the consumer
  tokenization's `.ds.metadata` per shard, the Hub publication record).
- `cited_code/`: the selection, rewriting and post-processing scripts the report cites, copied at the same relative
  paths (`cited_code/04_select/select_10b.py` = `04_select/select_10b.py` in the report). They are the as-run
  files; their sha256 prefixes match the report's §9. `06_lambda_grid/lambda_grid.npz` (470 MB) is not copied; it is
  cited by sha256 (`3ef427a5525a678e…`).
- `SHA256SUMS`: sha256 of every file in this folder.

The audit scripts read data locations on the preparation cluster (`/projects/bvandur1/...`); their inputs are the
published datasets (`blab-jhu/KYS-DCLM-Refinedweb-100M-Scored@dcbbc360`, `wytro/Know-Your-Sources@9e5ff241`,
`blab-jhu/KYS-Pre-Rewritten`) and the raw 100M DCLM-RefinedWeb pool.
