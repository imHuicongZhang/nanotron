"""The single list of raw-selected baseline settings and the segment schedule they share.

Every raw-baseline tool imports this module instead of carrying its own list: generate_configs.py,
render_config.py, assert_invariants.py, fill_placeholders.py, render_placeholders.py, plan_submit.py.
Adding a setting here (plus its corpus in blab-jhu/KYS-Pre-Rewritten and its templates) is the whole change.

Two families (configs/1.5B-baseline/WORKFLOW_RAW_BASELINES.md):

  strategy_linked  shared 5B anchor + a seed-42 uniform random 5B of the unique source documents of the
                   counterpart's final rewritten half (tools/kys_raw/build_raw_sources.py): equal token budget,
                   not identical documents; raw_rewire_inspired is conditional on REWIRE's post-rewrite filter
                   (configs/1.5B-baseline/reports/strategy_linked_audit/)
  global_top10b    the entire ~10B corpus is one global Top-10B selection over the Quality-Base universe;
                   NO anchor. Compared against the existing fastText quality_base arm
                   (configs/1.5B-baseline/reports/GLOBAL_TOP10B_SELECTION_REPORT.md).
  lambda_grid      a REWRITTEN arm of the Disagreement-Aware lambda grid run on the same schedule as the raw
                   baselines: shared 5B anchor + the 5B rewritten half selected by u = q + lambda*sqrt(var3).
                   It ships TOKENIZED (not as raw_text), so its expected total is recorded here
                   (`expected_corpus` = (tokens, shards)) instead of in KYS-Pre-Rewritten's manifest.json.
                   configs/1.5B-baseline/RUNBOOK.md, section "signal-disagreement-lambda0".
"""
from __future__ import annotations

RAW_SETTINGS = {
    'raw_diversity_oriented': {'family': 'strategy_linked', 'contains_anchor': True, 'comparator': 'diversity_oriented'},
    'raw_disagreement_aware': {'family': 'strategy_linked', 'contains_anchor': True, 'comparator': 'disagreement_aware'},
    'raw_random':             {'family': 'strategy_linked', 'contains_anchor': True, 'comparator': 'wrap_inspired'},
    'raw_rewire_inspired':    {'family': 'strategy_linked', 'contains_anchor': True, 'comparator': 'rewire_inspired'},
    'raw_top10b_fineweb_edu': {'family': 'global_top10b', 'contains_anchor': False, 'comparator': 'quality_base'},
    'raw_top10b_modernbert':  {'family': 'global_top10b', 'contains_anchor': False, 'comparator': 'quality_base'},
    'raw_top10b_consensus':   {'family': 'global_top10b', 'contains_anchor': False, 'comparator': 'quality_base'},
    # lambda = 0 (pure quality, no disagreement bonus) arm of the Disagreement-Aware grid; comparator = the
    # published lambda = 0.5 arm. Checked 2026-11-06: anchor identical to KYS-Pre-Rewritten's 4,120,164 ids,
    # order = pp_io.bucketed_shuffle(seed 42), 26 shards of datatrove 0.5.0 output (one per shuffled/ file).
    # 10,000,002,556 = the assembly budget 10,000,002,558 minus 2: two rewrites have empty text and datatrove
    # writes no document (so no </s>) for them (shuffled/ part_00020 and part_00023, one each).
    'signal-disagreement-lambda0': {'family': 'lambda_grid', 'contains_anchor': True,
                                    'comparator': 'disagreement_aware',
                                    'expected_corpus': (10_000_002_556, 26)},
}
SETTING_NAMES = list(RAW_SETTINGS)

# Segment schedule (values from tools/generate_configs.py, which owns them).
KINDS = ['trunk1', 'trunk2', 'trunk3', 'ep1', 'ep2', 'ep3']
DEPENDS = {'trunk1': None, 'trunk2': 'trunk1', 'trunk3': 'trunk2', 'ep1': 'trunk1', 'ep2': 'trunk2', 'ep3': 'trunk3'}
FIRST_STEP = {'trunk1': 0, 'trunk2': 4292, 'trunk3': 8583, 'ep1': 4292, 'ep2': 8583, 'ep3': 12875}
SEEDS = [42, 43, 44]


def template_names(seed: int, settings=None) -> list[str]:
    """The exact template file names expected for one seed."""
    return sorted(f'{s}_seed{seed}_{k}.yaml' for s in (settings or SETTING_NAMES) for k in KINDS)


def parse_settings(arg: str | None) -> list[str]:
    """Validate a comma-separated --settings value (None -> all settings, registry order)."""
    if not arg:
        return list(SETTING_NAMES)
    out = [s for s in arg.split(',') if s]
    bad = sorted(set(out) - set(SETTING_NAMES))
    if bad:
        raise SystemExit(f'unknown settings {bad}; known: {",".join(SETTING_NAMES)}')
    return out
