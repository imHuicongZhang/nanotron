"""The single list of raw-selected baseline settings and the segment schedule they share.

Every raw-baseline tool imports this module instead of carrying its own list: generate_configs.py,
render_config.py, assert_invariants.py, fill_placeholders.py, render_placeholders.py, plan_submit.py.
Adding a setting here (plus its corpus in blab-jhu/KYS-Pre-Rewritten and its templates) is the whole change.

Two families (configs/1.5B-baseline/WORKFLOW_RAW_BASELINES.md):

  strategy_linked  shared 5B anchor + a 5B raw strategy half linked to one rewritten arm (published earlier;
                   build code tools/kys_raw/build_raw_sources.py; interpretation under separate review)
  global_top10b    the entire ~10B corpus is one global Top-10B selection over the Quality-Base universe;
                   NO anchor. Compared against the existing fastText quality_base arm
                   (configs/1.5B-baseline/reports/GLOBAL_TOP10B_SELECTION_REPORT.md).
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
