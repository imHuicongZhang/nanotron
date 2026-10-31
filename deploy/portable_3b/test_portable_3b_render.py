"""Portable rendering of the four-setting grid with unrelated paths, W&B values and deployments (CPU only).

    PYTHONPATH=<pytest dir> $PYTHON -m pytest -q -p no:cacheprovider deploy/portable_3b/test_portable_3b_render.py

Asserts, for two deployments unlike Skipjack's (dp 16 x mbs 4, dp 8 x mbs 8 / zero 0):
  * 36 configs = 4 settings (quality-base included) x 9 segments, 12 annealed finals, in plan.json and in the Slurm
    chain, with the science (science_3b.science_view) of every config IDENTICAL to the reviewed committed configs of
    configs/3B-rewrite-600m (only paths, W&B project and the deployment differ);
  * every path is under the operator's roots; no publisher path, W&B entity/project or Slurm account anywhere;
  * W&B: online, the operator's entity and project in every .env, general.project = WANDB_PROJECT;
  * branch sources trunk/8583, /17166, /25749 with optimizer + LR state; 16 separate output folders; finals at
    9537 / 19073 / 28610 = 20,000,538,624 / 39,998,980,096 / 59,999,518,720 tokens;
  * kys3b check passes on the render and catches a hand edit; render is idempotent, refuses a different render
    without --force, and refuses a deployment change once training recorded one (kys3b_deployment.json).
"""
from __future__ import annotations

import json
import re

import pytest
import yaml

from conftest import CODE, ENTITY, PROJECT, operator_dict, with_changes, write
import kys3b as K
import science_3b as S

COMMITTED = {s: CODE / 'configs' / '3B-rewrite-600m' / ('seed42_prepared' if s == 'quality-base' else 'seed42')
             for s in S.GRID_SETTINGS}
DEPLOYMENTS = {
    'dp16_mbs4_z1': {},
    'dp8_mbs8_z0': {'launch.nnodes': 1, 'deployment.data_parallel_size': 8, 'deployment.micro_batch_size': 8,
                    'deployment.zero_stage': 0, 'deployment.recompute_layer': False},
}


@pytest.fixture(params=sorted(DEPLOYMENTS))
def rendered(request, tmp_path):
    op = with_changes(operator_dict(tmp_path), DEPLOYMENTS[request.param])
    path = write(op, tmp_path / 'operator.yaml')
    site = K.Site(K.load_operator(path))
    man = K.render(site)
    return site, man, path, tmp_path


def _publisher_free(text: str) -> bool:
    return not re.search(r'(^|[\s\'"=:,\[])(/weka/|/home/jhu/|/scratch/jhu/)|imwytro|zhc-3b|bvandur1_blabscify', text)


def test_grid_is_four_settings_with_reviewed_science(rendered):
    site, man, _, _ = rendered
    ymls = sorted(site.grid_dir.glob('*.yaml'))
    assert len(ymls) == 36 and len(sorted(site.grid_dir.glob('*.env'))) == 36
    for s in S.GRID_SETTINGS:
        for k in S.KINDS:
            mine = yaml.safe_load(site.cfg_path(s, k).read_text())
            reviewed = yaml.safe_load((COMMITTED[s] / f'{S.seg_name(s, k)}.yaml').read_text())
            assert S.science_view(mine) == S.science_view(reviewed), S.seg_name(s, k)
    plan = json.loads((site.config_root / 'plan.json').read_text())
    assert plan['settings'] == ['quality-first', 'diversity-oriented', 'disagreement-aware', 'quality-base']
    assert plan['n_segments'] == 36 and plan['n_final_models'] == 12
    finals = sorted({x['final_tokens'] for x in plan['segments'] if x['final_model']})
    assert finals == [20_000_538_624, 39_998_980_096, 59_999_518_720]
    assert {x['setting'] for x in plan['segments'] if x['final_model']} == set(S.GRID_SETTINGS)
    order = plan['order']
    for x in plan['segments']:                                   # a valid execution order
        assert order.index(x['depends_on']) < order.index(x['run'])


def test_paths_wandb_and_deployment_are_the_operators(rendered, tmp_path):
    site, _, _, root = rendered
    roots = tuple(str(root / p) for p in ('site', 'code', 'python'))
    for p in site.config_root.rglob('*'):
        if p.is_file():
            assert _publisher_free(p.read_text()), p
    for y in site.grid_dir.glob('*.yaml'):
        c = yaml.safe_load(y.read_text())
        paths = [v for _, v in K._flat(c) if isinstance(v, str) and v.startswith('/')]
        assert paths and all(v.startswith(roots) for v in paths), (y.name, paths)
        env = dict(ln[len('export '):].split('=', 1) for ln in y.with_suffix('.env').read_text().splitlines()
                   if ln.startswith('export '))
        assert (env['WANDB_ENTITY'], env['WANDB_PROJECT'], env['WANDB_MODE']) == (ENTITY, PROJECT, 'online')
        assert c['general']['project'] == PROJECT and env['WANDB_DIR'] == str(site.wandb_dir)
        t = c['tokens']
        assert (t['micro_batch_size'], t['batch_accumulation_per_replica'], c['parallelism']['dp']) == \
            (site.mbs, site.accum, site.dp)
        assert c['optimizer']['zero_stage'] == site.zero and c['parallelism']['recompute_layer'] == site.recompute


def test_branches_and_namespaces(rendered):
    site, _, _, _ = rendered
    outs = set()
    for s in S.GRID_SETTINGS:
        trunk = yaml.safe_load(site.cfg_path(s, 'trunk1').read_text())['checkpoints']['checkpoints_path']
        assert all(yaml.safe_load(site.cfg_path(s, k).read_text())['checkpoints']['checkpoints_path'] == trunk
                   for k in S.TRUNK_KINDS)
        outs.add(trunk)
        for k, b, h in zip(S.BRANCH_KINDS, (8583, 17166, 25749), (9537, 19073, 28610)):
            c = yaml.safe_load(site.cfg_path(s, k).read_text())
            ck, lrs = c['checkpoints'], c['optimizer']['learning_rate_scheduler']
            assert ck['resume_checkpoint_path'] == f'{trunk}/{b}' and ck['load_optimizer'] and ck['load_lr_scheduler']
            assert c['tokens']['train_steps'] == h and (lrs['lr_decay_starting_step'], lrs['lr_decay_steps']) == (b, h - b)
            assert not ck['checkpoints_path'].startswith(trunk) and ck['checkpoints_path'] not in outs
            outs.add(ck['checkpoints_path'])
        t1 = yaml.safe_load(site.cfg_path(s, 'trunk1').read_text())['checkpoints']
        assert (t1['load_optimizer'], t1['load_lr_scheduler']) == (False, False)        # weights-only from the init
        for k in S.TRUNK_KINDS[1:]:
            ck = yaml.safe_load(site.cfg_path(s, k).read_text())['checkpoints']
            assert ck['load_optimizer'] and ck['load_lr_scheduler'] and ck['resume_checkpoint_path'] == trunk
    assert len(outs) == 16


def test_slurm_chain_covers_all_four_settings(rendered):
    site, _, _, _ = rendered
    sub = (site.config_root / 'slurm' / 'submit_grid.sh').read_text()
    for s in S.GRID_SETTINGS:
        for k in S.KINDS:
            dep = '$INIT' if S.DEPENDS[k] is None else f'${S.slug(s).upper()}_{S.DEPENDS[k].upper()}'
            assert f'--dependency=afterok:{dep} "$D/{S.seg_name(s, k)}.sbatch"' in sub, S.seg_name(s, k)
        sb = (site.config_root / 'slurm' / f'{S.seg_name(s, "ep3")}.sbatch').read_text()
        assert f'--nodes={site.op["launch"]["nnodes"]}' in sb and '--gpus-per-node=8' in sb and '--account=marc-acct' in sb
        assert f'run_segment.sh {site.config_root}/site.env {S.seg_name(s, "ep3")}' in sb
    assert sub.count('sbatch --parsable') == 37


def test_check_passes_and_catches_edits(rendered):
    site, _, _, _ = rendered
    rep = K.check(site, corpus=False, hash_binaries=False)
    assert rep['ok'], rep['problems']
    p = site.cfg_path('quality-base', 'ep2')
    p.write_text(p.read_text().replace('learning_rate: 0.0004', 'learning_rate: 0.0003'))
    rep = K.check(site, corpus=False, hash_binaries=False)
    assert not rep['ok'] and any('3b_quality_base_seed42_ep2.yaml' in e for e in rep['problems'])


def test_rerender_rules(rendered):
    site, man, path, root = rendered
    assert K.render(site)['files'] == man['files']                  # idempotent
    op2 = with_changes(yaml.safe_load(path.read_text()), {'wandb.WANDB_PROJECT': 'another-project'})
    with pytest.raises(SystemExit, match='rendered from a different operator config'):
        K.render(K.Site(K.validate_operator(op2)))
    site.ckpt_root.mkdir(parents=True, exist_ok=True)
    (site.ckpt_root / 'kys3b_deployment.json').write_text(json.dumps({'deployment': site.deployment()}))
    dp = 4 if site.dp != 4 else 8
    op3 = with_changes(yaml.safe_load(path.read_text()), {'launch.nnodes': 1, 'launch.nproc_per_node': dp,
                                                          'scheduler.slurm.gpus_per_node': dp,
                                                          'deployment.data_parallel_size': dp})
    with pytest.raises(SystemExit, match='Checkpoints of a chain must keep one deployment'):
        K.render(K.Site(K.validate_operator(op3)), force=True)


def test_skipjack_configs_are_untouched_by_the_portable_builder():
    """The committed (reviewed) configs are what generate_configs_3b renders through the same science_3b builder."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(CODE / 'deploy/skipjack_3b_600m/assert_invariants_3b.py'), '--grid'],
                       capture_output=True, text=True, env={k: v for k, v in __import__('os').environ.items()
                                                            if not k.startswith('KYS3B_')})
    assert r.returncode == 0 and 'OK: 36 config(s)' in r.stdout, r.stdout + r.stderr
