"""Quality-Base (the prepared 4th setting) against the three live settings: CPU only, queues nothing.

    PYTHONPATH=<pytest dir> $P -m pytest -q -p no:cacheprovider deploy/skipjack_3b_600m/test_quality_base_3b.py
    $P deploy/skipjack_3b_600m/test_quality_base_3b.py --report <out.json>     # the four-setting comparison table

Asserts:
  * the live grid's 27 rendered configs (+ .env) are byte-identical to a fresh render (adding the 4th setting changed
    nothing the train8 lanes run), and the prepared quality-base configs are byte-identical to a fresh render;
  * for every segment kind, the quality-base config differs from each rewritten setting's config ONLY in the run name,
    the tokenized dataset folder and the checkpoint/resume paths (and the .env only in the W&B group and setting tag);
  * the scientific constants: 57 layers, original tokenizer, seed 42, 1024 x 2048 tokens/step, peak LR 4e-4, warmup
    1000, branch points 8583/17166/25749, finals 9537/19073/28610, trunk1 weights-only from the shared init, later
    segments and branches resume optimizer + LR state;
  * quality-base cannot be queued by the live tools (train8_3b plans exactly the 28 jobs of the three live settings);
  * every quality-base path is its own (no collision with another setting) and outside the 1.5B/100M namespaces.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'skipjack_1p5b'))
import generate_configs_3b as G  # noqa: E402
import registry_3b as R  # noqa: E402

QB = 'quality-base'
ALLOWED_YAML_DIFF = {'general.run', 'data_stages.0.data.dataset.dataset_folder.0', 'checkpoints.checkpoints_path',
                     'checkpoints.resume_checkpoint_path'}
ALLOWED_ENV_DIFF = {'WANDB_RUN_GROUP', 'WANDB_TAGS'}


def _flat(x, pre=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flat(v, f'{pre}{k}.')
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _flat(v, f'{pre}{i}.')
    else:
        yield pre[:-1], x


def _cfg(setting, kind):
    return yaml.safe_load((R.cfg_dir(setting) / f'{R.seg_name(setting, kind)}.yaml').read_text())


def _env(setting, kind):
    lines = (R.cfg_dir(setting) / f'{R.seg_name(setting, kind)}.env').read_text().splitlines()
    return dict(ln[len('export '):].split('=', 1) for ln in lines if ln.startswith('export '))


def compare() -> dict:
    """{other setting: {kind: {yaml key: [qb, other]}, env: {...}}} -- every difference between QB and each live setting."""
    out = {}
    for other in R.SETTINGS:
        rows = {}
        for k in R.KINDS:
            a, b = dict(_flat(_cfg(QB, k))), dict(_flat(_cfg(other, k)))
            ea, eb = _env(QB, k), _env(other, k)
            rows[k] = {'yaml': {key: [a.get(key), b.get(key)] for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)},
                       'env': {key: [ea.get(key), eb.get(key)] for key in sorted(set(ea) | set(eb)) if ea.get(key) != eb.get(key)}}
        out[other] = rows
    return out


def test_live_grid_unchanged(tmp_path):
    G.main(['--settings', *R.SETTINGS, '--out', str(tmp_path), '--layout', G.live_layout()])
    live = R.cfg_dir(R.SETTINGS[0])
    names = sorted(p.name for p in live.iterdir() if p.suffix in ('.yaml', '.env'))
    assert names == sorted(p.name for p in tmp_path.iterdir()) and len(names) == 2 * 27
    for n in names:
        assert (live / n).read_bytes() == (tmp_path / n).read_bytes(), n


def test_prepared_configs_are_current(tmp_path):
    G.main(['--settings', QB, '--out', str(tmp_path)])
    d = R.cfg_dir(QB)
    names = sorted(p.name for p in d.iterdir())
    assert names == sorted(p.name for p in tmp_path.iterdir()) and len(names) == 2 * len(R.KINDS)
    for n in names:
        assert (d / n).read_bytes() == (tmp_path / n).read_bytes(), n


def test_only_identifiers_differ():
    for other, rows in compare().items():
        for k, r in rows.items():
            assert set(r['yaml']) == ALLOWED_YAML_DIFF, (other, k, r['yaml'])
            assert set(r['env']) <= ALLOWED_ENV_DIFF, (other, k, r['env'])
            qb_tags, o_tags = (dict(t.split(':', 1) for t in v.split(',')) for v in r['env']['WANDB_TAGS'])
            assert {t for t in qb_tags if qb_tags[t] != o_tags.get(t)} == {'setting'}, (other, k)
            assert r['yaml']['data_stages.0.data.dataset.dataset_folder.0'][0] == str(R.TOKENIZED / QB)


def test_scientific_constants():
    assert R.HORIZONS == [9537, 19073, 28610] and R.BRANCH_POINTS == [8583, 17166, 25749]
    assert R.SEED == 42 and R.PEAK_LR == 4.0e-4 and R.WARMUP_STEPS == 1000 and R.TOKENS_PER_STEP == 1024 * 2048
    for k in R.KINDS:
        c = _cfg(QB, k)
        lrs, t, ck = c['optimizer']['learning_rate_scheduler'], c['tokens'], c['checkpoints']
        assert c['model']['model_config']['num_hidden_layers'] == 57 and c['general']['seed'] == 42
        assert c['tokenizer']['tokenizer_name_or_path'] == str(R.TOKENIZER)
        assert c['model']['dtype'] == 'bfloat16' and t['sequence_length'] == 2048
        assert t['micro_batch_size'] * t['batch_accumulation_per_replica'] * c['parallelism']['dp'] == 1024
        assert lrs['learning_rate'] == 4.0e-4 and lrs['lr_warmup_steps'] == 1000 and lrs['min_decay_lr'] == 0.0
        assert t['train_steps'] == R.END_STEP[k]
        if k.startswith('ep'):
            i = int(k[-1]) - 1
            assert (lrs['lr_decay_starting_step'], lrs['lr_decay_steps']) == (R.BRANCH_POINTS[i], R.HORIZONS[i] - R.BRANCH_POINTS[i])
            assert ck['resume_checkpoint_path'] == str(R.ckpt_dir(QB, 'trunk1') / str(R.BRANCH_POINTS[i]))
        else:
            assert (lrs['lr_decay_starting_step'], lrs['lr_decay_steps']) == (25749, 2861)
            assert t['train_steps'] <= 25749
        # trunk1 is weights-only from the shared init (segment_3b.sbatch loads optimizer + LR state once latest.txt > 0)
        assert ck['load_optimizer'] == ck['load_lr_scheduler'] == (k != 'trunk1')
    assert sorted(R.END_STEP[f'ep{i}'] for i in (1, 2, 3)) == R.HORIZONS


def test_not_submittable_by_live_tools():
    import train8_3b as T
    segs = T.load_segments()
    assert len(segs) == 28 and not any('quality_base' in n for n in segs)
    assert R.SETTINGS == ['quality-first', 'diversity-oriented', 'disagreement-aware']
    assert R.GRID_SETTINGS == R.SETTINGS + [QB] and QB not in R.EXCLUDED


def test_paths_are_own_namespace():
    from assert_invariants_3b import _paths
    mine = {p for k in R.KINDS for p in _paths(_cfg(QB, k))} - {str(R.TOKENIZER)}
    for p in mine:
        assert not any(p.startswith(f) for f in R.FORBIDDEN_PREFIXES), p
        assert '/quality-base' in p, p
    others = {p for s in R.SETTINGS for k in R.KINDS for p in _paths(_cfg(s, k))}
    assert not mine & others


def test_invariant_gates():
    for flag, n in (('--all', 27), ('--prepared', 9), ('--grid', 36)):
        r = subprocess.run([sys.executable, HERE / 'assert_invariants_3b.py', flag], capture_output=True, text=True)
        assert r.returncode == 0 and f'OK: {n} config(s)' in r.stdout, (flag, r.stdout, r.stderr)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', type=Path, required=True)
    rep = compare()
    summary = {o: sorted({key for r in rows.values() for key in r['yaml']} | {f'env:{key}' for r in rows.values() for key in r['env']})
               for o, rows in rep.items()}
    ap.parse_args().report.write_text(json.dumps({'differing_keys': summary, 'per_kind': rep}, indent=1))
    print(json.dumps(summary, indent=1))
