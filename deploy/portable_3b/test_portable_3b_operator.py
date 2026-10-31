"""Operator config validation and deployment accounting (CPU only, no network, nothing queued).

    PYTHONPATH=<pytest dir> $PYTHON -m pytest -q -p no:cacheprovider deploy/portable_3b/test_portable_3b_operator.py

Asserts:
  * the shipped template is refused with one actionable error per <<placeholder>> (20 of them), nothing defaulted;
  * missing / unknown / wrongly typed keys, relative paths, the publisher's paths, W&B entity and Slurm account are
    refused, naming the key;
  * micro_batch_size x accumulation x data_parallel_size = 1024 exactly: accumulation is derived as an integer for
    every valid combination, and combinations that do not divide 1024 or whose launch size != dp are refused;
  * tensor / pipeline parallelism stay 1 and no GPU model, count or site value is defaulted anywhere.
"""
from __future__ import annotations

import pytest
import yaml

from conftest import DELETE, HERE, operator_dict, with_changes, write
import kys3b as K
import science_3b as S


def problems(op) -> list[str]:
    with pytest.raises(K.OperatorError) as e:
        K.validate_operator(op)
    return e.value.problems


def test_template_refused_with_every_placeholder_named():
    tmpl = yaml.safe_load((HERE / 'operator.template.yaml').read_text())
    probs = problems(tmpl)
    ph = [p for p in probs if 'still the placeholder' in p]
    assert len(ph) == 20, probs
    for key in ('paths.CODE_ROOT', 'paths.DATA_ROOT', 'paths.CONFIG_ROOT', 'paths.INIT_ROOT', 'paths.CHECKPOINT_ROOT',
                'paths.LOG_ROOT', 'paths.REPORT_ROOT', 'paths.WANDB_DIR', 'environment.PYTHON', 'wandb.WANDB_ENTITY',
                'wandb.WANDB_PROJECT', 'deployment.data_parallel_size', 'deployment.micro_batch_size',
                'launch.nnodes', 'launch.nproc_per_node', 'launch.rdzv_endpoint', 'scheduler.type'):
        assert any(p.startswith(key + ':') for p in ph), key


def test_valid_operator_accepted(operator):
    op, _ = operator
    assert K.validate_operator(op) is op
    assert K.Site(op).accum == 16


@pytest.mark.parametrize('changes, needle', [
    ({'paths.DATA_ROOT': DELETE}, 'paths.DATA_ROOT: missing'),
    ({'paths.DATA_ROOT': 'relative/data'}, 'paths.DATA_ROOT: must be an absolute path'),
    ({'paths.CHECKPOINT_ROOT': '/weka/projects/bvandur1/zhuicon1/checkpoints/x'}, "publisher's (Skipjack) storage"),
    ({'paths.LOG_ROOT': '/home/jhu/zhuicon1/logs'}, "publisher's (Skipjack) storage"),
    ({'paths.EXTRA': '/x'}, 'paths.EXTRA: unknown key'),
    ({'wandb.WANDB_ENTITY': 'imwytro-johns-hopkins-university'}, "publisher's W&B entity"),
    ({'wandb.WANDB_PROJECT': 'has space'}, 'wandb.WANDB_PROJECT: must be a W&B name'),
    ({'deployment.micro_batch_size': '4'}, 'deployment.micro_batch_size: must be a positive integer'),
    ({'deployment.zero_stage': 2}, 'deployment.zero_stage: must be 0 or 1'),
    ({'deployment.recompute_layer': 'yes'}, 'deployment.recompute_layer: must be true or false'),
    ({'deployment.tensor_parallel_size': 2}, 'deployment.tensor_parallel_size: unknown key'),
    ({'launch.rdzv_endpoint': 'host:29500; rm -rf /'}, 'launch.rdzv_endpoint: must be a string without spaces'),
    ({'launch.rdzv_id': '$SLURM_JOB_ID'}, 'only ${NAME} references'),
    ({'scheduler.type': 'pbs'}, 'scheduler.type: must be one of slurm, none'),
    ({'scheduler.slurm.account': 'bvandur1_blabscify'}, "publisher's (Skipjack) allocation"),
    ({'scheduler.slurm.time': '48h'}, 'use [D-]HH:MM:SS'),
    ({'scheduler.slurm.gpus_per_node': 4}, 'gpus_per_node 4 != launch.nproc_per_node 8'),
    ({'scheduler.slurm': DELETE}, 'scheduler.slurm: missing or not a mapping'),
    ({'environment.PYTHON': '/nonexistent/python'}, 'not an executable'),
])
def test_refusals_name_the_key(operator, changes, needle):
    op, _ = operator
    probs = problems(with_changes(op, changes))
    assert any(needle in p for p in probs), probs


def test_scheduler_none_needs_no_slurm_block(operator):
    op, _ = operator
    K.validate_operator(with_changes(op, {'scheduler.type': 'none', 'scheduler.slurm': DELETE}))
    assert any('must be absent or null' in p for p in problems(with_changes(op, {'scheduler.type': 'none'})))


@pytest.mark.parametrize('nnodes, nproc, dp, mbs, accum', [
    (1, 4, 4, 16, 16), (1, 8, 8, 8, 16), (2, 8, 16, 4, 16), (4, 8, 32, 2, 16), (8, 8, 64, 16, 1), (1, 1, 1, 1, 1024),
    (16, 8, 128, 8, 1), (1, 2, 2, 32, 16)])
def test_accumulation_is_exact(operator, nnodes, nproc, dp, mbs, accum):
    op, _ = operator
    op = with_changes(op, {'launch.nnodes': nnodes, 'launch.nproc_per_node': nproc, 'deployment.data_parallel_size': dp,
                           'deployment.micro_batch_size': mbs, 'scheduler.slurm.gpus_per_node': nproc})
    site = K.Site(K.validate_operator(op))
    assert site.accum == accum and mbs * site.accum * dp == S.GLOBAL_BATCH_SEQ
    c = site.config('quality-base', 'trunk1')
    t = c['tokens']
    assert (t['micro_batch_size'], t['batch_accumulation_per_replica'], c['parallelism']['dp']) == (mbs, accum, dp)
    assert t['micro_batch_size'] * t['batch_accumulation_per_replica'] * c['parallelism']['dp'] * t['sequence_length'] \
        == S.TOKENS_PER_STEP == 2_097_152
    assert (c['parallelism']['tp'], c['parallelism']['pp']) == (1, 1)


@pytest.mark.parametrize('nnodes, nproc, dp, mbs, needle', [
    (1, 6, 6, 4, 'does not divide the global batch'), (3, 8, 24, 1, 'does not divide the global batch'),
    (1, 8, 8, 256, 'does not divide the global batch'), (2, 8, 8, 4, 'nnodes 2 x nproc_per_node 8 = 16 processes'),
    (1, 8, 16, 4, 'nnodes 1 x nproc_per_node 8 = 8 processes')])
def test_incompatible_deployments_refused(operator, nnodes, nproc, dp, mbs, needle):
    op, _ = operator
    op = with_changes(op, {'launch.nnodes': nnodes, 'launch.nproc_per_node': nproc, 'deployment.data_parallel_size': dp,
                           'deployment.micro_batch_size': mbs, 'scheduler.slurm.gpus_per_node': nproc})
    assert any(needle in p for p in problems(op)), problems(op)


def test_no_site_defaults_in_portable_code():
    """The portable package names no GPU model/count, account, partition, W&B identity or publisher path as a value
    (the publisher strings appear only in kys3b.PUBLISHER, the refusal list)."""
    import re
    for p in sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.sh')) + [HERE / 'operator.template.yaml']:
        if p.name.startswith('test_') or p.name == 'conftest.py':
            continue
        text = p.read_text()
        if p.name == 'kys3b.py':
            text = re.sub(r"PUBLISHER = \{.*?\n\}\n", '', text, flags=re.S)
        for bad in ('/weka/', 'bvandur1', 'zhuicon1', 'imwytro', 'zhc-3b', 'h100', 'H100', 'gpu:h100', 'jhu2'):
            assert bad not in text, f'{p.name} contains {bad!r}'
