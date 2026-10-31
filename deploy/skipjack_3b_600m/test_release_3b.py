"""Publisher guards of release_3b.py: CPU only, no network, queues nothing.

    PYTHONPATH=<pytest dir> $P -m pytest -q -p no:cacheprovider deploy/skipjack_3b_600m/test_release_3b.py

Asserts:
  * the dataset visibility the publisher enforces is public (owner authorization 2026-10-30);
  * the upload breaker stops, with the exact error recorded, on the third 401/403 (job 1385487's storage-quota 403
    message), and on 429s that continue past RATE_LIMIT_MAX_S, but not on 429s with gaps or on unrelated warnings;
  * the dataset card states the visibility, the upstream provenance, and makes no license claim of its own.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import release_3b as RL  # noqa: E402

QUOTA_403 = ('Failed to preupload LFS: (Request ID: Root=1-6ae53731-22225e9f342184af6aaf2fa8)\n\n403 Forbidden: Private '
             'storage limit reached for org blab-jhu, please upgrade your plan to increase your private storage limit.')
RATE_429 = 'HTTP Error 429 thrown while requesting POST https://huggingface.co/x\nRate limited. Waiting 294.0s before retry'


class _Exit(Exception):
    pass


@pytest.fixture
def breaker(tmp_path, monkeypatch):
    monkeypatch.setattr(RL, 'REC', tmp_path)
    monkeypatch.setattr(RL.os, '_exit', lambda code: (_ for _ in ()).throw(_Exit(code)))
    clock = [1000.0]
    monkeypatch.setattr(RL.time, 'time', lambda: clock[0])
    b = RL._Breaker()
    b.clock = clock
    return b


def _emit(b, msg, level=logging.ERROR):
    b.emit(logging.LogRecord('huggingface_hub', level, __file__, 1, msg, None, None))


def test_public():
    assert RL.PRIVATE is False and RL.VISIBILITY == 'public'


def test_stops_on_repeated_quota_error(breaker, tmp_path):
    for _ in range(RL.FATAL_MAX - 1):
        _emit(breaker, QUOTA_403)
    with pytest.raises(_Exit):
        _emit(breaker, QUOTA_403)
    rec = json.loads((tmp_path / 'upload.json').read_text())
    assert rec['ok'] is False and rec['error'] == 'quota or permission error'
    assert any('Private storage limit reached for org blab-jhu' in ln for ln in rec['message'])


def test_rate_limit_streak(breaker, tmp_path):
    for _ in range(10):                                   # 429s separated by gaps longer than RATE_LIMIT_GAP_S
        _emit(breaker, RATE_429, logging.WARNING)
        breaker.clock[0] += RL.RATE_LIMIT_GAP_S + 1
    _emit(breaker, 'Failed to commit 28 files at once. Will retry with less files in next batch.', logging.WARNING)
    assert not (tmp_path / 'upload.json').exists()
    with pytest.raises(_Exit):                            # then one continuous streak longer than RATE_LIMIT_MAX_S
        while True:
            _emit(breaker, RATE_429, logging.WARNING)
            breaker.clock[0] += 300
    assert 'rate limited' in json.loads((tmp_path / 'upload.json').read_text())['error']


def test_card():
    man = {'settings': {'quality-base': {'n_parts': 36, 'assembled_rows': 17942518, 'assembled_train_tokens': 20000000299,
                                         'empty_text_documents': 0, 'tokenized_tokens': 20000000299,
                                         'tokenized_documents': 17942518}}}
    c = RL.card(man, {'commit': '0' * 40})
    assert 'license: other' in c and 'Private.' not in c
    for s in ('DCLM-RefinedWeb', 'Common Crawl', 'Qwen/Qwen2.5-7B-Instruct', 'Llama 2 Community License'):
        assert s in c
