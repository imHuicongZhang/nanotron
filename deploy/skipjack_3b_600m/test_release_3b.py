"""Publisher guards of release_3b.py: CPU only, no network, queues nothing.

    PYTHONPATH=<pytest dir> $P -m pytest -q -p no:cacheprovider deploy/skipjack_3b_600m/test_release_3b.py

Asserts:
  * the dataset visibility the publisher enforces is public (owner authorization 2026-10-30);
  * the upload breaker stops, with the exact error recorded, on the third 401/403 (job 1385487's storage-quota 403
    message), and on 429s that continue past RATE_LIMIT_MAX_S, but not on 429s with gaps or on unrelated warnings;
  * upload_large_folder gets a copy of TOP (it appends its default ignore patterns in place, which put '.git' into the
    final commit's paths in job 1389362);
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


def test_upload_does_not_mutate_top(tmp_path, monkeypatch):
    seen = {}

    class FakeApi:
        def upload_large_folder(self, ignore_patterns, **kw):
            ignore_patterns += ['.git', '.git/*']                  # what huggingface_hub 1.13 does
            seen['ignore'] = ignore_patterns

        def create_commit(self, repo_id, operations, **kw):
            seen['ops'] = [op.path_in_repo for op in operations]
            return type('C', (), {'oid': 'c' * 40})()

    for n in RL.TOP:
        (tmp_path / n).write_text('{}')
    monkeypatch.setattr(RL, 'STAGE', tmp_path)
    monkeypatch.setattr(RL, 'api', FakeApi)
    monkeypatch.setattr(RL, '_remote_files', lambda a, revision=None: ({}, 'r' * 40, RL.PRIVATE))
    monkeypatch.setattr(RL, '_expected', lambda: {n: {'bytes': 2} for n in RL.TOP})
    assert RL.st_upload()['ok']
    assert isinstance(RL.TOP, tuple) and seen['ops'] == list(RL.TOP) and '.git' in seen['ignore']


def test_card():
    man = {'settings': {'quality-base': {'n_parts': 36, 'assembled_rows': 17942518, 'assembled_train_tokens': 20000000299,
                                         'empty_text_documents': 0, 'tokenized_tokens': 20000000299,
                                         'tokenized_documents': 17942518}}}
    c = RL.card(man, {'commit': '0' * 40})
    assert 'license: other' in c and 'Private.' not in c
    for s in ('DCLM-RefinedWeb', 'Common Crawl', 'Qwen/Qwen2.5-7B-Instruct', 'Llama 2 Community License'):
        assert s in c


def test_metadata_update_compares_content_and_drops_stale_receipt(tmp_path, monkeypatch):
    """Republishing for a new code commit changes HANDOFF/README content but not necessarily their size: the top files
    are compared by git blob sha1 and recommitted, and the previous receipt is deleted in the same commit."""
    seen = {}
    for n in RL.TOP:
        (tmp_path / n).write_text(f'new {n} 0123456789abcdef')
    old = {n: {'bytes': (tmp_path / n).stat().st_size, 'lfs_sha256': None, 'blob_id': '0' * 40} for n in RL.TOP}
    old['receipts/UPLOAD_RECEIPT.json'] = {'bytes': 10, 'lfs_sha256': None, 'blob_id': '1' * 40}

    class FakeApi:
        def upload_large_folder(self, ignore_patterns, **kw):
            seen['upload_ignore'] = list(ignore_patterns)

        def create_commit(self, repo_id, operations, **kw):
            seen['ops'] = [(type(op).__name__, op.path_in_repo) for op in operations]
            return type('C', (), {'oid': 'c' * 40})()

    monkeypatch.setattr(RL, 'STAGE', tmp_path)
    monkeypatch.setattr(RL, 'api', FakeApi)
    monkeypatch.setattr(RL, '_remote_files', lambda a, revision=None: (old, 'r' * 40, RL.PRIVATE))
    assert RL.st_upload()['top_commit'] == 'c' * 40
    assert sorted(seen['ops']) == sorted([('CommitOperationAdd', n) for n in RL.TOP] +
                                         [('CommitOperationDelete', 'receipts/UPLOAD_RECEIPT.json')])
    same = {n: {**old[n], 'blob_id': RL.git_blob_sha1(tmp_path / n)} for n in RL.TOP}       # already published
    monkeypatch.setattr(RL, '_remote_files', lambda a, revision=None: (same, 'r' * 40, RL.PRIVATE))
    seen.clear()
    assert RL.st_upload()['top_commit'] is None and 'ops' not in seen


def test_remote_check_treats_own_metadata_as_replaceable(tmp_path, monkeypatch):
    data = {'tokenized/quality-base/00000_unshuffled.ds': {'bytes': 4, 'sha256': 'a' * 64}}
    remote = {'tokenized/quality-base/00000_unshuffled.ds': {'bytes': 4, 'lfs_sha256': 'a' * 64, 'blob_id': 'x'},
              **{n: {'bytes': 1, 'lfs_sha256': None, 'blob_id': 'y'} for n in RL.TOP},
              'receipts/UPLOAD_RECEIPT.json': {'bytes': 1, 'lfs_sha256': None, 'blob_id': 'z'}, '.gitattributes': {}}
    monkeypatch.setattr(RL, 'api', lambda: None)
    monkeypatch.setattr(RL, '_remote_files', lambda a, revision=None: (remote, 'r' * 40, RL.PRIVATE))
    monkeypatch.setattr(RL, '_expected', lambda: {**data, **{n: {'bytes': 2, 'sha256': 'b' * 64} for n in RL.TOP}})
    monkeypatch.setattr(RL, 'WORK', tmp_path)
    r = RL.st_remote_check()
    assert r['ok'] and r['reused'] == 1 and r['conflicts'] == [] and \
        r['metadata_to_replace'] == sorted([*RL.TOP, 'receipts/UPLOAD_RECEIPT.json'])
    remote['tokenized/quality-base/00000_unshuffled.ds']['lfs_sha256'] = 'c' * 64              # a data file differs
    assert RL.st_remote_check()['conflicts'] == ['tokenized/quality-base/00000_unshuffled.ds']
