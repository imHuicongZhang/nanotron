"""The owner-approved .gitattributes exception of release_raw_3b.py (2026-11-13): CPU only, no network.

    PYTHONPATH=<pytest dir> $P -m pytest -q -p no:cacheprovider deploy/skipjack_3b_600m/test_release_raw_3b.py

The Hub appends one `<path> filter=lfs diff=lfs merge=lfs -text` line per new binary file to .gitattributes. That change
is accepted only if the old content is an exact byte prefix and every appended rule names exactly one new manifest file
that is not (and does not contain) an original release file; everything else is rejected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import release_raw_3b as RR  # noqa: E402

OLD = (b'*.parquet filter=lfs diff=lfs merge=lfs -text\n'
       b'# comment\n'
       b'tokenized/diversity_oriented/00000_unshuffled.ds filter=lfs diff=lfs merge=lfs -text\n')
LFS = ' filter=lfs diff=lfs merge=lfs -text\n'
NEW_FILES = {'tokenized/raw_diversity_oriented/00000_unshuffled.ds',
             'tokenized/raw_diversity_oriented/00000_unshuffled.ds.index',
             'tokenized/raw_disagreement_aware/00031_unshuffled.ds',
             'corpora/raw_diversity_oriented/part_00000.parquet'}
ORIGINAL = {'.gitattributes', 'README.md', 'RELEASE_MANIFEST.json', 'tokenized/diversity_oriented/00000_unshuffled.ds',
            'tokenized/diversity_oriented/00000_unshuffled.ds.index', 'corpora/diversity_oriented/part_00000.parquet'}


def check(added: str, old: bytes = OLD, new_files=NEW_FILES, original=ORIGINAL):
    return RR.gitattributes_append_check(old, old + added.encode(), new_files, original)


def rule(path: str) -> str:
    return path + LFS


# ---- allowed ------------------------------------------------------------------------------------------------------
def test_hub_append_of_new_manifest_files_is_allowed():
    added = ''.join(rule(p) for p in sorted(NEW_FILES) if p.startswith('tokenized'))
    r = check(added)
    assert r['ok'], r['problems']
    assert r['appended_rules'] == 3 and r['appended_paths'] == sorted(p for p in NEW_FILES if p.startswith('tokenized'))
    assert r['old_bytes'] == len(OLD) and r['new_bytes'] == len(OLD) + len(added)


def test_identical_content_is_allowed_with_zero_rules():
    r = check('')
    assert r['ok'] and r['appended_rules'] == 0


# ---- rejected: the original content changed --------------------------------------------------------------------------
@pytest.mark.parametrize('new', [
    OLD.replace(b'# comment\n', b''),                                                   # a line removed
    OLD.replace(b'-text\n', b'text\n', 1),                                              # an attribute edited
    b'*.ds filter=lfs diff=lfs merge=lfs -text\n' + OLD,                                # prepended
    OLD[:-1] + b' \n',                                                                  # last byte changed
])
def test_old_content_not_an_exact_prefix_is_rejected(new):
    r = RR.gitattributes_append_check(OLD, new + rule('tokenized/raw_diversity_oriented/00000_unshuffled.ds').encode(),
                                      NEW_FILES, ORIGINAL)
    assert not r['ok'] and 'exact byte prefix' in r['problems'][0]


def test_append_that_continues_an_unterminated_last_line_is_rejected():
    r = check(rule('tokenized/raw_diversity_oriented/00000_unshuffled.ds'), old=OLD.rstrip(b'\n'))
    assert not r['ok'] and 'newline' in r['problems'][0]


# ---- rejected: broad or out-of-scope appended rules -------------------------------------------------------------------
@pytest.mark.parametrize('line', [
    '*.ds' + LFS,                                                      # glob, applies to original shards
    'tokenized/**' + LFS,                                              # recursive glob
    'tokenized/raw_diversity_oriented/*' + LFS,                        # glob inside the new dir
    'tokenized/raw_diversity_oriented/0000[0-9]_unshuffled.ds' + LFS,  # character class
    '00000_unshuffled.ds' + LFS,                                       # no '/': matches at every level
    '/tokenized/raw_diversity_oriented/00000_unshuffled.ds' + LFS,     # not the Hub's form
    '!tokenized/raw_diversity_oriented/00000_unshuffled.ds' + LFS,     # negation
    '"tokenized/raw_diversity_oriented/00000_unshuffled.ds"' + LFS,    # quoted pattern
    '[attr]binary -diff -merge -text\n',                               # macro definition
    '# tokenized/raw_diversity_oriented/00000_unshuffled.ds\n',        # comment
    '\n',                                                              # blank line
    'tokenized/raw_diversity_oriented/../diversity_oriented/00000_unshuffled.ds' + LFS,
])
def test_broad_or_malformed_patterns_are_rejected(line):
    r = check(rule('tokenized/raw_diversity_oriented/00000_unshuffled.ds.index') + line)
    assert not r['ok'] and len(r['problems']) == 1 and 'appended line 2' in r['problems'][0]


@pytest.mark.parametrize('line, why', [
    ('tokenized/raw_diversity_oriented/00099_unshuffled.ds' + LFS, 'not a file of the raw-baseline manifest'),
    ('tokenized/diversity_oriented/00000_unshuffled.ds' + LFS, 'not a file of the raw-baseline manifest'),
    ('tokenized/raw_diversity_oriented/00000_unshuffled.ds filter=lfs diff=lfs merge=lfs\n', 'attributes are not'),
    ('tokenized/raw_diversity_oriented/00000_unshuffled.ds -text\n', 'attributes are not'),
    ('tokenized/raw_diversity_oriented/00000_unshuffled.ds filter=lfs diff=lfs merge=lfs -text eol=lf\n',
     'attributes are not'),
    ('tokenized/raw_diversity_oriented/00000_unshuffled.ds  filter=lfs diff=lfs merge=lfs -text\n', 'attributes are not'),
])
def test_out_of_scope_paths_or_attributes_are_rejected(line, why):
    r = check(line)
    assert not r['ok'] and why in r['problems'][0]


def test_rule_on_an_original_file_is_rejected_even_if_listed_as_new():
    p = 'tokenized/diversity_oriented/00000_unshuffled.ds'
    r = check(rule(p), new_files=NEW_FILES | {p})
    assert not r['ok'] and 'original release file' in r['problems'][0]


def test_rule_on_a_directory_holding_original_files_is_rejected():
    d = 'tokenized/diversity_oriented'
    r = check(rule(d), new_files=NEW_FILES | {d})
    assert not r['ok'] and 'original release file' in r['problems'][0]


def test_unterminated_or_duplicate_or_non_ascii_appends_are_rejected():
    p = 'tokenized/raw_diversity_oriented/00000_unshuffled.ds'
    assert 'does not end in a newline' in ' '.join(check(rule(p).rstrip('\n'))['problems'])
    assert 'duplicate' in ' '.join(check(rule(p) + rule(p))['problems'])
    r = RR.gitattributes_append_check(OLD, OLD + 'tokenized/é.ds'.encode() + LFS.encode(), NEW_FILES, ORIGINAL)
    assert not r['ok'] and 'ASCII' in r['problems'][0]


def test_one_bad_rule_among_valid_ones_fails_the_whole_check():
    good = ''.join(rule(p) for p in sorted(NEW_FILES) if p.startswith('tokenized'))
    r = check(good + '*.index' + LFS)
    assert not r['ok'] and r['appended_rules'] == 3 and len(r['problems']) == 1
