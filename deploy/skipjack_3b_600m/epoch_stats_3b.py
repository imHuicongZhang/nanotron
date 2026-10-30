#!/usr/bin/env python
"""Exact data geometry of one tokenized 3B corpus, as nanotron will read it (CPU only).

    python deploy/skipjack_3b_600m/epoch_stats_3b.py --setting diversity-oriented [--json out.json]

nanotron's Nanoset reads non-overlapping (seq_len + 1) = 2049-token windows from each .ds file (a file's tail
shorter than 2049 tokens is never read), permutes all windows of one pass once with the data seed, and repeats
that same permutation every pass, truncated at train_steps x 1024 samples (src/nanotron/data/nanoset.py
build_nanoset_index; datatrove DatatroveFileDataset). So per pass it trains on W windows = W x 2048 predicted tokens
while consuming W x 2049 corpus tokens; the remainder up to the corpus total is the dropped per-file tails.
This prints W, the dropped tail tokens, steps per pass, and the passes and trained tokens at each horizon.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry_3b as R  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--setting', choices=R.GRID_SETTINGS, required=True)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args(argv)
    folder = R.TOKENIZED / args.setting
    win = R.SEQ_LEN + 1
    windows = tail = total = 0
    for meta in sorted(folder.glob('*.ds.metadata')):
        n = int(meta.read_text().splitlines()[1])
        total += n
        windows += n // win
        tail += n % win
    assert total == R.TOKENIZED_TOKENS[args.setting], (total, R.TOKENIZED_TOKENS[args.setting])
    rep = {'setting': args.setting, 'corpus_tokens': total, 'manifest_corpus_train_tokens': R.CORPUS_TRAIN_TOKENS[args.setting],
           'empty_text_documents': R.CORPUS_EMPTY_DOCS[args.setting], 'windows_per_pass': windows,
           'tokens_read_per_pass': windows * win, 'trained_tokens_per_pass': windows * R.SEQ_LEN,
           'dropped_tail_tokens_per_pass': tail, 'steps_per_pass': windows / R.GLOBAL_BATCH_SEQ, 'horizons': []}
    for k, h in enumerate(R.HORIZONS, 1):
        samples = h * R.GLOBAL_BATCH_SEQ
        rep['horizons'].append({'epoch': k, 'step': h, 'samples': samples, 'trained_tokens': h * R.TOKENS_PER_STEP,
                                'corpus_tokens_read': samples * win, 'passes': samples / windows,
                                'full_passes': samples // windows, 'windows_into_last_pass': samples % windows})
    print(json.dumps(rep, indent=1))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
