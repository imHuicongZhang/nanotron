---
license: cc-by-4.0
language:
  - en
tags:
  - pretraining
  - data-selection
pretty_name: Know Your Sources — raw-selected (pre-rewrite) baseline corpora
---

# Know Your Sources: raw-selected baseline corpora (raw text)

No-rewrite controls for the Know-Your-Sources 1.5B grid. Each corpus trains a model on the **same
shared 5B anchor** and **the same selected source documents** as a rewritten arm, but keeps those
documents in their **original, unrewritten** form, subsampled at the document level to the same
5B-token strategy budget. Training on these against the rewritten arms separates the effect of
rewriting from the effect of source selection.

| folder | rewritten counterpart | strategy half |
|---|---|---|
| `raw_text/raw_diversity_oriented/` | `diversity_oriented` | original text of the documents the Diversity-Oriented arm rewrote |
| `raw_text/raw_disagreement_aware/` | `disagreement_aware` | original text of the documents the Disagreement-Aware arm rewrote |
| `raw_text/raw_random/` | `wrap_inspired` | original text of the uniform random sample the WRAP-Inspired arm rewrote |
| `raw_text/raw_rewire_inspired/` | `rewire_inspired` | original text of the source documents of the 5B kept by REWIRE's post-rewrite filter |

**`raw_random` is the no-selection reference for `wrap_inspired`**: its sources were drawn uniformly
at random with no quality or diversity criterion, so it is both the no-rewrite control for
`wrap_inspired` and the reference for what selection adds. **`raw_rewire_inspired`** starts from a
random pool too, but its sources are only the documents whose rewrites passed REWIRE's post-rewrite
filter — a selection made after rewriting.

## Files

Each `raw_text/<setting>/` holds 16 parquet files, `part-00000.parquet` … `part-00015.parquet`:

| column | type | meaning |
|---|---|---|
| `orig_doc_id` | int64 | position of the document in the 100M DCLM-RefinedWeb reservoir sample |
| `source` | string | `anchor` (the shared 5B) or `strategy` (this setting's 5B) |
| `text` | string | the original document text |

**This is the final training corpus, in training order**: anchor and strategy documents are already
merged and shuffled; rows follow the exact post-shuffle document order, file by file. Each folder is
about 10B tokens. [`manifest.json`](manifest.json) records every count, the expected token total after
tokenization, and the sha256 of every file.

**Tokenization is done by the consumer**, with `tools/kys_raw/tokenize_raw_text.sh` from
[`imHuicongZhang/nanotron`](https://github.com/imHuicongZhang/nanotron) (branch `huicong-dev`),
as described in `configs/1.5B-baseline/RUNBOOK.md`: 16 datatrove tasks, one `</s>` per document, no
merging or shuffling, then `tools/fix_ds_metadata.py`, then a check against the manifest's token total.

## How the corpora were built

1. **Source documents.** For each rewritten arm, the `orig_doc_id` of every non-anchor row
   (`source_prompt != 'original'`) of the published
   [`wytro/Know-Your-Sources`](https://huggingface.co/datasets/wytro/Know-Your-Sources) parquet,
   deduplicated across the rewriting prompts: the source documents whose rewrites were kept.
2. **Original text.** Read from the 100M reservoir sample by position (`orig_doc_id`: shard
   `id // 500000`, row `id % 500000`). No rewritten text is used anywhere.
3. **Token budget rule.** Tokens per document = `len(llama2_tokenizer(text, add_special_tokens=False)) + 1`
   (the `+1` is the end-of-document token). A source set above 5,000,000,000 tokens is cut by taking
   documents in the order of `numpy.random.default_rng(42).permutation` over the sorted unique ids and
   keeping the shortest prefix reaching 5B — whole documents only, at most one document of overshoot.
   At or below 5B it is used as is. The same rule applies to all four settings.
4. **Anchor.** The shared 5B anchor — 4,120,164 documents, 5,000,002,332 tokens, the rows with
   `source_prompt == 'original'`, identical in every arm — also read from the pool by position.
5. **Shuffle.** Anchor and strategy documents shuffled together at the document level with seed 42
   by `pp_io.bucketed_shuffle`, the same function every published arm was shuffled with.
6. **Leakage checks**, all passed before upload: every parquet read in the build code reads text only
   from the raw pool; all 4,120,164 anchor texts match the pool byte for byte and total exactly
   5,000,002,332 tokens; 200 sampled documents per setting match the pool and match none of the
   rewrites of the same source document; style features (markdown headings and lists, missing
   URLs/boilerplate) and document lengths sit with the raw pool, not with the rewritten arm.

Code: `tools/kys_raw/` (`build_raw_sources.py`, `assemble_raw_corpus.py`, `verify_raw_corpus.py`,
`publish_raw_text.py`); the exact commit is in `manifest.json`.

## Related

- Tokenizer: [`tokenizer/`](tokenizer) in this repo, the exact llama-2 tokenizer directory the grid was tokenized with (sha256 of each file in `manifest.json`, `tokenizer.sha256`). `tokenize_raw_text.sh` uses `<data_root>/tokenizer` by default.
- Init checkpoints: [`wytro/Know-Your-Sources-init`](https://huggingface.co/wytro/Know-Your-Sources-init), `_init_1.5B_seed{42,43,44}/0/`, with hash manifests.
- Training configs: `configs/1.5B-baseline-seed{42,43,44}/` and `configs/1.5B-baseline/RUNBOOK.md` in the code repo.
