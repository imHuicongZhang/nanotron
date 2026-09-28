---
license: cc-by-4.0
language:
  - en
tags:
  - pretraining
  - data-selection
pretty_name: Know Your Sources — raw (unrewritten) baseline corpora
---

# Know Your Sources: raw baseline corpora (raw text)

Seven raw, unrewritten training corpora for the Know-Your-Sources 1.5B grid. Each is about 10B training tokens.
They come in two families. For what each corpus is, and what each comparison can and cannot establish, see
[`reports/RAW_SELECTED_BASELINES_PROVENANCE.md`](reports/RAW_SELECTED_BASELINES_PROVENANCE.md) and
[`reports/GLOBAL_TOP10B_SELECTION_REPORT.md`](reports/GLOBAL_TOP10B_SELECTION_REPORT.md).

## Strategy-linked raw controls (4)

**Composition:** the **shared 5B anchor** plus a **5B strategy half**. The strategy half is a seed-42 random
subsample of the unique source documents of one rewritten arm's *final rewritten half*, kept as original text.

**Consequences:**
- Only documents whose rewrite succeeded and survived assembly are eligible.
- These controls match their rewritten arm's token budget.
- They do **not** contain the same documents: the strategy half covers 46–53% of the arm's source documents.

| folder | rewritten counterpart | strategy half |
|---|---|---|
| `raw_text/raw_diversity_oriented/` | `diversity_oriented` | random half of the arm's successfully rewritten sources |
| `raw_text/raw_disagreement_aware/` | `disagreement_aware` | random half of the arm's successfully rewritten sources |
| `raw_text/raw_random/` | `wrap_inspired` | random half (52%) of the uniform sample WRAP rewrote successfully; a uniform-sample reference, not WRAP's exact documents |
| `raw_text/raw_rewire_inspired/` | `rewire_inspired` | random half of the sources whose rewrites passed REWIRE's post-rewrite fastText filter |

**`raw_rewire_inspired` is a conditional ablation.** Its membership depends on how well each document's *rewrite*
scored, which is information that exists only after rewriting. It is not a raw-only selection policy.

**Quality-First has no raw arm.** Its raw comparison is the existing Quality-Base: the anchor + the fastText-best 5B
of Quality-First's own input.

## Global Top-10B controls (3, no anchor)

The whole ~10B corpus is one global Top-10B selection over the same universe as the original 1.5B Quality-Base:
the 99,949,162 scored documents of
[`blab-jhu/KYS-DCLM-Refinedweb-100M-Scored`](https://huggingface.co/datasets/blab-jhu/KYS-DCLM-Refinedweb-100M-Scored),
minus the 50,000-doc validation holdout.

**Selection rule:**
- score descending, with the original seeded tie-break;
- whole documents, until cumulative training tokens first reach 10B.

**What these corpora do not have:** no shared anchor, no rewriting, no floors, quotas, variance terms or domain
restrictions.

**Comparator:** the existing fastText Quality-Base, which is the global fastText Top-10B under the same conventions
up to 3 tail documents.

| folder | score |
|---|---|
| `raw_text/raw_top10b_fineweb_edu/` | `fineweb-edu-ranking-v2` (tie-aware global percentile) |
| `raw_text/raw_top10b_modernbert/` | `modernbert-ranking-v2` |
| `raw_text/raw_top10b_consensus/` | mean of the three percentiles: (fastText + FineWeb-Edu + ModernBERT) / 3 |

## Files

Each `raw_text/<setting>/` holds 16 parquet files, `part-00000.parquet` … `part-00015.parquet`.

| column | type | meaning |
|---|---|---|
| `orig_doc_id` | int64 | position in the 100M DCLM-RefinedWeb reservoir sample (shard `id // 500000`, row `id % 500000`) |
| `doc_id` | int64 | *global Top-10B only*: row of the scored pool, the selection key |
| `source` | string | `anchor` / `strategy` (strategy-linked) or `selected` (global Top-10B) |
| `text` | string | the original document text |

- **Each folder is the final training corpus, in training order.** Documents are shuffled once at the document
  level (seed 42, `pp_io.bucketed_shuffle`), and rows follow that order file by file.
- [`manifest.json`](manifest.json) records every count, each setting's `expected_total_tokens`, the sha256 of every
  file, the tokenizer's sha256, and the generation code commits.
- `settings_overview` lists every setting's family, anchor presence and comparator.
- `selection/<setting>/` holds the selected doc ids of the global Top-10B settings. The digest conventions are in
  `manifest.json` under `global_top10b`.

## Tokenization — done by the consumer

Use `tools/kys_raw/tokenize_raw_text.sh <data_root> <setting>` from
[`imHuicongZhang/nanotron`](https://github.com/imHuicongZhang/nanotron) (branch `huicong-dev`).

**What the script does:**
- runs 16 datatrove tasks with one `</s>` per document and no BOS;
- does no merging or shuffling;
- runs `tools/fix_ds_metadata.py`;
- checks the total against `expected_total_tokens`.

**Token convention:** `len(llama2_tokenizer(text, add_special_tokens=False)) + 1` per document.

**Check before publication:** all seven totals were confirmed with this exact script.

**Procedure:** `configs/1.5B-baseline/WORKFLOW_RAW_BASELINES.md` in the code repository.

## Provenance in brief

- **Text.** Every text is read from the raw 100M pool by position. No rewritten text is used anywhere.
- **Strategy-linked settings:**
  - source set = unique `orig_doc_id` of the non-anchor rows of the published
    [`wytro/Know-Your-Sources`](https://huggingface.co/datasets/wytro/Know-Your-Sources) arm;
  - 5B cut = `numpy.random.default_rng(42).permutation` order, shortest prefix reaching 5B;
  - anchor = 4,120,164 documents / 5,000,002,332 tokens, identical in every arm, merged in.
- **Global Top-10B settings:**
  - the selection code reproduces the original Quality-Base selection bit for bit;
  - the percentile columns reproduce exactly from the raw scores;
  - every document was re-tokenized at assembly and its length checked equal to the scored pool's `tokens-llama2`.
- **Code:** `tools/kys_raw/` in the code repository. The exact commits are in `manifest.json`.

## Related

- **Tokenizer:** [`tokenizer/`](tokenizer) in this repo, the exact llama-2 tokenizer directory the grid used.
- **Init checkpoints:** [`wytro/Know-Your-Sources-init`](https://huggingface.co/wytro/Know-Your-Sources-init).
- **Rewritten comparators (v2):** [`wytro/KYS-1.5B-Rewritten-v2`](https://huggingface.co/wytro/KYS-1.5B-Rewritten-v2).
- **Trained raw baselines:** `blab-jhu/KYS-1.5B-Raw-Selected-Baselines`.
