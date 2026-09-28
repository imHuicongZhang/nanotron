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
They come in two families. The three global Top-10B corpora are documented in
[`reports/GLOBAL_TOP10B_SELECTION_REPORT.md`](reports/GLOBAL_TOP10B_SELECTION_REPORT.md).

## Strategy-linked raw controls (4, published earlier, unchanged)

Each is the **shared 5B anchor** (4,120,164 documents / 5,000,002,332 training tokens, identical in all four) plus a
**5B raw strategy half** linked to one rewritten arm of
[`wytro/Know-Your-Sources`](https://huggingface.co/datasets/wytro/Know-Your-Sources). Their files, counts and
generation commit are recorded in `manifest.json` and were not modified by the addition of the global Top-10B
corpora.

| folder | rewritten counterpart |
|---|---|
| `raw_text/raw_diversity_oriented/` | `diversity_oriented` |
| `raw_text/raw_disagreement_aware/` | `disagreement_aware` |
| `raw_text/raw_random/` | `wrap_inspired` |
| `raw_text/raw_rewire_inspired/` | `rewire_inspired` |

**Interpretation under review.** How exactly each strategy half relates to its arm's rewriting input and rewritten
output (document overlap, conditioning on rewrite success or on REWIRE's post-rewrite filter) is being reviewed
separately. Until that review is published, treat these as equal-token-budget controls, not as document-matched
"same documents, unrewritten" interventions.

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

**Comparator:** the existing fastText Quality-Base. Rebuilt under the same conventions, its document set reproduces
the published Quality-Base digest, and it equals the global fastText Top-10B plus 3 tail documents.

**Scorer-training data.** The ModernBERT quality head was fit on 50,427 Claude-labelled documents; all DCLM ones were
removed from the scored pool before scoring, so none is in any corpus. A separate ~5M-document analysis sample was
not used to fit the head and was not removed, exactly as in the original Quality-Base universe; every selection
holds it at its pool rate (5.00%). Benchmark contamination was not tested. Details: selection report §2b.

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

**Check before publication:** for the three global Top-10B settings, the datatrove totals from this exact script
equal `expected_total_tokens` (selection report §8). The script enforces the same check for every setting.

**Procedure:** `configs/1.5B-baseline/WORKFLOW_RAW_BASELINES.md` in the code repository.

## Provenance in brief

- **Text.** Every text is read from the raw 100M pool by position. No rewritten text is used anywhere.
- **Global Top-10B settings:**
  - the selection code uses the original 1.5B selection primitives (universe, tie-break, whole-document cutoff);
    applied to fastText it reproduces the published Quality-Base document-set digest;
  - the three percentile columns recompute exactly from the raw scores (all 99,949,162 rows);
  - every document was re-tokenized at assembly and its length checked equal to the scored pool's `tokens-llama2`;
  - exported files were checked against the selection (document set, no duplicates, order, token total) and a
    sample of texts was compared byte for byte with the raw pool (selection report §7b).
- **Code:** `tools/kys_raw/` in the code repository. The exact commits are in `manifest.json`.

## Related

- **Tokenizer:** [`tokenizer/`](tokenizer) in this repo, the exact llama-2 tokenizer directory the grid used.
- **Init checkpoints:** [`wytro/Know-Your-Sources-init`](https://huggingface.co/wytro/Know-Your-Sources-init).
- **Rewritten comparators (v2):** [`wytro/KYS-1.5B-Rewritten-v2`](https://huggingface.co/wytro/KYS-1.5B-Rewritten-v2).
- **Trained raw baselines:** `blab-jhu/KYS-1.5B-Raw-Selected-Baselines`.
