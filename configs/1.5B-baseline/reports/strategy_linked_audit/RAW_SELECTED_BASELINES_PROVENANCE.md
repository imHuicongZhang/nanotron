# Raw-selected baselines — provenance audit (strategy-linked controls)

**Status: reviewed and published (2026-09-28).** The documentation corrections of §7d–7e have been applied to Marc's
guide, the nanotron READMEs and the dataset repo's README and manifest `role` descriptions. No data, selection,
tokenizer or training configuration was changed.

**Scope.** The four strategy-linked raw corpora in
[`blab-jhu/KYS-Pre-Rewritten@ed09db2a`](https://huggingface.co/datasets/blab-jhu/KYS-Pre-Rewritten/tree/ed09db2aa18b0319297735af181bcbe8b877e830)
(`raw_diversity_oriented`, `raw_disagreement_aware`, `raw_random`, `raw_rewire_inspired`), audited against the
production code and the published artifacts, not against README text; plus whether the existing Quality-Base supplies
Quality-First's raw comparison. The three global Top-10B baselines are documented separately
(`imHuicongZhang/nanotron@c953d14c`, `configs/1.5B-baseline/reports/GLOBAL_TOP10B_SELECTION_REPORT.md`).

**History.** First audit 2026-09-27 (commit `242f72a`, sections 1–9 below). Re-verified and extended 2026-09-28
(§R, §6, §7): the rewritten-data revision was pinned by checksum, every raw strategy half was rebuilt independently of
the raw-build outputs, the first audit's script was re-run from independently rebuilt inputs, the token stream was
checked, and all four published corpora were tokenized with the consumer script.

**Where this file lives.** Canonical: `docs/RAW_SELECTED_BASELINES_PROVENANCE.md` in the selection project
(`projects/rewrite`, which has no Git remote). Published copies, byte-identical:
- `imHuicongZhang/nanotron` (branch `huicong-dev`), `configs/1.5B-baseline/reports/strategy_linked_audit/`, together
  with the evidence: `15_kys_raw_audit/` (scripts + JSON) and `cited_code/` (the selection, rewriting and
  post-processing scripts cited below, at the paths used here), plus `SHA256SUMS`;
- `blab-jhu/KYS-Pre-Rewritten`, `reports/RAW_SELECTED_BASELINES_PROVENANCE.md`.

Paths in this report are relative to the selection project; in the nanotron copy, `04_select/…`, `10_postprocess/…`
etc. are under `cited_code/`. `06_lambda_grid/lambda_grid.npz` (470 MB) is cited by sha256 only.

---

## 0. Verdicts and recommendations

| setting | what the raw strategy half is | implementation | verdict | recommendation |
|---|---|---|---|---|
| `raw_diversity_oriented` | seed-42 random 5B of the unique sources of Diversity-Oriented's final rewritten half (51.4% of them) | correct; rebuilt exactly | **Verified, but interpretation/naming needs qualification** | keep unchanged; describe as an equal-token, same-distribution control, not a document-matched one |
| `raw_disagreement_aware` | the same for Disagreement-Aware (51.8%) | correct; rebuilt exactly | **Verified, but interpretation/naming needs qualification** | keep unchanged; same description |
| `raw_random` | seed-42 random 5B of the sources WRAP rewrote successfully (52.5%); WRAP's input was a uniform sample of the pool minus validation and anchor | correct; rebuilt exactly | **Verified, but interpretation/naming needs qualification** | keep unchanged; drop "no-selection reference" unless qualified (§7b) |
| `raw_rewire_inspired` | seed-42 random 5B of the sources whose **rewrites passed REWIRE's post-rewrite fastText filter** (45.9%) | correct; rebuilt exactly; matches its documented design | **Verified, but interpretation/naming needs qualification** — a *conditional* ablation | keep unchanged; label as conditional; read REWIRE's pipeline effect against `raw_random` (§7b) |

**No implementation defect was found** in any of the four (§7d). Each published corpus is exactly the documented
construction: anchor (identical to the counterpart's anchor) + a seed-42 random whole-document 5B subsample of the
unique sources of the counterpart's final rewritten half, shuffled once with seed 42, tokenized as text + `</s>`
exactly like the rewritten arms. What needs correcting is **how they are described**: none is a document-matched
"same documents, unrewritten" intervention, and one is conditioned on post-rewrite filtering.

**Principal findings.**
- **Equal training budget: yes.** Same 14,305 steps × 2,097,152 tokens; corpora of 10.000B raw vs 9.89–10.00B
  rewritten TRAIN tokens.
- **Identical source documents: no, by necessity.** Rewriting halves document length (rewritten/source token ratio
  0.46–0.53), so at a fixed 5B budget the raw half can hold only about half of the rewritten half's sources
  (45.9–52.5%). A uniform random half is the unbiased way to do that.
- **Same candidate distribution: yes** for diversity, disagreement and WRAP (the raw half is a uniform subsample of
  the same sources: topic TV ≤ 0.012 vs the rewriting input, identical mean percentiles). **No** for REWIRE.
- **Conditioned on rewriting outcome: yes for all four**, but negligibly for three (0.15–0.29% of input documents,
  mostly over-length ones, 2.7–4.8% of input tokens). **Substantially for REWIRE** (the filter kept 39.8% of input
  documents and moved the mean source fastText percentile 0.479 → 0.629).
- **Quality-First needs no separate raw arm:** Quality-Base's non-anchor block is the fastText-best half of
  Quality-First's own rewriting input (99.87% of it was rewritten into Quality-First's half). It is a stronger and
  unconditioned raw control (§6).

**Handoff (§7f).** All seven settings can proceed as configured; none of the four needs a data correction or
retraining. The qualification is in the interpretation; the documentation now states it (applied 2026-09-28, §7d). Keep micro-batch 32 for every new run; treat Quality-Base seed 42 (mbs 16) as a known,
small protocol deviation in the comparator (§7f).

---

## 1. Four properties that must not be conflated

| property | meaning | holds for the four raw arms? |
|---|---|---|
| **P1. Equal training-token budget** | raw and rewritten corpora have the same ~10B TRAIN tokens (5B anchor + 5B strategy half) and train for the same 14,305 steps | **Yes.** S = 5,000,000,553 … 5,000,002,737 tokens. The rewritten halves are 4,889,635,504 (diversity, a documented shortfall) to 5,000,000,351 tokens. |
| **P2. Identical source-document identities** | the raw half contains exactly the source documents that produced the rewritten half | **No.** S ⊂ R with Jaccard 0.46–0.53 (T3). |
| **P3. Matches the original rewriting-input distribution** | the raw half is distributed like the strategy's selection I | **Approximately for diversity, disagreement and random:** topic TV ≤ 0.012, identical mean percentiles, mean length 3–5% shorter (T5). **No for REWIRE:** the filter changes quality, topic and length. |
| **P4. Selection conditioned on rewriting outcome** | membership depends on whether or how well the document was rewritten | **Yes for all four:** successful rewrite (status 2), plus survival through assembly. **REWIRE additionally** depends on the fastText score of its *rewritten* text. |

A raw corpus resampled to reach 5B source tokens need not contain the documents that produced the 5B rewritten
tokens. These corpora do not, and must not be called strict document-matched interventions.

---

## R. Re-verification (2026-09-28)

Everything below was re-run from inputs rebuilt independently of the raw build's own outputs
(`/projects/bvandur1/zhuicon1/kys/raw_sources/` was not used except where stated).

**R1. Which rewritten data the raw corpora were built from.** The raw build read the local copy
`kys/hf_parquet/<arm>/` of `wytro/Know-Your-Sources` (`tools/kys_raw/build_raw_sources.py`), whose download did not
record a revision. All 123 local files (parquet + `metadata.json`) were hashed and matched against every commit of the
repo: they are byte-identical to **`wytro/Know-Your-Sources@9e5ff24149c2957c30f0c8fdd051a8eb3b75baad`**, the latest
commit and the only one holding all of them (`metadata.json` matched by size, being non-LFS). Each arm's
`metadata.json` states that its folder is "the complete 10B-token mixture the models actually trained on". The
locally tokenized rewritten mixtures sum exactly to the arms' recorded `token_count_llama2`
(diversity 9,889,637,833; disagreement 10,000,002,333; WRAP 10,000,002,419; REWIRE 10,000,002,683). The v2 checkpoints
themselves were trained on another cluster, so "the v2 models saw exactly these bytes" rests on that provenance record
and the equal totals, not on a byte comparison with the training cluster's files.
(`15_kys_raw_audit/pin_rewritten_revision.py`, `rewritten_revision_pin.json`.)

**R2. Independent rebuild of every raw strategy half** (`15_kys_raw_audit/independent_rebuild.py`,
`independent_rebuild.json`). R = unique `orig_doc_id` of the non-`original` rows, read directly from the pinned
rewritten parquet; TRAIN tokens from the scored pool (`tokens-llama2 + 1`); the documented rule re-applied. For all
four settings the rebuilt half **equals the published strategy rows exactly**, the published anchor **equals the
counterpart arm's anchor exactly**, anchor ∩ strategy = 0, and there are 0 duplicate documents.

| setting | R docs | rule: k docs / TRAIN tokens / overshoot | rebuilt = published | anchor = arm's anchor | published TRAIN tokens | S sorted-id sha256 | file-order sha256 |
|---|---:|---|---|---|---:|---|---|
| `raw_diversity_oriented` | 5,867,876 | 3,018,451 / 5,000,000,553 / 553 | yes | yes | 10,000,002,885 | `4cdbeb0fb93a4331…` | `22843235ff43fb00…` |
| `raw_disagreement_aware` | 5,592,424 | 2,898,764 / 5,000,002,737 / 2,737 | yes | yes | 10,000,005,069 | `08ae25c57439792e…` | `678ceca45644212e…` |
| `raw_random` | 10,573,523 | 5,554,525 / 5,000,001,419 / 1,419 | yes | yes | 10,000,003,751 | `75b91f4d6b96f7b9…` | `c4749e064980e4dd…` |
| `raw_rewire_inspired` | 8,445,785 | 3,872,986 / 5,000,000,799 / 799 | yes | yes | 10,000,003,131 | `c0693ae04e5ddad7…` | `5a89bd5c7ef20fec…` |

The anchor in all four arms (and in Quality-First) is the same 4,120,164 documents / 5,000,002,332 TRAIN tokens,
sorted-`orig_doc_id` sha256 `d8a6d22307c2e7c3…`, and equals the fastText top-5B re-derived from `select_10b.py`.
The first audit's anchor-text check compared all 4,120,164 raw anchor texts with the rewritten arm's anchor texts:
0 mismatches (`kys/verify/verify_raw_diversity_oriented.json`, `check2_anchor`).

**R3. First audit re-run.** `15_kys_raw_audit/audit_raw_provenance.py` re-run on flat arrays re-extracted from the
sha256-verified local copy of the scored pool (the six arrays are byte-identical to the ones used on 2026-09-27)
reproduces `raw_provenance_audit.json` exactly; the only differing field is the input path. So T1–T8 below stand.

**R4. BOS or EOS — resolved from the code and from the bytes.**
- Code: the Llama-2 `tokenizer.json` ships a post-processor that *prepends* `<s>` (id 1). datatrove 0.5.0's
  `DocumentTokenizer(eos_token="</s>")` **replaces** that post-processor with `TemplateProcessing("$A <EOS>")`
  (`datatrove/utils/tokenization.py` l.93-98). The raw consumer script (`tools/preprocess_data_parquet.py`) uses it;
  the rewritten arms' training streams show the identical structure (next bullet).
- Bytes: in the first 200,000 documents of shard 0, every document **ends with id 2** and **none starts with id 1**,
  in the raw corpus (`raw_diversity_oriented`) and in the rewritten arms' training streams (`diversity_oriented`,
  `wrap_inspired`) alike. For the first 1,000 raw documents the stream equals
  `encode(text, add_special_tokens=False) + [2]` exactly (1,000 / 1,000).
- The "+1 BOS" comments in `select_10b.py` l.18/93 and `10_postprocess/README.md` are therefore **wrong labels for a
  correct count**: the +1 is the appended `</s>`. Budgets are unaffected; the token *streams* of raw and rewritten
  corpora follow the same convention.
- In both pipelines, a literal `<s>` / `</s>` string inside document text is encoded as the special id (a few dozen
  per 200k documents). This is identical on both sides of every comparison.

**R5. Consumer tokenization of the four published corpora** (`tools/kys_raw/tokenize_raw_text.sh` from
`imHuicongZhang/nanotron@c953d14c`, unchanged; the script first checked every local parquet against the sha256 in
`manifest.json@ed09db2a`). This check had not been run on these four corpora before.

| setting | shards | datatrove total tokens | `expected_total_tokens` (manifest) | equal | `.ds` bytes | wall time (4 workers, 17 cores) |
|---|---:|---:|---:|---|---:|---:|
| `raw_diversity_oriented` | 16 | 10,000,002,885 | 10,000,002,885 | True | 20,000,005,770 | 17 min |
| `raw_disagreement_aware` | 16 | 10,000,005,069 | 10,000,005,069 | True | 20,000,010,138 | 16 min |
| `raw_random` | 16 | 10,000,003,751 | 10,000,003,751 | True | 20,000,007,502 | 16 min |
| `raw_rewire_inspired` | 16 | 10,000,003,131 | 10,000,003,131 | True | 20,000,006,262 | 16 min |

In every setting each document ends with `</s>` (id 2) and none starts with `<s>` (id 1) (first 200,000 documents of shard 0 checked). Records: `15_kys_raw_audit/consumer_tokenization.json`.

## 2. Shared facts (verified)

**Identifiers.**
- `orig_doc_id` is the position in the raw 100M DCLM-RefinedWeb reservoir pool: shard `id // 500000`, row
  `id % 500000`.
- `doc_id` is the contiguous 0 … 99,949,161 row index of the scored pool (`6_merged_clean`, published unchanged
  as `blab-jhu/KYS-DCLM-Refinedweb-100M-Scored@dcbbc360`). It was formed by physically removing 50,838 rows that
  overlap the ModernBERT-labeller set (`00_TMP/merge_remove_50k.py`, `01_explore/step2c_match_50k.py`).
- `orig_doc_id` is strictly increasing in `doc_id`, checked on all rows, so the two are interchangeable.

**Universe and exclusions** (`04_select/select_10b.py`, `main()`):
- The 50,000-doc validation holdout is `sort(default_rng(SeedSequence(42).spawn(8)[0]).choice(N, 50_000))`.
- The shared anchor (shared-top-5B) is the fastText-v2 order over (all − val), filled to 5e9 TRAIN tokens.
- REMAINING = all − val − anchor: 95,778,998 docs.
- Ties are broken by `default_rng(SeedSequence(42).spawn(8)[1]).permutation(N)` ascending (`order_desc`).
- The budget stops at the first document whose cumulative TRAIN tokens reach the target, and keeps that document
  whole (`fill_to`).
- **The 5M analysis sample is not excluded anywhere.**

**TRAIN-token convention:** `tokens-llama2 + 1`.
- `tokens-llama2` is `len(AutoTokenizer(llama2-unsloth)(text, add_special_tokens=False))`
  (`03_TokenCounts/count_tokens.py`).
- The +1 is the `</s>` that datatrove appends. It is **not** a BOS, despite the "+1 BOS" comments in
  `select_10b.py:18,93` (and, until 2026-09-28, in `10_postprocess/README.md`). The length is the same either way.
- Every `source_tokens.npy` of the raw build equals pool `tokens-llama2 + 1` exactly (T2).
- The published raw corpora sum to their `expected_total_tokens` under this convention (T6).

**Shared anchor.**
- Size: 4,120,164 docs, 5,000,002,332 TRAIN tokens, overshoot 2,332.
- Digests: sorted `orig_doc_id` sha256 `d8a6d223…c15de4`; sorted `doc_id` sha256 `c80e0e7f…41658b`.
- It is reproduced exactly by the `select_10b.py` rules (T8) and is identical in all four published rewritten arms
  (`tools/kys_raw/build_raw_sources.py`, which checks this).
- **Anchor ∩ strategy half = 0** in every setting. This holds by construction, because every strategy selected
  from REMAINING (T8: every input ∩ anchor = 0). It is also asserted at build time (`build_raw_sources.py`,
  `assemble_raw_corpus.py`) and checked on the published files (T6). No duplicate handling was ever needed.

**Merge, shuffle, layout.**
- Anchor and S are read from the raw pool by position (`assemble_raw_corpus.pool_texts`, the only text read) and
  staged per pool shard in `orig_doc_id` order.
- They are then shuffled together by `pp_io.bucketed_shuffle(seed=42, rows_per_shard=500_000)`:
  - pass 1 draws one `default_rng(42)` bucket id per row over the staged inputs in path order, with B = 16;
  - pass 2 applies a `default_rng([42, b])` permutation within each bucket and writes 500k-row parts.
- The result is exported by `publish_raw_text.export` into 16 contiguous, near-equal files (`part-00000` …
  `part-00015`, ceil(rows/16) rows each).
- Tokenizing with 16 datatrove tasks and no shuffling (`tools/kys_raw/tokenize_raw_text.sh`) makes task i read
  file i, so the shard order is the file order.
- `pp_io.py` sha256 is `5134dcc1…`. It is untracked in this repository; an identical copy is now in
  `nanotron/tools/kys_raw/pp_io.py`.

**Tokenization.**
- Tokenizer: the Llama-2 (llama2-unsloth) `tokenizer.json`, sha256 `81bb383c…`, vocab 32,000, BOS 1, EOS 2.
- Recipe: datatrove 0.5.0 `DocumentTokenizer(eos_token="</s>")`, which appends one `</s>` per document and no BOS.
- No text transformation is applied at any stage of the raw build.
- The consumer check that the datatrove total equals `expected_total_tokens` was first run on these four corpora on
  2026-09-28 (§R5).

**Production code and revisions.**

| artifact | identifier |
|---|---|
| raw build code | `imHuicongZhang/nanotron@4f9f968d` (`tools/kys_raw/{build_raw_sources,assemble_raw_corpus,verify_raw_corpus,publish_raw_text}.py`), as recorded in `manifest.json` |
| raw corpora | `blab-jhu/KYS-Pre-Rewritten@ce241737`; byte-unchanged at `ed09db2a` (the later commits only added the global Top-10B settings) |
| rewritten arms (the source of R) | `wytro/Know-Your-Sources@9e5ff241` (pinned by sha256 of all 123 files, §R1) |
| scored pool | `blab-jhu/KYS-DCLM-Refinedweb-100M-Scored@dcbbc360` |
| raw build manifests (preparation filesystem) | `/projects/bvandur1/zhuicon1/kys/raw_sources/manifest.json`, `raw_corpus/<s>/_raw_manifest.json`, `verify/verify_<s>.json` |
| selection/rewrite/postprocess code (this repo) | committed with this report; sha256 prefixes in §9 |

**The original selection outputs no longer exist.** `data_rewrite/experiments/...` (per-strategy `doc_ids`,
per-row rewrite `status`, `_manifest.json` files) is gone from the preparation filesystem. The rewriting inputs I
were therefore **re-derived** from the original selection code over the published scored pool. They are validated
by:
- the anchor reproducing exactly;
- every R being a subset of its I;
- the λ = 0.5 set equalling the independent `lambda_grid.npz` record (T8).

Per-document rewrite status cannot be recovered. I − R is therefore the union of three groups, which cannot be
separated per document:
- failed rewrites (status 0 = templated input over the length limit, status 1 = truncated generation);
- successful rewrites not needed by the assembly top-up;
- for REWIRE, rewrites rejected by the filter.

The aggregate status counts survive in `07_rewrite`/`09_Distill` `progress/*.json` and agree with the published
row counts.

---

## 3. Per setting

Common to all four:
- **Raw build** (`build_raw_sources.py`): R = unique `orig_doc_id` of rows with `source_prompt != 'original'`
  in `wytro/Know-Your-Sources/<arm>`.
- **5B cut**: `perm = default_rng(42).permutation(|R|)` over sorted R; keep the shortest prefix whose cumulative
  TRAIN tokens reach 5e9, whole documents, so the overshoot is less than one document.
  - The sampling unit is the unique source document. A source rewritten by two prompts enters once.
  - The RNG is plain `default_rng(42)`, not a SeedSequence child.
  - Re-executed on R here, the rule reproduces S exactly for every setting (T2).
- **Failed, missing, rejected and duplicated documents.**
  - Only status-2 rewrites exist in R.
  - Documents whose only rewrite failed are absent.
  - A source with two outputs (Wikipedia-style + distill, or WRAP style + distill) counts once in R and once in S.
  - There are no duplicate `orig_doc_id` in any corpus (T6).
- Merge, shuffle, layout and tokenization are as in §2.

### 3.1 `raw_diversity_oriented` ↔ `diversity_oriented` (internal `diversity-first`)

1. **Pool and exclusions:** REMAINING (§2).
2. **Strategy rule** (`select_10b.py` l.252-282): for each of the 24 topics, quota = 10e9 × topic's share of
   REMAINING TRAIN tokens; fill by consensus q = mean of the three v2 percentiles, descending, with the §2 tie
   rule. Result: I = 5,876,747 docs / 10,000,077,737 tokens, no top-up.
3. **Rewriting.**
   - Every I row was rewritten twice: Wikipedia-style (`07_rewrite`) and distill (`09_Distill`).
   - Status-2 counts are 5,861,683 and 5,868,627 of 5,876,747.
   - Final half (`10_postprocess/02_assemble_5B_diversity.py`), per topic quota = 5e9 × topic's share of I:
     all Wikipedia-style status-2 docs first, then distill, each sorted by fastText v2 descending within the
     topic, with no cross-topic backfill.
   - Result: 8,336,411 rows, 4,889,635,504 rewritten TRAIN tokens. This is 110,364,496 short in Adult, History,
     Literature and Religion, which is documented policy.
4. **Where S came from:** a random half of **successful rewrite outputs that survived assembly**; not the
   rewriting input, not a separate selection.
   - R has 5,867,876 docs, of which 6,193 are distill-only sources, i.e. their Wikipedia-style rewrite failed.
   - I − R = 8,871 docs holding 272M tokens, mean 30,658 TRAIN tokens per document, mostly over the status-0
     input limit.
5. **Handling:** see "Common" above. The 2,468,535 sources with two outputs count once.
6. **5B cut:** 3,018,451 docs, 5,000,000,553 tokens, overshoot 553.
7. **Anchor:** §2, no overlap.
8. **Final corpus:**
   - 7,138,615 docs, 10,000,002,885 TRAIN tokens, 16 files of 446,164 rows (last 446,155).
   - Doc-ID digests: S sorted `orig_doc_id` `4cdbeb0f…`; file order `22843235…`.
9. **Distribution:** S tracks I closely (topic TV 0.0069, identical mean percentiles) except for the missing very
   long documents (mean 1,656 vs 1,702 tokens per document).

### 3.2 `raw_disagreement_aware` ↔ `disagreement_aware` (internal `signal-disagreement-lambda05`)

1. **Pool and exclusions:** REMAINING.
2. **Strategy rule** (`05_select_s5_variants/select_s5.py` logic; the λ = 0.5 run's own script is not preserved;
   see §8):
   - U = ∪ of the fastText, FineWeb-Edu and ModernBERT top-10% (by TRAIN tokens) of REMAINING = 14,982,068 docs.
   - Floor q ≥ Q30(q over U) = 0.6956847; cap v ≤ Q90(v over U), where v is the population variance of the three
     percentiles in float32.
   - Rank the 10,435,667 survivors by u = q + 0.5·√v descending (§2 tie rule) and fill to 10e9.
   - Result: I = 5,602,476 docs / 10,000,002,827 tokens. It equals `lambda_grid.npz` bit 2 (T8).
3. **Rewriting and final half** (`02_assemble_5B.py`):
   - Status-2 counts: Wikipedia-style 5,587,114, distill 5,592,360.
   - Assembly takes all Wikipedia-style docs, then distill by u descending (stable argsort), filled to 5e9.
   - Result: 8,442,273 rows, 5,000,000,003 tokens.
4. **Where S came from:** successful rewrite outputs that survived assembly. R = 5,592,424 docs; I − R = 10,052
   docs / 353M tokens (mean 35,089 tokens per document).
5. **Handling:** as for diversity; 2,849,849 sources have two outputs.
6. **5B cut:** 2,898,764 docs, 5,000,002,737 tokens, overshoot 2,737.
7. **Final corpus:**
   - 7,018,928 docs, 10,000,005,069 tokens, 16 files.
   - Digests: S `08ae25c5…`; file order `678ceca4…`.
8. **Distribution:** topic TV S vs I 0.0106, mean percentiles identical.

### 3.3 `raw_random` ↔ `wrap_inspired` (internal `wrap`)

1. **Pool and exclusions:** REMAINING. The input is a uniform sample of the pool **minus validation and minus the
   fastText top-5B anchor**, not of the whole pool.
2. **Strategy rule** (`select_10b.py` l.239-240): `REMAINING[default_rng(SeedSequence(42).spawn(8)[2]).permutation]`
   filled to 10e9. Result: I = 10,604,458 docs / 10,000,000,190 tokens.
3. **Rewriting.**
   - Each I row got exactly one WRAP style from `default_rng([42, shard]).integers(0, 4)` over
     `["easy","hard","wiki","qa"]` (`07_rewrite/rewrite_worker.py:39,60`).
   - Status-2 by style: 2,646,480 / 2,631,853 / 2,644,343 / 2,643,215, totalling 10,565,891 of 10,604,458.
   - Each row also got a distill rewrite.
   - Final half (`02_assemble_5B_wrap.py`): all WRAP status-2 docs (4.155B tokens), then distill in
     `default_rng(42).permutation` order to 5e9, with no quality sort.
   - Result: 13,022,091 rows, 5,000,000,087 tokens.
4. **Where S came from:** successful WRAP (or distill) outputs in the final half. R = 10,573,523 docs = I minus
   30,935 docs / 479M tokens (mean 15,499 tokens per document).
5. **Handling:** 2,448,568 sources have two outputs (WRAP style + distill); each counts once.
6. **5B cut:** 5,554,525 docs, 5,000,001,419 tokens, overshoot 1,419.
7. **Final corpus:**
   - 9,674,689 docs, 10,000,003,751 tokens, 16 files.
   - Digests: S `75b91f4d…`; file order `c4749e06…`.
8. **WRAP distinction.**
   - **The exact source documents WRAP used** are I: all 10,604,458, and 10.0B source tokens.
   - `raw_random` contains a **random 52.4% of them** (50.0% of their tokens), drawn only from those that were
     rewritten successfully.
   - Statistically it is "another uniform sample under nearly the same rule", restricted to REMAINING and very
     slightly depleted of very long documents (T5).
   - As a no-selection reference it is sound, with that caveat. As "the documents WRAP rewrote" it is not.

### 3.4 `raw_rewire_inspired` ↔ `rewire_inspired` (internal `rewrite`)

1. **Pool and exclusions:** REMAINING.
2. **Strategy rule, stage 1** (`select_10b.py` l.245-249): `REMAINING[default_rng(SeedSequence(42).spawn(8)[3]).permutation]`
   filled to **20e9**, a compute-limited stand-in for "rewrite everything". Result: I = 21,214,299 docs /
   20,000,000,679 tokens.
3. **Rewriting and filter.**
   - Status-2 counts: Wikipedia-style 21,185,312, distill 21,195,043.
   - Pool (`02_build_pool_rewrite.py`): all status-2 outputs, 42,380,355 rows / 16.22B rewritten tokens.
   - Stage 2 (`03_fasttext_score_rewrite.py`, `04_filter_top5B_rewrite.py`): fastText-score the **rewritten
     text**, sort globally descending (stable), and keep the top 5e9 rewritten TRAIN tokens.
   - Result: 12,290,444 rows, 5,000,000,351 tokens.
4. **Where S came from:** **documents retained after post-rewrite filtering.**
   - R = 8,445,785 docs / 10.90B source tokens, i.e. 39.8% of I's documents.
   - The 12,768,514 dropped docs are mostly filter rejections, and are short and low-quality (mean 713 tokens,
     mean fastText 0.380).
5. **Handling:** 3,844,659 sources kept both outputs; each counts once in R.
6. **5B cut:** 3,872,986 docs, 5,000,000,799 tokens, overshoot 799.
7. **Final corpus:**
   - 7,993,150 docs, 10,000,003,131 tokens, 16 files.
   - Digests: S `c0693ae0…`; file order `5a89bd5c…`.
8. **Conditioning, labelled explicitly: this raw corpus depends on the post-rewrite filter.**
   - Membership is decided by the fastText score of each document's *rewrite*, information that exists only
     after rewriting. That pulls the *source* distribution far from I: mean fastText percentile 0.479 → 0.629,
     mean q 0.489 → 0.582, topic TV 0.083, longer documents.
   - It is therefore a **conditional ablation**, "the originals of the documents whose rewrites won", and not an
     independent raw-only selection policy.
   - An unconditioned raw counterpart of REWIRE's input would be a uniform sample of REMAINING. That population is
     statistically the same as `raw_random`'s: both I sets are uniform samples of REMAINING under independent
     RNG children.

---

## 4. The guide's "seed 42 from about 10B raw source tokens", checked per setting

| setting | pool the 5B was drawn from | its TRAIN tokens | rewriting input size | seed | verdict |
|---|---|---:|---:|---|---|
| raw_diversity_oriented | R | 9,728,111,104 | 10,000,077,737 | `default_rng(42)` | true for R, not for the input |
| raw_disagreement_aware | R | 9,647,286,028 | 10,000,002,827 | `default_rng(42)` | true for R, not for the input |
| raw_random | R | 9,520,542,699 | 10,000,000,190 | `default_rng(42)` | true for R, not for the input |
| raw_rewire_inspired | R (post-filter) | 10,899,228,624 | **20,000,000,679** | `default_rng(42)` | "≈10B" holds for R only; the input was 20B |

---

## 5. Raw / rewrite overlap tables

The tables below are generated (`15_kys_raw_audit/render_tables.py`). I = rewriting input, R = unique sources of
the final rewritten half, S = published raw strategy half. Token weights are **source** TRAIN tokens
(tokens-llama2 + 1) unless marked as output tokens.

<!-- BEGIN GENERATED TABLES -->
### T1. Input → successful rewrite sources → raw half (documents / source TRAIN tokens)

| setting | I: rewriting input | R: unique sources of final rewritten half | S: raw strategy half | I − R (dropped before training) |
|---|---:|---:|---:|---:|
| `raw_diversity_oriented` | 5,876,747 / 10,000,077,737 | 5,867,876 / 9,728,111,104 | 3,018,451 / 5,000,000,553 | 8,871 / 271,966,633 |
| `raw_disagreement_aware` | 5,602,476 / 10,000,002,827 | 5,592,424 / 9,647,286,028 | 2,898,764 / 5,000,002,737 | 10,052 / 352,716,799 |
| `raw_random` | 10,604,458 / 10,000,000,190 | 10,573,523 / 9,520,542,699 | 5,554,525 / 5,000,001,419 | 30,935 / 479,457,491 |
| `raw_rewire_inspired` | 21,214,299 / 20,000,000,679 | 8,445,785 / 10,899,228,624 | 3,872,986 / 5,000,000,799 | 12,768,514 / 9,100,772,055 |

### T2. Subset relations and rule re-execution

| setting | R ⊆ I | S ⊆ R | S ∩ anchor | source_tokens.npy = pool tokens-llama2+1 | seed-42 rule re-executed on R reproduces S |
|---|---|---|---:|---|---|
| `raw_diversity_oriented` | True | True | 0 | True | True |
| `raw_disagreement_aware` | True | True | 0 | True | True |
| `raw_random` | True | True | 0 | True | True |
| `raw_rewire_inspired` | True | True | 0 | True | True |

### T3. Directional coverage (documents; source-TRAIN-token weighted in parentheses)

Coverage of A by B = |A ∩ B| / |A|; token-weighted = Σ(tokens-llama2+1 of A ∩ B) / Σ(tokens-llama2+1 of A).

| setting | S in R | R in S | Jaccard(S,R) | S in I | I in S | Jaccard(S,I) | I in R |
|---|---:|---:|---:|---:|---:|---:|---:|
| `raw_diversity_oriented` | 100.00% (100.00%) | 51.44% (51.40%) | 0.5144 | 100.00% (100.00%) | 51.36% (50.00%) | 0.5136 | 99.85% (97.28%) |
| `raw_disagreement_aware` | 100.00% (100.00%) | 51.83% (51.83%) | 0.5183 | 100.00% (100.00%) | 51.74% (50.00%) | 0.5174 | 99.82% (96.47%) |
| `raw_random` | 100.00% (100.00%) | 52.53% (52.52%) | 0.5253 | 100.00% (100.00%) | 52.38% (50.00%) | 0.5238 | 99.71% (95.21%) |
| `raw_rewire_inspired` | 100.00% (100.00%) | 45.86% (45.87%) | 0.4586 | 100.00% (100.00%) | 18.26% (25.00%) | 0.1826 | 39.81% (54.50%) |

### T4. Raw half vs the rewritten outputs (one-to-many)

Rewritten rows are counted per output row; a source rewritten by two prompts contributes two rows. Output-token coverage = rewritten TRAIN tokens (rewritten_tokens + 1) whose source is in S, over all rewritten TRAIN tokens of the half.

| setting | rewrite rows (by prompt) | sources with 1 / 2 outputs | rewritten TRAIN tokens | rows with source in S | output-token coverage by S |
|---|---|---:|---:|---:|---:|
| `raw_diversity_oriented` | distill 2,474,728, wikipedia 5,861,683 | 3,399,341 / 2,468,535 | 4,889,635,504 | 4,288,640 | 51.46% |
| `raw_disagreement_aware` | distill 2,855,159, wikipedia 5,587,114 | 2,742,575 / 2,849,849 | 5,000,000,003 | 4,376,795 | 51.84% |
| `raw_random` | distill 2,456,200, wrap_easy 2,646,480, wrap_hard 2,631,853, wrap_qa 2,643,215, wrap_wiki 2,644,343 | 8,124,955 / 2,448,568 | 5,000,000,087 | 6,840,890 | 52.52% |
| `raw_rewire_inspired` | distill 7,024,123, wikipedia 5,266,321 | 4,601,126 / 3,844,659 | 5,000,000,351 | 5,636,442 | 45.87% |

### T5. Distribution drift caused by conditioning (TRAIN-token-weighted 24-topic TV / JS in bits; mean percentiles; length)

| setting | topic TV R vs I | topic TV S vs I | JS S vs I | mean fastText pct I → R → S | mean q I → R → S | mean TRAIN tokens/doc I → R → S | I − R: mean tokens/doc |
|---|---:|---:|---:|---|---|---|---:|
| `raw_diversity_oriented` | 0.0067 | 0.0069 | 0.00009 | 0.844 → 0.844 → 0.844 | 0.854 → 0.854 → 0.854 | 1702 → 1658 → 1656 | 30658 |
| `raw_disagreement_aware` | 0.0102 | 0.0106 | 0.00016 | 0.837 → 0.837 → 0.837 | 0.872 → 0.872 → 0.872 | 1785 → 1725 → 1725 | 35089 |
| `raw_random` | 0.0112 | 0.0117 | 0.00019 | 0.479 → 0.479 → 0.479 | 0.489 → 0.488 → 0.488 | 943 → 900 → 900 | 15499 |
| `raw_rewire_inspired` | 0.0834 | 0.0833 | 0.00647 | 0.479 → 0.629 → 0.629 | 0.489 → 0.582 → 0.582 | 943 → 1290 → 1291 | 713 |

### T6. Published corpora (local staging copy of the Hub files; sha256 listed in manifest.json)

| setting | rows | TRAIN tokens (pool tokens-llama2+1) | anchor rows | strategy rows | doc set = anchor ∪ S | duplicates | S digest (sorted orig_doc_id) | file-order digest |
|---|---:|---:|---:|---:|---|---:|---|---|
| `raw_diversity_oriented` | 7,138,615 | 10,000,002,885 | 4,120,164 | 3,018,451 | True | 0 | `4cdbeb0fb93a4331…` | `22843235ff43fb00…` |
| `raw_disagreement_aware` | 7,018,928 | 10,000,005,069 | 4,120,164 | 2,898,764 | True | 0 | `08ae25c57439792e…` | `678ceca45644212e…` |
| `raw_random` | 9,674,689 | 10,000,003,751 | 4,120,164 | 5,554,525 | True | 0 | `75b91f4d6b96f7b9…` | `c4749e064980e4dd…` |
| `raw_rewire_inspired` | 7,993,150 | 10,000,003,131 | 4,120,164 | 3,872,986 | True | 0 | `c0693ae04e5ddad7…` | `5a89bd5c7ef20fec…` |

### T7. Quality-First input vs the Quality-Base 5B block

| set | docs | TRAIN tokens | mean fastText pct | min fastText pct |
|---|---:|---:|---:|---:|
| Quality-First rewriting input (next 10B by fastText after the anchor) | 6,136,187 | 10,000,000,529 | 0.9280 | 0.8973 |
| unique sources of the final Quality-First rewritten half | 6,122,949 | 9,509,596,850 | 0.9281 | 0.8973 |
| Quality-Base non-anchor 5B block | 3,133,023 | 5,000,000,805 | 0.9431 | 0.9274 |

Quality-Base block ⊆ Quality-First input: True; block covered by R: 99.87% of docs (95.61% of tokens); R covered by the block: 51.10% (50.27%); block ∩ anchor = 0. `wytro/Know-Your-Sources` revision `9e5ff24149c2957c30f0c8fdd051a8eb3b75baad`.

### T8. Reconstruction checks

- shared-top-5B reproduced from `04_select/select_10b.py` rules equals the published anchor: **True** (4,120,164 docs, 5,000,002,332 TRAIN tokens; sorted orig_doc_id sha256 `d8a6d22307c2e7c325fdc20e52039ace950a6808dcc21f9b5e5b5687d2c15de4`, sorted doc_id sha256 `c80e0e7f8327f840d40028c7d8f9bc239d94d5e3a2c1408314a60cebda41658b`).
- λ=0.5 disagreement-aware input reproduced from `05_select_s5_variants/select_s5.py` rules equals bit 2 of `06_lambda_grid/lambda_grid.npz`: **True** (U = 14,982,068 docs, Q30(q) = 0.6956847310, Q90(v) = 0.1036491916, survivors 10,435,667).
- input `wrap`: 10,604,458 docs, 10,000,000,190 TRAIN tokens, ∩ anchor 0, ∩ val 0, sorted doc_id sha256 `a46a74b5a46c8c13…`
- input `rewrite`: 21,214,299 docs, 20,000,000,679 TRAIN tokens, ∩ anchor 0, ∩ val 0, sorted doc_id sha256 `9fc9f5bffccfb827…`
- input `diversity`: 5,876,747 docs, 10,000,077,737 TRAIN tokens, ∩ anchor 0, ∩ val 0, sorted doc_id sha256 `566649d9e0e4b018…`
- input `lambda05`: 5,602,476 docs, 10,000,002,827 TRAIN tokens, ∩ anchor 0, ∩ val 0, sorted doc_id sha256 `eb8f872a6ed03e3a…`
<!-- END GENERATED TABLES -->

---

## 6. Quality-First: why there is no raw arm, and whether Quality-Base supplies it

Verified on 2026-09-28 against the Quality-First rewritten mixture itself (`wytro/Know-Your-Sources@9e5ff241`,
`quality_first/`, ID columns only; `15_kys_raw_audit/quality_first_vs_base.py`, `quality_first_vs_base.json`).

**Construction (from code, confirmed on data).**
- `select_10b.py` l.219-231: after the anchor, one fastText-v2 order `qf_order` over REMAINING. Quality-Base's
  non-anchor block = `fill_to(qf_order, 5e9)`; Quality-First's rewriting input = `fill_to(qf_order, 10e9)`. The block
  is therefore the first half of the same order: **block ⊂ Quality-First input** (verified).
- Quality-Base = anchor + block, raw text. Rebuilt from the rules, its doc set has sha256 `90252ee5…`, equal to
  `provenance.doc_id_digest_sha256` in the published `quality_base/metadata.json`.
- Quality-First's rewritten half (`10_postprocess/02_assemble_5B.py`): **all** Wikipedia-style rewrites of the whole
  input (6,112,494 rows), then distill rewrites in fastText-v2-descending order until 5e9 rewritten TRAIN tokens
  (4,198,815 rows); 5,000,000,308 tokens. Its anchor equals the rule anchor (verified).

| set | docs | TRAIN tokens (source) | mean fastText pct | min fastText pct | mean tokens/doc |
|---|---:|---:|---:|---:|---:|
| Quality-First rewriting input | 6,136,187 | 10,000,000,529 | 0.9280 | 0.8973 | 1,630 |
| R_QF: unique sources of the Quality-First rewritten half | 6,122,949 | 9,509,596,850 | 0.9281 | 0.8973 | 1,553 |
| **Quality-Base non-anchor block** | 3,133,023 | 5,000,000,805 | **0.9431** | **0.9274** | 1,596 |
| a strategy-linked-style raw half (seed-42 random 5B of R_QF), hypothetical | 3,219,945 | 5,000,000,972 | 0.9281 | 0.8973 | 1,553 |

- 99.87% of the block's documents (95.61% of its tokens) have a rewrite in Quality-First's half; the block is 51.1%
  of R_QF's documents; 58.8% of Quality-First's rewritten tokens come from block sources (the distill top-up follows
  fastText order).

**Verdict: Quality-Base genuinely supplies the comparison, and a separate raw Quality-First arm is not needed.**
- Same anchor, same token budget, and the raw block is drawn from exactly Quality-First's candidate pool.
- Two differences in kind from the strategy-linked controls, both of which make it the *cleaner* control:
  1. it is the fastText-best half, not a random half (mean pct 0.943 vs 0.928), so it is a slightly *stronger* raw
     baseline: any rewriting gain measured against it is, if anything, understated;
  2. it is **not** conditioned on rewrite success, and it is an implementable raw-only policy.
- A `raw_quality_first` built like the other four would be a random half of R_QF: a weaker, rewrite-conditioned
  near-duplicate of Quality-Base. Not recommended.
- Caveat: Quality-Base seed 42 ran micro-batch 16 (§7f).

---

## 7. What each comparison can and cannot establish

Four properties (§1): P1 equal training-token budget; P2 identical source documents; P3 same selection procedure /
candidate distribution; P4 no conditioning on rewriting outcome.

| pair | P1 | P2 | P3 | P4 | supports | does not support |
|---|---|---|---|---|---|---|
| `diversity_oriented` vs `raw_diversity_oriented` | yes (9.89B vs 10.00B corpus; same 30B processed) | no: raw half = 51.4% of the rewritten half's sources | yes (TV 0.007) | ~yes (0.15% of input docs lost, over-length) | "for this strategy's selected sources, is a 5B slot better spent on rewrites or on original text?" | per-document effects; claims about the ~0.3B tokens of very long inputs that failed to rewrite |
| `disagreement_aware` vs `raw_disagreement_aware` | yes | no: 51.8% | yes (TV 0.011) | ~yes (0.18%) | same | same |
| `wrap_inspired` vs `raw_random` | yes | no: 52.5% | yes (both uniform over REMAINING) | ~yes (0.29%) | WRAP-style rewriting of a uniform sample vs a uniform raw sample of the same population | "WRAP's exact documents, unrewritten"; anything about the whole pool (anchor and validation are excluded from its population) |
| `rewire_inspired` vs `raw_rewire_inspired` | yes | no: 45.9% | **no** (filter-selected: fastText pct 0.479 → 0.629, topic TV 0.083) | **no**: membership uses the fastText score of each document's rewrite | "given the documents whose rewrites pass the filter, rewritten vs original text" | that REWIRE's pipeline beats raw-data training; any raw-only selection claim |
| `rewire_inspired` vs `raw_random` | yes | no | same population before the filter (both uniform over REMAINING, independent RNG children) | raw side unconditioned | **the REWIRE pipeline's total effect** (generate 20B → rewrite → filter) vs raw uniform data | isolating text transformation from the filter's selection |
| `raw_rewire_inspired` vs `raw_random` | yes | — | — | — | value of the filter's *source* selection on raw text | a deployable raw-only policy (it needs the rewrites) |
| `quality_first` vs `quality_base` | yes | no: block = 51.1% of R_QF, the best-scoring half | same candidate pool; raw side stronger | yes (raw side unconditioned) | rewriting vs the best raw half of the same fastText-ranked pool | a random-half control |
| global Top-10B arms vs `quality_base` | yes | — | — | yes | which single score, used to select the whole corpus, is best | any rewriting effect |

The anchor (50% of every corpus, identical everywhere) dilutes every effect: each comparison varies only the non-anchor half.

### 7a. Document matching vs token-budget matching

Rewriting changes lengths: the rewritten half used 0.50 (diversity), 0.52 (disagreement), 0.53 (WRAP) and 0.46
(REWIRE) rewritten tokens per source token, with 1.23–1.51 rewrite rows per source (two prompts per source for part
of each set). The mean rewrite row is 384–592 TRAIN tokens against 900–1,725 for its source. Consequently:
- a **document-matched** raw control (all of R as original text) would need 9.5–10.9B tokens for its half, i.e.
  ~2× the rewritten half's budget, and would break P1;
- a **token-matched** control can hold only ~half of R. Taking that half **uniformly at random** keeps the raw half
  distributed exactly like R (verified: identical mean percentiles and lengths, topic TV vs R ≈ 0), so the raw arm
  represents the same candidate distribution at the same budget.

The intended claim in the paper is budget-level: "at equal training tokens, does rewriting a strategy's documents
beat using them as is?" The token-matched, uniformly subsampled control is the correct control for that claim. It is
not the control for a per-document claim ("the same document, rewritten vs not"), and should not be described as one.
The rewritten arm also sees ~2× as many *distinct source documents* (in compressed form) as the raw arm; that
coverage gain is part of what rewriting does and is inside the measured effect, not a confound to be removed.

### 7b. Answers to the review questions

**Diversity-Oriented and Disagreement-Aware.** Subsampling 5B raw tokens from the recovered sources **preserves the
intended strategy comparison at the distribution level**: the raw half is a seed-42 uniform random 51.4% / 51.8% of
the sources represented in the final rewritten half, with the same topic mix (TV vs input 0.007 / 0.011), identical
mean percentiles, and the same mean length as R (1,656 vs 1,658; 1,725 vs 1,725 tokens/doc). Membership differs in
three ways: (i) the raw half holds ~half of R's documents, while the rewritten half holds rewrites of all of R;
(ii) 2,468,535 / 2,849,849 sources appear twice in the rewritten half (Wikipedia-style + distill) and once, if
sampled, in the raw half; (iii) the 8,871 / 10,052 input documents that never produced a usable rewrite (mean
30,658 / 35,089 tokens, mostly over the rewriting input limit) appear on neither side. Diversity's rewritten half is
4.89B (documented per-topic shortfall); its raw half is the full 5.00B.

**WRAP / `raw_random`.** It is **a subset of successful WRAP outputs' sources**: a seed-42 uniform random 52.4% of
WRAP's input documents (50.0% of its tokens), restricted to the 99.7% whose rewrite succeeded. It is **not** the exact
WRAP source selection (that is all 10,604,458 input documents) and **not** a separately drawn sample. Statistically it
is equivalent to "another uniform sample of REMAINING", very slightly depleted of the longest documents.
"**No-selection reference**" is accurate only for its strategy half and only relative to REMAINING: the corpus as a
whole is 50% fastText-top-5B anchor + 50% uniform sample of the rest, and the rest excludes that top 5B. Recommended
wording: "uniform-sample reference (WRAP's population: pool minus validation and anchor), plus the shared anchor".

**REWIRE / `raw_rewire_inspired`.** **Yes, raw documents are selected using the outcomes of the post-rewrite quality
filter.** The input was a uniform 20B sample of REMAINING; every input document was rewritten twice; the rewrites
were fastText-scored and the top 5B *rewritten* tokens kept (`03_fasttext_score_rewrite.py`,
`04_filter_top5B_rewrite.py`); the raw half is a random 45.9% of the sources of the kept rewrites. The filter kept
only 39.8% of input documents and shifted them strongly (mean fastText pct 0.479 → 0.629, mean length 943 → 1,290
tokens/doc, topic TV 0.083). The comparison it supports is **conditional**: "for documents whose rewrites win the
filter, is the rewritten text better than the original?" Invalid broader claims: that REWIRE beats training on raw
data, that the rewritten text *causes* the gain over an unselected raw corpus, or that `raw_rewire_inspired` is a raw
selection method anyone could deploy. **The unconditioned comparison already exists:** REWIRE's input population is
statistically identical to `raw_random`'s (both uniform over REMAINING, independent SeedSequence children), so
`rewire_inspired` vs `raw_random` measures the full pipeline's effect with no extra training.

**Quality-First.** §6: Quality-Base supplies the comparison and is the stronger, unconditioned control.

### 7c. What the four corpora share, stage by stage

original pool (99,949,162 scored docs; the 50,838 rows matching ModernBERT's labelling set removed earlier) →
minus 50,000 validation → anchor = fastText top-5B → REMAINING → strategy selection I (§3) → rewriting (Wikipedia-
style or WRAP style + distill; status 2 only) → assembly/filter → final rewritten half (R = its unique sources) →
raw half S = seed-42 random whole-document 5B of sorted R (`default_rng(42).permutation`, shortest prefix reaching
5e9 TRAIN tokens, overshoot < one document: 553 / 2,737 / 1,419 / 799) → anchor ∪ S (disjoint, no duplicates) → text
read from the raw 100M pool by `orig_doc_id` → `pp_io.bucketed_shuffle(seed=42)` → 16 contiguous parquet files →
datatrove, 16 tasks, text + `</s>`, no BOS.

### 7d. Implementation defects vs interpretation limitations

**Implementation defects: none found.** Every stage above was re-executed or checked on the bytes (R2–R5, T2, T6).

**Documentation errors (not data errors), corrected on 2026-09-28** in nanotron `configs/1.5B-baseline/`
(`WORKFLOW_RAW_BASELINES.md`, `README.md`, `RUNBOOK.md`), `tools/kys_raw/{registry.py,publish_raw_text.py:ROLE,
KYS-Pre-Rewritten.README.md}`, the dataset repo's `README.md` and the `role` fields of `manifest.json`:
- the earlier guide / HF README / `publish_raw_text.py:ROLE` / nanotron `configs/1.5B-baseline/README.md` (before
  2026-09-28) described the raw halves as "the original text of the documents the arm rewrote" and `raw_random` as
  "no new sample drawn"; both are inaccurate (random ~half of R; conditioned on rewrite success);
- the published manifest's `role` for `raw_random` said "no-selection reference (uniform random source sample)"
  without the anchor/REMAINING/successful-rewrite qualification; the four `role` strings were replaced (no other
  manifest field changed);
- "+1 BOS" wording in `10_postprocess/README.md` (corrected there) and in the code comments of `select_10b.py`
  l.18/93 and `pp_io.py` (left as they are: `select_10b.py` is archived as run and `pp_io.py` is sha256-pinned by
  the assembly code; the guide now states the correct convention);
- nanotron `deploy/clusters.yaml` l.138, `tools/assert_invariants.py` l.39, `tools/kys_raw/compare_configs.py` l.17 and
  `configs/1.5B-baseline/README.md` l.156 said mbs 32 "in 53 of 54" v2 checkpoints; the released configs show **51 of
  54** (the 3 exceptions are Quality-Base seed 42 ep1–ep3). Comment-only; no code path uses the number. Corrected.

**Interpretation limitations:** P2 fails for all four (by design, §7a); P4 fails materially for REWIRE.

### 7e. Recommendations

| setting | recommendation | data change | retraining | effect on Marc's handoff |
|---|---|---|---|---|
| `raw_diversity_oriented` | keep; describe as "equal-token, same-distribution raw control: seed-42 random 5B of the sources of the final rewritten half" | none | none | none (train as configured) |
| `raw_disagreement_aware` | same | none | none | none |
| `raw_random` | keep; rename the *interpretation* (not the setting name, which is a published path) to "uniform-sample reference (WRAP population) + shared anchor" | none | none | none |
| `raw_rewire_inspired` | keep; label "conditional on REWIRE's post-rewrite filter" everywhere; report `rewire_inspired` vs `raw_random` as the pipeline comparison | none | none | none |
| Quality-First | no new arm | — | — | — |

**No additional control is scientifically necessary** for the budget-level claims. Only if a per-document claim is
wanted would one be needed; the correct construction would then be *rewritten* half restricted to the rewrites of S
(so both sides contain exactly S's documents), which changes the rewritten arm's budget to ~2.5B and is a new
rewritten training arm, not a new raw one. Not recommended now.

### 7f. Handoff issues

**Which settings can proceed.** All seven. The four strategy-linked corpora are correctly built and need no
correction or regeneration, and the documentation now states how to interpret them (§7d). `plan_submit.py
--settings <subset>` can still stage the grid if wanted; the default plan submits all seven.

**Micro-batch 32 vs Quality-Base seed 42's 16.** Recommendation: **keep the uniform mbs 32 / accum 8 for all new
runs.**
- What mbs changes, as implemented: each micro-batch's loss is the mean over its unmasked label tokens
  (`masked_mean`, `models/llama.py`), divided by the number of micro-batches (`pipeline_parallel/engine.py` l.55) and
  averaged over dp. `label_mask` removes only the label at each document start (`data/clm_collator.py` l.83-94), so
  per-token weights differ between mbs 16 and 32 only through how the ~0.1% masked positions fall into micro-batches.
  The recorded measurement is a 1.43e-3 relative loss difference for mbs 4 vs 16 (`deploy/clusters.yaml`). It also
  changes bf16 reduction order, so runs are not bitwise comparable in any case.
- mbs 32 is what 51 of the 54 released v2 checkpoints used, including every rewritten comparator of the four
  strategy-linked settings and Quality-Base seeds 43/44. Matching Quality-Base seed 42 would make the new arms'
  seed 42 inconsistent with their own seeds 43/44 and with every rewritten arm.
- The deviation affects only comparisons against Quality-Base at seed 42 (the three global Top-10B arms, and
  `quality_first` vs `quality_base`). Its likely size is far below seed noise: Quality-Base's own ep3 Mean6 across
  seeds is 0.4566 / 0.4617 / 0.4698 (sd 0.0067; `kys-eval/reference/`), so the seed-42 cell cannot be attributed to
  mbs from existing data. Report seed-averaged results with the deviation footnoted, plus a seeds-43/44-only
  sensitivity check for the Quality-Base comparisons. Retraining Quality-Base seed 42 at mbs 32 would remove the
  caveat, at the cost of one chain (~100 node-hours at 22.8 s/it); optional, not required.

---

## 8. Verified facts vs unresolved items

**Verified** (recomputed; code and artifacts cited above):
- the selection rules of all five inputs, and the exact anchor reproduction;
- R ⊆ I, S ⊆ R, and S reproduced by the seed-42 rule;
- published doc sets = anchor ∪ S, with no duplicates;
- per-document token accounting;
- overlaps and drift (T1–T8);
- Quality-Base = anchor + first 5B of Quality-First's input.

**Added on 2026-09-28** (§R, §6):
- the rewritten data read by the raw build is byte-identical to `wytro/Know-Your-Sources@9e5ff241` (all 123 files);
- every published raw strategy half re-derived from that data alone (not from the raw build's outputs) equals the
  published one; every published anchor equals its counterpart arm's anchor; the first audit's script reproduces
  its JSON exactly from independently rebuilt inputs;
- raw and rewritten token streams are both `text + </s>`, no BOS (code and bytes);
- the datatrove totals of all four published corpora equal `expected_total_tokens` (§R5);
- Quality-First's rewritten half covers 99.78% of its input's documents and 99.87% of the Quality-Base block.

**Still not verifiable here (and why it does not change the verdicts):**
- that the v2 training cluster's tokenized files are byte-identical to the published rewritten mixtures: supported by
  the arms' provenance records and exact token totals, but the training cluster's files are not accessible;
- whether the ~0.2% of input documents missing from R failed rewriting or were not needed by assembly (per-row status
  is deleted); either way they are absent from both sides of the comparison.

**Unresolved or not verifiable:**
- **Per-document rewrite status** is gone, so I − R cannot be split between failure and not-needed-by-assembly.
  The aggregate counts are consistent.
- **The λ = 0.5 selection script as run is not preserved.** `select_s5.py` on disk builds λ ∈ {1.5, 2, 3}. The λ = 0.5
  input was reconstructed with its logic, and it matches the independent `lambda_grid.npz` record exactly, which
  itself was byte-gated against the original `doc_ids.npy` (`06_lambda_grid/select_lambda_grid.py`).
- **Validation holdout identity** is inferred from the exact anchor reproduction. The original `val_doc_ids.npy`
  is deleted.
- **`10_postprocess/DATASETS_SUMMARY.md` is stale.** For example, WRAP distill is given as 2,456,172 rows there vs
  2,456,200 published, and it says REWIRE was "not run". The published parquet is treated as ground truth.
- **This repository has no remote.** The cited code is committed locally with this report. Its sha256 values below
  let a reader match copies.

---

## 9. Code cited (this repository), sha256 prefixes at the time of the audit

| file | sha256 (16) | role |
|---|---|---|
| `00_TMP/merge_remove_50k.py` | `806bde0c88ac64c2` | 100M → 99,949,162 scored pool |
| `00_TMP/clean_v2_ranks.py` | `41bd4537cd01f494` | `*-ranking-v2` percentiles |
| `01_explore/step2c_match_50k.py` | `5031dbd5e0508812` | ModernBERT-labeller overlap keys |
| `03_TokenCounts/count_tokens.py` | `dc11c60c6dfaa0ab` | `tokens-llama2` |
| `04_select/select_10b.py` | `19fdb38708b45ca7` | val, anchor, Quality-Base, Quality-First, WRAP, REWIRE and Diversity inputs |
| `04_select/select_quality_base_15B.py` | `5b6e500ccf2b2ad8` | 15B-base (not used here) |
| `05_select_s5_variants/select_s5.py` | `342298e0f908d3dc` | disagreement-aware logic |
| `06_lambda_grid/select_lambda_grid.py`, `lambda_grid.npz` | `c3002304f94d6438`, `3ef427a5525a678e` | λ-grid record |
| `07_rewrite/rewrite_worker.py` | `58c054e5b868e821` | Wikipedia-style / WRAP rewriting, status codes |
| `09_Distill/rewrite_worker.py` | `cb3f3125048c7dc5` | distill rewriting |
| `10_postprocess/02_assemble_5B.py`, `_diversity.py`, `_wrap.py` | `b87c27280ccb1ef2`, `025aeaa36d21bb98`, `662fcaf88c55ad49` | rewritten-half assembly |
| `10_postprocess/02_build_pool_rewrite.py`, `03_fasttext_score_rewrite.py`, `04_filter_top5B_rewrite.py` | `8e35b2123fb44c34`, `de0204990a36376e`, `60d873dd526735b0` | REWIRE filter |
| `10_postprocess/03_mix_shared_top*.py`, `05_mix_shared_top_rewrite.py`, `04_assemble_base.py` | `98319126…`, `60f2ff94…`, `8e76d726…`, `aca38e0b…`, `ef859874…` | anchor merge |
| `10_postprocess/pp_io.py` | `5134dcc16bde1757` | `bucketed_shuffle` |
| `15_kys_raw_audit/audit_raw_provenance.py`, `pair_overlap.py`, `render_tables.py` | — | this audit (2026-09-27) |
| `15_kys_raw_audit/independent_rebuild.py`, `quality_first_vs_base.py` (+ their JSON) | — | re-verification (2026-09-28) |

Outside this repository (2026-09-28): `imHuicongZhang/nanotron@c953d14c` `tools/kys_raw/build_raw_sources.py`,
`assemble_raw_corpus.py`, `tokenize_raw_text.sh`, `tools/preprocess_data_parquet.py`, `src/nanotron/data/clm_collator.py`,
`src/nanotron/parallel/pipeline_parallel/engine.py`; datatrove 0.5.0 `utils/tokenization.py`. The temporary work
directory of the re-verification was deleted after publication. Its small records (audit re-run JSON, logs, per-shard
`.ds.metadata`, Hub publication record) are in `15_kys_raw_audit/run_records_2026-09-28/`; its large intermediates are
reproducible: flat arrays with nanotron `tools/kys_raw/extract_scored_columns.py`, tokenized corpora with
`tokenize_raw_text.sh`, the Quality-First ID extract with `15_kys_raw_audit/qf_sources.py`.
