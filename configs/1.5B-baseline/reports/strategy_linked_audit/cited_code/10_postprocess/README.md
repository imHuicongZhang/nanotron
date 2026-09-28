# 10_postprocess — finalize rewrites → 10B pretrain-ready datasets

Post-processes the completed rewrites for **signal-disagreement-lambda05** and
**quality-first** and assembles each arm's final **10B** pretraining corpus
(`shared-top-5B` original text + a freshly assembled `5B` rewritten set), shuffled at the
document level.

CPU-only (`cpu` partition, account `bvandur1`, no GPU). All steps are idempotent /
resumable and use atomic `.tmp` + `os.replace` writes.

## Inputs
- First-pass (Wikipedia-style) rewrites: `…/10B/<setting>/rewritten/` (200 shards)
- Second-pass (distill) rewrites: `…/10B/<setting>/distill/` (200 shards, row-aligned to
  `rewritten/` within each shard)
- Shared-top originals (NOT rewritten): `…/10B/shared-top-5B/` (200 shards)
- Llama-2 tokenizer: `…/tokenizers/llama2-unsloth-tokenizer`

`doc_id` is the canonical identity key. The same `doc_id` in `rewritten/` and `distill/`
is the same source document rewritten by two prompts — kept as **two** training examples.
`tokens-llama2` is the **source** length; `rewritten_tokens` is the **output** length —
kept strictly separate. Budget length = `(tokens + 1)`: the +1 is the one `</s>` datatrove appends per document at tokenization (there is no BOS; an earlier version of this line said "one leading BOS", which was a mislabel of the same count).

## Pipeline

### Step 1 — `01_strip_prefix.py`
Strips the artifact prefix `"Here is a paraphrased version:\n\n"` from every `status==2`
Wikipedia rewrite, **start-anchored only** (`text[len(prefix):]`, never `.replace`/
`.lstrip`), then **recounts** `rewritten_tokens` with the Llama-2 tokenizer
(`add_special_tokens=False`, raw — no BOS) and writes shards back in place.
Distill output is **scanned and reported only** (most-common first-50-chars +
preamble verdict) — distill text is never modified. → `_step1_summary.json`.

### Step 2 — `02_assemble_5B.py`
Assembles ~5B rewritten training tokens per setting: ALL `status==2` Wikipedia docs first,
then top up from `distill/` (status==2, NOT de-duplicated against wiki) until cumulative
`(rewritten_tokens + 1)` reaches 5B (last doc kept whole). If wiki+distill still fall short,
the shortfall is reported and the run STOPs (no padding).

**Quality-sort key** — always on the ORIGINAL document's precomputed columns (the rewritten
text is never scored), read straight from the shard:
- `quality-first`: `fasttext-ranking-v2` DESC.
- `signal-disagreement-lambda05`: `u = q + 0.5·√v` DESC, with `q = mean(3 v2 cols)`,
  `v = population variance (ddof=0)`, **float32**, λ=0.5 — matching `05_select_s5_variants/
  select_s5.py`.

Output → `pretrain/<setting>/rewritten/{wiki,distill}_NNNNN.parquet`. Rows carry `doc_id`,
`orig_doc_id`, `text` (= the rewrite), `rewritten_tokens`, `tokens-llama2`, `status`, the
three `*-ranking-v2` cols, `url`, `metadata`, `topic`, `source_prompt`
(`wikipedia`|`distill`), and the sort key (`u_score` materialized for signal-disagreement).
Also writes `_assembly_manifest.json` (incl. full cross-pass coverage) and
`_step2_summary.json`.

### Step 3 — `03_mix_shared_top.py`
- Physically **copies** `shared-top-5B/` into `pretrain/<setting>/shared-top-5B/` with
  `source_prompt="original"` (`text` stays original).
- **Shuffles** (seed=42, document level) shared-top + rewritten into
  `pretrain/<setting>/shuffled/part_NNNNN.parquet` (~500k rows/shard) — what training reads.
- Verifies **0** `doc_id` overlap between shared-top and the rewritten set's source docs.
- Reports combined budget (`shared-top + rewritten ≈ 10B`) and writes
  `_pretrain_manifest.json`.
- Assembles **`postprocess_report.md`** from the Step 1/2/3 summaries.

## Run
```bash
sbatch run_all.sh        # Step 1 → 2 → 3 on the cpu partition (96 cpu / 480G / 8h)
```
> ⚠️ Do not submit until the code has been reviewed and explicitly approved.

## Final layout
```
data_rewrite/pretrain/
  signal-disagreement-lambda05/{shared-top-5B/, rewritten/, shuffled/,
                                _assembly_manifest.json, _pretrain_manifest.json}
  quality-first/               {shared-top-5B/, rewritten/, shuffled/,
                                _assembly_manifest.json, _pretrain_manifest.json}
```
`shuffled/` is the training input. Report: `10_postprocess/postprocess_report.md`.
