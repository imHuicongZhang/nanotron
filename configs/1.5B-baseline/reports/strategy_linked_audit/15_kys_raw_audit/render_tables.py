#!/usr/bin/env python
"""Render the numeric tables of docs/RAW_SELECTED_BASELINES_PROVENANCE.md from the audit outputs.

    python 15_kys_raw_audit/render_tables.py > 15_kys_raw_audit/tables.md

Inputs: 15_kys_raw_audit/raw_provenance_audit.json (audit_raw_provenance.py) and
15_kys_raw_audit/pair_overlap.json (pair_overlap.py).
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
A = json.loads((H / 'raw_provenance_audit.json').read_text())
P = json.loads((H / 'pair_overlap.json').read_text())
ORDER = ['raw_diversity_oriented', 'raw_disagreement_aware', 'raw_random', 'raw_rewire_inspired']


def n(x):
    return f'{x:,}'


def pc(x):
    return f'{100 * x:.2f}%'


out = []
out += ['### T1. Input → successful rewrite sources → raw half (documents / source TRAIN tokens)', '',
        '| setting | I: rewriting input | R: unique sources of final rewritten half | S: raw strategy half | I − R (dropped before training) |',
        '|---|---:|---:|---:|---:|']
for s in ORDER:
    d = A['settings'][s]['sizes']
    m = A['settings'][s]['I_minus_R']
    out.append(f'| `{s}` | {n(d["I"]["docs"])} / {n(d["I"]["train_tokens"])} | {n(d["R"]["docs"])} / {n(d["R"]["train_tokens"])} | '
               f'{n(d["S"]["docs"])} / {n(d["S"]["train_tokens"])} | {n(m["docs"])} / {n(m["train_tokens"])} |')
out += ['', '### T2. Subset relations and rule re-execution', '',
        '| setting | R ⊆ I | S ⊆ R | S ∩ anchor | source_tokens.npy = pool tokens-llama2+1 | seed-42 rule re-executed on R reproduces S |',
        '|---|---|---|---:|---|---|']
for s in ORDER:
    d = A['settings'][s]
    out.append(f'| `{s}` | {d["R_subset_I"]} | {d["S_subset_R"]} | {d["S_overlap_anchor"]} | {d["source_tokens_match_pool"]} | {d["subsample_rule_reproduces_S"]} |')
out += ['', '### T3. Directional coverage (documents; source-TRAIN-token weighted in parentheses)', '',
        'Coverage of A by B = |A ∩ B| / |A|; token-weighted = Σ(tokens-llama2+1 of A ∩ B) / Σ(tokens-llama2+1 of A).', '',
        '| setting | S in R | R in S | Jaccard(S,R) | S in I | I in S | Jaccard(S,I) | I in R |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
for s in ORDER:
    c, j = A['settings'][s]['coverage'], A['settings'][s]['jaccard']
    f = lambda k: f'{pc(c[k]["docs"])} ({pc(c[k]["tokens"])})'  # noqa: E731
    out.append(f'| `{s}` | {f("S_in_R")} | {f("R_in_S")} | {j["S_R"]:.4f} | {f("S_in_I")} | {f("I_in_S")} | {j["S_I"]:.4f} | {f("I_in_R")} |')
out += ['', '### T4. Raw half vs the rewritten outputs (one-to-many)', '',
        'Rewritten rows are counted per output row; a source rewritten by two prompts contributes two rows. '
        'Output-token coverage = rewritten TRAIN tokens (rewritten_tokens + 1) whose source is in S, over all rewritten TRAIN tokens of the half.', '',
        '| setting | rewrite rows (by prompt) | sources with 1 / 2 outputs | rewritten TRAIN tokens | rows with source in S | output-token coverage by S |',
        '|---|---|---:|---:|---:|---:|']
for s in ORDER:
    p = P[s]
    pr = ', '.join(f'{k} {n(v)}' for k, v in p['prompts'].items())
    out.append(f'| `{s}` | {pr} | {n(p["multiplicity"]["1"])} / {n(p["multiplicity"]["2"])} | {n(p["rewritten_train_tokens"])} | '
               f'{n(p["rewritten_rows_with_source_in_raw"])} | {pc(p["rewritten_tokcov"])} |')
out += ['', '### T5. Distribution drift caused by conditioning (TRAIN-token-weighted 24-topic TV / JS in bits; mean percentiles; length)', '',
        '| setting | topic TV R vs I | topic TV S vs I | JS S vs I | mean fastText pct I → R → S | mean q I → R → S | mean TRAIN tokens/doc I → R → S | I − R: mean tokens/doc |',
        '|---|---:|---:|---:|---|---|---|---:|']
for s in ORDER:
    d = A['settings'][s]
    t, pr = d['topic_shift'], d['profile']
    out.append(f'| `{s}` | {t["R_vs_I"]["tv"]:.4f} | {t["S_vs_I"]["tv"]:.4f} | {t["S_vs_I"]["js_bits"]:.5f} | '
               f'{pr["I"]["mean_ft"]:.3f} → {pr["R"]["mean_ft"]:.3f} → {pr["S"]["mean_ft"]:.3f} | '
               f'{pr["I"]["mean_q"]:.3f} → {pr["R"]["mean_q"]:.3f} → {pr["S"]["mean_q"]:.3f} | '
               f'{pr["I"]["mean_train_tokens"]:.0f} → {pr["R"]["mean_train_tokens"]:.0f} → {pr["S"]["mean_train_tokens"]:.0f} | '
               f'{pr["I_minus_R"]["mean_train_tokens"]:.0f} |')
out += ['', '### T6. Published corpora (local staging copy of the Hub files; sha256 listed in manifest.json)', '',
        '| setting | rows | TRAIN tokens (pool tokens-llama2+1) | anchor rows | strategy rows | doc set = anchor ∪ S | duplicates | S digest (sorted orig_doc_id) | file-order digest |',
        '|---|---:|---:|---:|---:|---|---:|---|---|']
for s in ORDER:
    d = A['settings'][s]['published']
    out.append(f'| `{s}` | {n(d["rows"])} | {n(d["train_tokens"])} | {n(d["anchor_rows"])} | {n(d["strategy_rows"])} | '
               f'{d["docset_equals_anchor_plus_S"]} | {d["duplicates"]} | `{P[s]["sel_digest_sorted_int64le"][:16]}…` | `{d["file_order_orig_sha256"][:16]}…` |')
q = A['quality_first_vs_quality_base']
out += ['', '### T7. Quality-First input vs the Quality-Base 5B block', '',
        '| set | docs | TRAIN tokens | mean fastText pct | min fastText pct |', '|---|---:|---:|---:|---:|']
for k, lab in (('I_quality_first_input', 'Quality-First rewriting input (next 10B by fastText after the anchor)'),
               ('R_quality_first_rewritten_sources', 'unique sources of the final Quality-First rewritten half'),
               ('quality_base_block', 'Quality-Base non-anchor 5B block')):
    out.append(f'| {lab} | {n(q[k]["docs"])} | {n(q[k]["train_tokens"])} | {q[k]["mean_ft"]:.4f} | {q[k]["min_ft"]:.4f} |')
out += ['', f'Quality-Base block ⊆ Quality-First input: {q["qb_block_subset_of_I"]}; block covered by R: '
        f'{pc(q["qb_block_in_R"]["docs"])} of docs ({pc(q["qb_block_in_R"]["tokens"])} of tokens); R covered by the block: '
        f'{pc(q["R_in_qb_block"]["docs"])} ({pc(q["R_in_qb_block"]["tokens"])}); block ∩ anchor = {q["qb_overlap_anchor"]}. '
        f'`wytro/Know-Your-Sources` revision `{q["hf_revision"]}`.']
c = A['checks']
out += ['', '### T8. Reconstruction checks', '',
        f'- shared-top-5B reproduced from `04_select/select_10b.py` rules equals the published anchor: **{c["shared_top_equals_published_anchor"]}** '
        f'({n(c["anchor"]["docs"])} docs, {n(c["anchor"]["train_tokens"])} TRAIN tokens; sorted orig_doc_id sha256 `{c["anchor"]["orig_docset_sha256"]}`, '
        f'sorted doc_id sha256 `{c["anchor"]["docset_sha256"]}`).',
        f'- λ=0.5 disagreement-aware input reproduced from `05_select_s5_variants/select_s5.py` rules equals bit 2 of `06_lambda_grid/lambda_grid.npz`: '
        f'**{c["lambda05_equals_lambda_grid_bit2"]}** (U = {n(c["lambda05_meta"]["U_docs"])} docs, Q30(q) = {c["lambda05_meta"]["Q30"]:.10f}, '
        f'Q90(v) = {c["lambda05_meta"]["V90"]:.10g}, survivors {n(c["lambda05_meta"]["survivors"])}).']
for k, v in A['inputs'].items():
    out.append(f'- input `{k}`: {n(v["docs"])} docs, {n(v["train_tokens"])} TRAIN tokens, ∩ anchor {v["overlap_anchor"]}, ∩ val {v["overlap_val"]}, '
               f'sorted doc_id sha256 `{v["docset_sha256"][:16]}…`')
print('\n'.join(out))
