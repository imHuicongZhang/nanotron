#!/usr/bin/env python
"""Full-scale rewrite worker: one independent data-parallel vLLM process per GPU.

Each worker:
  - loads ONE copy of Qwen2.5-7B-Instruct (tensor_parallel_size=1) on its single GPU,
  - owns the parquet shards where (shard_index % num_workers == worker_id),
  - processes each owned shard with ONE llm.generate() call (continuous batching),
  - writes the rewritten output as <dataset>/rewritten/part_NNNNN.parquet (atomic),
  - checkpoints per-shard so a killed job resumes by simply skipping finished shards,
  - updates a progress JSON and a shared monitor file every --monitor-every docs.

Prompt assembly (read from files at runtime, NEVER hardcoded):
  - grounded: prompt_template.replace("[TEXT]", doc_text)
  - wrap    : wrap_prompts[style] + doc_text   (style assigned by a seeded RNG)

Status codes:
  0 = templated input > --input-drop tokens (NOT rewritten)
  1 = finish_reason == 'length' (truncated)
  2 = finish_reason == 'stop'   (completed)
"""
import argparse
import fcntl
import glob
import json
import os
import re
import time

# Stay fully offline: never reach out to the HF hub from a compute node.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SHARD_RE = re.compile(r"part_(\d+)\.parquet$")
WRAP_STYLES = ["easy", "hard", "wiki", "qa"]  # index order is part of the reproducible seed


# --------------------------------------------------------------------------- #
# prompt assembly
# --------------------------------------------------------------------------- #
def build_content(mode, doc_text, prompt_template, wrap_prompts, style):
    """Return the user-message content (pre chat-template) for one doc."""
    doc_text = doc_text or ""
    if mode == "grounded":
        return prompt_template.replace("[TEXT]", doc_text)
    # wrap: instruction string already ends with "Passage:\n"
    return wrap_prompts[style] + doc_text


def assign_wrap_styles(shard_index, n_rows, base_seed=42):
    """Deterministic per-(shard,row) style assignment, robust to variable shard sizes.

    Seeded only by (base_seed, shard_index) so row i in a given shard ALWAYS maps to the
    same style regardless of which worker runs it or when (resume-safe).
    """
    rng = np.random.default_rng([base_seed, shard_index])
    idx = rng.integers(0, len(WRAP_STYLES), size=n_rows)
    return [WRAP_STYLES[i] for i in idx]


# --------------------------------------------------------------------------- #
# monitoring heuristics
# --------------------------------------------------------------------------- #
def _word_set(s):
    return set(w for w in re.split(r"\W+", (s or "").lower()) if w)


def flag_format_only(src, out):
    """True if the output's tokens are >90% a subset of the input's tokens."""
    o = _word_set(out)
    if not o:
        return False
    i = _word_set(src)
    return (len(o & i) / len(o)) > 0.90


def flag_repetition(out, win=50, reps=5):
    """True if any 50-char substring repeats 5+ times in the output."""
    if not out or len(out) < win:
        return False
    seen = {}
    # stride to keep this cheap; a real degenerate loop will still be caught
    for k in range(0, len(out) - win + 1, 10):
        sub = out[k:k + win]
        c = seen.get(sub, 0) + 1
        seen[sub] = c
        if c >= reps:
            return True
    return False


def flag_short(out_tokens, in_tokens):
    return out_tokens < 20 and in_tokens > 200


def append_monitor(monitor_file, dataset, worker_id, samples):
    """Append a monitoring block; flock so 8 concurrent workers don't interleave."""
    lines = []
    for s in samples:
        warns = []
        if s["status"] == 2:
            if flag_format_only(s["src"], s["out"]):
                warns.append("FORMAT-ONLY (>90% token overlap)")
            if flag_repetition(s["out"]):
                warns.append("DEGENERATE REPETITION")
            if flag_short(s["out_tokens"], s["in_tokens"]):
                warns.append("SUSPICIOUSLY SHORT")
        hdr = f"- **{dataset}** | worker {worker_id} | doc_id={s['doc_id']} | status={s['status']} | in_tok={s['in_tokens']} out_tok={s['out_tokens']}"
        if warns:
            hdr += "  ⚠️ WARNING: " + "; ".join(warns)
        lines.append(hdr)
        lines.append(f"  - input[:500]: {s['src'][:500]!r}")
        lines.append(f"  - output: {s['out']!r}")
    block = (f"\n### {dataset} — worker {worker_id} — sample @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
             + "\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(monitor_file), exist_ok=True)
    with open(monitor_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(block)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
def write_progress(path, prog):
    prog["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prog, f, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True, help="dir containing part_*.parquet")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama2-tokenizer", required=True,
                    help="path to the Llama-2 tokenizer that produced tokens-llama2")
    ap.add_argument("--mode", choices=["grounded", "wrap"], required=True)
    ap.add_argument("--prompt-file", help="grounded prompt .md (contains [TEXT])")
    ap.add_argument("--wrap-prompts", help="wrap_prompts.json (easy/hard/wiki/qa)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--input-drop", type=int, default=30720,
                    help="status=0 if templated input exceeds this many tokens")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--monitor-file",
                    default="/scratch/bvandur1/zhuicon1/projects/rewrite/00_TMP/rewriting_monitor.md")
    ap.add_argument("--monitor-every", type=int, default=10000)
    ap.add_argument("--progress-dir",
                    default="/scratch/bvandur1/zhuicon1/projects/rewrite/07_rewrite/progress")
    ap.add_argument("--dry-run", action="store_true",
                    help="build prompts + token counts for the first ~20 rows of the first "
                         "owned shard and print them WITHOUT loading vLLM (review aid)")
    args = ap.parse_args()

    # ---- load prompts ----
    prompt_template, wrap_prompts = None, None
    if args.mode == "grounded":
        if not args.prompt_file or not os.path.exists(args.prompt_file):
            raise SystemExit(f"STOP: grounded prompt file missing: {args.prompt_file}")
        prompt_template = open(args.prompt_file).read()
        if "[TEXT]" not in prompt_template:
            raise SystemExit("STOP: grounded prompt has no [TEXT] placeholder")
    else:
        if not args.wrap_prompts or not os.path.exists(args.wrap_prompts):
            raise SystemExit(f"STOP: wrap prompts file missing: {args.wrap_prompts}")
        wrap_prompts = json.load(open(args.wrap_prompts))
        missing = [k for k in WRAP_STYLES if k not in wrap_prompts]
        if missing:
            raise SystemExit(f"STOP: wrap_prompts missing styles: {missing}")

    # ---- tokenizers ----
    from transformers import AutoTokenizer
    qtok = AutoTokenizer.from_pretrained(args.model)
    ltok = AutoTokenizer.from_pretrained(args.llama2_tokenizer)

    def n_llama(text):
        return len(ltok(text or "", add_special_tokens=False).input_ids)

    # ---- shard assignment ----
    all_shards = sorted(glob.glob(os.path.join(args.dataset_path, "part_*.parquet")))
    owned = []
    for p in all_shards:
        m = SHARD_RE.search(os.path.basename(p))
        si = int(m.group(1))
        if si % args.num_workers == args.worker_id:
            owned.append((si, p))
    out_dir = os.path.join(args.dataset_path, "rewritten")
    os.makedirs(out_dir, exist_ok=True)
    progress_path = os.path.join(args.progress_dir, f"{args.dataset_name}_worker{args.worker_id}.json")

    print(f"[w{args.worker_id}] dataset={args.dataset_name} mode={args.mode} "
          f"owns {len(owned)}/{len(all_shards)} shards", flush=True)

    if args.dry_run:
        if not owned:
            print("[dry-run] no owned shards", flush=True)
            return
        si, p = owned[0]
        tbl = pq.read_table(p, columns=[args.text_col])
        texts = tbl.column(args.text_col).to_pylist()[:20]
        styles = assign_wrap_styles(si, len(texts)) if args.mode == "wrap" else [None] * len(texts)
        st0 = 0
        for j, txt in enumerate(texts):
            content = build_content(args.mode, txt, prompt_template, wrap_prompts, styles[j])
            final = qtok.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
            n_in = len(qtok(final, add_special_tokens=False).input_ids)
            status0 = n_in > args.input_drop
            st0 += status0
            if j == 0:
                print("=" * 70)
                print(f"[dry-run] shard {si} sample row 0 style={styles[j]} "
                      f"n_in={n_in} status0={status0}")
                print("---- templated prompt (first 1200 chars) ----")
                print(final[:1200])
                print("=" * 70)
        print(f"[dry-run] first 20 rows of shard {si}: status0={st0}/20", flush=True)
        return

    # ---- load model ----
    import torch
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len)
    gpu_name = torch.cuda.get_device_name(0)

    # ---- progress state (cumulative across this worker's run) ----
    prog = {
        "dataset": args.dataset_name, "worker_id": args.worker_id,
        "shards_completed": 0, "shards_total": len(owned),
        "docs_completed": 0, "docs_status_0": 0, "docs_status_1": 0, "docs_status_2": 0,
        "total_input_tokens": 0, "total_output_tokens": 0,
        "elapsed_seconds": 0.0, "last_updated": None,
    }
    # account for already-finished shards on resume
    done_already = sum(1 for si, _ in owned
                       if os.path.exists(os.path.join(out_dir, f"part_{si:05d}.parquet")))
    prog["shards_completed"] = done_already

    mon_buf = []                 # rolling buffer of recent docs for monitoring
    docs_since_monitor = 0
    mon_rng = np.random.default_rng([7, args.worker_id])
    t_start = time.perf_counter()
    processed_docs_this_run = 0

    for si, shard_path in owned:
        out_path = os.path.join(out_dir, f"part_{si:05d}.parquet")
        if os.path.exists(out_path):
            print(f"[w{args.worker_id}] shard {si:05d} exists -> skip", flush=True)
            continue

        table = pq.read_table(shard_path)
        n_rows = table.num_rows
        texts = table.column(args.text_col).to_pylist()
        # input doc-token counts: reuse existing tokens-llama2 if present, else compute
        if "tokens-llama2" in table.column_names:
            in_doc_tokens = table.column("tokens-llama2").to_pylist()
        else:
            in_doc_tokens = [n_llama(t) for t in texts]
        styles = assign_wrap_styles(si, n_rows) if args.mode == "wrap" else [None] * n_rows

        # build prompts, decide status=0 by templated length
        finals = [None] * n_rows
        n_in_list = [0] * n_rows
        keep_idx, keep_prompts, keep_sp = [], [], []
        for j in range(n_rows):
            content = build_content(args.mode, texts[j], prompt_template, wrap_prompts, styles[j])
            final = qtok.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
            n_in = len(qtok(final, add_special_tokens=False).input_ids)
            n_in_list[j] = n_in
            finals[j] = final
            if n_in > args.input_drop:
                continue  # status=0
            keep_idx.append(j)
            keep_prompts.append(final)
            # per-doc cap so prompt+output never exceeds max_model_len (no vLLM overflow)
            max_new = min(args.max_tokens, args.max_model_len - n_in)
            keep_sp.append(SamplingParams(temperature=0, top_p=1.0, max_tokens=max(1, max_new)))

        # output columns (defaults are the status=0 values)
        rewritten = [""] * n_rows
        rewritten_tokens = [0] * n_rows
        status = [0] * n_rows
        finish_reason = [""] * n_rows

        t0 = time.perf_counter()
        if keep_prompts:
            outputs = llm.generate(keep_prompts, keep_sp)
            for o, j in zip(outputs, keep_idx):
                g = o.outputs[0]
                rewritten[j] = g.text
                rewritten_tokens[j] = n_llama(g.text)
                finish_reason[j] = g.finish_reason or ""
                status[j] = 1 if g.finish_reason == "length" else 2
        dt = time.perf_counter() - t0

        # ---- write output shard atomically ----
        out_table = (table
                     .append_column("rewritten", pa.array(rewritten, type=pa.large_string()))
                     .append_column("rewritten_tokens", pa.array(rewritten_tokens, type=pa.int32()))
                     .append_column("status", pa.array(status, type=pa.int8()))
                     .append_column("finish_reason", pa.array(finish_reason, type=pa.large_string()))
                     .append_column("input_tokens_qwen", pa.array(n_in_list, type=pa.int32())))
        if args.mode == "wrap":
            out_table = out_table.append_column("wrap_style", pa.array(styles, type=pa.large_string()))
        tmp_path = out_path + ".tmp"
        pq.write_table(out_table, tmp_path)
        os.replace(tmp_path, out_path)

        # ---- update counters ----
        s0 = status.count(0); s1 = status.count(1); s2 = status.count(2)
        prog["shards_completed"] += 1
        prog["docs_completed"] += n_rows
        prog["docs_status_0"] += s0
        prog["docs_status_1"] += s1
        prog["docs_status_2"] += s2
        prog["total_input_tokens"] += sum(n_in_list)
        prog["total_output_tokens"] += sum(rewritten_tokens)
        prog["elapsed_seconds"] = time.perf_counter() - t_start
        write_progress(progress_path, prog)
        processed_docs_this_run += n_rows

        # ---- monitoring buffer + periodic sample ----
        for j in range(n_rows):
            mon_buf.append({"doc_id": (table.column("doc_id")[j].as_py()
                                       if "doc_id" in table.column_names else -1),
                            "status": status[j], "src": texts[j] or "", "out": rewritten[j],
                            "in_tokens": in_doc_tokens[j], "out_tokens": rewritten_tokens[j]})
        docs_since_monitor += n_rows
        if docs_since_monitor >= args.monitor_every:
            k = min(10, len(mon_buf))
            pick = mon_rng.choice(len(mon_buf), size=k, replace=False)
            append_monitor(args.monitor_file, args.dataset_name, args.worker_id,
                           [mon_buf[i] for i in pick])
            mon_buf = []
            docs_since_monitor = 0

        # ---- stdout progress summary ----
        elapsed = prog["elapsed_seconds"]
        dps = processed_docs_this_run / elapsed if elapsed > 0 else 0.0
        remaining_shards = prog["shards_total"] - prog["shards_completed"]
        avg_per_shard = processed_docs_this_run / max(1, (prog["shards_completed"] - done_already))
        eta_s = (remaining_shards * avg_per_shard / dps) if dps > 0 else 0.0
        print(f"[w{args.worker_id}] {args.dataset_name} shard {si:05d} done "
              f"({prog['shards_completed']}/{prog['shards_total']}) rows={n_rows} "
              f"s0={s0} s1={s1} s2={s2} wall={dt:.1f}s docs={prog['docs_completed']} "
              f"docs/s={dps:.1f} ETA={eta_s/3600:.1f}h GPU={gpu_name}", flush=True)

    print(f"[w{args.worker_id}] {args.dataset_name} ALL OWNED SHARDS DONE "
          f"({prog['shards_completed']}/{prog['shards_total']})", flush=True)


if __name__ == "__main__":
    main()
