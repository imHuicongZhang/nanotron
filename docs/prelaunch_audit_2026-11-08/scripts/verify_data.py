#!/usr/bin/env python
"""§2 data verification for the 2026-11-08 rebuttal audit (CPU only).

Compares the local copies used by the audit against the published Hub metadata and recorded checksums:
  A. blab-jhu/KYS-Pre-Rewritten manifest.json (Hub, main) vs the local manifest.json
  B. raw_text/<setting>/*.parquet for the two rebuttal raw settings: sha256, bytes, rows vs manifest
  C. local tokenized raw shards: sum of .ds bytes / 2 and .ds.metadata line-2 total vs registry expected total
  D. signal-disagreement-lambda0/tokenized: sha256 vs local SHA256SUMS and Hub SHA256SUMS; token total vs registry
  E. tokenizer files: sha256 vs manifest
  F. Hub file listing for the three rebuttal settings (counts)
Writes a JSON summary to ../evidence/verify_data.json. Prints no credentials (uses the cached HF login only).

Usage: HF_HOME=/home/jhu/zhuicon1/.cache/huggingface python verify_data.py
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "evidence" / "verify_data.json"
RAW = Path("/home/jhu/zhuicon1/projects_bvandur1/zhuicon1/data/1p5_baseline/pretraining/raw")
L0 = Path("/home/jhu/zhuicon1/scratch_bvandur1/zhuicon1/datasets/rewrite/signal-disagreement-lambda0")
REPO = "blab-jhu/KYS-Pre-Rewritten"
EXPECTED = {"raw_diversity_oriented": (10_000_002_885, 16), "raw_disagreement_aware": (10_000_005_069, 16),
            "signal-disagreement-lambda0": (10_000_002_556, 26)}


def sha256(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(bs):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(*a, flush=True)


def main():
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    res = {}
    api = HfApi()
    info = api.dataset_info(REPO, files_metadata=True)
    res["hub_revision"] = info.sha
    hub_files = {s.rfilename: s for s in info.siblings}
    log("hub revision", info.sha, "files", len(hub_files))

    # A. manifest
    tmp = HERE.parent / "evidence" / "hub"
    hub_manifest = Path(hf_hub_download(REPO, "manifest.json", repo_type="dataset", revision=info.sha, local_dir=tmp))
    local_manifest = RAW / "manifest.json"
    res["A_manifest"] = {"hub_sha256": sha256(hub_manifest), "local_sha256": sha256(local_manifest)}
    res["A_manifest"]["identical"] = res["A_manifest"]["hub_sha256"] == res["A_manifest"]["local_sha256"]
    log("A manifest identical:", res["A_manifest"]["identical"])
    m = json.loads(hub_manifest.read_text())

    # B. parquet
    res["B_parquet"] = {}
    jobs = []
    for s in ("raw_diversity_oriented", "raw_disagreement_aware"):
        for name, meta in m["settings"][s]["files"].items():
            jobs.append((s, name, meta))
    def check(job):
        s, name, meta = job
        p = RAW / "raw_text" / s / name
        hf = hub_files.get(f"raw_text/{s}/{name}")
        hub_sha = getattr(getattr(hf, "lfs", None), "sha256", None) if hf else None
        if not p.is_file():
            return s, name, {"present": False}
        return s, name, {"present": True, "bytes_ok": p.stat().st_size == meta["bytes"],
                         "rows_ok": pq.ParquetFile(p).metadata.num_rows == meta["rows"],
                         "sha_ok": sha256(p) == meta["sha256"], "hub_lfs_sha_eq_manifest": hub_sha == meta["sha256"]}
    with ThreadPoolExecutor(8) as ex:
        for s, name, r in ex.map(check, jobs):
            res["B_parquet"].setdefault(s, {})[name] = r
    for s, d in res["B_parquet"].items():
        ok = all(all(v for v in r.values()) for r in d.values())
        rows = sum(m["settings"][s]["files"][n]["rows"] for n in d)
        res["B_parquet"][s] = {"all_ok": ok, "n_files": len(d), "manifest_rows": rows,
                               "final_merged_docs": m["settings"][s]["final_merged_docs"],
                               "expected_total_tokens": m["settings"][s]["expected_total_tokens"],
                               "failures": {n: r for n, r in d.items() if not all(r.values())}}
        log("B", s, res["B_parquet"][s]["all_ok"], len(d), rows)

    # C/D. tokenized totals
    def ds_totals(folder):
        ds = sorted(folder.glob("*.ds"))
        byte_tokens = sum(p.stat().st_size for p in ds) // 2
        meta_tokens = 0
        for p in ds:
            mp = Path(str(p) + ".metadata")
            meta_tokens += int(mp.read_text().splitlines()[1]) if mp.is_file() else 0
        return {"shards": len(ds), "bytes_div2": byte_tokens, "metadata_sum": meta_tokens}
    res["C_tokenized"] = {}
    for s in ("raw_diversity_oriented", "raw_disagreement_aware"):
        folder = RAW / s / "tokenized"
        sub = [d for d in folder.rglob("*") if d.is_dir()]
        cand = [folder] + sub
        best = max(cand, key=lambda d: len(list(d.glob("*.ds"))))
        t = ds_totals(best)
        t["folder"] = str(best)
        t["ok"] = (t["bytes_div2"], t["shards"]) == EXPECTED[s] == (t["metadata_sum"], t["shards"])
        res["C_tokenized"][s] = t
        log("C", s, t)

    tok = L0 / "tokenized"
    sums = {}
    for line in (L0 / "tokenized.SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, n = line.split(maxsplit=1)
            sums[n.strip().lstrip("*")] = h
    hub_sums_path = Path(hf_hub_download(REPO, "signal-disagreement-lambda0/tokenized/SHA256SUMS", repo_type="dataset",
                                         revision=info.sha, local_dir=tmp))
    hub_sums = {}
    for line in hub_sums_path.read_text().splitlines():
        if line.strip():
            h, n = line.split(maxsplit=1)
            hub_sums[n.strip().lstrip("*")] = h
    def l0check(item):
        n, h = item
        cands = [tok / n, L0 / n, tok / Path(n).name]
        p = next((c for c in cands if c.is_file()), None)
        return n, (None if p is None else sha256(p) == h)
    with ThreadPoolExecutor(8) as ex:
        l0 = dict(ex.map(l0check, sums.items()))
    names_local = {Path(n).name for n in sums}
    names_hub = {Path(n).name for n in hub_sums}
    same_hashes = all(hub_sums.get(n) == h or {Path(k).name: v for k, v in hub_sums.items()}.get(Path(n).name) == h
                      for n, h in sums.items())
    hub_listing = {f for f in hub_files if f.startswith("signal-disagreement-lambda0/tokenized/")}
    hub_lfs_vs_sums = {}
    for f in hub_listing:
        nm = Path(f).name
        hs = {Path(k).name: v for k, v in hub_sums.items()}.get(nm)
        lfs = getattr(getattr(hub_files[f], "lfs", None), "sha256", None)
        if hs is not None and lfs is not None:
            hub_lfs_vs_sums[nm] = lfs == hs
    t = ds_totals(tok)
    res["D_lambda0"] = {"local_sums_entries": len(sums), "local_files_match_local_sums": all(v is True for v in l0.values()),
                        "local_mismatch_or_missing": {n: v for n, v in l0.items() if v is not True},
                        "local_sums_eq_hub_sums": same_hashes and names_local == names_hub,
                        "hub_tokenized_files": len(hub_listing),
                        "hub_lfs_sha_eq_hub_sums": all(hub_lfs_vs_sums.values()) if hub_lfs_vs_sums else None,
                        "hub_lfs_checked": len(hub_lfs_vs_sums), **t,
                        "ok_total": (t["bytes_div2"], t["shards"]) == EXPECTED["signal-disagreement-lambda0"]}
    log("D", {k: v for k, v in res["D_lambda0"].items() if k != "local_mismatch_or_missing"})

    # E. tokenizer
    res["E_tokenizer"] = {n: sha256(RAW / "tokenizer" / n) == h for n, h in m["tokenizer"]["sha256"].items()
                          if (RAW / "tokenizer" / n).is_file()}
    res["E_tokenizer_hub_lfs"] = {n: getattr(getattr(hub_files.get(f"tokenizer/{n}"), "lfs", None), "sha256", None) == h
                                  for n, h in m["tokenizer"]["sha256"].items()}
    log("E", res["E_tokenizer"], res["E_tokenizer_hub_lfs"])

    # F. Hub listing
    res["F_hub_counts"] = {p: sum(1 for f in hub_files if f.startswith(p)) for p in
                           ("raw_text/raw_diversity_oriented/", "raw_text/raw_disagreement_aware/",
                            "signal-disagreement-lambda0/tokenized/", "tokenizer/")}
    log("F", res["F_hub_counts"])
    OUT.write_text(json.dumps(res, indent=1, default=str))
    log("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
