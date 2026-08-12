#!/usr/bin/env python3
"""
fix_ds_metadata.py — make datatrove .ds.metadata files point at the tokenizer DIRECTORY.

WHY (verified against nanotron 0.4):
  * datatrove preprocessing must be given the tokenizer.json FILE
    (datatrove/utils/tokenization.py::load_tokenizer -> Tokenizer.from_file; a directory errors
    "Is a directory"). datatrove writes that exact string into line 1 of every *.ds.metadata as
    "<tokenizer_string>|<token_size_bytes>".
  * nanotron training reads line 1 and calls AutoTokenizer.from_pretrained(<tokenizer_string>)
    (src/nanotron/config/config.py:192), which needs the tokenizer DIRECTORY (a .json file errors
    "Incorrect path_or_model_id"). It also asserts every *.ds.metadata across all dataset folders
    carries the IDENTICAL tokenizer string and token size (config.py:194-199).
  => No single local path satisfies both. Fix: preprocess with the .json file, then run this script
     to rewrite line 1's path part to the DIRECTORY (preserving the "|<token_size>" suffix), and
     train with the DIRECTORY in tokenizer.tokenizer_name_or_path.

USAGE:
  python3 tools/fix_ds_metadata.py \
      --output-folder <data_root>/<setting>/tokenized \
      --tokenizer-dir <data_root>/tokenizer

  This is a required step after downloading the published corpora — their metadata records the
  path they were tokenized under, not yours. See HANDOVER.md §9.2, which loops this over all six
  settings; assert_invariants.py fails the preflight until it has been run.

  Pass the SAME --tokenizer-dir to every block folder so the cross-folder assert holds.
  Idempotent: re-running on already-fixed files changes nothing. No third-party deps (stdlib only).
"""

import argparse
import glob
import os
import sys


def parse_line1(line1: str):
    """Return (path_part, size_part) by splitting on the LAST '|'. Raises if no '|'."""
    if "|" not in line1:
        raise ValueError(f"metadata line 1 has no '|' separator: {line1!r}")
    path_part, size_part = line1.rsplit("|", 1)
    if not size_part.isdigit():
        raise ValueError(f"token-size suffix after '|' is not an integer: {size_part!r} (line: {line1!r})")
    return path_part, size_part


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-folder", required=True,
                    help="Tokenized output dir containing *.ds.metadata files (one block).")
    ap.add_argument("--tokenizer-dir", required=True,
                    help="FIXED tokenizer DIRECTORY string to write as the path part of line 1 "
                         "(e.g. <data_root>/tokenizer). No trailing slash, not the .json.")
    args = ap.parse_args()

    tok_dir = args.tokenizer_dir.rstrip("/")
    if tok_dir.endswith(".json"):
        sys.exit(f"[ERROR] --tokenizer-dir must be the DIRECTORY, not a .json file: {args.tokenizer_dir}")
    if not os.path.isdir(args.output_folder):
        sys.exit(f"[ERROR] --output-folder is not a directory: {args.output_folder}")

    meta_files = sorted(glob.glob(os.path.join(args.output_folder, "*.ds.metadata")))
    if not meta_files:
        sys.exit(f"[ERROR] no *.ds.metadata files found under {args.output_folder}")

    print(f"[INFO] {len(meta_files)} metadata file(s) in {args.output_folder}")
    print(f"[INFO] target tokenizer dir: {tok_dir}")

    # --- rewrite pass ---
    for path in meta_files:
        with open(path, "r") as f:
            content = f.read()
        lines = content.split("\n")           # round-trips a trailing newline exactly
        old_path, size_part = parse_line1(lines[0])
        new_line1 = f"{tok_dir}|{size_part}"
        if lines[0] == new_line1:
            print(f"[SKIP] already fixed: {os.path.basename(path)}  (|{size_part})")
            continue
        lines[0] = new_line1
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"[FIX ] {os.path.basename(path)}: {old_path!r} -> {tok_dir!r}  (kept |{size_part})")

    # --- verify pass: re-read everything and assert ---
    first_lines = []
    for path in meta_files:
        with open(path, "r") as f:
            line1 = f.readline().rstrip("\n")
        path_part, size_part = parse_line1(line1)   # also re-asserts the |<int> suffix survived
        assert path_part == tok_dir, (
            f"[VERIFY FAIL] {path}: path part is {path_part!r}, expected {tok_dir!r}"
        )
        first_lines.append((path, line1))

    # byte-identical line 1 across ALL files in this folder (config.py:194-196 requires it)
    ref_path, ref_line1 = first_lines[0]
    for path, line1 in first_lines[1:]:
        assert line1 == ref_line1, (
            f"[VERIFY FAIL] line 1 differs:\n  {ref_path}: {ref_line1!r}\n  {path}: {line1!r}"
        )

    ref_size = ref_line1.rsplit("|", 1)[1]
    print(f"[OK] all {len(meta_files)} metadata line-1 byte-identical: {ref_line1!r}  (token_size={ref_size})")
    print("[OK] tokenizer-dir applied and |<token_size> suffix preserved.")


if __name__ == "__main__":
    main()
