#!/usr/bin/env python
"""Does the transformers version of the conversion environment change what kys-eval scores?

Step 1 (run in the nanotron training env, transformers 4.46.3):
    python convert_env_check.py convert
  converts the same v2 nanotron checkpoint as convert_roundtrip.py into ../work/converted_hf_tf446.
Step 2 (run in the kys-eval venv, transformers 5.6.2, the scoring env):
    python convert_env_check.py compare
  compares ../work/converted_hf_tf446 with the released v2 export: file bytes; tensors; the loaded config
  (AutoConfig.to_dict without version fields); the loaded tokenizer (special tokens, add_bos/eos behaviour, padding
  side, and the encodings of 2,000 real evaluation strings from the reference results' LightEval details are not
  available offline, so a fixed sample of strings is used: every MMLU/commonsense prompt template shape, unicode,
  whitespace, digits). Writes ../evidence/convert_env_check.json.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NANO = HERE.parents[2]
WORK = HERE.parent / "work"
EV = HERE.parent / "evidence"
CELL = "rewrite-1p5b/seed42/diversity_oriented/ep3"
TOK = Path("/home/jhu/zhuicon1/projects_bvandur1/zhuicon1/data/1p5_baseline/pretraining/raw/tokenizer")
OUT = WORK / "converted_hf_tf446"
REF = WORK / "v2" / CELL / "hf"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def convert():
    import transformers
    r = subprocess.run([sys.executable, NANO / "tools/kys_eval/convert_to_hf.py", "--checkpoint_path",
                        WORK / "v2" / CELL / "nanotron", "--save_path", OUT, "--tokenizer", TOK], capture_output=True, text=True)
    print("transformers", transformers.__version__, "rc", r.returncode, (r.stdout + r.stderr)[-800:])
    return r.returncode


STRINGS = [
    "Question: What is the capital of France?\nAnswer:", " Paris", " paris", "Paris",
    "The following are multiple choice questions (with answers) about abstract algebra.\n\nFind all c in Z_3",
    " A. 0\n B. 1\n C. 2\n D. 3\nAnswer:", " A", " B", "A man is sitting on a roof. He", " starts pulling up roofing",
    "To remove a stain from a shirt, you should", "Ünïcödé — “quotes” and emoji 😀 tabs\tand  double  spaces",
    "1234567890 3.14159 -42", "\n\n\n", " ", "", "</s>", "<s> text", "Carson relocated somewhere new. How would you "
    "describe Carson?", " mobile", " anxious", "Which of the following is a renewable resource?",
]


def compare():
    import torch  # noqa: F401
    import transformers
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    res = {"scoring_transformers": transformers.__version__}
    res["file_sha_equal"] = {n: sha(OUT / n) == sha(REF / n) for n in sorted(p.name for p in REF.iterdir())
                             if (OUT / n).exists()}
    res["files_converted"] = sorted(p.name for p in OUT.iterdir())
    a, b = safe_open(OUT / "model.safetensors", "pt"), safe_open(REF / "model.safetensors", "pt")
    res["tensor_names_equal"] = set(a.keys()) == set(b.keys())
    res["tensor_value_diffs"] = [k for k in sorted(set(a.keys()) & set(b.keys()))
                                 if a.get_tensor(k).dtype != b.get_tensor(k).dtype or not bool((a.get_tensor(k) == b.get_tensor(k)).all())]
    drop = {"transformers_version", "_name_or_path"}
    ca = {k: v for k, v in AutoConfig.from_pretrained(OUT).to_dict().items() if k not in drop}
    cb = {k: v for k, v in AutoConfig.from_pretrained(REF).to_dict().items() if k not in drop}
    res["loaded_config_diff"] = {k: (ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)}
    ta, tb = AutoTokenizer.from_pretrained(OUT), AutoTokenizer.from_pretrained(REF)
    attrs = ["bos_token", "eos_token", "unk_token", "pad_token", "padding_side", "truncation_side", "model_max_length",
             "add_bos_token", "add_eos_token", "vocab_size"]
    res["tokenizer_attr_diff"] = {k: (getattr(ta, k, None), getattr(tb, k, None)) for k in attrs
                                  if getattr(ta, k, None) != getattr(tb, k, None)}
    res["tokenizer_class"] = (type(ta).__name__, type(tb).__name__)
    enc_diffs = []
    for s in STRINGS:
        for kw in ({}, {"add_special_tokens": False}):
            if ta(s, **kw)["input_ids"] != tb(s, **kw)["input_ids"]:
                enc_diffs.append((s, kw))
    pair = ta(["Question: a", "longer question text here"], padding=True)["input_ids"] == \
        tb(["Question: a", "longer question text here"], padding=True)["input_ids"]
    res["encoding_diffs"] = enc_diffs
    res["batch_padding_equal"] = pair
    res["n_strings"] = len(STRINGS)
    print(json.dumps(res, indent=1, default=str))
    (EV / "convert_env_check.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(convert() if sys.argv[1] == "convert" else compare())
