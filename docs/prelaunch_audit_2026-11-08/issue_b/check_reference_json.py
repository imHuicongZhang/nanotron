"""Issue B, independent check: kys-eval reference/rewritten_v2_54_per_task.json == the 54 v2 LightEval JSONs, bit for
bit (acc and acc_norm, every task key), and its recorded source sha256 == the files' sha256."""
import csv, hashlib, json, sys
REF = "/weka/scratch/jhu/bvandur1/zhuicon1/projects/kys-eval/reference/rewritten_v2_54_per_task.json"
FILES = "/weka/scratch/jhu/bvandur1/zhuicon1/projects/rewrite/docs/rebuttal_q1_paired_analysis/result_files_used.csv"
ref = {(c["setting"], c["seed"], c["epoch"]): c for c in json.load(open(REF))["cells"]}
n = bad = sha_ok = 0
for f in list(csv.DictReader(open(FILES)))[:54]:
    c = ref[(f["setting"], int(f["seed"]), "ep" + f["epoch"])]
    raw = open(f["result_json"], "rb").read()
    sha_ok += hashlib.sha256(raw).hexdigest() == c["source_sha256"]
    res = json.loads(raw)["results"]
    for k, v in res.items():
        if k == "all":
            continue
        t = c["tasks"][k.split("|")[0]]
        n += 1
        bad += (t["acc"] != v["acc"]) or (t["acc_norm"] != v["acc_norm"])
print(f"cells {len(ref)}; task entries compared {n}; not bit-identical {bad}; source sha256 match {sha_ok}/54")
sys.exit(1 if bad or sha_ok != 54 or len(ref) != 54 else 0)
