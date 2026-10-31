"""SKIPJACK-ONLY registry of the 3B rewrite-600m grid: the historical Skipjack deployment (paths, Slurm site, GPU budget,
W&B owner, live 3-setting lanes) on top of the scientific contract.

The science (seed, settings, schedule, branch points, model, tokenizer hashes, corpus totals) is defined once in
deploy/portable_3b/science_3b.py and imported here unchanged; nothing scientific is defined in this file. Running the
grid anywhere else: deploy/portable_3b/ (kys3b.py + an operator config); it never imports this module.
Nothing in this file is shared with the 1.5B baseline grid (tools/kys_raw/registry.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'portable_3b'))

SEED = 42
SETTINGS = ['quality-first', 'diversity-oriented', 'disagreement-aware']      # 05_final/<setting>/shuffled; the train8 lanes
# Four-setting grid (owner, 2026-10-30): Quality-Base = the shared base + the next 10B along the same fastText order, raw
# (dataset/05_final/quality-base, prepared locally on CPU). It is PREPARED, not submittable: train8_3b.py, capped_submit_3b.py
# and the monitor iterate SETTINGS only, and its configs live in their own directory (cfg_dir) so the live 27-config gate
# (assert_invariants_3b --all, run in-job and by the train8 preflight) is unchanged. Only dataset/run/output identifiers
# differ from the rewritten arms (test_quality_base_3b.py). Moving it into SETTINGS is an owner decision at release.
PREPARED_SETTINGS = ['quality-base']
GRID_SETTINGS = SETTINGS + PREPARED_SETTINGS                                  # 4 settings x 3 annealed finals = 12
EXCLUDED = ['wrap-inspired', 'rewire-inspired']                              # never trained in this grid

# ---- paths (all new namespaces; none is shared with the 1.5B baselines) ----------------------------------
# The Skipjack layout. The KYS3B_* overrides exist for tests and historical tooling only; another site renders its configs
# with deploy/portable_3b/kys3b.py from its own operator config instead.
def _env_path(var: str, default: Path) -> Path:
    return Path(os.environ[var]) if os.environ.get(var) else default


DATA_ROOT_600M = _env_path('KYS3B_DATA_ROOT', Path('/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m'))
FINAL = DATA_ROOT_600M / 'dataset' / '05_final'                               # <setting>/shuffled/part_*.parquet
TRAIN_ROOT = DATA_ROOT_600M / 'training'                                     # tokenized/, tokenizer/, reports/, logs/, wandb/
TOKENIZED = _env_path('KYS3B_TOKENIZED', TRAIN_ROOT / 'tokenized')            # <setting>/NNNNN_unshuffled.ds{,.index,.metadata}
TOKENIZER = _env_path('KYS3B_TOKENIZER', TRAIN_ROOT / 'tokenizer')
# The exact tokenizer of the 1.5B models: the directory the 1.5B training data was tokenized with, pinned by the
# blab-jhu/KYS-Pre-Rewritten manifest.json. Its tokenizer.json is byte-identical to the one behind the 600M token counts
# (tokenizers/llama2-unsloth-tokenizer) and to the published 1.5B HF exports (rewrite-1p5b-hf/*/tokenizer.json).
TOKENIZER_SOURCE = Path('/weka/projects/bvandur1/zhuicon1/data/1p5_baseline/pretraining/raw/tokenizer')   # read only
REPORTS = TRAIN_ROOT / 'reports'
# Stage 05 validation that pins every shuffled part's sha256 (tokenize_3b.sbatch step 2)
ASSEMBLY_VALIDATION = {**{s: DATA_ROOT_600M / 'dataset' / 'reports' / 'stage04_assembly_20261016' / 'validate_assembly_scope.json'
                          for s in SETTINGS},
                       'quality-base': DATA_ROOT_600M / 'dataset' / 'reports' / 'quality_base_20261030' / 'validate_quality_base.json'}
STATE_ROOT = REPORTS / 'capped_submit'
LOG_DIR = TRAIN_ROOT / 'logs' / 'slurm'
WANDB_DIR = _env_path('KYS3B_WANDB_DIR', TRAIN_ROOT / 'wandb')
CKPT_BASE = _env_path('KYS3B_CKPT_BASE', Path('/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite3b_600m'))
INIT_ROOT = CKPT_BASE / 'init'                                               # _init_3B_seed42/0
CKPT_ROOT = CKPT_BASE / 'runs'                                               # seed42/<setting>/<trunk|epN>/<setting>/seed42/<step>
FORBIDDEN_PREFIXES = ['/weka/projects/bvandur1/zhuicon1/data/1p5_baseline', '/weka/projects/bvandur1/zhuicon1/checkpoints/1p5_baseline',
                      '/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b', '/weka/projects/bvandur1/zhuicon1/rewrite-3b/']

# ---- corpus, batch, schedule, model: the scientific contract, defined once in deploy/portable_3b/science_3b.py ----
# (the portable renderer builds Marc's configs through the same module, so the two cannot drift). Corpus notes:
# CORPUS_EMPTY_DOCS are rows whose text became empty in the Stage 04 strip (stage04_assembly_20261016/STATUS.md);
# corpus_train_tokens counts one EOS for each, datatrove skips them (tokenize job 1126253, 2026-10-17); quality-base
# is raw with 0 empty rows. diversity-oriented's shortfall vs 20B is accepted.
from science_3b import (  # noqa: E402
    ADAMW, BRANCH_KINDS, BRANCH_POINTS, CLIP_GRAD, CORPUS_EMPTY_DOCS, CORPUS_PARTS, CORPUS_ROWS, CORPUS_TRAIN_TOKENS,
    DEPENDS, END_STEP, EPOCH_TOKENS, FINAL_TOKENS, FIRST_STEP, GLOBAL_BATCH_SEQ, HORIZONS, INIT_STD, KINDS, MODEL,
    MODEL_1P5B, N_LAYERS, N_PARAMS, N_PARAMS_NON_EMBEDDING, PEAK_LR, SEQ_LEN, SHARED_BASE_TRAIN_TOKENS, STARTUP_STEPS,
    TOKENIZED_TOKENS, TOKENIZER_SHA256, TOKENS_PER_STEP, TRUNK_ENDS, TRUNK_KINDS, WARMUP_STEPS, WEIGHT_DECAY, slug,
    seg_name)
import science_3b as SCI  # noqa: E402

# ---- model: the 1.5B architecture with ONLY num_hidden_layers changed (28 -> 57) ------------------------------
# MODEL_1P5B is model.model_config of the 1.5B configs (configs/1.5B-baseline-seed42/filled/*, identical to the published
# checkpoints' model_config.json); assert_invariants_3b re-reads the baseline config and refuses any other difference.
# 57 layers is the depth closest to 3e9 parameters: per layer 4*d^2 + 3*d*ffn + 2*d = 51,384,320; fixed (tied 32000 x 2048
# embedding + final RMSNorm) = 65,538,048; total 57 x 51,384,320 + 65,538,048 = 2,994,444,288 (56 -> 2,943,059,968;
# 58 -> 3,045,828,608). Verified with transformers LlamaForCausalLM on the meta device.
# Init (unchanged rule, nanotron StandardParametrizator, init_method.scaling_method default NUM_LAYERS):
#   q/k/v, gate/up, tied embedding ~ N(0, 0.02); RMSNorm weights = 1;
#   row-parallel o_proj and down_proj ~ N(0, 0.02 / sqrt(2 * L)): L=28 -> 0.0026726, L=57 -> 0.0018732.
BASELINE_MODEL_CONFIG = Path('/weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys/configs/1.5B-baseline-seed42/filled/'
                             'raw_diversity_oriented_seed42_trunk1.yaml')       # read only
if not BASELINE_MODEL_CONFIG.is_file():   # off Skipjack: this branch's byte-identical copy (same commit 42c130ab)
    BASELINE_MODEL_CONFIG = Path(__file__).resolve().parents[2] / 'configs' / '1.5B-baseline-seed42' / 'filled' / BASELINE_MODEL_CONFIG.name

# ---- layout on 4 x H100 80GB (DP=4, TP=1, PP=1); mbs is fixed for all three settings ---------------------
DP = 4                      # Skipjack: one 4-GPU node per segment
TP, PP = SCI.TP, SCI.PP
LAYOUTS = {   # chosen by the smoke test; never mixed across settings (masked_mean is per micro-batch)
    'A': {'zero_stage': 1, 'micro_batch_size': 16, 'recompute_layer': True},   # default, estimated ~52 GiB peak
    'B': {'zero_stage': 0, 'micro_batch_size': 8, 'recompute_layer': True},    # fallback, estimated ~68 GiB peak
    'C': {'zero_stage': 1, 'micro_batch_size': 32, 'recompute_layer': True},   # optional, ~70.5 GiB: the 1.5B grid's mbs 32
}
DEFAULT_LAYOUT = 'A'

JOB_PREFIX = 'kys_3b_'          # kys_ so the 1.5B tool and this tool both count these GPUs against the shared 20
GPU_CEILING = 20
GPUS_PER_SEGMENT = 4

# ---- dedicated 8-GPU budget (owner authorization 2026-10-20; supersedes the shared-20 handoff path above) -----------
# Every 3B GPU job (startup validation, segments, retries, evaluation) is named rw3b_gpu_*: NOT kys_, so the 1.5B tool,
# whose submit refuses while any other kys_ job is live, keeps working, and its 20-GPU budget is not consumed.
# At most two 4-GPU jobs at once, enforced by two serial afterok lanes (train8_3b.py).
TRAIN8_PREFIX = 'rw3b_gpu_'
TRAIN8_GPU_CEILING = 8
TRAIN8_LANES = TRAIN8_GPU_CEILING // GPUS_PER_SEGMENT
TRAIN8_DIR = REPORTS / 'train8'
TRAIN8_STATE = TRAIN8_DIR / f'seed{SEED}'
STARTUP = f'3b_seed{SEED}_startup'      # bounded startup validation, run on quality-first trunk1 itself (steps 0 -> 30)
# W&B destination of the Skipjack runs (the Skipjack owner's account, online). Never used elsewhere: the portable
# renderer requires the operator's own entity and project and refuses this entity.
SKIPJACK_WANDB_ENTITY = 'imwytro-johns-hopkins-university'
WANDB_PROJECT = 'zhc-3b-600m-20b-wsd'
WANDB_ENTITY = SKIPJACK_WANDB_ENTITY

# ---- site (same account/QoS/partition as the 1.5B baselines; the 20-GPU cap is this project's, not the QoS's)
REPO = Path(__file__).resolve().parents[2]     # this checkout (on Skipjack the 3B worktree), never the active 1.5B checkout
CFG_ROOT = REPO / 'configs' / '3B-rewrite-600m'
SITE = {
    'repo_dir': str(REPO),
    'env_activate': '/weka/scratch/jhu/bvandur1/zhuicon1/envs/pretrain/bin/activate',   # read-only use of the shared env
    'python_include': '/weka/scratch/jhu/bvandur1/zhuicon1/basic/miniconda3/envs/agents/include/python3.11',
    'dp': DP, 'tp': TP, 'pp': PP,
    'slurm': {'partition': 'h100', 'account': 'bvandur1_blabscify', 'qos': 'jhu2', 'nodes': 1,
              'gres': 'gpu:h100:4', 'cpus_per_task': 32, 'time': '2-00:00:00'},
}
CPU_SLURM = {'partition': 'med', 'account': 'bvandur1_blabscify', 'qos': 'jhu2'}   # as the 600M Stage 04 CPU jobs


def cfg_dir(setting: str, seed: int = SEED) -> Path:
    """Rendered configs: the live grid's directory, or a separate one for a prepared (not yet submittable) setting."""
    assert setting in GRID_SETTINGS, setting
    return CFG_ROOT / (f'seed{seed}' if setting in SETTINGS else f'seed{seed}_prepared')


def ckpt_dir(setting: str, kind: str, seed: int = SEED) -> Path:
    return SCI.ckpt_dir(CKPT_ROOT, setting, kind, seed)


assert SEED == SCI.SEED and GRID_SETTINGS == SCI.GRID_SETTINGS, 'registry_3b and science_3b disagree on the grid'
