"""Single source of truth for the 3B rewrite-600m grid (one seed, three settings, trunk + 3 decay branches).

Everything scientific that the configs, the submitter, the invariant check and the runbook use is defined
here once. Nothing in this file is shared with the 1.5B baseline grid (tools/kys_raw/registry.py).

Step schedule (derived, not copied):
    tokens/step  T  = 1024 seq x 2048 = 2,097,152
    horizon k    H_k = round(k * 20e9 / T)               -> 9537 / 19073 / 28610
    branch point B_k = round(0.9 * k * 20e9 / T)         -> 8583 / 17166 / 25749
    decay        H_k - B_k                               -> 954 / 1907 / 2861 (the final ~10%)
Applied to the 1.5B grid (10e9 per epoch) the same two formulas give exactly its constants
4768/9537/14305 and 4292/8583/12875, so this is the 1.5B rule, scaled by tokens per epoch.

The trunk is cut into six segments (<= 4292 steps each, so one segment fits the 48 h job limit up to
~39 s/it). Every branch point is a segment end, so each branch resumes from a step directory the trunk
writes with save_final_state.
"""
from __future__ import annotations

from pathlib import Path

SEED = 42
SETTINGS = ['quality-first', 'diversity-oriented', 'disagreement-aware']      # 05_final/<setting>/shuffled
EXCLUDED = ['quality-base', 'wrap-inspired', 'rewire-inspired']              # never trained in this grid

# ---- paths (all new namespaces; none is shared with the 1.5B baselines) ----------------------------------
DATA_ROOT_600M = Path('/weka/projects/bvandur1/zhuicon1/rewrite-3b-600m')
FINAL = DATA_ROOT_600M / 'dataset' / '05_final'                               # <setting>/shuffled/part_*.parquet
TRAIN_ROOT = DATA_ROOT_600M / 'training'                                     # tokenized/, tokenizer/, reports/, logs/, wandb/
TOKENIZED = TRAIN_ROOT / 'tokenized'
TOKENIZER = TRAIN_ROOT / 'tokenizer'
# The exact tokenizer of the 1.5B models: the directory the 1.5B training data was tokenized with, pinned by the
# blab-jhu/KYS-Pre-Rewritten manifest.json. Its tokenizer.json is byte-identical to the one behind the 600M token counts
# (tokenizers/llama2-unsloth-tokenizer) and to the published 1.5B HF exports (rewrite-1p5b-hf/*/tokenizer.json).
TOKENIZER_SOURCE = Path('/weka/projects/bvandur1/zhuicon1/data/1p5_baseline/pretraining/raw/tokenizer')   # read only
TOKENIZER_SHA256 = {
    'tokenizer.json': '81bb383c6138159665cd350a59fbcb5aa1fe65712cf5ed9b6481bc802cdbd8e5',
    'tokenizer_config.json': '2bffa41435e22dde97e37b2c5b97e13dbfcd1e41e921f2051bef14623b474fc2',
    'tokenizer_report.json': '4dcd1212911e7acfd59068eab1b96ed8077584fecded75fe0335035d9913c8c1',
}
REPORTS = TRAIN_ROOT / 'reports'
STATE_ROOT = REPORTS / 'capped_submit'
LOG_DIR = TRAIN_ROOT / 'logs' / 'slurm'
WANDB_DIR = TRAIN_ROOT / 'wandb'
CKPT_BASE = Path('/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite3b_600m')
INIT_ROOT = CKPT_BASE / 'init'                                               # _init_3B_seed42/0
CKPT_ROOT = CKPT_BASE / 'runs'                                               # seed42/<setting>/<trunk|epN>/<setting>/seed42/<step>
FORBIDDEN_PREFIXES = ['/weka/projects/bvandur1/zhuicon1/data/1p5_baseline', '/weka/projects/bvandur1/zhuicon1/checkpoints/1p5_baseline',
                      '/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b', '/weka/projects/bvandur1/zhuicon1/rewrite-3b/']

# ---- corpus (dataset/05_final/<setting>/_pretrain_manifest.json, validated 2026-10-17) -------------------
CORPUS_TRAIN_TOKENS = {'quality-first': 20_000_000_270, 'diversity-oriented': 19_834_314_433, 'disagreement-aware': 20_000_000_227}
CORPUS_ROWS = {'quality-first': 32_341_160, 'diversity-oriented': 26_692_847, 'disagreement-aware': 26_743_280}
CORPUS_PARTS = {'quality-first': 65, 'diversity-oriented': 54, 'disagreement-aware': 54}
SHARED_BASE_TRAIN_TOKENS = 10_000_000_186

# ---- batch and schedule ---------------------------------------------------------------------------------
SEQ_LEN = 2048
GLOBAL_BATCH_SEQ = 1024
TOKENS_PER_STEP = GLOBAL_BATCH_SEQ * SEQ_LEN                                 # 2,097,152
EPOCH_TOKENS = 20_000_000_000
HORIZONS = [round(k * EPOCH_TOKENS / TOKENS_PER_STEP) for k in (1, 2, 3)]    # [9537, 19073, 28610]
BRANCH_POINTS = [round(0.9 * k * EPOCH_TOKENS / TOKENS_PER_STEP) for k in (1, 2, 3)]   # [8583, 17166, 25749]
TRUNK_ENDS = [4292, 8583, 12875, 17166, 21458, 25749]                        # trunk1..trunk6 absolute end steps
PEAK_LR = 4.0e-4
WARMUP_STEPS = 1000

TRUNK_KINDS = [f'trunk{i}' for i in range(1, len(TRUNK_ENDS) + 1)]
BRANCH_KINDS = ['ep1', 'ep2', 'ep3']
KINDS = TRUNK_KINDS + BRANCH_KINDS
END_STEP = {**{k: e for k, e in zip(TRUNK_KINDS, TRUNK_ENDS)}, **{f'ep{i + 1}': h for i, h in enumerate(HORIZONS)}}
FIRST_STEP = {**{k: s for k, s in zip(TRUNK_KINDS, [0] + TRUNK_ENDS[:-1])},
              **{f'ep{i + 1}': b for i, b in enumerate(BRANCH_POINTS)}}
DEPENDS = {**{k: (TRUNK_KINDS[i - 1] if i else None) for i, k in enumerate(TRUNK_KINDS)},
           **{f'ep{i + 1}': TRUNK_KINDS[TRUNK_ENDS.index(b)] for i, b in enumerate(BRANCH_POINTS)}}   # ep1<-trunk2, ep2<-trunk4, ep3<-trunk6

assert HORIZONS == [9537, 19073, 28610] and BRANCH_POINTS == [8583, 17166, 25749]
assert all(b in TRUNK_ENDS for b in BRANCH_POINTS) and TRUNK_ENDS[-1] == BRANCH_POINTS[-1]
assert max(e - s for s, e in zip([0] + TRUNK_ENDS[:-1], TRUNK_ENDS)) == 4292

# ---- model: the 1.5B architecture with ONLY num_hidden_layers changed (28 -> 57) ------------------------------
# MODEL_1P5B is model.model_config of the 1.5B configs (configs/1.5B-baseline-seed42/filled/*, identical to the published
# checkpoints' model_config.json); assert_invariants_3b re-reads the baseline config and refuses any other difference.
# 57 layers is the depth closest to 3e9 parameters: per layer 4*d^2 + 3*d*ffn + 2*d = 51,384,320; fixed (tied 32000 x 2048
# embedding + final RMSNorm) = 65,538,048; total 57 x 51,384,320 + 65,538,048 = 2,994,444,288 (56 -> 2,943,059,968;
# 58 -> 3,045,828,608). Verified with transformers LlamaForCausalLM on the meta device.
MODEL_1P5B = {
    'is_llama_config': True, 'hidden_size': 2048, 'num_hidden_layers': 28, 'num_attention_heads': 16,
    'num_key_value_heads': 16, 'intermediate_size': 5632, 'vocab_size': 32000, 'tie_word_embeddings': True,
    'hidden_act': 'silu', 'max_position_embeddings': SEQ_LEN, 'rope_theta': 10000.0, 'rope_interleaved': False,
    'rms_norm_eps': 1.0e-05, 'attention_bias': False, 'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': None,
    'pretraining_tp': 1, 'use_cache': False,
}
N_LAYERS = 57
MODEL = {**MODEL_1P5B, 'num_hidden_layers': N_LAYERS}
N_PARAMS = 2_994_444_288
N_PARAMS_NON_EMBEDDING = 2_928_908_288
assert N_PARAMS == N_LAYERS * (4 * 2048 ** 2 + 3 * 2048 * 5632 + 2 * 2048) + 32000 * 2048 + 2048
# Init (unchanged rule, nanotron StandardParametrizator, init_method.scaling_method default NUM_LAYERS):
#   q/k/v, gate/up, tied embedding ~ N(0, 0.02); RMSNorm weights = 1;
#   row-parallel o_proj and down_proj ~ N(0, 0.02 / sqrt(2 * L)): L=28 -> 0.0026726, L=57 -> 0.0018732.
INIT_STD = 0.02
BASELINE_MODEL_CONFIG = Path('/weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys/configs/1.5B-baseline-seed42/filled/'
                             'raw_diversity_oriented_seed42_trunk1.yaml')       # read only

# ---- layout on 4 x H100 80GB (DP=4, TP=1, PP=1); mbs is fixed for all three settings ---------------------
DP, TP, PP = 4, 1, 1
LAYOUTS = {   # chosen by the smoke test; never mixed across settings (masked_mean is per micro-batch)
    'A': {'zero_stage': 1, 'micro_batch_size': 16, 'recompute_layer': True},   # default, estimated ~52 GiB peak
    'B': {'zero_stage': 0, 'micro_batch_size': 8, 'recompute_layer': True},    # fallback, estimated ~68 GiB peak
    'C': {'zero_stage': 1, 'micro_batch_size': 32, 'recompute_layer': True},   # optional, ~70.5 GiB: the 1.5B grid's mbs 32
}
DEFAULT_LAYOUT = 'A'

JOB_PREFIX = 'kys_3b_'          # kys_ so the 1.5B tool and this tool both count these GPUs against the shared 20
GPU_CEILING = 20
GPUS_PER_SEGMENT = 4
WANDB_PROJECT = 'zhc-3b-600m-20b-wsd'
WANDB_ENTITY = 'imwytro-johns-hopkins-university'

# ---- site (same account/QoS/partition as the 1.5B baselines; the 20-GPU cap is this project's, not the QoS's)
REPO = Path('/weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys-3b600m')     # this worktree, never the active 1.5B checkout
SITE = {
    'repo_dir': str(REPO),
    'env_activate': '/weka/scratch/jhu/bvandur1/zhuicon1/envs/pretrain/bin/activate',   # read-only use of the shared env
    'python_include': '/weka/scratch/jhu/bvandur1/zhuicon1/basic/miniconda3/envs/agents/include/python3.11',
    'dp': DP, 'tp': TP, 'pp': PP,
    'slurm': {'partition': 'h100', 'account': 'bvandur1_blabscify', 'qos': 'jhu2', 'nodes': 1,
              'gres': 'gpu:h100:4', 'cpus_per_task': 32, 'time': '2-00:00:00'},
}
CPU_SLURM = {'partition': 'med', 'account': 'bvandur1_blabscify', 'qos': 'jhu2'}   # as the 600M Stage 04 CPU jobs


def slug(setting: str) -> str:
    return setting.replace('-', '_')


def seg_name(setting: str, kind: str, seed: int = SEED) -> str:
    return f'3b_{slug(setting)}_seed{seed}_{kind}'


def ckpt_dir(setting: str, kind: str, seed: int = SEED) -> Path:
    phase = 'trunk' if kind.startswith('trunk') else kind
    return CKPT_ROOT / f'seed{seed}' / setting / phase / setting / f'seed{seed}'
