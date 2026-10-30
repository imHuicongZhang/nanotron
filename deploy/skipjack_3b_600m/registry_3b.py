"""Single source of truth for the 3B rewrite-600m grid (one seed, four settings, trunk + 3 decay branches).

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
# Stage 05 validation that pins every shuffled part's sha256 (tokenize_3b.sbatch step 2)
ASSEMBLY_VALIDATION = {**{s: DATA_ROOT_600M / 'dataset' / 'reports' / 'stage04_assembly_20261016' / 'validate_assembly_scope.json'
                          for s in SETTINGS},
                       'quality-base': DATA_ROOT_600M / 'dataset' / 'reports' / 'quality_base_20261030' / 'validate_quality_base.json'}
STATE_ROOT = REPORTS / 'capped_submit'
LOG_DIR = TRAIN_ROOT / 'logs' / 'slurm'
WANDB_DIR = TRAIN_ROOT / 'wandb'
CKPT_BASE = Path('/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite3b_600m')
INIT_ROOT = CKPT_BASE / 'init'                                               # _init_3B_seed42/0
CKPT_ROOT = CKPT_BASE / 'runs'                                               # seed42/<setting>/<trunk|epN>/<setting>/seed42/<step>
FORBIDDEN_PREFIXES = ['/weka/projects/bvandur1/zhuicon1/data/1p5_baseline', '/weka/projects/bvandur1/zhuicon1/checkpoints/1p5_baseline',
                      '/weka/projects/bvandur1/zhuicon1/checkpoints/rewrite-1p5b', '/weka/projects/bvandur1/zhuicon1/rewrite-3b/']

# ---- corpus (dataset/05_final/<setting>/_pretrain_manifest.json, validated 2026-10-17; quality-base 2026-10-30) ----
CORPUS_TRAIN_TOKENS = {'quality-first': 20_000_000_270, 'diversity-oriented': 19_834_314_433, 'disagreement-aware': 20_000_000_227,
                       'quality-base': 20_000_000_299}   # shared base 10,000,000,186 + next-10B fastText block 10,000,000,113
CORPUS_ROWS = {'quality-first': 32_341_160, 'diversity-oriented': 26_692_847, 'disagreement-aware': 26_743_280,
               'quality-base': 17_942_518}           # 10,527,887 + 7,414,631
CORPUS_PARTS = {'quality-first': 65, 'diversity-oriented': 54, 'disagreement-aware': 54, 'quality-base': 36}
# Rows whose text became empty in the Stage 04 strip (stage04_assembly_20261016/STATUS.md, "empty after"; retained with
# tokens_llama2 = 0). corpus_train_tokens counts one EOS for each; datatrove's reader skips a document without text
# (the same recipe as every KYS corpus), so the tokenized stream holds exactly corpus_train_tokens - empty documents,
# and datatrove reads CORPUS_ROWS - empty documents. Verified by tokenize job 1126253 (2026-10-17). Data unchanged.
# quality-base is raw (no strip): 0 zero-token rows in base or block by the verified flat token array; the Stage 05
# validator recounts empty text on every shuffled row, and tokenize_3b's exact-total check refuses any drift.
CORPUS_EMPTY_DOCS = {'quality-first': 27, 'diversity-oriented': 2, 'disagreement-aware': 8, 'quality-base': 0}
TOKENIZED_TOKENS = {s: CORPUS_TRAIN_TOKENS[s] - CORPUS_EMPTY_DOCS[s] for s in CORPUS_TRAIN_TOKENS}
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
STARTUP_STEPS = (20, 30)                 # save at 20, resume 20 -> 30 with optimizer + LR state
WANDB_PROJECT = 'zhc-3b-600m-20b-wsd'
WANDB_ENTITY = 'imwytro-johns-hopkins-university'

# ---- site (same account/QoS/partition as the 1.5B baselines; the 20-GPU cap is this project's, not the QoS's)
REPO = Path('/weka/scratch/jhu/bvandur1/zhuicon1/projects/nanotron-kys-3b600m')     # this worktree, never the active 1.5B checkout
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


def slug(setting: str) -> str:
    return setting.replace('-', '_')


def seg_name(setting: str, kind: str, seed: int = SEED) -> str:
    return f'3b_{slug(setting)}_seed{seed}_{kind}'


def cfg_dir(setting: str, seed: int = SEED) -> Path:
    """Rendered configs: the live grid's directory, or a separate one for a prepared (not yet submittable) setting."""
    assert setting in GRID_SETTINGS, setting
    return CFG_ROOT / (f'seed{seed}' if setting in SETTINGS else f'seed{seed}_prepared')


def ckpt_dir(setting: str, kind: str, seed: int = SEED) -> Path:
    phase = 'trunk' if kind.startswith('trunk') else kind
    return CKPT_ROOT / f'seed{seed}' / setting / phase / setting / f'seed{seed}'
