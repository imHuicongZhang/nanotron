"""The scientific contract of the KYS 3B rewrite-600m grid: one seed, four settings, trunk + 3 decay branches.

Pure constants and one config builder. No paths, no site, no W&B identity, no hardware: build_config() takes the
operator's paths and deployment explicitly. Both the portable renderer (kys3b.py) and the historical Skipjack
generator (deploy/skipjack_3b_600m/generate_configs_3b.py, via registry_3b.py) build every config through it, so
the science cannot drift between them.

Step schedule (derived, not copied):
    tokens/step  T  = 1024 seq x 2048 = 2,097,152
    horizon k    H_k = round(k * 20e9 / T)               -> 9537 / 19073 / 28610
    branch point B_k = round(0.9 * k * 20e9 / T)         -> 8583 / 17166 / 25749
    decay        H_k - B_k                               -> 954 / 1907 / 2861 (the final ~10%)
The trunk is cut into six segments (<= 4292 steps each). Every branch point is a segment end, so each branch
resumes from a step directory the trunk writes with save_final_state.

Per segment:
  trunk1        weights-only load of the shared seed-42 init (seeded into trunk/0); writes trunk/<step>
  trunk2..6     resume optimizer + LR state + data position from trunk/latest.txt
  ep1/ep2/ep3   resume from the trunk STEP directory of their branch point (8583 / 17166 / 25749) with optimizer +
                LR state, decay linearly to 0, write to their own ep<N>/ namespace: branch weights never feed back
                into the trunk.
Trunk segments carry ep3's decay parameters (start 25749, 2861 steps) and stop at or before 25749, so their LR is
warmup-then-constant.
"""
from __future__ import annotations

import copy
from pathlib import Path

SEED = 42
# The four settings, in release order. All four are trained in the portable grid (4 x 3 annealed finals = 12).
GRID_SETTINGS = ['quality-first', 'diversity-oriented', 'disagreement-aware', 'quality-base']

# ---- tokenizer: the exact 1.5B KYS llama-2 tokenizer, as published in tokenizer/ of the dataset ----------------------
TOKENIZER_SHA256 = {
    'tokenizer.json': '81bb383c6138159665cd350a59fbcb5aa1fe65712cf5ed9b6481bc802cdbd8e5',
    'tokenizer_config.json': '2bffa41435e22dde97e37b2c5b97e13dbfcd1e41e921f2051bef14623b474fc2',
    'tokenizer_report.json': '4dcd1212911e7acfd59068eab1b96ed8077584fecded75fe0335035d9913c8c1',
}

# ---- corpus (validated 2026-10-17; quality-base 2026-10-30) ---------------------------------------------------------
CORPUS_TRAIN_TOKENS = {'quality-first': 20_000_000_270, 'diversity-oriented': 19_834_314_433,
                       'disagreement-aware': 20_000_000_227, 'quality-base': 20_000_000_299}
CORPUS_ROWS = {'quality-first': 32_341_160, 'diversity-oriented': 26_692_847, 'disagreement-aware': 26_743_280,
               'quality-base': 17_942_518}
CORPUS_PARTS = {'quality-first': 65, 'diversity-oriented': 54, 'disagreement-aware': 54, 'quality-base': 36}
# Rows with empty text: counted as one EOS in corpus_train_tokens, skipped by datatrove's tokenizer, so the tokenized
# stream holds corpus_train_tokens - empty documents. diversity-oriented's 165,685,567-token shortfall is accepted.
CORPUS_EMPTY_DOCS = {'quality-first': 27, 'diversity-oriented': 2, 'disagreement-aware': 8, 'quality-base': 0}
TOKENIZED_TOKENS = {s: CORPUS_TRAIN_TOKENS[s] - CORPUS_EMPTY_DOCS[s] for s in CORPUS_TRAIN_TOKENS}
SHARED_BASE_TRAIN_TOKENS = 10_000_000_186

# ---- batch and schedule ---------------------------------------------------------------------------------------------
SEQ_LEN = 2048
GLOBAL_BATCH_SEQ = 1024
TOKENS_PER_STEP = GLOBAL_BATCH_SEQ * SEQ_LEN                                 # 2,097,152
EPOCH_TOKENS = 20_000_000_000
HORIZONS = [round(k * EPOCH_TOKENS / TOKENS_PER_STEP) for k in (1, 2, 3)]    # [9537, 19073, 28610]
BRANCH_POINTS = [round(0.9 * k * EPOCH_TOKENS / TOKENS_PER_STEP) for k in (1, 2, 3)]   # [8583, 17166, 25749]
TRUNK_ENDS = [4292, 8583, 12875, 17166, 21458, 25749]                        # trunk1..trunk6 absolute end steps
PEAK_LR = 4.0e-4
WARMUP_STEPS = 1000
ADAMW = {'adam_beta1': 0.9, 'adam_beta2': 0.95, 'adam_eps': 1.0e-8, 'name': 'adamW', 'torch_adam_is_fused': True}
WEIGHT_DECAY = 0.1
CLIP_GRAD = 1.0

TRUNK_KINDS = [f'trunk{i}' for i in range(1, len(TRUNK_ENDS) + 1)]
BRANCH_KINDS = ['ep1', 'ep2', 'ep3']
KINDS = TRUNK_KINDS + BRANCH_KINDS
END_STEP = {**{k: e for k, e in zip(TRUNK_KINDS, TRUNK_ENDS)}, **{f'ep{i + 1}': h for i, h in enumerate(HORIZONS)}}
FIRST_STEP = {**{k: s for k, s in zip(TRUNK_KINDS, [0] + TRUNK_ENDS[:-1])},
              **{f'ep{i + 1}': b for i, b in enumerate(BRANCH_POINTS)}}
DEPENDS = {**{k: (TRUNK_KINDS[i - 1] if i else None) for i, k in enumerate(TRUNK_KINDS)},
           **{f'ep{i + 1}': TRUNK_KINDS[TRUNK_ENDS.index(b)] for i, b in enumerate(BRANCH_POINTS)}}   # ep1<-trunk2, ep2<-trunk4, ep3<-trunk6
FINAL_TOKENS = [h * TOKENS_PER_STEP for h in HORIZONS]                      # 20,000,538,624 / 39,998,980,096 / 59,999,518,720

assert HORIZONS == [9537, 19073, 28610] and BRANCH_POINTS == [8583, 17166, 25749]
assert FINAL_TOKENS == [20_000_538_624, 39_998_980_096, 59_999_518_720]
assert all(b in TRUNK_ENDS for b in BRANCH_POINTS) and TRUNK_ENDS[-1] == BRANCH_POINTS[-1]
assert max(e - s for s, e in zip([0] + TRUNK_ENDS[:-1], TRUNK_ENDS)) == 4292

# ---- model: the 1.5B architecture with ONLY num_hidden_layers changed (28 -> 57) --------------------------------------
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
INIT_STD = 0.02
MODEL_BLOCK_FIXED = {'ddp_bucket_cap_mb': 25, 'dtype': 'bfloat16', 'make_vocab_size_divisible_by': 1}

# ---- parallelism: tensor and pipeline parallelism are part of the established setup ----------------------------------
TP, PP = 1, 1
TRUNK_CKPT_INTERVAL = 1000       # trunk checkpoints every 1000 updates; segment ends are saved regardless
BRANCH_CKPT_INTERVAL = 100000    # branches save only their final state (a restart re-runs the branch from its source)
STARTUP_STEPS = (20, 30)         # bounded startup validation: save at 20, resume 20 -> 30 with optimizer + LR state
INIT_DIRNAME = f'_init_3B_seed{SEED}'


def slug(setting: str) -> str:
    return setting.replace('-', '_')


def seg_name(setting: str, kind: str, seed: int = SEED) -> str:
    return f'3b_{slug(setting)}_seed{seed}_{kind}'


def ckpt_dir(runs_root, setting: str, kind: str, seed: int = SEED) -> Path:
    """<runs_root>/seed42/<setting>/<trunk|epN>/<setting>/seed42: one trunk namespace per setting (all six trunk
    segments) and one per branch, so branch outputs never land in the trunk's folder."""
    phase = 'trunk' if kind.startswith('trunk') else kind
    return Path(runs_root) / f'seed{seed}' / setting / phase / setting / f'seed{seed}'


def accumulation(micro_batch_size: int, dp: int) -> int:
    """Gradient accumulation such that micro_batch_size x accumulation x dp = 1024 sequences exactly."""
    if not (isinstance(micro_batch_size, int) and isinstance(dp, int)) or micro_batch_size < 1 or dp < 1:
        raise ValueError(f'micro_batch_size and data-parallel size must be positive integers (got {micro_batch_size!r}, {dp!r})')
    if GLOBAL_BATCH_SEQ % (micro_batch_size * dp):
        raise ValueError(f'micro_batch_size {micro_batch_size} x data-parallel size {dp} = {micro_batch_size * dp} does not '
                         f'divide the global batch of {GLOBAL_BATCH_SEQ} sequences; choose values whose product divides '
                         f'{GLOBAL_BATCH_SEQ} (e.g. dp 8 x mbs 8 -> accumulation 16)')
    return GLOBAL_BATCH_SEQ // (micro_batch_size * dp)


def build_config(setting: str, kind: str, *, tokenized_root, tokenizer_dir, runs_root, project: str,
                 dp: int, micro_batch_size: int, zero_stage: int, recompute_layer: bool) -> dict:
    """The nanotron config of one segment. Only the arguments are deployment- or site-specific; everything else is
    the scientific contract above."""
    assert setting in GRID_SETTINGS and kind in KINDS, (setting, kind)
    if zero_stage not in (0, 1):
        raise ValueError(f'zero_stage must be 0 or 1 (got {zero_stage!r})')
    if kind.startswith('trunk'):
        decay_start, decay_steps = BRANCH_POINTS[-1], HORIZONS[-1] - BRANCH_POINTS[-1]
        resume = str(ckpt_dir(runs_root, setting, kind))
        interval = TRUNK_CKPT_INTERVAL
    else:
        i = int(kind[-1]) - 1
        decay_start, decay_steps = BRANCH_POINTS[i], HORIZONS[i] - BRANCH_POINTS[i]
        resume = str(ckpt_dir(runs_root, setting, 'trunk1') / str(decay_start))
        interval = BRANCH_CKPT_INTERVAL
    accum = accumulation(micro_batch_size, dp)
    return {
        'general': {'project': project, 'run': seg_name(setting, kind), 'seed': SEED},
        'model': {'ddp_bucket_cap_mb': MODEL_BLOCK_FIXED['ddp_bucket_cap_mb'], 'dtype': MODEL_BLOCK_FIXED['dtype'],
                  'init_method': {'std': INIT_STD}, 'make_vocab_size_divisible_by': MODEL_BLOCK_FIXED['make_vocab_size_divisible_by'],
                  'model_config': dict(MODEL)},
        'tokenizer': {'tokenizer_name_or_path': str(tokenizer_dir)},
        'tokens': {'train_steps': END_STEP[kind], 'val_check_interval': -1, 'limit_val_batches': 0,
                   'limit_test_batches': 0, 'micro_batch_size': micro_batch_size, 'batch_accumulation_per_replica': accum,
                   'sequence_length': SEQ_LEN},
        'data_stages': [{'name': 'S0_rewrite600m_20B', 'start_training_step': 1,
                         'data': {'dataset': {'dataset_folder': [str(Path(tokenized_root) / setting)]},
                                  'num_loading_workers': 1, 'seed': SEED}}],
        'checkpoints': {'load_optimizer': kind != 'trunk1', 'load_lr_scheduler': kind != 'trunk1',
                        'checkpoint_interval': interval, 'save_initial_state': False, 'save_final_state': True,
                        'checkpoints_path': str(ckpt_dir(runs_root, setting, kind)), 'resume_checkpoint_path': resume},
        'optimizer': {'weight_decay': WEIGHT_DECAY, 'clip_grad': CLIP_GRAD, 'accumulate_grad_in_fp32': True,
                      'learning_rate_scheduler': {'learning_rate': PEAK_LR, 'lr_warmup_steps': WARMUP_STEPS,
                                                  'lr_warmup_style': 'linear', 'lr_decay_starting_step': decay_start,
                                                  'lr_decay_style': 'linear', 'lr_decay_steps': decay_steps,
                                                  'min_decay_lr': 0.0},
                      'optimizer_factory': dict(ADAMW),
                      'zero_stage': zero_stage},
        'logging': {'iteration_step_info_interval': 1},
        'parallelism': {'dp': dp, 'tp': TP, 'pp': PP, 'expert_parallel_size': 1, 'pp_engine': '1f1b',
                        'tp_mode': 'REDUCE_SCATTER', 'tp_linear_async_communication': True,
                        'recompute_layer': recompute_layer},
    }


def science_view(cfg: dict) -> dict:
    """The fields of a rendered config that are NOT operator inputs (paths, W&B project, deployment): two configs with
    equal science_view are the same experiment segment."""
    c = copy.deepcopy(cfg)
    c['general'].pop('project', None)
    c['tokenizer'].pop('tokenizer_name_or_path', None)
    c['data_stages'][0]['data']['dataset'].pop('dataset_folder', None)
    for k in ('checkpoints_path', 'resume_checkpoint_path'):
        c['checkpoints'].pop(k, None)
    for k in ('micro_batch_size', 'batch_accumulation_per_replica'):
        c['tokens'].pop(k, None)
    c['optimizer'].pop('zero_stage', None)
    for k in ('dp', 'recompute_layer'):
        c['parallelism'].pop(k, None)
    return c
