# Model Config Guide — Nanotron Llama (1.1B & 1.5B from scratch)

This explains the Llama architecture knobs in Nanotron, the **exact** parameter-count
formula so you can hit ~1.1B / ~1.5B yourself, the tokenizer/vocab facts, and the
checkpoint-interval arithmetic. It does **not** fill in your target numbers — it gives you
the formula and the levers.

All field names and behaviors below were read from this checkout:
- `src/nanotron/config/models_config.py` → `LlamaConfig` (the `model.model_config` block)
- `src/nanotron/config/config.py` → `ModelArgs`, `TokensArgs`, `CheckpointsArgs`, validation asserts
- `src/nanotron/helpers.py` → `_vocab_size_with_padding`
- `examples/config_tiny_llama.yaml` / `examples/config_nanoset.yaml` → reference YAML shape

---

## 1. The architectural hyperparameters

These live under `model.model_config` and map 1:1 to `LlamaConfig` fields
(`src/nanotron/config/models_config.py`). Symbols in brackets are used in the formula (§3).

| YAML field | Symbol | What it controls | Effect on param count |
|---|---|---|---|
| `hidden_size` | **H** | model/residual width | **Quadratic** — every attn & MLP matrix scales with H; embeddings scale linearly |
| `num_hidden_layers` | **L** | number of transformer blocks | **Linear** — total per-layer params × L |
| `num_attention_heads` | **A** | query heads; `head_dim = H / A` | Sets head_dim; attn Q/O are H×H regardless of A |
| `num_key_value_heads` | **KV** | GQA key/value heads (≤ A) | Shrinks K/V projections to `H × (KV · head_dim)` |
| `intermediate_size` | **I** | SwiGLU MLP width | **Linear & large** — MLP is `3·H·I`, usually the biggest per-layer term |
| `vocab_size` | **V** | token embedding rows | Embeddings = `2·V·H` (untied; see §4) |
| `max_position_embeddings` | — | max sequence length supported by RoPE | No param cost (RoPE is parameter-free) |
| `tie_word_embeddings` | — | share input & output embedding matrices | **`false` for Llama-2** → embeddings counted twice (§4) |
| `hidden_act` | — | `silu` (SwiGLU) | Fixes the `3·H·I` MLP structure |
| `rms_norm_eps`, `rope_theta`, `initializer_range`, `bos/eos_token_id` | — | numerics / RoPE base / init / special ids | No param cost |
| `attention_bias` | — | bias on attn projections (default `false`) | `false` → no bias params (assumed in §3) |

### Constraints you MUST satisfy (asserts in `src/nanotron/config/config.py`)
For tensor-parallel size `tp` (from `parallelism.tp`):
- `num_attention_heads % tp == 0`
- `num_attention_heads >= num_key_value_heads`
- `num_key_value_heads >= tp`
- `num_attention_heads % num_key_value_heads == 0`

Also keep `head_dim = hidden_size / num_attention_heads` an integer (commonly 64 or 128).

---

## 2. Reference YAML structure (Llama)

Skeleton with the param-relevant fields. `<SET>` = you must choose; values shown for
non-architectural fields are typical defaults, not prescriptions.

```yaml
model:
  ddp_bucket_cap_mb: 25
  dtype: bfloat16
  init_method:
    std: 0.02            # often tied to 1/sqrt(hidden_size); your choice
  make_vocab_size_divisible_by: 1     # see §4 (padding); 1 = no extra padding
  model_config:
    is_llama_config: true
    # ---- architecture: these set the parameter count (§3) ----
    hidden_size: <SET>            # H
    num_hidden_layers: <SET>      # L
    num_attention_heads: <SET>    # A   (H / A = head_dim)
    num_key_value_heads: <SET>    # KV  (<= A, divides A, >= tp)
    intermediate_size: <SET>      # I   (SwiGLU MLP width)
    # ---- tokenizer-bound ----
    vocab_size: 32000             # MUST equal the tokenizer vocab (§4) — do NOT pre-pad
    tie_word_embeddings: false    # Llama-2: separate input/output embeddings (§4)
    # ---- sequence / rope ----
    max_position_embeddings: <SET>   # >= tokens.sequence_length
    rope_theta: 10000.0
    rope_interleaved: false          # false to match HF Llama layout
    # ---- numerics / ids ----
    rms_norm_eps: 1.0e-5
    hidden_act: silu
    attention_bias: false
    bos_token_id: 1
    eos_token_id: 2
    pad_token_id: null
    pretraining_tp: 1
    use_cache: true
```

(For the surrounding sections — `tokens`, `checkpoints`, `optimizer`, `parallelism`,
`data_stages` — start from `examples/config_nanoset.yaml` and overwrite the model block.)

---

## 3. The explicit parameter-count formula

Assumptions (true for Nanotron Llama with the defaults above): SwiGLU MLP (3 matrices),
no biases (`attention_bias: false`), RMSNorm (weight only, no bias), RoPE (no params),
untied embeddings (`tie_word_embeddings: false`).

Let `head_dim = H / A`.

```
P =  EMBED  +  L · ( ATTN + MLP + NORMS_layer )  +  NORM_final

EMBED         = 2 · V · H                      # input embedding + output (lm_head), untied
ATTN          = 2·H²  +  2 · H · (KV · head_dim)
                └─ q_proj (H×H) + o_proj (H×H) = 2H²
                   k_proj (H × KV·head_dim) + v_proj (H × KV·head_dim) = 2·H·KV·head_dim
              = 2·H²  +  2·H²·(KV / A)          # since KV·head_dim = H·KV/A
MLP (SwiGLU)  = 3 · H · I                       # gate_proj + up_proj + down_proj
NORMS_layer   = 2 · H                           # input_layernorm + post_attention_layernorm (RMSNorm)
NORM_final    = H                               # final RMSNorm before lm_head
```

So, fully expanded:

```
P = 2·V·H  +  L · [ 2H²·(1 + KV/A)  +  3·H·I  +  2H ]  +  H
```

**How to use it:**
- Plug `V = 32000` and your candidate `H, L, A, KV, I` to get `P`. Adjust to land on
  ~1.1B (Model A) and ~1.5B (Model B).
- **Embeddings are fixed-ish and large at this scale:** `2·V·H = 64000·H`. For H≈2048 that's
  ~131M params *before any layers* — non-negligible for a 1.1B model, so don't ignore the
  embedding term (this is exactly why untied vs tied matters — §4).
- **Biggest lever:** `I` and `L` (both linear), then `H` (quadratic in the layer term,
  linear in embeddings). MHA vs GQA: with `KV = A` the attn term is `4H²`; with GQA
  (`KV < A`) it shrinks toward `2H²`.

> Verify your hand-computed `P` against the model's own report: when you launch training,
> Nanotron logs `Total number of parameters: …` (seen in the smoke run as
> "Total number of parameters: 13.7M"). Use a 1-step dry run to confirm your config hits
> the intended count **before** committing to a 30B-token run.

A note on what counts as "1.1B"/"1.5B": people sometimes quote **non-embedding** params or
**total**. This formula gives **total** (incl. both embeddings). Decide which convention you
mean for your "~1.1B / ~1.5B" targets and be consistent.

---

## 4. Tokenizer / vocab precision (verified)

- **Actual vocab size = 32000.** Verified two ways: `len(model.vocab)` in
  `…/llama2-unsloth-tokenizer/tokenizer.json` is 32000, and `tokenizers.Tokenizer.from_file(...).get_vocab_size()`
  returns 32000 (with `tokenizers>=0.20.3`; the env's 0.19.1 cannot load it — see the data guide §3.2).
  Special tokens are within range: `<unk>`=0, `<s>`(bos)=1, `</s>`(eos)=2. Source repo: `unsloth/llama-2-7b`.
- **Set `vocab_size: 32000` exactly.** With a Nanoset, `src/nanotron/config/config.py`
  asserts `model.model_config.vocab_size == dataset.vocab_size`, where `dataset.vocab_size =
  len(AutoTokenizer.from_pretrained(tokenizer_name).get_vocab())` = 32000. **Pre-padding the
  config's `vocab_size` (e.g. to 32064) will fail this assert.** Do not do it.
- **Padding to a multiple of 64 is handled internally, not in `vocab_size`.** At model build
  Nanotron pads the *embedding matrix* via `_vocab_size_with_padding`
  (`src/nanotron/helpers.py`):
  ```
  multiple   = make_vocab_size_divisible_by * tp
  padded_V   = ceil(orig_V / multiple) * multiple
  ```
  This pads rows for TP/efficiency **without** changing the config `vocab_size` field.
  **32000 is already 256·125** (divisible by 32, 64, 128, 256), so for any reasonable
  `make_vocab_size_divisible_by` (1, 64, 128) and small `tp`, `padded_V == 32000` — **no
  padding tokens are added**, and the embedding term stays exactly `2·32000·H`. (If you
  later pick a `tp` such that `make_vocab_size_divisible_by * tp` does not divide 32000,
  padding rows *would* be added and the real param count would rise accordingly — recheck.)
- **`tie_word_embeddings = false` for Llama-2.** Input embedding and output `lm_head` are
  **separate** matrices, so the embedding contribution is `2 · V · H` (not `V · H`). This is
  the `LlamaConfig` default (`tie_word_embeddings: bool = False`) — but the
  `examples/config_tiny_llama.yaml` sets it to `true`, so **set it explicitly to `false`**
  in your configs.

---

## 5. Checkpointing — fields and the interval arithmetic

### 5.1 Fields that control checkpointing (`CheckpointsArgs`, `src/nanotron/config/config.py`)
```yaml
checkpoints:
  checkpoints_path: /scratch/.../checkpoints/<run>   # where ckpts are written
  checkpoint_interval: <SET>        # save every N TRAINING STEPS  (this is the only cadence knob)
  save_initial_state: false         # if true, also saves at step 0 (not a token milestone)
  save_final_state: true            # save at the last step
  resume_checkpoint_path: null
```
**Key fact:** `checkpoint_interval` is a single integer in **steps** — saving happens every
`checkpoint_interval` steps. There is no built-in list-of-milestones schedule. So your token
milestones must be converted to a step cadence.

### 5.2 Tokens per step (verify this number yourself)
From `src/nanotron/config/config.py`:
```
global_batch_size            = micro_batch_size · batch_accumulation_per_replica · dp     # (in SEQUENCES)
global_batch_size_in_tokens  = global_batch_size · sequence_length                        # (in TOKENS)
```
So:
```
tokens_per_step = micro_batch_size × batch_accumulation_per_replica × dp × sequence_length
```
**Note what is NOT in the formula:** tensor-parallel (`tp`) and pipeline-parallel (`pp`) do
**not** change tokens/step — only **data-parallel `dp`** multiplies the token throughput per
step. (`tp`/`pp` change how one replica is split across GPUs, not how many tokens a step consumes.)

These fields live in the `tokens:` block (`TokensArgs`): `sequence_length`,
`micro_batch_size`, `batch_accumulation_per_replica`, `train_steps`; and `dp` in `parallelism:`.

### 5.3 Worked arithmetic for your milestones
You want checkpoints at **1B**, then **3B, 6B, 9B, …, 30B** (10 marks) → **11 checkpoints**,
and you target **~2M tokens/step**.

Step to reach `X` tokens: `step(X) = X / tokens_per_step`. For a milestone to land *exactly*
on a checkpoint, `X` must be an integer multiple of `tokens_per_step`.

**Example with a clean 2,000,000 tokens/step** (e.g. `seq_len=2048`, and
`mbs·accum·dp = 1000000/1024 …` — pick factors so the product × 2048 = 2,000,000; verify it
divides cleanly, since 2,000,000 / 2048 is **not** an integer → 2,000,000 is *not* reachable
with seq_len 2048. This is exactly the kind of mismatch to catch up front):

- `2,000,000 / 2048 = 976.5625` → **not integer**. So with `seq_len=2048` you cannot get
  exactly 2,000,000 tokens/step. Either pick `seq_len` that divides your target, or accept a
  power-of-two token count.

**Example with 2,097,152 = 2²¹ tokens/step** (`seq_len=2048`, `mbs·accum·dp = 1024`):
- `tokens_per_step = 1024 × 2048 = 2,097,152`.
- `step(1B)  = 1e9 / 2,097,152   = 476.84`   → **not integer** (1B is not a clean multiple).
- `step(3B)  = 3e9 / 2,097,152   = 1430.5`   → **not integer**.
  → power-of-two steps don't align with decimal (1B/3B) token milestones.

**To make milestones land exactly, choose `tokens_per_step` that divides 1,000,000,000.**
The milestones are `1B` and multiples of `3B`; `gcd(1, 3, 6, …, 30) = 1` (in units of B), so
their greatest common divisor is **1B**. Pick `tokens_per_step` dividing 1e9, e.g.
**2,000,000** is *not* a divisor friendly to seq_len 2048, but **2,000,000 = 2·10⁶ does
divide 1e9** (1e9 / 2e6 = 500). To realize 2,000,000 tokens/step cleanly, pick a `seq_len`
that divides 2,000,000 — e.g. `seq_len=2000` (unusual) or `seq_len=1000`/`4000`. More
practically, choose a round `tokens_per_step` like **1,000,000** or **2,000,000** with a
matching `seq_len`, or keep `seq_len=2048` and accept that milestones are approximate.

If you fix **`tokens_per_step = 2,000,000`** (and a `seq_len` that divides it):
```
step(1B)  = 1,000,000,000 / 2,000,000 =   500
step(3B)  = 3,000,000,000 / 2,000,000 = 1,500
step(6B)  =                            3,000
…
step(30B) = 30,000,000,000 / 2,000,000 = 15,000   = train_steps
```

### 5.4 The single-interval limitation (important)
`checkpoint_interval` is **one** number, so it can only produce an arithmetic series
`k · interval`. Your desired set `{1B, 3B, 6B, …, 30B}` is **not** a single arithmetic series
(the gap 1B→3B is 2B, then 3B thereafter). With `tokens_per_step = 2,000,000`:

- **Option 1 — interval = step(1B) = 500.** Saves every 1B: `1,2,3,…,30B` = **30 checkpoints**.
  This is a *superset* of what you want (includes all the 3B marks) but writes 20 extra
  checkpoints (storage cost — each is full model+optimizer state).
- **Option 2 — interval = step(3B) = 1,500.** Saves `3,6,…,30B` = **10 checkpoints**, but
  **misses 1B** (500 is not a multiple of 1500). To still get the 1B checkpoint, either:
  (a) run with `interval=1500` and separately stop/checkpoint at step 500 via a short first
  run then `resume_checkpoint_path`; or (b) set `save_initial_state` won't help (that's step 0).
- **Option 3 — interval = step(1B) but prune.** Use Option 1 and delete the unwanted
  checkpoints afterward.

There is no config field for an arbitrary milestone list, so pick Option 1 (simplest, more
storage) or Option 2 + a manual 1B capture. Compute `train_steps = step(30B)` for whichever
`tokens_per_step` you settle on, and set `checkpoint_interval` to the step value above.

### 5.5 Storage note
Each checkpoint stores model weights **+ optimizer state** (Adam → ~2× model bytes for
moments) **+ LR scheduler**. For a 1.5B model in bf16 weights with fp32 Adam moments, expect
**~15–20 GB per checkpoint**. Option 1 (30 ckpts) ⇒ ~0.5 TB; budget scratch accordingly, or
prefer Option 2.

---

## 6. Checklist before a real run
1. Pick `H, L, A, KV, I`; compute `P` with the §3 formula; **dry-run 1 step** and confirm
   Nanotron's logged "Total number of parameters" matches your target.
2. `vocab_size: 32000`, `tie_word_embeddings: false`, `make_vocab_size_divisible_by` such
   that no surprise padding (32000 is already divisible by 256).
3. Satisfy the head asserts (§1) for your chosen `tp`.
4. Choose `tokens_per_step` (= `mbs·accum·dp·seq_len`) so 1B/3B land on integer steps (§5.3);
   set `train_steps = step(30B)` and `checkpoint_interval` per §5.4.
5. `tokenizer.tokenizer_name_or_path` must equal the string baked into the dataset's
   `.ds.metadata` (see data guide §3.3).
```
</content>
