source "$HOME/kys_rebuttal_env.sh"
NANOTRON_SHA="PIN_SET_IN_NEXT_COMMIT"
KYS_EVAL_SHA="00bbca3483b35e932e20f75e6a49c18a57a225e4"
git clone --branch huicong-dev https://github.com/imHuicongZhang/nanotron.git "$CODE_ROOT"
git -C "$CODE_ROOT" checkout "$NANOTRON_SHA"
git -C "$CODE_ROOT" log -1 --format=%H        # must print $NANOTRON_SHA; record it
git clone https://github.com/imHuicongZhang/kys-eval.git "$EVAL_ROOT"
git -C "$EVAL_ROOT" checkout "$KYS_EVAL_SHA"
git -C "$EVAL_ROOT" log -1 --format=%H        # must print $KYS_EVAL_SHA; record it

# training environment (INSTALL.md §2; Python 3.11; order matters)
conda create -n kys python=3.11 -y && conda activate kys
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.3 --no-build-isolation
pip install "datatrove[io]==0.5.0" "numpy==2.0.2" "huggingface_hub<1.0" \
            "transformers==4.46.3" "tokenizers==0.20.3" numba==0.60.0 pybind11
pip install -e "$CODE_ROOT"
mkdir -p "$PWD/.tmp"
TMPDIR="$PWD/.tmp" TORCH_CUDA_ARCH_LIST=9.0 pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/fanshiqing/grouped_gemm@efe8c40eaf4c8ef57191e0ea9aa4117aa5b1a8f2"
pip install "wandb==0.30.0"                   # INSTALL.md does not install wandb; nanotron logs to it
python -c "import wandb; print('wandb', wandb.__version__)"
python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"   # 0.x
hf --help > /dev/null && echo "hf CLI available"

# evaluation environment (in a new shell, after source "$HOME/kys_rebuttal_env.sh")
cd "$EVAL_ROOT"
./install.sh                                  # venv + pinned packages + patched LightEval; ends with check_install
source .venv/bin/activate
export HF_HOME="$EVAL_ROOT/hf_cache"
python -m kys_eval.check_install              # must end "OK: environment matches the pinned v2 setup"
python -m kys_eval.selftest                   # must print "selftest passed"
python -m kys_eval.prefetch                   # evaluation datasets at pinned revisions
hf --help > /dev/null && echo "hf CLI available"
