# Source this from the project root:  source env.sh
# Keeps every cache inside the project folder and activates the project's own venv.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export HF_HOME="$PROJECT_ROOT/.cache/hf"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.cache/uv-python"
export NLTK_DATA="$PROJECT_ROOT/.cache/nltk"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export JUPYTER_CONFIG_DIR="$PROJECT_ROOT/.cache/jupyter/config"
export JUPYTER_DATA_DIR="$PROJECT_ROOT/.cache/jupyter/data"
export JUPYTER_RUNTIME_DIR="$PROJECT_ROOT/.cache/jupyter/runtime"
export IPYTHONDIR="$PROJECT_ROOT/.cache/ipython"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1   # job logs show progress as it happens

# Settings that stay out of git, such as the Slurm account:
#   echo 'export SBATCH_ACCOUNT=<your account>' > env.local.sh
if [ -f "$PROJECT_ROOT/env.local.sh" ]; then
    source "$PROJECT_ROOT/env.local.sh"
fi

# Another project's venv may already be active in this shell; drop it.
unset VIRTUAL_ENV
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi
