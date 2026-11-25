#!/bin/bash
# Reads the project's small files in parallel before Python starts.
# The first read of a file from Slate is slow on a fresh compute node, and importing
# transformers reads about 3,000 files one by one, which took over ten minutes.
#
#   bash slurm/warm_cache.sh           # Python packages
#   bash slurm/warm_cache.sh images    # data/images (run in the background with &)

: "${PROJECT_ROOT:?source env.sh first}"
start=$(date +%s)

read_parallel() {
    xargs -0 -r -P 48 -n 16 cat > /dev/null 2>&1 || true
}

if [ "${1:-packages}" = "images" ]; then
    find "$PROJECT_ROOT/data/images" -type f -name '*.jpg' -print0 2>/dev/null | read_parallel
    echo "warm_cache: images read in $(( $(date +%s) - start )) s"
else
    site="$PROJECT_ROOT/.venv/lib/python3.12/site-packages"
    packages=(transformers torch torchvision numpy tokenizers huggingface_hub safetensors PIL
              sympy networkx accelerate regex yaml tqdm pandas nltk pycocoevalcap jinja2 mpmath)
    paths=()
    for name in "${packages[@]}"; do
        [ -d "$site/$name" ] && paths+=("$site/$name")
    done
    find "${paths[@]}" "$PROJECT_ROOT/src" -type f \( -name '*.py' -o -name '*.pyc' \) -print0 \
        2>/dev/null | read_parallel
    echo "warm_cache: python packages read in $(( $(date +%s) - start )) s"
fi
