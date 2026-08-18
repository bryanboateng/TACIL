#!/bin/bash
#SBATCH --job-name=sweep-agent
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/%x.%A_%a.out
#SBATCH --array=1-4

# One agent per array task. Each agent pulls runs from the same sweep, so the
# array tasks search the space in parallel.
#
# Submit with:
#   sbatch --export=ALL,SWEEP_PATH=<entity>/<project>/<sweep_id> sweep_agent.sbatch

set -euo pipefail

: "${SWEEP_PATH:?Set SWEEP_PATH=<entity>/<project>/<sweep_id>}"

# uv lives in the home directory, not in the system path.
source "$HOME/.local/bin/env"

# Caches default to the home directory, which is mirrored and nearly full.
# They hold thousands of small files, and BeeGFS synchronises file permissions
# with its metadata servers, so many files slow every operation down. The venv
# needs no override: uv puts it beside the project, which already sits here.
export UV_CACHE_DIR="/scratch/$USER/uv-cache"
export WANDB_CACHE_DIR="/scratch/$USER/wandb-cache"
export WANDB_DATA_DIR="/scratch/$USER/wandb-staging"

# --no-sync: the environment is built once on the frontend. Compute nodes must
# not resolve or install anything.
uv run --no-sync wandb agent "$SWEEP_PATH"
