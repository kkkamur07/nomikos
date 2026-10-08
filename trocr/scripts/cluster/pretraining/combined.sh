#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
fi
cd "${REPO_ROOT}"

command -v uv >/dev/null 2>&1 || {
    echo "Error: uv is not available on PATH." >&2
    exit 127
}

export PYTHONUNBUFFERED=1

echo "Repository: ${REPO_ROOT}"
echo "Slurm job: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

exec uv run --extra train --group parity python -m src.train.trocr \
    data=pretraining/combined \
    tokenizer=gpt_500 \
    encoder=base \
    model.freeze_encoder=false \
    decoder.reinitialize=token_layers \
    lora_adaptors.enabled=false \
    training.epochs=60 \
    output.root=var/runs/trocr/combined-pretraining \
    wandb.name=trocr-pretraining-combined \
    "$@"
