#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
    REPO_ROOT="${SCRIPT_REPO_ROOT}"
fi
cd "${REPO_ROOT}"

if [[ "${1:-}" == "create" ]]; then
    exec uv run --extra train --group parity wandb sweep \
        --entity personal-space-astha \
        --project OCR_pretraining \
        config/trocr/wanbd_sweep/sweep.yaml
fi

if [[ "${1:-}" != "agent" ]] || (( $# != 3 )); then
    echo "Usage:" >&2
    echo "  bash $0 create" >&2
    echo "  sbatch --array=1-<gpu-count> $0 agent <sweep-id> <runs-per-gpu>" >&2
    exit 2
fi

sweep_id="$2"
runs_per_gpu="$3"
[[ "${runs_per_gpu}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Error: runs-per-gpu must be a positive integer." >&2
    exit 2
}

echo "Slurm job: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-1}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

exec uv run --extra train --group parity wandb agent \
    --forward-signals \
    --count "${runs_per_gpu}" \
    --entity personal-space-astha \
    --project OCR_pretraining \
    "${sweep_id}"