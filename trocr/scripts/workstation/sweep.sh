#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWEEP_CONFIG="config/trocr/wanbd_sweep/sweep.yaml"
WANDB_ENTITY="personal-space-astha"
WANDB_PROJECT="OCR"

cd "${REPO_ROOT}"

case "${1:-}" in
    create)
        exec uv run --extra train wandb sweep \
            --entity "${WANDB_ENTITY}" \
            --project "${WANDB_PROJECT}" \
            "${SWEEP_CONFIG}"
        ;;
    agent)
        if (( $# < 2 || $# > 3 )); then
            echo "Usage: $0 agent <sweep-id> [run-count]" >&2
            exit 2
        fi
        sweep_id="$2"
        run_count="${3:-20}"
        exec uv run --extra train --group parity wandb agent \
            --forward-signals \
            --count "${run_count}" \
            --entity "${WANDB_ENTITY}" \
            --project "${WANDB_PROJECT}" \
            "${sweep_id}"
        ;;
    *)
        echo "Usage: $0 create | agent <sweep-id> [run-count]" >&2
        exit 2
        ;;
esac
