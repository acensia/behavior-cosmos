#!/usr/bin/env bash
# 5-iter, 1-GPU smoke of the action_policy_behavior1k_edge recipe (validates
# dataset decode -> packing -> model load -> train step -> DCP save end-to-end).
# Run under srun:
#   srun -p h100 --gres=gpu:1 --cpus-per-task=16 --mem=200G --time=00:50:00 \
#     bash scripts/smoke_behavior1k_edge.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_behavior1k_edge.sh"

cd "$REPO_ROOT/cosmos-framework"
EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 trainer.logging_iter=1 trainer.grad_accum_iter=1 model.config.parallelism.data_parallel_shard_degree=1 dataloader_train.max_samples_per_batch=4 checkpoint.save_iter=5" \
NPROC_PER_NODE=1 LOG_FILENAME=behavior1k_edge_smoke.log \
  bash examples/launch_sft_action_policy_behavior1k_edge.sh
