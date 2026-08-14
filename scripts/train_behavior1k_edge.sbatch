#!/usr/bin/env bash
#SBATCH --job-name=b1k-edge-sft
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

# Cosmos3-Edge BEHAVIOR-1K (turning_on_radio) action-policy SFT.
# FSDP 4x1 on one node, global batch 32 x 4 x grad_accum 4 = 512,
# max_iter 2000, DCP checkpoint every 500 iters.
# Submit from the behavior-cosmos repo root: sbatch scripts/train_behavior1k_edge.sbatch
# NOTE: slurm runs a spooled copy of this file; resolve the repo root via
# REPO_ROOT env (export before sbatch on other machines) or the default below.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/mnt/cepheid/users/acensia/nvidia/behavior-cosmos}"
source "$REPO_ROOT/scripts/env_behavior1k_edge.sh"

cd "$REPO_ROOT/cosmos-framework"
NPROC_PER_NODE=4 bash examples/launch_sft_action_policy_behavior1k_edge.sh
