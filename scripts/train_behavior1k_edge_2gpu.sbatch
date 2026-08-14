#!/usr/bin/env bash
#SBATCH --job-name=b1k-edge-sft-2gpu
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out

# 2-GPU Cosmos3-Edge BEHAVIOR-1K (turning_on_radio) action-policy SFT.
# FSDP 2x1, grad_accum 8 -> same global batch 512 / schedule as the 4-GPU
# recipe, ~2x wall-clock. Outputs land under a separate job name
# (action_policy_behavior1k_edge_2gpu) so it never clobbers the 4-GPU run.
# Submit: sbatch scripts/train_behavior1k_edge_2gpu.sbatch
# NOTE: slurm runs a spooled copy of this file; resolve the repo root via
# REPO_ROOT env (export before sbatch on other machines) or the default below.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/mnt/cepheid/users/acensia/nvidia/behavior-cosmos}"
source "$REPO_ROOT/scripts/env_behavior1k_edge.sh"

cd "$REPO_ROOT/cosmos-framework"
NPROC_PER_NODE=2 bash examples/launch_sft_action_policy_behavior1k_edge_2gpu.sh
