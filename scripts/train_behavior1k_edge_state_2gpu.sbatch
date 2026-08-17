#!/usr/bin/env bash
#SBATCH --job-name=b1k-edge-sft-state-2gpu
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out

# 2-GPU Cosmos3-Edge BEHAVIOR-1K action-policy SFT, STATE-CONDITIONED variant
# (use_state=True: 61D observation.state -> 23D action-layout row 0, conditioned).
# FSDP 2x1, grad_accum 8 -> same global batch 512 / schedule as the 4-GPU state
# recipe. Outputs land under action_policy_behavior1k_edge_state_2gpu.
# Submit: sbatch scripts/train_behavior1k_edge_state_2gpu.sbatch
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/mnt/cepheid/users/acensia/nvidia/behavior-cosmos}"
source "$REPO_ROOT/scripts/env_behavior1k_edge.sh"

cd "$REPO_ROOT/cosmos-framework"
TOML_FILE=examples/toml/sft_config/action_policy_behavior1k_edge_state_2gpu.toml \
NPROC_PER_NODE=2 bash examples/launch_sft_action_policy_behavior1k_edge_2gpu.sh
