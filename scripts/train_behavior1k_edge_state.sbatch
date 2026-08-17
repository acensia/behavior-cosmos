#!/usr/bin/env bash
#SBATCH --job-name=b1k-edge-sft-state
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

# Cosmos3-Edge BEHAVIOR-1K action-policy SFT, STATE-CONDITIONED variant:
# the 61D observation.state is projected to the 23D action layout and prepended
# as a conditioned initial action row (use_state=True, DROID-style).
# Same sizing as the vision-only run: FSDP 4x1, global batch 512, max_iter 2000.
# Checkpoints land under .../action_policy_behavior1k_edge_state/ (no collision
# with the finished vision-only run).
# Submit from the behavior-cosmos repo root: sbatch scripts/train_behavior1k_edge_state.sbatch
# NOTE: ~45.5 s/iter observed for the vision-only run -> 2000 iters ≈ 25 h; if it
# hits the 24 h wall, resubmit — the trainer auto-resumes from latest_checkpoint.txt.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/mnt/cepheid/users/acensia/nvidia/behavior-cosmos}"
source "$REPO_ROOT/scripts/env_behavior1k_edge.sh"

cd "$REPO_ROOT/cosmos-framework"
TOML_FILE=examples/toml/sft_config/action_policy_behavior1k_edge_state.toml \
NPROC_PER_NODE=4 bash examples/launch_sft_action_policy_behavior1k_edge.sh
