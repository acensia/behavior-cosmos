#!/usr/bin/env bash
#SBATCH --job-name=b1k-sft-serve
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.out

# Long-running vla-eval model server for the BEHAVIOR-1K Edge SFT checkpoint.
# Binds ws://0.0.0.0:8000 — point the harness at ws://<node>:8000 (the node
# name is echoed below), tunneling if evaluating from outside the cluster.
# Submit from the repo root: sbatch scripts/serve_behavior1k_sft.sbatch
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/mnt/cepheid/users/acensia/nvidia/behavior-cosmos}"
cd "$REPO_ROOT"
echo ">>> serving on ws://$(hostname):8000"
exec uv run servers/cosmos3_policy_server.py \
  --config configs/cosmos3_policy/behavior1k_sft.yaml --address 0.0.0.0:8000
