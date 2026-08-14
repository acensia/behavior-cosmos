# Shared environment for the Cosmos3-Edge BEHAVIOR-1K post-train recipe.
# Source from smoke/sbatch scripts. Repo root inferred from this file's location.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export BEHAVIOR1K_ROOT="$REPO_ROOT/data/behavior1k/task-0000"
export EDGE_HF_PATH="$REPO_ROOT/checkpoints/Cosmos3-Edge"
export BASE_CHECKPOINT_PATH="$REPO_ROOT/checkpoints/Cosmos3-Edge-DCP"
export WAN_VAE_PATH="$REPO_ROOT/checkpoints/Wan2.2_VAE.pth"
export OUTPUT_ROOT="$REPO_ROOT/outputs/train"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export HF_HUB_OFFLINE=1
# torchrun must come from the cosmos-framework venv (not pip-installed; PYTHONPATH
# is set by the launcher).
export PATH="$REPO_ROOT/cosmos-framework/.venv/bin:$PATH"
