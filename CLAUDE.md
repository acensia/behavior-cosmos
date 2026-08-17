# behavior-cosmos

Serve **NVIDIA Cosmos 3 policy** models (nvidia/cosmos-framework) behind the
**vla-evaluation-harness** model-server interface, so they can be evaluated on the harness's
benchmarks (LIBERO first; BEHAVIOR-1K is the eventual target implied by the project name).

> **History**: the project originally targeted NVlabs/cosmos-policy (the Predict2-2B-based
> "Cosmos Policy" line). The user redirected to the Cosmos 3 line (released 05/2026,
> `github.com/nvidia/cosmos-framework`). The old server (`servers/cosmos_policy_server.py`,
> `configs/cosmos_policy/`) is **parked** — functional up to model load but blocked on the
> gated `nvidia/Cosmos-Predict2-2B-Video2World` base repo (needs HF license + token).

## Layout

```
behavior-cosmos/
├── vla-evaluation-harness/     # submodule: github.com/allenai/vla-evaluation-harness
├── cosmos-framework/           # submodule: github.com/NVIDIA/cosmos-framework (Cosmos 3)
├── cosmos-policy/              # submodule: NVlabs/cosmos-policy (PARKED, predict2 line)
├── servers/cosmos3_policy_server.py  # ACTIVE: Cosmos 3 wrapper server (PEP 723 uv script)
├── servers/cosmos_policy_server.py   # parked predict2 server
├── configs/cosmos3_policy/libero.yaml  # ACTIVE config
├── checkpoints/                # gitignored: staged model weights (see below)
└── CLAUDE.md
```

## Cosmos 3 architecture (two processes)

`servers/cosmos3_policy_server.py` is a **light** vla-eval `PredictModelServer` (no torch).
On startup it spawns cosmos-framework's official HTTP policy server
(`cosmos_framework.scripts.action_policy_server_libero`) via `cosmos-framework/.venv/bin/python`
and translates per request. This mirrors cosmos-framework's own client/server eval split and
keeps the heavy env (torch 2.10 cu128, flash-attn-3) isolated in the submodule's venv.

Environment (once, login node OK):
```bash
cd cosmos-framework && uv sync --all-extras --group=cu128-train --group=policy-server
```
(cu128 chosen for the DGX-H100 driver; cu130-train also exists. `.python-version` pins 3.13.)

Checkpoints staged in `checkpoints/` (gitignored) — **user directive: Edge (4B) models only,
no Nano** (Nano LIBERO SFTs are ~90 GB/iter raw fp32 DCP — vetoed as too big):
- `Cosmos3-Edge-Policy-DROID/` — official nvidia policy (8.6 G, ungated, diffusers layout).
  Serve with `cosmos_framework.scripts.action_policy_server_robolab` + mandatory
  `--format-prompt-as-json True` (Edge export has no checkpoint.json → nothing inferred).
  Contract: 3 cams → 640×540 canvas, 8-D `[joint(7), gripper]` proprio in, 32×8 **absolute
  joint positions** out, raw units, gripper flipped both boundaries. Smoke:
  `srun -p h100 --gres=gpu:1 --cpus-per-task=8 --mem=96G bash scripts/smoke_droid_edge.sh`.
  No harness DROID benchmark exists (MolmoSpaces closest; adapter = future work, user said
  not now).
- `cosmos-edge-robocasa-bs512/` — `easyminnn/cosmos-edge-robocasa-bs512` (6.3 G consolidated
  export, iter_000004000; chunk 16, fps 20, domain_name "robocasa"). **BLOCKED on the
  training fork**: trained with `action_policy_robocasa_edge` experiment from an rlwrld fork;
  upstream cosmos-framework has NO "robocasa" domain (`domain_utils.get_domain_id` raises) —
  need the fork (or its domain id + action-space details) to serve faithfully. Harness has a
  ready `robocasa365` benchmark (12-D actions; mirror `robocasa365_groot.py`).
- `Cosmos3-Edge/` — vanilla 4B base (8.6 G), staged per user request (post-train substrate;
  note upstream registers a `behavior1k_lerobot` domain, id 22, 23-D R1Pro actions —
  BEHAVIOR-1K post-training is natively supported → the project's endgame path).
- `Wan2.2_VAE.pth` — from `Wan-AI/Wan2.2-TI2V-5B`; needed when serving DCP checkpoints whose
  config bakes a trainer-local VAE path (`--experiment-overrides
  model.config.tokenizer.vae_path=...`). Consolidated exports bundle their own VAE.

## Critical mapping details (LIBERO, Cosmos 3) — replicated from cosmos-framework's own client

Source of truth: `cosmos_framework/simulation/libero/closed_loop_eval.py` +
`docs/action_policy_libero_posttrain.md` (in the submodule).

1. **Images**: client sends `rotate_180`-ed frames (`img[::-1, ::-1]`); the harness's
   `preprocess_libero_image` double-flip IS a 180° rotation → **use harness images as-is**
   (unlike the predict2 line which needed an unflip). agentview + wrist each resized to
   256×256, concatenated horizontally (`concat_view`, 256×512), base64 PNG.
2. **Prompt**: task description + "This video contains concatenated views from multiple
   camera perspectives." + "The left half shows the third-person view; the right half shows
   the wrist-mounted camera." (byte-identical to training-time augmentors). The cosmos server
   appends duration/fps/resolution sentences itself (config-driven) — don't add them here.
3. **No proprioception** — the LIBERO recipe conditions on images only.
4. **Actions**: cosmos server returns denormalized (`quantile_rot` stats bundled in the
   submodule) 16×10 chunks in `frame_wise_relative` space `[dpos(3), rot6d(6), gripper(1)]`.
   rot6d = two **columns** + cross product + SVD projection to SO(3)
   (`pose_utils.convert_rotation`). Wrapper emits LIBERO-native 7-D
   `[dpos, axis-angle, gripper]`; spec `POSITION_DELTA / ROTATION_AA / GRIPPER_CLOSE_POS`.
5. **Gripper**: model emits [0,1]; env wants [-1,1] negative=open → `-(2g-1)` (`zero_one`
   mode; harness benchmark binarizes by sign afterwards). If grasps fail with a weak
   checkpoint, cosmos docs suggest binarizing `-sign(2g-1)` — flip `gripper_mode` if needed.
6. **Serving knobs** (defaults match the official eval): fps=20, num_steps=30 denoising,
   guidance=1.0, seed=0, chunk_size=16 open-loop. `LIBERO_ROOT` env must exist at spawn
   (config interpolates `${oc.env:LIBERO_ROOT}` for unused dataloaders; wrapper sets /tmp).

## Environment gotchas (learned the hard way)

- **Script names must not shadow packages**: `cosmos_policy.py` shadowed the `cosmos_policy`
  package (script dir lands on `sys.path`) — hence `*_server.py` names.
- **uv ignores a path dependency's `[tool.uv.sources]`/`[tool.uv.index]`** — a PEP 723 script
  must replicate wheel indexes itself (the parked predict2 server does this for the
  cosmos-dependencies cu128 index; flash-attn cu128 wheels are cp310-only → its script pins
  python 3.10). The cosmos3 wrapper avoids all of this by having no heavy deps.
- **Login node has NO GPU** — anything loading the model needs `srun -p h100 --gres=gpu:1`
  (a100 partition also exists). `nvidia-smi` doesn't exist on login-1.
- **Predict2 line HF gating**: `nvidia/Cosmos-Predict2-2B-Video2World` is gated (tokenizer +
  base weights); `$HF_HOME` = /mnt/cepheid/users/acensia/cache/huggingface, no token present.

## Commands

```bash
# Smoke test (GPU; harness venv + cosmos-framework venv + checkpoints must be staged):
srun -p h100 --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=01:00:00 \
  bash -lc 'vla-evaluation-harness/.venv/bin/vla-eval test -c configs/cosmos3_policy/libero.yaml -v --timeout 2700'
# Harness venv (once): cd vla-evaluation-harness && uv sync --python 3.11 --all-extras --dev
# Real eval: vla-eval run with a LIBERO benchmark config + this server config; 1 episode first.
```

## Status / next steps

- [x] Repo + submodules (harness @ e1ee9ad, cosmos-policy parked, cosmos-framework added)
- [x] cosmos-framework env synced: `.venv` (877 pkgs, torch 2.10.0+cu128, python 3.13,
      openpi-server); run scripts with `PYTHONPATH=<cosmos-framework dir>` (not pip-installed).
      First sync died on "Disk quota exceeded" (2 TiB cepheid quota) — watch checkpoint bulk.
- [x] Three Edge checkpoints + Wan VAE staged (see above); all heavy files on /mnt/cepheid
      (NEVER /mnt/home — user directive)
- [x] `servers/cosmos3_policy_server.py` (LIBERO wrapper, spawns cosmos HTTP server; CLI
      verified) + `configs/cosmos3_policy/libero.yaml` (checkpoint not staged — see note in file)
- [x] `scripts/smoke_droid_edge.sh` + `scripts/smoke_droid_client.py` (openpi WS client) —
      DROID Edge smoke, srun-ready
- [x] **BEHAVIOR-1K connection check wired** (user priority: raw zero-shot results first, SFT later):
      `configs/cosmos3_policy/behavior1k.yaml` serves vanilla `Cosmos3-Edge` via the wrapper's
      `action_mode: joint_passthrough` — 3 cams (head/left_wrist/right_wrist) → 256×768 concat
      canvas; (16, 23) raw joint chunks out; spec `joints/23/joint_positions_r1pro` matches the
      benchmark byte-for-byte (harness cross-checks at HELLO). No stats → un-denormalized output
      by design. Benchmark side lives on `acensia/feat/behavior1k` harness branch (fetched as
      remote `acensia` in the submodule; contract identical to main — branch only adds
      OmniGibson v3.9 / 2026-challenge compat). Wrapper handles missing cameras in
      `vla-eval test`'s 2-camera stub obs via warn+substitute fallback.
- [x] **behavior1k connection check PASSED** (2026-08-14, h100): `vla-eval test -c
      configs/cosmos3_policy/behavior1k.yaml` → 3 steps, success=True, 67 s. Three startup
      blockers fixed to get there:
      1. Guardrails: the cosmos server hard-enables them (no CLI flag) and their weights are
         a gated HF repo (`nvidia/Cosmos-Guardrail1`, no token) → wrapper now spawns via
         `servers/cosmos3_action_server_main.py`, a shim that applies the framework's own
         `guardrails=False` override. Guardrails are a generation content filter; unused in
         policy serving.
      2. Config: vanilla `Cosmos3-Edge` is a public diffusers export — its root `config.json`
         has no `model.config.vlm_config.tokenizer` (KeyError in `OmniInference._create`).
         behavior1k.yaml now passes `config_file:` = the registry's
         `inference/configs/model/Cosmos3-Edge.yaml`.
      3. VAE: that registry YAML bakes `bucket_name: bucket` + trainer-local `vae_path` →
         S3-credential crash. Wrapper's `wan_vae_path` now also emits
         `model.config.tokenizer.bucket_name=''` (local-path mode) alongside the vae_path
         override, as ONE `--experiment-overrides` flag (tyro list[str]: repeated flags keep
         only the last).
- [x] **BEHAVIOR-1K fine-tune pipeline built & smoke-tested** (2026-08-14). Single task
      first (user choice): `turning_on_radio` = task 0 of the 100-task
      `behavior-1k/2026-challenge-demos` (LeRobot v3, 30 fps, 23-D action, 61-D state,
      3 RGB cams: zed head 720², realsense wrists 480²). Facts learned:
      - Full challenge set is 3.27 TB (doesn't fit 2 TiB quota; ~990 GB already used).
        Per-task RGB-only subset ≈ 1.9 GB via `scripts/stage_behavior1k_task.py`
        (reads `meta/episodes/chunk-NNN/file-000.parquet` = exactly task NNN's 200 eps,
        downloads only referenced data parquets + RGB mp4s → valid local LeRobot tree).
        Staged at `data/behavior1k/task-0000`. Older `lerobot/behavior1k-taskNNNN` HF
        repos are a DIFFERENT release (256-D state, other cam keys) — don't use.
      - Upstream cosmos-framework has ONLY the domain constants (id 22, 23-D) plus a
        purpose-registered `"2,3"` 512×768 canvas in the **480 tier** of
        `VIDEO_RES_SIZE_INFO` (`data/generator/utils.py`) — the intended b1k layout is a
        DROID-style portrait composite: head 720² top, wrists 360² side-by-side below
        → 720×1080 (h/w=1.5, zero padding). `resolution="480"` must be pinned
        (`resolution=None` picks the 720 tier → wrong canvas).
      - Observed 23-D action layout (from data, upstream comment is looser):
        base(3) trunk(4) left_arm(7) left_gripper(1) right_arm(7) right_gripper(1);
        grippers exactly [-1,1] at dims 14 & 22; all dims within ±2.5; trained RAW
        (`action_normalization=None`, DROID joint_pos precedent) = passthrough serving.
      - Everything else authored on submodule branch **`feat/behavior1k-edge-posttrain`**:
        `Behavior1KLeRobotDataset` (mirrors LIBERO loader; pyav decode forced —
        lerobot's torchcodec default needs system libav*, absent on nodes),
        `get_action_behavior1k_sft_dataset`, `action_policy_behavior1k_edge` experiment
        (EDGE_MODEL_CONFIG + selective ckpting, `encode_exact_durations=[17]` for
        chunk 16, fresh action heads, offline processor via `EDGE_HF_PATH` local
        snapshot), TOML (FSDP 4×1, global batch 32×4×accum4=512, max_iter 2000) +
        launcher; `utils/distributed.py` affinity fix (EINVAL under slurm cgroup —
        intersect with `sched_getaffinity`).
      - Trainer needs DCP: `checkpoints/Cosmos3-Edge-DCP` (6.3 G) via
        `convert_model_to_dcp` (CPU-safe; on login-1 needed venv symlink
        `nvidia/cuda_cudart → cuda_runtime` for transformer_engine's cudart probe).
      - 1-GPU 5-iter smoke PASSED (`scripts/smoke_behavior1k_edge.sh`, loss 14.3→13.8,
        DCP save OK; ~0.65 s/iter at batch 4). Hydra tail-override path for parallelism
        is `model.config.parallelism.*` (NOT `model.parallelism.*`).
- [x] **Training run COMPLETE** (2026-08-15): 2000/2000 iters, loss 14.3 → ~0.88.
      Ran as job 169728 (TIMEOUT at 24 h @ iter 1928 — real speed ~45.5 s/iter, not the
      ≤12 h estimate) + resume job 170486 (auto-resumed from `latest_checkpoint.txt` at
      iter 1500; 6h53m). DCP checkpoints at iters 500/1000/1500/**2000** (30 G each incl.
      optimizer state) →
      `outputs/train/cosmos3_action_behavior1k/action_sft/action_policy_behavior1k_edge/checkpoints/`.
      Run-1 log archived as `logs/action_policy_behavior1k_edge_sft.run1-iter0-1928.log`
      (the launcher's tee truncates on restart). A 2-GPU variant recipe
      (`*_2gpu.toml`, grad_accum 8 = same global batch 512) is committed for other
      machines; branches pushed to `acensia/behavior-cosmos@feat/behavior1k-edge-finetune`
      + `acensia/cosmos-framework@feat/behavior1k-edge-posttrain` (.gitmodules points at
      the fork — the submodule commit doesn't exist upstream).
- [ ] **Serve the fine-tune**: serve straight from DCP (skip `export_model` — its
      `_build_edge_policy_metadata` expects the internal dataloader layout and crashes on
      the OSS `PackingDataLoader` shape). Update the wrapper for behavior1k SFT serving:
      camera composite must switch to the TRAINING layout (head top 720², wrists 360²
      below → 512×768 canvas) + byte-identical prompt sentence
      (`Behavior1KLeRobotDataset.CONCAT_VIEW_LAYOUT_DESCRIPTION`), NOT the zero-shot
      256×768 horizontal strip in `configs/cosmos3_policy/behavior1k.yaml`.
- [x] **State-conditioned recipe added** (2026-08-17): the vision-only fine-tune ignores the
      demos' 61-D `observation.state` (Cosmos 3 ingests state via the action channel —
      `use_state` prepends state as a conditioned action row 0, DROID `joint_pos` precedent;
      LIBERO recipe, which ours mirrored, doesn't). New experiment
      `action_policy_behavior1k_edge_state` (separate ckpt dir, no resume collision):
      `Behavior1KLeRobotDataset(use_state=True)` projects 61-D state → 23-D action layout
      (`state_to_action_layout`; mapping found empirically at corr 0.99+/dim: base←state[0:3],
      trunk←[53:57], Larm←[3:10], Lgrip←[24], Rarm←[28:35], Rgrip←[50]; gripper aperture
      [0,0.05] m → ±1) and prepends it → 17 action rows vs 17 frames = transforms "Case B"
      (row 0 conditioned). TOML_FILE now overridable in both launchers; new
      `*_state{,_2gpu}.toml` + `scripts/train_behavior1k_edge_state{,_2gpu}.sbatch` +
      `scripts/smoke_behavior1k_edge_state.sh`. Retrain required (old ckpt never saw a state
      row). NOTE serving the state ckpt needs wrapper work: map harness proprio → same 23-D
      action-layout row 0 + `send_proprio: true` (libero server has no use_state path —
      mirror `action_policy_server_robolab.py` lines 520-540, strip row 0 from output);
      harness proprio layout is UNVERIFIED vs the demos' 61-D state ordering — validate first.
- [ ] Scale up: more tasks via `stage_behavior1k_task.py --task N` (~2 GB each), swap
      BEHAVIOR1K_ROOT to a merged root or multiple dataset entries.
- [ ] **User runs the DROID smoke** (`scripts/smoke_droid_edge.sh`) if wanted
- [ ] robocasa SFT serving: get the rlwrld fork (or robocasa domain id + action space +
      normalization) from the user → then wire a robocasa365 wrapper mirroring the LIBERO one
- [ ] LIBERO: needs an Edge LIBERO SFT (none exists; post-train with cosmos-framework recipe,
      swapping nano → edge experiment) or accept a Nano DCP
