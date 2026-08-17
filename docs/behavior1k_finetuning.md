# Cosmos 3 Edge → BEHAVIOR-1K fine-tuning: code structure

How the BEHAVIOR-1K (R1Pro) fine-tuning pipeline for **Cosmos3-Edge (4B)** is
built, file by file, and how the pieces fit together — from raw
`behavior-1k/2026-challenge-demos` data to a served vla-eval model server.

The code lives in two repos:

- **cosmos-framework** submodule, branch `feat/behavior1k-edge-posttrain`
  (fork: `github.com/acensia/cosmos-framework`) — dataset, experiment recipe,
  TOML, launchers. Upstream only ships the domain *constants* for BEHAVIOR-1K
  (id 22, 23-D) plus a purpose-registered 512×768 canvas; everything that makes
  it trainable is on this branch.
- **behavior-cosmos** parent repo — data staging, slurm scripts, the serving
  wrapper, and configs.

```
                       TRAINING                                   SERVING
┌─────────────────────────────────────────────────┐   ┌───────────────────────────────┐
│ scripts/stage_behavior1k_task.py                │   │ vla-eval harness (other box)  │
│   HF → data/behavior1k/task-0000 (LeRobot v3)   │   │   ws://<node>:8000            │
│                                                 │   └──────────────┬────────────────┘
│ scripts/train_behavior1k_edge.sbatch            │                  │ WebSocket (obs → action)
│   → scripts/env_behavior1k_edge.sh (env vars)   │   ┌──────────────▼────────────────┐
│   → examples/launch_sft_..._edge.sh (torchrun)  │   │ servers/cosmos3_policy_server │
│     → cosmos_framework.scripts.train            │   │   (light, no torch)           │
│         --sft-toml action_policy_behavior1k_edge│   │   composes cams, builds prompt│
│                                                 │   └──────────────┬────────────────┘
│ experiment: action_policy_behavior1k_edge       │                  │ HTTP /predict (localhost)
│ dataset:    Behavior1KLeRobotDataset            │   ┌──────────────▼────────────────┐
│ base ckpt:  checkpoints/Cosmos3-Edge-DCP        │   │ cosmos action server (shim:   │
│                                                 │   │  cosmos3_action_server_main)  │
│ output: outputs/train/.../iter_000002000  ──────┼──▶│  loads the DCP checkpoint     │
└─────────────────────────────────────────────────┘   └───────────────────────────────┘
```

## 1. Data

### Source dataset

`behavior-1k/2026-challenge-demos` (HF): LeRobot **v3.0**, 100 tasks × 200
episodes, 30 fps, **3.27 TB** total. Per frame:

| field | shape / value |
|---|---|
| `action` | 23-D float — see layout below |
| `observation.state` | 61-D (unused by this recipe) |
| `observation.rgb.zed_link_camera_0` | head, 720×720 |
| `observation.rgb.left/right_realsense_link_camera_0` | wrists, 480×480 |

**23-D action layout** (verified against the data, not just upstream comments):
`[base(3), trunk(4), left_arm(7), left_gripper(1)@dim14, right_arm(7), right_gripper(1)@dim22]`
— absolute joint targets in raw hybrid units (base velocity-like, arms/trunk
radians, grippers exactly ±1). Trained **raw** (`action_normalization=None`,
same precedent as the DROID `joint_pos` recipe) → serving is a pure
passthrough, no stats file anywhere.

> Do NOT use the older `lerobot/behavior1k-taskNNNN` HF repos — different
> release (256-D state, other camera keys).

### Staging — `scripts/stage_behavior1k_task.py` (parent repo)

The full set doesn't fit the 2 TiB quota, so this PEP 723 uv script downloads a
**per-task RGB-only subset** (~1.9 GB/task). Key trick: in this dataset,
`meta/episodes/chunk-NNN/file-000.parquet` holds exactly task NNN's 200
episodes, so the script reads that one parquet, then fetches only the data
parquets and RGB mp4s it references — the result is a *valid standalone
LeRobot tree* at `data/behavior1k/task-NNNN`. Task 0 = `turning_on_radio` is
staged at `data/behavior1k/task-0000`.

Scale-up: `uv run scripts/stage_behavior1k_task.py --task N`, then point
`BEHAVIOR1K_ROOT` elsewhere (or add multiple dataset entries to the experiment's
`datasets` dict with ratios).

## 2. Dataset class (cosmos-framework branch)

`cosmos_framework/data/generator/action/datasets/behavior1k_lerobot_dataset.py`
— `Behavior1KLeRobotDataset(ActionBaseDataset)`, mirrors the LIBERO loader.

- **Embodiment/domain**: `behavior1k_lerobot`, domain id **22** (upstream
  constant). `viewpoint="concat_view"`.
- **Frame index**: reads only `index / episode_index / task_index / timestamp /
  action` columns from all data parquets into contiguous numpy arrays sorted by
  global frame index (copy-on-write friendly across DataLoader workers).
  Valid windows are within-episode only: `count − chunk_length` per episode,
  addressed through a cumulative-sum (`_valid_cum`) → flat index.
- **Split**: deterministic per-episode train/val (`val_ratio=0.01`, seeded, same
  on every rank).
- **Video decode**: per-frame real timestamps via lerobot's
  `decode_video_frames(..., backend="pyav")` — pyav is **forced** because
  lerobot's default (torchcodec) needs system libav\*, absent on cluster nodes.
  The loader is FPS-agnostic (trusts `meta/info.json` fps = 30).
- **Camera composite** (`_compose_multi_view`) — must match serving exactly:

  ```
  ┌──────────────┐
  │     head     │  720×720 native
  ├───────┬──────┤
  │ left  │ right│  wrists bilinear-downscaled to 360×360
  └───────┴──────┘   → 720(W) × 1080(H), h/w = 1.5
  ```

  At `resolution="480"` this snaps to the purpose-registered exact-aspect
  `"2,3"` **512×768 canvas** (`data/generator/utils.py`,
  `VIDEO_RES_SIZE_INFO["480"]`) with zero padding. **`resolution="480"` is
  mandatory** — `None` picks the 720 tier, which has no 2:3 canvas and would
  pad on 3:4.
- **Actions**: window of 16 raw 23-D rows, no transform.
- **Prompt pieces**: `ai_caption` = the raw task string from `tasks.parquet`
  (**`turning_on_radio`**, underscored — serving must match, see §5), plus
  `extras["additional_view_description"] = CONCAT_VIEW_LAYOUT_DESCRIPTION`:

  > "The top row is from the head-mounted camera. The bottom row contains the
  > left and right wrist-mounted camera views, concatenated horizontally."

  This sentence must stay **byte-identical** between this class and the serving
  config — it lands in the JSON prompt's `cinematography.framing`.
- **Robustness**: `__getitem__` retries up to 8 random resamples on decode
  failure.

Factory: `get_action_behavior1k_sft_dataset` in `action_sft_dataset.py` wraps
the dataset in `ActionTransformPipeline` (canvas resize, JSON prompt formatting
`format_prompt_as_json=True`, cfg dropout 0.1, packing metadata). Export added
to `datasets/__init__.py`.

## 3. Experiment config (cosmos-framework branch)

`configs/base/experiment/action/posttrain_config/action_policy_behavior1k_edge.py`,
registered in Hydra's ConfigStore (import added to `configs/base/config.py`).
Mirrors `action_policy_libero_nano` but on `EDGE_MODEL_CONFIG`.

Model config deltas (`_action_policy_behavior1k_edge_model_config`):

- `max_num_tokens_after_packing=74000` — the 512×768 canvas packs far fewer
  samples than LIBERO's 192×320; this caps packed sequence length for H100 mem.
- `activation_checkpointing.mode="selective"` (save `fmha` ops).
- `diffusion_expert_config.load_weights_from_pretrained=False` — fresh
  diffusion-expert init.
- `rectified_flow_training_config.loss_scale=10.0`, `image_loss_scale=None`.
- `tokenizer.encode_exact_durations=[17]` — chunk 16 + 1 obs frame (DROID pins
  `[33]` for chunk 32 the same way).
- `vlm_config.tokenizer = build_processor_lazy(tokenizer_type=${oc.env:EDGE_HF_PATH})`
  — offline-safe: sources the HF processor from the local `Cosmos3-Edge`
  snapshot instead of a hub fetch (`HF_HUB_OFFLINE=1` works).

Optimizer: FusedAdam, fp32 master weights, `eps=1e-8` (bf16 + 1e-6 diverged on
the LIBERO action loss). Trainable key set = generation + action heads
(`moe_gen, time_embedder, vae2llm, llm2vae, action2llm, llm2action,
action_modality_embed, k_norm_und_for_gen`), base lr `5e-5` with **5×
multipliers on the three action-head keys**.

Checkpoint: `keys_to_skip_loading = [net_ema., action2llm, llm2action,
action_modality_embed, action_pos_embed]` — the public Edge base ships trained
action heads, but for *other* domains' layouts; the 23-D behavior1k layout
shares no convention with them, so the heads start fresh (EMA warm-starts from
net). `strict_resume=False` for the base-init load.

Dataloader: `PackingDataLoader(max_samples_per_batch=32) →
RankPartitionedDataLoader(batch_size=1, workers=4)`; shuffling is
dataset-side (`iterable_shuffle=True` over per-episode blocks from
`get_shuffle_blocks`). Root comes from `${oc.env:BEHAVIOR1K_ROOT}`.

In-file scheduler/trainer values are smoke-sized (max_iter 100); the real run
sizes come from the TOML.

## 4. Run configs & launch chain

### TOMLs — `examples/toml/sft_config/`

| file | shard×replicate | grad_accum | global batch |
|---|---|---|---|
| `action_policy_behavior1k_edge.toml` | 4×1 (one 4-GPU node) | 4 | 32 × 4 × 4 = **512** |
| `action_policy_behavior1k_edge_2gpu.toml` | 2×1 | 8 | 32 × 2 × 8 = **512** (identical optimization) |

Both: bf16, `max_iter=2000`, `save_iter=500`, warmup 250, cycle 16000 (linear
decay), `load_path=${oc.env:BASE_CHECKPOINT_PATH}`,
`model.tokenizer.vae_path=${oc.env:WAN_VAE_PATH}`.

### Launch chain (top → bottom)

1. **`scripts/train_behavior1k_edge.sbatch`** (parent repo; `_2gpu` variant
   exists) — slurm shell: 4×H100, 24 h. Resolves `REPO_ROOT` (overridable env,
   since slurm runs a spooled copy), sources the env script, cd's into the
   submodule, runs the launcher.
2. **`scripts/env_behavior1k_edge.sh`** — the single place all paths are
   defined: `BEHAVIOR1K_ROOT`, `EDGE_HF_PATH` (local HF snapshot →
   processor), `BASE_CHECKPOINT_PATH` (DCP base), `WAN_VAE_PATH`,
   `OUTPUT_ROOT`/`IMAGINAIRE_OUTPUT_ROOT`, `HF_HUB_OFFLINE=1`, and puts the
   submodule venv's `torchrun` on PATH.
3. **`examples/launch_sft_action_policy_behavior1k_edge.sh`** (submodule) —
   validates the env vars, then delegates to the shared
   `_sft_launcher_common.sh` → `torchrun … cosmos_framework.scripts.train
   --sft-toml <toml>`. Extra Hydra overrides via `EXTRA_TAIL_OVERRIDES`
   (space-separated), e.g. the smoke test's
   `model.config.parallelism.data_parallel_shard_degree=1`.

**Hydra override paths**: tail overrides for parallelism are
`model.config.parallelism.*` — `model.parallelism.*` is TOML-section syntax
only and is rejected on the CLI.

### Base checkpoint

The trainer loads **DCP**, not diffusers: `checkpoints/Cosmos3-Edge-DCP`
(6.3 GB) was produced from the public `nvidia/Cosmos3-Edge` snapshot with
`cosmos_framework.scripts.convert_model_to_dcp` (CPU-safe; on the GPU-less
login node transformer_engine needs the venv symlink
`nvidia/cuda_cudart → cuda_runtime` to find cudart).

### Smoke & full run

- `scripts/smoke_behavior1k_edge.sh` — 1 GPU, 5 iters, DCP save check.
- Full run (done 2026-08-15): 2000 iters, loss 14.3 → 0.88, ~45.5 s/iter at
  global batch 512 on 4×H100 (≈25 h → hit the 24 h wall at iter 1928, resubmit
  **auto-resumed** from `latest_checkpoint.txt` at iter 1500). Checkpoints
  (30 GB each, incl. optimizer state) at
  `outputs/train/cosmos3_action_behavior1k/action_sft/action_policy_behavior1k_edge/checkpoints/iter_000000{500,1000,1500,2000}`.
  Note: the launcher's `tee` **truncates** the log on restart — archive before
  resubmitting.

## 5. Serving (parent repo)

Serving goes **straight from the DCP checkpoint** — `export_model` is broken
for OSS Edge action runs (`_build_edge_policy_metadata` assumes the internal
dataloader layout).

Two processes on one GPU node:

1. **`servers/cosmos3_policy_server.py`** — a *light* vla-eval
   `PredictModelServer` (PEP 723 uv script, no torch). Spawns the heavy server
   below via the submodule venv's python, then per request: composes the 3
   harness camera frames with the **same head-top / wrists-half-below layout**
   as `_compose_multi_view`, base64-PNGs the 512×768 canvas, builds the
   prompt, POSTs to `/predict`, and returns the (16, 23) chunk raw
   (`action_mode: joint_passthrough`). Relevant knobs (all in the yaml):
   `camera_layout: head_top_wrists_bottom`, `caption_style: underscored`
   (harness sends humanized "turning on radio"; training saw
   `turning_on_radio`), `view_description` (the byte-identical layout
   sentence), `extra_env` (feeds `BEHAVIOR1K_ROOT`/`EDGE_HF_PATH` to the
   experiment config's `${oc.env:…}` interpolations).
2. **`servers/cosmos3_action_server_main.py`** — a shim around
   `cosmos_framework.scripts.action_policy_server_libero` that (a) applies the
   framework's own `guardrails=False` override (guardrail weights are a gated
   HF repo and irrelevant to policy serving) and (b) monkeypatches
   `ActionPromptJsonFormatter` to inject `additional_view_description` from the
   `COSMOS3_VIEW_DESCRIPTION` env var — reproducing what the dataset's
   `extras` does at training time.

Config: `configs/cosmos3_policy/behavior1k_sft.yaml` (points at
`iter_000002000`, `experiment: action_policy_behavior1k_edge`, fps 30,
chunk 16, `image_size: 768`). Note: **no `port:` key** — the standalone
`run_server` rejects it; bind via `--address` in
`scripts/serve_behavior1k_sft.sbatch` (1 GPU, 3-day limit, ws://\<node\>:8000).

Contract checks that already bit us once:

| training | serving must match |
|---|---|
| caption `turning_on_radio` | `caption_style: underscored` |
| `CONCAT_VIEW_LAYOUT_DESCRIPTION` sentence | `view_description:` byte-identical |
| head-top composite, 512×768, fps 30 | `camera_layout`, `image_size: 768`, `fps: 30` |
| raw actions (`action_normalization=None`) | `joint_passthrough`, empty `action_stats_path` |

A golden-trajectory probe (real demo frames → live server vs. ground-truth
actions) validated the whole path: 0.99+ correlation per joint block.

## 6. State-conditioned variant (`action_policy_behavior1k_edge_state`)

The recipe above is vision-only — the demos' 61-D `observation.state` never
reaches the model (see the "61-D Blind Spot" findings memo). Cosmos 3 ingests
robot state through the *action channel*: with `use_state=True` the current
state, expressed in the action layout, is prepended as action row 0 and the
transform marks it as a condition frame (held fixed during denoising — the
DROID `joint_pos` recipe's mechanism). The `_state` experiment enables this:

- `Behavior1KLeRobotDataset(use_state=True)` reads `observation.state` and
  projects it to the 23-D action layout via `state_to_action_layout`. The
  61→23 mapping was identified empirically (corr 0.99+ per dim):
  base←state[0:3] (measured base velocity), trunk←state[53:57],
  left arm←state[3:10], left gripper←state[24], right arm←state[28:35],
  right gripper←state[50]; grippers are finger aperture in meters [0, 0.05]
  (0.05 = open = action +1), rescaled to ±1.
- 17 action rows against 17 video frames triggers the transform's "Case B" →
  row 0 conditioned, rows 1–16 predicted. `encode_exact_durations=[17]`
  already matches.
- Registered as a **separate experiment** (separate job name → separate
  checkpoint dir) so a new run never auto-resumes from the finished
  vision-only checkpoints.
- Launch: same launchers with `TOML_FILE=examples/toml/sft_config/action_policy_behavior1k_edge_state{,_2gpu}.toml`,
  or from the parent repo `sbatch scripts/train_behavior1k_edge_state.sbatch`
  (smoke: `scripts/smoke_behavior1k_edge_state.sh`).
- **Serving the state checkpoint needs wrapper changes** (not yet done): send
  proprio from the harness, map it onto the same 23-D action-layout row, fill
  action row 0, and strip it from the returned chunk — mirror
  `action_policy_server_robolab.py`'s `use_state` handling; the libero server
  script the wrapper spawns has no such path today. Caveat: the harness's
  proprio layout is NOT verified to equal the demos' 61-D `observation.state`
  ordering — validate the mapping against demo data before trusting it.

## 7. Gotchas index

- `resolution="480"` mandatory (the 2:3 canvas only exists in that tier).
- `backend="pyav"` for video decode (no system libav\* on nodes).
- CPU-affinity fix in `cosmos_framework/utils/distributed.py`: intersect
  `device.get_cpu_affinity()` with `os.sched_getaffinity(0)` — raw
  `sched_setaffinity` gets EINVAL under slurm cgroups.
- Hydra CLI parallelism overrides: `model.config.parallelism.*`.
- `--experiment-overrides` is a tyro `list[str]`: repeated flags keep only the
  last — pass multiple overrides in ONE flag.
- Login node: no GPU, no `nvidia-smi`; anything touching the model needs
  `srun -p h100 --gres=gpu:1`.
- Heavy files on `/mnt/cepheid` only (2 TiB quota — check with cephfs xattrs,
  `getfattr -n ceph.dir.rbytes`, not `du`).
- Launcher `tee` truncates logs on restart; auto-resume reads
  `latest_checkpoint.txt`.
