# behavior-cosmos

Serve the NVIDIA **Cosmos Policy** model (NVlabs/cosmos-policy, "cosmos3 policy") behind the
**vla-evaluation-harness** model-server interface, so it can be evaluated on the harness's
benchmarks (LIBERO first; BEHAVIOR-1K is the eventual target implied by the project name).

## Layout

```
behavior-cosmos/
├── vla-evaluation-harness/   # submodule: github.com/allenai/vla-evaluation-harness
├── cosmos-policy/            # submodule: github.com/NVlabs/cosmos-policy
├── servers/cosmos_policy.py  # PEP 723 uv-script model server (PredictModelServer subclass)
├── configs/cosmos_policy/    # server config YAMLs (libero.yaml so far)
└── CLAUDE.md
```

Decision: the server lives **out-of-tree** (here, not inside the harness submodule), with the
uv-script header pointing at both submodules via relative `path` sources. Move it in-tree on a
harness fork only if upstreaming to allenai later. Another standalone harness checkout exists at
`~/vla-evaluation-harness` — ignore it; this repo's submodule is authoritative for this project.

## How the pieces fit

- The harness spawns the server via `uv run servers/cosmos_policy.py` (config `script:` path
  resolves against **CWD** — run `vla-eval` from this repo root) and talks WebSocket+msgpack.
  See `vla-evaluation-harness/CLAUDE.md` and `CONTRIBUTING.md` ("Adding a Model Server").
- `CosmosPolicyModelServer` subclasses `PredictModelServer` with `chunk_size=16`; cosmos returns
  16-step action chunks, the harness chunk buffer serves one action per step (open-loop replay,
  matching cosmos's own `num_open_loop_steps=16`).
- Model loading uses `cosmos_policy.experiments.robot.cosmos_utils`:
  `get_model`, `load_dataset_stats`, `init_t5_text_embeddings_cache`, `get_action`.
  `PolicyEvalConfig` can't be imported (it lives in `run_libero_eval.py`, which imports the
  LIBERO simulator), so `servers/cosmos_policy.py` has a local `_CosmosCfg` mirror — keep its
  defaults in sync with `run_libero_eval.PolicyEvalConfig` if the submodule is bumped.

## Critical mapping details (LIBERO) — verify before trusting results

1. **Image flip**: harness `preprocess_libero_image` flips both axes (`img[::-1, ::-1]`);
   cosmos `get_libero_image` flips vertically only (`np.flipud`). Server undoes the horizontal
   flip (`img[:, ::-1]`, `unflip_horizontal: true`). If eval scores are near-zero, check this first.
2. **Proprio**: harness sends 8-D `[eef_pos(3), axis_angle(3), gripper_qpos(2)]`; cosmos expects
   9-D `[gripper_qpos(2), eef_pos(3), eef_quat_xyzw(4)]`. Server converts axis-angle→quat.
   Risk: harness canonicalizes quat sign (w≥0, `vla_eval/rotation.py:quat_to_axisangle`) while
   robosuite's raw quat may have either sign; cosmos normalizes proprio with dataset stats, so a
   sign flip could shift inputs. The harness LIBERO benchmark has a `quat_no_antipodal` option —
   compare a few episodes against cosmos's native eval (`run_libero_eval.py`) if success rates lag.
3. **Actions**: cosmos outputs unnormalized native LIBERO actions
   `[dxyz(3), axis-angle delta(3), gripper]`, gripper −1=open/+1=close → passthrough, no inversion
   (unlike groot which needs `invert_gripper`). Declared spec: `POSITION_DELTA / ROTATION_AA /
   GRIPPER_CLOSE_POS`.
4. **T5 embeddings**: instruction embeddings come from the checkpoint's pickle cache. Harness task
   descriptions (`task["name"]`) must match cosmos's instruction strings; a cache miss triggers an
   on-the-fly T5 encode (big download, slow first step). Warmup uses the first cached key.
5. **Resolution/JPEG**: benchmark sends 256px; cosmos resizes to 224 + JPEG-compresses inside
   `get_action` (`use_jpeg_compression=True`, `trained_with_image_aug=True`) — don't pre-resize.

## Commands

```bash
# Smoke-test the server (loads model — needs GPU + HF access):
cd vla-evaluation-harness && uv sync --python 3.11 --all-extras --dev   # once
uv run vla-eval test -c ../configs/cosmos_policy/libero.yaml            # from harness dir; or use full paths
# Real eval (from repo root so script: resolves):
#   vla-eval run with a LIBERO benchmark config + this server config; 1 episode first.
```

## Status / next steps

- [x] Repo + submodules (harness @ e1ee9ad v0.3.0-36, cosmos-policy @ 18a2acc main)
- [x] Server scaffold + LIBERO config — **not yet run**; first run will surface uv dependency
      resolution issues (cosmos-policy `cu128` extra, flash-attn no-build-isolation, torch 2.7 cu128)
- [ ] First inference smoke test (`vla-eval test`), fix env/dep fallout
- [ ] 1-episode LIBERO run, then compare success rate vs cosmos native eval (seeds 195/196/197,
      LIBERO-10: paper numbers in cosmos-policy/LIBERO.md)
- [ ] RoboCasa config (needs `secondary_image` — third camera; see `get_action`'s robocasa branch)
- [ ] BEHAVIOR-1K: harness has `configs/benchmarks/behavior1k/` + `behavior1k_baseline.py` server
      example; no cosmos BEHAVIOR checkpoint exists — likely requires post-training (see
      nvidia-cosmos/cosmos-cookbook) before serving is meaningful.
