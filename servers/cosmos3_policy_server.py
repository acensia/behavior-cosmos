# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vla-eval",
#     "numpy>=1.24",
#     "pillow>=9.4",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../vla-evaluation-harness", editable = true }
# ///
"""Cosmos 3 policy model server for vla-evaluation-harness (LIBERO).

Two-process design, mirroring cosmos-framework's own eval split (their docs
keep the sim client and the model server in separate environments on purpose):

  1. This script is a *light* vla-eval ``PredictModelServer`` — no torch, no
     cosmos deps.  On startup it spawns NVIDIA cosmos-framework's official
     HTTP policy server (``cosmos_framework.scripts.action_policy_server_libero``)
     as a subprocess inside the ``cosmos-framework/.venv`` environment
     (build it once: ``cd cosmos-framework && uv sync --all-extras
     --group=cu128-train --group=policy-server``).
  2. Each ``predict()`` translates harness observations to the cosmos server's
     ``POST /predict`` contract and converts the returned action chunk back to
     the harness's LIBERO action space.

The translation replicates cosmos-framework's reference LIBERO client
(``cosmos_framework/simulation/libero/closed_loop_eval.py``) byte-for-byte:

  - **Images**: the client sends ``rotate_180``-ed sim frames; the harness's
    ``preprocess_libero_image`` already flips both axes (``img[::-1, ::-1]``),
    which IS a 180° rotation — so harness images are used as-is.  Each camera
    (agentview, wrist) is resized to ``image_size`` square and concatenated
    horizontally (``concat_view``, 256x512), then base64-PNG encoded.
  - **Prompt**: task description + the training-time concat_view sentence +
    the left/right layout sentence (``_augment_task_prompt_with_viewpoint``).
  - **No proprioception** — the LIBERO recipe conditions on images only.
  - **Actions**: the cosmos server returns a denormalized ``(chunk, 10)``
    chunk in ``frame_wise_relative`` space: ``[dpos(3), rot6d(6), gripper(1)]``.
    rot6d decodes column-wise with an SVD projection to SO(3) (matching
    ``pose_utils.convert_rotation``), giving per-step ``[dpos, axis-angle,
    gripper]`` — LIBERO's native 7-D command.  Gripper: model emits [0, 1];
    env wants [-1, 1] with negative=open -> ``-(2g - 1)`` (``zero_one`` mode;
    the harness benchmark binarizes by sign afterwards).
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import logging
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.rotation import matrix_to_quat, quat_to_axisangle
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    DimSpec,
)
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

# Prompt augmentation — byte-identical to cosmos-framework's
# viewpoint_utils.DEFAULT_VIEWPOINT_TEMPLATES["concat_view"] and
# closed_loop_eval._concat_view_layout_description / _CAMERA_PROMPT_NAMES.
_CONCAT_VIEW_SENTENCE = "This video contains concatenated views from multiple camera perspectives."
_CAMERA_PROMPT_NAMES = {
    "agentview": "third-person view",
    "wrist": "wrist-mounted camera",
    "head": "head-mounted camera",
    "left_wrist": "left wrist-mounted camera",
    "right_wrist": "right wrist-mounted camera",
}


def _append_prompt_sentence(prompt: str, sentence: str) -> str:
    if sentence in prompt:
        return prompt
    prompt = prompt.rstrip()
    if not prompt:
        return sentence.rstrip()
    separator = " " if prompt.endswith(".") else ". "
    return prompt + separator + sentence.rstrip()


def _layout_sentence(cameras: list[str]) -> str:
    names = [_CAMERA_PROMPT_NAMES.get(c, c) for c in cameras]
    if len(names) == 2:
        return f"The left half shows the {names[0]}; the right half shows the {names[1]}."
    layout = ", ".join(names)
    return f"The views are concatenated horizontally from left to right as: {layout}."


def _rot6d_to_matrix(v6: np.ndarray) -> np.ndarray:
    """Cosmos convention: first 3 = column 0, next 3 = column 1, col2 = cross,
    then SVD projection onto SO(3) (pose_utils.convert_rotation, rot6d branch)."""
    col0 = v6[:3]
    col1 = v6[3:6]
    col2 = np.cross(col0, col1)
    mat = np.stack((col0, col1, col2), axis=-1).astype(np.float64)
    u, _, vt = np.linalg.svd(mat)
    normalized = u @ vt
    if np.linalg.det(normalized) < 0:
        u[:, -1] *= -1.0
        normalized = u @ vt
    return normalized


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Cosmos3PolicyModelServer(PredictModelServer):
    """Cosmos 3 (cosmos-framework) LIBERO policy behind the harness interface."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/cosmos3-libero10-paper-2k/checkpoints/iter_000001500",
        wan_vae_path: str = "checkpoints/Wan2.2_VAE.pth",
        cosmos_framework_dir: str = "cosmos-framework",
        experiment: str = "action_policy_libero_nano",
        config_file: str = "",
        action_stats_path: str = (
            "cosmos_framework/data/generator/action/normalizer_stats/"
            "libero_native_frame_wise_relative_rot6d.json"
        ),
        action_normalization: str = "quantile_rot",
        raw_action_dim: int = 10,
        action_mode: str = "libero_rot6d",
        fps: int = 20,
        num_steps: int = 30,
        guidance: float = 1.0,
        seed: int = 0,
        image_size: int = 256,
        domain_name: str = "libero",
        gripper_mode: str = "zero_one",
        format_prompt_as_json: bool | None = None,
        send_proprio: bool = False,
        cameras: list[str] | None = None,
        server_url: str = "",
        cosmos_port: int = 0,
        startup_timeout: float = 1800.0,
        request_timeout: float = 300.0,
        warmup: bool = True,
        *,
        chunk_size: int = 16,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        if action_mode not in ("libero_rot6d", "joint_passthrough"):
            raise ValueError(f"Unknown action_mode={action_mode!r}. Use libero_rot6d/joint_passthrough.")
        self.image_size = image_size
        self.domain_name = domain_name
        self.gripper_mode = gripper_mode
        self.cameras = list(cameras) if cameras else ["agentview", "wrist"]
        self.raw_action_dim = raw_action_dim
        self.action_mode = action_mode
        self.send_proprio = send_proprio
        self.request_timeout = request_timeout
        self._proc: subprocess.Popen | None = None

        if server_url:
            self.server_url = server_url.rstrip("/")
            logger.info("Using externally managed cosmos server at %s", self.server_url)
        else:
            cosmos_dir = self._resolve(cosmos_framework_dir)
            venv_python = cosmos_dir / ".venv" / "bin" / "python"
            if not venv_python.exists():
                raise RuntimeError(
                    f"cosmos-framework venv not found at {venv_python}.\n"
                    f"Build it once with:\n"
                    f"  cd {cosmos_dir} && uv sync --all-extras "
                    f"--group=cu128-train --group=policy-server"
                )
            port = cosmos_port or _free_port()
            self.server_url = f"http://127.0.0.1:{port}"

            # Shim = action_policy_server_libero with guardrails disabled
            # (gated nvidia/Cosmos-Guardrail1 repo; see shim docstring).
            shim = Path(__file__).resolve().parent / "cosmos3_action_server_main.py"
            cmd = [
                str(venv_python),
                str(shim),
                f"--checkpoint-path={self._resolve(checkpoint_path)}",
                f"--raw-action-dim={raw_action_dim}",
                f"--action-chunk-size={chunk_size}",
                f"--fps={fps}",
                f"--num-steps={num_steps}",
                f"--guidance={guidance}",
                f"--seed={seed}",
                "--host=127.0.0.1",
                f"--port={port}",
            ]
            if action_stats_path:
                # Without stats the cosmos server returns actions un-denormalized
                # (raw model output) — what you want for zero-shot/raw checks.
                cmd.append(f"--action-normalization={action_normalization}")
                cmd.append(f"--action-stats-path={self._resolve_in(action_stats_path, cosmos_dir)}")
            if config_file:
                cmd.append(f"--config-file={self._resolve(config_file)}")
            elif experiment:
                # No experiment/config: a consolidated HF checkpoint dir supplies
                # its own config.json (CheckpointOverrides fallback).
                cmd.append(f"--experiment={experiment}")
            if wan_vae_path:
                # Registry model YAMLs set bucket_name -> Wan2pt2VAEInterface
                # prepends s3://<bucket>/ and needs object-store credentials.
                # Empty bucket_name = documented local-path mode. tyro parses
                # list[str] as one flag with N values (a repeated flag would
                # keep only the last), so pass both overrides together.
                cmd.append("--experiment-overrides")
                cmd.append(f"model.config.tokenizer.vae_path={self._resolve(wan_vae_path)}")
                cmd.append("model.config.tokenizer.bucket_name=''")
            if format_prompt_as_json is not None:
                cmd.append(f"--format-prompt-as-json={format_prompt_as_json}")

            env = dict(os.environ)
            # cosmos-framework is not pip-installed into its venv; its own docs
            # run scripts with PYTHONPATH=. from the repo root.
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{cosmos_dir}{os.pathsep}{existing_pp}" if existing_pp else str(cosmos_dir)
            # The experiment config interpolates ${oc.env:LIBERO_ROOT} for its
            # (unused at inference) dataloaders — any existing dir satisfies it.
            env.setdefault("LIBERO_ROOT", "/tmp")

            logger.info("Spawning cosmos action server: %s", " ".join(cmd))
            self._proc = subprocess.Popen(cmd, cwd=str(cosmos_dir), env=env)
            atexit.register(self._shutdown_subprocess)

        self._wait_for_server(startup_timeout)
        if warmup:
            self._warmup()

    # ------------------------------------------------------------------
    # Subprocess / HTTP plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(path: str) -> Path:
        """Resolve a config path against CWD (the repo root when run per docs)."""
        return Path(path).expanduser().resolve()

    @staticmethod
    def _resolve_in(path: str, base: Path) -> Path:
        p = Path(path).expanduser()
        return p.resolve() if p.is_absolute() else (base / p).resolve()

    def _shutdown_subprocess(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        logger.info("Terminating cosmos action server (pid %d)", self._proc.pid)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    def _wait_for_server(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        logger.info("Waiting for cosmos action server at %s (timeout %.0fs)...", self.server_url, timeout)
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"cosmos action server exited with code {self._proc.returncode} during startup"
                )
            try:
                with urllib.request.urlopen(self.server_url + "/", timeout=5) as resp:
                    if resp.status == 200:
                        logger.info("cosmos action server is up")
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(3.0)
        raise TimeoutError(f"cosmos action server did not come up within {timeout:.0f}s")

    def _post_predict(self, image_b64: str, prompt: str) -> list[list[float]]:
        payload = json.dumps(
            {
                "image": image_b64,
                "prompt": prompt,
                "domain_name": self.domain_name,
                "image_size": self.image_size,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.server_url + "/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        actions = body.get("action") or []
        if not actions:
            raise RuntimeError(f"cosmos action server returned no actions: {body.get('error', body)}")
        return actions

    # ------------------------------------------------------------------
    # Observation / action translation
    # ------------------------------------------------------------------

    def _encode_concat_view(self, images: dict[str, np.ndarray]) -> str:
        views: list[np.ndarray] = []
        values = list(images.values())
        for i, cam in enumerate(self.cameras):
            img = images.get(cam)
            if img is None:
                # Positional fallback, then last-available (keeps `vla-eval test`'s
                # 2-camera stub obs working for 3-camera configs; real benchmarks
                # send all named cameras so this never triggers there).
                img = values[i] if i < len(values) else values[-1]
                logger.warning(
                    "Camera %r missing from observation (have %s); substituting %s",
                    cam,
                    list(images),
                    f"images[{min(i, len(values) - 1)}]",
                )
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = (arr * 255.0).round().astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            pil = Image.fromarray(arr)
            if pil.size != (self.image_size, self.image_size):
                pil = pil.resize((self.image_size, self.image_size), resample=Image.Resampling.BILINEAR)
            views.append(np.asarray(pil, dtype=np.uint8))
        concat = np.concatenate(views, axis=1) if len(views) > 1 else views[0]
        buf = io.BytesIO()
        Image.fromarray(concat).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _augment_prompt(self, task_description: str) -> str:
        if len(self.cameras) <= 1:
            return task_description
        prompt = _append_prompt_sentence(task_description, _CONCAT_VIEW_SENTENCE)
        return _append_prompt_sentence(prompt, _layout_sentence(self.cameras))

    def _chunk_to_env_actions(self, raw: list[list[float]]) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < self.raw_action_dim:
            raise ValueError(f"Expected (T, >={self.raw_action_dim}) action chunk, got {arr.shape}")
        arr = arr[:, : self.raw_action_dim]
        if self.action_mode == "joint_passthrough":
            # e.g. BEHAVIOR-1K R1Pro: 23-D absolute joint positions, no decoding.
            return np.ascontiguousarray(arr)
        out = np.zeros((arr.shape[0], 7), dtype=np.float32)
        for t in range(arr.shape[0]):
            out[t, 0:3] = arr[t, 0:3]  # frame-wise delta position, controller units
            mat = _rot6d_to_matrix(arr[t, 3:9].astype(np.float64))
            out[t, 3:6] = quat_to_axisangle(matrix_to_quat(mat))
            out[t, 6] = self._remap_gripper(float(arr[t, 9]))
        return out

    def _remap_gripper(self, g: float) -> float:
        """closed_loop_eval._remap_gripper: model raw -> LIBERO [-1,1], negative=open."""
        if self.gripper_mode == "zero_one":
            return float(np.clip(g * 2.0 - 1.0, -1.0, 1.0)) * -1.0
        if self.gripper_mode == "pm_one":
            return float(np.clip(g, -1.0, 1.0))
        if self.gripper_mode == "pm_one_flip":
            return float(np.clip(-g, -1.0, 1.0))
        raise ValueError(f"Unknown gripper_mode={self.gripper_mode!r}. Use zero_one/pm_one/pm_one_flip.")

    def _warmup(self) -> None:
        logger.info("Warmup inference...")
        dummy = np.zeros((self.image_size, self.image_size * max(1, len(self.cameras)), 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(dummy).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        self._post_predict(b64, self._augment_prompt("pick up the object"))
        logger.info("Warmup complete")

    # ------------------------------------------------------------------
    # Harness interface
    # ------------------------------------------------------------------

    def get_observation_params(self) -> dict[str, Any]:
        if self.action_mode == "joint_passthrough":
            return {"send_proprio": self.send_proprio}
        return {"send_wrist_image": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        if self.action_mode == "joint_passthrough":
            return {"joints": DimSpec("joints", self.raw_action_dim, "joint_positions_r1pro")}
        return {"position": POSITION_DELTA, "rotation": ROTATION_AA, "gripper": GRIPPER_CLOSE_POS}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "language": LANGUAGE}

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        images: dict[str, np.ndarray] = obs.get("images", {})
        if not images:
            raise ValueError("No images in observation")
        image_b64 = self._encode_concat_view(images)
        prompt = self._augment_prompt(obs.get("task_description", ""))
        raw_chunk = self._post_predict(image_b64, prompt)
        return {"actions": self._chunk_to_env_actions(raw_chunk)}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(Cosmos3PolicyModelServer)
