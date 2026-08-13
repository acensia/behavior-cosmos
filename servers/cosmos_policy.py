# /// script
# requires-python = "~=3.10"
# dependencies = [
#     "vla-eval",
#     "cosmos-policy[cu128]",
#     "numpy>=1.24",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../vla-evaluation-harness", editable = true }
# cosmos-policy = { path = "../cosmos-policy", editable = true }
#
# [tool.uv]
# no-build-isolation-package = ["flash-attn"]
# ///
"""Cosmos Policy model server for vla-evaluation-harness.

Wraps NVlabs/cosmos-policy inference (``cosmos_policy.experiments.robot.cosmos_utils``)
behind the harness's ``PredictModelServer`` interface. Defaults target the LIBERO
checkpoint (nvidia/Cosmos-Policy-LIBERO-Predict2-2B).

Observation mapping (harness -> cosmos), LIBERO suite:
  - Harness images are flipped on BOTH axes (``preprocess_libero_image``:
    ``img[::-1, ::-1]``); cosmos was trained on vertically-flipped-only images
    (``np.flipud``). We undo the horizontal flip here (``img[:, ::-1]``).
  - Harness ``states`` is 8-D ``[eef_pos(3), axis_angle(3), gripper_qpos(2)]``;
    cosmos ``proprio`` is 9-D ``[gripper_qpos(2), eef_pos(3), eef_quat_xyzw(4)]``.
    We convert axis-angle back to a quaternion. NOTE: the harness canonicalizes
    quaternion sign (w >= 0) before the axis-angle conversion, so the recovered
    quaternion may differ in sign from robosuite's raw output — validate against
    cosmos's own eval on a few episodes (see CLAUDE.md).
  - Actions come back unnormalized in LIBERO's native 7-D action space
    ``[dx, dy, dz, axis-angle delta(3), gripper]`` with gripper -1=open/+1=close,
    executed open-loop as a 16-step chunk (harness chunk buffer handles this).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from vla_eval.rotation import axisangle_to_matrix, matrix_to_quat
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)
from vla_eval.types import Action, Observation

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer

logger = logging.getLogger(__name__)


@dataclass
class _CosmosCfg:
    """Local mirror of the fields cosmos_utils reads off PolicyEvalConfig.

    We can't import PolicyEvalConfig from run_libero_eval: that module pulls in
    the LIBERO simulator, which is not (and should not be) installed in this
    server's environment. Defaults copied from run_libero_eval.PolicyEvalConfig.
    """

    suite: str = "libero"
    model_family: str = "cosmos"
    config: str = ""
    ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""

    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = True
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True
    chunk_size: int = 16
    num_open_loop_steps: int = 16
    seed: int = 7
    randomize_seed: bool = False


class CosmosPolicyModelServer(PredictModelServer):
    """Cosmos Policy (Predict2-2B post-trained) model server."""

    def __init__(
        self,
        ckpt_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        config_name: str = "cosmos_predict2_2b_480p_libero__inference_only",
        dataset_stats_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
        t5_text_embeddings_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
        suite: str = "libero",
        num_denoising_steps_action: int = 5,
        seed: int = 7,
        randomize_seed: bool = False,
        unflip_horizontal: bool = True,
        warmup: bool = True,
        *,
        chunk_size: int = 16,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.suite = suite
        self.seed = seed
        self.randomize_seed = randomize_seed
        self.num_denoising_steps_action = num_denoising_steps_action
        self.unflip_horizontal = unflip_horizontal

        self._cfg = _CosmosCfg(
            suite=suite,
            config=config_name,
            ckpt_path=ckpt_path,
            dataset_stats_path=dataset_stats_path,
            t5_text_embeddings_path=t5_text_embeddings_path,
            num_denoising_steps_action=num_denoising_steps_action,
            chunk_size=chunk_size,
            num_open_loop_steps=chunk_size,
            seed=seed,
            randomize_seed=randomize_seed,
        )

        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_model,
            init_t5_text_embeddings_cache,
            load_dataset_stats,
        )

        logger.info("Loading Cosmos Policy from %s (config=%s)", ckpt_path, config_name)
        self._model, self._model_config = get_model(self._cfg)
        self._dataset_stats = load_dataset_stats(dataset_stats_path)
        self._t5_cache = init_t5_text_embeddings_cache(t5_text_embeddings_path)
        logger.info("Loaded %d cached T5 instruction embeddings", len(self._t5_cache))

        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """One dummy forward so the first HELLO-gated request isn't JIT-cold.

        Uses an instruction already in the T5 cache — an unknown instruction
        would trigger an on-the-fly T5 encode (large model download).
        """
        if not self._t5_cache:
            logger.warning("T5 embedding cache is empty; skipping warmup")
            return
        instruction = next(iter(self._t5_cache))
        dummy = {
            "primary_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "proprio": np.zeros(9, dtype=np.float32),
        }
        logger.info("Warmup inference (instruction=%r)...", instruction)
        self._run_inference(dummy, instruction)
        logger.info("Warmup complete")

    def _run_inference(self, observation: dict[str, Any], task_description: str) -> np.ndarray:
        from cosmos_policy.experiments.robot.cosmos_utils import get_action

        ret = get_action(
            self._cfg,
            self._model,
            self._dataset_stats,
            observation,
            task_description,
            seed=self.seed,
            randomize_seed=self.randomize_seed,
            num_denoising_steps_action=self.num_denoising_steps_action,
        )
        actions = np.asarray(ret["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        return actions

    # ------------------------------------------------------------------
    # Harness interface
    # ------------------------------------------------------------------

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_wrist_image": True, "send_state": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"position": POSITION_DELTA, "rotation": ROTATION_AA, "gripper": GRIPPER_CLOSE_POS}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": STATE_EEF_POS_AA_GRIP, "language": LANGUAGE}

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        images: dict[str, np.ndarray] = obs.get("images", {})
        img_values = list(images.values())
        primary = images.get("agentview", img_values[0] if img_values else None)
        wrist = images.get("wrist", img_values[1] if len(img_values) > 1 else None)
        if primary is None:
            raise ValueError("No primary image in observation")
        if wrist is None:
            raise ValueError(
                "No wrist image in observation — cosmos policy requires it "
                "(send_wrist_image should be auto-merged from get_observation_params)"
            )

        primary = np.ascontiguousarray(np.asarray(primary, dtype=np.uint8))
        wrist = np.ascontiguousarray(np.asarray(wrist, dtype=np.uint8))
        if self.unflip_horizontal:
            # Harness flips both axes; cosmos expects vertical-only flip.
            primary = np.ascontiguousarray(primary[:, ::-1])
            wrist = np.ascontiguousarray(wrist[:, ::-1])

        state = obs.get("states", obs.get("state"))
        if state is None:
            raise ValueError("No proprioceptive state in observation (send_state)")
        state = np.asarray(state, dtype=np.float32).flatten()
        proprio = self._harness_state_to_proprio(state)

        task_description = obs.get("task_description", "")
        actions = self._run_inference(
            {"primary_image": primary, "wrist_image": wrist, "proprio": proprio},
            task_description,
        )
        return {"actions": actions}

    @staticmethod
    def _harness_state_to_proprio(state: np.ndarray) -> np.ndarray:
        """[eef_pos(3), axis_angle(3), gripper_qpos(2)] -> [gripper_qpos(2), eef_pos(3), quat_xyzw(4)]."""
        if len(state) < 8:
            raise ValueError(f"Expected 8-D LIBERO state, got {len(state)}-D")
        eef_pos = state[0:3]
        quat_xyzw = matrix_to_quat(axisangle_to_matrix(state[3:6]))
        gripper_qpos = state[6:8]
        return np.concatenate([gripper_qpos, eef_pos, quat_xyzw]).astype(np.float32)


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(CosmosPolicyModelServer)
