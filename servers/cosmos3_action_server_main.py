"""Entry shim for ``cosmos_framework.scripts.action_policy_server_libero``.

Runs the official cosmos action server with guardrails disabled. The server
hard-codes ``SetupOverrides``' default (``guardrails=True``) and exposes no CLI
flag for it, but the guardrail models live in the gated ``nvidia/Cosmos-Guardrail1``
HF repo (no token/approval on this cluster), so startup dies in the download.
``guardrails=False`` is a first-class framework knob (``GuardrailOverrides``)
and every inference call site handles ``guardrails is None`` — the guardrails
are a text/video content filter for generation outputs, unused in closed-loop
policy serving where prompts are fixed task descriptions.

Must run under cosmos-framework's venv python with PYTHONPATH pointing at the
cosmos-framework repo root (the wrapper server arranges both).
"""

import os

from cosmos_framework.scripts import action_policy_server_libero as _srv

_orig_build = _srv.ActionServerArgs.build_setup_overrides


def _build_without_guardrails(self):
    overrides = _orig_build(self)
    overrides.guardrails = False
    return overrides


_srv.ActionServerArgs.build_setup_overrides = _build_without_guardrails

# Training-parity view description (COSMOS3_VIEW_DESCRIPTION env, set by the
# wrapper): the stock server's ``_build_json_prompt`` omits
# ``additional_view_description``, so JSON prompts get only the bare
# concat_view template in ``cinematography.framing``. Datasets that train with
# a layout sentence (e.g. Behavior1KLeRobotDataset.CONCAT_VIEW_LAYOUT_DESCRIPTION)
# need it injected to keep serve-time prompts byte-identical to training.
_view_description = os.environ.get("COSMOS3_VIEW_DESCRIPTION")
if _view_description:
    from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter

    _orig_format = ActionPromptJsonFormatter.__call__

    def _format_with_view_description(self, data_dict):
        data_dict.setdefault("additional_view_description", _view_description)
        return _orig_format(self, data_dict)

    ActionPromptJsonFormatter.__call__ = _format_with_view_description

if __name__ == "__main__":
    _srv.main()
