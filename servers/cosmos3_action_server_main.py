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

from cosmos_framework.scripts import action_policy_server_libero as _srv

_orig_build = _srv.ActionServerArgs.build_setup_overrides


def _build_without_guardrails(self):
    overrides = _orig_build(self)
    overrides.guardrails = False
    return overrides


_srv.ActionServerArgs.build_setup_overrides = _build_without_guardrails

if __name__ == "__main__":
    _srv.main()
