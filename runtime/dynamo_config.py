"""Set the dynamo knobs ComfyUI needs, where its worker thread can actually see them.

Two knobs, both for failures that were measured on an RTX 5090 with torch 2.13.0+cu130.

``automatic_dynamic_shapes``
    Left at True, dynamo re-specialises a changed shape as symbolic and
    ``_produce_dyn_sizes_from_int_tuple`` raises "Expect size to be a plain tuple of ints".
    That is what makes generate_api_v2 and inpaint_api_v2 fail outright with the plain
    TorchCompileModel node they ship with: ComfyUI clones the model with disable_dynamic=True,
    but the process-wide knob still routes compilation down the dynamic path.
    TorchCompileModelAdvanced only avoided it by passing dynamic="false" explicitly.

``recompile_limit``
    81 inpaint buckets x 2 batch sizes = 162 shapes against a default budget of 8. Four core
    DiT frames (comfy/ops.py, ldm/flux/layers.py:192 and :316, ldm/flux/math.py) hit the ceiling
    and fall back to eager without saying so.

Assigning them is not enough. ``torch._dynamo.config`` keeps user overrides in a
``contextvars.ContextVar``, and ComfyUI executes prompts on a worker thread that starts with an
empty context, so an assignment made at import time is invisible where compilation happens.
Measured on the pod, torch 2.13.0+cu130::

    override_type                                  ContextVar
    read_after_assign_main_thread                  256
    read_from_other_thread_after_assign              8      <- the leak
    read_from_other_thread_after_default_override  512

Overwriting the entry's ``default`` is context-independent, which is what this does. The
assignment is kept as well so the current context matches.
"""

from __future__ import annotations

SETTINGS = {
    "recompile_limit": 512,
    "accumulated_recompile_limit": 8192,
    "automatic_dynamic_shapes": False,
}


def apply(verbose: bool = True) -> dict[str, object]:
    """Apply the settings and return what ended up in effect.

    Import failures are not fatal: without torch this package still has to load for the tests
    that exercise the pure-numpy nodes.
    """
    try:
        import torch._dynamo.config as config
    except Exception as exc:  # pragma: no cover - depends on the host having torch
        if verbose:
            print(f"[dynamo_config] torch недоступен, настройки не применены: {exc}", flush=True)
        return {}

    applied: dict[str, object] = {}
    for name, value in SETTINGS.items():
        entry = config._config.get(name)
        if entry is None:
            if verbose:
                print(f"[dynamo_config] нет такой настройки: {name}", flush=True)
            continue
        entry.default = value
        setattr(config, name, value)
        applied[name] = entry.default

    if verbose and applied:
        print("[dynamo_config] " + "  ".join(f"{k}={v}" for k, v in applied.items()), flush=True)
    return applied
