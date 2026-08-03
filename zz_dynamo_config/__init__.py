"""Set the dynamo knobs ComfyUI needs, in a way that survives its worker thread.

torch._dynamo.config keeps user overrides in a contextvars.ContextVar, so a plain assignment is
visible only inside the context that made it. ComfyUI executes prompts on a worker thread, and a
thread starts with an empty context, so the assignment is invisible there and the read falls back
to the entry's default. Measured on this pod: assign 256 on the main thread, read it from another
thread, get 8. Overwriting the entry's *default* is context-independent and survives.

Two knobs, both for a failure that was measured, not anticipated:

recompile_limit
    81 inpaint buckets x 2 batch sizes = 162 shapes against a default of 8. Four core DiT frames
    (comfy/ops.py, ldm/flux/layers.py:192 and :316, ldm/flux/math.py) hit the ceiling and fell
    back to eager without saying so.

automatic_dynamic_shapes
    Left at True, dynamo re-specialises a changed shape as symbolic and
    _produce_dyn_sizes_from_int_tuple raises "Expect size to be a plain tuple of ints". That is
    what makes generate_api_v2 and inpaint_api_v2 fail outright with the plain TorchCompileModel
    node they ship with — ComfyUI clones the model with disable_dynamic=True, but the global knob
    still routes compilation down the dynamic path. TorchCompileModelAdvanced only worked because
    it passes dynamic="false" explicitly.
"""

import torch._dynamo.config as config

SETTINGS = {
    "recompile_limit": 512,
    "accumulated_recompile_limit": 8192,
    "automatic_dynamic_shapes": False,
}

for name, value in SETTINGS.items():
    entry = config._config.get(name)
    if entry is None:
        print(f"[zz_dynamo_limit] нет такой настройки: {name}", flush=True)
        continue
    entry.default = value
    setattr(config, name, value)

print("[zz_dynamo_limit] " + "  ".join(
    f"{name}={config._config[name].default}" for name in SETTINGS if name in config._config),
    flush=True)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
