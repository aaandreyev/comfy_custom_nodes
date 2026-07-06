from __future__ import annotations


def apply_seamfix_prefix(text: str, enabled: bool) -> str:
    if not enabled or not text:
        return text
    first = text.split("\n", 1)[0].strip()
    if first == "SEAMFIX":
        return text
    return "SEAMFIX\n" + text


class SeamfixCLIPTextEncode:
    """
    Same as CLIP Text Encode but prepends ``SEAMFIX`` + newline to the prompt (for inpainting
    workflows). Plug this instead of the positive ``CLIP Text Encode`` when sampling inpaints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": True},
                ),
                "prepend_seamfix": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "If True, sends SEAMFIX + newline before your text."},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"
    DESCRIPTION = (
        "Encodes prompt like CLIP Text Encode but prefixes SEAMFIX and a newline for inpainting pipelines."
    )

    def encode(self, clip, text, prepend_seamfix=True):
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\n"
                "If the clip comes from checkpoint loader your checkpoint may not include CLIP / text encoder."
            )
        final_text = apply_seamfix_prefix(str(text), bool(prepend_seamfix))
        tokens = clip.tokenize(final_text)
        return (clip.encode_from_tokens_scheduled(tokens),)
