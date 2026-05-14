from __future__ import annotations

import torch
import torch.nn.functional as F


def _parse_color(color: str) -> list[float]:
    color = color.strip()
    values: list[float] = []
    if color.startswith("#"):
        hex_color = color.lstrip("#")
        if len(hex_color) == 3:
            values = [int(c * 2, 16) / 255.0 for c in hex_color]
        elif len(hex_color) == 4:
            values = [int(c * 2, 16) / 255.0 for c in hex_color]
        elif len(hex_color) == 6:
            values = [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
        elif len(hex_color) == 8:
            values = [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4, 6)]
        else:
            raise ValueError(f"Invalid hex color format: {color}")
    else:
        for raw in color.split(","):
            component = float(raw.strip())
            values.append(component / 255.0 if component > 1.0 else component)
    if len(values) < 3:
        raise ValueError("Color must have at least 3 components")
    return values


class DrawMaskOverlayAdvancedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "overlay_mode": (["solid_color", "mask_grayscale"], {"default": "mask_grayscale"}),
                "color": ("STRING", {"default": "255, 255, 255", "tooltip": "Used in solid_color mode. Supports RGB/RGBA or hex."}),
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "device": (["cpu", "gpu"], {"default": "cpu"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = "image/masking"
    DESCRIPTION = "Overlay a mask preview over the image. Supports solid-color fill and semi-transparent grayscale mask preview."

    def apply(self, image, mask, overlay_mode, color, opacity, device="cpu"):
        del device  # Keep API parity with the source node; geometry stays deterministic on CPU.
        batch, height, width, channels = image.shape

        work_image = image.clone().float()
        work_mask = mask.clone().float()
        if work_mask.ndim == 4:
            work_mask = work_mask[:, 0]
        if work_mask.shape[-2:] != (height, width):
            mode = "nearest-exact" if "nearest-exact" in torch._C._nn.__dict__ else "nearest"
            work_mask = F.interpolate(work_mask.unsqueeze(1), size=(height, width), mode=mode)[:, 0]
        if work_mask.shape[0] < batch:
            repeats = (batch + work_mask.shape[0] - 1) // work_mask.shape[0]
            work_mask = work_mask.repeat(repeats, 1, 1)[:batch]
        elif work_mask.shape[0] > batch:
            work_mask = work_mask[:batch]

        work_mask = work_mask.clamp(0.0, 1.0)
        out: list[torch.Tensor] = []

        color_values = _parse_color(color)
        solid_rgb = torch.tensor(color_values[:3], dtype=work_image.dtype, device=work_image.device).view(1, 1, 3)
        color_alpha = float(color_values[3]) if len(color_values) >= 4 else 1.0
        opacity = float(max(0.0, min(1.0, opacity)))

        for index in range(batch):
            current_image = work_image[index]
            current_mask = work_mask[index]
            if overlay_mode == "solid_color":
                blend = (current_mask.unsqueeze(-1) * opacity * color_alpha).clamp(0.0, 1.0)
                overlay_rgb = solid_rgb.expand(height, width, 3)
            elif overlay_mode == "mask_grayscale":
                blend = torch.full((height, width, 1), opacity, dtype=work_image.dtype, device=work_image.device)
                overlay_rgb = current_mask.unsqueeze(-1).expand(height, width, 3)
            else:
                raise ValueError(f"Unsupported overlay_mode: {overlay_mode}")

            if channels == 4:
                image_rgb = current_image[..., :3]
                image_alpha = current_image[..., 3:4]
                mixed_rgb = image_rgb * (1.0 - blend) + overlay_rgb * blend
                mixed_alpha = image_alpha
                out.append(torch.cat((mixed_rgb, mixed_alpha), dim=-1))
            else:
                out.append(current_image * (1.0 - blend) + overlay_rgb * blend)

        return (torch.stack(out, dim=0).to(image.dtype).cpu(),)
