from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import FiLMGenerator
from .harmonizer_blocks import NAFBlockLite, NAFEncoderLite, resize_inner


DEFAULT_CORRECTION_LIMITS = {
    "gain_limit": 1.20,
    "gamma_limit": 1.20,
    "bias_limit": 0.50,
    "mix_limit": 0.90,
    "detail_limit": 0.35,
    "gate_bias": -0.10,
}

DEFAULT_STREAM_STRENGTHS = {
    "gain": 1.0,
    "gamma": 1.0,
    "bias": 1.0,
    "mix": 1.0,
    "detail": 1.0,
    "gate": 1.0,
    "attention": 1.0,
}


def _identity_color_matrix(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3, 1, 1)
    return eye.expand(batch, 3, 3, height, width)


def apply_local_color_matrix(inner_rgb: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bijhw,bjhw->bihw", matrix, inner_rgb)


def reconstruct_corrected_strip(
    strip_rgb: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    *,
    outer_width: int = 128,
    gain_limit: float = DEFAULT_CORRECTION_LIMITS["gain_limit"],
    gamma_limit: float = DEFAULT_CORRECTION_LIMITS["gamma_limit"],
    bias_limit: float = DEFAULT_CORRECTION_LIMITS["bias_limit"],
    mix_limit: float = DEFAULT_CORRECTION_LIMITS["mix_limit"],
    detail_limit: float = DEFAULT_CORRECTION_LIMITS["detail_limit"],
    gate_bias: float = DEFAULT_CORRECTION_LIMITS["gate_bias"],
    disable_spatial_gate: bool = False,
    stream_strengths: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    strengths = dict(DEFAULT_STREAM_STRENGTHS)
    if stream_strengths:
        strengths.update({key: float(value) for key, value in stream_strengths.items() if key in strengths})
    height = strip_rgb.shape[-2]
    inner_width = strip_rgb.shape[-1] - outer_width
    outer = strip_rgb[..., :outer_width]
    inner = strip_rgb[..., outer_width:]
    gain = resize_inner(outputs["gain_lowres"], height, inner_width)
    gamma = resize_inner(outputs["gamma_lowres"], height, inner_width)
    gate = resize_inner(outputs["gate_lowres"], height, inner_width)
    bias = F.interpolate(outputs["bias_lowres"], size=(height, inner_width), mode="bilinear", align_corners=False)
    detail = F.interpolate(outputs["detail_lowres"], size=(height, inner_width), mode="bilinear", align_corners=False)
    mix = F.interpolate(outputs["mix_lowres"].flatten(1, 2), size=(height, inner_width), mode="bilinear", align_corners=False)
    mix = mix.view(strip_rgb.shape[0], 3, 3, height, inner_width)
    gamma_map = torch.exp(gamma_limit * strengths["gamma"] * torch.tanh(gamma))
    curved = inner.clamp(1e-7, 1.0).pow(gamma_map)
    color_matrix = _identity_color_matrix(strip_rgb.shape[0], height, inner_width, strip_rgb.device, strip_rgb.dtype)
    color_matrix = color_matrix + mix_limit * strengths["mix"] * torch.tanh(mix)
    mixed = apply_local_color_matrix(curved, color_matrix)
    gain_map = torch.exp(gain_limit * strengths["gain"] * torch.tanh(gain))
    bias_map = bias_limit * strengths["bias"] * torch.tanh(bias)
    detail_map = detail_limit * strengths["detail"] * torch.tanh(detail)
    proposed = mixed * gain_map + bias_map + detail_map
    confidence = strengths["gate"] * torch.sigmoid(gate + gate_bias)
    confidence = confidence.clamp(0.0, 1.0)
    attn_lowres = outputs.get("attention_lowres")
    if attn_lowres is not None and not disable_spatial_gate:
        attn_full = F.interpolate(attn_lowres, size=(height, inner_width), mode="bilinear", align_corners=False)
        spatial_gate = (strengths["attention"] * attn_full * 2.0).clamp(0.0, 1.0)
    else:
        spatial_gate = torch.ones_like(confidence)
    corrected_inner = inner + confidence * spatial_gate * (proposed - inner)
    corrected_inner = corrected_inner.clamp(0.0, 1.0)
    corrected_strip = torch.cat([outer, corrected_inner], dim=-1)
    return {
        "corrected_inner": corrected_inner,
        "corrected_strip": corrected_strip,
        "confidence": confidence,
    }


class DecoderFuse(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.block1 = NAFBlockLite(out_channels)
        self.block2 = NAFBlockLite(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(self.in_proj(x), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = x + self.skip_proj(skip)
        x = self.block1(x)
        return self.block2(x)


class SeamHarmonizerV3(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        channels: tuple[int, ...] = (32, 64, 128, 192),
        blocks: tuple[int, ...] = (2, 2, 4, 6),
        outer_width: int = 128,
        boundary_band_px: int = 24,
        correction_limits: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.outer_width = outer_width
        self.boundary_band_px = boundary_band_px
        self.correction_limits = dict(DEFAULT_CORRECTION_LIMITS)
        if correction_limits:
            self.correction_limits.update({key: float(value) for key, value in correction_limits.items()})
        self.encoder = NAFEncoderLite(in_channels=in_channels, channels=channels, blocks=blocks)
        self.bottleneck = nn.Sequential(NAFBlockLite(channels[-1]), NAFBlockLite(channels[-1]))
        self.decode2 = DecoderFuse(channels[-1], channels[2], channels[2])
        self.decode1 = DecoderFuse(channels[2], channels[1], channels[1])
        self.decode0 = DecoderFuse(channels[1], channels[0], channels[0])
        self.context_film = FiLMGenerator(channels[-1])
        self.coarse_adapter = nn.Conv2d(channels[0] + channels[1] + channels[2], channels[2], kernel_size=1)
        self.coarse_head = nn.Sequential(
            nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels[2], 18, kernel_size=1),
        )
        self.attention_head = nn.Sequential(
            nn.Conv2d(channels[2], channels[2] // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels[2] // 2, 1, kernel_size=1),
        )
        self._init_identity()

    def _init_identity(self) -> None:
        nn.init.zeros_(self.coarse_head[-1].weight)
        nn.init.zeros_(self.coarse_head[-1].bias)
        nn.init.zeros_(self.attention_head[-1].weight)
        nn.init.zeros_(self.attention_head[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        disable_spatial_gate: bool = False,
        stream_strengths: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        f0, f1, f2, f3 = self.encoder(x)
        bottleneck = self.bottleneck(f3)
        gamma, beta = self.context_film(bottleneck)
        bottleneck = bottleneck * (1.0 + torch.tanh(gamma)) + beta
        d2 = self.decode2(bottleneck, f2)
        d1 = self.decode1(d2, f1)
        d0 = self.decode0(d1, f0)
        coarse_h = max(16, x.shape[-2] // 4)
        coarse_w = max(8, (x.shape[-1] - self.outer_width) // 4)
        fused = torch.cat(
            [
                F.interpolate(d0, size=(coarse_h, coarse_w), mode="bilinear", align_corners=False),
                F.interpolate(d1, size=(coarse_h, coarse_w), mode="bilinear", align_corners=False),
                F.interpolate(d2, size=(coarse_h, coarse_w), mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        adapted = self.coarse_adapter(fused)
        coarse = self.coarse_head(adapted)
        gain_lowres, gamma_lowres, bias_lowres, mix_flat, detail_lowres, gate_lowres = torch.split(coarse, [1, 1, 3, 9, 3, 1], dim=1)
        mix_lowres = mix_flat.view(x.shape[0], 3, 3, coarse_h, coarse_w)
        attention_lowres = torch.sigmoid(self.attention_head(adapted))
        recon = reconstruct_corrected_strip(
            x[:, :3],
            {
                "gain_lowres": gain_lowres,
                "gamma_lowres": gamma_lowres,
                "bias_lowres": bias_lowres,
                "mix_lowres": mix_lowres,
                "detail_lowres": detail_lowres,
                "gate_lowres": gate_lowres,
                "attention_lowres": attention_lowres,
            },
            outer_width=self.outer_width,
            disable_spatial_gate=disable_spatial_gate,
            stream_strengths=stream_strengths,
            **self.correction_limits,
        )
        return {
            "gain_lowres": gain_lowres,
            "gamma_lowres": gamma_lowres,
            "bias_lowres": bias_lowres,
            "mix_lowres": mix_lowres,
            "detail_lowres": detail_lowres,
            "gate_lowres": gate_lowres,
            "attention_lowres": attention_lowres,
            **recon,
        }
