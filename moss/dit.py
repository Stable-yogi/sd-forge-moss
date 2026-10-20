# MOSS-Image DiT for Forge Neo.
# Ported from ComfyUI 0.42.1 comfy/ldm/moss_image/model.py (plain torch, no comfy_kitchen kernels).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.attention import attention_function
from backend.memory_management import cast_to
from backend.nn.flux import timestep_embedding


def situ(hidden, gate):
    # soft-capped gated activation
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


class RMSNorm(nn.Module):
    """Plain RMSNorm (weight multiplies, no +1), kept out of Forge's op patching so quant files never touch it."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x):
        w = cast_to(self.weight, dtype=torch.float32, device=x.device)
        return F.rms_norm(x.float(), (x.shape[-1],), weight=w, eps=self.eps).to(x.dtype)


def modulated_rms_norm(x, norm, scale, shift):
    w = cast_to(norm.weight, dtype=torch.float32, device=x.device)
    y = F.rms_norm(x.float(), (x.shape[-1],), weight=w, eps=norm.eps)
    return torch.addcmul(shift.float(), y, 1 + scale.float()).to(x.dtype)


def rope_split_half(x, cos, sin):
    # x: B, N, H, D ; cos/sin: N, D/2 -> pairs (x[i], x[i + D/2])
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).to(x.dtype)


def attention(q, k, v, q_norm, k_norm, rope, heads, transformer_options):
    B, N, _ = q.shape
    q = q_norm(q.view(B, N, heads, -1))
    k = k_norm(k.view(B, N, heads, -1))
    cos, sin = rope
    q = rope_split_half(q, cos, sin).transpose(1, 2)
    k = rope_split_half(k, cos, sin).transpose(1, 2)
    v = v.view(B, N, heads, -1).transpose(1, 2)
    return attention_function(q, k, v, heads, skip_reshape=True, transformer_options=transformer_options)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.q_norm = RMSNorm(dim // num_heads)
        self.k_norm = RMSNorm(dim // num_heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, transformer_options={}):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        x = x + self.proj(attention(q, k, v, self.q_norm, self.k_norm, rope, self.num_heads, transformer_options))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextEmbedder(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, num_heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, num_heads) for _ in range(2)])

    def forward(self, x, rope, transformer_options={}):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, rope, transformer_options)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, fused_qkv=False):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.q_norm = RMSNorm(dim // num_heads)
        self.k_norm = RMSNorm(dim // num_heads)
        names = ("qkv", "o") if fused_qkv else ("q", "k", "v", "o")
        self.proj = nn.ModuleDict({name: nn.Linear(dim, 3 * dim if name == "qkv" else dim, bias=False) for name in names})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, mod, rope, transformer_options={}):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
        y = modulated_rms_norm(x, self.norm1, scale_a, shift_a)
        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, dim=-1)
        else:
            q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        out = attention(q, k, v, self.q_norm, self.k_norm, rope, self.num_heads, transformer_options)
        x = torch.addcmul(x, self.proj["o"](out), gate_a)
        y = modulated_rms_norm(x, self.norm2, scale_f, shift_f)
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, dim=-1)
        return torch.addcmul(x, self.proj["ffn_down"](situ(hidden, gate)), gate_f)


class MOSSImage(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, latent_channels=128, fused_qkv=False, image_model=None, **kwargs):
        super().__init__()
        self.latent_channels = latent_channels
        self.head_dim = dim // num_heads
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextEmbedder(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads, fused_qkv=fused_qkv) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def rope(self, h, w, text_len, device):
        # (r, f, h, w): the image sits after the text on f, its rows and columns centered
        ids = torch.zeros(h * w + text_len, 4, device=device, dtype=torch.float64)
        ids[: h * w, 1] = text_len
        ids[: h * w, 2] = (torch.arange(h, device=device, dtype=torch.float64) - (h - 1) / 2).repeat_interleave(w)
        ids[: h * w, 3] = (torch.arange(w, device=device, dtype=torch.float64) - (w - 1) / 2).repeat(h)
        ids[h * w :, 1] = torch.arange(text_len, device=device, dtype=torch.float64)
        axis = self.head_dim // 4
        scale = torch.arange(0, axis, 2, dtype=torch.float64, device=device) / axis
        omega = 1.0 / (10000 ** scale)
        freqs = torch.cat([ids[:, i : i + 1] * omega[None] for i in range(4)], dim=-1)  # L, head_dim/2
        return freqs.cos().float(), freqs.sin().float()

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        B, _, H, W = x.shape
        dtype, device = x.dtype, x.device
        context = context.to(dtype)
        cos, sin = self.rope(H, W, context.shape[1], device)
        HW = H * W

        vec = self.time_embedding(timestep_embedding(timestep, 256).to(dtype))
        mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
        head_shift, head_scale = self.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)

        txt = self.text_embedder(context, (cos[HW:], sin[HW:]), transformer_options)
        x = torch.cat((self.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)
        del txt

        rope = (cos, sin)
        for block in self.layers:
            x = block(x, mod, rope, transformer_options)

        out = self.final_linear_video(modulated_rms_norm(x[:, :HW], self.final_norm_video, head_scale, head_shift))
        return out.transpose(1, 2).reshape(B, self.latent_channels, H, W)
