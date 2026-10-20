# Qwen3.5 text model (hybrid Gated DeltaNet + gated full attention), text-only, prefill-only.
# Ported from ComfyUI 0.42.1 comfy/text_encoders/qwen35.py for the MOSS-Image text encoder.
# State dict layout: model.embed_tokens / model.layers.N.* / model.norm (vision, mtp, lm_head are dropped by the loader).

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.memory_management import cast_to
from backend.nn.llm.llama import apply_rope, precompute_freqs_cis


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, add=False):
        super().__init__()
        self.eps = eps
        self.add = add
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x):
        w = cast_to(self.weight, dtype=torch.float32, device=x.device)
        if self.add:
            w = w + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=w, eps=self.eps).to(x.dtype)


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64):
    initial_dtype = query.dtype
    query = F.normalize(query.float(), dim=-1)
    key = F.normalize(key.float(), dim=-1)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_new = v_i - (k_cumdecay[:, :, i]) @ state
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        state = state * g[:, :, i, -1, None, None].exp() + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        hidden = c["hidden_size"]
        self.num_key_heads = c["linear_num_key_heads"]
        self.num_value_heads = c["linear_num_value_heads"]
        self.key_head_dim = c["linear_key_head_dim"]
        self.value_head_dim = c["linear_value_head_dim"]
        self.conv_kernel_size = c["conv_kernel_size"]
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Linear(hidden, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.num_value_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_value_heads))
        # depthwise causal conv, kept as a raw parameter (works for every weight format)
        self.conv1d = nn.Module()
        self.conv1d.weight = nn.Parameter(torch.empty(conv_dim, 1, self.conv_kernel_size))
        self.norm = RMSNorm(self.value_head_dim, eps=c["rms_norm_eps"], add=False)

    def forward(self, x):
        B, L, _ = x.shape
        proj = self.in_proj_qkv(x)
        z = self.in_proj_z(x)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        w = cast_to(self.conv1d.weight, dtype=proj.dtype, device=x.device)
        mixed = F.conv1d(proj.transpose(1, 2), w, None, padding=self.conv_kernel_size - 1, groups=w.shape[0])[:, :, :L]
        mixed = F.silu(mixed).transpose(1, 2)
        query, key, value = mixed.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)

        g_decay = -cast_to(self.A_log, dtype=torch.float32, device=x.device).exp()
        dt_bias = cast_to(self.dt_bias, dtype=torch.float32, device=x.device)
        beta = b.float().sigmoid()
        g = g_decay * F.softplus(a.float() + dt_bias)
        query = query.reshape(B, L, self.num_key_heads, self.key_head_dim)
        key = key.reshape(B, L, self.num_key_heads, self.key_head_dim)
        value = value.reshape(B, L, self.num_value_heads, self.value_head_dim)
        if self.num_value_heads != self.num_key_heads:
            rep = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        out = torch_chunk_gated_delta_rule(query, key, value, g=g, beta=beta)
        out = self.norm(out.reshape(-1, self.value_head_dim)) * F.silu(z.reshape(-1, self.value_head_dim).float()).to(out.dtype)
        return self.out_proj(out.reshape(B, L, -1))


class GatedAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.num_heads = c["num_attention_heads"]
        self.num_kv_heads = c["num_key_value_heads"]
        self.head_dim = c["head_dim"]
        inner = self.num_heads * self.head_dim
        self.rotary_dim = int(self.head_dim * c["partial_rotary_factor"])
        self.q_proj = nn.Linear(c["hidden_size"], inner * 2, bias=False)
        self.k_proj = nn.Linear(c["hidden_size"], self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(c["hidden_size"], self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(inner, c["hidden_size"], bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=c["rms_norm_eps"], add=True)
        self.k_norm = RMSNorm(self.head_dim, eps=c["rms_norm_eps"], add=True)

    def forward(self, x, freqs_cis):
        B, L, _ = x.shape
        qg = self.q_proj(x).view(B, L, self.num_heads, self.head_dim * 2)
        xq, gate = qg[..., : self.head_dim], qg[..., self.head_dim :]
        gate = gate.reshape(B, L, -1)
        xq = self.q_norm(xq).transpose(1, 2)
        xk = self.k_norm(self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        xv = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        r = self.rotary_dim
        q_rot, k_rot = apply_rope(xq[..., :r].float().contiguous(), xk[..., :r].float().contiguous(), freqs_cis)
        xq = torch.cat([q_rot.to(xq.dtype), xq[..., r:]], dim=-1)
        xk = torch.cat([k_rot.to(xk.dtype), xk[..., r:]], dim=-1)

        rep = self.num_heads // self.num_kv_heads
        xk = xk.repeat_interleave(rep, dim=1)
        xv = xv.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out * gate.sigmoid())


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(c["intermediate_size"], c["hidden_size"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        interval = c.get("full_attention_interval", 4)
        self.linear = (index + 1) % interval != 0
        if self.linear:
            self.linear_attn = GatedDeltaNet(c)
        else:
            self.self_attn = GatedAttention(c)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"], add=True)
        self.post_attention_layernorm = RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"], add=True)

    def forward(self, x, freqs_cis):
        h = self.input_layernorm(x)
        h = self.linear_attn(h) if self.linear else self.self_attn(h, freqs_cis)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen35Body(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = nn.ModuleList([Block(c, i) for i in range(c["num_hidden_layers"])])
        self.norm = RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"], add=True)


class Qwen35TextModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.num_layers = config["num_hidden_layers"]
        self.model = Qwen35Body(config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids, dtype=torch.bfloat16):
        """Returns the final-norm last hidden state, [B, L, hidden]."""
        x = self.model.embed_tokens(input_ids).to(dtype)
        L = x.shape[1]
        rotary_dim = int(self.config["head_dim"] * self.config["partial_rotary_factor"])
        position_ids = torch.arange(L, device=x.device).unsqueeze(0)
        freqs_cis = precompute_freqs_cis(rotary_dim, position_ids, self.config["rope_theta"], device=x.device)
        for layer in self.model.layers:
            x = layer(x, freqs_cis)
        return self.model.norm(x)
