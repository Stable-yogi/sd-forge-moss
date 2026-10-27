"""Spectrum-style step forecast for the MOSS-Image DiT (optional speed-up, off by default).

Idea: Han, Shi, Li, Ye, Guo, Ermon, "Adaptive Spectral Feature Forecasting for Diffusion Sampling
Acceleration" (arXiv:2603.01623, https://github.com/hanjq17/Spectrum, MIT licence).
On most steps the network does not run. The image tokens that leave the last block are predicted from
the ones saved on earlier real steps (Chebyshev polynomials over the step number, ridge regression),
and only the small output head runs on them. This file is our own code, written from the paper;
no code of the ComfyUI node "ComfyUI-Spectrum-MOSS" (GPL-3.0) was copied.

How it plugs in (no core edit): a Forge unet function wrapper. A real step runs the full DiT and
saves the output of the last block (a forward hook). A forecast step swaps the block list for one
tiny module that writes the prediction into the image tokens, so the embedders and the head run
as normal and the 42 blocks are skipped.
"""
import math

import torch
import torch.nn as nn

MARK = "_moss_spectrum"


def build_plan(steps, warmup, window, flex, tail):
    """List of bool per step: True = run the network. Same window idea as the paper:
    after the warm-up, one real step every floor(window) steps, the window grows by `flex` each time."""
    steps = int(steps)
    real = [False] * steps
    for n in range(min(int(warmup), steps)):
        real[n] = True
    for n in range(max(steps - int(tail), 0), steps):
        real[n] = True
    cached, ws = 0, float(window)
    for n in range(int(warmup), steps):
        if real[n]:
            continue
        if (cached + 1) % max(math.floor(ws), 1) == 0:
            real[n] = True
            ws = round(ws + float(flex), 3)
            cached = 0
        else:
            cached += 1
    return real


def _cheb(tau, degree):
    """(len(tau), degree+1) Chebyshev T0..Tdegree on tau in [-1, 1]."""
    tau = tau.reshape(-1, 1)
    cols = [torch.ones_like(tau)]
    if degree >= 1:
        cols.append(tau)
    for _ in range(2, degree + 1):
        cols.append(2 * tau * cols[-1] - cols[-2])
    return torch.cat(cols[: degree + 1], dim=1)


class _Inject(nn.Module):
    """Stands in for the block list on a forecast step: puts the prediction into the image tokens."""

    def __init__(self, pred, hw):
        super().__init__()
        self.pred, self.hw = pred, hw

    def forward(self, x, *args, **kwargs):
        x[:, : self.hw] = self.pred.to(x.dtype)
        return x


class _Stream:
    """History and step counter of one cond / uncond stream."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.t = []
        self.h = []
        self.last_ts = None


class Spectrum:
    def __init__(self, steps, warmup=4, window=2.0, flex=0.75, tail=2, degree=4, lam=0.1, blend=1.0, history=8):
        self.steps = int(steps)
        self.plan = build_plan(steps, warmup, window, flex, tail)
        self.degree, self.lam, self.blend, self.history = int(degree), float(lam), float(blend), int(history)
        self.streams = {}
        self.real_calls = 0
        self.skip_calls = 0

    # -------------------------------------------------------------- forecast
    def _predict(self, s, t_star):
        k = len(s.t)
        end = max(self.steps - 1, 1)
        tau = lambda v: 2.0 * torch.as_tensor(v, dtype=torch.float64) / end - 1.0
        X = _cheb(tau(torch.tensor(s.t, dtype=torch.float64)), self.degree)  # (K, P)
        xs = _cheb(tau(t_star), self.degree)  # (1, P)
        A = X.T @ X + self.lam * torch.eye(X.shape[1], dtype=torch.float64)
        w = (xs @ torch.linalg.solve(A, X.T)).reshape(-1).tolist()  # one weight per saved step
        out = None
        for wk, hk in zip(w, s.h):
            term = hk.float() * wk
            out = term if out is None else out.add_(term)
        if self.blend < 1.0 and k >= 2 and out is not None:  # mix with a straight line through the last two saved steps
            dt = max(s.t[-1] - s.t[-2], 1e-8)
            lin = s.h[-1].float() + ((t_star - s.t[-1]) / dt) * (s.h[-1].float() - s.h[-2].float())
            out = self.blend * out + (1.0 - self.blend) * lin
        return out

    # -------------------------------------------------------------- one model call
    def run(self, apply_model, key, x, timestep, c):
        s = self.streams.setdefault(key, _Stream())
        ts = float(timestep.flatten()[0])
        if s.last_ts is not None and ts > s.last_ts + 1e-6:  # time went up: a new sampling began
            s.reset()
        s.last_ts = ts
        n = s.n
        s.n += 1
        dm = apply_model.__self__.diffusion_model
        if not hasattr(dm, "layers"):
            return apply_model(x, timestep, **c)
        hw = x.shape[-2] * x.shape[-1]
        real = n >= len(self.plan) or self.plan[n] or len(s.h) < 2 or s.h[-1].shape[0] != x.shape[0]
        if real:
            self.real_calls += 1
            grab = []
            hook = dm.layers[-1].register_forward_hook(lambda _m, _i, o: grab.append(o[:, :hw].detach().clone()))
            try:
                out = apply_model(x, timestep, **c)
            finally:
                hook.remove()
            if grab:
                s.t.append(float(n))
                s.h.append(grab[0])
                if len(s.h) > self.history:
                    s.t, s.h = s.t[-self.history:], s.h[-self.history:]
            return out
        self.skip_calls += 1
        pred = self._predict(s, float(n))
        saved = dm.layers
        dm.layers = nn.ModuleList([_Inject(pred, hw)])
        try:
            return apply_model(x, timestep, **c)
        finally:
            dm.layers = saved

    def summary(self):
        return "Spectrum: %d real + %d forecast model calls (plan %d of %d steps real)" % (self.real_calls, self.skip_calls, sum(self.plan), self.steps)


def make_wrapper(state):
    """Unet function wrapper: every model call goes through the Spectrum state."""

    def wrapper(apply_model, args):
        return state.run(apply_model, tuple(args["cond_or_uncond"]), args["input"], args["timestep"], args["c"])

    setattr(wrapper, MARK, state)
    return wrapper
