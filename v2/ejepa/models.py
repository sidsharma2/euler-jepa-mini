"""Shared architecture for every learned method.

Spectrogram trunk    T: x_{t-6:t} -> h_t (C x 64)    causal time-frequency CNN (feature map per frame)
Keypoint readout     R: h_t -> r_t                    1-D conv over frequency + spatial soft-argmax + pools
Belief encoder       E: (r_{0:t}, u_{-1:t-1}) -> z_t  causal GRU over observation features and past commands
Predictor            P: (z_t, u_t) -> z_{t+1}         action-conditioned, residual, re-normalised

z is LayerNorm-normalised and the predictor is residual and re-normalised every step, which
keeps 200-step latent rollouts bounded.  The methods differ only in the training signal:

  jepa   two stages (the V-JEPA 2-AC recipe).  (1) T is pre-trained alone by masked
         time-frequency latent prediction with an EMA target (no actions, no labels).
         (2) T is frozen.  R, E, P and a readout q are trained so that q(P^k(z_t, u))
         predicts the frozen, normalised feature maps of the future frames, y_{t+k} = pool(LN(T(x)_{t+k})).
         Targets are fixed, so the latent cannot collapse onto the action history.
  recon  T, R, E, P trained end to end to decode every latent back to the spectrum
  sup    T, R, E, P trained end to end to regress the true speed (needs labels)
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

N_BINS = 64
RECEPTIVE = 7          # frames of causal temporal context in T (three (3, 5) convolutions)
TARGET_POOL = 2        # frequency pooling of the JEPA target maps (64 -> 32 cells)


class SpectroTrunk(nn.Module):
    """Causal time-frequency CNN: (B,T,64) [+ mask] -> feature map (B,C,T,64)."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.register_buffer("coord", torch.linspace(-1, 1, N_BINS), persistent=False)
        self.layers = nn.ModuleList([nn.Conv2d(3, 32, (3, 5)), nn.Conv2d(32, 32, (3, 5)),
                                     nn.Conv2d(32, channels, (3, 5))])
        self.channels = channels

    def forward(self, x, mask=None):
        b, t, f = x.shape
        coord = self.coord.view(1, 1, 1, f).expand(b, 1, t, f)
        m = torch.zeros_like(x) if mask is None else mask
        h = torch.cat([(x * (1 - m))[:, None], coord, m[:, None]], dim=1)
        for i, conv in enumerate(self.layers):
            h = conv(F.pad(h, (2, 2, 2, 0)))          # causal in time, 'same' in frequency
            if i < len(self.layers) - 1:
                h = F.gelu(h)
        return h


def target_features(h):
    """JEPA target: per-position LayerNorm over channels, frequency-pooled, flattened -> (B,T,C*32)."""
    y = F.layer_norm(h.permute(0, 2, 3, 1), (h.shape[1],))           # (B,T,64,C)
    y = F.avg_pool1d(y.permute(0, 1, 3, 2).flatten(0, 1), TARGET_POOL)  # (B*T,C,32)
    return y.reshape(h.shape[0], h.shape[2], -1)


class KeypointReadout(nn.Module):
    """Per-frame read-out with a spatial soft-argmax (Finn et al. 2016; Levine et al. 2016).

    Each of K maps becomes a distribution over frequency bins; its expected coordinate is a
    keypoint that is linear in where a spectral line sits.  Generic; encodes no physics.
    """

    def __init__(self, channels: int = 32, keypoints: int = 16, out_dim: int = 128):
        super().__init__()
        self.register_buffer("coord", torch.linspace(-1, 1, N_BINS), persistent=False)
        self.conv = nn.Sequential(nn.Conv1d(channels, 32, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(32, keypoints, 5, padding=2))
        self.log_temp = nn.Parameter(torch.full((1, keypoints, 1), 1.0))
        self.mlp = nn.Sequential(nn.Linear(3 * keypoints, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.out_dim = out_dim

    def forward(self, h):                                              # (B,C,T,64) -> (B,T,out)
        b, c, t, f = h.shape
        g = self.conv(h.permute(0, 2, 1, 3).reshape(b * t, c, f))       # (BT,K,64)
        attn = torch.softmax(g * torch.exp(self.log_temp), dim=-1)
        feats = torch.cat([(attn * self.coord).sum(-1), g.mean(-1), g.amax(-1)], dim=-1)
        return self.mlp(feats).reshape(b, t, -1)


class SpectroJEPA(nn.Module):
    """Stage 1: masked time-frequency latent prediction (I-JEPA / A-JEPA style, CNN backbone).

    The context trunk sees the crop with masked blocks removed (plus a mask channel); the EMA
    target trunk sees the full crop.  A convolutional predictor maps context features plus the
    mask to the target map; L1 on masked positions against per-position LayerNorm targets.
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        self.context = SpectroTrunk(channels)
        self.target = copy.deepcopy(self.context)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Conv2d(channels + 1, 64, (3, 5), padding=(1, 2)), nn.GELU(),
            nn.Conv2d(64, 64, (3, 5), padding=(1, 2)), nn.GELU(),
            nn.Conv2d(64, channels, (3, 5), padding=(1, 2)),
        )

    @torch.no_grad()
    def ema_update(self, m: float):
        for t, o in zip(self.target.parameters(), self.context.parameters()):
            t.mul_(m).add_(o.detach(), alpha=1 - m)

    def loss(self, x, mask):
        h_ctx = self.context(x, mask)
        pred = self.predictor(torch.cat([h_ctx, mask[:, None]], dim=1))
        with torch.no_grad():
            tgt = self.target(x)
            tgt = F.layer_norm(tgt.permute(0, 2, 3, 1), (tgt.shape[1],)).permute(0, 3, 1, 2)
        m = mask[:, None].expand_as(pred)
        return (torch.abs(pred - tgt) * m).sum() / m.sum().clamp_min(1.0), h_ctx


def sample_masks(b: int, t: int, gen: torch.Generator, device) -> torch.Tensor:
    """Time-frequency block masks: 2-3 frequency bands over random time spans, plus a future block."""
    mask = torch.zeros(b, t, N_BINS, device=device)
    r = lambda lo, hi: int(torch.randint(lo, hi + 1, (1,), generator=gen))  # noqa: E731
    for i in range(b):
        for _ in range(r(2, 3)):
            w, span = r(6, 16), r(6, t // 2)
            f0, t0 = r(0, N_BINS - w), r(0, t - span)
            mask[i, t0:t0 + span, f0:f0 + w] = 1
        mask[i, t - r(3, 8):] = 1                                   # predict the near future
    return mask


class BeliefEncoder(nn.Module):
    def __init__(self, latent_dim: int = 48, hidden: int = 128, channels: int = 32):
        super().__init__()
        self.trunk = SpectroTrunk(channels)
        self.trunk_frozen = False
        self.readout = KeypointReadout(channels)
        self.inp = nn.Sequential(nn.Linear(self.readout.out_dim + 1, 128), nn.GELU())
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.head = nn.Linear(hidden, latent_dim)
        self.latent_dim = latent_dim

    def trunk_map(self, obs):
        if self.trunk_frozen:
            with torch.no_grad():
                return self.trunk(obs)
        return self.trunk(obs)

    def from_features(self, r, u_prev, h0=None):
        seq, h = self.gru(self.inp(torch.cat([r, u_prev[..., None]], dim=-1)), h0)
        return F.layer_norm(self.head(seq), (self.latent_dim,)), h

    def forward(self, obs, u_prev, h0=None, return_map: bool = False):
        """obs (B,T,64), u_prev (B,T) -> z (B,T,D), h (1,B,H) [, trunk map]."""
        hmap = self.trunk_map(obs)
        z, h = self.from_features(self.readout(hmap), u_prev, h0)
        return (z, h, hmap) if return_map else (z, h)


class Predictor(nn.Module):
    """z_{t+1} = LN(z_t + f(z_t, u_t [, c])).

    With ctx_dim > 0 the predictor is conditioned on a per-episode context embedding c
    (set with `set_context`), inferred from observed transitions -- an adaptive world model.
    """

    def __init__(self, latent_dim: int = 48, hidden: int = 256, ctx_dim: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1 + ctx_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.latent_dim, self.ctx_dim = latent_dim, ctx_dim
        self.c = None

    def set_context(self, c):
        self.c = c

    def _ctx(self, z):
        c = self.c
        if z.dim() == 3:
            return c[:, None].expand(-1, z.shape[1], -1)
        if z.shape[0] == c.shape[0]:
            return c
        return c.repeat_interleave(z.shape[0] // c.shape[0], dim=0)

    def forward(self, z, u):
        parts = [z, u[..., None]] + ([self._ctx(z)] if self.ctx_dim else [])
        return F.layer_norm(z + self.net(torch.cat(parts, dim=-1)), (self.latent_dim,))

    def rollout(self, z0, u_seq):
        """z0 (N,D), u_seq (N,K) -> (N,K,D) latents z_1..z_K."""
        out, z = [], z0
        for k in range(u_seq.shape[1]):
            z = self(z, u_seq[:, k])
            out.append(z)
        return torch.stack(out, dim=1)


class WorldModel(nn.Module):
    """Belief encoder + predictor + method-specific heads."""

    def __init__(self, method: str, latent_dim: int = 48, ctx_dim: int = 0):
        super().__init__()
        self.method = method
        self.encoder = BeliefEncoder(latent_dim)
        self.predictor = Predictor(latent_dim, ctx_dim=ctx_dim)
        self.ctx_dim = ctx_dim
        if ctx_dim:
            # Permutation-invariant transition encoder: mean over (z_t, u_t, z_{t+1}) -> c.
            self.ctx_enc = nn.Sequential(nn.Linear(2 * latent_dim + 1, 128), nn.GELU(),
                                         nn.Linear(128, 128), nn.GELU(), nn.Linear(128, ctx_dim))
        if method == "jepa":
            tdim = self.encoder.trunk.channels * N_BINS // TARGET_POOL
            self.target_head = nn.Sequential(nn.Linear(latent_dim, 256), nn.GELU(), nn.Linear(256, tdim))
        if method == "recon":
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.GELU(),
                                         nn.Linear(256, 256), nn.GELU(), nn.Linear(256, N_BINS))
        # Speed head: trained for 'sup' / 'jepa_ft'; 'jepa' / 'recon' use a frozen ridge probe.
        self.speed_head = nn.Linear(latent_dim, 1)

    def context(self, z, u):
        """Machine embedding from observed transitions: z (B,T,D), u (B,T) -> c (B,ctx)."""
        x = torch.cat([z[:, :-1], u[:, :-1, None], z[:, 1:]], dim=-1)
        return F.layer_norm(self.ctx_enc(x).mean(1), (self.ctx_dim,))

    def set_context_from(self, z, u):
        """Set the predictor context (no-op for non-adaptive models)."""
        if self.ctx_dim:
            self.predictor.set_context(self.context(z, u))


def effective_rank(z):
    """exp(entropy of normalised singular values) of centred latents (M, D)."""
    z = z - z.mean(0)
    s = torch.linalg.svdvals(z.double())
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
