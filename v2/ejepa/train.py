"""Training loops.

Every world-model method shares the belief encoder, predictor, optimiser, data stream, batch
size, rollout length K and step budget.  Only the training signal differs:

  jepa1   stage 1: masked time-frequency latent prediction on spectrogram crops (no actions,
          no labels); EMA target with a cosine momentum schedule.
  jepa    stage 2: frozen stage-1 encoder S.  q(z_t) ~ s_t,  q(P(z_t,u_t)) ~ s_{t+1} (teacher
          forcing) and q(P^k(z_t,u)) ~ s_{t+k} for k = 1..K (rollout), all L1 against the frozen
          observation embeddings.  Targets cannot collapse because S is frozen.
  recon   same rollouts, MSE between Dec(z) and the 64-bin spectrum (encoded + predicted)
  sup     same rollouts, MSE between a speed head and the true omega (labelled episodes only)
  jepa_ft start from a trained JEPA world model, then the 'sup' loss on labelled episodes
  jepa_naive  the single-stage variant with actions inside the EMA-target encoder (kept to
          document the action-shortcut collapse; see the report)
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import Episodes, norm_omega
from .models import SpectroJEPA, WorldModel, effective_rank, sample_masks, target_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")


@dataclass
class TrainConfig:
    method: str
    seed: int = 0
    steps: int = 2500
    batch: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    rollout_k: int = 64
    w_consist: float = 1.0
    anchors: int = 24
    warmup: int = 16
    ema: tuple = (0.996, 1.0)
    latent_dim: int = 48
    ctx_dim: int = 0                     # >0: adaptive (context-conditioned) predictor
    labelled: int | None = None          # number of labelled training episodes (sup / jepa_ft)
    init_from: str | None = None         # stage-1 checkpoint (jepa) or JEPA world model (jepa_ft)
    eval_every: int = 250
    crop: int | None = 200               # random 2 s training windows (evaluation uses full episodes)
    stage1_crop: int = 32
    tag: str = ""
    extra: dict = field(default_factory=dict)


def _batch(ep: Episodes, idx, u_prev_all, crop: int | None = None, rng=None):
    """Episodes idx, optionally a random time crop of length `crop` (encoder restarts at the crop)."""
    t0 = 0 if crop is None else int(rng.integers(0, ep.obs.shape[1] - crop + 1))
    sl = slice(t0, None if crop is None else t0 + crop)
    obs = torch.from_numpy(ep.obs[idx, sl].astype(np.float32)).to(DEVICE, non_blocking=True)
    u = torch.from_numpy(ep.u[idx, sl]).to(DEVICE)
    u_prev = torch.from_numpy(u_prev_all[idx, sl]).to(DEVICE)
    w = torch.from_numpy(norm_omega(ep.omega[idx, sl])).to(DEVICE)
    return obs, u, u_prev, w


def _gather_future(x, t_anchor, k):
    """x (B,T,...) ; t_anchor (B,A) -> x[b, t_anchor + 1 .. t_anchor + k]  as (B*A, k, ...)."""
    b = x.shape[0]
    t = t_anchor[..., None] + torch.arange(1, k + 1, device=x.device)
    bi = torch.arange(b, device=x.device)[:, None, None].expand_as(t)
    return x[bi, t].reshape(-1, k, *x.shape[2:])


def _gather_u(u, t_anchor, k):
    b = u.shape[0]
    t = t_anchor[..., None] + torch.arange(0, k, device=u.device)
    bi = torch.arange(b, device=u.device)[:, None, None].expand_as(t)
    return u[bi, t].reshape(-1, k)


def compute_loss(model: WorldModel, cfg: TrainConfig, obs, u, u_prev, w):
    b, t_len, _ = obs.shape
    k, wu = cfg.rollout_k, cfg.warmup
    z, _, hmap = model.encoder(obs, u_prev, return_map=True)        # (B,T,D), (B,C,T,64)
    # Adaptive models: the machine embedding c is inferred from transitions in the first half
    # of the crop only; every predictor target lies in the second half (no leakage).
    t1 = wu
    if model.ctx_dim:
        t1 = t_len // 2
        model.set_context_from(z[:, wu:t1], u[:, wu:t1])
    t_anchor = torch.randint(t1, t_len - k, (b, cfg.anchors), device=obs.device)
    bi = torch.arange(b, device=obs.device)[:, None].expand_as(t_anchor)
    z0 = z[bi, t_anchor].reshape(-1, z.shape[-1])
    z_roll = model.predictor.rollout(z0, _gather_u(u, t_anchor, k))  # (B*A,K,D)
    z_tf = model.predictor(z[:, t1:-1], u[:, t1:-1])                 # predicts t = t1+1 .. T-1
    parts = {}
    if model.method == "jepa":
        q = model.target_head
        with torch.no_grad():
            s = target_features(hmap)                               # frozen future targets (B,T,C*32)
        parts["now"] = F.l1_loss(q(z[:, wu:]), s[:, wu:])
        parts["roll"] = F.l1_loss(q(z_roll), _gather_future(s, t_anchor, k))
        parts["tf"] = F.l1_loss(q(z_tf), s[:, t1 + 1:])
        loss = parts["now"] + parts["roll"] + parts["tf"]
    elif model.method == "recon":
        dec = model.decoder
        parts["ae"] = F.mse_loss(dec(z[:, wu:]), obs[:, wu:])
        parts["roll"] = F.mse_loss(dec(z_roll), _gather_future(obs, t_anchor, k))
        parts["tf"] = F.mse_loss(dec(z_tf), obs[:, t1 + 1:])
        loss = parts["ae"] + parts["roll"] + parts["tf"]
    elif model.method in ("sup", "jepa_ft"):
        head = lambda x: model.speed_head(x).squeeze(-1)  # noqa: E731
        parts["ae"] = F.mse_loss(head(z[:, wu:]), w[:, wu:])
        parts["roll"] = F.mse_loss(head(z_roll), _gather_future(w, t_anchor, k))
        parts["tf"] = F.mse_loss(head(z_tf), w[:, t1 + 1:])
        loss = parts["ae"] + parts["roll"] + parts["tf"]
    else:
        raise ValueError(model.method)
    # Latent consistency (TD-MPC2 style), identical for every method: predicted latents track
    # stop-gradient encoder latents, so open-loop rollouts stay on the manifold that probes and
    # heads are fitted on.  It cannot cause collapse: z must still explain the method's targets.
    parts["consist"] = F.mse_loss(z_roll, _gather_future(z.detach(), t_anchor, k))
    loss = loss + cfg.w_consist * parts["consist"]
    parts["loss"] = loss
    return loss, parts, z


def _write_history(out_dir, cfg, history):
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    with (out_dir / "history.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted({k for r in history for k in r}))
        writer.writeheader(); writer.writerows(history)


def _cosine(cfg):
    warm = max(1, cfg.steps // 20)
    return lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps))


def train_stage1(cfg: TrainConfig, train_ep: Episodes, val_ep: Episodes, out_dir: Path, log=print):
    """Masked spectrogram JEPA; saves the EMA target encoder (with feature standardisation)."""
    from .evaluate import fit_ridge, apply_ridge

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = SpectroJEPA().to(DEVICE)
    params = list(model.context.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _cosine(cfg))
    history, t_start = [], time.time()
    batch = cfg.batch * 2
    u_prev_all = train_ep.u_prev()
    for step in range(1, cfg.steps + 1):
        idx = np.sort(rng.choice(train_ep.n, size=batch, replace=False))
        obs, *_ = _batch(train_ep, idx, u_prev_all, cfg.stage1_crop, rng)
        mask = sample_masks(batch, cfg.stage1_crop, gen, DEVICE)
        loss, h_ctx = model.loss(obs, mask)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        m0, m1 = cfg.ema
        model.ema_update(m1 - (m1 - m0) * 0.5 * (1 + math.cos(math.pi * step / cfg.steps)))
        if step % cfg.eval_every == 0 or step == 1 or step == cfg.steps:
            with torch.no_grad():
                feats = target_features(model.target(torch.from_numpy(val_ep.obs[:32].astype(np.float32)).to(DEVICE)))
                f = feats[:, 8:].reshape(-1, feats.shape[-1]).cpu().numpy()
                y = norm_omega(val_ep.omega[:32, 8:]).reshape(-1)
                half = len(f) // 2
                probe = fit_ridge(f[:half], y[:half], 1e-4)
                frame_rmse = float(np.sqrt(np.mean((apply_ridge(probe, f[half:]) - y[half:]) ** 2)) * 100)
                rank = effective_rank(feats.reshape(-1, feats.shape[-1])[:4096])
            row = {"step": step, "loss": float(loss), "frame_probe_rmse": frame_rmse, "eff_rank": rank,
                   "elapsed_s": time.time() - t_start}
            history.append(row); log(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}))
    torch.save(model.target.state_dict(), out_dir / "encoder.pt")
    torch.save(model.state_dict(), out_dir / "stage1_full.pt")
    _write_history(out_dir, cfg, history)
    return model.target


def train(cfg: TrainConfig, train_ep: Episodes, val_ep: Episodes, out_dir: Path, log=print):
    """Train one world model; returns it.  Writes config.json, history.csv, model.pt to out_dir."""
    from .evaluate import quick_validation   # local import to avoid a cycle

    if cfg.method == "jepa1":
        return train_stage1(cfg, train_ep, val_ep, out_dir, log)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = WorldModel("sup" if cfg.method == "jepa_ft" else cfg.method, cfg.latent_dim, cfg.ctx_dim).to(DEVICE)
    model.method = cfg.method
    if cfg.method == "jepa":
        if cfg.init_from:          # None -> random frozen trunk (ablation: is stage-1 pre-training needed?)
            model.encoder.trunk.load_state_dict(torch.load(cfg.init_from, map_location=DEVICE))
        model.encoder.trunk_frozen = True
        for p in model.encoder.trunk.parameters():
            p.requires_grad_(False)
        log(f"stage-2 JEPA: frozen spectrogram trunk from {cfg.init_from or 'random initialisation'}")
    elif cfg.init_from:
        state = torch.load(cfg.init_from, map_location=DEVICE)
        res = model.load_state_dict({k: v for k, v in state.items() if not k.startswith("target_head")}, strict=False)
        log(f"init from {cfg.init_from}; missing={res.missing_keys}")
    pool = np.arange(train_ep.n) if cfg.labelled is None else np.arange(min(cfg.labelled, train_ep.n))
    u_prev_all = train_ep.u_prev()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _cosine(cfg))
    history, best, best_state, t_start = [], float("inf"), None, time.time()
    supervised = cfg.method in ("sup", "jepa_ft")
    for step in range(1, cfg.steps + 1):
        model.train()
        idx = rng.choice(pool, size=min(cfg.batch, len(pool)), replace=len(pool) < cfg.batch)
        loss, parts, z = compute_loss(model, cfg, *_batch(train_ep, np.sort(idx), u_prev_all, cfg.crop, rng))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % cfg.eval_every == 0 or step == 1 or step == cfg.steps:
            row = {"step": step, **{k: float(v.detach()) for k, v in parts.items()},
                   "eff_rank": effective_rank(z[:, cfg.warmup:].reshape(-1, z.shape[-1])[:4096].detach()),
                   "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t_start}
            row.update(quick_validation(model, train_ep, val_ep))
            history.append(row)
            log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))
            # Supervised methods are early-stopped on labelled validation speed error;
            # self-supervised methods keep the final checkpoint (no label-based selection).
            if supervised and row["val_roll_rmse"] < best:
                best = row["val_roll_rmse"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model.pt")
    _write_history(out_dir, cfg, history)
    model.eval()
    return model


def load_model(run_dir: Path) -> WorldModel:
    cfg = json.loads((run_dir / "config.json").read_text())
    model = WorldModel("sup" if cfg["method"] == "jepa_ft" else cfg["method"], cfg["latent_dim"], cfg.get("ctx_dim", 0)).to(DEVICE)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=DEVICE))
    model.method = cfg["method"]
    model.encoder.trunk_frozen = cfg["method"] == "jepa"
    model.eval()
    return model
