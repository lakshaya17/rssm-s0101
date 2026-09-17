"""
Evaluation utilities: dreaming, linear probes, ablation variants.
"""

import numpy as np
import torch

from .config import H, S, CONTEXT, HORIZON, HOLD_OUT, DEVICE
from .model import RSSM
from .data import SO101Dataset


@torch.no_grad()
def dream(
    model: RSSM,
    episode: dict,
    dataset: SO101Dataset,
    start: int,
    context: int = CONTEXT,
    horizon: int = HORIZON,
    device: torch.device = DEVICE,
):
    """
    Give the model `context` real frames, then dream for `horizon` steps
    using only the actions (camera off).

    Returns
    -------
    real_frames : np.ndarray   (context+horizon, 64, 64, 3) float32 [0,1]
    dreamed_frames : np.ndarray  (horizon, 64, 64, 3) float32 [0,1]
    dreamed_joints : np.ndarray  (horizon, 6)
    real_states_normed : np.ndarray  (context+horizon, 6)
    """
    model.eval()
    T = context + horizon
    fr = torch.tensor(
        episode["frames"][start : start + T] / 255.0,
        dtype=torch.float32, device=device,
    )
    st = (episode["states"][start : start + T] - dataset.s_mean) / dataset.s_std
    ac = torch.tensor(
        (episode["actions"][start : start + T] - dataset.a_mean) / dataset.a_std,
        dtype=torch.float32, device=device,
    )

    emb = model.enc(fr)
    h, s = model.init_state(1, device)

    # warm up on real frames
    for t in range(context):
        if t > 0:
            h = model.step_h(h, s, ac[t - 1 : t])
        qm, _ = model.posterior(h, emb[t : t + 1])
        s = qm

    # dream with camera off
    frames, joints = [], []
    for k in range(horizon):
        h = model.step_h(h, s, ac[context + k - 1 : context + k])
        pm, _ = model.prior(h)
        s = pm
        frames.append(model.dec(h, s)[0].clamp(0, 1).cpu().numpy())
        joints.append(
            model.joint_head(torch.cat([h, s], -1))[0].cpu().numpy()
        )

    return fr.cpu().numpy(), np.array(frames), np.array(joints), st


@torch.no_grad()
def linear_probe(
    model: RSSM,
    dataset: SO101Dataset,
    device: torch.device = DEVICE,
):
    """
    Fit a linear map from h to joint angles on training episodes.

    Returns
    -------
    r2 : np.ndarray (6,)        per-joint R^2
    r2_control : np.ndarray (6,) shuffled-label control
    """
    model.eval()
    Hs, Ys = [], []

    for e in dataset.train_episodes:
        fr = torch.tensor(
            e["frames"] / 255.0, dtype=torch.float32, device=device,
        )
        ac = torch.tensor(
            (e["actions"] - dataset.a_mean) / dataset.a_std,
            dtype=torch.float32, device=device,
        )
        emb = model.enc(fr)
        h, s = model.init_state(1, device)

        for t in range(len(fr)):
            if t > 0:
                h = model.step_h(h, s, ac[t - 1 : t])
            qm, _ = model.posterior(h, emb[t : t + 1])
            s = qm
            Hs.append(h[0].cpu().numpy())
            Ys.append(
                (e["states"][t] - dataset.s_mean) / dataset.s_std
            )

    Hs = np.array(Hs, np.float32)
    Ys = np.array(Ys, np.float32)

    # add bias column
    A = np.concatenate([Hs, np.ones((len(Hs), 1), np.float32)], axis=1)
    W, *_ = np.linalg.lstsq(A, Ys, rcond=None)
    r2 = 1 - ((A @ W - Ys) ** 2).sum(0) / ((Ys - Ys.mean(0)) ** 2).sum(0)

    # shuffled control
    idx = np.random.permutation(len(Ys))
    Wc, *_ = np.linalg.lstsq(A, Ys[idx], rcond=None)
    r2_ctrl = 1 - ((A @ Wc - Ys[idx]) ** 2).sum(0) / (
        (Ys[idx] - Ys[idx].mean(0)) ** 2
    ).sum(0)

    return r2, r2_ctrl


@torch.no_grad()
def dream_variant(
    model: RSSM,
    episode: dict,
    dataset: SO101Dataset,
    start: int,
    mode: str = "both",
    horizon: int = 40,
    device: torch.device = DEVICE,
):
    """
    Dream with modified wiring for the ablation study.

    mode: 'both'     -- full RSSM (as designed)
          'freeze_s' -- keep the last warm-up sample forever
          'starve_h' -- zero out s before feeding it to the GRU
    """
    model.eval()
    T = CONTEXT + horizon
    fr = torch.tensor(
        episode["frames"][start : start + T] / 255.0,
        dtype=torch.float32, device=device,
    )
    ac = torch.tensor(
        (episode["actions"][start : start + T] - dataset.a_mean) / dataset.a_std,
        dtype=torch.float32, device=device,
    )

    emb = model.enc(fr)
    h, s = model.init_state(1, device)

    for t in range(CONTEXT):
        if t > 0:
            h = model.step_h(h, s, ac[t - 1 : t])
        qm, _ = model.posterior(h, emb[t : t + 1])
        s = qm

    out = []
    for k in range(horizon):
        if mode == "starve_h":
            h = model.step_h(h, torch.zeros_like(s), ac[CONTEXT + k - 1 : CONTEXT + k])
        else:
            h = model.step_h(h, s, ac[CONTEXT + k - 1 : CONTEXT + k])

        if mode != "freeze_s":
            pm, ps = model.prior(h)
            s = pm if mode == "both" else pm + ps * torch.randn_like(ps)

        out.append(model.dec(h, s)[0].clamp(0, 1).cpu().numpy())

    return fr.cpu().numpy(), out


@torch.no_grad()
def uncertainty_over_time(
    model: RSSM,
    episode: dict,
    dataset: SO101Dataset,
    device: torch.device = DEVICE,
):
    """
    Run the posterior over an entire episode and return the mean std at each step.
    """
    model.eval()
    fr = torch.tensor(
        episode["frames"] / 255.0, dtype=torch.float32, device=device,
    )
    ac = torch.tensor(
        (episode["actions"] - dataset.a_mean) / dataset.a_std,
        dtype=torch.float32, device=device,
    )

    emb = model.enc(fr)
    h, s = model.init_state(1, device)
    sigmas = []

    for t in range(len(fr)):
        if t > 0:
            h = model.step_h(h, s, ac[t - 1 : t])
        qm, qs = model.posterior(h, emb[t : t + 1])
        s = qm
        sigmas.append(float(qs.mean()))

    return sigmas
