"""
Training loop for the RSSM.
"""

import time

import numpy as np
import torch

from .config import (
    H, S, DEVICE,
    BATCH_SIZE, SEQ_LEN, LR, FREE_NATS, TRAIN_STEPS,
)
from .model import RSSM, kl_divergence
from .data import SO101Dataset


def train(
    model: RSSM,
    dataset: SO101Dataset,
    steps: int = TRAIN_STEPS,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    lr: float = LR,
    free_nats: float = FREE_NATS,
    log_every: int = 100,
    device: torch.device = DEVICE,
) -> np.ndarray:
    """
    Train the RSSM and return a (steps, 3) array of
    [reconstruction_loss, joint_loss, kl_loss] per step.
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    history = []
    t_start = time.time()

    for step in range(steps):
        fr, st, ac = dataset.sample_batch(batch_size, seq_len, device)
        B, L = batch_size, seq_len

        # encode all frames at once
        emb = model.enc(fr.reshape(B * L, 64, 64, 3)).view(B, L, -1)

        h, s = model.init_state(B, device)
        l_rec = l_kl = l_joint = 0.0

        for t in range(L):
            if t > 0:
                h = model.step_h(h, s, ac[:, t - 1])

            pm, ps = model.prior(h)
            qm, qs = model.posterior(h, emb[:, t])
            s = qm + qs * torch.randn_like(qs)  # sample from posterior

            # loss 1: repaint the frame
            l_rec += (
                (model.dec(h, s) - fr[:, t]) ** 2
            ).sum(dim=(1, 2, 3)).mean()

            # auxiliary joint-angle readout
            l_joint += (
                (model.joint_head(torch.cat([h, s], -1)) - st[:, t]) ** 2
            ).sum(-1).mean()

            # loss 2: KL between blind guess and informed guess
            # KL balancing: 80% gradient to prior, 20% to posterior
            k = (
                0.8 * kl_divergence(qm.detach(), qs.detach(), pm, ps).mean()
                + 0.2 * kl_divergence(qm, qs, pm.detach(), ps.detach()).mean()
            )
            l_kl += torch.clamp(k, min=free_nats)

        loss = (l_rec + 10.0 * l_joint + 1.0 * l_kl) / L

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()

        rec_val = float(l_rec) / L
        jnt_val = float(l_joint) / L
        kl_val  = float(l_kl) / L
        history.append((rec_val, jnt_val, kl_val))

        if step % log_every == 0:
            elapsed = time.time() - t_start
            print(
                f"step {step:5d}  "
                f"repaint {rec_val:8.1f}   "
                f"joints {jnt_val:6.3f}   "
                f"KL {kl_val:5.2f}   "
                f"({elapsed:.0f}s)"
            )

    return np.array(history)
