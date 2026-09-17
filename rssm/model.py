"""
RSSM model: Encoder, Decoder, and the recurrent state-space model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import H, S, EMB, A_DIM, MIN_STD


class Encoder(nn.Module):
    """frame (64x64x3) -> 1024-dim embedding"""

    def __init__(self, emb_dim: int = EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  nn.ELU(),    # -> 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(),    # -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(),   # -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(),  # -> 4x4
            nn.Flatten(),
            nn.Linear(256 * 16, emb_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64, 64, 3) float in [0,1]
        return self.net(x.permute(0, 3, 1, 2) - 0.5)


class Decoder(nn.Module):
    """[h, s] -> reconstructed frame.  Reads BOTH halves of the state."""

    def __init__(self, h_dim: int = H, s_dim: int = S):
        super().__init__()
        self.fc = nn.Linear(h_dim + s_dim, 256 * 16)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.ELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.fc(torch.cat([h, s], -1)).view(-1, 256, 4, 4)
        return self.net(x).permute(0, 2, 3, 1) + 0.5


class RSSM(nn.Module):
    """
    Recurrent State-Space Model.

    State at each timestep = (h, s)
      - h: deterministic, 256-d, updated by a GRU. Facts live here.
      - s: stochastic, 32-d, sampled every step. Doubt lives here.
    """

    def __init__(
        self,
        a_dim: int = A_DIM,
        h_dim: int = H,
        s_dim: int = S,
        emb_dim: int = EMB,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.s_dim = s_dim

        self.enc = Encoder(emb_dim)
        self.dec = Decoder(h_dim, s_dim)

        # auxiliary head: read joint angles out of the latent (for legibility)
        self.joint_head = nn.Sequential(
            nn.Linear(h_dim + s_dim, 256), nn.ELU(),
            nn.Linear(256, 6),
        )

        # input MLP that merges (s, a) before feeding the GRU
        self.in_mlp = nn.Sequential(
            nn.Linear(s_dim + a_dim, 256), nn.ELU(),
        )

        # the deterministic belt
        self.gru = nn.GRUCell(256, h_dim)

        # prior: guess s from h alone (blind -- this is what dreaming uses)
        self.prior_mlp = nn.Sequential(
            nn.Linear(h_dim, 256), nn.ELU(),
            nn.Linear(256, 2 * s_dim),
        )

        # posterior: guess s from h + encoder embedding (peeking at the frame)
        self.post_mlp = nn.Sequential(
            nn.Linear(h_dim + emb_dim, 256), nn.ELU(),
            nn.Linear(256, 2 * s_dim),
        )

    def dist(self, out: torch.Tensor):
        """Split network output into (mean, std) of a diagonal Gaussian."""
        mu, raw_std = out.chunk(2, dim=-1)
        return mu, F.softplus(raw_std) + MIN_STD

    def step_h(self, h: torch.Tensor, s: torch.Tensor, a: torch.Tensor):
        """Advance the deterministic belt: h_t = GRU(h_{t-1}, [s_{t-1}, a_{t-1}])."""
        return self.gru(self.in_mlp(torch.cat([s, a], -1)), h)

    def prior(self, h: torch.Tensor):
        """Blind guess: predict s from h alone (no frame)."""
        return self.dist(self.prior_mlp(h))

    def posterior(self, h: torch.Tensor, emb: torch.Tensor):
        """Informed guess: predict s from h + encoded frame."""
        return self.dist(self.post_mlp(torch.cat([h, emb], -1)))

    def init_state(self, batch_size: int = 1, device=None):
        """Return zero-initialized (h, s)."""
        device = device or next(self.parameters()).device
        h = torch.zeros(batch_size, self.h_dim, device=device)
        s = torch.zeros(batch_size, self.s_dim, device=device)
        return h, s


def kl_divergence(m1, s1, m2, s2):
    """KL( N(m1,s1) || N(m2,s2) ), summed over the last dimension."""
    return (
        torch.log(s2 / s1)
        + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2)
        - 0.5
    ).sum(-1)
