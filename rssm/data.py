"""
Data downloading, loading, and batch sampling.
"""

import os
import urllib.request

import numpy as np
import torch

from .config import (
    DATA_DIR, CHECKPOINT_DIR, REMOTE_BASE, REMOTE_FILES,
    HOLD_OUT, DEVICE,
)


def download_data(force: bool = False):
    """Download the dataset, normalization stats, and pretrained checkpoint."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for local_name, remote_path in REMOTE_FILES.items():
        if local_name.endswith(".pt"):
            dest = os.path.join(CHECKPOINT_DIR, local_name)
        else:
            dest = os.path.join(DATA_DIR, local_name)

        if not force and os.path.exists(dest):
            continue

        url = REMOTE_BASE + remote_path
        print(f"downloading {remote_path} ...")
        urllib.request.urlretrieve(url, dest)

    print("all files ready")


class SO101Dataset:
    """
    Holds the 12-episode SO-101 pick-and-place dataset in memory.

    Attributes
    ----------
    episodes : list[dict]
        Each dict has keys 'frames' (T,64,64,3 uint8),
        'states' (T,6), 'actions' (T,6).
    s_mean, s_std, a_mean, a_std : np.ndarray
        Per-joint normalization statistics.
    """

    def __init__(self, data_dir: str = DATA_DIR):
        z = np.load(os.path.join(data_dir, "so101_mini.npz"))
        n = int(z["n_episodes"])
        self.episodes = [
            {
                "frames":  z[f"f{i}"],
                "states":  z[f"s{i}"],
                "actions": z[f"a{i}"],
            }
            for i in range(n)
        ]

        norm = np.load(os.path.join(data_dir, "so101_norm.npz"))
        self.s_mean = norm["s_mean"]
        self.s_std  = norm["s_std"]
        self.a_mean = norm["a_mean"]
        self.a_std  = norm["a_std"]

    @property
    def train_episodes(self):
        return self.episodes[:-HOLD_OUT]

    @property
    def eval_episodes(self):
        return self.episodes[-HOLD_OUT:]

    def summary(self) -> str:
        total_frames = sum(len(e["frames"]) for e in self.episodes)
        shape = self.episodes[0]["frames"].shape[1:]
        return (
            f"{len(self.episodes)} episodes, "
            f"{total_frames:,} frames, "
            f"frame shape {shape}"
        )

    def sample_batch(
        self,
        batch_size: int = 8,
        seq_len: int = 24,
        device: torch.device = DEVICE,
    ):
        """
        Sample a random training batch.

        Returns (frames, states, actions) each of shape (B, L, ...) as tensors.
        """
        eps = self.train_episodes
        B, L = batch_size, seq_len

        fr = np.zeros((B, L, 64, 64, 3), np.float32)
        st = np.zeros((B, L, 6), np.float32)
        ac = np.zeros((B, L, 6), np.float32)

        for b in range(B):
            e = eps[np.random.randint(len(eps))]
            t = np.random.randint(0, len(e["frames"]) - L)
            fr[b] = e["frames"][t : t + L] / 255.0
            st[b] = (e["states"][t : t + L] - self.s_mean) / self.s_std
            ac[b] = (e["actions"][t : t + L] - self.a_mean) / self.a_std

        return (
            torch.tensor(fr, device=device),
            torch.tensor(st, device=device),
            torch.tensor(ac, device=device),
        )
