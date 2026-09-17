"""
Hyperparameters, paths, and plotting style for the RSSM SO-101 project.
"""

import os
import torch
from matplotlib import rcParams

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def run_output_dir(h=None, s=None, steps=None, tag=None):
    """
    Build a run-specific output directory like:
        outputs/H256_S32_steps2000/
        outputs/H32_S32_steps2000/
        outputs/H256_S32_pretrained/
        outputs/H256_S32_steps600_koopman/   (with a custom tag)

    Falls back to module-level defaults for any arg left as None.
    """
    from . import config as _cfg
    h = h if h is not None else _cfg.H
    s = s if s is not None else _cfg.S
    parts = [f"H{h}", f"S{s}"]
    if steps is not None:
        parts.append(f"steps{steps}")
    else:
        parts.append("pretrained")
    if tag:
        parts.append(tag)
    name = "_".join(parts)
    path = os.path.join(PROJECT_ROOT, "outputs", name)
    os.makedirs(path, exist_ok=True)
    return path

REMOTE_BASE = (
    "https://raw.githubusercontent.com/RajatDandekar/"
    "build-a-world-model-from-scratch/main/lecture-04-dreams-that-last/"
)

REMOTE_FILES = {
    "so101_mini.npz": "data/so101_mini.npz",
    "so101_norm.npz": "data/so101_norm.npz",
    "rssm_so101.pt":  "data/rssm_so101.pt",
}

# ── Model dimensions ───────────────────────────────────────────────────────────
H = 32          # deterministic state (GRU hidden)
S = 32           # stochastic state
EMB = 1024       # encoder embedding
A_DIM = 6        # action dimension (6 joints)
MIN_STD = 0.1

# ── Training defaults ──────────────────────────────────────────────────────────
SEED = 0
BATCH_SIZE = 8
SEQ_LEN = 24
LR = 3e-4
FREE_NATS = 1.0
TRAIN_STEPS = 600
HOLD_OUT = 4     # last 4 episodes held out for evaluation

# ── Evaluation ─────────────────────────────────────────────────────────────────
CONTEXT = 5      # frames of real context before dreaming
HORIZON = 60     # dream steps with camera off

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ── Joint names ────────────────────────────────────────────────────────────────
JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# ── Plotting style ─────────────────────────────────────────────────────────────
PAPER = "#FBF9F1"
INK   = "#16130D"
MUTED = "#6D665A"
TEAL  = "#2E8F8F"
GOLD  = "#DD9F3E"
CLAY  = "#C96442"


def apply_plot_style():
    """Apply the lecture's matplotlib style globally."""
    rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    })
