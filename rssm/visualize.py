"""
Plotting utilities for the RSSM SO-101 project.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from .config import (
    JOINTS, CONTEXT, HORIZON, OUTPUT_DIR,
    PAPER, INK, MUTED, TEAL, GOLD, CLAY,
    apply_plot_style,
)

apply_plot_style()


# Module-level override -- set by main.py before any plotting
_output_dir = OUTPUT_DIR


def set_output_dir(path: str):
    """Override the directory where plots are saved."""
    global _output_dir
    _output_dir = path
    os.makedirs(path, exist_ok=True)


def _savefig(fig, name: str):
    os.makedirs(_output_dir, exist_ok=True)
    path = os.path.join(_output_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


# ── Dataset inspection ─────────────────────────────────────────────────────────

def plot_one_row(episode: dict, t: int = 60, save: bool = True):
    """Show one observation + state/action numbers."""
    fig = plt.figure(figsize=(12, 4.4))
    ax = fig.add_axes([0.02, 0.08, 0.34, 0.82])
    ax.imshow(episode["frames"][t] / 255.0)
    ax.axis("off")
    ax.set_title(
        "the OBSERVATION\n64 x 64 x 3 = 12,288 numbers",
        color=TEAL, fontsize=12,
    )

    axt = fig.add_axes([0.42, 0.08, 0.56, 0.82])
    axt.axis("off")
    y = 0.94
    for title, vals, col in [
        ("observation.state  (6 numbers)", episode["states"][t], TEAL),
        ("action  (6 numbers)", episode["actions"][t], CLAY),
    ]:
        axt.text(0, y, title, fontsize=12.5, color=col, fontweight="bold")
        y -= 0.10
        for name, v in zip(JOINTS, vals):
            axt.text(0.03, y, f"{name:<16s}", fontsize=11, family="monospace")
            axt.text(
                0.45, y, f"{v:8.2f}", fontsize=11,
                family="monospace", color=col, fontweight="bold",
            )
            y -= 0.072
        y -= 0.03

    fig.suptitle("ONE row of the dataset", fontsize=14)
    if save:
        _savefig(fig, "01_one_row.png")
    plt.close(fig)


def plot_gripper_fork(episode: dict, save: bool = True):
    """Show the fork point where the gripper closes."""
    grip = episode["actions"][:, 5]
    fork = int(np.argmax(np.abs(np.diff(grip))))

    offs = [-8, -4, 0, 4, 8]
    fig, axes = plt.subplots(1, len(offs), figsize=(13, 3.0))
    for ax, o in zip(axes, offs):
        t = int(np.clip(fork + o, 0, len(episode["frames"]) - 1))
        ax.imshow(episode["frames"][t] / 255.0)
        ax.axis("off")
        ax.set_title(
            "the gripper closes" if o == 0 else f"{o:+d} steps",
            fontsize=10,
            color=CLAY if o == 0 else MUTED,
            fontweight="bold" if o == 0 else "normal",
        )

    fig.suptitle(
        "The fork: from here the future genuinely has more than one answer",
        y=1.06,
    )
    if save:
        _savefig(fig, "02_gripper_fork.png")
    plt.close(fig)
    print(f"the gripper closes fastest at timestep {fork}")


# ── Training curves ────────────────────────────────────────────────────────────

def plot_training_curves(history: np.ndarray, save: bool = True):
    """Plot the three training losses. history shape: (steps, 3)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    labels = [
        "loss 1: repaint the frame",
        "joint-angle readout",
        "loss 2: make the guesses agree",
    ]
    colors = [TEAL, GOLD, CLAY]

    for ax, col, lab, c in zip(axes, history.T, labels, colors):
        ax.plot(col, lw=1.6, color=c)
        ax.set_title(lab, fontsize=11)
        ax.set_xlabel("training step")

    fig.suptitle(
        "Both losses falling together: the model is getting sharper "
        "AND more predictable",
        y=1.04,
    )
    if save:
        _savefig(fig, "03_training_curves.png")
    plt.close(fig)


# ── Linear probe ───────────────────────────────────────────────────────────────

def plot_probe(r2: np.ndarray, r2_ctrl: np.ndarray, save: bool = True):
    """Horizontal bar chart of per-joint R^2 vs shuffled control."""
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    yy = np.arange(6)
    ax.barh(yy, r2, color=TEAL, height=0.55, label="from the memory h alone")
    ax.barh(yy, r2_ctrl, color=CLAY, height=0.22, label="shuffled control")
    ax.set_yticks(yy)
    ax.set_yticklabels(JOINTS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("fraction of the joint angle recovered by a LINEAR readout")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    if save:
        _savefig(fig, "04_linear_probe.png")
    plt.close(fig)
    print("per-joint R^2:", np.round(r2, 4))


# ── Dream frames ───────────────────────────────────────────────────────────────

def plot_dream_frames(
    real: np.ndarray,
    dreamed: np.ndarray,
    context: int = CONTEXT,
    save: bool = True,
):
    """Side-by-side real vs dreamed frames at selected timesteps."""
    picks = [0, 11, 23, 35, 47, 59]
    fig, axes = plt.subplots(2, len(picks), figsize=(13, 4.6))

    for i, k in enumerate(picks):
        axes[0, i].imshow(real[context + k])
        axes[0, i].axis("off")
        axes[0, i].set_title(f"+{k+1} steps", fontsize=10, color=MUTED)
        axes[1, i].imshow(dreamed[k])
        axes[1, i].axis("off")

    axes[0, 0].text(
        -0.12, 0.5, "the real robot",
        transform=axes[0, 0].transAxes,
        ha="right", va="center", fontsize=12, color=TEAL, fontweight="bold",
    )
    axes[1, 0].text(
        -0.12, 0.5, "imagined\n(camera off)",
        transform=axes[1, 0].transAxes,
        ha="right", va="center", fontsize=12, color=CLAY, fontweight="bold",
    )
    fig.suptitle(
        "Sixty steps of pure imagination on a held-out episode", y=1.02,
    )
    plt.subplots_adjust(left=0.13, wspace=0.05, hspace=0.08)
    if save:
        _savefig(fig, "05_dream_frames.png")
    plt.close(fig)


def plot_dream_joints(
    dreamed_joints: np.ndarray,
    real_states: np.ndarray,
    context: int = CONTEXT,
    horizon: int = HORIZON,
    save: bool = True,
):
    """Plot dreamed vs real joint angles over the dream horizon."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.4))
    for j in range(6):
        ax = axes[j // 3, j % 3]
        ax.plot(
            real_states[context : context + horizon, j],
            lw=2.4, color=TEAL, label="real",
        )
        ax.plot(dreamed_joints[:, j], lw=2.4, ls="--", color=CLAY, label="dreamed")
        ax.set_title(JOINTS[j], fontsize=11)
        if j >= 3:
            ax.set_xlabel("dream step")

    axes[0, 0].legend(frameon=False, fontsize=10)
    fig.suptitle(
        "Dreamed joint angles vs the real robot: 60 open-loop steps", y=1.01,
    )
    plt.tight_layout()
    if save:
        _savefig(fig, "06_dream_joints.png")
    plt.close(fig)

    err = ((dreamed_joints - real_states[context : context + horizon]) ** 2).mean(-1)
    print(
        f"joint error at step 1: {err[0]:.4f}   "
        f"step 30: {err[29]:.4f}   "
        f"step 60: {err[59]:.4f}"
    )


# ── Ablation ───────────────────────────────────────────────────────────────────

def plot_ablation(
    real_frames: np.ndarray,
    frozen_frames: list,
    starved_frames: list,
    both_frames: list,
    context: int = CONTEXT,
    save: bool = True,
):
    """4-row comparison: real / s frozen / s starved / full RSSM."""
    picks = [0, 8, 16, 24, 32, 39]
    rows = [
        ("the real robot", None, INK),
        ("s frozen", frozen_frames, TEAL),
        ("s starved from h", starved_frames, GOLD),
        ("both, as designed", both_frames, CLAY),
    ]
    fig, axes = plt.subplots(4, len(picks), figsize=(13, 8.4))

    for r, (lab, frames, col) in enumerate(rows):
        for i, k in enumerate(picks):
            if frames is None:
                img = real_frames[context + k]
            else:
                img = frames[k]
            axes[r, i].imshow(np.clip(img, 0, 1))
            axes[r, i].axis("off")
            if r == 0:
                axes[r, i].set_title(f"+{k+1}", fontsize=10, color=MUTED)
        axes[r, 0].text(
            -0.12, 0.5, lab,
            transform=axes[r, 0].transAxes,
            ha="right", va="center", fontsize=12,
            fontweight="bold", color=col,
        )

    fig.suptitle(
        "One trained model, three wirings: "
        "the two paths are load-bearing for each other",
        y=1.01,
    )
    plt.subplots_adjust(left=0.15, wspace=0.05, hspace=0.08)
    if save:
        _savefig(fig, "07_ablation.png")
    plt.close(fig)


# ── Uncertainty ────────────────────────────────────────────────────────────────

def plot_uncertainty(sigmas: list, save: bool = True):
    """Plot posterior uncertainty (mean std) over time."""
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.plot(sigmas, lw=2, color=CLAY)
    ax.set_xlabel("timestep")
    ax.set_ylabel("uncertainty (mean width)")
    ax.set_title("Maximum uncertainty at t=0: one frame collapses it")
    plt.tight_layout()
    if save:
        _savefig(fig, "08_uncertainty.png")
    plt.close(fig)
