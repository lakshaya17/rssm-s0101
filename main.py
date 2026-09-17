#!/usr/bin/env python3
"""
Dreams That Last -- building an RSSM on a real robot (SO-101).
Local runner for Lecture 4 of "Build a World Model from Scratch" (Vizuara AI).

Usage
-----
  python main.py                    # run everything (skip training on CPU)
  python main.py --train            # force training even on CPU
  python main.py --steps 2000       # custom step count
  python main.py --skip-plots       # numbers only, no matplotlib windows
"""

import argparse
import os

import numpy as np
import torch

from rssm.config import (
    SEED, DEVICE, CONTEXT, HORIZON, HOLD_OUT, H, S,
    CHECKPOINT_DIR, TRAIN_STEPS, apply_plot_style,
    run_output_dir,
)
from rssm.data import download_data, SO101Dataset
from rssm.model import RSSM
from rssm.train import train
from rssm.evaluate import (
    dream, linear_probe, dream_variant, uncertainty_over_time,
)
from rssm.visualize import (
    set_output_dir,
    plot_one_row, plot_gripper_fork, plot_training_curves,
    plot_probe, plot_dream_frames, plot_dream_joints,
    plot_ablation, plot_uncertainty,
)


def main():
    parser = argparse.ArgumentParser(description="RSSM SO-101 local runner")
    parser.add_argument("--train", action="store_true",
                        help="force training even without a GPU")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS,
                        help="training steps (default: 600)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="skip all matplotlib visualizations")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="skip loading the pretrained checkpoint (use your own trained model)")
    parser.add_argument("--tag", type=str, default=None,
                        help="extra label for the output folder (e.g. 'koopman', 'ablation')")
    args = parser.parse_args()

    # ── Seed ───────────────────────────────────────────────────────────────
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"device: {DEVICE}")

    # ── Set up run-specific output directory ───────────────────────────────
    steps_label = args.steps if args.no_pretrained or args.train else None
    out_dir = run_output_dir(h=H, s=S, steps=steps_label, tag=args.tag)
    set_output_dir(out_dir)
    print(f"outputs -> {out_dir}")

    if not args.skip_plots:
        apply_plot_style()

    # ── Download data ──────────────────────────────────────────────────────
    print("\n=== Setup: downloading data & checkpoint ===")
    download_data()

    dataset = SO101Dataset()
    print(dataset.summary())

    # ── Part 1: Inspect the data ───────────────────────────────────────────
    if not args.skip_plots:
        print("\n=== Part 1: What the model tracks ===")
        ep = dataset.episodes[-3]
        plot_one_row(ep, t=60)
        plot_gripper_fork(ep)

    # ── Part 2: Build and optionally train ─────────────────────────────────
    print("\n=== Part 2: The machine (RSSM) ===")
    model = RSSM().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RSSM parameters: {n_params:,}")

    run_training = args.train or args.no_pretrained or (DEVICE.type in ("cuda", "mps"))

    if run_training:
        print(f"\nTraining for {args.steps} steps on {DEVICE} ...")
        hist = train(model, dataset, steps=args.steps, device=DEVICE)
        if not args.skip_plots:
            plot_training_curves(hist)
        # save checkpoint into the run-specific output dir
        ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"saved checkpoint to {ckpt_path}")
    else:
        print(
            f"\nNo GPU/MPS detected. Skipping training; "
            f"loading pretrained checkpoint..."
        )

    # ── Load the fully-trained checkpoint ──────────────────────────────────
    if not args.no_pretrained:
        print("\n=== Loading fully-trained checkpoint ===")
        ckpt_path = os.path.join(CHECKPOINT_DIR, "rssm_so101.pt")
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = RSSM().to(DEVICE)
        model.load_state_dict(ck)
        model.eval()
        print("loaded the trained RSSM")
    else:
        print("\n=== Skipping pretrained checkpoint, using locally trained model ===")
        model.eval()

    # ── Test 1: Linear probe ───────────────────────────────────────────────
    print("\n=== Test 1: What is the belt actually carrying? ===")
    r2, r2_ctrl = linear_probe(model, dataset, device=DEVICE)
    if not args.skip_plots:
        plot_probe(r2, r2_ctrl)

    # ── Test 2: Dream with camera off ──────────────────────────────────────
    print("\n=== Test 2: The dream (60 steps, camera off) ===")
    ep = dataset.episodes[-2]
    start = max(0, len(ep["frames"]) // 3 - CONTEXT)
    real, dreamed, joints, st = dream(
        model, ep, dataset, start, device=DEVICE,
    )
    if not args.skip_plots:
        plot_dream_frames(real, dreamed)
        plot_dream_joints(joints, st)

    # ── Test 3: Ablation ───────────────────────────────────────────────────
    print("\n=== Test 3: Cut either path and watch it break ===")
    real40, both_out = dream_variant(
        model, ep, dataset, start, mode="both", device=DEVICE,
    )
    _, frozen_out = dream_variant(
        model, ep, dataset, start, mode="freeze_s", device=DEVICE,
    )
    _, starved_out = dream_variant(
        model, ep, dataset, start, mode="starve_h", device=DEVICE,
    )
    if not args.skip_plots:
        plot_ablation(real40, frozen_out, starved_out, both_out)

    # ── Uncertainty over time ──────────────────────────────────────────────
    print("\n=== Uncertainty over time ===")
    sigmas = uncertainty_over_time(
        model, dataset.episodes[-1], dataset, device=DEVICE,
    )
    if not args.skip_plots:
        plot_uncertainty(sigmas)

    print(f"\n=== Done. All outputs saved to {out_dir} ===")


if __name__ == "__main__":
    main()
