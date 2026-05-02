"""
Phase portraits across the three ablation models.

For a chosen position and trajectory variant, plot the (a - a_0,
grad_a - grad_a_0) phase portrait of selected units in each of the
three models, side by side. This is the natural visual companion to
compare_ablations.py's distribution plots.

Two views are produced:

  single_sample_portraits.png
      Trajectories on a single chosen input pair (--sample-idx).
      Sensitive to per-input noise, but reveals input-dependent
      behavior.

  mean_trajectory_portraits.png
      Trajectories averaged across all 256 input pairs.
      Isolates the consistent (input-independent) component of the
      trajectory. If the mean is close to a clean spiral, the
      rotational structure is robust across inputs; if it collapses
      to nothing, the per-unit dynamics are mostly input-driven.

Default unit selection picks units spanning the |R| range of a chosen
reference model (default: full). Pass --units to override.

Usage:
  python phase_portraits_across_models.py \
      --root ablation_run \
      --out-dir ablation_compare/portraits_pos5_cum \
      --position 5 \
      --variant cumulative

  python phase_portraits_across_models.py \
      --variant attn_delta \
      --units 0 5 12 23 47 63
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from toy_transformer_addition import capture_residual_streams
from train_ablations import AblatedTransformer, AblatedConfig
from analyze_toy_residuals import rotation_count_per_unit


ALL_MODES = ["full", "attn_only", "mlp_only"]
MODE_COLORS = {"full": "C0", "attn_only": "C1", "mlp_only": "C2"}
VARIANTS = ["cumulative", "combined_delta", "attn_delta", "mlp_delta"]


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = AblatedConfig(**ckpt["config"])
    model = AblatedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def collect_streams_at_position(model, position, batch_size=64):
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]
    _, captures = capture_residual_streams(
        model, pairs, device, batch_size=batch_size
    )
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    at_pos = stacked[:, :, position, :]
    return (at_pos.permute(1, 0, 2).contiguous()
                  .numpy().astype(np.float32))


def select_variant(streams, variant):
    diffs = np.diff(streams, axis=1)
    if variant == "cumulative":
        return streams
    if variant == "combined_delta":
        return diffs
    if variant == "attn_delta":
        return diffs[:, ::2, :]
    if variant == "mlp_delta":
        return diffs[:, 1::2, :]
    raise ValueError(f"unknown variant {variant}")


def is_meaningful(streams_v, eps=1e-8):
    return np.any(np.abs(streams_v) > eps)


def select_units_default(R_ref, n_units):
    """Pick n_units evenly spaced through the |R| range."""
    order = np.argsort(np.abs(R_ref))
    n = len(order)
    idxs = np.linspace(0, n - 1, n_units).round().astype(int)
    return [int(order[i]) for i in idxs]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def draw_portrait(ax, traj, mode_color, n_steps_label_unit=False):
    """Draw a single phase portrait into the given axis.

    traj: 1D array of length n_steps (the trace of one unit across depth).
    """
    n_steps = len(traj)
    grad = np.gradient(traj)
    x = traj - traj[0]
    y = grad - grad[0]
    ax.plot(x, y, "-", linewidth=0.7, alpha=0.9, color=mode_color)
    ax.scatter(x, y, c=np.arange(n_steps), cmap="viridis", s=12)
    ax.scatter([0], [0], marker="x", color="red", s=30, zorder=5)
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.axvline(0, color="gray", linewidth=0.4)


def plot_grid(
    streams_v_by_mode,
    R_by_mode,
    units,
    use_mean,
    sample_idx,
    variant,
    position,
    save_path,
):
    """Rows = units, columns = models. Each panel is the phase portrait
    of that unit in that model."""
    n_units = len(units)
    fig, axes = plt.subplots(
        n_units, len(ALL_MODES),
        figsize=(3.6 * len(ALL_MODES), 2.8 * n_units),
        squeeze=False,
    )
    for col, mode in enumerate(ALL_MODES):
        s_v = streams_v_by_mode.get(mode)
        for row, u in enumerate(units):
            ax = axes[row, col]
            if s_v is None or not is_meaningful(s_v):
                ax.set_facecolor("#eeeeee")
                ax.text(
                    0.5, 0.5,
                    "trivial / empty"
                    if s_v is not None
                    else "no checkpoint",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                if use_mean:
                    traj = s_v[:, :, u].mean(axis=0)
                else:
                    traj = s_v[sample_idx, :, u]
                draw_portrait(ax, traj, MODE_COLORS[mode])
                ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(mode, fontsize=11)
            if col == 0:
                # Y-label: unit index plus R for each model that has it
                R_strs = []
                for m in ALL_MODES:
                    if R_by_mode.get(m) is not None:
                        R_strs.append(f"{R_by_mode[m][u]:.2f}")
                    else:
                        R_strs.append("--")
                ax.set_ylabel(
                    f"unit {u}\nR = [{', '.join(R_strs)}]",
                    fontsize=9,
                )
            if row == n_units - 1:
                ax.set_xlabel("a - a_0", fontsize=8)

    title_kind = "mean trajectory" if use_mean else f"sample idx {sample_idx}"
    fig.suptitle(
        f"Phase portraits across ablation models "
        f"(variant={variant}, position={position}, {title_kind})\n"
        f"R = [full, attn_only, mlp_only]",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="ablation_run")
    parser.add_argument("--out-dir", type=str,
                        default="ablation_compare/portraits")
    parser.add_argument("--position", type=int, default=5)
    parser.add_argument("--variant", type=str, default="cumulative",
                        choices=VARIANTS)
    parser.add_argument("--n-units", type=int, default=12,
                        help="Number of units to plot (if --units not given). "
                             "Spans the |R| range of the reference model.")
    parser.add_argument("--units", type=int, nargs="*", default=None,
                        help="Explicit unit indices to plot")
    parser.add_argument("--ref-model", type=str, default="full",
                        choices=ALL_MODES,
                        help="Which model's |R| ordering to use for default "
                             "unit selection")
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    streams_v_by_mode = {}
    R_by_mode = {}
    for mode in ALL_MODES:
        ckpt = root / mode / "model_trained.pt"
        if not ckpt.exists():
            print(f"[{mode}] no checkpoint at {ckpt}; will leave panels empty")
            streams_v_by_mode[mode] = None
            R_by_mode[mode] = None
            continue
        print(f"[{mode}] loading and capturing ...")
        model, _ = load_checkpoint(ckpt)
        model = model.to(device)
        streams = collect_streams_at_position(model, args.position)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        streams_v = select_variant(streams, args.variant)
        streams_v_by_mode[mode] = streams_v
        if is_meaningful(streams_v):
            R_by_mode[mode] = rotation_count_per_unit(streams_v)
        else:
            R_by_mode[mode] = np.zeros(streams_v.shape[2])
        print(f"  {args.variant} shape: {streams_v.shape}, "
              f"meaningful: {is_meaningful(streams_v)}")

    # Pick units
    if args.units:
        units = list(args.units)
        print(f"\nUsing user-specified units: {units}")
    else:
        ref_R = R_by_mode.get(args.ref_model)
        if ref_R is None:
            # Fall back to any model that has data
            for m in ALL_MODES:
                if R_by_mode.get(m) is not None:
                    ref_R = R_by_mode[m]
                    print(f"Reference model {args.ref_model} unavailable, "
                          f"using {m} instead.")
                    break
        if ref_R is None:
            raise SystemExit("No checkpoints available; nothing to plot.")
        units = select_units_default(ref_R, args.n_units)
        print(f"\nDefault unit selection (spanning |R| in ref={args.ref_model}, "
              f"n={args.n_units}): {units}")
        for u in units:
            R_strs = []
            for m in ALL_MODES:
                v = R_by_mode.get(m)
                R_strs.append(f"{v[u]:+.2f}" if v is not None else "--")
            print(f"  unit {u}: R = [{', '.join(R_strs)}]")

    print("\nMaking single-sample grid ...")
    plot_grid(
        streams_v_by_mode, R_by_mode, units,
        use_mean=False, sample_idx=args.sample_idx,
        variant=args.variant, position=args.position,
        save_path=out_dir / "single_sample_portraits.png",
    )

    print("Making mean-trajectory grid ...")
    plot_grid(
        streams_v_by_mode, R_by_mode, units,
        use_mean=True, sample_idx=args.sample_idx,
        variant=args.variant, position=args.position,
        save_path=out_dir / "mean_trajectory_portraits.png",
    )

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
