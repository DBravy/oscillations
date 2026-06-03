"""
Cross-checkpoint phase portrait comparison
==========================================

Plots phase portraits for the SAME residual stream units across
multiple OLMo checkpoint files, so unit-by-unit evolution is visible.
Works on the trajectories.npz / rotations.npz produced by
phase_portraits_olmo_bloom.py or phase_portraits_olmo_checkpoints.py.

The trajectories file stores mean_trajectory of shape (2L, D), so any
unit's phase portrait can be re-rendered from the data without re-running
the model.

Inputs:
  Positional: one or more *_trajectories.npz files, in the order you want
  them displayed (left to right). For each one, the script tries to locate
  the matching *_rotations.npz by suffix swap; if not found, it just skips
  the R annotation but still plots.

Unit selection (mutually exclusive):
  (default)             use portrait_units from the FIRST checkpoint's
                        rotations file as the canonical unit set
  --units 12 480 ...    explicit unit indices
  --units_from PATH     use portrait_units from a specific rotations.npz
  --top_k_from PATH K   top-K abs(R) units from a specific rotations.npz
  --quartiles_from P    8 units spread along abs(R) quartiles of a
                        rotations.npz (same logic the original scripts use)

Layout: rows = units, columns = checkpoints. Within each row, axes share
limits across checkpoints by default so you can see whether the orbit
grows, shrinks, or changes shape with training. Pass --no_share_axes to
let each cell auto-scale.

Example:
    python compare_phase_portraits.py \\
        olmo_step100000_trajectories.npz olmo_trajectories.npz \\
        --labels "step 100k" "mature" \\
        --out compare_step100k_vs_mature.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_rotations_path(trajectories_path):
    """Locate the rotations.npz sibling of a trajectories.npz file."""
    if trajectories_path.endswith("_trajectories.npz"):
        candidate = trajectories_path.replace("_trajectories.npz",
                                              "_rotations.npz")
        if os.path.exists(candidate):
            return candidate
    return None


def load_checkpoint(trajectories_path, rotations_path=None):
    """
    Load mean_trajectory plus (optionally) rotations and metadata.
    Returns a dict with: mean_traj, T, D, model, step, rotations,
    portrait_units, path.
    """
    if rotations_path is None:
        rotations_path = find_rotations_path(trajectories_path)

    tdata = np.load(trajectories_path, allow_pickle=True)
    mean_traj = tdata["mean_trajectory"]
    T, D = mean_traj.shape

    model = str(tdata["model"]) if "model" in tdata.files else os.path.basename(trajectories_path)
    step = int(tdata["step"]) if "step" in tdata.files else None

    rotations = None
    portrait_units = None
    if rotations_path is not None and os.path.exists(rotations_path):
        rdata = np.load(rotations_path, allow_pickle=True)
        if "rotations" in rdata.files:
            rotations = rdata["rotations"]
        if "portrait_units" in rdata.files:
            portrait_units = np.asarray(rdata["portrait_units"]).astype(int)

    return {
        "path": trajectories_path,
        "rotations_path": rotations_path,
        "mean_traj": mean_traj.astype(np.float64),
        "T": T,
        "D": D,
        "model": model,
        "step": step,
        "rotations": rotations,
        "portrait_units": portrait_units,
    }


# ---------------------------------------------------------------------------
# Rotation count for a single unit (matches the originals)
# ---------------------------------------------------------------------------

def rotation_count_one_unit(trajectory_1d):
    """Replicates compute_rotations_vec for a single unit. Returns scalar."""
    a = np.asarray(trajectory_1d, dtype=np.float64).reshape(-1, 1)
    g = np.gradient(a, axis=0)
    x = a - a[0:1, :]
    y = g - g[0:1, :]
    dx = np.diff(x, axis=0)
    dy = np.diff(y, axis=0)
    theta = np.arctan2(dy, dx)
    dtheta = np.diff(theta, axis=0)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(dtheta) / (2 * np.pi))


# ---------------------------------------------------------------------------
# Unit selection
# ---------------------------------------------------------------------------

def pick_quartile_units(abs_rotations, n=8):
    """Match the original 8-unit quartile sampling scheme."""
    D = len(abs_rotations)
    order = np.argsort(abs_rotations)[::-1]
    return np.array([
        order[0], order[1],
        order[int(D * 0.25)], order[int(D * 0.25) + 1],
        order[int(D * 0.50)], order[int(D * 0.50) + 1],
        order[int(D * 0.75)], order[int(D * 0.75) + 1],
    ])


def resolve_units(args, checkpoints):
    """
    Apply the unit-selection flags to produce the final unit index list.
    """
    if args.units is not None:
        return np.array(args.units, dtype=int)

    if args.units_from is not None:
        ref = np.load(args.units_from, allow_pickle=True)
        if "portrait_units" not in ref.files:
            raise SystemExit(f"{args.units_from} has no portrait_units field")
        return np.asarray(ref["portrait_units"]).astype(int)

    if args.top_k_from is not None:
        path, k = args.top_k_from
        ref = np.load(path, allow_pickle=True)
        if "rotations" not in ref.files:
            raise SystemExit(f"{path} has no rotations field")
        abs_R = np.abs(ref["rotations"])
        return np.argsort(abs_R)[::-1][:int(k)]

    if args.quartiles_from is not None:
        ref = np.load(args.quartiles_from, allow_pickle=True)
        if "rotations" not in ref.files:
            raise SystemExit(f"{args.quartiles_from} has no rotations field")
        return pick_quartile_units(np.abs(ref["rotations"]), n=8)

    # Default: portrait_units from the first checkpoint's rotations file
    first = checkpoints[0]
    if first["portrait_units"] is None:
        raise SystemExit(
            f"No portrait_units found alongside {first['path']}. "
            f"Pass --units / --units_from / --top_k_from / --quartiles_from."
        )
    return first["portrait_units"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def compute_xy(mean_traj, unit):
    """Centered (activation, gradient) for a single unit."""
    a = mean_traj[:, unit]
    g = np.gradient(mean_traj, axis=0)[:, unit]
    return a - a[0], g - g[0]


def plot_grid(checkpoints, units, labels, share_axes, out_path, suptitle):
    """
    Rows = units, columns = checkpoints.
    Within each row, axes share limits across columns (when share_axes).
    """
    n_units = len(units)
    n_cps = len(checkpoints)
    cmap = plt.cm.viridis

    # Pre-compute xy per (unit, checkpoint) so we can set shared limits.
    xy = {}
    for i, u in enumerate(units):
        for j, cp in enumerate(checkpoints):
            if u >= cp["D"]:
                xy[(i, j)] = None
                continue
            xy[(i, j)] = compute_xy(cp["mean_traj"], int(u))

    # Per-row limits
    row_limits = []
    for i in range(n_units):
        m = 0.0
        for j in range(n_cps):
            if xy[(i, j)] is None:
                continue
            x, y = xy[(i, j)]
            m = max(m, float(np.abs(x).max()), float(np.abs(y).max()))
        row_limits.append(m * 1.1 + 1e-8)

    # T for the colorbar. Use the first checkpoint's T; warn if heterogeneous.
    T_ref = checkpoints[0]["T"]
    heterogeneous_T = any(cp["T"] != T_ref for cp in checkpoints)
    if heterogeneous_T:
        print("  WARNING: checkpoints have different T (effective layer counts); "
              "colorbar uses the first checkpoint's T.")

    fig_w = max(3.0 * n_cps + 1.0, 6.0)
    fig_h = max(2.9 * n_units, 3.0)
    fig, axes = plt.subplots(
        n_units, n_cps,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        squeeze=False,
    )

    for i, u in enumerate(units):
        for j, cp in enumerate(checkpoints):
            ax = axes[i, j]
            pair = xy[(i, j)]
            if pair is None:
                ax.text(0.5, 0.5, f"unit {u} out of range\n(D={cp['D']})",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="#888")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            x, y = pair
            T = cp["T"]
            points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
            segs = np.concatenate([points[:-1], points[1:]], axis=1)
            layer_idx = np.arange(T - 1)
            norm = plt.Normalize(vmin=0, vmax=T - 1)
            lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.3)
            lc.set_array(layer_idx)
            ax.add_collection(lc)
            ax.scatter([0], [0], s=18, c="black", marker="o", zorder=5)

            if share_axes:
                m = row_limits[i]
            else:
                m = max(float(np.abs(x).max()),
                        float(np.abs(y).max())) * 1.1 + 1e-8

            ax.set_xlim(-m, m)
            ax.set_ylim(-m, m)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.3)

            # R annotation for this unit at this checkpoint
            R_here = rotation_count_one_unit(cp["mean_traj"][:, int(u)])
            title = f"unit {int(u)}  R={R_here:.2f}"
            ax.set_title(title, fontsize=9)

            # Column header on the top row
            if i == 0:
                ax.annotate(
                    labels[j],
                    xy=(0.5, 1.22), xycoords="axes fraction",
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                )

            # Row label on the left column
            if j == 0:
                ax.set_ylabel(f"unit {int(u)}\ngradient (centered)",
                              fontsize=9)
            else:
                ax.set_ylabel("")

            if i == n_units - 1:
                ax.set_xlabel("activation (centered)", fontsize=9)

    # Shared colorbar on the right
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(0, T_ref - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6,
                 label="Effective layer")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Argument plumbing
# ---------------------------------------------------------------------------

def derive_label(cp, fallback):
    if cp["step"] is not None:
        return f"step {cp['step']}"
    return cp["model"] if cp["model"] else fallback


def main():
    p = argparse.ArgumentParser(
        description="Compare phase portraits across checkpoints, "
                    "using a single set of units across all of them.")
    p.add_argument("trajectory_files", nargs="+",
                   help="Trajectories npz files, in the order you want "
                        "them shown left-to-right.")
    p.add_argument("--rotations_files", nargs="+", default=None,
                   help="Optional explicit rotations npz paths "
                        "(same order, same count as trajectory_files). "
                        "If omitted, each is inferred by suffix swap.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Column labels (one per checkpoint). If omitted, "
                        "derived from the 'step' or 'model' field.")
    p.add_argument("--out", default="compare_phase_portraits.png",
                   help="Output PNG path.")
    p.add_argument("--suptitle", default=None,
                   help="Figure-level title (optional).")

    # Unit selection (mutually exclusive)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--units", nargs="+", type=int, default=None,
                     help="Explicit unit indices.")
    grp.add_argument("--units_from", default=None,
                     help="Path to a rotations.npz whose portrait_units "
                          "field provides the unit set.")
    grp.add_argument("--top_k_from", nargs=2, default=None,
                     metavar=("PATH", "K"),
                     help="Top-K |R| units from a rotations.npz.")
    grp.add_argument("--quartiles_from", default=None,
                     help="Pick 8 units along |R| quartiles of a "
                          "rotations.npz (same scheme as the originals).")

    p.add_argument("--no_share_axes", action="store_true",
                   help="Disable shared axis limits within a row. "
                        "Default is to share so growth is visible.")
    args = p.parse_args()

    # Validate rotations_files count if given
    if args.rotations_files is not None and \
       len(args.rotations_files) != len(args.trajectory_files):
        raise SystemExit("rotations_files must match trajectory_files in count")

    # Load all checkpoints
    print("Loading checkpoints:")
    checkpoints = []
    for i, tp in enumerate(args.trajectory_files):
        rp = args.rotations_files[i] if args.rotations_files else None
        cp = load_checkpoint(tp, rp)
        print(f"  [{i}] {tp}")
        print(f"      model={cp['model']!r}  step={cp['step']}  "
              f"T={cp['T']}  D={cp['D']}  "
              f"rotations={'yes' if cp['rotations'] is not None else 'no'}")
        checkpoints.append(cp)

    # Sanity: all D should match for unit-by-unit comparison to be meaningful
    Ds = [cp["D"] for cp in checkpoints]
    if len(set(Ds)) > 1:
        print(f"  WARNING: checkpoints have different D values: {Ds}. "
              f"Out-of-range units will be left blank.")

    # Resolve unit set
    units = resolve_units(args, checkpoints)
    print(f"Using {len(units)} units: {units.tolist()}")

    # Resolve labels
    if args.labels is None:
        labels = [derive_label(cp, f"ckpt {i}")
                  for i, cp in enumerate(checkpoints)]
    else:
        if len(args.labels) != len(checkpoints):
            raise SystemExit("labels must match number of checkpoints")
        labels = args.labels
    print(f"Labels: {labels}")

    # Plot
    plot_grid(
        checkpoints=checkpoints,
        units=units,
        labels=labels,
        share_axes=not args.no_share_axes,
        out_path=args.out,
        suptitle=args.suptitle,
    )

    print("Done.")


if __name__ == "__main__":
    main()
