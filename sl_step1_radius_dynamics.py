"""
Step 1 of the Stuart-Landau bootstrap analysis: characterize the
amplitude trajectory across depth and identify the explosion sublayer L*.

Two phase-plane conventions, computed side by side:

  PER-UNIT plane:
    For each unit u, define its own 2D phase plane
      (a_u(l) - a_u(0), grad_l a_u(l) - grad_l a_u(0))
    and r_u(l) = distance from origin in that plane.
    This is the convention the prior pipeline used.

  GLOBAL plane:
    Find the dominant rotating mode of the residual stream's depth
    trajectory. We do this with a complex-valued PCA on the analytic
    signal of the centered depth trajectory. Concretely, for each
    input we form
      Z(l) = stream(l) + i * Hilbert_l[stream(l)]
    a (n_sublayers, D) complex matrix, and take the top right singular
    vector w in C^D as the global mode. Each unit u's amplitude in this
    mode at sublayer l is |Z(l) . w| projected onto unit u, that is,
    |proj_u(Z(l)) onto direction w|. We instead use the simpler scalar:
      A_global(l) = |Z(l) . w| (one number per sublayer per input)
    and decompose the per-unit contribution to that mode using
      r_u^global(l) = |Z_u(l) * conj(w_u)|  (alignment of unit u
                                              with global mode at l)
    Lots of conventions are possible; this one preserves the role of
    the unit's own oscillation while referring to a shared plane.

For each convention, per position (5, 6, 7) and each input pair, compute
r_u(l) for every unit u and sublayer l. Aggregate:

  - Mean and stddev across units of r_u(l) at each l (for the run-of-
    the-mill bulk dynamics).
  - Mean and stddev across input pairs of r_u(l) for fixed u, l.
  - Second derivative across l of mean radius. The maximum is L*.
  - Per-input L* (peak second derivative on each input separately) to
    check the "remarkably consistent explosion point" claim.

Outputs in --out-dir:
  radius_trajectories_per_unit.png       per-unit-plane: bulk vs depth
  radius_trajectories_global.png         global-plane: bulk vs depth
  second_derivative_pos{5,6,7}.png       d2 mean_radius / dl2
  L_star_consistency.png                 distribution of per-input L*
                                         across input pairs (per position
                                         and convention)
  radius_step1_summary.json

Usage:
  python sl_step1_radius_dynamics.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_sl_step1_trained
  python sl_step1_radius_dynamics.py \
      --checkpoint toy_transformer_run/model_random_init.pt \
      --out-dir toy_sl_step1_random
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from toy_transformer_addition import (
    ToyTransformer,
    ModelConfig,
    capture_residual_streams,
)


POSITIONS = [5, 6, 7]
POSITION_LABELS = {
    5: 'pos 5 ("=")',
    6: "pos 6 (c2)",
    7: "pos 7 (c1)",
}
POSITION_COLORS = {5: "C0", 6: "C1", 7: "C2"}


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = ToyTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def collect_streams(model, batch_size=64):
    """One forward pass; return cumulative streams at each position."""
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]
    _, captures = capture_residual_streams(
        model, pairs, device, batch_size=batch_size
    )
    sublayer_meta = [(c["kind"], c["layer"]) for c in captures]
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    out = {}
    for pos in POSITIONS:
        s = stacked[:, :, pos, :].permute(1, 0, 2).contiguous()
        out[pos] = s.numpy().astype(np.float32)
    return out, sublayer_meta


# ---------------------------------------------------------------------------
# Per-unit phase-plane radius
# ---------------------------------------------------------------------------

def radius_per_unit_plane(streams):
    """
    streams: (N, L, D)
    Returns r: (N, L, D) where r[n, l, u] is the distance in unit u's
    own phase plane from the trajectory's depth-0 origin.
    """
    N, L, D = streams.shape
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    return np.sqrt(x ** 2 + y ** 2)


# ---------------------------------------------------------------------------
# Global rotating mode
# ---------------------------------------------------------------------------

def global_mode_amplitudes(streams):
    """
    streams: (N, L, D)

    Method: For each input n, build the analytic signal along the depth
    axis with the discrete Hilbert transform. This yields a complex
    matrix Z[n, l, u] whose magnitude is the amplitude envelope of the
    depth oscillation and whose phase tracks the rotation angle. We
    then find the global rotating-mode direction w by SVD on the
    sample-flattened matrix Z.reshape(N*L, D), taking the top right
    singular vector as a complex unit vector in C^D.

    Returns:
      Z:                  (N, L, D) complex
      w:                  (D,) complex unit vector
      r_global_per_unit:  (N, L, D) per-unit alignment magnitude with
                          the global mode, |Z[n, l, u] * conj(w[u])|
      A_global:           (N, L) projected scalar amplitude
                          | sum_u Z[n, l, u] * conj(w[u]) |
    """
    N, L, D = streams.shape
    centered = streams - streams.mean(axis=1, keepdims=True)
    Z = hilbert(centered, axis=1)               # (N, L, D), complex

    flat = Z.reshape(N * L, D)
    # Top right singular vector via SVD on the complex matrix.
    # We only need the leading right singular vector. Use the smaller
    # operator: D x D Gram matrix flat^H flat.
    G = flat.conj().T @ flat                    # (D, D), Hermitian
    eigvals, eigvecs = np.linalg.eigh(G)
    w = eigvecs[:, -1]                          # leading eigenvector
    w = w / (np.linalg.norm(w) + 1e-12)

    r_global_per_unit = np.abs(Z * np.conj(w[None, None, :]))
    A_global = np.abs(Z @ np.conj(w))           # (N, L)

    return Z, w, r_global_per_unit, A_global


# ---------------------------------------------------------------------------
# Identify L*: layer of peak second derivative of mean radius
# ---------------------------------------------------------------------------

def identify_L_star(mean_r):
    """
    mean_r: (L,) mean radius across units (and possibly inputs).
    Returns:
      L_star, second_deriv (length L array using np.gradient twice),
      peak_value
    """
    d1 = np.gradient(mean_r)
    d2 = np.gradient(d1)
    L_star = int(np.argmax(d2))
    return L_star, d2, float(d2[L_star])


def per_input_L_star(r_NL):
    """
    r_NL: (N, L) per-input radius (already aggregated across units).
    Returns array of length N with each input's argmax of d2.
    """
    L_stars = np.zeros(r_NL.shape[0], dtype=int)
    for n in range(r_NL.shape[0]):
        d1 = np.gradient(r_NL[n])
        d2 = np.gradient(d1)
        L_stars[n] = int(np.argmax(d2))
    return L_stars


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_radius_trajectories(r_by_pos, mode_name, save_path):
    """Three panels (one per position), each showing:
       mean over units and inputs (thick line), shaded inter-quartile
       across inputs of the mean-over-units, and faint per-unit means."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, pos in zip(axes, POSITIONS):
        r = r_by_pos[pos]                   # (N, L, D)
        N, L, D = r.shape
        # Per-unit mean across inputs: (L, D)
        per_unit_mean = r.mean(axis=0)
        for u in range(D):
            ax.plot(
                np.arange(L), per_unit_mean[:, u],
                color=POSITION_COLORS[pos], alpha=0.07, linewidth=0.5,
            )
        # Mean over both units and inputs
        global_mean = r.mean(axis=(0, 2))
        # Per-input mean over units, then percentiles across inputs
        per_input_unit_mean = r.mean(axis=2)   # (N, L)
        q25 = np.percentile(per_input_unit_mean, 25, axis=0)
        q75 = np.percentile(per_input_unit_mean, 75, axis=0)
        ax.fill_between(
            np.arange(L), q25, q75,
            color=POSITION_COLORS[pos], alpha=0.25,
            label="IQR across inputs of mean-over-units",
        )
        ax.plot(
            np.arange(L), global_mean,
            color=POSITION_COLORS[pos], linewidth=2.4,
            label="mean over inputs and units",
        )
        L_star, _, peak = identify_L_star(global_mean)
        ax.axvline(
            L_star, color="red", linewidth=1.0, linestyle="--",
            label=f"L* = {L_star}",
        )
        ax.set_xlabel("sublayer index l")
        ax.set_ylabel(f"radius r ({mode_name})")
        ax.set_title(POSITION_LABELS[pos], fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"Radius dynamics across depth, "
        f"{mode_name} convention",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_second_derivatives(r_by_pos_per_unit, r_by_pos_global,
                            save_path):
    """One panel per position. Plots both d2 mean_r curves
    (per-unit-plane and global-plane). Marks L* on each."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, pos in zip(axes, POSITIONS):
        for r_by_pos, name, ls in [
            (r_by_pos_per_unit, "per-unit",  "-"),
            (r_by_pos_global,   "global",    "--"),
        ]:
            r = r_by_pos[pos]                # (N, L, D)
            mean_r = r.mean(axis=(0, 2))
            L_star, d2, peak = identify_L_star(mean_r)
            ax.plot(
                np.arange(len(d2)), d2, ls, linewidth=1.6,
                label=f"{name}: L*={L_star}",
            )
            ax.scatter(
                [L_star], [peak], s=60, marker="o",
                edgecolors="red", facecolors="none", zorder=5,
            )
        ax.axhline(0, color="gray", linewidth=0.4, linestyle=":")
        ax.set_xlabel("sublayer l")
        ax.set_ylabel("d2 mean_r / dl2")
        ax.set_title(POSITION_LABELS[pos], fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Second derivative of mean radius across depth",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_L_star_consistency(L_stars_per_unit, L_stars_global, save_path):
    """One panel per position; histogram of per-input L* values for
    both conventions. Tests the 'consistent explosion point' claim."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, pos in zip(axes, POSITIONS):
        for L_arr, name, color in [
            (L_stars_per_unit[pos], "per-unit", "C0"),
            (L_stars_global[pos],   "global",   "C1"),
        ]:
            ax.hist(
                L_arr, bins=np.arange(-0.5, L_arr.max() + 1.5),
                density=False, histtype="step", linewidth=1.6,
                color=color,
                label=(f"{name}: median={int(np.median(L_arr))}, "
                       f"std={np.std(L_arr):.2f}"),
            )
        ax.set_xlabel("per-input L* (argmax d2 mean_r)")
        ax.set_ylabel("count over input pairs")
        ax.set_title(POSITION_LABELS[pos], fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Per-input L* consistency across input pairs",
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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="toy_sl_step1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}")

    print("\nCollecting cumulative residual streams ...")
    streams_by_pos, sublayer_meta = collect_streams(model)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------- per-unit-plane radius -------------
    print("\nComputing per-unit-plane radii ...")
    r_per_unit_by_pos = {
        pos: radius_per_unit_plane(streams_by_pos[pos])
        for pos in POSITIONS
    }

    # ------------- global-plane radius -------------
    print("Computing global rotating-mode radii ...")
    r_global_by_pos = {}
    A_global_by_pos = {}
    w_by_pos = {}
    for pos in POSITIONS:
        Z, w, r_global, A_global = global_mode_amplitudes(
            streams_by_pos[pos]
        )
        r_global_by_pos[pos] = r_global
        A_global_by_pos[pos] = A_global
        w_by_pos[pos] = w
        print(f"  pos {pos}: |w|={np.linalg.norm(w):.3f}  "
              f"top-3 |w_u| = "
              f"{np.sort(np.abs(w))[-3:][::-1]}")

    # ------------- L* identification -------------
    print("\nIdentifying L* ...")
    L_stars_per_unit = {}
    L_stars_global = {}
    summary_per_pos = {}
    for pos in POSITIONS:
        # Aggregate-over-everything L*
        mean_pu = r_per_unit_by_pos[pos].mean(axis=(0, 2))
        L_pu, d2_pu, peak_pu = identify_L_star(mean_pu)
        mean_gl = r_global_by_pos[pos].mean(axis=(0, 2))
        L_gl, d2_gl, peak_gl = identify_L_star(mean_gl)
        # Per-input L* (uses mean over units within each input)
        per_input_pu = r_per_unit_by_pos[pos].mean(axis=2)
        per_input_gl = r_global_by_pos[pos].mean(axis=2)
        L_stars_per_unit[pos] = per_input_L_star(per_input_pu)
        L_stars_global[pos] = per_input_L_star(per_input_gl)

        summary_per_pos[str(pos)] = {
            "label":  POSITION_LABELS[pos],
            "per_unit_plane": {
                "L_star_aggregate":   int(L_pu),
                "peak_d2":            float(peak_pu),
                "L_star_per_input_median": float(
                    np.median(L_stars_per_unit[pos])
                ),
                "L_star_per_input_std":    float(
                    np.std(L_stars_per_unit[pos])
                ),
                "mean_radius_at_l0":  float(mean_pu[0]),
                "mean_radius_at_lminus1": float(mean_pu[-1]),
                "growth_factor":      float(
                    mean_pu[-1] / (mean_pu[1] + 1e-12)
                ),
            },
            "global_plane": {
                "L_star_aggregate":   int(L_gl),
                "peak_d2":            float(peak_gl),
                "L_star_per_input_median": float(
                    np.median(L_stars_global[pos])
                ),
                "L_star_per_input_std":    float(
                    np.std(L_stars_global[pos])
                ),
                "mean_radius_at_l0":  float(mean_gl[0]),
                "mean_radius_at_lminus1": float(mean_gl[-1]),
                "growth_factor":      float(
                    mean_gl[-1] / (mean_gl[1] + 1e-12)
                ),
                "top_w_unit_idx_top5": [
                    int(i) for i in np.argsort(-np.abs(w_by_pos[pos]))[:5]
                ],
                "top_w_unit_mag_top5": [
                    float(np.abs(w_by_pos[pos])[i])
                    for i in np.argsort(-np.abs(w_by_pos[pos]))[:5]
                ],
            },
        }

    # ------------- Plots -------------
    print("\nMaking plots ...")
    plot_radius_trajectories(
        r_per_unit_by_pos, "per-unit",
        out_dir / "radius_trajectories_per_unit.png",
    )
    plot_radius_trajectories(
        r_global_by_pos, "global mode",
        out_dir / "radius_trajectories_global.png",
    )
    plot_second_derivatives(
        r_per_unit_by_pos, r_global_by_pos,
        out_dir / "second_derivative_comparison.png",
    )
    plot_L_star_consistency(
        L_stars_per_unit, L_stars_global,
        out_dir / "L_star_consistency.png",
    )

    # ------------- Save derived data for step 2 -------------
    np.savez(
        out_dir / "radius_data.npz",
        **{
            f"streams_pos{pos}":   streams_by_pos[pos]
            for pos in POSITIONS
        },
        **{
            f"r_per_unit_pos{pos}": r_per_unit_by_pos[pos]
            for pos in POSITIONS
        },
        **{
            f"r_global_pos{pos}":   r_global_by_pos[pos]
            for pos in POSITIONS
        },
        **{
            f"A_global_pos{pos}":   A_global_by_pos[pos]
            for pos in POSITIONS
        },
        **{
            f"w_pos{pos}":          w_by_pos[pos]
            for pos in POSITIONS
        },
        **{
            f"L_stars_per_unit_pos{pos}": L_stars_per_unit[pos]
            for pos in POSITIONS
        },
        **{
            f"L_stars_global_pos{pos}":   L_stars_global[pos]
            for pos in POSITIONS
        },
    )

    summary = {
        "checkpoint":     str(args.checkpoint),
        "n_sublayers":    int(streams_by_pos[5].shape[1]),
        "d_model":        int(streams_by_pos[5].shape[2]),
        "n_input_pairs":  int(streams_by_pos[5].shape[0]),
        "sublayer_meta":  [
            {"kind": k, "layer": l} for (k, l) in sublayer_meta
        ],
        "per_position":   summary_per_pos,
    }
    with open(out_dir / "radius_step1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Step 1 summary ---")
    print(f"\nL* (aggregate, peak of d2 mean_r over inputs+units):")
    print(f"{'position':<14} {'per-unit L*':>14} {'global L*':>14}")
    for pos in POSITIONS:
        s = summary["per_position"][str(pos)]
        print(f"{POSITION_LABELS[pos]:<14} "
              f"{s['per_unit_plane']['L_star_aggregate']:>14d} "
              f"{s['global_plane']['L_star_aggregate']:>14d}")

    print(f"\nL* per-input variability "
          f"(consistency across input pairs):")
    print(f"{'position':<14} "
          f"{'pu median':>10} {'pu std':>8} "
          f"{'gl median':>10} {'gl std':>8}")
    for pos in POSITIONS:
        s = summary["per_position"][str(pos)]
        print(
            f"{POSITION_LABELS[pos]:<14} "
            f"{s['per_unit_plane']['L_star_per_input_median']:>10.1f} "
            f"{s['per_unit_plane']['L_star_per_input_std']:>8.2f} "
            f"{s['global_plane']['L_star_per_input_median']:>10.1f} "
            f"{s['global_plane']['L_star_per_input_std']:>8.2f}"
        )

    print(f"\nRadius growth factor (mean_r[-1] / mean_r[1]):")
    print(f"{'position':<14} {'per-unit':>10} {'global':>10}")
    for pos in POSITIONS:
        s = summary["per_position"][str(pos)]
        print(f"{POSITION_LABELS[pos]:<14} "
              f"{s['per_unit_plane']['growth_factor']:>10.2f} "
              f"{s['global_plane']['growth_factor']:>10.2f}")

    print(f"\nOutputs in {out_dir.resolve()}")
    print("  radius_trajectories_per_unit.png")
    print("  radius_trajectories_global.png")
    print("  second_derivative_comparison.png")
    print("  L_star_consistency.png")
    print("  radius_data.npz   (used by step 2)")
    print("  radius_step1_summary.json")


if __name__ == "__main__":
    main()
