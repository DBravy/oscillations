"""
Layer-by-layer Fiedler overlap analysis (v2).

Same coupling and Fiedler-vector construction as v1, but:

  - Layer 0's coupling matrix is degenerate (z=0 by construction), so
    its Fiedler vector is not meaningful. v1 set V = I at layer 0 which
    produced an artificial overlap of 1.0 there. v2 EXCLUDES layer 0
    from the analysis entirely.

  - Saturation layer l* is computed against a baseline at sublayer
    --start (default 1), not at layer 0.

  - Multiple thresholds (50%, 75%, 90%, 95%) are reported so the curve
    shape is visible, not just a single point.

  - Layer 0 is shown blank in the Fiedler heatmaps and omitted from the
    overlap line plots.

Usage:
  python sl_step3b_fiedler_overlap.py \
      --radius-data toy_sl_step1_trained/radius_data.npz \
      --out-dir toy_sl_step3b_trained
  python sl_step3b_fiedler_overlap.py \
      --radius-data toy_sl_step1_random/radius_data.npz \
      --out-dir toy_sl_step3b_random \
      --start 2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


POSITIONS = [5, 6, 7]
POSITION_LABELS = {
    5: 'pos 5 ("=")',
    6: "pos 6 (c2)",
    7: "pos 7 (c1)",
}
POSITION_COLORS = {5: "C0", 6: "C1", 7: "C2"}

PAIR_LABELS = {
    (5, 6): "pos 5 vs pos 6",
    (5, 7): "pos 5 vs pos 7",
    (6, 7): "pos 6 vs pos 7",
}
PAIR_COLORS = {
    (5, 6): "tab:purple",
    (5, 7): "tab:cyan",
    (6, 7): "tab:olive",
}

THRESHOLDS = [0.50, 0.75, 0.90, 0.95]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_step1(npz_path):
    z = np.load(npz_path)
    return {pos: z[f"streams_pos{pos}"] for pos in POSITIONS}


def per_unit_z(streams):
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    return x + 1j * y


def coupling_matrix_at_layer(z_layer):
    cs = z_layer[:, :, None] * np.conj(z_layer[:, None, :])
    cs_mean = cs.mean(axis=0)
    C = np.abs(cs_mean)
    np.fill_diagonal(C, 0.0)
    return C


def laplacian_eigendecomp(C):
    deg = C.sum(axis=1)
    L = np.diag(deg) - C
    L = (L + L.T) / 2.0
    w, V = np.linalg.eigh(L)
    return w, V


def sign_align_trajectory(V_traj, valid_mask):
    """V_traj: (L, D). valid_mask: (L,) of bools. Sign-align starting
    from first valid layer."""
    out = V_traj.copy()
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) <= 1:
        return out
    for i in range(1, len(valid_idx)):
        l_curr = valid_idx[i]
        l_prev = valid_idx[i - 1]
        if np.dot(out[l_prev], out[l_curr]) < 0:
            out[l_curr] = -out[l_curr]
    return out


def saturation_at_thresholds(O_traj, start, thresholds):
    """O_traj: (L,). Returns dict mapping each threshold to first
    layer index >= start where O reaches base + thr * (asymp - base).
    asymp = mean of last 3 layers; base = O[start]."""
    L = len(O_traj)
    asymp = float(np.mean(O_traj[max(start, L - 3):]))
    base  = float(O_traj[start])
    rise  = asymp - base
    out = {
        "asymptote": asymp,
        "baseline":  base,
        "rise":      rise,
        "by_threshold": {},
    }
    for thr in thresholds:
        target = base + thr * rise
        candidates = np.where(O_traj[start:] >= target)[0]
        if len(candidates):
            l_star = int(candidates[0]) + start
        else:
            l_star = None
        out["by_threshold"][f"{thr:.2f}"] = {
            "target":            float(target),
            "saturation_layer":  l_star,
        }
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overlap_vs_layer(overlaps, saturations, start, D, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    L = len(next(iter(overlaps.values())))
    xs = np.arange(start, L)
    for (p, q), O in overlaps.items():
        color = PAIR_COLORS[(p, q)]
        ax.plot(xs, O[start:], "-", linewidth=1.6, color=color,
                label=(f"{PAIR_LABELS[(p, q)]}: "
                       f"asymp={saturations[(p, q)]['asymptote']:.3f}, "
                       f"base={saturations[(p, q)]['baseline']:.3f}"))
        ax.axhline(saturations[(p, q)]["asymptote"],
                   color=color, linewidth=0.5, linestyle=":", alpha=0.6)
        # Mark the 90% saturation point if present
        sat_90 = saturations[(p, q)]["by_threshold"]["0.90"]["saturation_layer"]
        if sat_90 is not None:
            ax.scatter(
                [sat_90], [O[sat_90]],
                s=70, marker="o", facecolors="none",
                edgecolors=color, linewidths=1.5, zorder=5,
            )
            ax.annotate(
                f"l*_90={sat_90}", (sat_90, O[sat_90]),
                xytext=(5, -12), textcoords="offset points",
                fontsize=8, color=color,
            )
    # Random-vector overlap baseline
    baseline = 2 / np.sqrt(np.pi * D)
    ax.axhline(baseline, color="gray", linewidth=0.6, linestyle="--",
               label=f"random baseline 2/sqrt(pi*D), D={D}")
    ax.set_xlabel("sublayer l")
    ax.set_ylabel("Fiedler overlap |v_2^p(l) . v_2^q(l)|")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(start - 0.5, L - 0.5)
    ax.set_title(
        f"Cross-position Fiedler overlap vs depth "
        f"(analysis from sublayer {start})",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_eigenvalue_gap_vs_layer(gaps_by_pos, start, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    L = len(next(iter(gaps_by_pos.values())))
    xs = np.arange(start, L)
    for pos in POSITIONS:
        ax.plot(xs, gaps_by_pos[pos][start:], "-",
                linewidth=1.4, color=POSITION_COLORS[pos],
                label=POSITION_LABELS[pos])
    ax.set_xlabel("sublayer l")
    ax.set_ylabel("eigenvalue gap lambda_3 - lambda_2")
    ax.set_title("Eigenvalue gap vs depth (Fiedler reliability)",
                 fontsize=12)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_fiedler_heatmaps(V2_traj_by_pos, valid_mask, save_path):
    fig, axes = plt.subplots(
        len(POSITIONS), 1, figsize=(13, 3.5 * len(POSITIONS)),
        sharex=True,
    )
    for ax, pos in zip(axes, POSITIONS):
        V2 = V2_traj_by_pos[pos].copy()
        # Mask invalid layers (set to nan so they show blank)
        V2[~valid_mask] = np.nan
        M = V2.T
        masked = np.ma.array(M, mask=np.isnan(M))
        vmax = np.nanmax(np.abs(masked)) if np.any(~np.isnan(M)) else 1.0
        im = ax.imshow(
            masked, aspect="auto", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, interpolation="nearest",
        )
        ax.set_ylabel("unit u")
        ax.set_title(
            f"{POSITION_LABELS[pos]}: Fiedler v_2(u, l), sign-aligned",
            fontsize=11,
        )
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    axes[-1].set_xlabel("sublayer l (layer 0 omitted)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_multi_eigenvector_overlap(multi_overlaps, start, D, save_path,
                                    n_eig=4):
    fig, axes = plt.subplots(
        n_eig, 1, figsize=(11, 3 * n_eig), sharex=True
    )
    L = len(next(iter(next(iter(multi_overlaps.values())).values())))
    xs = np.arange(start, L)
    for k_idx, ax in enumerate(axes):
        k = k_idx + 2
        for pair, O in multi_overlaps[k].items():
            color = PAIR_COLORS[pair]
            ax.plot(xs, O[start:], "-", linewidth=1.4,
                    color=color, label=PAIR_LABELS[pair])
        ax.axhline(2 / np.sqrt(np.pi * D), color="gray",
                   linewidth=0.6, linestyle="--",
                   label=f"random baseline (D={D})")
        ax.set_ylabel(f"|v_{k}^p . v_{k}^q|")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Eigenvector index {k}", fontsize=10)
        if k_idx == 0:
            ax.legend(fontsize=8, loc="lower right")
    axes[-1].set_xlabel("sublayer l")
    fig.suptitle(
        "Cross-position overlap for first non-trivial eigenvectors",
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
    parser.add_argument("--radius-data", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="toy_sl_step3b")
    parser.add_argument(
        "--start", type=int, default=1,
        help=("First sublayer included in saturation analysis and "
              "plotting. Default 1 (skip layer 0 embedding)."),
    )
    parser.add_argument("--n-eig", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.radius_data}")
    streams_by_pos = load_step1(args.radius_data)
    L_total = streams_by_pos[5].shape[1]
    D = streams_by_pos[5].shape[2]
    print(f"  n_sublayers={L_total}, d_model={D}")
    print(f"  Analysis starts at sublayer {args.start}")

    # ------------- per-layer coupling and eigendecomp -------------
    print("\nBuilding per-layer coupling matrices ...")
    eigvals_by_pos = {pos: [] for pos in POSITIONS}
    eigvecs_by_pos = {pos: [] for pos in POSITIONS}
    z_by_pos = {pos: per_unit_z(streams_by_pos[pos])
                for pos in POSITIONS}
    valid_mask = np.zeros(L_total, dtype=bool)
    valid_mask[1:] = True   # layer 0 is degenerate; everything else valid

    for pos in POSITIONS:
        z = z_by_pos[pos]
        for l in range(L_total):
            if not valid_mask[l]:
                w = np.full(args.n_eig + 2, np.nan)
                V = np.full((D, args.n_eig + 2), np.nan)
            else:
                C = coupling_matrix_at_layer(z[:, l, :])
                w_full, V_full = laplacian_eigendecomp(C)
                w = w_full[:args.n_eig + 2]
                V = V_full[:, :args.n_eig + 2]
            eigvals_by_pos[pos].append(w)
            eigvecs_by_pos[pos].append(V)
        eigvals_by_pos[pos] = np.stack(eigvals_by_pos[pos], axis=0)
        eigvecs_by_pos[pos] = np.stack(eigvecs_by_pos[pos], axis=0)

    # ------------- sign-align v_2 trajectories -------------
    V2_traj_by_pos = {}
    for pos in POSITIONS:
        v2 = eigvecs_by_pos[pos][:, :, 1]
        V2_traj_by_pos[pos] = sign_align_trajectory(v2, valid_mask)

    # ------------- cross-position overlaps -------------
    pairs = [(5, 6), (5, 7), (6, 7)]
    overlap_v2 = {}
    multi_overlaps = {k: {} for k in range(2, args.n_eig + 2)}
    for (p, q) in pairs:
        Vp = eigvecs_by_pos[p]
        Vq = eigvecs_by_pos[q]
        # Fiedler overlap (abs, sign-flip invariant)
        overlap = np.abs((Vp[:, :, 1] * Vq[:, :, 1]).sum(axis=1))
        # Set invalid layers to NaN explicitly
        overlap[~valid_mask] = np.nan
        overlap_v2[(p, q)] = overlap
        for k in range(2, args.n_eig + 2):
            o = np.abs((Vp[:, :, k - 1] * Vq[:, :, k - 1]).sum(axis=1))
            o[~valid_mask] = np.nan
            multi_overlaps[k][(p, q)] = o

    # ------------- saturation analysis -------------
    print("\nSaturation analysis (multiple thresholds) ...")
    saturations = {}
    for (p, q) in pairs:
        sat = saturation_at_thresholds(
            overlap_v2[(p, q)], start=args.start,
            thresholds=THRESHOLDS,
        )
        saturations[(p, q)] = sat
        print(f"  {PAIR_LABELS[(p, q)]:<18}  "
              f"baseline={sat['baseline']:.3f}, "
              f"asymp={sat['asymptote']:.3f}, "
              f"rise={sat['rise']:+.3f}")
        for thr_str, info in sat["by_threshold"].items():
            ls = info["saturation_layer"]
            ls_str = f"{ls}" if ls is not None else "n/a"
            print(f"      {thr_str} threshold "
                  f"(target {info['target']:.3f}): l*={ls_str}")

    # ------------- eigenvalue gaps -------------
    gaps_by_pos = {}
    for pos in POSITIONS:
        gap = (
            eigvals_by_pos[pos][:, 2] - eigvals_by_pos[pos][:, 1]
        )
        gaps_by_pos[pos] = gap

    # ------------- Plots -------------
    print("\nMaking plots ...")
    plot_overlap_vs_layer(
        overlap_v2, saturations, args.start, D,
        out_dir / "fiedler_overlap_vs_layer.png",
    )
    plot_eigenvalue_gap_vs_layer(
        gaps_by_pos, args.start,
        out_dir / "eigenvalue_gap_vs_layer.png",
    )
    plot_fiedler_heatmaps(
        V2_traj_by_pos, valid_mask,
        out_dir / "fiedler_heatmaps_per_position.png",
    )
    plot_multi_eigenvector_overlap(
        multi_overlaps, args.start, D,
        out_dir / "multi_eigenvector_overlap.png",
        n_eig=args.n_eig,
    )

    # ------------- Save data -------------
    save_dict = {}
    for pos in POSITIONS:
        save_dict[f"V_traj_pos{pos}"]     = eigvecs_by_pos[pos]
        save_dict[f"eigvals_pos{pos}"]    = eigvals_by_pos[pos]
        save_dict[f"V2_aligned_pos{pos}"] = V2_traj_by_pos[pos]
    for (p, q) in pairs:
        save_dict[f"overlap_v2_pos{p}_pos{q}"] = overlap_v2[(p, q)]
        for k in range(2, args.n_eig + 2):
            save_dict[f"overlap_v{k}_pos{p}_pos{q}"] = (
                multi_overlaps[k][(p, q)]
            )
    np.savez(out_dir / "step3b_data.npz", **save_dict)

    summary = {
        "radius_data":   args.radius_data,
        "n_sublayers":   int(L_total),
        "d_model":       int(D),
        "analysis_start": int(args.start),
        "thresholds":    THRESHOLDS,
        "saturation": {
            f"pos{p}_pos{q}": {
                "asymptote":     saturations[(p, q)]["asymptote"],
                "baseline":      saturations[(p, q)]["baseline"],
                "rise":          saturations[(p, q)]["rise"],
                "by_threshold":  saturations[(p, q)]["by_threshold"],
            }
            for (p, q) in pairs
        },
        "fiedler_overlap_at_final_layer": {
            f"pos{p}_pos{q}": float(overlap_v2[(p, q)][-1])
            for (p, q) in pairs
        },
        "fiedler_overlap_at_start_layer": {
            f"pos{p}_pos{q}": float(overlap_v2[(p, q)][args.start])
            for (p, q) in pairs
        },
        "random_baseline_2_over_sqrt_piD": float(
            2 / np.sqrt(np.pi * D)
        ),
        "eigenvalue_gap_summary": {
            str(pos): {
                "min":      float(np.nanmin(gaps_by_pos[pos][args.start:])),
                "median":   float(np.nanmedian(gaps_by_pos[pos][args.start:])),
                "max":      float(np.nanmax(gaps_by_pos[pos][args.start:])),
                "argmin_l": int(
                    np.nanargmin(gaps_by_pos[pos][args.start:]) + args.start
                ),
            }
            for pos in POSITIONS
        },
    }
    with open(out_dir / "saturation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
