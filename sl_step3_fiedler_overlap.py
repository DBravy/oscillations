"""
Layer-by-layer Fiedler overlap analysis.

For each sublayer l and each position p in {5, 6, 7}, build a coupling
matrix C^p(l), compute its unnormalized Laplacian L = D - C, and take
the Fiedler vector v_2^p(l). Then compute cross-position overlap

    O_pq(l) = | v_2^p(l) . v_2^q(l) |

and identify the layer l* where most of the asymptotic overlap is
established.

Coupling matrix used:
    C^p(l)[u, v] = | mean_n  z_u^p(n, l) * conj(z_v^p(n, l)) |
where
    z_u^p(n, l) = (a_u^p(n, l) - a_u^p(n, 0))
                + i * (grad_u^p(n, l) - grad_u^p(n, 0))

This is a time-localized version of cospec at sublayer l (no FFT).

Outputs in --out-dir:
  fiedler_overlap_vs_layer.png        O_pq(l) for the three position pairs
  eigenvalue_gap_vs_layer.png         lambda_3 - lambda_2 per position;
                                       tells you where Fiedler is reliable
  fiedler_heatmaps_per_position.png   v_2(u, l) per position, sign-aligned
                                       across l
  multi_eigenvector_overlap.png       overlaps for first 4 non-trivial
                                       eigenvectors
  saturation_summary.json             l* values, asymptotes, gap stats

Usage:
  python sl_step3_fiedler_overlap.py \
      --radius-data toy_sl_step1_trained/radius_data.npz \
      --out-dir toy_sl_step3_trained
  python sl_step3_fiedler_overlap.py \
      --radius-data toy_sl_step1_random/radius_data.npz \
      --out-dir toy_sl_step3_random
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_step1(npz_path):
    z = np.load(npz_path)
    return {pos: z[f"streams_pos{pos}"] for pos in POSITIONS}


def per_unit_z(streams):
    """streams: (N, L, D). Returns z: (N, L, D) complex."""
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    return x + 1j * y


def coupling_matrix_at_layer(z_layer):
    """z_layer: (N, D) complex. Returns C: (D, D) real, |mean cospec|."""
    cs = z_layer[:, :, None] * np.conj(z_layer[:, None, :])  # (N, D, D)
    cs_mean = cs.mean(axis=0)                                # (D, D), complex
    C = np.abs(cs_mean)
    np.fill_diagonal(C, 0.0)
    return C


def laplacian_eigendecomp(C):
    """Unnormalized Laplacian L = D - C; return ascending eigenvalues
    and corresponding eigenvectors."""
    deg = C.sum(axis=1)
    L = np.diag(deg) - C
    L = (L + L.T) / 2.0
    w, V = np.linalg.eigh(L)
    return w, V


def sign_align_trajectory(V_traj):
    """V_traj: (L, D) eigenvector across sublayers. Flip signs to
    maximize inner product with previous layer's vector. Skip the first
    layer (used as anchor)."""
    out = V_traj.copy()
    for l in range(1, V_traj.shape[0]):
        if np.dot(out[l - 1], out[l]) < 0:
            out[l] = -out[l]
    return out


def find_saturation_layer(O_traj, threshold=0.9, baseline_layer=0):
    """O_traj: (L,). Returns first layer index where O exceeds
    threshold * asymptote (asymptote = mean of last 3 layers)."""
    L = len(O_traj)
    asymp = float(np.mean(O_traj[max(0, L - 3):]))
    base  = float(O_traj[baseline_layer])
    target = base + threshold * (asymp - base)
    crossings = np.where(O_traj >= target)[0]
    l_star = int(crossings[0]) if len(crossings) else None
    return {
        "asymptote":     asymp,
        "baseline":      base,
        "threshold":     float(threshold),
        "target_value":  float(target),
        "saturation_layer": l_star,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overlap_vs_layer(overlaps, asymptotes, save_path,
                           saturation=None, gap_stats=None):
    """overlaps: dict {(p, q): array of length L}. One panel."""
    fig, ax = plt.subplots(figsize=(11, 6))
    L = len(next(iter(overlaps.values())))
    for (p, q), O in overlaps.items():
        color = PAIR_COLORS[(p, q)]
        ax.plot(np.arange(L), O, "-", linewidth=1.6, color=color,
                label=PAIR_LABELS[(p, q)])
        # Asymptote line
        ax.axhline(asymptotes[(p, q)], color=color, linewidth=0.5,
                   linestyle=":", alpha=0.6)
        # Saturation marker
        if saturation and saturation[(p, q)]["saturation_layer"] is not None:
            ls_sat = saturation[(p, q)]["saturation_layer"]
            ax.scatter(
                [ls_sat], [O[ls_sat]],
                s=60, marker="o", facecolors="none",
                edgecolors=color, linewidths=1.5, zorder=5,
            )
    # Random-vector overlap baseline: 2/sqrt(pi*D) for two random unit vectors
    D = 64    # documented
    if D:
        ax.axhline(
            2 / np.sqrt(np.pi * D), color="gray", linewidth=0.6,
            linestyle="--",
            label=f"random baseline 2/sqrt(pi*D), D=64",
        )
    ax.set_xlabel("sublayer l")
    ax.set_ylabel("Fiedler overlap |v_2^p(l) . v_2^q(l)|")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Cross-position Fiedler overlap vs depth",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_eigenvalue_gap_vs_layer(gaps_by_pos, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    L = len(next(iter(gaps_by_pos.values())))
    for pos in POSITIONS:
        gap = gaps_by_pos[pos]
        ax.plot(np.arange(L), gap, "-", linewidth=1.4,
                color=POSITION_COLORS[pos],
                label=POSITION_LABELS[pos])
    ax.set_xlabel("sublayer l")
    ax.set_ylabel("eigenvalue gap lambda_3 - lambda_2")
    ax.set_title(
        "Eigenvalue gap vs depth (Fiedler reliability)",
        fontsize=12,
    )
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_fiedler_heatmaps(V2_traj_by_pos, save_path):
    """One heatmap per position: rows = units, cols = sublayers,
    color = Fiedler vector entry (sign-aligned)."""
    fig, axes = plt.subplots(
        len(POSITIONS), 1, figsize=(13, 3.5 * len(POSITIONS)),
        sharex=True,
    )
    for ax, pos in zip(axes, POSITIONS):
        V2 = V2_traj_by_pos[pos]                # (L, D)
        # Plot as (units, layers): transpose
        M = V2.T                                # (D, L)
        vmax = np.abs(M).max()
        im = ax.imshow(
            M, aspect="auto", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, interpolation="nearest",
        )
        ax.set_ylabel("unit u")
        ax.set_title(
            f"{POSITION_LABELS[pos]}: Fiedler vector v_2(u, l), "
            "sign-aligned across l",
            fontsize=11,
        )
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    axes[-1].set_xlabel("sublayer l")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_multi_eigenvector_overlap(multi_overlaps, save_path, n_eig=4):
    """For each pair of positions, plot |v_k^p(l) . v_k^q(l)| for
    k in {2, 3, 4, 5} (i.e. first n_eig non-trivial eigenvectors)."""
    fig, axes = plt.subplots(
        n_eig, 1, figsize=(11, 3 * n_eig), sharex=True
    )
    for k_idx, ax in enumerate(axes):
        k = k_idx + 2          # 2, 3, 4, 5 (skipping the trivial v_1)
        for pair, O in multi_overlaps[k].items():
            color = PAIR_COLORS[pair]
            ax.plot(np.arange(len(O)), O, "-", linewidth=1.4,
                    color=color, label=PAIR_LABELS[pair])
        D = 64
        ax.axhline(
            2 / np.sqrt(np.pi * D), color="gray", linewidth=0.6,
            linestyle="--", label="random baseline (D=64)",
        )
        ax.set_ylabel(f"|v_{k}^p(l) . v_{k}^q(l)|")
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
    parser.add_argument("--out-dir", type=str,
                        default="toy_sl_step3")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Fraction of asymptote for l* threshold")
    parser.add_argument("--n-eig", type=int, default=4,
                        help="Number of non-trivial eigenvectors to track")
    parser.add_argument("--warmup", type=int, default=1,
                        help=("Skip the first this-many sublayers when "
                              "displaying the trajectory; layer 0 has "
                              "z=0 so coupling is degenerate."))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.radius_data}")
    streams_by_pos = load_step1(args.radius_data)
    L_total = streams_by_pos[5].shape[1]
    D = streams_by_pos[5].shape[2]
    print(f"  n_sublayers={L_total}, d_model={D}")

    # ------------- per-layer coupling and Fiedler vectors -------------
    print("\nBuilding per-layer coupling matrices and Fiedler vectors ...")
    eigvals_by_pos = {pos: [] for pos in POSITIONS}    # (L, n_eig+1)
    eigvecs_by_pos = {pos: [] for pos in POSITIONS}    # list of (D, n_eig+1)
    z_by_pos = {pos: per_unit_z(streams_by_pos[pos])
                for pos in POSITIONS}

    for pos in POSITIONS:
        z = z_by_pos[pos]
        for l in range(L_total):
            if l == 0:
                # z = 0 at layer 0; coupling matrix is degenerate
                w = np.zeros(D)
                V = np.eye(D)
            else:
                C = coupling_matrix_at_layer(z[:, l, :])
                w, V = laplacian_eigendecomp(C)
            eigvals_by_pos[pos].append(w[:args.n_eig + 2])
            eigvecs_by_pos[pos].append(V[:, :args.n_eig + 2])

        eigvals_by_pos[pos] = np.stack(eigvals_by_pos[pos], axis=0)
        # eigvecs: shape (L, D, n_eig+2). Convert to per-eigenindex
        # trajectories of shape (L, D).
        all_V = np.stack(eigvecs_by_pos[pos], axis=0)
        eigvecs_by_pos[pos] = all_V

    # ------------- sign-align v_2 trajectories per position -------------
    V2_traj_by_pos = {}
    for pos in POSITIONS:
        v2_traj = eigvecs_by_pos[pos][:, :, 1]    # (L, D)
        V2_traj_by_pos[pos] = sign_align_trajectory(v2_traj)

    # ------------- compute cross-position overlaps -------------
    pairs = [(5, 6), (5, 7), (6, 7)]
    overlap_v2 = {}
    multi_overlaps = {k: {} for k in range(2, args.n_eig + 2)}
    for (p, q) in pairs:
        Vp = eigvecs_by_pos[p]
        Vq = eigvecs_by_pos[q]
        # Fiedler-only overlap (using sign-aligned vectors is fine,
        # and the abs value makes it sign-invariant anyway)
        overlap = np.abs((Vp[:, :, 1] * Vq[:, :, 1]).sum(axis=1))
        overlap_v2[(p, q)] = overlap
        for k in range(2, args.n_eig + 2):
            multi_overlaps[k][(p, q)] = np.abs(
                (Vp[:, :, k - 1] * Vq[:, :, k - 1]).sum(axis=1)
            )

    # ------------- saturation analysis -------------
    print("\nSaturation analysis ...")
    saturation = {}
    asymptotes = {}
    for (p, q) in pairs:
        sat = find_saturation_layer(
            overlap_v2[(p, q)], threshold=args.threshold,
        )
        saturation[(p, q)] = sat
        asymptotes[(p, q)] = sat["asymptote"]
        print(
            f"  {PAIR_LABELS[(p, q)]:<18}  "
            f"asymptote={sat['asymptote']:.3f}, "
            f"l*={sat['saturation_layer']} "
            f"(>= {sat['target_value']:.3f} = "
            f"{args.threshold:.0%} of (asymptote - baseline))"
        )

    # ------------- eigenvalue gap (Fiedler reliability) -------------
    gaps_by_pos = {}
    for pos in POSITIONS:
        # gap = lambda_3 - lambda_2 (we want strictly positive gap above v_2)
        gaps_by_pos[pos] = (
            eigvals_by_pos[pos][:, 2] - eigvals_by_pos[pos][:, 1]
        )

    # ------------- Plots -------------
    print("\nMaking plots ...")
    plot_overlap_vs_layer(
        overlap_v2, asymptotes,
        out_dir / "fiedler_overlap_vs_layer.png",
        saturation=saturation, gap_stats=gaps_by_pos,
    )
    plot_eigenvalue_gap_vs_layer(
        gaps_by_pos,
        out_dir / "eigenvalue_gap_vs_layer.png",
    )
    plot_fiedler_heatmaps(
        V2_traj_by_pos,
        out_dir / "fiedler_heatmaps_per_position.png",
    )
    plot_multi_eigenvector_overlap(
        multi_overlaps,
        out_dir / "multi_eigenvector_overlap.png",
        n_eig=args.n_eig,
    )

    # ------------- Save data -------------
    save_dict = {}
    for pos in POSITIONS:
        save_dict[f"V_traj_pos{pos}"]   = eigvecs_by_pos[pos]
        save_dict[f"eigvals_pos{pos}"]  = eigvals_by_pos[pos]
        save_dict[f"V2_aligned_pos{pos}"] = V2_traj_by_pos[pos]
    for (p, q) in pairs:
        save_dict[f"overlap_v2_pos{p}_pos{q}"] = overlap_v2[(p, q)]
        for k in range(2, args.n_eig + 2):
            save_dict[f"overlap_v{k}_pos{p}_pos{q}"] = (
                multi_overlaps[k][(p, q)]
            )
    np.savez(out_dir / "step3_data.npz", **save_dict)

    summary = {
        "radius_data":    args.radius_data,
        "n_sublayers":    int(L_total),
        "d_model":        int(D),
        "threshold":      float(args.threshold),
        "saturation": {
            f"pos{p}_pos{q}": {
                "asymptote":      saturation[(p, q)]["asymptote"],
                "baseline":       saturation[(p, q)]["baseline"],
                "target_value":   saturation[(p, q)]["target_value"],
                "saturation_layer": saturation[(p, q)]["saturation_layer"],
            }
            for (p, q) in pairs
        },
        "fiedler_overlap_at_final_layer": {
            f"pos{p}_pos{q}": float(overlap_v2[(p, q)][-1])
            for (p, q) in pairs
        },
        "eigenvalue_gap_per_position_lambda3_minus_lambda2_summary": {
            str(pos): {
                "min":     float(np.min(gaps_by_pos[pos][1:])),
                "median":  float(np.median(gaps_by_pos[pos][1:])),
                "max":     float(np.max(gaps_by_pos[pos][1:])),
                "min_layer": int(np.argmin(gaps_by_pos[pos][1:]) + 1),
            }
            for pos in POSITIONS
        },
    }
    with open(out_dir / "saturation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs in {out_dir.resolve()}")
    print("  fiedler_overlap_vs_layer.png   (main)")
    print("  eigenvalue_gap_vs_layer.png    (Fiedler reliability)")
    print("  fiedler_heatmaps_per_position.png")
    print("  multi_eigenvector_overlap.png")
    print("  step3_data.npz                 (raw arrays)")
    print("  saturation_summary.json")


if __name__ == "__main__":
    main()
