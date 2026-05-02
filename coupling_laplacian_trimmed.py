"""
Functional coupling matrix and graph Laplacian spectrum.

For each position (5, 6, 7) and each coupling flavor, builds the D x D
symmetric coupling matrix C between residual units, then computes:

  Unnormalized Laplacian:    L      = D_C - C
  Symmetric normalized:      L_sym  = I - D_C^(-1/2) C D_C^(-1/2)

where D_C is the diagonal degree matrix (with degrees taken as the row
sum of |C| for the signed flavor, otherwise just row sum of C).

Three coupling flavors:
  cospec_mag    C[u,v] = |cospec_mag(u,v)|        amplitude + consistency
  plv           C[u,v] = PLV(u,v)                  pure phase consistency
  signed        C[u,v] = cospec_mag * cos(phase)   in-phase = +, anti-phase = -

Outputs in --out-dir:
  coupling_matrices.png        D x D heatmaps, reordered by Fiedler vector
  spectrum_unnormalized.png    eigenvalues of L
  spectrum_normalized.png      eigenvalues of L_sym
  fiedler_vectors.png          Fiedler vector per position / flavor
  low_eigenvectors.png         first 4 non-trivial eigenvectors as a heatmap
  laplacian_summary.json       lambda_2, gaps, suggested k

Usage:
  python coupling_laplacian.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_coupling

  # Edge-control run: drop first/last 2 captured sublayers before cospec/PLV.
  python coupling_laplacian_trimmed.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_coupling_trim2 \
      --trim-sublayers 2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from toy_transformer_addition import (
    ToyTransformer,
    ModelConfig,
    capture_residual_streams,
)
from disentangle_cospec import (
    compute_metrics,
    POSITIONS,
    POSITION_LABELS,
    POSITION_COLORS,
)


COUPLING_FLAVORS = ["cospec_mag", "plv", "signed"]
FLAVOR_LABELS = {
    "cospec_mag": "|cospec|",
    "plv":        "PLV",
    "signed":     "signed (cospec * cos phi)",
}


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = ToyTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def collect_streams(model, batch_size=64):
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]
    _, captures = capture_residual_streams(
        model, pairs, device, batch_size=batch_size
    )
    sublayer_meta = [(c["kind"], c["layer"]) for c in captures]
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    streams_by_pos = {}
    for pos in POSITIONS:
        s = stacked[:, :, pos, :].permute(1, 0, 2).contiguous()
        streams_by_pos[pos] = s.numpy().astype(np.float32)
    return streams_by_pos, sublayer_meta


def trim_streams_by_depth(streams_by_pos, sublayer_meta, trim):
    """Drop boundary sublayers before cross-spectral / PLV analysis.

    trim=0 keeps the original stream. trim=k keeps indices k : L-k.
    This is applied before compute_metrics(), so every coupling matrix is
    based only on the retained interior depth trajectory.
    """
    if trim < 0:
        raise ValueError("--trim-sublayers must be >= 0")

    any_pos = next(iter(streams_by_pos))
    L = streams_by_pos[any_pos].shape[1]
    if 2 * trim >= L:
        raise ValueError(
            f"--trim-sublayers={trim} removes all/too many sublayers "
            f"for L={L}. Need 2*trim < L."
        )

    if trim == 0:
        used_indices = list(range(L))
        return streams_by_pos, sublayer_meta, used_indices

    used_indices = list(range(trim, L - trim))
    trimmed = {
        pos: arr[:, trim:L - trim, :].copy()
        for pos, arr in streams_by_pos.items()
    }
    trimmed_meta = [sublayer_meta[i] for i in used_indices]
    return trimmed, trimmed_meta, used_indices


# ---------------------------------------------------------------------------
# Coupling and Laplacian construction
# ---------------------------------------------------------------------------

def build_coupling_matrix(metrics, flavor):
    """Return dense symmetric D x D coupling matrix with zero diagonal."""
    D = metrics["bin1_power"].shape[0]
    u = metrics["u_idx"]
    v = metrics["v_idx"]
    if flavor == "cospec_mag":
        w = metrics["cospec_mag"]
    elif flavor == "plv":
        w = metrics["plv"]
    elif flavor == "signed":
        w = metrics["cospec_mag"] * np.cos(metrics["cospec_phase"])
    else:
        raise ValueError(flavor)
    C = np.zeros((D, D), dtype=np.float64)
    C[u, v] = w
    C[v, u] = w
    np.fill_diagonal(C, 0.0)
    return C


def laplacian_unnormalized(C, signed=False):
    """L = D - C. Degrees are row sums of C (or row sums of |C| if signed)."""
    if signed:
        d = np.abs(C).sum(axis=1)
    else:
        d = C.sum(axis=1)
    return np.diag(d) - C, d


def laplacian_normalized(C, signed=False):
    """L_sym = I - D^(-1/2) C D^(-1/2)."""
    if signed:
        d = np.abs(C).sum(axis=1)
    else:
        d = C.sum(axis=1)
    eps = 1e-12
    d_inv_sqrt = 1.0 / np.sqrt(d + eps)
    norm_C = (C * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    L_sym = np.eye(C.shape[0]) - norm_C
    return L_sym, d


def eigendecompose(L):
    """Symmetric eigendecomposition. Returns sorted ascending."""
    L = (L + L.T) / 2.0
    w, V = np.linalg.eigh(L)
    return w, V


def fiedler_vector(L):
    w, V = eigendecompose(L)
    return V[:, 1], w[1]


def suggest_k_from_gap(eigs, k_max=12):
    """Return the index k for which eigs[k] - eigs[k-1] is largest
    (1-indexed gap, scanning k = 1..k_max-1). Heuristic for number
    of communities."""
    eigs = np.asarray(eigs)
    upper = min(k_max, len(eigs) - 1)
    gaps = np.diff(eigs[:upper + 1])
    if len(gaps) == 0:
        return 1, 0.0
    k = int(np.argmax(gaps)) + 1
    return k, float(gaps[k - 1])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_coupling_matrices(C_by_pos_flavor, save_path):
    """Rows = positions, cols = flavors. Each cell shows the coupling
    matrix reordered by the Fiedler vector of that matrix's
    unnormalized Laplacian."""
    n_rows = len(POSITIONS)
    n_cols = len(COUPLING_FLAVORS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows),
    )
    if n_rows == 1:
        axes = np.array([axes])
    for r, pos in enumerate(POSITIONS):
        for c, flavor in enumerate(COUPLING_FLAVORS):
            ax = axes[r, c]
            C = C_by_pos_flavor[pos][flavor]
            signed = flavor == "signed"
            L, _ = laplacian_unnormalized(C, signed=signed)
            f_vec, _ = fiedler_vector(L)
            order = np.argsort(f_vec)
            C_sorted = C[order][:, order]
            if signed:
                vmax = np.abs(C_sorted).max()
                im = ax.imshow(
                    C_sorted, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax,
                    aspect="equal", interpolation="nearest",
                )
            else:
                vmax = C_sorted.max()
                im = ax.imshow(
                    C_sorted, cmap="viridis",
                    vmin=0, vmax=vmax,
                    aspect="equal", interpolation="nearest",
                )
            if r == 0:
                ax.set_title(FLAVOR_LABELS[flavor], fontsize=11)
            if c == 0:
                ax.set_ylabel(POSITION_LABELS[pos], fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Coupling matrices, reordered by Fiedler vector",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_spectrum(eigs_by_pos_flavor, save_path, title_suffix,
                  k_zoom=20):
    """Two-panel plot: full spectrum on the left, low end zoom on the right."""
    fig, axes = plt.subplots(2, len(COUPLING_FLAVORS),
                              figsize=(5 * len(COUPLING_FLAVORS), 9))
    for c, flavor in enumerate(COUPLING_FLAVORS):
        ax_full = axes[0, c]
        ax_zoom = axes[1, c]
        for pos in POSITIONS:
            eigs = eigs_by_pos_flavor[pos][flavor]
            x = np.arange(len(eigs))
            ax_full.plot(
                x, eigs, "o-", markersize=3, linewidth=1.0,
                color=POSITION_COLORS[pos],
                label=POSITION_LABELS[pos],
            )
            ax_zoom.plot(
                x[:k_zoom], eigs[:k_zoom], "o-", markersize=5,
                linewidth=1.4, color=POSITION_COLORS[pos],
                label=POSITION_LABELS[pos],
            )
        ax_full.set_title(
            f"{FLAVOR_LABELS[flavor]} (full)", fontsize=11,
        )
        ax_full.set_xlabel("eigenvalue index")
        ax_full.set_ylabel("eigenvalue")
        ax_full.legend(fontsize=8)

        ax_zoom.set_title(
            f"{FLAVOR_LABELS[flavor]} (low end, k <= {k_zoom})",
            fontsize=11,
        )
        ax_zoom.set_xlabel("eigenvalue index")
        ax_zoom.set_ylabel("eigenvalue")
        ax_zoom.axhline(0, color="gray", linewidth=0.4, linestyle=":")
        ax_zoom.legend(fontsize=8)

    fig.suptitle(
        f"Laplacian spectrum  ({title_suffix})", fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_fiedler_vectors(V_by_pos_flavor, save_path):
    """Bar plot of the Fiedler vector per (position, flavor)."""
    fig, axes = plt.subplots(
        len(POSITIONS), len(COUPLING_FLAVORS),
        figsize=(5 * len(COUPLING_FLAVORS), 3 * len(POSITIONS)),
        sharex=True,
    )
    if len(POSITIONS) == 1:
        axes = np.array([axes])
    for r, pos in enumerate(POSITIONS):
        for c, flavor in enumerate(COUPLING_FLAVORS):
            ax = axes[r, c]
            v = V_by_pos_flavor[pos][flavor][:, 1]  # 2nd eigenvector
            colors = ["C3" if x < 0 else "C2" for x in v]
            ax.bar(np.arange(len(v)), v, color=colors)
            ax.axhline(0, color="black", linewidth=0.5)
            if r == 0:
                ax.set_title(FLAVOR_LABELS[flavor], fontsize=11)
            if c == 0:
                ax.set_ylabel(
                    f"{POSITION_LABELS[pos]}\nFiedler v[u]",
                    fontsize=10,
                )
            if r == len(POSITIONS) - 1:
                ax.set_xlabel("unit u")
            ax.tick_params(labelsize=7)
    fig.suptitle(
        "Fiedler vector (eigenvector of lambda_2) per position / flavor",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_low_eigenvectors(V_by_pos_flavor, save_path, n_eig=4):
    """Heatmap of the first n_eig non-trivial eigenvectors stacked,
    one row per (pos, flavor)."""
    rows = []
    row_labels = []
    for pos in POSITIONS:
        for flavor in COUPLING_FLAVORS:
            V = V_by_pos_flavor[pos][flavor]
            for k in range(1, n_eig + 1):
                rows.append(V[:, k])
                row_labels.append(
                    f"{POSITION_LABELS[pos]} | "
                    f"{FLAVOR_LABELS[flavor]} | v_{k+1}"
                )
    M = np.stack(rows, axis=0)
    fig, ax = plt.subplots(figsize=(12, 0.35 * len(rows) + 1))
    vmax = np.abs(M).max()
    im = ax.imshow(
        M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("unit u")
    ax.set_title(
        f"First {n_eig} non-trivial eigenvectors per "
        "(position, coupling flavor)",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="toy_coupling")
    parser.add_argument(
        "--trim-sublayers", type=int, default=0, metavar="K",
        help=("Drop the first K and last K captured sublayers before "
              "compute_metrics()/cospec/PLV/Laplacian analysis. "
              "Default: 0."),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  d_model={cfg.d_model}")

    print("\nCollecting streams ...")
    streams_by_pos, sublayer_meta = collect_streams(model)
    original_sublayer_meta = list(sublayer_meta)
    original_n_sublayers = streams_by_pos[POSITIONS[0]].shape[1]
    streams_by_pos, sublayer_meta, used_sublayer_indices = trim_streams_by_depth(
        streams_by_pos, sublayer_meta, args.trim_sublayers
    )
    if args.trim_sublayers:
        print(
            f"Trimmed first/last {args.trim_sublayers} sublayers before analysis: "
            f"L {original_n_sublayers} -> {streams_by_pos[POSITIONS[0]].shape[1]}"
        )
        print(f"  retained original sublayer indices: {used_sublayer_indices}")
        print(f"  retained sublayers: {sublayer_meta}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nComputing coupling and Laplacian decompositions ...")
    metrics_by_pos = {pos: compute_metrics(streams_by_pos[pos])
                      for pos in POSITIONS}

    C_by_pos_flavor = {pos: {} for pos in POSITIONS}
    eigs_unnorm = {pos: {} for pos in POSITIONS}
    eigs_norm   = {pos: {} for pos in POSITIONS}
    V_unnorm    = {pos: {} for pos in POSITIONS}
    V_norm      = {pos: {} for pos in POSITIONS}
    summary     = {
        "checkpoint": str(args.checkpoint),
        "trim_sublayers": int(args.trim_sublayers),
        "n_sublayers": int(streams_by_pos[POSITIONS[0]].shape[1]),
        "original_n_sublayers": int(original_n_sublayers),
        "used_sublayer_indices": [int(i) for i in used_sublayer_indices],
        "sublayer_meta": [
            {"kind": k, "layer": l} for (k, l) in sublayer_meta
        ],
        "original_sublayer_meta": [
            {"kind": k, "layer": l} for (k, l) in original_sublayer_meta
        ],
        "per_position": {},
    }

    for pos in POSITIONS:
        print(f"\n=== {POSITION_LABELS[pos]} ===")
        summary["per_position"][str(pos)] = {
            "label": POSITION_LABELS[pos], "flavors": {},
        }
        for flavor in COUPLING_FLAVORS:
            signed = flavor == "signed"
            C = build_coupling_matrix(metrics_by_pos[pos], flavor)
            C_by_pos_flavor[pos][flavor] = C

            L, deg = laplacian_unnormalized(C, signed=signed)
            w_u, V_u = eigendecompose(L)
            eigs_unnorm[pos][flavor] = w_u
            V_unnorm[pos][flavor]    = V_u

            L_sym, _ = laplacian_normalized(C, signed=signed)
            w_n, V_n = eigendecompose(L_sym)
            eigs_norm[pos][flavor] = w_n
            V_norm[pos][flavor]    = V_n

            k_unn, gap_unn = suggest_k_from_gap(w_u)
            k_nor, gap_nor = suggest_k_from_gap(w_n)

            entry = {
                "lambda_1_unnorm":             float(w_u[0]),
                "lambda_2_unnorm":             float(w_u[1]),
                "lambda_3_unnorm":             float(w_u[2]),
                "spectral_radius_unnorm":      float(w_u[-1]),
                "lambda_1_norm":               float(w_n[0]),
                "lambda_2_norm":               float(w_n[1]),
                "lambda_3_norm":               float(w_n[2]),
                "spectral_radius_norm":        float(w_n[-1]),
                "suggested_k_unnorm":          k_unn,
                "biggest_gap_unnorm":          gap_unn,
                "suggested_k_norm":            k_nor,
                "biggest_gap_norm":            gap_nor,
                "degree_mean":                 float(deg.mean()),
                "degree_std":                  float(deg.std()),
                "degree_min":                  float(deg.min()),
                "degree_max":                  float(deg.max()),
            }
            summary["per_position"][str(pos)]["flavors"][flavor] = entry
            print(
                f"  {flavor:12s}  lambda_2(unn)={w_u[1]:.4f}  "
                f"lambda_2(nor)={w_n[1]:.4f}  k_hat={k_nor}  "
                f"deg=[{deg.min():.2f}, {deg.max():.2f}]"
            )

    print("\nMaking plots ...")
    plot_coupling_matrices(
        C_by_pos_flavor, out_dir / "coupling_matrices.png"
    )
    plot_spectrum(
        eigs_unnorm, out_dir / "spectrum_unnormalized.png",
        title_suffix="L = D - C",
    )
    plot_spectrum(
        eigs_norm, out_dir / "spectrum_normalized.png",
        title_suffix="L_sym = I - D^(-1/2) C D^(-1/2)",
    )
    plot_fiedler_vectors(
        V_norm, out_dir / "fiedler_vectors.png"
    )
    plot_low_eigenvectors(
        V_norm, out_dir / "low_eigenvectors.png", n_eig=4
    )

    np.savez(
        out_dir / "laplacian_data.npz",
        used_sublayer_indices=np.asarray(used_sublayer_indices, dtype=np.int64),
        **{
            f"C_{pos}_{flavor}":         C_by_pos_flavor[pos][flavor]
            for pos in POSITIONS for flavor in COUPLING_FLAVORS
        },
        **{
            f"eigs_unnorm_{pos}_{flavor}": eigs_unnorm[pos][flavor]
            for pos in POSITIONS for flavor in COUPLING_FLAVORS
        },
        **{
            f"eigs_norm_{pos}_{flavor}":   eigs_norm[pos][flavor]
            for pos in POSITIONS for flavor in COUPLING_FLAVORS
        },
        **{
            f"V_norm_{pos}_{flavor}":      V_norm[pos][flavor]
            for pos in POSITIONS for flavor in COUPLING_FLAVORS
        },
    )

    with open(out_dir / "laplacian_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.trim_sublayers:
        print(
            f"\nNOTE: all coupling matrices/eigenvectors use the trimmed depth axis; "
            f"original indices are saved as used_sublayer_indices."
        )

    print(f"\nOutputs in {out_dir.resolve()}")
    print("  coupling_matrices.png")
    print("  spectrum_unnormalized.png")
    print("  spectrum_normalized.png")
    print("  fiedler_vectors.png")
    print("  low_eigenvectors.png")
    print("  laplacian_data.npz       (full matrices and eigenvectors)")
    print("  laplacian_summary.json")


if __name__ == "__main__":
    main()
