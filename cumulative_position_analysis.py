"""
Cumulative residual stream: per-unit phase portraits and top-pair tables
across positions 5, 6, 7.

Two outputs:

1. Per-unit phase portraits comparing positions side by side.
   Rows = units, columns = positions (5, 6, 7). For each unit, the
   trajectory in (a, da/dl) phase space at that position. Same color
   gradient across all panels (viridis = sublayer index 0 to 16).
   Lets you see how a single unit's depth-dynamics differ across
   positions.

2. Top N pairs by cross-spectral magnitude at the slow bin, per position.
   Saved as CSV with columns:
     rank, u, v, cospec_mag, phase, phase_cluster,
     trace_corr, R_u, R_v, mean_abs_u, mean_abs_v
   plus a combined scatter plot of (phase, log10 magnitude) for the
   top-N pairs at each position.

Usage:
  python cumulative_position_analysis.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_cumulative_analysis \
      --top-n 100 \
      --n-units-plot 12

  # Or specify exact units to plot:
  python cumulative_position_analysis.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --units 0 5 12 23 47 63

  # Trim first/last 2 sublayers to reduce edge effects:
  python cumulative_position_analysis.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --trim-sublayers 2
"""

import argparse
import csv
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
from analyze_toy_residuals import (
    rotation_count_per_unit,
    per_unit_spectra,
    all_pair_indices,
    trace_correlations,
    pair_cross_spectral,
    QUAD_BAND,
    SEED,
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


def collect_cumulative_streams(model, cfg, batch_size=64):
    """One forward pass; return cumulative streams at each position."""
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]
    _, captures = capture_residual_streams(
        model, pairs, device, batch_size=batch_size
    )
    sublayer_meta = [(c["kind"], c["layer"]) for c in captures]
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    streams_by_pos = {}
    for pos in POSITIONS:
        at_pos = stacked[:, :, pos, :]
        s = at_pos.permute(1, 0, 2).contiguous()
        streams_by_pos[pos] = s.numpy().astype(np.float32)
    return streams_by_pos, sublayer_meta


# ---------------------------------------------------------------------------
# Per-position cumulative analysis
# ---------------------------------------------------------------------------

def trim_streams_by_depth(streams_by_pos, sublayer_meta, trim):
    """Drop boundary sublayers before analysis.

    trim=0 keeps the original stream. trim=k keeps indices k : L-k.
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


def analyze_cumulative(streams):
    """Compute everything needed for both outputs from a single
    (N, n_sublayers, D) cumulative stream."""
    R = rotation_count_per_unit(streams)
    spectra = per_unit_spectra(streams)
    n_freq = spectra.shape[1]
    freq_bin = 1 if n_freq > 1 else 0

    u_idx, v_idx = all_pair_indices(streams.shape[2])
    trace_corr = trace_correlations(streams, u_idx, v_idx)
    phase, mag = pair_cross_spectral(spectra, u_idx, v_idx, freq_bin=freq_bin)

    # Per-unit "loudness": mean |activation| across samples and sublayers
    loudness = np.abs(streams).mean(axis=(0, 1))

    return {
        "streams":       streams,
        "R":             R,
        "spectra":       spectra,
        "freq_bin":      freq_bin,
        "u_idx":         u_idx,
        "v_idx":         v_idx,
        "trace_corr":    trace_corr,
        "phase":         phase,
        "mag":           mag,
        "loudness":      loudness,
    }


def classify_phase(phi, band=QUAD_BAND):
    """Return one of {'near_0', 'near_pi_over_2', 'near_minus_pi_over_2',
    'near_pi', 'between'} based on which cluster phi is closest to."""
    if abs(phi) < band:
        return "near_0"
    if abs(phi - np.pi / 2) < band:
        return "near_pi_over_2"
    if abs(phi + np.pi / 2) < band:
        return "near_minus_pi_over_2"
    if (abs(phi - np.pi) < band) or (abs(phi + np.pi) < band):
        return "near_pi"
    return "between"


def top_pairs_table(result, top_n):
    """Return list-of-dicts for the top-N pairs by |cospec|."""
    order = np.argsort(-result["mag"])
    rows = []
    R = result["R"]
    loud = result["loudness"]
    for rank, idx in enumerate(order[:top_n], start=1):
        u = int(result["u_idx"][idx])
        v = int(result["v_idx"][idx])
        rows.append({
            "rank":          rank,
            "u":             u,
            "v":             v,
            "cospec_mag":    float(result["mag"][idx]),
            "phase":         float(result["phase"][idx]),
            "phase_cluster": classify_phase(result["phase"][idx]),
            "trace_corr":    float(result["trace_corr"][idx]),
            "R_u":           float(R[u]),
            "R_v":           float(R[v]),
            "mean_abs_u":    float(loud[u]),
            "mean_abs_v":    float(loud[v]),
        })
    return rows


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def select_units_default(R_pos5, n_units):
    """Pick n_units spanning the |R| range at position 5."""
    order = np.argsort(np.abs(R_pos5))
    n = len(order)
    idxs = np.linspace(0, n - 1, n_units).round().astype(int)
    return [int(order[i]) for i in idxs]


def plot_per_unit_portraits_across_positions(
    results_by_pos, units, sample_idx, save_path
):
    """Rows = units, cols = positions. Each panel plots the cumulative
    trajectory of that unit in (a - a_0, grad a - grad a_0)."""
    n_units = len(units)
    fig, axes = plt.subplots(
        n_units, len(POSITIONS),
        figsize=(3.6 * len(POSITIONS), 2.8 * n_units),
        squeeze=False,
    )
    for row, u in enumerate(units):
        for col, pos in enumerate(POSITIONS):
            ax = axes[row, col]
            data = results_by_pos[pos]
            sample = data["streams"][sample_idx]
            n_steps = sample.shape[0]
            a = sample[:, u]
            ga = np.gradient(a)
            x = a - a[0]
            y = ga - ga[0]
            ax.plot(x, y, "-", linewidth=0.7, alpha=0.9)
            ax.scatter(x, y, c=np.arange(n_steps), cmap="viridis", s=12)
            ax.scatter([0], [0], marker="x", color="red", s=30, zorder=5)
            if row == 0:
                ax.set_title(POSITION_LABELS[pos], fontsize=11)
            if col == 0:
                ax.set_ylabel(
                    f"unit {u}\n"
                    f"R = ["
                    f"{results_by_pos[POSITIONS[0]]['R'][u]:.2f}, "
                    f"{results_by_pos[POSITIONS[1]]['R'][u]:.2f}, "
                    f"{results_by_pos[POSITIONS[2]]['R'][u]:.2f}]",
                    fontsize=9,
                )
            ax.set_xlabel("a - a_0", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="gray", linewidth=0.4)
            ax.axvline(0, color="gray", linewidth=0.4)
    fig.suptitle(
        f"Per-unit cumulative phase portraits across positions "
        f"(sample idx {sample_idx})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_per_unit_portraits_mean_trajectory(
    results_by_pos, units, save_path
):
    """Same layout, but plot the mean trajectory across all input pairs
    rather than a single sample. Useful to see what's consistent vs
    sample-specific."""
    n_units = len(units)
    fig, axes = plt.subplots(
        n_units, len(POSITIONS),
        figsize=(3.6 * len(POSITIONS), 2.8 * n_units),
        squeeze=False,
    )
    for row, u in enumerate(units):
        for col, pos in enumerate(POSITIONS):
            ax = axes[row, col]
            data = results_by_pos[pos]
            mean_traj = data["streams"][:, :, u].mean(axis=0)
            n_steps = mean_traj.shape[0]
            ga = np.gradient(mean_traj)
            x = mean_traj - mean_traj[0]
            y = ga - ga[0]
            ax.plot(x, y, "-", linewidth=0.9, alpha=0.9)
            ax.scatter(x, y, c=np.arange(n_steps), cmap="viridis", s=12)
            ax.scatter([0], [0], marker="x", color="red", s=30, zorder=5)
            if row == 0:
                ax.set_title(POSITION_LABELS[pos], fontsize=11)
            if col == 0:
                ax.set_ylabel(
                    f"unit {u}\n"
                    f"R = ["
                    f"{results_by_pos[POSITIONS[0]]['R'][u]:.2f}, "
                    f"{results_by_pos[POSITIONS[1]]['R'][u]:.2f}, "
                    f"{results_by_pos[POSITIONS[2]]['R'][u]:.2f}]",
                    fontsize=9,
                )
            ax.set_xlabel("mean a - a_0", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="gray", linewidth=0.4)
            ax.axvline(0, color="gray", linewidth=0.4)
    fig.suptitle(
        "Per-unit cumulative phase portraits (mean across input pairs)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_top_pairs_scatter(top_pairs_by_pos, save_path):
    """Scatter plot of (phase, log10 mag) for top-N pairs per position."""
    cluster_to_marker = {
        "near_0":              "o",
        "near_pi_over_2":      "^",
        "near_minus_pi_over_2": "v",
        "near_pi":             "s",
        "between":             "x",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, pos in zip(axes, POSITIONS):
        pairs = top_pairs_by_pos[pos]
        for cluster, marker in cluster_to_marker.items():
            xs = [p["phase"] for p in pairs if p["phase_cluster"] == cluster]
            ys = [np.log10(p["cospec_mag"] + 1e-12)
                  for p in pairs if p["phase_cluster"] == cluster]
            if xs:
                ax.scatter(xs, ys, marker=marker, s=30, alpha=0.7,
                           label=cluster)
        for x in (-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi):
            ax.axvline(x, color="gray", linewidth=0.4, linestyle=":")
        ax.set_xlabel("phase difference")
        ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
        ax.set_title(
            f"{POSITION_LABELS[pos]}: top {len(pairs)} pairs",
            fontsize=11,
        )
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("log10(cospec magnitude)")
    fig.suptitle(
        "Top pairs by cospec magnitude: phase vs magnitude",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_top_pairs_cluster_counts(top_pairs_by_pos, save_path):
    """Bar chart: how many of the top-N pairs at each position fall
    into each cluster."""
    clusters = ["near_0", "near_pi_over_2", "near_minus_pi_over_2",
                "near_pi", "between"]
    cluster_labels = ["0", "+pi/2", "-pi/2", "pi", "between"]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    x_base = np.arange(len(clusters))
    for i, pos in enumerate(POSITIONS):
        pairs = top_pairs_by_pos[pos]
        counts = [
            sum(1 for p in pairs if p["phase_cluster"] == c)
            for c in clusters
        ]
        offset = (i - 1) * width
        ax.bar(
            x_base + offset, counts, width,
            color=POSITION_COLORS[pos],
            label=POSITION_LABELS[pos],
        )
    ax.set_xticks(x_base)
    ax.set_xticklabels(cluster_labels)
    ax.set_xlabel("phase cluster")
    ax.set_ylabel("count")
    ax.set_title(
        f"Top pairs by cospec magnitude: cluster counts "
        f"(band = +/- {QUAD_BAND:.3f} rad)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str,
                        default="toy_cumulative_analysis")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of top pairs to dump per position")
    parser.add_argument("--n-units-plot", type=int, default=12,
                        help="If --units not given, pick this many "
                             "spanning the |R| range at position 5")
    parser.add_argument("--units", type=int, nargs="*", default=None,
                        help="Explicit unit indices to plot")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Which input pair to draw single-sample "
                             "phase portraits for")
    parser.add_argument(
        "--trim-sublayers", type=int, default=0, metavar="K",
        help=("Drop the first K and last K captured sublayers before "
              "analysis. Default: 0."),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}")

    print("\nCollecting cumulative residual streams "
          "at positions 5, 6, 7 ...")
    streams_by_pos, sublayer_meta = collect_cumulative_streams(model, cfg)
    for pos, s in streams_by_pos.items():
        print(f"  pos {pos}: {s.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    original_sublayer_meta = list(sublayer_meta)
    original_n_sublayers = streams_by_pos[POSITIONS[0]].shape[1]
    streams_by_pos, sublayer_meta, used_sublayer_indices = trim_streams_by_depth(
        streams_by_pos, sublayer_meta, args.trim_sublayers
    )
    if args.trim_sublayers:
        print(
            f"\nTrimmed first/last {args.trim_sublayers} sublayers: "
            f"L {original_n_sublayers} -> "
            f"{streams_by_pos[POSITIONS[0]].shape[1]}"
        )
        print(f"  retained original sublayer indices: {used_sublayer_indices}")

    print("\nAnalyzing each position ...")
    results_by_pos = {}
    for pos in POSITIONS:
        print(f"  pos {pos} ...")
        results_by_pos[pos] = analyze_cumulative(streams_by_pos[pos])

    # Pick units to plot
    if args.units:
        units = list(args.units)
        print(f"\nUsing user-specified units: {units}")
    else:
        units = select_units_default(
            results_by_pos[5]["R"], args.n_units_plot
        )
        print(f"\nDefault unit selection (spanning |R| at pos 5, n="
              f"{args.n_units_plot}): {units}")
        for u in units:
            print(f"  unit {u}: |R_pos5|={abs(results_by_pos[5]['R'][u]):.2f}, "
                  f"|R_pos6|={abs(results_by_pos[6]['R'][u]):.2f}, "
                  f"|R_pos7|={abs(results_by_pos[7]['R'][u]):.2f}")

    print("\nMaking per-unit phase portrait plots ...")
    plot_per_unit_portraits_across_positions(
        results_by_pos, units, args.sample_idx,
        out_dir / "per_unit_portraits_single_sample.png",
    )
    plot_per_unit_portraits_mean_trajectory(
        results_by_pos, units,
        out_dir / "per_unit_portraits_mean_trajectory.png",
    )

    print("\nBuilding top-pair tables ...")
    top_pairs_by_pos = {}
    for pos in POSITIONS:
        rows = top_pairs_table(results_by_pos[pos], args.top_n)
        top_pairs_by_pos[pos] = rows
        csv_path = out_dir / f"top_{args.top_n}_pairs_pos{pos}.csv"
        write_csv(rows, csv_path)
        # Cluster counts
        from collections import Counter
        counts = Counter(r["phase_cluster"] for r in rows)
        print(f"  pos {pos}: wrote {csv_path.name}; "
              f"cluster counts: {dict(counts)}")

    print("\nMaking top-pair plots ...")
    plot_top_pairs_scatter(
        top_pairs_by_pos, out_dir / "top_pairs_scatter.png"
    )
    plot_top_pairs_cluster_counts(
        top_pairs_by_pos, out_dir / "top_pairs_cluster_counts.png"
    )

    summary = {
        "checkpoint":   str(args.checkpoint),
        "positions":    POSITIONS,
        "top_n":        args.top_n,
        "n_samples":    int(streams_by_pos[POSITIONS[0]].shape[0]),
        "n_sublayers":  int(streams_by_pos[POSITIONS[0]].shape[1]),
        "original_n_sublayers": int(original_n_sublayers),
        "trim_sublayers": int(args.trim_sublayers),
        "used_sublayer_indices": [int(i) for i in used_sublayer_indices],
        "d_model":      int(streams_by_pos[POSITIONS[0]].shape[2]),
        "freq_bin":     int(results_by_pos[POSITIONS[0]]["freq_bin"]),
        "units_plotted": units,
        "sublayer_meta": [
            {"kind": k, "layer": l} for (k, l) in sublayer_meta
        ],
        "original_sublayer_meta": [
            {"kind": k, "layer": l} for (k, l) in original_sublayer_meta
        ],
        "top_pair_cluster_counts": {
            str(pos): {
                c: sum(1 for r in top_pairs_by_pos[pos]
                       if r["phase_cluster"] == c)
                for c in ["near_0", "near_pi_over_2",
                          "near_minus_pi_over_2", "near_pi", "between"]
            }
            for pos in POSITIONS
        },
        "top_pair_mag_summary": {
            str(pos): {
                "max":    float(top_pairs_by_pos[pos][0]["cospec_mag"]),
                "min_in_top": float(top_pairs_by_pos[pos][-1]["cospec_mag"]),
                "median_in_top": float(np.median(
                    [r["cospec_mag"] for r in top_pairs_by_pos[pos]]
                )),
            }
            for pos in POSITIONS
        },
    }
    with open(out_dir / "cumulative_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs in {out_dir.resolve()}")
    print(f"  CSVs:  top_<N>_pairs_pos{{5,6,7}}.csv")
    print(f"  Plots: per_unit_portraits_single_sample.png")
    print(f"         per_unit_portraits_mean_trajectory.png")
    print(f"         top_pairs_scatter.png")
    print(f"         top_pairs_cluster_counts.png")
    print(f"  JSON:  cumulative_analysis_summary.json")


if __name__ == "__main__":
    main()
