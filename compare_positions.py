"""
Compare phase / rotation analysis across positions 5, 6, 7 of the
toy transformer.

Position 5 = "=" token: model has just been shown the full problem and
            must predict c2. Carries propagate into c2, so all three
            answer digits are effectively committed by this position.
Position 6 = c2 token:  model has emitted c2, must predict c1.
Position 7 = c1 token:  model has emitted c2, c1, must predict c0.

For each position we run the same four-variant decomposition
(cumulative, combined_delta, attn_delta, mlp_delta) and the same
per-unit / per-pair analyses, then make overlay plots so the
positions can be compared directly.

Outputs in --out-dir:
  - R_distribution_by_position.png
  - phase_diff_distribution_by_position.png
  - phase_clustering_bars.png
  - cospec_magnitude_by_position.png
  - position_comparison_summary.json

Usage:
  python compare_positions.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_position_comparison
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
from analyze_toy_residuals import (
    make_variants,
    analyze_variant,
    summarize_variant,
    VARIANTS,
    QUAD_BAND,
    SEED,
    N_SHUFFLE_RUNS,
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


def collect_streams_all_positions(model, cfg, batch_size=64):
    """
    Run the model on all 256 (a, b) pairs once and capture residuals
    at every sublayer. Return a dict {position: streams} for the
    target positions.

    Returns:
      streams_by_pos: {pos: (N, n_sublayers, D)}
      sublayer_meta:  list of (kind, layer_idx)
    """
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]
    _, captures = capture_residual_streams(
        model, pairs, device, batch_size=batch_size
    )
    sublayer_meta = [(c["kind"], c["layer"]) for c in captures]
    # (n_sublayers, N, T, D)
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    streams_by_pos = {}
    for pos in POSITIONS:
        at_pos = stacked[:, :, pos, :]              # (n_sublayers, N, D)
        s = at_pos.permute(1, 0, 2).contiguous()    # (N, n_sublayers, D)
        streams_by_pos[pos] = s.numpy().astype(np.float32)
    return streams_by_pos, sublayer_meta


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def plot_R_distribution_by_position(results_by_pos, save_path=None):
    """One subplot per variant. Within each subplot, overlay R histograms
    for positions 5, 6, 7 (real distributions; nulls drawn dotted)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, label in zip(axes.flat, VARIANTS):
        for pos in POSITIONS:
            data = results_by_pos[pos][label]
            color = POSITION_COLORS[pos]
            ax.hist(
                data["R"], bins=30, density=True,
                histtype="step", linewidth=1.6, color=color,
                label=(f"{POSITION_LABELS[pos]} "
                       f"(R={data['R'].mean():.2f}, "
                       f"null={data['null_R'].mean():.2f})"),
            )
            ax.hist(
                data["null_R"].flatten(), bins=30, density=True,
                histtype="step", linewidth=0.7, color=color,
                linestyle=":", alpha=0.4,
            )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("rotation count R")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Per-unit rotation count by position (real solid, null dotted)",
        fontsize=12,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_diff_distribution_by_position(results_by_pos, save_path=None):
    """One subplot per variant. Overlay phase-difference histograms for
    positions 5, 6, 7."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, label in zip(axes.flat, VARIANTS):
        for pos in POSITIONS:
            data = results_by_pos[pos][label]
            color = POSITION_COLORS[pos]
            ax.hist(
                data["phase"], bins=60, range=(-np.pi, np.pi),
                density=True, histtype="step", linewidth=1.6,
                color=color, label=POSITION_LABELS[pos],
            )
        for x in (-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi):
            ax.axvline(x, color="gray", linewidth=0.4, linestyle=":")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("phase difference at slow bin")
        ax.set_ylabel("density")
        ax.legend(fontsize=9)
    fig.suptitle(
        "Pairwise phase difference distribution by position (all pairs)",
        fontsize=12,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_clustering_bars(results_by_pos, save_path=None):
    """For each variant, a grouped bar chart showing the fraction of pairs
    near each phase cluster (0, +pi/2, -pi/2, pi), grouped by position."""
    cluster_names = ["near_0", "near_pi_over_2",
                     "near_minus_pi_over_2", "near_pi"]
    cluster_labels = ["0", "+pi/2", "-pi/2", "pi"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    width = 0.25
    x_base = np.arange(len(cluster_names))

    for ax, label in zip(axes.flat, VARIANTS):
        for i, pos in enumerate(POSITIONS):
            data = results_by_pos[pos][label]
            phase = data["phase"]

            def near(target, band=QUAD_BAND):
                if target == "pi":
                    return ((np.abs(phase - np.pi) < band)
                            | (np.abs(phase + np.pi) < band))
                return np.abs(phase - target) < band

            fractions = [
                near(0).mean(),
                near(np.pi / 2).mean(),
                near(-np.pi / 2).mean(),
                near("pi").mean(),
            ]
            offset = (i - 1) * width
            ax.bar(
                x_base + offset, fractions, width,
                color=POSITION_COLORS[pos],
                label=POSITION_LABELS[pos],
            )
        # Reference: uniform fraction = 2 * band / (2 * pi) per cluster
        uniform = QUAD_BAND / np.pi
        ax.axhline(uniform, color="gray", linewidth=0.6,
                   linestyle="--",
                   label=f"uniform ({uniform:.3f})")
        ax.set_xticks(x_base)
        ax.set_xticklabels(cluster_labels)
        ax.set_xlabel("cluster center")
        ax.set_ylabel("fraction of pairs")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Phase clustering by position (band = +/- {QUAD_BAND:.3f} rad)",
        fontsize=12,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_cospec_magnitude_by_position(results_by_pos, save_path=None):
    """For each variant, overlay the cospec-magnitude distribution
    (log scale) for the three positions. A position with more
    pair-locking shows mass at higher magnitudes."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, label in zip(axes.flat, VARIANTS):
        for pos in POSITIONS:
            data = results_by_pos[pos][label]
            mag = data["mag"]
            mag = mag[mag > 0]
            ax.hist(
                np.log10(mag + 1e-12), bins=60, density=True,
                histtype="step", linewidth=1.5,
                color=POSITION_COLORS[pos],
                label=(f"{POSITION_LABELS[pos]} "
                       f"(p95={np.percentile(data['mag'], 95):.3f})"),
            )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("log10(cospec magnitude)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Cospec magnitude distribution by position",
        fontsize=12,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str,
                        default="toy_position_comparison")
    parser.add_argument("--n-shuffle", type=int, default=N_SHUFFLE_RUNS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print(f"Loading checkpoint from {args.checkpoint} ...")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}")

    print("\nCollecting residual streams at positions 5, 6, 7 "
          "on all 256 (a, b) pairs (single forward pass) ...")
    streams_by_pos, sublayer_meta = collect_streams_all_positions(model, cfg)
    for pos, s in streams_by_pos.items():
        print(f"  pos {pos}: streams shape {s.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nAnalyzing each position ...")
    results_by_pos = {}
    for pos in POSITIONS:
        print(f"\n=== Position {pos} ({POSITION_LABELS[pos]}) ===")
        variants = make_variants(streams_by_pos[pos])
        results = {}
        for label in VARIANTS:
            results[label] = analyze_variant(
                variants[label], label, args.n_shuffle,
                np.random.default_rng(
                    SEED + (pos * 10) + hash(label) % 1000
                ),
            )
        results_by_pos[pos] = results

    print("\nMaking comparison plots ...")
    plot_R_distribution_by_position(
        results_by_pos,
        save_path=out_dir / "R_distribution_by_position.png",
    )
    plot_phase_diff_distribution_by_position(
        results_by_pos,
        save_path=out_dir / "phase_diff_distribution_by_position.png",
    )
    plot_phase_clustering_bars(
        results_by_pos,
        save_path=out_dir / "phase_clustering_bars.png",
    )
    plot_cospec_magnitude_by_position(
        results_by_pos,
        save_path=out_dir / "cospec_magnitude_by_position.png",
    )

    summary = {
        "checkpoint":    str(args.checkpoint),
        "positions":     POSITIONS,
        "n_samples":     int(streams_by_pos[POSITIONS[0]].shape[0]),
        "n_sublayers":   int(streams_by_pos[POSITIONS[0]].shape[1]),
        "d_model":       int(streams_by_pos[POSITIONS[0]].shape[2]),
        "sublayer_meta": [
            {"kind": k, "layer": l} for (k, l) in sublayer_meta
        ],
        "results": {
            str(pos): {
                "label": POSITION_LABELS[pos],
                "variants": {
                    label: summarize_variant(results_by_pos[pos][label])
                    for label in VARIANTS
                },
            }
            for pos in POSITIONS
        },
    }
    with open(out_dir / "position_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Position comparison summary ---")
    print(f"{'variant':<18} {'metric':<28} "
          + " ".join(f"{POSITION_LABELS[p]:>16}" for p in POSITIONS))
    for label in VARIANTS:
        print()
        for metric_path in [
            ("R", "abs_mean"),
            ("R", "std"),
            ("null_R", "mean"),
            ("cospec_mag", "p95"),
            ("phase_clustering_fractions", "near_0"),
            ("phase_clustering_fractions", "near_pi"),
            ("phase_clustering_fractions", "near_pi_over_2"),
            ("phase_clustering_fractions", "near_minus_pi_over_2"),
        ]:
            vals = [
                summary["results"][str(p)]["variants"][label][metric_path[0]][metric_path[1]]
                for p in POSITIONS
            ]
            metric_str = ".".join(metric_path)
            print(f"{label:<18} {metric_str:<28} "
                  + " ".join(f"{v:>16.4f}" for v in vals))

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
