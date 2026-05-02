"""
Find residual units whose phase-space trajectory oscillates without
expanding. The motivating observation from the bigger-model findings:
the typical trajectory is a growing spiral (constant tangent angular
speed, growing radius). This script asks whether some units buck that
pattern.

What "expansion" means here:

  For a unit u at a given position, the depth trajectory is the pair
      a(l)     = residual_u  at sublayer l
      da/dl(l) = layer-derivative of a at sublayer l
  In phase space (a, da/dl) this traces a curve. We measure radius from
  the centroid of that curve:
      r(l) = sqrt( (a(l) - mean_l a)^2 + (da/dl(l) - mean_l da/dl)^2 )

  Expansion metrics per unit (averaged across input pairs):
    r_first_half_mean  = mean of r(l) for l in first half of depth
    r_second_half_mean = mean of r(l) for l in second half of depth
    expansion_ratio    = r_second_half_mean / r_first_half_mean
    log_r_slope        = slope of log(r(l)) vs l, by linear regression

  Interpretation:
    expansion_ratio approx 1, log_r_slope approx 0:
        stationary rotation (closed orbit)
    expansion_ratio > 1, log_r_slope > 0:
        growing spiral (the typical case in your findings)
    expansion_ratio < 1, log_r_slope < 0:
        contracting spiral

  These are computed per (position, unit). We can then ask whether any
  units are non-expanding (or even contracting), and whether expansion
  behavior differs across positions.

Outputs in --out-dir:
  - expansion_metrics.csv                per-(position, unit) table
  - expansion_distributions.png          histograms of expansion ratio
                                          and log-r slope per position
  - radius_profiles_overlay.png          r(l) curves overlaid, colored
                                          by expansion ratio, per position
  - extreme_units_phase_portraits.png    phase portraits of the most
                                          and least expanding units,
                                          across positions
  - expansion_vs_other_metrics.png       expansion ratio vs R, vs bin-1
                                          power, vs loudness
  - expansion_summary.json               aggregate summary

Usage:
  python find_non_expanding_units.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_expansion_analysis
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
# Expansion metrics
# ---------------------------------------------------------------------------

def compute_radius_profiles(streams):
    """
    For each (sample, unit), compute the radius profile r(l) where
    r is measured from the trajectory's centroid in (a, da/dl) space.

    Returns:
      r_per_sample: (n_samples, n_layers, d_model)
      r_mean:       (n_layers, d_model) mean across samples
    """
    a = streams                                  # (N, L, D)
    da = np.gradient(a, axis=1)                  # (N, L, D)
    a_centered = a - a.mean(axis=1, keepdims=True)
    da_centered = da - da.mean(axis=1, keepdims=True)
    r = np.sqrt(a_centered ** 2 + da_centered ** 2)
    r_mean = r.mean(axis=0)
    return r, r_mean


def expansion_metrics(r_mean):
    """
    r_mean: (n_layers, d_model)
    Returns dict of per-unit (D,) arrays:
      expansion_ratio: mean r in second half / mean r in first half
      log_r_slope:     slope of log(r) vs layer
      r_max_over_min:  max(r) / min(r)
      r_first_mean, r_second_mean
    """
    n_layers, d_model = r_mean.shape
    half = n_layers // 2
    r_first_mean = r_mean[:half].mean(axis=0)
    r_second_mean = r_mean[half:].mean(axis=0)
    eps = 1e-12
    expansion_ratio = r_second_mean / (r_first_mean + eps)

    # log-r slope by least squares on each unit
    layers = np.arange(n_layers, dtype=np.float64)
    log_r = np.log(r_mean + eps)
    layers_centered = layers - layers.mean()
    denom = (layers_centered ** 2).sum()
    log_r_centered = log_r - log_r.mean(axis=0, keepdims=True)
    log_r_slope = (
        (layers_centered[:, None] * log_r_centered).sum(axis=0) / denom
    )

    r_max_over_min = r_mean.max(axis=0) / (r_mean.min(axis=0) + eps)

    return {
        "expansion_ratio":  expansion_ratio,
        "log_r_slope":      log_r_slope,
        "r_max_over_min":   r_max_over_min,
        "r_first_mean":     r_first_mean,
        "r_second_mean":    r_second_mean,
    }


def per_unit_bin1_power(streams, freq_bin=1):
    spectra = per_unit_spectra(streams)
    return (np.abs(spectra[:, freq_bin, :]) ** 2).mean(axis=0)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_expansion_distributions(metrics_by_pos, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for pos in POSITIONS:
        m = metrics_by_pos[pos]["expansion"]
        ax.hist(
            m["expansion_ratio"], bins=30,
            histtype="step", linewidth=1.6,
            color=POSITION_COLORS[pos],
            label=(f"{POSITION_LABELS[pos]} "
                   f"(median={np.median(m['expansion_ratio']):.2f}, "
                   f"min={m['expansion_ratio'].min():.2f})"),
        )
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--",
               label="ratio = 1 (no expansion)")
    ax.set_xlabel("expansion ratio  (r_second_half / r_first_half)")
    ax.set_ylabel("count of units")
    ax.set_title("Per-unit expansion ratio")
    ax.legend(fontsize=8)

    ax = axes[1]
    for pos in POSITIONS:
        m = metrics_by_pos[pos]["expansion"]
        ax.hist(
            m["log_r_slope"], bins=30,
            histtype="step", linewidth=1.6,
            color=POSITION_COLORS[pos],
            label=(f"{POSITION_LABELS[pos]} "
                   f"(median={np.median(m['log_r_slope']):.3f}, "
                   f"min={m['log_r_slope'].min():.3f})"),
        )
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--",
               label="slope = 0 (no expansion)")
    ax.set_xlabel("log r vs layer slope (per layer)")
    ax.set_ylabel("count of units")
    ax.set_title("Per-unit log-radius slope")
    ax.legend(fontsize=8)

    fig.suptitle("Expansion metric distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_radius_profiles_overlay(metrics_by_pos, save_path):
    """For each position, draw r(l) for every unit, colored by
    expansion ratio. Highlights the non-expanders in cool colors."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, pos in zip(axes, POSITIONS):
        r_mean = metrics_by_pos[pos]["r_mean"]
        ratios = metrics_by_pos[pos]["expansion"]["expansion_ratio"]
        # Use log of ratio for color scale (symmetric around 0)
        log_ratio = np.log(ratios)
        v = np.max(np.abs(log_ratio))
        cmap = plt.colormaps.get_cmap("coolwarm")
        layers = np.arange(r_mean.shape[0])
        for u in range(r_mean.shape[1]):
            color = cmap(0.5 + 0.5 * log_ratio[u] / (v + 1e-12))
            ax.plot(layers, r_mean[:, u], "-",
                    color=color, linewidth=0.8, alpha=0.7)
        ax.set_xlabel("sublayer index")
        ax.set_title(POSITION_LABELS[pos], fontsize=11)
    axes[0].set_ylabel("mean radius r(l)")
    fig.suptitle(
        "Radius profile per unit (cool colors: low expansion; "
        "warm colors: high expansion)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_radius_profiles_log_overlay(metrics_by_pos, save_path):
    """Same but log y-axis. A growing spiral with constant rate is a
    straight line in log-r vs layer; non-expanding shows up as flat."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, pos in zip(axes, POSITIONS):
        r_mean = metrics_by_pos[pos]["r_mean"]
        ratios = metrics_by_pos[pos]["expansion"]["expansion_ratio"]
        log_ratio = np.log(ratios)
        v = np.max(np.abs(log_ratio))
        cmap = plt.colormaps.get_cmap("coolwarm")
        layers = np.arange(r_mean.shape[0])
        for u in range(r_mean.shape[1]):
            color = cmap(0.5 + 0.5 * log_ratio[u] / (v + 1e-12))
            ax.plot(layers, r_mean[:, u], "-",
                    color=color, linewidth=0.8, alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel("sublayer index")
        ax.set_title(POSITION_LABELS[pos], fontsize=11)
    axes[0].set_ylabel("mean radius r(l)  (log scale)")
    fig.suptitle(
        "Radius profile per unit, log y-axis "
        "(flat = non-expanding, sloped = expanding)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_extreme_units_phase_portraits(
    metrics_by_pos, streams_by_pos, n_per_group, sample_idx, save_path
):
    """Phase portraits of the n_per_group least-expanding and
    n_per_group most-expanding units (selected by position 5
    expansion ratio), shown across all three positions."""
    pos5_metrics = metrics_by_pos[5]["expansion"]
    order = np.argsort(pos5_metrics["expansion_ratio"])
    least_expanding = order[:n_per_group].tolist()
    most_expanding = order[-n_per_group:][::-1].tolist()

    units = least_expanding + most_expanding
    group_labels = (
        ["non-expanding"] * n_per_group +
        ["expanding"] * n_per_group
    )

    n_units = len(units)
    fig, axes = plt.subplots(
        n_units, len(POSITIONS),
        figsize=(3.6 * len(POSITIONS), 2.8 * n_units),
        squeeze=False,
    )
    for row, (u, group) in enumerate(zip(units, group_labels)):
        for col, pos in enumerate(POSITIONS):
            ax = axes[row, col]
            streams = streams_by_pos[pos]
            sample = streams[sample_idx]
            n_steps = sample.shape[0]
            a = sample[:, u]
            ga = np.gradient(a)
            x = a - a.mean()
            y = ga - ga.mean()
            ax.plot(x, y, "-", linewidth=0.7, alpha=0.9)
            ax.scatter(x, y, c=np.arange(n_steps),
                       cmap="viridis", s=12)
            ax.scatter([0], [0], marker="x", color="red", s=30, zorder=5)
            if row == 0:
                ax.set_title(POSITION_LABELS[pos], fontsize=11)
            if col == 0:
                ratio_5 = metrics_by_pos[5]["expansion"]["expansion_ratio"][u]
                ratio_6 = metrics_by_pos[6]["expansion"]["expansion_ratio"][u]
                ratio_7 = metrics_by_pos[7]["expansion"]["expansion_ratio"][u]
                ax.set_ylabel(
                    f"{group}\n"
                    f"unit {u}\n"
                    f"ratio=[{ratio_5:.2f}, {ratio_6:.2f}, {ratio_7:.2f}]",
                    fontsize=8,
                )
            ax.set_xlabel("a - mean(a)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="gray", linewidth=0.4)
            ax.axvline(0, color="gray", linewidth=0.4)
    fig.suptitle(
        f"Phase portraits: least vs most expanding units "
        f"(selected by pos 5 expansion ratio, sample idx {sample_idx})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_expansion_vs_other_metrics(metrics_by_pos, save_path):
    fig, axes = plt.subplots(len(POSITIONS), 3,
                             figsize=(15, 4.5 * len(POSITIONS)),
                             squeeze=False)
    for row, pos in enumerate(POSITIONS):
        m = metrics_by_pos[pos]
        ratio = m["expansion"]["expansion_ratio"]
        R = m["R"]
        bin1_power = m["bin1_power"]
        loud = m["loudness"]

        ax = axes[row, 0]
        ax.scatter(ratio, np.abs(R), s=14, alpha=0.6,
                   c=POSITION_COLORS[pos])
        ax.axvline(1.0, color="black", linewidth=0.6, linestyle=":")
        ax.set_xlabel("expansion ratio")
        ax.set_ylabel("|R| (rotation count)")
        ax.set_title(f"{POSITION_LABELS[pos]}: |R| vs expansion",
                     fontsize=10)

        ax = axes[row, 1]
        ax.scatter(ratio, bin1_power, s=14, alpha=0.6,
                   c=POSITION_COLORS[pos])
        ax.axvline(1.0, color="black", linewidth=0.6, linestyle=":")
        ax.set_yscale("log")
        ax.set_xlabel("expansion ratio")
        ax.set_ylabel("bin-1 power (log)")
        ax.set_title(f"{POSITION_LABELS[pos]}: power vs expansion",
                     fontsize=10)

        ax = axes[row, 2]
        ax.scatter(ratio, loud, s=14, alpha=0.6,
                   c=POSITION_COLORS[pos])
        ax.axvline(1.0, color="black", linewidth=0.6, linestyle=":")
        ax.set_xlabel("expansion ratio")
        ax.set_ylabel("mean |activation| (loudness)")
        ax.set_title(f"{POSITION_LABELS[pos]}: loudness vs expansion",
                     fontsize=10)
    fig.suptitle(
        "Per-unit expansion vs rotation, power, loudness",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV and summary
# ---------------------------------------------------------------------------

def write_csv(metrics_by_pos, save_path):
    rows = []
    d_model = len(metrics_by_pos[5]["expansion"]["expansion_ratio"])
    for u in range(d_model):
        row = {"unit": u}
        for pos in POSITIONS:
            m = metrics_by_pos[pos]
            row[f"expansion_ratio_pos{pos}"] = float(
                m["expansion"]["expansion_ratio"][u]
            )
            row[f"log_r_slope_pos{pos}"] = float(
                m["expansion"]["log_r_slope"][u]
            )
            row[f"r_first_mean_pos{pos}"] = float(
                m["expansion"]["r_first_mean"][u]
            )
            row[f"r_second_mean_pos{pos}"] = float(
                m["expansion"]["r_second_mean"][u]
            )
            row[f"R_pos{pos}"] = float(m["R"][u])
            row[f"bin1_power_pos{pos}"] = float(m["bin1_power"][u])
            row[f"loudness_pos{pos}"] = float(m["loudness"][u])
        rows.append(row)
    if rows:
        with open(save_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)
    return rows


def categorize(ratio, low_thresh=1.1, contract_thresh=0.95):
    if ratio < contract_thresh:
        return "contracting"
    if ratio < low_thresh:
        return "non_expanding"
    return "expanding"


def build_summary(metrics_by_pos, low_thresh=1.1, contract_thresh=0.95):
    out = {
        "thresholds": {
            "non_expanding_below": low_thresh,
            "contracting_below":   contract_thresh,
        },
        "per_position": {},
    }
    for pos in POSITIONS:
        ratio = metrics_by_pos[pos]["expansion"]["expansion_ratio"]
        slope = metrics_by_pos[pos]["expansion"]["log_r_slope"]
        cats = [categorize(r, low_thresh, contract_thresh) for r in ratio]
        from collections import Counter
        counts = Counter(cats)
        non_exp_units = [
            int(u) for u, c in enumerate(cats) if c != "expanding"
        ]
        out["per_position"][str(pos)] = {
            "label": POSITION_LABELS[pos],
            "expansion_ratio": {
                "mean":   float(ratio.mean()),
                "median": float(np.median(ratio)),
                "min":    float(ratio.min()),
                "max":    float(ratio.max()),
                "p10":    float(np.percentile(ratio, 10)),
                "p90":    float(np.percentile(ratio, 90)),
            },
            "log_r_slope": {
                "mean":   float(slope.mean()),
                "median": float(np.median(slope)),
                "min":    float(slope.min()),
                "max":    float(slope.max()),
            },
            "category_counts": dict(counts),
            "non_expanding_or_contracting_units": non_exp_units,
        }
    # Cross-position consistency: which units are non-expanding at all 3?
    cats_by_pos = {
        pos: [categorize(r) for r in
              metrics_by_pos[pos]["expansion"]["expansion_ratio"]]
        for pos in POSITIONS
    }
    d_model = len(cats_by_pos[5])
    consistent_non_expanders = [
        u for u in range(d_model)
        if all(cats_by_pos[p][u] != "expanding" for p in POSITIONS)
    ]
    consistent_expanders = [
        u for u in range(d_model)
        if all(cats_by_pos[p][u] == "expanding" for p in POSITIONS)
    ]
    out["cross_position"] = {
        "consistent_non_expanders": consistent_non_expanders,
        "consistent_expanders_count": len(consistent_expanders),
    }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str,
                        default="toy_expansion_analysis")
    parser.add_argument("--n-per-group", type=int, default=4,
                        help="How many least-expanding and most-expanding "
                             "units to show in phase portrait grid")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--non-expanding-thresh", type=float, default=1.1,
                        help="expansion_ratio below this is considered "
                             "non-expanding")
    parser.add_argument("--contracting-thresh", type=float, default=0.95,
                        help="expansion_ratio below this is considered "
                             "contracting")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("\nCollecting cumulative streams ...")
    streams_by_pos, sublayer_meta = collect_cumulative_streams(model, cfg)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nComputing radius profiles and expansion metrics ...")
    metrics_by_pos = {}
    for pos in POSITIONS:
        print(f"  pos {pos} ...")
        streams = streams_by_pos[pos]
        _, r_mean = compute_radius_profiles(streams)
        exp_metrics = expansion_metrics(r_mean)
        metrics_by_pos[pos] = {
            "r_mean":     r_mean,
            "expansion":  exp_metrics,
            "R":          rotation_count_per_unit(streams),
            "bin1_power": per_unit_bin1_power(streams),
            "loudness":   np.abs(streams).mean(axis=(0, 1)),
        }

    print("\nMaking plots ...")
    plot_expansion_distributions(
        metrics_by_pos, out_dir / "expansion_distributions.png"
    )
    plot_radius_profiles_overlay(
        metrics_by_pos, out_dir / "radius_profiles_overlay.png"
    )
    plot_radius_profiles_log_overlay(
        metrics_by_pos, out_dir / "radius_profiles_log_overlay.png"
    )
    plot_extreme_units_phase_portraits(
        metrics_by_pos, streams_by_pos,
        args.n_per_group, args.sample_idx,
        out_dir / "extreme_units_phase_portraits.png",
    )
    plot_expansion_vs_other_metrics(
        metrics_by_pos, out_dir / "expansion_vs_other_metrics.png"
    )

    print("\nWriting CSV ...")
    write_csv(metrics_by_pos, out_dir / "expansion_metrics.csv")

    print("\nBuilding summary ...")
    summary = build_summary(
        metrics_by_pos,
        low_thresh=args.non_expanding_thresh,
        contract_thresh=args.contracting_thresh,
    )
    with open(out_dir / "expansion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print headline stats
    print("\n--- Expansion summary ---")
    print(f"Categorization thresholds: contracting < "
          f"{args.contracting_thresh}, "
          f"non_expanding < {args.non_expanding_thresh} "
          f"(else expanding).\n")
    for pos in POSITIONS:
        s = summary["per_position"][str(pos)]
        print(f"{POSITION_LABELS[pos]}:")
        print(f"  expansion ratio:  median={s['expansion_ratio']['median']:.3f}, "
              f"min={s['expansion_ratio']['min']:.3f}, "
              f"max={s['expansion_ratio']['max']:.3f}")
        print(f"  log-r slope:      median={s['log_r_slope']['median']:.3f}")
        print(f"  category counts: {s['category_counts']}")
        nx = s['non_expanding_or_contracting_units']
        if nx:
            print(f"  non-expanding/contracting units ({len(nx)}): "
                  f"{nx}")
        print()

    print(f"Units that are non-expanding at ALL three positions: "
          f"{summary['cross_position']['consistent_non_expanders']}")
    print(f"Units that are expanding at ALL three positions: "
          f"{summary['cross_position']['consistent_expanders_count']} "
          f"(of {cfg.d_model})")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
