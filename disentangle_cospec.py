"""
Disentangle why cospec magnitude differs across positions 5, 6, 7.

cospec magnitude at bin 1 conflates two things:

  1. Per-unit bin-1 POWER: how much each unit oscillates at the slow
     frequency. If positions 6 and 7 just have less depth-extensive
     processing, every unit's bin-1 power drops, and pairwise cospec
     drops too even with identical phase relationships.

  2. Phase CONSISTENCY across input pairs: even with identical power
     levels, if the phase relationship between two units is consistent
     across inputs at one position but variable at another, the cospec
     magnitude (which is the magnitude of an *average over samples*) is
     larger at the consistent position.

We compute both directly:

  Per-unit:
    bin1_power(u)    = mean_s |spec_u[1]|^2
    bin1_mag(u)      = mean_s |spec_u[1]|

  Per-pair:
    cospec_mag(u,v)  = |mean_s spec_u[1] * conj(spec_v[1])|       (the original)
    amp_prod(u,v)    = mean_s |spec_u[1]| * |spec_v[1]|           (potential mag)
    plv(u,v)         = |mean_s exp(i * angle(spec_u * conj(spec_v)))|
                                                                  (pure consistency)

The exact relationship is:
    cospec_mag = amp_prod * weighted_consistency
where weighted_consistency = |mean_s amp_prod_per_sample * exp(i*phase_diff)|
                            / mean_s amp_prod_per_sample
weighted_consistency is between 0 and 1, equal to 1 iff phase is exactly
constant across samples. PLV is the unweighted version and is usually
close to weighted_consistency but easier to interpret.

Key questions answered:

  Q1. Does per-unit bin-1 power drop from pos 5 to pos 6, 7?
      -> if yes, part of the cospec drop is "less oscillation"
  Q2. Does PLV drop from pos 5 to pos 6, 7?
      -> if yes, part of the cospec drop is "less consistent phase"
  Q3. For the top-100 pairs at pos 5, how do their cospec, amp_prod,
      and PLV change at pos 6, 7?
      -> apportions the cospec drop into the two factors
  Q4. Are the top-100 pairs at pos 5 also top at pos 6, 7?
      -> tells us whether pos 6, 7 carry attenuated-but-same structure,
         or different structure altogether

Outputs in --out-dir:
  - per_unit_power_comparison.png
  - plv_distribution_comparison.png
  - amp_prod_distribution_comparison.png
  - cospec_vs_plv_scatter.png
  - decomposition_top100_at_pos5.png
  - top_pair_overlap_jaccard.png
  - top100_by_cospec_overlap.csv     (which pairs make top-100 at which positions)
  - disentangle_summary.json
"""

import argparse
import json
from collections import Counter
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
# Disentanglement metrics
# ---------------------------------------------------------------------------

def compute_metrics(streams, freq_bin=1):
    """
    Returns a dict containing everything we need for the comparison.
    Per-unit arrays are length D, per-pair arrays are length n_pairs
    (with n_pairs = D*(D-1)/2).
    """
    spectra = per_unit_spectra(streams)               # (N, n_freq, D), complex
    n_samples, n_freq, D = spectra.shape
    spec_b = spectra[:, freq_bin, :]                  # (N, D), complex

    # Per-unit
    bin1_power = (np.abs(spec_b) ** 2).mean(axis=0)   # (D,)
    bin1_mag   = np.abs(spec_b).mean(axis=0)          # (D,)

    # Per-pair
    u_idx, v_idx = all_pair_indices(D)
    u_spec = spec_b[:, u_idx]                         # (N, n_pairs)
    v_spec = spec_b[:, v_idx]                         # (N, n_pairs)

    cs = u_spec * np.conj(v_spec)                     # (N, n_pairs), complex
    cs_mean = cs.mean(axis=0)                         # (n_pairs,), complex
    cospec_mag = np.abs(cs_mean)
    cospec_phase = np.angle(cs_mean)

    # amplitude product (sample-by-sample, then averaged)
    amp_per_sample = np.abs(u_spec) * np.abs(v_spec)  # (N, n_pairs)
    amp_prod = amp_per_sample.mean(axis=0)            # (n_pairs,)

    # PLV: pure phase consistency, ignores amplitudes
    eps = 1e-12
    cs_unit = cs / (np.abs(cs) + eps)                 # unit complex per sample
    plv = np.abs(cs_unit.mean(axis=0))                # (n_pairs,)

    # Weighted consistency: cospec_mag = amp_prod * weighted_consistency exactly.
    weighted_consistency = cospec_mag / (amp_prod + eps)

    R = rotation_count_per_unit(streams)
    loudness = np.abs(streams).mean(axis=(0, 1))

    return {
        "freq_bin":             freq_bin,
        "bin1_power":           bin1_power,
        "bin1_mag":             bin1_mag,
        "u_idx":                u_idx,
        "v_idx":                v_idx,
        "cospec_mag":           cospec_mag,
        "cospec_phase":         cospec_phase,
        "amp_prod":             amp_prod,
        "plv":                  plv,
        "weighted_consistency": weighted_consistency,
        "R":                    R,
        "loudness":             loudness,
    }


def classify_phase(phi, band=QUAD_BAND):
    if abs(phi) < band:
        return "near_0"
    if abs(phi - np.pi / 2) < band:
        return "near_pi_over_2"
    if abs(phi + np.pi / 2) < band:
        return "near_minus_pi_over_2"
    if (abs(phi - np.pi) < band) or (abs(phi + np.pi) < band):
        return "near_pi"
    return "between"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_per_unit_power(metrics_by_pos, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear-scale histogram of bin-1 magnitude
    ax = axes[0]
    for pos in POSITIONS:
        m = metrics_by_pos[pos]
        ax.hist(
            m["bin1_mag"], bins=30, density=True,
            histtype="step", linewidth=1.6,
            color=POSITION_COLORS[pos],
            label=(f"{POSITION_LABELS[pos]} "
                   f"(mean={m['bin1_mag'].mean():.3f}, "
                   f"med={np.median(m['bin1_mag']):.3f})"),
        )
    ax.set_xlabel("per-unit bin-1 magnitude  mean_s |spec_u[1]|")
    ax.set_ylabel("density")
    ax.set_title("Per-unit bin-1 magnitude distribution")
    ax.legend(fontsize=8)

    # Log-scale histogram of bin-1 power
    ax = axes[1]
    for pos in POSITIONS:
        m = metrics_by_pos[pos]
        log_p = np.log10(m["bin1_power"] + 1e-12)
        ax.hist(
            log_p, bins=30, density=True,
            histtype="step", linewidth=1.6,
            color=POSITION_COLORS[pos],
            label=(f"{POSITION_LABELS[pos]} "
                   f"(mean log10 P={log_p.mean():.2f})"),
        )
    ax.set_xlabel("log10 per-unit bin-1 power")
    ax.set_ylabel("density")
    ax.set_title("Per-unit bin-1 power distribution (log)")
    ax.legend(fontsize=8)

    fig.suptitle("Q1: Does per-unit bin-1 power drop from pos 5 to 6, 7?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_plv_distribution(metrics_by_pos, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # All pairs
    ax = axes[0]
    for pos in POSITIONS:
        m = metrics_by_pos[pos]
        ax.hist(
            m["plv"], bins=40, density=True, range=(0, 1),
            histtype="step", linewidth=1.6,
            color=POSITION_COLORS[pos],
            label=(f"{POSITION_LABELS[pos]} "
                   f"(mean={m['plv'].mean():.3f}, "
                   f"p95={np.percentile(m['plv'], 95):.3f})"),
        )
    ax.set_xlabel("PLV (phase locking value)")
    ax.set_ylabel("density")
    ax.set_title("All pairs")
    ax.legend(fontsize=8)

    # High-amp pairs only (top 25% by amp_prod, to remove noise floor)
    ax = axes[1]
    for pos in POSITIONS:
        m = metrics_by_pos[pos]
        thresh = np.percentile(m["amp_prod"], 75)
        keep = m["amp_prod"] >= thresh
        ax.hist(
            m["plv"][keep], bins=40, density=True, range=(0, 1),
            histtype="step", linewidth=1.6,
            color=POSITION_COLORS[pos],
            label=(f"{POSITION_LABELS[pos]} "
                   f"(n={keep.sum()}, "
                   f"mean PLV={m['plv'][keep].mean():.3f})"),
        )
    ax.set_xlabel("PLV (phase locking value)")
    ax.set_ylabel("density")
    ax.set_title("Pairs with amp_prod above its 75th percentile")
    ax.legend(fontsize=8)

    fig.suptitle("Q2: Does phase consistency drop, even at fixed amplitude?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_amp_prod_distribution(metrics_by_pos, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for pos in POSITIONS:
        m = metrics_by_pos[pos]
        log_a = np.log10(m["amp_prod"] + 1e-12)
        ax.hist(
            log_a, bins=40, density=True,
            histtype="step", linewidth=1.6,
            color=POSITION_COLORS[pos],
            label=(f"{POSITION_LABELS[pos]} "
                   f"(median={np.median(m['amp_prod']):.4f})"),
        )
    ax.set_xlabel("log10 mean amplitude product mean_s |spec_u||spec_v|")
    ax.set_ylabel("density")
    ax.set_title("Pairwise amplitude product (the upper bound on cospec_mag)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_cospec_vs_plv_scatter(metrics_by_pos, save_path):
    """For each position, scatter cospec_mag vs PLV, with point size
    proportional to amp_prod. Lets you read off the joint distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, pos in zip(axes, POSITIONS):
        m = metrics_by_pos[pos]
        # Subsample if too many for plotting
        idx = np.arange(len(m["plv"]))
        if len(idx) > 5000:
            idx = np.random.default_rng(SEED).choice(
                idx, size=5000, replace=False
            )
        sizes = 4 + 60 * (m["amp_prod"][idx] / m["amp_prod"].max())
        ax.scatter(
            m["plv"][idx], m["cospec_mag"][idx],
            s=sizes, alpha=0.3, c=POSITION_COLORS[pos],
            edgecolors="none",
        )
        ax.set_xlabel("PLV (phase consistency)")
        ax.set_xlim(0, 1)
        ax.set_title(POSITION_LABELS[pos], fontsize=11)
    axes[0].set_ylabel("cospec magnitude")
    fig.suptitle(
        "cospec_mag vs PLV (point size proportional to amp_prod)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_decomposition_top100(metrics_by_pos, top_n, save_path):
    """For the top-N pairs at pos 5 (by cospec_mag), track their
    cospec_mag, amp_prod, and PLV at all three positions.
    Shows whether the cospec drop is power or consistency."""
    pos5 = metrics_by_pos[5]
    top_idx = np.argsort(-pos5["cospec_mag"])[:top_n]

    metrics_to_show = [
        ("cospec_mag", "cospec magnitude"),
        ("amp_prod",   "mean amplitude product"),
        ("plv",        "PLV (phase consistency)"),
        ("weighted_consistency",
         "weighted consistency = cospec / amp_prod"),
    ]
    fig, axes = plt.subplots(1, len(metrics_to_show),
                             figsize=(5 * len(metrics_to_show), 5))

    for ax, (key, title) in zip(axes, metrics_to_show):
        data_per_pos = [
            metrics_by_pos[pos][key][top_idx] for pos in POSITIONS
        ]
        bp = ax.boxplot(
            data_per_pos,
            tick_labels=[POSITION_LABELS[p] for p in POSITIONS],
            showfliers=True,
            patch_artist=True,
        )
        for patch, pos in zip(bp["boxes"], POSITIONS):
            patch.set_facecolor(POSITION_COLORS[pos])
            patch.set_alpha(0.5)
        for i, vals in enumerate(data_per_pos):
            ax.text(
                i + 1, np.median(vals),
                f"med={np.median(vals):.3f}",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Q3: For the top-{top_n} pairs at pos 5, how do the three "
        f"components track across positions?",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_top_pair_overlap_jaccard(metrics_by_pos, save_path):
    """Jaccard overlap of top-N pair sets across positions, as a
    function of N, for two ranking metrics: cospec_mag and PLV."""
    n_pairs_total = len(metrics_by_pos[5]["cospec_mag"])
    Ns = sorted(set(
        [n for n in [25, 50, 100, 200, 400, 800] if n <= n_pairs_total]
    ))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric_key, metric_label in [
        (axes[0], "cospec_mag", "ranked by cospec_mag"),
        (axes[1], "plv",        "ranked by PLV"),
    ]:
        # For each pair of positions, plot Jaccard vs N
        for (a, b) in [(5, 6), (5, 7), (6, 7)]:
            jaccards = []
            for N in Ns:
                top_a = set(np.argsort(-metrics_by_pos[a][metric_key])[:N])
                top_b = set(np.argsort(-metrics_by_pos[b][metric_key])[:N])
                inter = top_a & top_b
                union = top_a | top_b
                jaccards.append(len(inter) / len(union) if union else 0.0)
            ax.plot(
                Ns, jaccards, "o-",
                label=f"pos {a} vs pos {b}",
                linewidth=1.6, markersize=6,
            )

        # Reference: random overlap. For random sets of size N drawn from
        # n_pairs_total, expected Jaccard ~= N / (2 * n_pairs_total - N)
        ref = [N / (2 * n_pairs_total - N) for N in Ns]
        ax.plot(
            Ns, ref, "k--", linewidth=0.8,
            label="random baseline",
        )

        ax.set_xlabel("top-N (logarithmic)")
        ax.set_xscale("log")
        ax.set_ylabel("Jaccard overlap")
        ax.set_title(metric_label, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Q4: Are the top pairs the same across positions, "
        "or are positions 6, 7 separate structures?",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output table for cross-position pair overlap
# ---------------------------------------------------------------------------

def write_top100_overlap_csv(metrics_by_pos, top_n, save_path):
    """For the union of top-N pairs at any of the three positions,
    write a row showing each pair's metrics at every position."""
    union_idx = set()
    for pos in POSITIONS:
        top = np.argsort(-metrics_by_pos[pos]["cospec_mag"])[:top_n]
        union_idx.update(top.tolist())
    union_idx = sorted(union_idx)

    pos5 = metrics_by_pos[5]
    rows = []
    for idx in union_idx:
        u = int(pos5["u_idx"][idx])
        v = int(pos5["v_idx"][idx])
        ranks = {}
        for pos in POSITIONS:
            order = np.argsort(-metrics_by_pos[pos]["cospec_mag"])
            rank_map = np.empty(len(order), dtype=int)
            rank_map[order] = np.arange(len(order))
            ranks[pos] = int(rank_map[idx]) + 1
        row = {
            "u": u, "v": v,
            "rank_pos5":   ranks[5],
            "rank_pos6":   ranks[6],
            "rank_pos7":   ranks[7],
            "in_top_pos5": ranks[5] <= top_n,
            "in_top_pos6": ranks[6] <= top_n,
            "in_top_pos7": ranks[7] <= top_n,
        }
        for pos in POSITIONS:
            m = metrics_by_pos[pos]
            row[f"cospec_pos{pos}"]   = float(m["cospec_mag"][idx])
            row[f"plv_pos{pos}"]      = float(m["plv"][idx])
            row[f"amp_prod_pos{pos}"] = float(m["amp_prod"][idx])
            row[f"phase_pos{pos}"]    = float(m["cospec_phase"][idx])
            row[f"cluster_pos{pos}"]  = classify_phase(
                m["cospec_phase"][idx]
            )
        rows.append(row)

    rows.sort(key=lambda r: r["rank_pos5"])
    if rows:
        import csv
        with open(save_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(metrics_by_pos, top_n, overlap_rows):
    n_pairs_total = len(metrics_by_pos[5]["cospec_mag"])

    # Top-N overlap counts (not Jaccard, just intersection size)
    overlap_counts = {}
    for metric in ("cospec_mag", "plv"):
        overlap_counts[metric] = {}
        tops = {
            pos: set(
                np.argsort(-metrics_by_pos[pos][metric])[:top_n].tolist()
            )
            for pos in POSITIONS
        }
        for (a, b) in [(5, 6), (5, 7), (6, 7)]:
            inter = tops[a] & tops[b]
            overlap_counts[metric][f"pos{a}_pos{b}"] = {
                "intersection":  len(inter),
                "union":         len(tops[a] | tops[b]),
                "jaccard":       len(inter) / len(tops[a] | tops[b])
                                  if (tops[a] | tops[b]) else 0.0,
            }
        triple = tops[5] & tops[6] & tops[7]
        overlap_counts[metric]["triple_intersection"] = len(triple)

    # Per-position aggregates
    agg = {}
    for pos in POSITIONS:
        m = metrics_by_pos[pos]
        agg[str(pos)] = {
            "label": POSITION_LABELS[pos],
            "per_unit_bin1_mag": {
                "mean":   float(m["bin1_mag"].mean()),
                "median": float(np.median(m["bin1_mag"])),
                "p95":    float(np.percentile(m["bin1_mag"], 95)),
            },
            "per_unit_bin1_power": {
                "mean":   float(m["bin1_power"].mean()),
                "median": float(np.median(m["bin1_power"])),
                "p95":    float(np.percentile(m["bin1_power"], 95)),
            },
            "cospec_mag": {
                "mean":   float(m["cospec_mag"].mean()),
                "median": float(np.median(m["cospec_mag"])),
                "p95":    float(np.percentile(m["cospec_mag"], 95)),
                "max":    float(m["cospec_mag"].max()),
            },
            "amp_prod": {
                "mean":   float(m["amp_prod"].mean()),
                "median": float(np.median(m["amp_prod"])),
            },
            "plv_all_pairs": {
                "mean":   float(m["plv"].mean()),
                "median": float(np.median(m["plv"])),
                "p95":    float(np.percentile(m["plv"], 95)),
            },
        }
        thresh = np.percentile(m["amp_prod"], 75)
        keep = m["amp_prod"] >= thresh
        agg[str(pos)]["plv_high_amp_pairs_only"] = {
            "n":      int(keep.sum()),
            "mean":   float(m["plv"][keep].mean()),
            "median": float(np.median(m["plv"][keep])),
        }

    # Top-N at pos 5 tracking
    pos5 = metrics_by_pos[5]
    top5 = np.argsort(-pos5["cospec_mag"])[:top_n]
    top5_tracking = {}
    for pos in POSITIONS:
        m = metrics_by_pos[pos]
        top5_tracking[str(pos)] = {
            "cospec_mag_median":           float(np.median(m["cospec_mag"][top5])),
            "amp_prod_median":             float(np.median(m["amp_prod"][top5])),
            "plv_median":                  float(np.median(m["plv"][top5])),
            "weighted_consistency_median": float(np.median(m["weighted_consistency"][top5])),
        }

    # Cluster counts in top-N at each position (by cospec_mag)
    cluster_counts = {}
    for pos in POSITIONS:
        m = metrics_by_pos[pos]
        top = np.argsort(-m["cospec_mag"])[:top_n]
        clusters = [classify_phase(m["cospec_phase"][i]) for i in top]
        cluster_counts[str(pos)] = dict(Counter(clusters))

    return {
        "n_pairs_total":        n_pairs_total,
        "top_n":                top_n,
        "per_position":         agg,
        "top_n_at_pos5_tracked_across_positions": top5_tracking,
        "top_n_overlap_counts": overlap_counts,
        "top_n_cluster_counts_by_position":     cluster_counts,
        "overlap_csv_n_rows":   len(overlap_rows),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="toy_disentangle")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("\nCollecting cumulative streams at positions 5, 6, 7 ...")
    streams_by_pos, sublayer_meta = collect_cumulative_streams(model, cfg)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nComputing per-position metrics ...")
    metrics_by_pos = {}
    for pos in POSITIONS:
        print(f"  pos {pos} ...")
        metrics_by_pos[pos] = compute_metrics(streams_by_pos[pos])

    print("\nMaking plots ...")
    plot_per_unit_power(
        metrics_by_pos, out_dir / "per_unit_power_comparison.png"
    )
    plot_plv_distribution(
        metrics_by_pos, out_dir / "plv_distribution_comparison.png"
    )
    plot_amp_prod_distribution(
        metrics_by_pos, out_dir / "amp_prod_distribution_comparison.png"
    )
    plot_cospec_vs_plv_scatter(
        metrics_by_pos, out_dir / "cospec_vs_plv_scatter.png"
    )
    plot_decomposition_top100(
        metrics_by_pos, args.top_n,
        out_dir / "decomposition_top100_at_pos5.png",
    )
    plot_top_pair_overlap_jaccard(
        metrics_by_pos, out_dir / "top_pair_overlap_jaccard.png"
    )

    print("\nWriting overlap CSV ...")
    overlap_rows = write_top100_overlap_csv(
        metrics_by_pos, args.top_n,
        out_dir / f"top{args.top_n}_by_cospec_overlap.csv",
    )
    print(f"  wrote {len(overlap_rows)} rows")

    summary = build_summary(metrics_by_pos, args.top_n, overlap_rows)
    with open(out_dir / "disentangle_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print a human-readable summary table
    print("\n--- Disentanglement summary ---")
    print(f"\nPer-unit bin-1 magnitude (mean across units):")
    for pos in POSITIONS:
        v = summary["per_position"][str(pos)]["per_unit_bin1_mag"]["mean"]
        print(f"  {POSITION_LABELS[pos]:>14}: {v:.4f}")

    print(f"\nPLV (mean over all pairs):")
    for pos in POSITIONS:
        v = summary["per_position"][str(pos)]["plv_all_pairs"]["mean"]
        print(f"  {POSITION_LABELS[pos]:>14}: {v:.4f}")

    print(f"\nPLV (mean over high-amp pairs only):")
    for pos in POSITIONS:
        v = summary["per_position"][str(pos)]["plv_high_amp_pairs_only"]["mean"]
        print(f"  {POSITION_LABELS[pos]:>14}: {v:.4f}")

    print(f"\ncospec_mag (median over all pairs):")
    for pos in POSITIONS:
        v = summary["per_position"][str(pos)]["cospec_mag"]["median"]
        print(f"  {POSITION_LABELS[pos]:>14}: {v:.5f}")

    print(f"\nTop-{args.top_n} at pos 5: median values when tracked across positions:")
    print(f"  {'metric':<22}  "
          + "  ".join(f"{POSITION_LABELS[p]:>14}" for p in POSITIONS))
    for key in ("cospec_mag_median", "amp_prod_median",
                "plv_median", "weighted_consistency_median"):
        vals = [
            summary["top_n_at_pos5_tracked_across_positions"][str(p)][key]
            for p in POSITIONS
        ]
        print(f"  {key:<22}  "
              + "  ".join(f"{v:>14.4f}" for v in vals))

    print(f"\nTop-{args.top_n} pair overlap (intersection counts):")
    for metric in ("cospec_mag", "plv"):
        print(f"  ranked by {metric}:")
        oc = summary["top_n_overlap_counts"][metric]
        for k, v in oc.items():
            if isinstance(v, dict):
                print(f"    {k}: |inter|={v['intersection']}, "
                      f"jaccard={v['jaccard']:.3f}")
            else:
                print(f"    {k}: {v}")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
