"""
Compare the phase structure that emerges in each of the three ablation
models trained by train_ablations.py.

For each (mode, variant) combination, computes:
  - per-unit rotation count R and shuffle null
  - per-unit bin-1 magnitude (loudness in the slow oscillation)
  - pairwise cospec phase distribution (binned at 0, +/-pi/2, pi)
  - PLV restricted to high-amplitude pairs

Outputs in --out-dir:
  - training_curves.png            test sequence accuracy over steps
  - per_position_accuracy.png      final test accuracy at positions 5, 6, 7
  - phase_distribution_grid.png    cluster histograms by (mode, variant)
  - R_distribution_grid.png        rotation count distributions by (mode, variant)
  - plv_high_amp_comparison.png    high-amp PLV by (mode, variant)
  - phase_clustering_bars.png      summary bar chart of cluster fractions
  - ablation_comparison_summary.json

Usage:
  python compare_ablations.py --root ablation_run --out-dir ablation_compare
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from toy_transformer_addition import capture_residual_streams
from train_ablations import AblatedTransformer, AblatedConfig
from analyze_toy_residuals import (
    rotation_count_per_unit,
    shuffle_null_R,
    per_unit_spectra,
    all_pair_indices,
    QUAD_BAND,
    SEED,
)


ALL_MODES = ["full", "attn_only", "mlp_only"]
MODE_COLORS = {"full": "C0", "attn_only": "C1", "mlp_only": "C2"}
VARIANTS = ["cumulative", "combined_delta", "attn_delta", "mlp_delta"]
DEFAULT_POSITION = 5


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
    sublayer_meta = [(c["kind"], c["layer"]) for c in captures]
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    at_pos = stacked[:, :, position, :]
    streams = at_pos.permute(1, 0, 2).contiguous().numpy().astype(np.float32)
    return streams, sublayer_meta


def make_variants(streams):
    diffs = np.diff(streams, axis=1)
    return {
        "cumulative":     streams,
        "combined_delta": diffs,
        "attn_delta":     diffs[:, ::2, :],
        "mlp_delta":      diffs[:, 1::2, :],
    }


def is_variant_meaningful(streams_v, eps=1e-8):
    """Check whether a variant has any nontrivial signal. For attn_only,
    the mlp_delta will be exactly zero (since MLP added zero contribution)."""
    return np.any(np.abs(streams_v) > eps)


def analyze_variant(streams_v, n_shuffle, freq_bin=1):
    """Compute the per-variant metrics needed for the comparison plots."""
    if not is_variant_meaningful(streams_v):
        return None

    R = rotation_count_per_unit(streams_v)
    null_R = shuffle_null_R(
        streams_v, n_shuffle, np.random.default_rng(SEED + 1)
    )
    spectra = per_unit_spectra(streams_v)
    n_freq = spectra.shape[1]
    fb = freq_bin if freq_bin < n_freq else 0

    spec_b = spectra[:, fb, :]
    bin1_mag = np.abs(spec_b).mean(axis=0)

    u_idx, v_idx = all_pair_indices(streams_v.shape[2])
    u_spec = spec_b[:, u_idx]
    v_spec = spec_b[:, v_idx]
    cs = u_spec * np.conj(v_spec)
    cs_mean = cs.mean(axis=0)
    cospec_mag = np.abs(cs_mean)
    cospec_phase = np.angle(cs_mean)
    amp_prod = (np.abs(u_spec) * np.abs(v_spec)).mean(axis=0)

    cs_unit = cs / (np.abs(cs) + 1e-12)
    plv = np.abs(cs_unit.mean(axis=0))

    return {
        "R":            R,
        "null_R":       null_R,
        "bin1_mag":     bin1_mag,
        "cospec_mag":   cospec_mag,
        "cospec_phase": cospec_phase,
        "amp_prod":     amp_prod,
        "plv":          plv,
        "freq_bin":     int(fb),
        "u_idx":        u_idx,
        "v_idx":        v_idx,
    }


# ---------------------------------------------------------------------------
# Cluster-fraction summary helpers
# ---------------------------------------------------------------------------

def cluster_fractions(phase, band=QUAD_BAND):
    near_0     = (np.abs(phase) < band).mean()
    near_pp    = (np.abs(phase - np.pi / 2) < band).mean()
    near_mp    = (np.abs(phase + np.pi / 2) < band).mean()
    near_pi    = ((np.abs(phase - np.pi) < band)
                  | (np.abs(phase + np.pi) < band)).mean()
    return {
        "near_0":  float(near_0),
        "near_pi": float(near_pi),
        "near_pi_over_2": float(near_pp),
        "near_minus_pi_over_2": float(near_mp),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(histories, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for mode, h in histories.items():
        if h is None:
            continue
        steps = [e["step"] for e in h["history"]]
        train_acc = [e["train_seq_acc"] for e in h["history"]]
        test_acc = [e["test_seq_acc"] for e in h["history"]]
        axes[0].plot(steps, train_acc, "-", color=MODE_COLORS[mode],
                     label=mode, linewidth=1.6)
        axes[1].plot(steps, test_acc, "-", color=MODE_COLORS[mode],
                     label=mode, linewidth=1.6)
    for ax, title in zip(axes, ["train sequence accuracy",
                                 "test sequence accuracy"]):
        ax.set_xlabel("step")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_per_position_accuracy(histories, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    positions = [5, 6, 7]
    x_base = np.arange(len(positions))
    for i, mode in enumerate(ALL_MODES):
        h = histories.get(mode)
        if h is None:
            continue
        per_pos = h["final_full"]["per_pos_acc"]
        vals = [per_pos[str(p)] if str(p) in per_pos else per_pos[p]
                for p in positions]
        offset = (i - 1) * width
        ax.bar(x_base + offset, vals, width,
               color=MODE_COLORS[mode], label=mode)
    ax.set_xticks(x_base)
    ax.set_xticklabels([f'pos 5 ("=") -> c2',
                        "pos 6 (c2) -> c1",
                        "pos 7 (c1) -> c0"])
    ax.set_ylabel("token accuracy on full dataset")
    ax.set_title("Per-position task accuracy (final, full dataset)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_distribution_grid(results, save_path):
    """Rows = modes, cols = variants. Each panel is a phase histogram.
    Empty where the variant is trivial (e.g., attn_only's mlp_delta)."""
    fig, axes = plt.subplots(
        len(ALL_MODES), len(VARIANTS),
        figsize=(4 * len(VARIANTS), 3 * len(ALL_MODES)),
        squeeze=False,
    )
    for i, mode in enumerate(ALL_MODES):
        for j, var in enumerate(VARIANTS):
            ax = axes[i, j]
            data = results.get(mode, {}).get(var)
            if data is None:
                ax.set_facecolor("#eeeeee")
                ax.text(
                    0.5, 0.5, "trivial / empty",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.hist(
                    data["cospec_phase"], bins=60,
                    range=(-np.pi, np.pi),
                    density=True, histtype="step",
                    color=MODE_COLORS[mode], linewidth=1.5,
                )
                for x in (-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi):
                    ax.axvline(x, color="gray", linewidth=0.4,
                               linestyle=":")
                ax.set_xlim(-np.pi, np.pi)
            if i == 0:
                ax.set_title(var, fontsize=11)
            if j == 0:
                ax.set_ylabel(mode, fontsize=11)
            if i == len(ALL_MODES) - 1:
                ax.set_xlabel("phase difference at slow bin", fontsize=9)
    fig.suptitle(
        f"Phase difference distribution at position {DEFAULT_POSITION} "
        f"by (model, variant)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_R_distribution_grid(results, save_path):
    fig, axes = plt.subplots(
        len(ALL_MODES), len(VARIANTS),
        figsize=(4 * len(VARIANTS), 3 * len(ALL_MODES)),
        squeeze=False,
    )
    for i, mode in enumerate(ALL_MODES):
        for j, var in enumerate(VARIANTS):
            ax = axes[i, j]
            data = results.get(mode, {}).get(var)
            if data is None:
                ax.set_facecolor("#eeeeee")
                ax.text(
                    0.5, 0.5, "trivial / empty",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.hist(
                    data["R"], bins=20, density=True, histtype="step",
                    color=MODE_COLORS[mode], linewidth=1.5,
                    label=f"R mean={data['R'].mean():.2f}",
                )
                ax.hist(
                    data["null_R"].flatten(), bins=20, density=True,
                    histtype="step", color="gray", linewidth=0.7,
                    linestyle=":",
                    label=f"null mean={data['null_R'].mean():.2f}",
                )
                ax.legend(fontsize=7)
            if i == 0:
                ax.set_title(var, fontsize=11)
            if j == 0:
                ax.set_ylabel(mode, fontsize=11)
            if i == len(ALL_MODES) - 1:
                ax.set_xlabel("rotation count R", fontsize=9)
    fig.suptitle(
        f"Per-unit rotation count at position {DEFAULT_POSITION} "
        f"by (model, variant)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_plv_high_amp_comparison(results, save_path):
    """For each variant, overlay the high-amp PLV distribution from
    each model. The cleanest direct comparison of phase consistency."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, var in zip(axes.flat, VARIANTS):
        for mode in ALL_MODES:
            data = results.get(mode, {}).get(var)
            if data is None:
                continue
            thresh = np.percentile(data["amp_prod"], 75)
            keep = data["amp_prod"] >= thresh
            if keep.sum() == 0:
                continue
            ax.hist(
                data["plv"][keep], bins=30, range=(0, 1),
                density=True, histtype="step",
                color=MODE_COLORS[mode], linewidth=1.6,
                label=f"{mode} (n={int(keep.sum())}, "
                      f"mean={data['plv'][keep].mean():.3f})",
            )
        ax.set_title(var, fontsize=11)
        ax.set_xlabel("PLV (high-amp pairs only)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Phase consistency on high-amp pairs, per variant",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_clustering_bars(results, save_path):
    """Three bar charts (one per variant), each showing the fraction of pairs
    near 0, +/-pi/2, pi for each model. Quickest visual summary."""
    cluster_keys = ["near_0", "near_pi_over_2",
                    "near_minus_pi_over_2", "near_pi"]
    cluster_labels = ["0", "+pi/2", "-pi/2", "pi"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    width = 0.25
    x_base = np.arange(len(cluster_keys))

    for ax, var in zip(axes.flat, VARIANTS):
        any_data = False
        for i, mode in enumerate(ALL_MODES):
            data = results.get(mode, {}).get(var)
            if data is None:
                continue
            any_data = True
            fracs = cluster_fractions(data["cospec_phase"])
            vals = [fracs[k] for k in cluster_keys]
            offset = (i - 1) * width
            ax.bar(x_base + offset, vals, width,
                   color=MODE_COLORS[mode], label=mode)
        uniform = QUAD_BAND / np.pi
        ax.axhline(uniform, color="gray", linewidth=0.6, linestyle="--",
                   label=f"uniform ({uniform:.3f})")
        ax.set_xticks(x_base)
        ax.set_xticklabels(cluster_labels)
        ax.set_xlabel("cluster center")
        ax.set_ylabel("fraction of pairs")
        ax.set_title(var, fontsize=11)
        if any_data:
            ax.legend(fontsize=8)
    fig.suptitle(
        f"Phase clustering by model, per variant "
        f"(band = +/- {QUAD_BAND:.3f} rad)",
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
    parser.add_argument("--root", type=str, default="ablation_run",
                        help="Root directory containing full/, attn_only/, mlp_only/")
    parser.add_argument("--out-dir", type=str, default="ablation_compare")
    parser.add_argument("--position", type=int, default=DEFAULT_POSITION)
    parser.add_argument("--n-shuffle", type=int, default=30)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for mode in ALL_MODES:
        ckpt_path = root / mode / "model_trained.pt"
        hist_path = root / mode / "training_history.json"
        if not ckpt_path.exists():
            print(f"[{mode}] no checkpoint at {ckpt_path}; skipping")
            histories[mode] = None
            results[mode] = None
            continue

        print(f"\n=== {mode} ===")
        with open(hist_path) as f:
            histories[mode] = json.load(f)

        model, cfg = load_checkpoint(ckpt_path)
        model = model.to(device)

        print(f"  capturing streams at position {args.position} ...")
        streams, sublayer_meta = collect_streams_at_position(
            model, args.position
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  streams shape: {streams.shape}")
        print(f"  analyzing variants ...")
        variants = make_variants(streams)
        results[mode] = {}
        for var in VARIANTS:
            r = analyze_variant(variants[var], args.n_shuffle)
            results[mode][var] = r
            if r is None:
                print(f"    {var}: trivial (skipped)")
            else:
                print(f"    {var}: R mean={r['R'].mean():.2f} "
                      f"(null mean={r['null_R'].mean():.2f}), "
                      f"PLV all-pairs mean={r['plv'].mean():.3f}")

    print("\nMaking comparison plots ...")
    plot_training_curves(histories, out_dir / "training_curves.png")
    plot_per_position_accuracy(
        histories, out_dir / "per_position_accuracy.png"
    )
    plot_phase_distribution_grid(
        results, out_dir / "phase_distribution_grid.png"
    )
    plot_R_distribution_grid(
        results, out_dir / "R_distribution_grid.png"
    )
    plot_plv_high_amp_comparison(
        results, out_dir / "plv_high_amp_comparison.png"
    )
    plot_phase_clustering_bars(
        results, out_dir / "phase_clustering_bars.png"
    )

    summary = {
        "position": args.position,
        "models": {},
    }
    for mode in ALL_MODES:
        if results.get(mode) is None:
            summary["models"][mode] = None
            continue
        h = histories[mode]
        per_mode = {
            "n_params":     h["n_params"],
            "final_seq_acc_full_dataset": float(
                h["final_full"]["sequence_acc"]
            ),
            "final_per_pos_acc": h["final_full"]["per_pos_acc"],
            "variants": {},
        }
        for var in VARIANTS:
            r = results[mode].get(var)
            if r is None:
                per_mode["variants"][var] = None
                continue
            thresh = np.percentile(r["amp_prod"], 75)
            keep = r["amp_prod"] >= thresh
            per_mode["variants"][var] = {
                "R_abs_mean":     float(np.abs(r["R"]).mean()),
                "R_std":          float(r["R"].std()),
                "null_R_mean":    float(r["null_R"].mean()),
                "bin1_mag_mean":  float(r["bin1_mag"].mean()),
                "cospec_mag_p95": float(np.percentile(r["cospec_mag"], 95)),
                "plv_all_mean":   float(r["plv"].mean()),
                "plv_high_amp_mean": float(r["plv"][keep].mean())
                                       if keep.any() else None,
                "cluster_fractions": cluster_fractions(r["cospec_phase"]),
            }
        summary["models"][mode] = per_mode

    with open(out_dir / "ablation_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Stdout summary
    print("\n--- Ablation comparison summary ---")
    for mode in ALL_MODES:
        m = summary["models"].get(mode)
        if m is None:
            print(f"\n[{mode}] (no checkpoint)")
            continue
        print(f"\n[{mode}] params={m['n_params']}, "
              f"final test seq acc={m['final_seq_acc_full_dataset']:.3f}, "
              f"per_pos={m['final_per_pos_acc']}")
        for var in VARIANTS:
            v = m["variants"].get(var)
            if v is None:
                print(f"  {var}: (trivial / empty)")
                continue
            cf = v["cluster_fractions"]
            print(f"  {var}: |R| mean={v['R_abs_mean']:.2f} "
                  f"(null={v['null_R_mean']:.2f}), "
                  f"PLV high-amp={v['plv_high_amp_mean']!r}, "
                  f"clusters: 0={cf['near_0']:.2f} "
                  f"+pi/2={cf['near_pi_over_2']:.2f} "
                  f"-pi/2={cf['near_minus_pi_over_2']:.2f} "
                  f"pi={cf['near_pi']:.2f}")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
