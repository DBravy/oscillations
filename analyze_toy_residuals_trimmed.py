"""
Toy-transformer residual-stream phase analysis.

Adapts the TinyLlama / SmolLM2 four-variant analysis to the d_model=64,
8-layer toy model trained on MSB-first base-4 two-digit addition.

The 17-sublayer capture list is:
    [embed, attn_0, mlp_0, attn_1, mlp_1, ..., attn_7, mlp_7]
Diffs of this list:
    diffs[0]  = post_attn_0 - embed         (attn 0 contribution)
    diffs[1]  = post_mlp_0 - post_attn_0    (mlp 0 contribution)
    diffs[2]  = post_attn_1 - post_mlp_0    (attn 1 contribution)
    ...
So diffs[::2] are attn deltas (length 8), diffs[1::2] are mlp deltas (length 8).

Outputs in toy_analysis_out/ (configurable):
  - R_distribution.png            rotation count for all 4 variants + null
  - phase_diff_distribution.png   phase clustering for all 4
  - phase_diff_by_proximity.png   three-regime phase diff per variant
  - cospec_heatmap.png            phase-vs-mag heatmap per variant (furthest)
  - phase_portraits.png           sample units across R range per variant
  - toy_summary.json              numerical summary

Usage:
  python analyze_toy_residuals.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_analysis_out_trained \
      --position 5

  python analyze_toy_residuals.py \
      --checkpoint toy_transformer_run/model_random_init.pt \
      --out-dir toy_analysis_out_random \
      --position 5

  # Edge-control run: drop first/last 2 cumulative stream samples before
  # constructing cumulative/delta variants and computing cospec/phase stats.
  python analyze_toy_residuals_trimmed.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_analysis_out_trained_trim2 \
      --position 5 \
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


SEED = 0
QUAD_BAND = np.pi / 6
N_SHUFFLE_RUNS = 30

VARIANTS = ["cumulative", "combined_delta", "attn_delta", "mlp_delta"]


# ---------------------------------------------------------------------------
# Stream collection: load model, capture residuals, pick a token position
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = ToyTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def collect_streams_at_position(model, cfg, position, batch_size=64):
    """
    Run the model on every (a, b) in [0, 16) x [0, 16), capture the residual
    stream at every sublayer, and pull out a single token position.

    Returns:
      streams: (N, n_sublayers, d_model) float32, where N = 256
      tokens:  (N, seq_len) long, the input sequences
      sublayer_meta: list of (kind, layer_idx) tuples, length n_sublayers
    """
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]
    tokens, captures = capture_residual_streams(
        model, pairs, device, batch_size=batch_size
    )
    # captures: list of dicts with keys {kind, layer, residual: (N, T, D)}
    sublayer_meta = [(c["kind"], c["layer"]) for c in captures]
    # Stack into (n_sublayers, N, T, D) then pick position -> (n_sublayers, N, D)
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    at_pos = stacked[:, :, position, :]            # (n_sublayers, N, D)
    streams = at_pos.permute(1, 0, 2).contiguous() # (N, n_sublayers, D)
    return streams.numpy().astype(np.float32), tokens.numpy(), sublayer_meta



def trim_streams_by_depth(streams, sublayer_meta, trim):
    """Drop boundary cumulative residual samples before variant construction.

    trim=0 keeps the original stream. trim=k keeps original indices k:L-k.
    Deltas are then computed *inside* this trimmed interval, so the first and
    last retained samples do not involve the original boundary sublayers.
    """
    if trim < 0:
        raise ValueError("--trim-sublayers must be >= 0")

    L = streams.shape[1]
    if 2 * trim >= L:
        raise ValueError(
            f"--trim-sublayers={trim} removes all/too many sublayers "
            f"for L={L}. Need 2*trim < L."
        )

    if trim == 0:
        return streams, sublayer_meta, list(range(L))

    used_indices = list(range(trim, L - trim))
    return streams[:, trim:L - trim, :].copy(), [sublayer_meta[i] for i in used_indices], used_indices


# ---------------------------------------------------------------------------
# Trajectory variants
# ---------------------------------------------------------------------------

def make_variants(streams):
    """
    streams: (N, n_sublayers, D) where n_sublayers = 2L + 1 (with embed).
    diffs has length 2L. diffs[::2] = attn deltas (L), diffs[1::2] = mlp deltas (L).
    """
    diffs = np.diff(streams, axis=1)
    return {
        "cumulative":     streams,
        "combined_delta": diffs,
        "attn_delta":     diffs[:, ::2, :],
        "mlp_delta":      diffs[:, 1::2, :],
    }


# ---------------------------------------------------------------------------
# Per-unit and per-pair analyses (verbatim from the reference)
# ---------------------------------------------------------------------------

def rotation_count_per_unit(streams):
    """Mean R per unit. Same convention as the reference script."""
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    theta = np.arctan2(dy, dx)
    dtheta = np.diff(theta, axis=1)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    return (dtheta.sum(axis=1) / (2 * np.pi)).mean(axis=0)


def shuffle_null_R(streams, n_runs, rng):
    n_samples, n_steps, d_model = streams.shape
    null_means = np.zeros((n_runs, d_model))
    for r in range(n_runs):
        perm = rng.permutation(n_steps)
        shuffled = streams[:, perm, :]
        null_means[r] = rotation_count_per_unit(shuffled)
    return null_means


def per_unit_spectra(streams):
    ms = streams - streams.mean(axis=1, keepdims=True)
    return np.fft.rfft(ms, axis=1)


def all_pair_indices(d_model):
    """All ordered pairs with u < v."""
    u, v = np.triu_indices(d_model, k=1)
    return u, v


def trace_correlations(streams, u_idx, v_idx):
    n_samples = streams.shape[0]
    out = np.zeros(len(u_idx))
    for s in range(n_samples):
        sample = streams[s]
        sample_centered = sample - sample.mean(axis=0, keepdims=True)
        sample_std = sample_centered.std(axis=0) + 1e-8
        u_traces = sample_centered[:, u_idx] / sample_std[u_idx]
        v_traces = sample_centered[:, v_idx] / sample_std[v_idx]
        out += np.mean(u_traces * v_traces, axis=0)
    return out / n_samples


def pair_cross_spectral(spectra, u_idx, v_idx, freq_bin=1):
    n_samples = spectra.shape[0]
    coeffs = np.zeros(len(u_idx), dtype=complex)
    for s in range(n_samples):
        u_vals = spectra[s, freq_bin, u_idx]
        v_vals = spectra[s, freq_bin, v_idx]
        coeffs += u_vals * np.conj(v_vals)
    coeffs /= n_samples
    return np.angle(coeffs), np.abs(coeffs)


def analyze_variant(streams, label, n_shuffle, rng):
    print(f"\n[{label}] shape={streams.shape}")
    d_model = streams.shape[2]

    print(f"  R ...")
    R = rotation_count_per_unit(streams)

    print(f"  shuffle null ({n_shuffle} runs) ...")
    null_R = shuffle_null_R(
        streams, n_shuffle, np.random.default_rng(SEED + 1)
    )

    print(f"  spectra ...")
    spectra = per_unit_spectra(streams)

    u_idx, v_idx = all_pair_indices(d_model)
    print(f"  trace corr on {len(u_idx)} pairs ...")
    trace_corr = trace_correlations(streams, u_idx, v_idx)

    n_freq = spectra.shape[1]
    freq_bin = 1 if n_freq > 1 else 0
    print(f"  cross-spec at bin {freq_bin} (n_freq={n_freq}) ...")
    phase, mag = pair_cross_spectral(
        spectra, u_idx, v_idx, freq_bin=freq_bin
    )

    return {
        "label":         label,
        "streams":       streams,
        "R":             R,
        "null_R":        null_R,
        "spectra":       spectra,
        "u_idx":         u_idx,
        "v_idx":         v_idx,
        "trace_corr":    trace_corr,
        "phase":         phase,
        "mag":           mag,
        "freq_bin_used": freq_bin,
    }


# ---------------------------------------------------------------------------
# Plots (mostly copied from the reference, light tweaks for small d_model)
# ---------------------------------------------------------------------------

def plot_R_distribution(results, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in VARIANTS:
        data = results[label]
        ax.hist(data["R"], bins=30, density=True,
                histtype="step", linewidth=1.5,
                label=f"{label} "
                      f"(mean={data['R'].mean():.2f}, "
                      f"null={data['null_R'].mean():.2f})")
        ax.hist(data["null_R"].flatten(), bins=30, density=True,
                histtype="step", linewidth=0.7, alpha=0.4,
                color="gray", linestyle=":")
    ax.set_xlabel("rotation count R")
    ax.set_ylabel("density")
    ax.set_title("Per-unit rotation count: real (colored) vs null (gray)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_diff_distribution(results, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in VARIANTS:
        data = results[label]
        ax.hist(data["phase"], bins=60, range=(-np.pi, np.pi),
                density=True, histtype="step", linewidth=1.5,
                label=label)
    for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
        ax.axvline(x, color="gray", linewidth=0.4, linestyle=":")
    ax.set_xlabel("phase difference at slow bin")
    ax.set_ylabel("density")
    ax.set_title("Pairwise phase difference distribution (all pairs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_diff_by_proximity(results, save_path=None):
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharey=True)
    for col, label in enumerate(VARIANTS):
        data = results[label]
        prox = 1.0 - np.abs(data["trace_corr"])
        edges = np.quantile(prox, [0, 1/3, 2/3, 1])
        for row, (regime_label, mask) in enumerate([
            ("closest",  prox <= edges[1]),
            ("middle",   (prox > edges[1]) & (prox <= edges[2])),
            ("furthest", prox > edges[2]),
        ]):
            ax = axes[row, col]
            ax.hist(data["phase"][mask], bins=60, range=(-np.pi, np.pi),
                    density=True, histtype="step", linewidth=1.5)
            for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
                ax.axvline(x, color="gray", linewidth=0.4, linestyle=":")
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{regime_label}\ndensity", fontsize=10)
            if row == 2:
                ax.set_xlabel("phase difference")
    fig.suptitle("Phase difference by proximity regime, per variant",
                 fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_cospec_heatmap(results, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, label in zip(axes, VARIANTS):
        data = results[label]
        prox = 1.0 - np.abs(data["trace_corr"])
        edges = np.quantile(prox, [0, 1/3, 2/3, 1])
        furthest = prox > edges[2]
        p = data["phase"][furthest]
        m = data["mag"][furthest]
        if len(m) == 0:
            ax.set_title(f"{label}: empty")
            continue
        m_floor = max(np.percentile(m, 1), 1e-10)
        keep = m >= m_floor
        p = p[keep]
        m = m[keep]
        if len(p) == 0:
            ax.set_title(f"{label}: empty after floor")
            continue
        h, xedges, yedges = np.histogram2d(
            p, np.log10(m + 1e-12),
            bins=[40, 40],
            range=[[-np.pi, np.pi],
                   [np.log10(m_floor),
                    np.log10(m.max() + 1e-12)]],
        )
        im = ax.imshow(
            h.T, origin="lower", aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis",
        )
        ax.set_xlabel("phase difference")
        ax.set_ylabel("log10(cospec magnitude)")
        ax.set_title(f"{label} (n={len(p)})", fontsize=10)
        for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
            ax.axvline(x, color="white", linewidth=0.5,
                       linestyle="--", alpha=0.5)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Cospec heatmap (furthest regime)", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_portraits(results, save_path=None, sample_idx=0):
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for col, label in enumerate(VARIANTS):
        data = results[label]
        order = np.argsort(np.abs(data["R"]))
        selected = [order[len(order) // 10],
                    order[len(order) // 2],
                    order[-len(order) // 10]]
        labels = ["low |R|", "median |R|", "high |R|"]
        sample = data["streams"][sample_idx]
        n_steps = sample.shape[0]
        for row, (u, lbl) in enumerate(zip(selected, labels)):
            ax = axes[row, col]
            a = sample[:, u]
            ga = np.gradient(a)
            x = a - a[0]
            y = ga - ga[0]
            ax.plot(x, y, "-", linewidth=0.7, alpha=0.9)
            ax.scatter(x, y, c=np.arange(n_steps),
                       cmap="viridis", s=10)
            ax.set_title(
                f"{label}\n{lbl}: u={u}, R={data['R'][u]:.2f}",
                fontsize=9,
            )
            ax.set_xlabel("a - a_0", fontsize=8)
            ax.set_ylabel("grad a - grad a_0", fontsize=8)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_variant(data):
    R = data["R"]
    null_R = data["null_R"]
    p = data["phase"]
    m = data["mag"]
    tc = data["trace_corr"]

    def near(target, band=QUAD_BAND):
        if target == "pi":
            return (np.abs(p - np.pi) < band) | (np.abs(p + np.pi) < band)
        return np.abs(p - target) < band

    return {
        "label":         data["label"],
        "n_steps":       int(data["streams"].shape[1]),
        "freq_bin_used": int(data["freq_bin_used"]),
        "R": {
            "mean":     float(R.mean()),
            "std":      float(R.std()),
            "median":   float(np.median(R)),
            "abs_mean": float(np.abs(R).mean()),
        },
        "null_R": {
            "mean": float(null_R.mean()),
            "std":  float(null_R.std()),
        },
        "trace_corr": {
            "mean":     float(tc.mean()),
            "std":      float(tc.std()),
            "abs_mean": float(np.abs(tc).mean()),
        },
        "cospec_mag": {
            "mean":   float(m.mean()),
            "median": float(np.median(m)),
            "p95":    float(np.percentile(m, 95)),
        },
        "phase_clustering_fractions": {
            "near_0":              float(near(0).mean()),
            "near_pi":             float(near("pi").mean()),
            "near_pi_over_2":      float(near(np.pi / 2).mean()),
            "near_minus_pi_over_2": float(near(-np.pi / 2).mean()),
        },
        "phase_clustering_mag_means": {
            "near_0":              float(m[near(0)].mean())              if near(0).any()              else None,
            "near_pi":             float(m[near("pi")].mean())           if near("pi").any()           else None,
            "near_pi_over_2":      float(m[near(np.pi / 2)].mean())      if near(np.pi / 2).any()      else None,
            "near_minus_pi_over_2": float(m[near(-np.pi / 2)].mean())    if near(-np.pi / 2).any()     else None,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model_trained.pt or model_random_init.pt",
    )
    parser.add_argument(
        "--out-dir", type=str, default="toy_analysis_out",
        help="Directory to write plots and summary into",
    )
    parser.add_argument(
        "--position", type=int, default=5,
        help=("Token position to analyze. "
              "5 = '=' token (just committed to answer). "
              "6 = c2 (just emitted MSB, must retain c1, c0). "
              "7 = c1. 8 = c0."),
    )
    parser.add_argument(
        "--n-shuffle", type=int, default=N_SHUFFLE_RUNS,
        help="Number of shuffle null runs",
    )
    parser.add_argument(
        "--trim-sublayers", type=int, default=0, metavar="K",
        help=("Drop first K and last K cumulative residual stream samples "
              "before constructing variants and computing cospec/phase stats. "
              "Default: 0."),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print(f"Loading checkpoint from {args.checkpoint} ...")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}")

    print(f"\nCollecting residual streams at position {args.position} "
          f"on all 256 (a, b) pairs ...")
    streams, tokens, sublayer_meta = collect_streams_at_position(
        model, cfg, args.position
    )
    print(f"streams shape: {streams.shape}")
    print(f"sublayers: {sublayer_meta}")

    original_sublayer_meta = list(sublayer_meta)
    original_n_sublayers = streams.shape[1]
    streams, sublayer_meta, used_sublayer_indices = trim_streams_by_depth(
        streams, sublayer_meta, args.trim_sublayers
    )
    if args.trim_sublayers:
        print(
            f"\nTrimmed first/last {args.trim_sublayers} sublayers before analysis: "
            f"L {original_n_sublayers} -> {streams.shape[1]}"
        )
        print(f"  retained original sublayer indices: {used_sublayer_indices}")
        print(f"  retained sublayers: {sublayer_meta}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nMaking variants ...")
    variants = make_variants(streams)
    for k, v in variants.items():
        print(f"  {k}: shape {v.shape}")

    print("\nAnalyzing variants ...")
    results = {}
    for label in VARIANTS:
        results[label] = analyze_variant(
            variants[label], label, args.n_shuffle,
            np.random.default_rng(SEED + hash(label) % 1000),
        )

    print("\nMaking plots ...")
    plot_R_distribution(
        results, save_path=out_dir / "R_distribution.png"
    )
    plot_phase_diff_distribution(
        results, save_path=out_dir / "phase_diff_distribution.png"
    )
    plot_phase_diff_by_proximity(
        results, save_path=out_dir / "phase_diff_by_proximity.png"
    )
    plot_cospec_heatmap(
        results, save_path=out_dir / "cospec_heatmap.png"
    )
    plot_phase_portraits(
        results, save_path=out_dir / "phase_portraits.png"
    )

    summary = {
        "checkpoint":   str(args.checkpoint),
        "position":     int(args.position),
        "n_samples":    int(streams.shape[0]),
        "n_sublayers":  int(streams.shape[1]),
        "original_n_sublayers": int(original_n_sublayers),
        "trim_sublayers": int(args.trim_sublayers),
        "used_sublayer_indices": [int(i) for i in used_sublayer_indices],
        "d_model":      int(streams.shape[2]),
        "sublayer_meta": [
            {"kind": k, "layer": l} for (k, l) in sublayer_meta
        ],
        "original_sublayer_meta": [
            {"kind": k, "layer": l} for (k, l) in original_sublayer_meta
        ],
        "variants": {
            label: summarize_variant(results[label])
            for label in VARIANTS
        },
    }
    with open(out_dir / "toy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    for label in VARIANTS:
        s = summary["variants"][label]
        print(f"\n[{label}] n_steps={s['n_steps']}, "
              f"freq_bin={s['freq_bin_used']}")
        print(f"  R: mean {s['R']['mean']:.2f}, "
              f"std {s['R']['std']:.2f}, "
              f"null mean {s['null_R']['mean']:.2f}")
        print(f"  Trace corr: std {s['trace_corr']['std']:.3f}")
        print(f"  Cospec mag: mean {s['cospec_mag']['mean']:.4f}, "
              f"p95 {s['cospec_mag']['p95']:.4f}")
        f0 = s["phase_clustering_fractions"]
        print(f"  Phase fractions: 0={f0['near_0']:.3f}, "
              f"pi/2={f0['near_pi_over_2']:.3f}, "
              f"-pi/2={f0['near_minus_pi_over_2']:.3f}, "
              f"pi={f0['near_pi']:.3f}")

    if args.trim_sublayers:
        print(
            f"\nNOTE: all summaries/plots use the trimmed depth axis; "
            f"original indices are saved as used_sublayer_indices."
        )

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
