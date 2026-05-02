"""
Characterize the high-magnitude quadrature subpopulation in the
furthest regime.

We've established that pairs with:
  - low trace correlation (furthest regime)
  - high cospec magnitude
  - phase difference near ±π/2
form a concentrated subpopulation. This script asks:
  1. How does the ±π/2 concentration sharpen as we raise the magnitude
     threshold?
  2. Which units participate in this subpopulation, and do they cluster?
  3. Does the subpopulation have internal structure (e.g., are the
     same units paired with many others, or do pairs come from a wide
     pool)?

Outputs in quadrature_out_<model>/:
  - threshold_sweep.png       phase distribution at increasing
                              magnitude thresholds
  - peak_sharpness.png        |R_2| (axial concentration around π/2 axis)
                              vs threshold
  - unit_participation.png    histogram of how often each unit appears
                              in selected pairs; per-unit position
                              along the d_model index
  - pair_index_2d.png         2D scatter of (u_idx, v_idx) for selected
                              pairs, to see any index-space clustering
  - participation_vs_freq.png does participation correlate with the
                              unit's dominant frequency or its
                              activation magnitude?
  - quadrature_summary.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"
N_SAMPLES = 256
SEQ_LEN = 128
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PAIRS = 200_000

# Threshold sweep: top fractions of pairs by cospec magnitude
THRESHOLDS = [0.50, 0.25, 0.10, 0.05, 0.02, 0.01]

# How wide a band around ±π/2 counts as "in the quadrature peak"
QUAD_BAND = np.pi / 6


# ---------------------------------------------------------------------------
# Boilerplate (same as before)
# ---------------------------------------------------------------------------

def get_hook_targets(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
        attn_ln, mlp_ln = "ln_1", "ln_2"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
        attn_ln, mlp_ln = "input_layernorm", "post_attention_layernorm"
    else:
        raise RuntimeError("Could not locate transformer blocks.")
    targets = []
    for block in blocks:
        targets.append(getattr(block, attn_ln))
        targets.append(getattr(block, mlp_ln))
    return targets


def get_d_model(model):
    cfg = model.config
    for attr in ("n_embd", "hidden_size", "d_model"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise RuntimeError("d_model unknown")


def get_n_layers(model):
    cfg = model.config
    for attr in ("n_layer", "num_hidden_layers"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise RuntimeError("n_layers unknown")


def collect_streams(model, tokenizer, texts, seq_len):
    model.eval()
    targets = get_hook_targets(model)
    out = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt",
                        max_length=seq_len, truncation=True)
        ids = enc["input_ids"].to(DEVICE)
        if ids.shape[1] < 8:
            continue
        captures = []
        hooks = [t.register_forward_hook(
            lambda m, i, o, c=captures: c.append(i[0].detach())
        ) for t in targets]
        try:
            with torch.no_grad():
                model(ids)
        finally:
            for h in hooks:
                h.remove()
        last = torch.stack([c[0, -1, :].float() for c in captures], dim=0)
        out.append(last.cpu().numpy())
    return np.stack(out, axis=0)


def per_unit_spectra(streams):
    ms = streams - streams.mean(axis=1, keepdims=True)
    return np.fft.rfft(ms, axis=1)


def dominant_freq_per_unit(spectra):
    power = np.mean(np.abs(spectra) ** 2, axis=0)
    if power.shape[0] <= 1:
        return np.zeros(power.shape[1], dtype=int)
    return np.argmax(power[1:], axis=0) + 1


def sample_pair_indices(d_model, n_pairs, rng):
    a = rng.integers(0, d_model, size=n_pairs * 2)
    b = rng.integers(0, d_model, size=n_pairs * 2)
    mask = a != b
    a = a[mask][:n_pairs]
    b = b[mask][:n_pairs]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return lo, hi


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


def pair_cross_spectral(spectra, freq_per_pair, u_idx, v_idx):
    n_samples = spectra.shape[0]
    coeffs = np.zeros(len(u_idx), dtype=complex)
    for s in range(n_samples):
        spec_s = spectra[s]
        u_vals = spec_s[freq_per_pair, u_idx]
        v_vals = spec_s[freq_per_pair, v_idx]
        coeffs += u_vals * np.conj(v_vals)
    coeffs /= n_samples
    return np.angle(coeffs), np.abs(coeffs)


# ---------------------------------------------------------------------------
# Helper: detect quadrature concentration
# ---------------------------------------------------------------------------

def axial_quadrature_index(phases):
    """
    A scalar in [0, 1] measuring how much the phase distribution is
    concentrated around ±π/2 relative to 0 and π.

    Method: shift the phase by π/2 so that ±π/2 maps to 0 and ±π,
    then take the second circular moment R_2 of the shifted phases.
    R_2 is high if the shifted phases pile at 0 and ±π.

    Equivalently: |E[exp(2i(θ - π/2))]| = |E[exp(2iθ - iπ)]|
                                        = |-E[exp(2iθ)]|
                                        = |E[exp(2iθ)]|

    So R_2 of the unshifted phases also captures axial concentration,
    but it doesn't distinguish the 0/π axis from the ±π/2 axis. We use
    a sign convention: positive value means concentrated at ±π/2,
    negative means concentrated at 0 and π.
    """
    z2 = np.mean(np.exp(2j * phases))
    # If z2 ≈ +1: phases concentrate around 0 or π.
    # If z2 ≈ -1: phases concentrate around ±π/2.
    return float(-np.real(z2))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_threshold_sweep(phase, mag, regime_mask, thresholds,
                         save_path=None):
    """
    Phase distribution within the furthest regime at increasing
    magnitude thresholds.
    """
    p = phase[regime_mask]
    m = mag[regime_mask]

    fig, ax = plt.subplots(figsize=(9, 5))
    for frac in thresholds:
        cutoff = np.quantile(m, 1 - frac)
        keep = m >= cutoff
        n_kept = keep.sum()
        ax.hist(
            p[keep], bins=80, range=(-np.pi, np.pi),
            density=True, histtype="step", linewidth=1.5,
            label=f"top {int(frac*100)}% (n={n_kept})",
        )
    for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
        ax.axvline(x, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("phase difference")
    ax.set_ylabel("density")
    ax.set_title("Furthest regime: phase distribution at increasing "
                 "magnitude thresholds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_peak_sharpness(phase, mag, regime_mask, thresholds,
                        save_path=None):
    """
    Quadrature axial index as a function of magnitude threshold.
    Computed both for the furthest regime and (for comparison) the
    closest regime.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, mask in [("furthest", regime_mask),
                        ("closest", ~regime_mask)]:
        p = phase[mask]
        m = mag[mask]
        idx_values = []
        for frac in thresholds:
            cutoff = np.quantile(m, 1 - frac)
            keep = m >= cutoff
            if keep.sum() > 10:
                idx_values.append(axial_quadrature_index(p[keep]))
            else:
                idx_values.append(np.nan)
        ax.plot(
            [int(f * 100) for f in thresholds], idx_values,
            "o-", label=label,
        )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("top X% by cospec magnitude")
    ax.set_ylabel("quadrature axial index "
                  "(positive = peaks at ±π/2)")
    ax.set_title("Sharpness of ±π/2 peaks vs magnitude threshold")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_unit_participation(d_model, u_sel, v_sel, save_path=None):
    """
    Histogram of how often each unit appears in selected pairs.
    Plus: scatter of participation count vs unit index.
    """
    counts = np.zeros(d_model, dtype=int)
    np.add.at(counts, u_sel, 1)
    np.add.at(counts, v_sel, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(counts, bins=60)
    axes[0].set_xlabel("appearances per unit")
    axes[0].set_ylabel("number of units")
    axes[0].set_title("How often each unit appears in selected pairs")
    axes[0].axvline(2 * len(u_sel) / d_model, color="red", linewidth=1,
                    linestyle="--",
                    label=f"uniform expectation "
                          f"({2 * len(u_sel) / d_model:.1f})")
    axes[0].legend(fontsize=8)

    axes[1].scatter(np.arange(d_model), counts, s=4, alpha=0.5)
    axes[1].set_xlabel("unit index (0..d_model-1)")
    axes[1].set_ylabel("appearances")
    axes[1].set_title("Participation vs unit index")
    axes[1].axhline(2 * len(u_sel) / d_model, color="red",
                    linewidth=1, linestyle="--")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return counts


def plot_pair_index_2d(d_model, u_sel, v_sel, save_path=None):
    """
    2D scatter of selected pair indices, to see whether pairs cluster
    in index space.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(u_sel, v_sel, s=2, alpha=0.3)
    ax.set_xlim(0, d_model)
    ax.set_ylim(0, d_model)
    ax.set_xlabel("u index")
    ax.set_ylabel("v index")
    ax.set_title(f"Selected pairs in (u, v) index space "
                 f"(n={len(u_sel)})")
    ax.set_aspect("equal")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_participation_vs_unit_features(counts, dom_freq, mean_abs,
                                        save_path=None):
    """
    Does participation correlate with a unit's dominant frequency,
    or with how loud it is?
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].scatter(dom_freq, counts, s=4, alpha=0.4)
    axes[0].set_xlabel("dominant FFT bin")
    axes[0].set_ylabel("appearances in selected pairs")
    axes[0].set_title("Participation vs dominant frequency")

    # Use log-x for activation magnitude since it's heavy-tailed
    axes[1].scatter(mean_abs, counts, s=4, alpha=0.4)
    axes[1].set_xlabel("mean |activation| (log)")
    axes[1].set_xscale("log")
    axes[1].set_ylabel("appearances in selected pairs")
    axes[1].set_title("Participation vs unit loudness")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    args = parser.parse_args()

    out_dir = Path(f"quadrature_out_{slugify(args.model)}")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(DEVICE)

    d_model = get_d_model(model)
    n_layers = get_n_layers(model)
    print(f"d_model={d_model}, n_layers={n_layers}")

    print("Loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]

    print("Collecting streams ...")
    streams = collect_streams(model, tokenizer, texts, args.seq_len)
    print(f"streams shape: {streams.shape}")

    norms = np.linalg.norm(streams, axis=2, keepdims=True)
    streams_n = streams / np.maximum(norms, 1e-8)

    # Per-unit features for later analysis
    mean_abs = np.mean(np.abs(streams_n), axis=(0, 1))  # (d_model,)

    print("Computing spectra and dominant freqs ...")
    spectra = per_unit_spectra(streams_n)
    dom_freq = dominant_freq_per_unit(spectra)

    print(f"Sampling {args.n_pairs} pairs ...")
    u_idx, v_idx = sample_pair_indices(d_model, args.n_pairs, rng)

    print("Trace correlations ...")
    trace_corr = trace_correlations(streams_n, u_idx, v_idx)

    print("Cross-spectral phase / mag at u's dominant bin ...")
    phase, mag = pair_cross_spectral(spectra, dom_freq[u_idx], u_idx, v_idx)

    prox = 1.0 - np.abs(trace_corr)
    edges = np.quantile(prox, [0, 1/3, 2/3, 1])
    furthest_mask = prox > edges[2]

    print("Plot 1: threshold sweep on furthest regime ...")
    plot_threshold_sweep(
        phase, mag, furthest_mask, THRESHOLDS,
        save_path=out_dir / "threshold_sweep.png",
    )

    print("Plot 2: peak sharpness vs threshold ...")
    plot_peak_sharpness(
        phase, mag, furthest_mask, THRESHOLDS,
        save_path=out_dir / "peak_sharpness.png",
    )

    # Selection: top 10% by cospec magnitude in furthest regime,
    # restricted to phases within QUAD_BAND of ±π/2.
    p_furthest = phase[furthest_mask]
    m_furthest = mag[furthest_mask]
    u_furthest = u_idx[furthest_mask]
    v_furthest = v_idx[furthest_mask]

    cutoff = np.quantile(m_furthest, 0.90)
    high_mag = m_furthest >= cutoff

    near_pi_over_2 = (
        (np.abs(p_furthest - np.pi / 2) < QUAD_BAND) |
        (np.abs(p_furthest + np.pi / 2) < QUAD_BAND)
    )

    selected = high_mag & near_pi_over_2
    u_sel = u_furthest[selected]
    v_sel = v_furthest[selected]
    print(f"Selected pairs: {selected.sum()} "
          f"({100 * selected.sum() / len(p_furthest):.2f}% of furthest, "
          f"{100 * selected.sum() / args.n_pairs:.2f}% of all sampled pairs)")

    print("Plot 3: unit participation ...")
    counts = plot_unit_participation(
        d_model, u_sel, v_sel,
        save_path=out_dir / "unit_participation.png",
    )

    print("Plot 4: pair indices in 2D ...")
    plot_pair_index_2d(
        d_model, u_sel, v_sel,
        save_path=out_dir / "pair_index_2d.png",
    )

    print("Plot 5: participation vs unit features ...")
    plot_participation_vs_unit_features(
        counts, dom_freq, mean_abs,
        save_path=out_dir / "participation_vs_freq.png",
    )

    # Numerical summary
    summary = {
        "model": args.model,
        "n_samples": int(streams.shape[0]),
        "n_sublayers": int(streams.shape[1]),
        "d_model": int(d_model),
        "n_pairs_sampled": int(args.n_pairs),
        "n_furthest": int(furthest_mask.sum()),
        "selected_subpopulation": {
            "magnitude_threshold_quantile": 0.90,
            "phase_band_radians": float(QUAD_BAND),
            "n_selected": int(selected.sum()),
            "fraction_of_furthest": float(
                selected.sum() / max(len(p_furthest), 1)
            ),
            "fraction_of_all": float(selected.sum() / args.n_pairs),
        },
        "participation": {
            "uniform_expectation": float(2 * len(u_sel) / d_model),
            "max_count": int(counts.max()),
            "median_count": float(np.median(counts)),
            "n_units_above_2x_uniform": int(
                (counts > 2 * (2 * len(u_sel) / d_model)).sum()
            ),
            "n_units_zero_participation": int((counts == 0).sum()),
            # Top 20 most-participating units
            "top_units_by_participation":
                np.argsort(counts)[-20:][::-1].tolist(),
            "top_unit_counts":
                counts[np.argsort(counts)[-20:][::-1]].tolist(),
        },
        "axial_quadrature_index_by_threshold": {
            f"top_{int(frac*100)}pct": float(
                axial_quadrature_index(
                    p_furthest[m_furthest >= np.quantile(m_furthest, 1 - frac)]
                )
            )
            for frac in THRESHOLDS
        },
        "correlation_with_unit_features": {
            "participation_vs_dom_freq":
                float(np.corrcoef(counts, dom_freq)[0, 1]),
            "participation_vs_mean_abs":
                float(np.corrcoef(counts, mean_abs)[0, 1]),
        },
    }
    with open(out_dir / "quadrature_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(f"Selected subpopulation: {summary['selected_subpopulation']['n_selected']} "
          f"pairs ({100 * summary['selected_subpopulation']['fraction_of_all']:.2f}% "
          f"of all sampled)")
    print(f"Uniform expected count per unit: "
          f"{summary['participation']['uniform_expectation']:.2f}")
    print(f"Max count: {summary['participation']['max_count']}, "
          f"median: {summary['participation']['median_count']:.1f}")
    print(f"Units with > 2x uniform participation: "
          f"{summary['participation']['n_units_above_2x_uniform']}")
    print(f"Units with zero participation: "
          f"{summary['participation']['n_units_zero_participation']}")
    print(f"\nQuadrature axial index by threshold:")
    for k, v in summary["axial_quadrature_index_by_threshold"].items():
        print(f"  {k}: {v:+.3f}")
    print(f"\nParticipation correlation with dominant freq: "
          f"{summary['correlation_with_unit_features']['participation_vs_dom_freq']:+.3f}")
    print(f"Participation correlation with mean |activation|: "
          f"{summary['correlation_with_unit_features']['participation_vs_mean_abs']:+.3f}")
    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()