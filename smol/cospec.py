"""
Diagnostic plots focused on the 'furthest' regime, testing whether the
±π/2 peaks correspond to phase-locked-at-quadrature pairs (high cospec
magnitude) or to weak/unrelated pairs (low cospec magnitude).

Five plots:
  1. cospec_vs_phase_furthest.png
       2D histogram of (phase_diff, cospec_mag) restricted to the
       furthest regime. Tells us whether the ±π/2 peaks are
       concentrated in high-magnitude pairs.
  2. cospec_distribution_by_regime.png
       Cospec magnitude distribution overlaid for the three regimes.
  3. phase_diff_high_cospec.png
       Phase-difference distribution for the top X% of pairs by
       cospec magnitude (in each regime).
  4. fixed_bin_phase_diff.png
       Phase-difference distribution computed at a single fixed
       frequency bin instead of per-pair dominant bin.
  5. phase_diff_freqmatch.png
       Phase-difference distribution split by whether the pair shares
       a dominant frequency.

Usage:
  python diagnostics.py --model HuggingFaceTB/SmolLM2-360M
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
FIXED_BIN = 1               # for the fixed-bin variant
TOP_COSPEC_FRACTION = 0.10  # top X% by cospec magnitude


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
    """
    Cross-spectral coefficient at the given per-pair frequency, averaged
    over samples. Returns (phase_diff, magnitude) of shape (n_pairs,).
    """
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
# Plots
# ---------------------------------------------------------------------------

def plot_cospec_vs_phase_furthest(phase, mag, regime_mask, save_path=None):
    """
    2D histogram (phase_diff, cospec_mag) for pairs in the furthest
    regime. Y-axis log-scaled because cospec mags span a wide range.
    """
    p = phase[regime_mask]
    m = mag[regime_mask]

    # Drop the bottom 1% to keep log scale readable
    m_floor = max(np.percentile(m, 1), 1e-8)
    keep = m >= m_floor
    p = p[keep]
    m = m[keep]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 2D histogram
    h, xedges, yedges = np.histogram2d(
        p, np.log10(m + 1e-12),
        bins=[60, 60],
        range=[[-np.pi, np.pi],
               [np.log10(m_floor), np.log10(m.max() + 1e-12)]],
    )
    im = axes[0].imshow(
        h.T, origin="lower", aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="viridis",
    )
    axes[0].set_xlabel("phase difference (radians)")
    axes[0].set_ylabel("log10(cospec magnitude)")
    axes[0].set_title(f"Furthest regime: phase vs |cospec| "
                      f"(n={len(p)})")
    for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
        axes[0].axvline(x, color="white", linewidth=0.5,
                        linestyle="--", alpha=0.5)
    fig.colorbar(im, ax=axes[0], label="pair count")

    # Mean magnitude as a function of phase, for the same regime
    nbins = 40
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_mag = np.zeros(nbins)
    for i in range(nbins):
        mask = (p >= edges[i]) & (p < edges[i + 1])
        mean_mag[i] = m[mask].mean() if mask.any() else np.nan
    axes[1].plot(centers, mean_mag, "o-")
    axes[1].set_xlabel("phase difference")
    axes[1].set_ylabel("mean cospec magnitude")
    axes[1].set_title("Mean |cospec| as a function of phase (furthest regime)")
    for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
        axes[1].axvline(x, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_cospec_distribution_by_regime(mag, prox, save_path=None):
    """
    Histogram of cospec magnitudes for closest, middle, furthest bins
    (defined by quantiles of prox).
    """
    edges = np.quantile(prox, [0, 1/3, 2/3, 1])
    closest = mag[prox <= edges[1]]
    middle = mag[(prox > edges[1]) & (prox <= edges[2])]
    furthest = mag[prox > edges[2]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    bins = np.logspace(
        np.log10(max(mag.min(), 1e-8)),
        np.log10(mag.max() + 1e-12),
        60,
    )
    for arr, label in [(closest, "closest"), (middle, "middle"),
                       (furthest, "furthest")]:
        axes[0].hist(arr, bins=bins, histtype="step", linewidth=1.5,
                     density=True, label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("cospec magnitude (log)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Cospec magnitude by regime")
    axes[0].legend()

    # Linear-axis version, zoomed to most of the mass
    cap = np.percentile(mag, 99)
    bins_lin = np.linspace(0, cap, 60)
    for arr, label in [(closest, "closest"), (middle, "middle"),
                       (furthest, "furthest")]:
        axes[1].hist(arr, bins=bins_lin, histtype="step",
                     linewidth=1.5, density=True, label=label)
    axes[1].set_xlabel("cospec magnitude (linear, top 1% clipped)")
    axes[1].set_ylabel("density")
    axes[1].set_title("Cospec magnitude by regime (linear)")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_diff_high_cospec(phase, mag, prox, top_frac,
                                save_path=None):
    """
    Phase-difference distribution for the top fraction of pairs by
    cospec magnitude, broken down by regime.
    """
    edges = np.quantile(prox, [0, 1/3, 2/3, 1])
    regimes = [
        ("closest", prox <= edges[1]),
        ("middle", (prox > edges[1]) & (prox <= edges[2])),
        ("furthest", prox > edges[2]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             sharey=True)
    for ax, (label, mask) in zip(axes, regimes):
        in_regime_mag = mag[mask]
        in_regime_phase = phase[mask]
        if len(in_regime_mag) == 0:
            ax.set_title(f"{label}: empty")
            continue
        threshold = np.quantile(in_regime_mag, 1 - top_frac)
        keep = in_regime_mag >= threshold
        ax.hist(in_regime_phase[keep], bins=60, range=(-np.pi, np.pi),
                density=True, alpha=0.5, label=f"top {int(top_frac*100)}%")
        ax.hist(in_regime_phase, bins=60, range=(-np.pi, np.pi),
                density=True, histtype="step", linewidth=1.2,
                label="all", color="black")
        ax.set_title(f"{label}")
        ax.set_xlabel("phase difference")
        for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
            ax.axvline(x, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle(
        f"Phase difference: high-cospec subset vs. all pairs, by regime",
        fontsize=11,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_fixed_bin_phase_diff(phase_fixed, mag_fixed, prox, save_path=None):
    """
    Same regime breakdown but at a single fixed FFT bin for every pair.
    """
    edges = np.quantile(prox, [0, 1/3, 2/3, 1])
    regimes = [
        ("closest", prox <= edges[1]),
        ("middle", (prox > edges[1]) & (prox <= edges[2])),
        ("furthest", prox > edges[2]),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, mask in regimes:
        ax.hist(phase_fixed[mask], bins=60, range=(-np.pi, np.pi),
                density=True, histtype="step", linewidth=1.5, label=label)
    for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
        ax.axvline(x, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("phase difference at fixed FFT bin")
    ax.set_ylabel("density")
    ax.set_title("Phase difference at a single fixed frequency bin")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_diff_freqmatch(phase, dom_freq_u, dom_freq_v, save_path=None):
    """
    Phase-difference distribution split by whether u and v share their
    dominant frequency bin.
    """
    matched = dom_freq_u == dom_freq_v
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(phase[matched], bins=60, range=(-np.pi, np.pi),
            density=True, histtype="step", linewidth=1.5,
            label=f"matched dom freq (n={matched.sum()})")
    ax.hist(phase[~matched], bins=60, range=(-np.pi, np.pi),
            density=True, histtype="step", linewidth=1.5,
            label=f"unmatched (n={(~matched).sum()})")
    for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
        ax.axvline(x, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("phase difference")
    ax.set_ylabel("density")
    ax.set_title("Phase difference, split by frequency match")
    ax.legend()
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
    parser.add_argument("--fixed-bin", type=int, default=FIXED_BIN)
    args = parser.parse_args()

    out_dir = Path(f"diagnostics_out_{slugify(args.model)}")
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

    print("Computing spectra and per-unit dominant freqs ...")
    spectra = per_unit_spectra(streams_n)
    dom_freq = dominant_freq_per_unit(spectra)

    print(f"Sampling {args.n_pairs} pairs ...")
    u_idx, v_idx = sample_pair_indices(d_model, args.n_pairs, rng)

    print("Computing trace correlations ...")
    trace_corr = trace_correlations(streams_n, u_idx, v_idx)

    print("Computing cross-spectral phase / mag at u's dominant bin ...")
    phase_dom, mag_dom = pair_cross_spectral(
        spectra, dom_freq[u_idx], u_idx, v_idx
    )

    print(f"Computing cross-spectral phase / mag at fixed bin "
          f"k={args.fixed_bin} ...")
    fixed_freq_per_pair = np.full(len(u_idx), args.fixed_bin, dtype=int)
    phase_fixed, mag_fixed = pair_cross_spectral(
        spectra, fixed_freq_per_pair, u_idx, v_idx
    )

    # Proximity (1 - |trace correlation|)
    prox = 1.0 - np.abs(trace_corr)

    # Define furthest regime mask
    edges = np.quantile(prox, [0, 1/3, 2/3, 1])
    furthest_mask = prox > edges[2]

    print("Saving plots ...")
    plot_cospec_vs_phase_furthest(
        phase_dom, mag_dom, furthest_mask,
        save_path=out_dir / "cospec_vs_phase_furthest.png",
    )
    plot_cospec_distribution_by_regime(
        mag_dom, prox,
        save_path=out_dir / "cospec_distribution_by_regime.png",
    )
    plot_phase_diff_high_cospec(
        phase_dom, mag_dom, prox, TOP_COSPEC_FRACTION,
        save_path=out_dir / "phase_diff_high_cospec.png",
    )
    plot_fixed_bin_phase_diff(
        phase_fixed, mag_fixed, prox,
        save_path=out_dir / "fixed_bin_phase_diff.png",
    )
    plot_phase_diff_freqmatch(
        phase_dom, dom_freq[u_idx], dom_freq[v_idx],
        save_path=out_dir / "phase_diff_freqmatch.png",
    )

    # Numerical summary
    summary = {
        "model": args.model,
        "n_samples": int(streams.shape[0]),
        "n_sublayers": int(streams.shape[1]),
        "d_model": int(d_model),
        "n_pairs": int(args.n_pairs),
        "fixed_bin": int(args.fixed_bin),
        "top_cospec_fraction": float(TOP_COSPEC_FRACTION),
        "regime_edges": [float(e) for e in edges],
        "by_regime": {},
    }
    for label, mask in [
        ("closest", prox <= edges[1]),
        ("middle", (prox > edges[1]) & (prox <= edges[2])),
        ("furthest", prox > edges[2]),
    ]:
        m = mag_dom[mask]
        p = phase_dom[mask]
        # Mean cospec magnitude near each canonical phase
        def near(angle, width=np.pi/8):
            d = np.abs(np.angle(np.exp(1j * (p - angle))))
            return d < width

        summary["by_regime"][label] = {
            "n_pairs": int(mask.sum()),
            "mean_cospec_mag": float(m.mean()),
            "median_cospec_mag": float(np.median(m)),
            "mean_mag_at_phase_0": float(m[near(0)].mean())
                if near(0).any() else None,
            "mean_mag_at_phase_pi_over_2": float(m[near(np.pi/2)].mean())
                if near(np.pi/2).any() else None,
            "mean_mag_at_phase_minus_pi_over_2": float(
                m[near(-np.pi/2)].mean()
            ) if near(-np.pi/2).any() else None,
            "mean_mag_at_phase_pi": float(
                m[near(np.pi) | near(-np.pi)].mean()
            ) if (near(np.pi) | near(-np.pi)).any() else None,
        }

    with open(out_dir / "diagnostics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    for label, stats in summary["by_regime"].items():
        print(f"{label}: n={stats['n_pairs']}, "
              f"mean |cospec|={stats['mean_cospec_mag']:.4f}, "
              f"|cospec| at φ=0: {stats['mean_mag_at_phase_0']}, "
              f"at ±π/2: {stats['mean_mag_at_phase_pi_over_2']}, "
              f"{stats['mean_mag_at_phase_minus_pi_over_2']}, "
              f"at ±π: {stats['mean_mag_at_phase_pi']}")
    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()