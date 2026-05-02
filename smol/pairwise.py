"""
Pairwise structure between residual units: frequency alignment and
phase relationships, with multiple notions of "proximity."

We compute four pairwise quantities for unit pairs:
  - trace_correlation      Pearson correlation of cross-layer traces
                           (averaged across samples)
  - freq_distance          difference in dominant FFT bin
  - phase_difference       angle of cross-spectral coefficient at
                           dominant frequency, averaged across samples
  - cospectral_magnitude   magnitude of the same coefficient

Then we test these against three notions of proximity:
  - standard-basis index distance |u - v|
  - data-driven proximity from trace correlation
  - sorted-by-dominant-frequency proximity

The question we're answering: do units that are 'close' under any of
these notions have stable phase relationships, and what are they?

Outputs in pairwise_out_<model>/:
  - pairwise_summary.json
  - phase_diff_distribution.png      histogram + polar scatter of phase diffs
  - basis_proximity_curves.png       four pairwise quantities vs |u-v|
  - corr_proximity_curves.png        same vs trace-correlation rank
  - freq_sort_curves.png             same vs frequency-sorted rank
  - high_corr_pair_examples.png      example trace pairs across the
                                     correlation distribution
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

# Number of pairs to sample for pairwise analysis. d_model^2 is too big.
# We use a random sample and stratify by some proximity notions.
N_PAIRS = 200_000


# ---------------------------------------------------------------------------
# Boilerplate: hooks, residual stream collection
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
    for i, block in enumerate(blocks):
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
    expected = len(targets)
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
        if len(captures) != expected:
            raise RuntimeError("hook count mismatch")
        last = torch.stack([c[0, -1, :].float() for c in captures], dim=0)
        out.append(last.cpu().numpy())
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Per-unit features
# ---------------------------------------------------------------------------

def per_unit_spectra(streams):
    """
    For each (sample, unit), FFT the cross-layer trace (mean-subtracted)
    and store the complex spectrum. Returns:
      spectra: (n_samples, n_freq, d_model) complex
      freqs: (n_freq,) integer bin indices
    """
    n_samples, n_sub, d_model = streams.shape
    # Mean-subtract along sublayer axis
    ms = streams - streams.mean(axis=1, keepdims=True)
    # rfft over sublayer axis
    spec = np.fft.rfft(ms, axis=1)
    n_freq = spec.shape[1]
    freqs = np.arange(n_freq)
    return spec, freqs


def dominant_freq_per_unit(spectra):
    """
    Per unit, find the dominant non-DC frequency bin (averaged power
    over samples).
    Returns: (d_model,) int array
    """
    power = np.mean(np.abs(spectra) ** 2, axis=0)  # (n_freq, d_model)
    # ignore DC bin; argmax over non-DC bins
    if power.shape[0] <= 1:
        return np.zeros(power.shape[1], dtype=int)
    idx = np.argmax(power[1:], axis=0) + 1
    return idx


# ---------------------------------------------------------------------------
# Pairwise analysis
# ---------------------------------------------------------------------------

def sample_pair_indices(d_model, n_pairs, rng):
    """Random sample of unordered unit pairs (u != v)."""
    a = rng.integers(0, d_model, size=n_pairs * 2)
    b = rng.integers(0, d_model, size=n_pairs * 2)
    mask = a != b
    a = a[mask][:n_pairs]
    b = b[mask][:n_pairs]
    # canonicalize order
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return lo, hi


def trace_correlations(streams, u_idx, v_idx):
    """
    Pearson correlation of cross-layer traces for each pair, averaged
    over samples. Returns (n_pairs,)
    """
    n_samples, n_sub, d_model = streams.shape
    # For each sample, correlate trace_u and trace_v across sublayers,
    # then average.
    out = np.zeros(len(u_idx))
    for s in range(n_samples):
        sample = streams[s]  # (n_sub, d_model)
        sample_centered = sample - sample.mean(axis=0, keepdims=True)
        sample_std = sample_centered.std(axis=0) + 1e-8
        # Compute correlations only for the requested pairs
        u_traces = sample_centered[:, u_idx] / sample_std[u_idx]
        v_traces = sample_centered[:, v_idx] / sample_std[v_idx]
        out += np.mean(u_traces * v_traces, axis=0)
    out /= n_samples
    return out


def pair_cross_spectral(spectra, dom_freq_per_pair, u_idx, v_idx):
    """
    For each pair, compute the cross-spectral coefficient at the pair's
    dominant frequency, averaged over samples:
        c_uv = E_s [ spec_s[k_uv, u] * conj(spec_s[k_uv, v]) ]
    where k_uv is the dominant bin of the pair.

    Returns (phase_diff, magnitude) of shape (n_pairs,).
    """
    n_samples = spectra.shape[0]
    n_pairs = len(u_idx)
    coeffs = np.zeros(n_pairs, dtype=complex)
    for s in range(n_samples):
        spec_s = spectra[s]  # (n_freq, d_model)
        # gather: spec_s[k_uv, u] for each pair
        u_vals = spec_s[dom_freq_per_pair, u_idx]
        v_vals = spec_s[dom_freq_per_pair, v_idx]
        coeffs += u_vals * np.conj(v_vals)
    coeffs /= n_samples
    return np.angle(coeffs), np.abs(coeffs)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_phase_diff_distribution(phase_diffs, magnitudes, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(phase_diffs, bins=80, weights=magnitudes,
                 range=(-np.pi, np.pi))
    axes[0].set_xlabel("phase difference (radians)")
    axes[0].set_ylabel("magnitude-weighted count")
    axes[0].set_title("Pairwise phase differences at pair's dominant freq")
    for x, lbl in [(0, "0"), (np.pi/2, "π/2"), (-np.pi/2, "-π/2"),
                   (np.pi, "±π"), (-np.pi, "")]:
        axes[0].axvline(x, color="gray", linewidth=0.5, linestyle="--")

    ax = plt.subplot(1, 2, 2, projection="polar")
    # subsample for polar scatter (don't draw a million points)
    n_show = min(20000, len(phase_diffs))
    idx = np.random.choice(len(phase_diffs), size=n_show, replace=False)
    ax.scatter(phase_diffs[idx], magnitudes[idx], s=1, alpha=0.2)
    ax.set_title("Phase diff vs co-spectral magnitude")
    ax.set_rticks([])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_proximity_curves(prox, trace_corr, freq_dist, phase_diff,
                          cospec_mag, n_bins=40,
                          save_path=None, xlabel="proximity"):
    """
    Bin pairs by `prox` (a per-pair scalar). For each bin, show the mean
    of trace_corr, freq_dist, and cospec_mag, and a histogram of phase
    differences for the smallest-prox bin.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Bin the proximity axis
    quantile_edges = np.quantile(prox, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.digitize(prox, quantile_edges[1:-1])
    bin_centers = 0.5 * (quantile_edges[:-1] + quantile_edges[1:])

    def by_bin(values):
        means = np.array([
            values[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
            for b in range(n_bins)
        ])
        return means

    axes[0, 0].plot(bin_centers, by_bin(trace_corr), "o-")
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel("mean trace correlation")
    axes[0, 0].set_title("Trace correlation vs proximity")
    axes[0, 0].axhline(0, color="gray", linewidth=0.5)

    axes[0, 1].plot(bin_centers, by_bin(freq_dist), "o-")
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel("mean |freq_u - freq_v|")
    axes[0, 1].set_title("Frequency distance vs proximity")

    axes[1, 0].plot(bin_centers, by_bin(cospec_mag), "o-")
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel("mean co-spectral magnitude")
    axes[1, 0].set_title("Co-spectral magnitude vs proximity")

    # For phase: show histogram of phase diffs in the lowest-proximity
    # bin (closest pairs), middle, and highest (furthest) for contrast.
    closest = phase_diff[bin_idx == 0]
    middle = phase_diff[bin_idx == n_bins // 2]
    furthest = phase_diff[bin_idx == n_bins - 1]
    axes[1, 1].hist(
        [closest, middle, furthest],
        bins=40, range=(-np.pi, np.pi),
        label=["closest", "middle", "furthest"],
        density=True, histtype="step", linewidth=1.5,
    )
    axes[1, 1].set_xlabel("phase difference")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title("Phase difference distribution by proximity")
    axes[1, 1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_corr_pair_examples(streams, u_idx, v_idx, trace_corr,
                            save_path=None):
    """
    Show example unit-trace pairs across the trace_correlation
    distribution: the most-correlated, most-anti-correlated, and
    near-zero correlation. Three rows, one example per row.
    """
    fig, axes = plt.subplots(3, 3, figsize=(13, 8))
    n_show = 3

    # most positive
    idx_pos = np.argsort(trace_corr)[-n_show:]
    # most negative
    idx_neg = np.argsort(trace_corr)[:n_show]
    # near zero
    idx_zero = np.argsort(np.abs(trace_corr))[:n_show]

    rows = [
        ("highest +corr", idx_pos),
        ("highest -corr", idx_neg),
        ("near 0 corr", idx_zero),
    ]
    for r, (label, idxs) in enumerate(rows):
        for c, p in enumerate(idxs):
            u, v = u_idx[p], v_idx[p]
            ax = axes[r, c]
            # plot a few samples for each
            for s in range(min(8, streams.shape[0])):
                ax.plot(streams[s, :, u], color="C0", alpha=0.4, linewidth=0.7)
                ax.plot(streams[s, :, v], color="C1", alpha=0.4, linewidth=0.7)
            ax.set_title(
                f"{label}: u={u}, v={v}, r={trace_corr[p]:.2f}",
                fontsize=9,
            )
            ax.set_xlabel("sublayer", fontsize=8)
            ax.tick_params(labelsize=7)

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

    out_dir = Path(f"pairwise_out_{slugify(args.model)}")
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

    # Normalize each sublayer to unit norm so we're looking at directions
    norms = np.linalg.norm(streams, axis=2, keepdims=True)
    streams_n = streams / np.maximum(norms, 1e-8)

    print("Computing per-unit spectra ...")
    spectra, freqs = per_unit_spectra(streams_n)
    dom_freq = dominant_freq_per_unit(spectra)
    print(f"dominant freq histogram: "
          f"{np.bincount(dom_freq, minlength=len(freqs))}")

    print(f"Sampling {args.n_pairs} pairs ...")
    u_idx, v_idx = sample_pair_indices(d_model, args.n_pairs, rng)

    print("Pairwise: trace correlation ...")
    trace_corr = trace_correlations(streams_n, u_idx, v_idx)

    print("Pairwise: dominant freq distance ...")
    freq_dist = np.abs(dom_freq[u_idx] - dom_freq[v_idx]).astype(float)

    # For phase analysis, use the average of the two units' dominant
    # frequencies, rounded. (If they differ, we evaluate at the closer
    # one to v's.) Simpler: use u's dominant freq.
    pair_freq = dom_freq[u_idx]

    print("Pairwise: cross-spectral phase + magnitude ...")
    phase_diff, cospec_mag = pair_cross_spectral(
        spectra, pair_freq, u_idx, v_idx
    )

    # Three notions of proximity:
    print("Computing proximity notions ...")
    prox_basis = np.abs(u_idx - v_idx).astype(float)

    # Trace-correlation proximity: smaller distance = more correlated.
    # Define distance = 1 - |corr|.
    prox_corr = 1.0 - np.abs(trace_corr)

    # Frequency-sorted proximity: rank units by dominant freq, then
    # take rank distance.
    rank_by_freq = np.argsort(np.argsort(dom_freq))
    prox_freqsort = np.abs(rank_by_freq[u_idx] -
                            rank_by_freq[v_idx]).astype(float)

    print("Saving plots ...")
    plot_phase_diff_distribution(
        phase_diff, cospec_mag,
        save_path=out_dir / "phase_diff_distribution.png",
    )
    plot_proximity_curves(
        prox_basis, trace_corr, freq_dist, phase_diff, cospec_mag,
        save_path=out_dir / "basis_proximity_curves.png",
        xlabel="|u - v| (basis distance)",
    )
    plot_proximity_curves(
        prox_corr, trace_corr, freq_dist, phase_diff, cospec_mag,
        save_path=out_dir / "corr_proximity_curves.png",
        xlabel="1 - |trace correlation|",
    )
    plot_proximity_curves(
        prox_freqsort, trace_corr, freq_dist, phase_diff, cospec_mag,
        save_path=out_dir / "freq_sort_curves.png",
        xlabel="freq-sorted rank distance",
    )
    plot_corr_pair_examples(
        streams_n, u_idx, v_idx, trace_corr,
        save_path=out_dir / "high_corr_pair_examples.png",
    )

    # Summary
    summary = {
        "model": args.model,
        "n_samples": int(streams.shape[0]),
        "n_sublayers": int(streams.shape[1]),
        "d_model": int(d_model),
        "n_pairs": int(args.n_pairs),
        "trace_corr": {
            "mean": float(trace_corr.mean()),
            "std": float(trace_corr.std()),
            "abs_mean": float(np.abs(trace_corr).mean()),
            "p5": float(np.percentile(trace_corr, 5)),
            "p95": float(np.percentile(trace_corr, 95)),
        },
        "freq_dist": {
            "mean": float(freq_dist.mean()),
            "median": float(np.median(freq_dist)),
            "frac_zero": float((freq_dist == 0).mean()),
        },
        "phase_diff": {
            "mean_resultant_length_unweighted": float(
                np.abs(np.mean(np.exp(1j * phase_diff)))
            ),
            "mean_resultant_length_weighted": float(
                np.abs(
                    np.sum(cospec_mag * np.exp(1j * phase_diff))
                    / np.sum(cospec_mag)
                )
            ),
        },
        "cospec_mag": {
            "mean": float(cospec_mag.mean()),
            "median": float(np.median(cospec_mag)),
        },
        # Correlations between proximity and the four pairwise quantities.
        # Spearman would be more appropriate but pearson is fine for
        # a quick read.
        "proximity_correlations": {
            "basis_vs_trace_corr": float(np.corrcoef(prox_basis, trace_corr)[0, 1]),
            "basis_vs_freq_dist": float(np.corrcoef(prox_basis, freq_dist)[0, 1]),
            "basis_vs_cospec_mag": float(np.corrcoef(prox_basis, cospec_mag)[0, 1]),
            "freqsort_vs_trace_corr": float(np.corrcoef(prox_freqsort, trace_corr)[0, 1]),
            "freqsort_vs_cospec_mag": float(np.corrcoef(prox_freqsort, cospec_mag)[0, 1]),
        },
    }
    with open(out_dir / "pairwise_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(f"Model: {args.model}")
    print(f"Pairs analyzed: {args.n_pairs}")
    print(f"Trace correlation: mean={summary['trace_corr']['mean']:.3f}, "
          f"|mean|={summary['trace_corr']['abs_mean']:.3f}, "
          f"5-95% range = "
          f"[{summary['trace_corr']['p5']:.2f}, "
          f"{summary['trace_corr']['p95']:.2f}]")
    print(f"Phase-diff resultant length (weighted): "
          f"{summary['phase_diff']['mean_resultant_length_weighted']:.3f}")
    print(f"Frequency match: "
          f"{summary['freq_dist']['frac_zero']*100:.1f}% of pairs have "
          f"identical dominant freq")
    print(f"Basis proximity vs trace corr: "
          f"r = {summary['proximity_correlations']['basis_vs_trace_corr']:.3f}")
    print(f"Freq-sorted proximity vs trace corr: "
          f"r = {summary['proximity_correlations']['freqsort_vs_trace_corr']:.3f}")
    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()