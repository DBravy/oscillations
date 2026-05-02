"""
TinyLlama 1.1B: phase / rotation analysis on four trajectory variants:
  1. Cumulative residual stream (h(ℓ))
  2. Combined deltas (h(ℓ+1) - h(ℓ))
  3. Attention-only deltas (post-attention - pre-attention)
  4. MLP-only deltas (post-mlp - pre-mlp, which is pre-attention next - pre-mlp)

For each: compute per-unit rotation count R, FFT spectrum, pairwise phase
distribution at bin 1 (or at a slow bin if n_steps is small). Compare
each against a shuffle null for R.

Outputs in tinyllama_out/:
  - R_distribution.png        rotation count for all 4 variants + null
  - phase_diff_distribution.png  phase clustering for all 4
  - cospec_heatmap.png        2D phase-vs-mag heatmap for all 4 (furthest regime)
  - phase_portraits.png       sample units across R range for each variant
  - tinyllama_summary.json
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


N_SAMPLES = 128       # smaller than SmolLM2 since model is bigger
SEQ_LEN = 128
SEED = 0
DEVICE_MAP = "auto"
N_PAIRS = 200_000
QUAD_BAND = np.pi / 6
N_SHUFFLE_RUNS = 30


# ---------------------------------------------------------------------------
# Model loading (Llama-style architecture)
# ---------------------------------------------------------------------------

def load_model():
    print("Loading TinyLlama ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map=DEVICE_MAP,
        output_hidden_states=True,
    )
    model.eval()
    print(f"Loaded: {model.config.num_hidden_layers} layers, "
          f"hidden_size={model.config.hidden_size}")
    return model, tokenizer


def get_hook_targets(model):
    """Llama-style: each block has input_layernorm (pre-attn) and
    post_attention_layernorm (pre-mlp)."""
    blocks = model.model.layers
    targets = []
    for block in blocks:
        targets.append(block.input_layernorm)
        targets.append(block.post_attention_layernorm)
    return targets


# ---------------------------------------------------------------------------
# Stream collection
# ---------------------------------------------------------------------------

def collect_streams(model, tokenizer, texts, seq_len):
    targets = get_hook_targets(model)
    out = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt",
                        max_length=seq_len, truncation=True)
        ids = enc["input_ids"].to(next(model.parameters()).device)
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
        # Cast to float32 for stable downstream math
        last = torch.stack(
            [c[0, -1, :].float() for c in captures], dim=0
        )
        out.append(last.cpu().numpy())
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Trajectory variants
# ---------------------------------------------------------------------------

def make_variants(streams):
    """
    Given streams of shape (n_samples, 2*L, d_model), produce four
    trajectory variants:
      cumulative: streams as-is, length 2L
      combined:   diff along axis 1, length 2L - 1
      attn:       even-indexed diffs (post-attn - pre-attn), length L
      mlp:        odd-indexed diffs (post-mlp - pre-mlp = next pre-attn - pre-mlp), length L
    """
    diffs = np.diff(streams, axis=1)  # (n_samples, 2L-1, d_model)
    attn_deltas = diffs[:, ::2, :]     # captures 0->1, 2->3, etc
    mlp_deltas = diffs[:, 1::2, :]     # captures 1->2, 3->4, etc

    return {
        "cumulative": streams,
        "combined_delta": diffs,
        "attn_delta": attn_deltas,
        "mlp_delta": mlp_deltas,
    }


# ---------------------------------------------------------------------------
# Per-unit and per-pair analyses
# ---------------------------------------------------------------------------

def rotation_count_per_unit(streams):
    """Mean R per unit, vectorized."""
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
    """Per-unit null: shuffle the layer ordering and recompute R.
    Returns (n_runs, d_model) of per-unit means across samples."""
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


def sample_pair_indices(d_model, n_pairs, rng):
    a = rng.integers(0, d_model, size=n_pairs * 2)
    b = rng.integers(0, d_model, size=n_pairs * 2)
    mask = a != b
    a = a[mask][:n_pairs]
    b = b[mask][:n_pairs]
    return np.minimum(a, b), np.maximum(a, b)


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


def analyze_variant(streams, label, args, rng):
    print(f"\n[{label}] shape={streams.shape}")
    d_model = streams.shape[2]

    print(f"  R ...")
    R = rotation_count_per_unit(streams)

    print(f"  shuffle null ({args.n_shuffle} runs) ...")
    null_R = shuffle_null_R(streams, args.n_shuffle,
                              np.random.default_rng(SEED + 1))

    print(f"  spectra ...")
    spectra = per_unit_spectra(streams)

    print(f"  sampling {args.n_pairs} pairs ...")
    u_idx, v_idx = sample_pair_indices(d_model, args.n_pairs, rng)

    print(f"  trace correlations ...")
    trace_corr = trace_correlations(streams, u_idx, v_idx)

    # Pick a "slow" frequency bin: bin 1 if available, else bin 0 + 1
    n_freq = spectra.shape[1]
    freq_bin = 1 if n_freq > 1 else 0
    print(f"  cross-spectral at bin {freq_bin} (n_freq={n_freq}) ...")
    phase, mag = pair_cross_spectral(spectra, u_idx, v_idx,
                                       freq_bin=freq_bin)

    return {
        "label": label,
        "streams": streams,
        "R": R,
        "null_R": null_R,
        "spectra": spectra,
        "u_idx": u_idx,
        "v_idx": v_idx,
        "trace_corr": trace_corr,
        "phase": phase,
        "mag": mag,
        "freq_bin_used": freq_bin,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

VARIANTS = ["cumulative", "combined_delta", "attn_delta", "mlp_delta"]


def plot_R_distribution(results, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in VARIANTS:
        data = results[label]
        ax.hist(data["R"], bins=60, density=True,
                histtype="step", linewidth=1.5,
                label=f"{label} "
                      f"(mean={data['R'].mean():.2f}, "
                      f"null={data['null_R'].mean():.2f})")
        # Overlay the null as a thin grey curve
        ax.hist(data["null_R"].flatten(), bins=60, density=True,
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
    """Phase difference distribution (all pairs) for each variant."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in VARIANTS:
        data = results[label]
        ax.hist(data["phase"], bins=80, range=(-np.pi, np.pi),
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
    """Three-regime phase diff for each variant: 4 columns × 3 rows."""
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
            ax.hist(data["phase"][mask], bins=80, range=(-np.pi, np.pi),
                    density=True, histtype="step", linewidth=1.5)
            for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
                ax.axvline(x, color="gray", linewidth=0.4,
                           linestyle=":")
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
    """Furthest-regime 2D heatmap of phase vs log mag, per variant."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, label in zip(axes, VARIANTS):
        data = results[label]
        prox = 1.0 - np.abs(data["trace_corr"])
        edges = np.quantile(prox, [0, 1/3, 2/3, 1])
        furthest = prox > edges[2]
        p = data["phase"][furthest]
        m = data["mag"][furthest]
        m_floor = max(np.percentile(m, 1), 1e-10)
        keep = m >= m_floor
        p = p[keep]
        m = m[keep]
        if len(p) == 0:
            ax.set_title(f"{label}: empty")
            continue
        h, xedges, yedges = np.histogram2d(
            p, np.log10(m + 1e-12),
            bins=[60, 60],
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


def plot_phase_portraits(results, save_path=None):
    """For each variant, three units across the |R| range."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    sample_idx = 0
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
            ax.set_ylabel("∇a - ∇a_0", fontsize=8)
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
        "label": data["label"],
        "n_steps": int(data["streams"].shape[1]),
        "freq_bin_used": int(data["freq_bin_used"]),
        "R": {
            "mean": float(R.mean()),
            "std": float(R.std()),
            "median": float(np.median(R)),
            "abs_mean": float(np.abs(R).mean()),
        },
        "null_R": {
            "mean": float(null_R.mean()),
            "std": float(null_R.std()),
        },
        "trace_corr": {
            "mean": float(tc.mean()),
            "std": float(tc.std()),
            "abs_mean": float(np.abs(tc).mean()),
        },
        "cospec_mag": {
            "mean": float(m.mean()),
            "median": float(np.median(m)),
            "p95": float(np.percentile(m, 95)),
        },
        "phase_clustering_fractions": {
            "near_0": float(near(0).mean()),
            "near_pi": float(near("pi").mean()),
            "near_pi_over_2": float(near(np.pi / 2).mean()),
            "near_minus_pi_over_2": float(near(-np.pi / 2).mean()),
        },
        "phase_clustering_mag_means": {
            "near_0": float(m[near(0)].mean())
                if near(0).any() else None,
            "near_pi": float(m[near("pi")].mean())
                if near("pi").any() else None,
            "near_pi_over_2": float(m[near(np.pi / 2)].mean())
                if near(np.pi / 2).any() else None,
            "near_minus_pi_over_2": float(m[near(-np.pi / 2)].mean())
                if near(-np.pi / 2).any() else None,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    parser.add_argument("--n-shuffle", type=int, default=N_SHUFFLE_RUNS)
    args = parser.parse_args()

    out_dir = Path("tinyllama_out")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("Loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]

    model, tokenizer = load_model()

    print("\nCollecting streams ...")
    streams = collect_streams(model, tokenizer, texts, args.seq_len)
    print(f"streams shape: {streams.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("\nMaking variants ...")
    variants = make_variants(streams)
    for k, v in variants.items():
        print(f"  {k}: shape {v.shape}")

    print("\nAnalyzing variants ...")
    results = {}
    for label in VARIANTS:
        results[label] = analyze_variant(
            variants[label], label, args,
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
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "n_samples": int(streams.shape[0]),
        "n_sublayers": int(streams.shape[1]),
        "d_model": int(streams.shape[2]),
        "variants": {label: summarize_variant(results[label])
                     for label in VARIANTS},
    }
    with open(out_dir / "tinyllama_summary.json", "w") as f:
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
              f"π/2={f0['near_pi_over_2']:.3f}, "
              f"-π/2={f0['near_minus_pi_over_2']:.3f}, "
              f"π={f0['near_pi']:.3f}")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()