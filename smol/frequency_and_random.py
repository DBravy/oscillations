"""
Two-part analysis:

1. Frequency-pairing: how do units cluster in frequency space, and
   how does frequency match relate to the phase structures we
   already found?
2. Trained vs random: re-run all key analyses on a random-init
   model with the same architecture, to separate learned structure
   from architectural baseline.

Outputs in train_vs_init_out/ with subdirectories for trained and
random_init runs. Each subdirectory contains:
  - frequency_distribution.png      per-unit tangent speed histogram
  - frequency_distance_distribution.png   pairwise speed differences
  - phase_vs_freq_distance.png      do quadrature pairs match in freq?
  - rotations_summary.json
  - per_unit_metrics.npz            for cross-comparison

Plus a top-level comparison plot:
  - trained_vs_init_comparison.png
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
)
from datasets import load_dataset


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"
N_SAMPLES = 256
SEQ_LEN = 128
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PAIRS = 200_000
QUAD_BAND = np.pi / 6
HIGH_MAG_QUANTILE = 0.90


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


# ---------------------------------------------------------------------------
# Per-unit metrics
# ---------------------------------------------------------------------------

def compute_per_unit_metrics(streams):
    """
    Per-unit:
      - mean tangent angular speed (rad/step, vector of size d_model)
      - power-weighted mean frequency (FFT bin units)
      - per-unit complex coefficient at FFT bin 1 (for phase analysis)
      - mean |activation|
    Plus the full streams for pair-level analysis.
    """
    n_samples, n_sub, d_model = streams.shape

    # Tangent angular speed: vectorized
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    theta = np.arctan2(dy, dx)
    dtheta = np.diff(theta, axis=1)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    # mean over samples and over sublayer transitions
    tangent_speed = np.abs(dtheta).mean(axis=(0, 1))  # (d_model,)

    # FFT power spectrum and power-weighted mean freq
    ms = streams - streams.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(ms, axis=1)
    power = np.mean(np.abs(spec) ** 2, axis=0)  # (n_freq, d_model)
    n_freq = power.shape[0]
    if n_freq > 1:
        ks = np.arange(1, n_freq).reshape(-1, 1)
        denom = np.maximum(power[1:].sum(axis=0), 1e-12)
        pw_mean = (ks * power[1:]).sum(axis=0) / denom
    else:
        pw_mean = np.zeros(d_model)

    # Per-unit complex coefficient at bin 1, averaged across samples
    bin1 = spec[:, 1, :].mean(axis=0)  # (d_model,) complex

    # Loudness
    mean_abs = np.mean(np.abs(streams), axis=(0, 1))

    return {
        "tangent_speed": tangent_speed,
        "pw_mean_freq": pw_mean,
        "bin1_coeff": bin1,
        "mean_abs": mean_abs,
        "spec": spec,
    }


# ---------------------------------------------------------------------------
# Pairwise analysis
# ---------------------------------------------------------------------------

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


def pair_cross_spectral(spec, u_idx, v_idx, freq_bin=1):
    """Cross-spectral coefficient at a fixed frequency bin, averaged
    over samples."""
    n_samples = spec.shape[0]
    coeffs = np.zeros(len(u_idx), dtype=complex)
    for s in range(n_samples):
        u_vals = spec[s, freq_bin, u_idx]
        v_vals = spec[s, freq_bin, v_idx]
        coeffs += u_vals * np.conj(v_vals)
    coeffs /= n_samples
    return np.angle(coeffs), np.abs(coeffs)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_frequency_distribution(metrics, save_path=None):
    """Per-unit tangent speed and power-weighted mean freq."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ts = metrics["tangent_speed"]
    pw = metrics["pw_mean_freq"]

    axes[0].hist(ts, bins=60)
    axes[0].set_xlabel("tangent angular speed (rad/step)")
    axes[0].set_ylabel("# units")
    axes[0].set_title(
        f"Tangent speed: mean={ts.mean():.3f}, std={ts.std():.3f}"
    )

    axes[1].hist(pw, bins=60)
    axes[1].set_xlabel("power-weighted mean FFT bin")
    axes[1].set_ylabel("# units")
    axes[1].set_title(
        f"Power-weighted mean: mean={pw.mean():.2f}, std={pw.std():.2f}"
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_freq_distance_distribution(metrics, u_idx, v_idx, save_path=None):
    """Histogram of pairwise frequency distances."""
    ts = metrics["tangent_speed"]
    pw = metrics["pw_mean_freq"]

    ts_dist = np.abs(ts[u_idx] - ts[v_idx])
    pw_dist = np.abs(pw[u_idx] - pw[v_idx])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(ts_dist, bins=80)
    axes[0].set_xlabel("|tangent_speed_u - tangent_speed_v|")
    axes[0].set_ylabel("# pairs")
    axes[0].set_title("Pairwise tangent-speed distance")
    axes[0].set_yscale("log")

    axes[1].hist(pw_dist, bins=80)
    axes[1].set_xlabel("|pw_mean_freq_u - pw_mean_freq_v|")
    axes[1].set_ylabel("# pairs")
    axes[1].set_title("Pairwise power-weighted-freq distance")
    axes[1].set_yscale("log")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_vs_freq_distance(phase, mag, ts_dist, save_path=None):
    """
    For pairs in the high-magnitude quadrature subset, scatter
    (frequency distance, phase difference). If quadrature pairs are
    truly frequency-matched, they should cluster at small ts_dist.
    """
    cutoff_mag = np.quantile(mag, 0.90)
    high_mag = mag >= cutoff_mag
    near_quad = (
        (np.abs(phase - np.pi / 2) < QUAD_BAND) |
        (np.abs(phase + np.pi / 2) < QUAD_BAND)
    )
    near_zero = np.abs(phase) < QUAD_BAND
    near_pi = (np.abs(phase - np.pi) < QUAD_BAND) | \
              (np.abs(phase + np.pi) < QUAD_BAND)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    for ax, mask, title in [
        (axes[0], high_mag & near_zero, "High-mag in-phase pairs"),
        (axes[1], high_mag & near_quad, "High-mag quadrature pairs"),
        (axes[2], high_mag & near_pi, "High-mag anti-phase pairs"),
    ]:
        if mask.sum() == 0:
            ax.set_title(f"{title}: empty")
            continue
        ax.hist(ts_dist[mask], bins=60, density=True,
                histtype="step", linewidth=1.5,
                label=f"n={mask.sum()}")
        ax.hist(ts_dist, bins=60, density=True,
                histtype="step", linewidth=1.0, color="gray",
                label=f"all pairs")
        ax.set_xlabel("|tangent_speed_u - tangent_speed_v|")
        ax.set_title(f"{title}", fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_comparison(trained_metrics, init_metrics,
                    trained_label, init_label, save_path=None):
    """Side-by-side comparison of trained vs init for key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # tangent speed
    bins = np.linspace(
        min(trained_metrics["tangent_speed"].min(),
            init_metrics["tangent_speed"].min()),
        max(trained_metrics["tangent_speed"].max(),
            init_metrics["tangent_speed"].max()),
        80,
    )
    axes[0, 0].hist(trained_metrics["tangent_speed"], bins=bins,
                    histtype="step", linewidth=1.5,
                    density=True, label=trained_label)
    axes[0, 0].hist(init_metrics["tangent_speed"], bins=bins,
                    histtype="step", linewidth=1.5,
                    density=True, label=init_label)
    axes[0, 0].set_xlabel("tangent angular speed")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].set_title("Per-unit tangent speed")
    axes[0, 0].legend()

    # power-weighted mean
    bins = np.linspace(
        min(trained_metrics["pw_mean_freq"].min(),
            init_metrics["pw_mean_freq"].min()),
        max(trained_metrics["pw_mean_freq"].max(),
            init_metrics["pw_mean_freq"].max()),
        80,
    )
    axes[0, 1].hist(trained_metrics["pw_mean_freq"], bins=bins,
                    histtype="step", linewidth=1.5,
                    density=True, label=trained_label)
    axes[0, 1].hist(init_metrics["pw_mean_freq"], bins=bins,
                    histtype="step", linewidth=1.5,
                    density=True, label=init_label)
    axes[0, 1].set_xlabel("power-weighted mean freq")
    axes[0, 1].set_title("Per-unit pw mean frequency")
    axes[0, 1].legend()

    # mean |activation|
    bins = np.logspace(
        np.log10(max(min(trained_metrics["mean_abs"].min(),
                          init_metrics["mean_abs"].min()), 1e-8)),
        np.log10(max(trained_metrics["mean_abs"].max(),
                      init_metrics["mean_abs"].max()) + 1e-12),
        80,
    )
    axes[1, 0].hist(trained_metrics["mean_abs"], bins=bins,
                    histtype="step", linewidth=1.5,
                    density=True, label=trained_label)
    axes[1, 0].hist(init_metrics["mean_abs"], bins=bins,
                    histtype="step", linewidth=1.5,
                    density=True, label=init_label)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("mean |activation| (log)")
    axes[1, 0].set_title("Per-unit loudness")
    axes[1, 0].legend()

    # bin-1 coefficient magnitude
    trained_mag = np.abs(trained_metrics["bin1_coeff"])
    init_mag = np.abs(init_metrics["bin1_coeff"])
    bins = np.logspace(
        np.log10(max(min(trained_mag.min(), init_mag.min()), 1e-8)),
        np.log10(max(trained_mag.max(), init_mag.max()) + 1e-12),
        80,
    )
    axes[1, 1].hist(trained_mag, bins=bins,
                    histtype="step", linewidth=1.5,
                    density=True, label=trained_label)
    axes[1, 1].hist(init_mag, bins=bins,
                    histtype="step", linewidth=1.5,
                    density=True, label=init_label)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("|bin-1 coefficient| (log)")
    axes[1, 1].set_title("Per-unit bin-1 oscillation strength")
    axes[1, 1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main per-model analysis
# ---------------------------------------------------------------------------

def analyze_model(model, tokenizer, texts, args, out_dir, label):
    """Run all analyses on a single model. Returns the metrics dict."""
    print(f"\n=== Analyzing {label} ===")
    rng = np.random.default_rng(SEED)

    print("Collecting streams ...")
    streams = collect_streams(model, tokenizer, texts, args.seq_len)
    print(f"Streams shape: {streams.shape}")

    print("Computing per-unit metrics ...")
    metrics = compute_per_unit_metrics(streams)
    d_model = streams.shape[2]

    print("Plotting frequency distributions ...")
    plot_frequency_distribution(
        metrics, save_path=out_dir / "frequency_distribution.png"
    )

    print(f"Sampling {args.n_pairs} pairs ...")
    u_idx, v_idx = sample_pair_indices(d_model, args.n_pairs, rng)

    print("Pairwise: trace correlation ...")
    trace_corr = trace_correlations(streams, u_idx, v_idx)

    print("Pairwise: cross-spectral at bin 1 ...")
    phase, mag = pair_cross_spectral(
        metrics["spec"], u_idx, v_idx, freq_bin=1
    )

    ts_dist = np.abs(metrics["tangent_speed"][u_idx] -
                      metrics["tangent_speed"][v_idx])
    pw_dist = np.abs(metrics["pw_mean_freq"][u_idx] -
                      metrics["pw_mean_freq"][v_idx])

    print("Plotting frequency distance distribution ...")
    plot_freq_distance_distribution(
        metrics, u_idx, v_idx,
        save_path=out_dir / "frequency_distance_distribution.png"
    )

    print("Plotting phase vs frequency distance ...")
    plot_phase_vs_freq_distance(
        phase, mag, ts_dist,
        save_path=out_dir / "phase_vs_freq_distance.png"
    )

    # Save per-unit metrics for cross-comparison
    np.savez(
        out_dir / "per_unit_metrics.npz",
        tangent_speed=metrics["tangent_speed"],
        pw_mean_freq=metrics["pw_mean_freq"],
        bin1_coeff=metrics["bin1_coeff"],
        mean_abs=metrics["mean_abs"],
    )

    summary = {
        "model_label": label,
        "n_samples": int(streams.shape[0]),
        "n_sublayers": int(streams.shape[1]),
        "d_model": int(d_model),
        "tangent_speed": {
            "mean": float(metrics["tangent_speed"].mean()),
            "std": float(metrics["tangent_speed"].std()),
            "min": float(metrics["tangent_speed"].min()),
            "max": float(metrics["tangent_speed"].max()),
            "median": float(np.median(metrics["tangent_speed"])),
        },
        "pw_mean_freq": {
            "mean": float(metrics["pw_mean_freq"].mean()),
            "std": float(metrics["pw_mean_freq"].std()),
            "median": float(np.median(metrics["pw_mean_freq"])),
        },
        "mean_abs": {
            "mean": float(metrics["mean_abs"].mean()),
            "std": float(metrics["mean_abs"].std()),
            "median": float(np.median(metrics["mean_abs"])),
        },
        "bin1_coeff_magnitude": {
            "mean": float(np.abs(metrics["bin1_coeff"]).mean()),
            "std": float(np.abs(metrics["bin1_coeff"]).std()),
            "median": float(np.median(np.abs(metrics["bin1_coeff"]))),
        },
        "ts_distance": {
            "mean": float(ts_dist.mean()),
            "median": float(np.median(ts_dist)),
            "p95": float(np.percentile(ts_dist, 95)),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return metrics, summary


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
    parser.add_argument("--skip-init", action="store_true",
                        help="Skip random-init run (faster)")
    args = parser.parse_args()

    base_out = Path(f"train_vs_init_out_{slugify(args.model)}")
    base_out.mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("Loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]

    print(f"Loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Trained model
    print(f"\n--- Trained model ---")
    print(f"Loading {args.model} (trained) ...")
    trained_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(DEVICE)
    trained_dir = base_out / "trained"
    trained_dir.mkdir(exist_ok=True)
    trained_metrics, trained_summary = analyze_model(
        trained_model, tokenizer, texts, args, trained_dir, "trained"
    )
    del trained_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.skip_init:
        print("\nSkipping random-init as requested.")
        return

    # Random-init model
    print(f"\n--- Random-init model ---")
    print(f"Loading {args.model} config and instantiating with random weights ...")
    config = AutoConfig.from_pretrained(args.model)
    init_model = AutoModelForCausalLM.from_config(config).to(DEVICE)
    init_model = init_model.float()
    init_dir = base_out / "random_init"
    init_dir.mkdir(exist_ok=True)
    init_metrics, init_summary = analyze_model(
        init_model, tokenizer, texts, args, init_dir, "random_init"
    )
    del init_model

    # Comparison plot
    print("\nMaking comparison plot ...")
    plot_comparison(
        trained_metrics, init_metrics,
        "trained", "random_init",
        save_path=base_out / "trained_vs_init_comparison.png"
    )

    # Combined summary
    combined = {
        "model": args.model,
        "trained": trained_summary,
        "random_init": init_summary,
    }
    with open(base_out / "combined_summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("\n--- Trained vs Random Comparison ---")
    print(f"Tangent speed: trained "
          f"{trained_summary['tangent_speed']['mean']:.3f} "
          f"(std {trained_summary['tangent_speed']['std']:.3f}) "
          f"vs random "
          f"{init_summary['tangent_speed']['mean']:.3f} "
          f"(std {init_summary['tangent_speed']['std']:.3f})")
    print(f"Bin-1 coeff magnitude: trained "
          f"{trained_summary['bin1_coeff_magnitude']['mean']:.4f} "
          f"vs random "
          f"{init_summary['bin1_coeff_magnitude']['mean']:.4f}")
    print(f"Mean |activation|: trained "
          f"{trained_summary['mean_abs']['mean']:.3f} "
          f"vs random "
          f"{init_summary['mean_abs']['mean']:.3f}")
    print(f"\nOutputs in {base_out.resolve()}")


if __name__ == "__main__":
    main()