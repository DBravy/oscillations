"""
Compute Fernando-style rotation count R for each residual unit and
compare it to the FFT-based observables we've been using.

For each unit, we compute:
  - R: cumulative tangent-vector turning, divided by 2π, exactly as
       described in Fernando & Guitchounts 2025 §2.6
  - dominant FFT bin: where the strongest non-DC power lies
  - power-weighted mean frequency: Σ k · P(k) / Σ P(k), over k≥1
  - full power spectrum

We then ask:
  1. What's the distribution of R in our data, and does it match
     Fernando's reported mean of ~10.74?
  2. How does R relate to the dominant FFT bin? They should diverge
     if higher-frequency content is contributing to R but not to
     the bin-1-dominated FFT readout.
  3. What does the per-unit power spectrum look like? Is bin 1 a
     sharp peak with little else, or is there substantial power
     spread across higher bins?

Outputs in rotations_out_<model>/:
  - rotation_distribution.png   histogram of R, overlaid with the
                                shuffle-null distribution
  - rotation_vs_dom_freq.png    scatter of R vs dominant bin
  - rotation_vs_powmean.png     scatter of R vs power-weighted
                                mean frequency
  - mean_power_spectrum.png     average power spectrum across units
                                (log y-axis), showing how power
                                distributes across bins
  - example_traces_and_portraits.png  for a few units of varying R,
                                show the raw trace and phase portrait
                                side by side
  - rotations_summary.json
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
N_SHUFFLE_RUNS = 200  # per-unit shuffle null (Fernando used 1000)


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
# Fernando's R, exactly as described
# ---------------------------------------------------------------------------

def rotation_count(trace):
    """
    Compute the rotation count R for a single 1D trace across sublayers,
    following Fernando & Guitchounts §2.6.

    Steps:
      x_ℓ = a_ℓ - a_0
      y_ℓ = ∇a_ℓ - ∇a_0
      Δx_ℓ = x_{ℓ+1} - x_ℓ
      Δy_ℓ = y_{ℓ+1} - y_ℓ
      θ_ℓ = atan2(Δy_ℓ, Δx_ℓ)
      Δθ_ℓ = θ_{ℓ+1} - θ_ℓ, wrapped to (-π, π]
      R = (1/2π) Σ Δθ_ℓ
    """
    a = trace
    grad_a = np.gradient(a)
    x = a - a[0]
    y = grad_a - grad_a[0]
    dx = np.diff(x)
    dy = np.diff(y)
    theta = np.arctan2(dy, dx)
    dtheta = np.diff(theta)
    # Wrap into (-π, π]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(dtheta) / (2 * np.pi))


def per_unit_rotation_counts(streams):
    """
    For each unit, compute R per sample and then the mean across samples.
    Returns: (d_model,) mean R, and (n_samples, d_model) per-sample R.
    """
    n_samples, n_sub, d_model = streams.shape
    R_per_sample = np.zeros((n_samples, d_model))
    for s in range(n_samples):
        for u in range(d_model):
            R_per_sample[s, u] = rotation_count(streams[s, :, u])
    return R_per_sample.mean(axis=0), R_per_sample


def shuffle_null_R(streams, n_runs, rng):
    """
    Per-unit null distribution: shuffle the layer ordering and
    recompute R. Average across samples per shuffle, then aggregate
    across shuffles. Returns (n_runs, d_model) of mean-across-samples R.
    """
    n_samples, n_sub, d_model = streams.shape
    null_means = np.zeros((n_runs, d_model))
    for r in range(n_runs):
        perm = rng.permutation(n_sub)
        shuffled = streams[:, perm, :]
        R_per_sample = np.zeros((n_samples, d_model))
        for s in range(n_samples):
            for u in range(d_model):
                R_per_sample[s, u] = rotation_count(shuffled[s, :, u])
        null_means[r] = R_per_sample.mean(axis=0)
    return null_means


# ---------------------------------------------------------------------------
# Spectral observables for comparison
# ---------------------------------------------------------------------------

def per_unit_power_spectrum(streams):
    """
    Mean power spectrum across samples, per unit.
    Returns: (n_freq, d_model) where n_freq = n_sub // 2 + 1.
    DC bin (k=0) is set to zero after mean-subtraction.
    """
    ms = streams - streams.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(ms, axis=1)
    power = np.mean(np.abs(spec) ** 2, axis=0)  # (n_freq, d_model)
    return power


def dominant_bin(power):
    """Per-unit dominant non-DC FFT bin."""
    if power.shape[0] <= 1:
        return np.zeros(power.shape[1], dtype=int)
    return np.argmax(power[1:], axis=0) + 1


def power_weighted_mean_freq(power):
    """Per-unit Σ k · P(k) / Σ P(k), over k >= 1."""
    n_freq, d_model = power.shape
    ks = np.arange(n_freq).reshape(-1, 1)
    ac = power[1:]
    ks_ac = ks[1:]
    denom = ac.sum(axis=0)
    denom = np.where(denom > 0, denom, 1.0)
    return (ks_ac * ac).sum(axis=0) / denom


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_rotation_distribution(R_real, null_means, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(
        min(R_real.min(), null_means.min()) - 0.5,
        max(R_real.max(), null_means.max()) + 0.5,
        80,
    )
    ax.hist(R_real, bins=bins, histtype="step", linewidth=1.8,
            density=True, label=f"real (mean={R_real.mean():.2f})")
    ax.hist(null_means.flatten(), bins=bins, histtype="step",
            linewidth=1.5, density=True, color="gray",
            label=f"shuffle null "
                  f"(mean={null_means.mean():.2f})")
    ax.set_xlabel("rotation count R")
    ax.set_ylabel("density")
    ax.set_title("Per-unit rotation count: real vs shuffle null")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_rotation_vs_dom_freq(R_real, dom_freq, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(dom_freq, R_real, s=8, alpha=0.5)
    ax.set_xlabel("dominant FFT bin")
    ax.set_ylabel("rotation count R")
    ax.set_title("R vs dominant FFT bin (per unit)")
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_rotation_vs_powmean(R_real, pw_mean, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pw_mean, R_real, s=8, alpha=0.5)
    # Reference line y = x for visual comparison
    lim = max(pw_mean.max(), R_real.max())
    ax.plot([0, lim], [0, lim], "--", color="gray",
            linewidth=0.7, label="y = x")
    ax.set_xlabel("power-weighted mean frequency")
    ax.set_ylabel("rotation count R")
    ax.set_title("R vs power-weighted mean frequency (per unit)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_mean_power_spectrum(power, save_path=None):
    """
    Show the mean (across units) power spectrum and a few example
    per-unit spectra.
    """
    n_freq = power.shape[0]
    ks = np.arange(n_freq)

    # Average across units
    mean_p = power.mean(axis=1)
    median_p = np.median(power, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(ks, mean_p, "o-", label="mean across units")
    axes[0].plot(ks, median_p, "s-", label="median across units")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("FFT bin")
    axes[0].set_ylabel("power (log)")
    axes[0].set_title("Average per-unit power spectrum")
    axes[0].legend()

    # A handful of example units
    rng = np.random.default_rng(0)
    sample_units = rng.choice(power.shape[1], size=12, replace=False)
    for u in sample_units:
        axes[1].plot(ks, power[:, u], alpha=0.5, linewidth=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("FFT bin")
    axes[1].set_ylabel("power (log)")
    axes[1].set_title("12 example per-unit power spectra")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_example_traces_and_portraits(streams, R_real, save_path=None):
    """
    For three units chosen across the R range (low, median, high),
    show the raw trace and the phase portrait side by side, for one
    representative sample.
    """
    order = np.argsort(R_real)
    selected = [order[len(order) // 10],         # low
                order[len(order) // 2],          # median
                order[-len(order) // 10]]        # high
    labels = ["low R", "median R", "high R"]

    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    sample_idx = 0
    sample = streams[sample_idx]
    n_sub = sample.shape[0]

    for row, (u, lbl) in enumerate(zip(selected, labels)):
        a = sample[:, u]
        ga = np.gradient(a)
        # Trace
        axes[row, 0].plot(a, "-", linewidth=1)
        axes[row, 0].set_title(f"{lbl}: unit {u}, R={R_real[u]:.2f} "
                               f"(sample 0 trace)", fontsize=10)
        axes[row, 0].set_xlabel("sublayer", fontsize=9)
        axes[row, 0].set_ylabel("a", fontsize=9)
        # Phase portrait
        x = a - a[0]
        y = ga - ga[0]
        axes[row, 1].plot(x, y, "-", linewidth=0.8, alpha=0.9)
        axes[row, 1].scatter(x, y, c=np.arange(n_sub),
                             cmap="viridis", s=12)
        axes[row, 1].set_xlabel("a - a_0", fontsize=9)
        axes[row, 1].set_ylabel("∇a - ∇a_0", fontsize=9)
        axes[row, 1].set_title(f"{lbl}: phase portrait", fontsize=10)
        axes[row, 1].axhline(0, color="gray", linewidth=0.5)
        axes[row, 1].axvline(0, color="gray", linewidth=0.5)

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
    parser.add_argument("--n-shuffle", type=int, default=N_SHUFFLE_RUNS)
    args = parser.parse_args()

    out_dir = Path(f"rotations_out_{slugify(args.model)}")
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
    print(f"d_model={d_model}")

    print("Loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]

    print("Collecting streams ...")
    streams = collect_streams(model, tokenizer, texts, args.seq_len)
    print(f"Streams shape: {streams.shape}")

    print("Computing per-unit rotation counts (real) ...")
    R_real, R_per_sample = per_unit_rotation_counts(streams)
    print(f"R range: [{R_real.min():.2f}, {R_real.max():.2f}], "
          f"mean = {R_real.mean():.2f}, "
          f"median = {np.median(R_real):.2f}")

    print(f"Computing shuffle null (n_runs={args.n_shuffle}) ...")
    null_means = shuffle_null_R(streams, args.n_shuffle, rng)
    print(f"Null mean: {null_means.mean():.2f}, "
          f"std: {null_means.std():.2f}")

    print("Computing power spectra ...")
    power = per_unit_power_spectrum(streams)
    dom_freq = dominant_bin(power)
    pw_mean = power_weighted_mean_freq(power)

    print("Saving plots ...")
    plot_rotation_distribution(
        R_real, null_means,
        save_path=out_dir / "rotation_distribution.png",
    )
    plot_rotation_vs_dom_freq(
        R_real, dom_freq,
        save_path=out_dir / "rotation_vs_dom_freq.png",
    )
    plot_rotation_vs_powmean(
        R_real, pw_mean,
        save_path=out_dir / "rotation_vs_powmean.png",
    )
    plot_mean_power_spectrum(
        power,
        save_path=out_dir / "mean_power_spectrum.png",
    )
    plot_example_traces_and_portraits(
        streams, R_real,
        save_path=out_dir / "example_traces_and_portraits.png",
    )

    summary = {
        "model": args.model,
        "n_samples": int(streams.shape[0]),
        "n_sublayers": int(streams.shape[1]),
        "d_model": int(d_model),
        "rotation_count_R": {
            "mean": float(R_real.mean()),
            "median": float(np.median(R_real)),
            "std": float(R_real.std()),
            "min": float(R_real.min()),
            "max": float(R_real.max()),
        },
        "shuffle_null_R": {
            "mean": float(null_means.mean()),
            "std": float(null_means.std()),
            "abs_mean": float(np.abs(null_means).mean()),
        },
        "dominant_bin": {
            "mean": float(dom_freq.mean()),
            "median": float(np.median(dom_freq)),
            "std": float(dom_freq.std()),
        },
        "power_weighted_mean_freq": {
            "mean": float(pw_mean.mean()),
            "median": float(np.median(pw_mean)),
            "std": float(pw_mean.std()),
        },
        "correlations": {
            "R_vs_dominant_bin": float(np.corrcoef(R_real, dom_freq)[0, 1]),
            "R_vs_powmean": float(np.corrcoef(R_real, pw_mean)[0, 1]),
        },
        "mean_power_per_bin": power.mean(axis=1).tolist(),
    }
    with open(out_dir / "rotations_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(f"Mean rotation count R: {summary['rotation_count_R']['mean']:.2f}")
    print(f"Median rotation count R: "
          f"{summary['rotation_count_R']['median']:.2f}")
    print(f"Shuffle-null mean R: {summary['shuffle_null_R']['mean']:.2f} "
          f"(std {summary['shuffle_null_R']['std']:.3f})")
    print(f"Mean dominant bin: {summary['dominant_bin']['mean']:.2f}")
    print(f"Mean power-weighted mean freq: "
          f"{summary['power_weighted_mean_freq']['mean']:.2f}")
    print(f"R vs dominant bin correlation: "
          f"{summary['correlations']['R_vs_dominant_bin']:.3f}")
    print(f"R vs power-weighted mean freq correlation: "
          f"{summary['correlations']['R_vs_powmean']:.3f}")
    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()