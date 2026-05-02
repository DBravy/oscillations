"""
Comparison of trained vs random-init residual stream dynamics.

Runs the same set of phase analyses on both, using consistent
methodology so the results can be compared directly.

For each model:
  - Phase portraits of selected units across the rotation count range
  - Phase difference distribution by proximity regime
  - Cospec heatmap (2D hist of phase vs log magnitude) in the furthest regime
  - Phase difference at a fixed bin (bin 1) for all pairs
  - Cospec magnitude distribution by regime
  - Trace correlation distribution

Outputs in compare_out_<model>/:
  - phase_portraits.png        side by side, trained vs init
  - phase_diff_by_proximity.png  three regimes, trained and init overlaid
  - cospec_heatmap.png         furthest regime heatmap, side by side
  - phase_diff_bin1.png        all pairs, fixed bin
  - cospec_distribution.png    by regime, trained and init
  - trace_corr_distribution.png  for context
  - compare_summary.json
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


# ---------------------------------------------------------------------------
# Boilerplate
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
# Per-model analysis
# ---------------------------------------------------------------------------

def rotation_count_per_unit(streams):
    """
    Compute Fernando-style R per unit (mean across samples).
    Vectorized.
    """
    n_samples, n_sub, d_model = streams.shape
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    theta = np.arctan2(dy, dx)
    dtheta = np.diff(theta, axis=1)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    R_per_sample = dtheta.sum(axis=1) / (2 * np.pi)
    return R_per_sample.mean(axis=0)  # (d_model,)


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


def analyze(model, tokenizer, texts, args):
    """Run the analysis for one model. Returns a dict of computed
    arrays for use in cross-model plotting."""
    print("Collecting streams ...")
    streams = collect_streams(model, tokenizer, texts, args.seq_len)
    print(f"Streams shape: {streams.shape}")

    print("Computing rotation counts ...")
    R = rotation_count_per_unit(streams)

    print("Computing spectra ...")
    spectra = per_unit_spectra(streams)

    print(f"Sampling {args.n_pairs} pairs ...")
    rng = np.random.default_rng(SEED)
    u_idx, v_idx = sample_pair_indices(streams.shape[2], args.n_pairs, rng)

    print("Trace correlations ...")
    trace_corr = trace_correlations(streams, u_idx, v_idx)

    print("Cross-spectral at bin 1 ...")
    phase_bin1, mag_bin1 = pair_cross_spectral(
        spectra, u_idx, v_idx, freq_bin=1
    )

    return {
        "streams": streams,
        "R": R,
        "spectra": spectra,
        "u_idx": u_idx,
        "v_idx": v_idx,
        "trace_corr": trace_corr,
        "phase_bin1": phase_bin1,
        "mag_bin1": mag_bin1,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_phase_portraits(trained, init, save_path=None):
    """
    Six units total: low-R, median-R, high-R from each model.
    Two columns (trained, init), three rows (low/med/high R).
    """
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))

    sample_idx = 0
    for col, (data, label) in enumerate([(trained, "trained"),
                                          (init, "random_init")]):
        order = np.argsort(np.abs(data["R"]))
        selected = [order[len(order) // 10],
                    order[len(order) // 2],
                    order[-len(order) // 10]]
        labels = ["low |R|", "median |R|", "high |R|"]

        sample = data["streams"][sample_idx]
        n_sub = sample.shape[0]
        for row, (u, lbl) in enumerate(zip(selected, labels)):
            ax = axes[row, col]
            a = sample[:, u]
            ga = np.gradient(a)
            x = a - a[0]
            y = ga - ga[0]
            ax.plot(x, y, "-", linewidth=0.8, alpha=0.9)
            ax.scatter(x, y, c=np.arange(n_sub),
                       cmap="viridis", s=12)
            ax.set_title(
                f"{label}, {lbl}: unit {u}, R={data['R'][u]:.2f}",
                fontsize=10,
            )
            ax.set_xlabel("a - a_0", fontsize=9)
            ax.set_ylabel("∇a - ∇a_0", fontsize=9)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_diff_by_proximity(trained, init, save_path=None):
    """
    For each model, three regimes: closest, middle, furthest.
    Plot phase-difference density curves. Trained and init overlaid
    in the same panel for direct comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for data, label, style in [(trained, "trained", "-"),
                                (init, "random_init", "--")]:
        prox = 1.0 - np.abs(data["trace_corr"])
        edges = np.quantile(prox, [0, 1/3, 2/3, 1])
        for ax, (regime_label, mask) in zip(axes, [
            ("closest",  prox <= edges[1]),
            ("middle",   (prox > edges[1]) & (prox <= edges[2])),
            ("furthest", prox > edges[2]),
        ]):
            p = data["phase_bin1"][mask]
            ax.hist(p, bins=80, range=(-np.pi, np.pi),
                    density=True, histtype="step", linewidth=1.5,
                    label=f"{label}", linestyle=style)
            ax.set_title(regime_label)
            ax.set_xlabel("phase difference at bin 1")
            for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
                ax.axvline(x, color="gray", linewidth=0.4,
                           linestyle=":")
    axes[0].set_ylabel("density")
    for ax in axes:
        ax.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_cospec_heatmap(trained, init, save_path=None):
    """
    2D histogram of (phase, log magnitude), restricted to furthest
    regime, side by side for trained and init.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (data, label) in zip(axes, [(trained, "trained"),
                                          (init, "random_init")]):
        prox = 1.0 - np.abs(data["trace_corr"])
        edges = np.quantile(prox, [0, 1/3, 2/3, 1])
        furthest = prox > edges[2]
        p = data["phase_bin1"][furthest]
        m = data["mag_bin1"][furthest]
        m_floor = max(np.percentile(m, 1), 1e-10)
        keep = m >= m_floor
        p = p[keep]
        m = m[keep]
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
        ax.set_xlabel("phase difference (radians)")
        ax.set_ylabel("log10(cospec magnitude)")
        ax.set_title(f"{label} — furthest regime, n={len(p)}")
        for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
            ax.axvline(x, color="white", linewidth=0.5,
                       linestyle="--", alpha=0.5)
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_diff_bin1(trained, init, save_path=None):
    """All pairs, phase difference at bin 1 (no regime split)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for data, label, style in [(trained, "trained", "-"),
                                (init, "random_init", "--")]:
        ax.hist(data["phase_bin1"], bins=80, range=(-np.pi, np.pi),
                density=True, histtype="step", linewidth=1.5,
                label=label, linestyle=style)
    for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
        ax.axvline(x, color="gray", linewidth=0.4, linestyle=":")
    ax.set_xlabel("phase difference at bin 1 (all pairs)")
    ax.set_ylabel("density")
    ax.set_title("Phase difference distribution: all pairs at bin 1")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_cospec_distribution(trained, init, save_path=None):
    """Cospec magnitude distribution by regime, both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (data, label) in zip(axes, [(trained, "trained"),
                                          (init, "random_init")]):
        prox = 1.0 - np.abs(data["trace_corr"])
        edges = np.quantile(prox, [0, 1/3, 2/3, 1])
        m = data["mag_bin1"]
        bins = np.logspace(
            np.log10(max(m.min(), 1e-10)),
            np.log10(m.max() + 1e-12),
            60,
        )
        for regime_label, mask in [
            ("closest", prox <= edges[1]),
            ("middle", (prox > edges[1]) & (prox <= edges[2])),
            ("furthest", prox > edges[2]),
        ]:
            ax.hist(m[mask], bins=bins, histtype="step",
                    linewidth=1.5, density=True, label=regime_label)
        ax.set_xscale("log")
        ax.set_xlabel("cospec magnitude (log)")
        ax.set_ylabel("density")
        ax.set_title(f"{label}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_trace_corr_distribution(trained, init, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for data, label, style in [(trained, "trained", "-"),
                                (init, "random_init", "--")]:
        ax.hist(data["trace_corr"], bins=80, range=(-1, 1),
                density=True, histtype="step", linewidth=1.5,
                label=label, linestyle=style)
    ax.set_xlabel("trace correlation")
    ax.set_ylabel("density")
    ax.set_title("Pairwise trace correlation distribution")
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_R_distribution(trained, init, save_path=None):
    """Per-unit rotation count distribution."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for data, label, style in [(trained, "trained", "-"),
                                (init, "random_init", "--")]:
        ax.hist(data["R"], bins=80, density=True,
                histtype="step", linewidth=1.5,
                label=f"{label} (mean={data['R'].mean():.2f}, "
                      f"std={data['R'].std():.2f})",
                linestyle=style)
    ax.set_xlabel("rotation count R")
    ax.set_ylabel("density")
    ax.set_title("Per-unit rotation count")
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


def summarize(data, label):
    """Compact summary stats for the JSON dump."""
    R = data["R"]
    tc = data["trace_corr"]
    p = data["phase_bin1"]
    m = data["mag_bin1"]

    # phase clustering at canonical values, weighted by mag
    def near(target, band=QUAD_BAND):
        if target == "pi":
            return (np.abs(p - np.pi) < band) | (np.abs(p + np.pi) < band)
        return np.abs(p - target) < band

    return {
        "label": label,
        "n_units": int(R.shape[0]),
        "n_pairs": int(p.shape[0]),
        "rotation_count": {
            "mean": float(R.mean()),
            "std": float(R.std()),
            "median": float(np.median(R)),
            "abs_mean": float(np.abs(R).mean()),
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
            "near_0": float(m[near(0)].mean()) if near(0).any() else None,
            "near_pi": float(m[near("pi")].mean()) if near("pi").any() else None,
            "near_pi_over_2": float(m[near(np.pi / 2)].mean()) if near(np.pi / 2).any() else None,
            "near_minus_pi_over_2": float(m[near(-np.pi / 2)].mean()) if near(-np.pi / 2).any() else None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    args = parser.parse_args()

    out_dir = Path(f"compare_out_{slugify(args.model)}")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("Loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]

    print(f"Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Trained
    print("\n=== TRAINED ===")
    trained_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(DEVICE)
    trained = analyze(trained_model, tokenizer, texts, args)
    del trained_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Random init
    print("\n=== RANDOM INIT ===")
    config = AutoConfig.from_pretrained(args.model)
    init_model = AutoModelForCausalLM.from_config(config).to(DEVICE).float()
    init = analyze(init_model, tokenizer, texts, args)
    del init_model

    print("\nMaking plots ...")
    plot_phase_portraits(
        trained, init, save_path=out_dir / "phase_portraits.png"
    )
    plot_phase_diff_by_proximity(
        trained, init, save_path=out_dir / "phase_diff_by_proximity.png"
    )
    plot_cospec_heatmap(
        trained, init, save_path=out_dir / "cospec_heatmap.png"
    )
    plot_phase_diff_bin1(
        trained, init, save_path=out_dir / "phase_diff_bin1.png"
    )
    plot_cospec_distribution(
        trained, init, save_path=out_dir / "cospec_distribution.png"
    )
    plot_trace_corr_distribution(
        trained, init, save_path=out_dir / "trace_corr_distribution.png"
    )
    plot_R_distribution(
        trained, init, save_path=out_dir / "R_distribution.png"
    )

    summary = {
        "model": args.model,
        "trained": summarize(trained, "trained"),
        "random_init": summarize(init, "random_init"),
    }
    with open(out_dir / "compare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    for label, s in [("trained", summary["trained"]),
                     ("random_init", summary["random_init"])]:
        print(f"\n[{label}]")
        print(f"  R: mean {s['rotation_count']['mean']:.2f}, "
              f"std {s['rotation_count']['std']:.2f}")
        print(f"  Cospec mag: mean {s['cospec_mag']['mean']:.4f}, "
              f"p95 {s['cospec_mag']['p95']:.4f}")
        print(f"  Phase clustering fractions: "
              f"near_0={s['phase_clustering_fractions']['near_0']:.3f}, "
              f"near_π/2={s['phase_clustering_fractions']['near_pi_over_2']:.3f}, "
              f"near_-π/2={s['phase_clustering_fractions']['near_minus_pi_over_2']:.3f}, "
              f"near_π={s['phase_clustering_fractions']['near_pi']:.3f}")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()