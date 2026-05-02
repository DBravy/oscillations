"""
Compare cumulative residual stream analyses to combined-delta analyses.

For a single model, we compute:
  - h(ℓ): cumulative residual stream values across sublayers
  - Δ(ℓ) = h(ℓ+1) - h(ℓ): per-sublayer contributions

Run the same battery of analyses on both, side by side:
  - per-unit rotation count R
  - phase portraits at low/median/high R
  - pairwise phase difference distribution by proximity regime
  - cospec heatmap (phase vs log magnitude) in furthest regime
  - cospec magnitude distribution by regime
  - trace correlation distribution

Outputs in delta_compare_out_<model>/:
  - phase_portraits.png
  - phase_diff_by_proximity.png
  - cospec_heatmap.png
  - cospec_distribution.png
  - trace_corr_distribution.png
  - R_distribution.png
  - delta_compare_summary.json
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
# Per-unit / per-pair analyses
# ---------------------------------------------------------------------------

def rotation_count_per_unit(streams):
    """
    Compute Fernando-style R per unit (mean across samples), vectorized.
    Works for either cumulative h or deltas; just operates on whatever
    is passed in.
    """
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    theta = np.arctan2(dy, dx)
    dtheta = np.diff(theta, axis=1)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    R_per_sample = dtheta.sum(axis=1) / (2 * np.pi)
    return R_per_sample.mean(axis=0)


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


def analyze(streams, label, args):
    """Run the standard analyses on a (n_samples, n_steps, d_model) tensor.
    streams can be either h (cumulative) or Δ (deltas)."""
    print(f"  {label}: shape {streams.shape}")
    rng = np.random.default_rng(SEED)
    d_model = streams.shape[2]

    print(f"  computing R ...")
    R = rotation_count_per_unit(streams)

    print(f"  computing spectra ...")
    spectra = per_unit_spectra(streams)

    print(f"  sampling {args.n_pairs} pairs ...")
    u_idx, v_idx = sample_pair_indices(d_model, args.n_pairs, rng)

    print(f"  trace correlations ...")
    trace_corr = trace_correlations(streams, u_idx, v_idx)

    print(f"  cross-spectral at bin 1 ...")
    phase_bin1, mag_bin1 = pair_cross_spectral(
        spectra, u_idx, v_idx, freq_bin=1
    )

    return {
        "label": label,
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
# Plots (reused from compare script, adapted to label "cumulative" vs "delta")
# ---------------------------------------------------------------------------

def plot_phase_portraits(cum, delta, save_path=None):
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    sample_idx = 0
    for col, data in enumerate([cum, delta]):
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
            ax.plot(x, y, "-", linewidth=0.8, alpha=0.9)
            ax.scatter(x, y, c=np.arange(n_steps),
                       cmap="viridis", s=12)
            ax.set_title(
                f"{data['label']}, {lbl}: unit {u}, "
                f"R={data['R'][u]:.2f}",
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


def plot_phase_diff_by_proximity(cum, delta, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for data, style in [(cum, "-"), (delta, "--")]:
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
                    label=data["label"], linestyle=style)
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


def plot_cospec_heatmap(cum, delta, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data in zip(axes, [cum, delta]):
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
        ax.set_title(f"{data['label']} — furthest regime, n={len(p)}")
        for x in (-np.pi, -np.pi/2, 0, np.pi/2, np.pi):
            ax.axvline(x, color="white", linewidth=0.5,
                       linestyle="--", alpha=0.5)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_cospec_distribution(cum, delta, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data in zip(axes, [cum, delta]):
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
        ax.set_title(data["label"])
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_trace_corr_distribution(cum, delta, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for data, style in [(cum, "-"), (delta, "--")]:
        ax.hist(data["trace_corr"], bins=80, range=(-1, 1),
                density=True, histtype="step", linewidth=1.5,
                label=data["label"], linestyle=style)
    ax.set_xlabel("trace correlation")
    ax.set_ylabel("density")
    ax.set_title("Pairwise trace correlation distribution")
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_R_distribution(cum, delta, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for data, style in [(cum, "-"), (delta, "--")]:
        ax.hist(data["R"], bins=80, density=True,
                histtype="step", linewidth=1.5,
                label=f"{data['label']} (mean={data['R'].mean():.2f}, "
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


def summarize(data):
    R = data["R"]
    p = data["phase_bin1"]
    m = data["mag_bin1"]

    def near(target, band=QUAD_BAND):
        if target == "pi":
            return (np.abs(p - np.pi) < band) | (np.abs(p + np.pi) < band)
        return np.abs(p - target) < band

    return {
        "label": data["label"],
        "n_steps": int(data["streams"].shape[1]),
        "R": {
            "mean": float(R.mean()),
            "std": float(R.std()),
            "median": float(np.median(R)),
            "abs_mean": float(np.abs(R).mean()),
        },
        "trace_corr": {
            "mean": float(data["trace_corr"].mean()),
            "std": float(data["trace_corr"].std()),
            "abs_mean": float(np.abs(data["trace_corr"]).mean()),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    args = parser.parse_args()

    out_dir = Path(f"delta_compare_out_{slugify(args.model)}")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("Loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(DEVICE)

    print("\nCollecting streams ...")
    streams = collect_streams(model, tokenizer, texts, args.seq_len)
    print(f"streams shape: {streams.shape}")

    print("Computing combined deltas ...")
    deltas = np.diff(streams, axis=1)
    print(f"deltas shape: {deltas.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== CUMULATIVE ===")
    cum = analyze(streams, "cumulative", args)
    print("\n=== DELTAS ===")
    delta = analyze(deltas, "delta", args)

    print("\nMaking plots ...")
    plot_phase_portraits(
        cum, delta, save_path=out_dir / "phase_portraits.png"
    )
    plot_phase_diff_by_proximity(
        cum, delta, save_path=out_dir / "phase_diff_by_proximity.png"
    )
    plot_cospec_heatmap(
        cum, delta, save_path=out_dir / "cospec_heatmap.png"
    )
    plot_cospec_distribution(
        cum, delta, save_path=out_dir / "cospec_distribution.png"
    )
    plot_trace_corr_distribution(
        cum, delta, save_path=out_dir / "trace_corr_distribution.png"
    )
    plot_R_distribution(
        cum, delta, save_path=out_dir / "R_distribution.png"
    )

    summary = {
        "model": args.model,
        "cumulative": summarize(cum),
        "delta": summarize(delta),
    }
    with open(out_dir / "delta_compare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    for key in ("cumulative", "delta"):
        s = summary[key]
        print(f"\n[{key}] (n_steps={s['n_steps']})")
        print(f"  R: mean {s['R']['mean']:.2f}, "
              f"std {s['R']['std']:.2f}, "
              f"abs_mean {s['R']['abs_mean']:.2f}")
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