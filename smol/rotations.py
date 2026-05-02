"""
Exploratory: what geometric structure underlies per-unit rotations in
the residual stream?

We already know that individual units rotate in (a, da/dl) phase space
across sublayers. This script asks: are those per-unit rotations
projections of one underlying low-dim rotation in d-space, or many
unrelated rotations? Three analyses:

  1. SVD of the mean trajectory (across samples), with each sublayer
     normalized to unit norm. Shows the dominant rotation plane and how
     much of the trajectory's variance lives there.
  2. Sample-resolved trajectories projected into the mean basis. Tells
     us whether individual samples follow the mean arc or scatter.
  3. Per-unit phase extraction at the dominant FFT bin. Tells us
     whether unit phases cluster (one underlying rotation) or are
     uniform (many independent rotations).

Outputs in svd_out_<model>/:
  - svd_trajectory.png     mean trajectory in top-2 SVD plane
  - svd_spectrum.png       singular value spectrum
  - sample_overlay.png     individual samples in the mean basis
  - unit_phases.png        per-unit phase distribution at dominant bin
  - svd_summary.json
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


DEFAULT_MODEL = "gpt2"
N_SAMPLES = 256          # more samples for cleaner mean
SEQ_LEN = 128
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Architecture detection (same as before)
# ---------------------------------------------------------------------------

def get_hook_targets(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
        attn_ln_attr = "ln_1"
        mlp_ln_attr = "ln_2"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
        attn_ln_attr = "input_layernorm"
        mlp_ln_attr = "post_attention_layernorm"
    else:
        raise RuntimeError("Could not locate transformer blocks.")

    targets = []
    for i, block in enumerate(blocks):
        targets.append((i, "pre_attn", getattr(block, attn_ln_attr)))
        targets.append((i, "pre_mlp", getattr(block, mlp_ln_attr)))
    return targets


def get_d_model(model):
    cfg = model.config
    for attr in ("n_embd", "hidden_size", "d_model"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise RuntimeError("Could not infer d_model.")


def get_n_layers(model):
    cfg = model.config
    for attr in ("n_layer", "num_hidden_layers"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise RuntimeError("Could not infer n_layers.")


def collect_residual_stream(model, tokenizer, texts, seq_len):
    model.eval()
    n_layers = get_n_layers(model)
    targets = get_hook_targets(model)
    expected = 2 * n_layers

    all_streams = []
    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt",
            max_length=seq_len, truncation=True,
        )
        input_ids = enc["input_ids"].to(DEVICE)
        if input_ids.shape[1] < 8:
            continue

        captures = []
        hooks = []
        for (_, _, ln_module) in targets:
            def make_hook():
                def hook(module, inp, out):
                    captures.append(inp[0].detach())
                return hook
            hooks.append(ln_module.register_forward_hook(make_hook()))

        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        last_token_states = torch.stack(
            [c[0, -1, :].float() for c in captures], dim=0
        )
        all_streams.append(last_token_states.cpu().numpy())

    return np.stack(all_streams, axis=0)


# ---------------------------------------------------------------------------
# Analysis: separating direction from magnitude
# ---------------------------------------------------------------------------

def normalize_per_sublayer(streams):
    """
    Normalize each sublayer's vector to unit norm, per sample.
    This puts every state on the unit sphere so we can see direction
    independently of the magnitude growth Fernando reported.
    """
    norms = np.linalg.norm(streams, axis=2, keepdims=True)
    return streams / np.maximum(norms, 1e-8)


def mean_trajectory(streams_normed):
    """
    Mean over samples, per sublayer, then re-normalize.
    Shape: (n_sublayers, d_model)
    """
    mean = streams_normed.mean(axis=0)
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    return mean / np.maximum(norms, 1e-8)


def svd_of_trajectory(traj):
    """
    SVD of a (n_sublayers, d_model) trajectory after centering across
    sublayers. Returns U, S, Vt where Vt's first 2 rows are the basis
    of the dominant 2D plane in d_model space.
    """
    centered = traj - traj.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return U, S, Vt


def project(traj, Vt, k=2):
    """Project (n_sublayers, d_model) trajectory onto the first k right
    singular vectors. Returns (n_sublayers, k)."""
    centered = traj - traj.mean(axis=0, keepdims=True)
    return centered @ Vt[:k].T


def per_unit_phases(streams_normed, freq_bin=1):
    """
    For each unit u, compute the complex Fourier coefficient of its
    trace at the given frequency bin (across sublayers), averaged over
    samples. The argument of that coefficient is u's phase at that
    frequency; the magnitude is u's participation in that mode.

    Returns:
      phases: array of shape (d_model,) in (-pi, pi]
      magnitudes: array of shape (d_model,)
    """
    n_samples, n_sub, d_model = streams_normed.shape
    coeffs = np.zeros(d_model, dtype=complex)
    for u in range(d_model):
        # mean trace across samples
        trace = streams_normed[:, :, u].mean(axis=0)
        ms = trace - trace.mean()
        spec = np.fft.rfft(ms)
        coeffs[u] = spec[freq_bin]
    phases = np.angle(coeffs)
    magnitudes = np.abs(coeffs)
    return phases, magnitudes


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_svd_trajectory(traj_2d, save_path=None, title=""):
    """Plot the mean trajectory in top-2 SVD plane, colored by sublayer."""
    fig, ax = plt.subplots(figsize=(6, 6))
    n = traj_2d.shape[0]
    sc = ax.scatter(
        traj_2d[:, 0], traj_2d[:, 1],
        c=np.arange(n), cmap="viridis", s=40,
    )
    ax.plot(traj_2d[:, 0], traj_2d[:, 1], "-", linewidth=0.7, alpha=0.7)
    for i in (0, n - 1):
        ax.annotate(
            f"sublayer {i}",
            (traj_2d[i, 0], traj_2d[i, 1]),
            fontsize=9,
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("SVD component 1")
    ax.set_ylabel("SVD component 2")
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    fig.colorbar(sc, ax=ax, label="sublayer")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_svd_spectrum(S, save_path=None, k_show=20):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    k = min(k_show, len(S))
    axes[0].plot(np.arange(1, k + 1), S[:k], "o-")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("singular value index")
    axes[0].set_ylabel("singular value (log)")
    axes[0].set_title("SVD spectrum")

    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    axes[1].plot(np.arange(1, len(S) + 1), cumvar, "o-")
    axes[1].set_xlabel("number of components")
    axes[1].set_ylabel("cumulative explained variance")
    axes[1].set_title("Cumulative variance")
    axes[1].axhline(0.9, color="gray", linewidth=0.5, linestyle="--")
    axes[1].set_xlim(0, min(20, len(S)))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_sample_overlay(streams_normed, Vt, save_path=None, n_overlay=32):
    """Project each individual sample's trajectory onto the mean basis."""
    n_samples = min(n_overlay, streams_normed.shape[0])
    fig, ax = plt.subplots(figsize=(7, 7))
    for s in range(n_samples):
        traj = streams_normed[s]
        proj = project(traj, Vt, k=2)
        ax.plot(proj[:, 0], proj[:, 1], "-", linewidth=0.5, alpha=0.4)

    # Plot the mean on top
    mean_traj = mean_trajectory(streams_normed)
    mean_proj = project(mean_traj, Vt, k=2)
    n = mean_proj.shape[0]
    ax.scatter(
        mean_proj[:, 0], mean_proj[:, 1],
        c=np.arange(n), cmap="viridis", s=40, zorder=5,
    )
    ax.plot(
        mean_proj[:, 0], mean_proj[:, 1],
        "-", linewidth=1.5, color="black", alpha=0.7, zorder=4,
    )
    ax.set_xlabel("SVD component 1 (of mean)")
    ax.set_ylabel("SVD component 2 (of mean)")
    ax.set_title(
        f"Per-sample trajectories in mean basis "
        f"({n_samples} samples shown)"
    )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_unit_phases(phases, magnitudes, save_path=None):
    """
    Two views: histogram of phases (weighted by magnitude), and
    a polar scatter where each unit is a point at angle=phase,
    radius=magnitude.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Weighted histogram
    axes[0].hist(
        phases, bins=60, weights=magnitudes,
        range=(-np.pi, np.pi),
    )
    axes[0].set_xlabel("phase (radians)")
    axes[0].set_ylabel("magnitude-weighted count")
    axes[0].set_title("Per-unit phases at dominant FFT bin")
    axes[0].axvline(0, color="gray", linewidth=0.5)

    # Polar scatter
    ax_polar = plt.subplot(1, 2, 2, projection="polar")
    ax_polar.scatter(
        phases, magnitudes, s=4, alpha=0.4,
    )
    ax_polar.set_title("Phase / magnitude per unit (polar)")
    ax_polar.set_rticks([])

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
    args = parser.parse_args()

    out_dir = Path(f"svd_out_{slugify(args.model)}")
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
    print(f"d_model = {d_model}, n_layers = {n_layers}")

    print("Loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    candidate_texts = [
        x["text"] for x in ds
        if 200 < len(x["text"]) < 1500
    ]
    rng.shuffle(candidate_texts)
    texts = candidate_texts[:args.n_samples]
    print(f"Using {len(texts)} text samples.")

    print("Collecting residual stream ...")
    streams = collect_residual_stream(model, tokenizer, texts, args.seq_len)
    print(f"Stream shape: {streams.shape}")

    print("Normalizing each sublayer to unit norm ...")
    streams_normed = normalize_per_sublayer(streams)

    print("Computing mean trajectory and its SVD ...")
    mean_traj = mean_trajectory(streams_normed)
    U, S, Vt = svd_of_trajectory(mean_traj)
    traj_2d = project(mean_traj, Vt, k=2)

    print("Computing per-unit phases at dominant frequency bin ...")
    # Use bin 1 (slowest non-DC), since that was the dominant bin in
    # both models in the previous round
    phases, magnitudes = per_unit_phases(streams_normed, freq_bin=1)

    print("Saving figures ...")
    plot_svd_trajectory(
        traj_2d,
        save_path=out_dir / "svd_trajectory.png",
        title=f"{args.model} mean trajectory in top-2 SVD plane",
    )
    plot_svd_spectrum(
        S,
        save_path=out_dir / "svd_spectrum.png",
    )
    plot_sample_overlay(
        streams_normed, Vt,
        save_path=out_dir / "sample_overlay.png",
    )
    plot_unit_phases(
        phases, magnitudes,
        save_path=out_dir / "unit_phases.png",
    )

    # Summary stats
    var_explained_top2 = float(np.sum(S[:2] ** 2) / np.sum(S ** 2))
    var_explained_top5 = float(np.sum(S[:5] ** 2) / np.sum(S ** 2))
    n_for_90pct = int(
        np.searchsorted(np.cumsum(S ** 2) / np.sum(S ** 2), 0.9) + 1
    )

    # Phase clustering: circular variance. Low = clustered, ~1 = uniform.
    weights = magnitudes / np.sum(magnitudes)
    mean_resultant = np.abs(np.sum(weights * np.exp(1j * phases)))
    circular_variance = 1 - mean_resultant

    out_json = {
        "model": args.model,
        "n_samples": int(streams.shape[0]),
        "n_sublayers": int(streams.shape[1]),
        "d_model": int(d_model),
        "svd": {
            "var_explained_top2": var_explained_top2,
            "var_explained_top5": var_explained_top5,
            "n_components_for_90pct": n_for_90pct,
            "top_singular_values": S[:10].tolist(),
        },
        "phases": {
            "circular_variance": float(circular_variance),
            "mean_resultant_length": float(mean_resultant),
            "n_units": int(d_model),
        },
    }
    with open(out_dir / "svd_summary.json", "w") as f:
        json.dump(out_json, f, indent=2)

    print("\n--- Summary ---")
    print(f"Model: {args.model}")
    print(f"Variance in top-2 SVD components:  "
          f"{var_explained_top2*100:.1f}%")
    print(f"Variance in top-5 SVD components:  "
          f"{var_explained_top5*100:.1f}%")
    print(f"Components needed for 90% variance: {n_for_90pct}")
    print(f"Mean resultant length of phases:   "
          f"{mean_resultant:.3f} (1 = perfectly aligned, 0 = uniform)")
    print(f"Circular variance:                 "
          f"{circular_variance:.3f} (0 = clustered, 1 = uniform)")
    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()