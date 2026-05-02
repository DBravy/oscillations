"""
Exploratory: residual unit dynamics in GPT-2 and SmolLM2.

Goal: characterize cross-layer dynamics of residual units. Replicate the
basic Fernando-style observation (rotational orbits in (a, da/dl) phase
space), and add an FFT-based metric that distinguishes genuine oscillation
from noise on a ramp.

Outputs (in exploratory_out_<model_name>/):
  - phase_portraits.png      phase portraits for selected units
  - unit_traces.png          raw activation traces for the same units
  - summary_distributions.png  histograms of per-unit summary stats
  - fft_diagnostics.png      AC/total power and dominant freq distributions
  - summary.json             numeric summary

Usage:
  python exploratory.py --model gpt2
  python exploratory.py --model HuggingFaceTB/SmolLM2-360M
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt2"
N_SAMPLES = 64           # number of text samples
SEQ_LEN = 128            # tokens per sample (last token is the analyzed one)
N_UNITS_TO_PLOT = 12     # how many individual units to show in the figure
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Architecture detection: where to hook for pre-attn / pre-mlp residuals
# ---------------------------------------------------------------------------

def get_hook_targets(model):
    """
    Return a list of (block_idx, ln_name, ln_module) tuples in the order
    pre_attn_L0, pre_mlp_L0, pre_attn_L1, pre_mlp_L1, ...

    Supports GPT-2-style models (ln_1, ln_2) and Llama-style models
    (input_layernorm, post_attention_layernorm), which covers both
    GPT-2 and SmolLM2.
    """
    # Find the list of transformer blocks
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
        attn_ln_attr = "ln_1"
        mlp_ln_attr = "ln_2"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
        attn_ln_attr = "input_layernorm"
        mlp_ln_attr = "post_attention_layernorm"
    else:
        raise RuntimeError(
            "Could not locate transformer blocks. Expected either "
            "model.transformer.h (GPT-2 style) or model.model.layers "
            "(Llama style)."
        )

    targets = []
    for i, block in enumerate(blocks):
        targets.append((i, "pre_attn", getattr(block, attn_ln_attr)))
        targets.append((i, "pre_mlp", getattr(block, mlp_ln_attr)))
    return targets


def get_d_model(model):
    """Return the residual stream dimension."""
    cfg = model.config
    for attr in ("n_embd", "hidden_size", "d_model"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise RuntimeError("Could not infer d_model from config.")


def get_n_layers(model):
    cfg = model.config
    for attr in ("n_layer", "num_hidden_layers"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise RuntimeError("Could not infer n_layers from config.")


# ---------------------------------------------------------------------------
# Capture residual stream at every sublayer
# ---------------------------------------------------------------------------

def collect_residual_stream(model, tokenizer, texts, seq_len):
    """
    Run the model on each text and capture the residual stream vector
    just before each sublayer's LayerNorm reads it. Take the LAST token
    position only, matching Fernando's protocol.

    Returns: array of shape (n_samples, 2 * n_layers, d_model)
    """
    model.eval()
    n_layers = get_n_layers(model)
    targets = get_hook_targets(model)
    expected = 2 * n_layers
    assert len(targets) == expected, (
        f"hook targets ({len(targets)}) != 2 * n_layers ({expected})"
    )

    all_streams = []

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
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

        if len(captures) != expected:
            raise RuntimeError(
                f"Expected {expected} captures but got {len(captures)}"
            )

        # Last token position for each sublayer
        last_token_states = torch.stack(
            [c[0, -1, :].float() for c in captures], dim=0
        )  # (2 * n_layers, d_model)
        all_streams.append(last_token_states.cpu().numpy())

    return np.stack(all_streams, axis=0)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def per_unit_summary(streams):
    """
    Per-unit summary statistics over cross-sublayer trajectories.

    streams: (n_samples, n_sublayers, d_model)

    Returns dict with per-unit arrays of shape (d_model,):
      - mean_abs_activation
      - layerwise_correlation     correlation between consecutive sublayers
      - sign_changes              sign-changes of da/dl, mean over samples
      - amplitude_ratio           late-quarter |a| / early-quarter |a|
      - ac_power_fraction         fraction of FFT power in non-DC bins
      - dominant_freq_bin         FFT bin (1..N/2) with peak non-DC power
    """
    n_samples, n_sub, d_model = streams.shape

    mean_abs = np.mean(np.abs(streams), axis=(0, 1))

    # Layerwise correlation: corr(h_l, h_{l+1}) across samples, per unit,
    # averaged over the n_sub-1 layer transitions.
    correlations = np.zeros(d_model)
    for u in range(d_model):
        per_layer = streams[:, :, u].T  # (n_sub, n_samples)
        rs = []
        for l in range(n_sub - 1):
            x = per_layer[l]
            y = per_layer[l + 1]
            if x.std() > 1e-8 and y.std() > 1e-8:
                rs.append(np.corrcoef(x, y)[0, 1])
        correlations[u] = np.mean(rs) if rs else 0.0

    # Sign-changes of the layer-derivative
    diffs = np.diff(streams, axis=1)
    signs = np.sign(diffs)
    sign_changes_per_sample = np.sum(
        (signs[:, 1:, :] * signs[:, :-1, :]) < 0, axis=1
    )
    sign_changes = sign_changes_per_sample.mean(axis=0)

    # Amplitude growth
    q = max(1, n_sub // 4)
    early_amp = np.mean(np.abs(streams[:, :q, :]), axis=(0, 1))
    late_amp = np.mean(np.abs(streams[:, -q:, :]), axis=(0, 1))
    amp_ratio = late_amp / np.maximum(early_amp, 1e-8)

    # FFT-based metrics. For each (sample, unit), take the trace across
    # sublayers, subtract its sample mean, FFT it, look at one-sided power
    # spectrum. Then compute:
    #   - ac_power_fraction: power in non-DC bins / total power of the
    #     RAW (non-mean-subtracted) trace. This way a pure-DC trace gives 0
    #     and a pure-oscillation trace gives ~1.
    #   - dominant_freq_bin: argmax over non-DC bins of |FFT|^2.
    #
    # Average over samples to get one value per unit.
    ac_frac_per_sample = np.zeros((n_samples, d_model))
    dom_freq_per_sample = np.zeros((n_samples, d_model), dtype=int)

    for s in range(n_samples):
        for u in range(d_model):
            trace = streams[s, :, u]
            # power of the raw trace, used as denominator
            total_power = np.sum(trace ** 2) + 1e-12
            # FFT of the mean-subtracted trace (so DC bin is exactly 0)
            ms = trace - trace.mean()
            spec = np.fft.rfft(ms)
            power = np.abs(spec) ** 2
            # power[0] is DC of mean-subtracted = 0; non-DC = power[1:]
            ac_power = np.sum(power)  # already mean-subtracted
            # Express AC power as a fraction of the original trace's power.
            # This penalizes large DC offsets / monotonic ramps.
            ac_frac_per_sample[s, u] = ac_power / total_power
            if len(power) > 1:
                dom_freq_per_sample[s, u] = int(np.argmax(power[1:])) + 1
            else:
                dom_freq_per_sample[s, u] = 0

    ac_power_fraction = ac_frac_per_sample.mean(axis=0)
    dominant_freq_bin = np.median(dom_freq_per_sample, axis=0)

    return {
        "mean_abs_activation": mean_abs,
        "layerwise_correlation": correlations,
        "sign_changes": sign_changes,
        "amplitude_ratio": amp_ratio,
        "ac_power_fraction": ac_power_fraction,
        "dominant_freq_bin": dominant_freq_bin,
        "n_sublayers": n_sub,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_phase_portraits(streams, unit_indices, sample_idx=0, save_path=None):
    n = len(unit_indices)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows))
    axes = np.atleast_2d(axes)

    sample = streams[sample_idx]
    derivs = np.gradient(sample, axis=0)

    for i, u in enumerate(unit_indices):
        ax = axes[i // cols, i % cols]
        a = sample[:, u]
        da = derivs[:, u]
        ax.plot(a, da, "-", linewidth=0.8, alpha=0.9)
        ax.scatter(a, da, c=np.arange(len(a)), cmap="viridis", s=10)
        ax.set_title(f"unit {u}", fontsize=9)
        ax.set_xlabel("a", fontsize=8)
        ax.set_ylabel("da/dl", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle(
        f"Phase portraits of selected units (sample {sample_idx})",
        fontsize=11,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_unit_traces(streams, unit_indices, save_path=None, n_overlay=8):
    n = len(unit_indices)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows))
    axes = np.atleast_2d(axes)

    n_samples = streams.shape[0]
    overlay_samples = min(n_overlay, n_samples)

    for i, u in enumerate(unit_indices):
        ax = axes[i // cols, i % cols]
        for s in range(overlay_samples):
            ax.plot(streams[s, :, u], linewidth=0.7, alpha=0.6)
        ax.set_title(f"unit {u}", fontsize=9)
        ax.set_xlabel("sublayer", fontsize=8)
        ax.set_ylabel("a", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="gray", linewidth=0.5)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle("Raw activation trajectories of selected units", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_summary_distributions(summary, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    items = [
        ("mean_abs_activation", "Mean |activation| per unit"),
        ("layerwise_correlation", "Mean consecutive-layer correlation"),
        ("sign_changes", "Sign changes of da/dl"),
        ("amplitude_ratio", "Late/early amplitude ratio"),
    ]
    for ax, (key, title) in zip(axes.flat, items):
        vals = summary[key]
        ax.hist(vals, bins=60)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(key, fontsize=9)
        ax.set_ylabel("# units", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_fft_diagnostics(summary, save_path=None):
    """
    The two new plots that should let us see the regime split (if real):
      - histogram of ac_power_fraction across units
      - histogram of dominant_freq_bin across units
      - scatter of (ac_power_fraction, sign_changes), since the regimes
        should separate in this 2D plane: exponential-ramp units have
        low ac_power_fraction but possibly normal sign_changes from noise.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ac = summary["ac_power_fraction"]
    df = summary["dominant_freq_bin"]
    sc = summary["sign_changes"]

    axes[0].hist(ac, bins=60)
    axes[0].set_title("AC / total power per unit", fontsize=10)
    axes[0].set_xlabel("ac_power_fraction", fontsize=9)
    axes[0].set_ylabel("# units", fontsize=9)

    axes[1].hist(df, bins=int(max(df) + 1) if max(df) > 0 else 1)
    axes[1].set_title("Dominant FFT bin per unit", fontsize=10)
    axes[1].set_xlabel("dominant_freq_bin (1 = slowest non-DC)", fontsize=9)
    axes[1].set_ylabel("# units", fontsize=9)

    axes[2].scatter(ac, sc, s=4, alpha=0.5)
    axes[2].set_title("Sign-changes vs AC power fraction", fontsize=10)
    axes[2].set_xlabel("ac_power_fraction", fontsize=9)
    axes[2].set_ylabel("sign_changes", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(name):
    """Turn a model name into a filesystem-safe directory suffix."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = parser.parse_args()

    out_dir = Path(f"exploratory_out_{slugify(args.model)}")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
    ).to(DEVICE)

    d_model = get_d_model(model)
    n_layers = get_n_layers(model)
    print(f"d_model = {d_model}, n_layers = {n_layers} "
          f"(=> {2 * n_layers} sublayers)")

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
    streams = collect_residual_stream(
        model, tokenizer, texts, args.seq_len
    )
    print(f"Residual stream tensor shape: {streams.shape}")

    print("Computing per-unit summary ...")
    summary = per_unit_summary(streams)

    # Pick units to plot: span the AC-power range so we get both regimes
    ac = summary["ac_power_fraction"]
    order = np.argsort(ac)  # low ac first (likely exponential), high last
    n_third = N_UNITS_TO_PLOT // 3
    low = order[:n_third]                    # likely "exponential" regime
    high = order[-n_third:]                  # likely "wave" regime
    rest_n = N_UNITS_TO_PLOT - len(low) - len(high)
    mid_idx = len(order) // 2
    mid = order[mid_idx - rest_n // 2 : mid_idx - rest_n // 2 + rest_n]
    units_to_plot = np.concatenate([low, mid, high])

    print("Saving figures ...")
    plot_phase_portraits(
        streams, units_to_plot,
        sample_idx=0,
        save_path=out_dir / "phase_portraits.png",
    )
    plot_unit_traces(
        streams, units_to_plot,
        save_path=out_dir / "unit_traces.png",
    )
    plot_summary_distributions(
        summary,
        save_path=out_dir / "summary_distributions.png",
    )
    plot_fft_diagnostics(
        summary,
        save_path=out_dir / "fft_diagnostics.png",
    )

    out_json = {
        "model": args.model,
        "n_samples": int(streams.shape[0]),
        "n_sublayers": int(summary["n_sublayers"]),
        "d_model": int(d_model),
        "stats": {
            key: {
                "mean": float(np.mean(summary[key])),
                "median": float(np.median(summary[key])),
                "std": float(np.std(summary[key])),
                "min": float(np.min(summary[key])),
                "max": float(np.max(summary[key])),
            }
            for key in (
                "mean_abs_activation",
                "layerwise_correlation",
                "sign_changes",
                "amplitude_ratio",
                "ac_power_fraction",
                "dominant_freq_bin",
            )
        },
        "plotted_units": [int(u) for u in units_to_plot],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out_json, f, indent=2)

    print("\n--- Summary ---")
    print(f"Model: {args.model}")
    print(f"Samples: {streams.shape[0]}, "
          f"sublayers: {streams.shape[1]}, d_model: {d_model}")
    s = out_json["stats"]
    print(f"AC / total power per unit:")
    print(f"  median = {s['ac_power_fraction']['median']:.3f}, "
          f"mean = {s['ac_power_fraction']['mean']:.3f}, "
          f"min = {s['ac_power_fraction']['min']:.3f}, "
          f"max = {s['ac_power_fraction']['max']:.3f}")
    print(f"Dominant FFT bin per unit:")
    print(f"  median = {s['dominant_freq_bin']['median']:.1f}, "
          f"mean = {s['dominant_freq_bin']['mean']:.2f}, "
          f"max = {s['dominant_freq_bin']['max']:.0f}")
    print(f"Sign-change count per unit:")
    print(f"  median = {s['sign_changes']['median']:.2f}, "
          f"mean = {s['sign_changes']['mean']:.2f}")
    print(f"Mean consecutive-layer correlation per unit:")
    print(f"  median = {s['layerwise_correlation']['median']:.3f}, "
          f"mean = {s['layerwise_correlation']['mean']:.3f}")
    print(f"Late/early amplitude ratio:")
    print(f"  median = {s['amplitude_ratio']['median']:.2f}, "
          f"mean = {s['amplitude_ratio']['mean']:.2f}")
    print(f"\nOutputs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()