"""
Measure how angular speed evolves across sublayers, per residual unit.

For each unit's trajectory in (x, y) = (a - a_0, ∇a - ∇a_0) space,
compute two angular-speed signals:

  - Tangent angular speed: how fast the direction of motion turns,
    per sublayer. This is what Fernando's R sums up.
  - Position angular speed: how fast the position vector rotates
    around the origin, per sublayer. This is "how fast the orbit
    is winding."

Also track:
  - Radius: distance from origin at each sublayer.
  - Linear step size: how far the trajectory moves between sublayers.

We aggregate across units and samples to see whether these quantities
have a consistent shape across sublayer depth.

Outputs in angular_speed_out_<model>/:
  - mean_angular_speed.png       mean angular speeds vs sublayer
  - mean_radius.png              mean radius vs sublayer
  - per_unit_traces.png          a few example units' angular speed
  - speed_ratio_distribution.png early/late speed ratio per unit
  - angular_speed_summary.json
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
# Angular speed computation
# ---------------------------------------------------------------------------

def compute_angular_signals(streams):
    """
    For each (sample, unit), compute four signals across sublayers:
      - tangent_speed[ℓ]: |Δθ_ℓ| where θ is the angle of (Δx, Δy)
      - position_speed[ℓ]: |Δφ_ℓ| where φ is the angle of (x, y)
      - radius[ℓ]: sqrt(x² + y²)
      - linear_step[ℓ]: sqrt(Δx² + Δy²)

    Vectorized across samples and units for speed.

    Returns dict of arrays:
      tangent_speed: (n_samples, n_sub - 2, d_model)
      position_speed: (n_samples, n_sub - 1, d_model)
      radius: (n_samples, n_sub, d_model)
      linear_step: (n_samples, n_sub - 1, d_model)
    """
    # gradient along sublayer axis (axis=1)
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]

    # radius
    radius = np.sqrt(x ** 2 + y ** 2)

    # position angle
    phi = np.arctan2(y, x)
    dphi = np.diff(phi, axis=1)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    position_speed = np.abs(dphi)

    # linear step
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    linear_step = np.sqrt(dx ** 2 + dy ** 2)

    # tangent angle
    theta = np.arctan2(dy, dx)
    dtheta = np.diff(theta, axis=1)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    tangent_speed = np.abs(dtheta)

    return {
        "tangent_speed": tangent_speed,
        "position_speed": position_speed,
        "radius": radius,
        "linear_step": linear_step,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_mean_angular_speed(signals, save_path=None):
    """
    Mean angular speed across (samples, units), as a function of
    sublayer index. Shows tangent and position speed on the same plot.
    """
    ts = signals["tangent_speed"].mean(axis=(0, 2))
    ps = signals["position_speed"].mean(axis=(0, 2))
    ls = signals["linear_step"].mean(axis=(0, 2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(ts, "o-", label="tangent angular speed")
    axes[0].plot(ps, "s-", label="position angular speed")
    axes[0].set_xlabel("sublayer transition index")
    axes[0].set_ylabel("|Δangle| (radians)")
    axes[0].set_title("Mean angular speeds across sublayers")
    axes[0].legend()
    axes[0].axhline(0, color="gray", linewidth=0.5)

    axes[1].plot(ls, "o-", color="C2", label="linear step size")
    axes[1].set_xlabel("sublayer transition index")
    axes[1].set_ylabel("step length")
    axes[1].set_title("Mean linear step size across sublayers")
    axes[1].set_yscale("log")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_mean_radius(signals, save_path=None):
    radius = signals["radius"].mean(axis=(0, 2))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(radius, "o-")
    ax.set_xlabel("sublayer index")
    ax.set_ylabel("mean radius")
    ax.set_title("Mean radius (distance from origin) across sublayers")
    ax.set_yscale("log")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_per_unit_traces(signals, n_units=12, save_path=None):
    """
    A few example units, showing tangent angular speed across
    sublayers (averaged over samples).
    """
    ts = signals["tangent_speed"].mean(axis=0)  # (n_sub - 2, d_model)
    n_sub_minus_2, d_model = ts.shape

    rng = np.random.default_rng(0)
    selected = rng.choice(d_model, size=n_units, replace=False)

    cols = 4
    rows = (n_units + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(3.0 * cols, 2.4 * rows))
    axes = np.atleast_2d(axes)
    for i, u in enumerate(selected):
        ax = axes[i // cols, i % cols]
        ax.plot(ts[:, u], linewidth=0.9)
        ax.set_title(f"unit {u}", fontsize=9)
        ax.set_xlabel("sublayer trans.", fontsize=8)
        ax.set_ylabel("|Δθ| (rad)", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n_units, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_speed_ratio_distribution(signals, save_path=None):
    """
    Per unit: mean angular speed in the first quarter vs the last
    quarter of the trace. If speed is constant, the ratio is ~1.
    If speed slows down, the ratio is > 1. If it speeds up, < 1.
    """
    ts = signals["tangent_speed"]
    n_sub_minus_2 = ts.shape[1]
    q = max(1, n_sub_minus_2 // 4)
    early = ts[:, :q, :].mean(axis=(0, 1))  # (d_model,)
    late = ts[:, -q:, :].mean(axis=(0, 1))
    ratio_tan = early / np.maximum(late, 1e-12)

    ps = signals["position_speed"]
    n_sub_minus_1 = ps.shape[1]
    q2 = max(1, n_sub_minus_1 // 4)
    early_p = ps[:, :q2, :].mean(axis=(0, 1))
    late_p = ps[:, -q2:, :].mean(axis=(0, 1))
    ratio_pos = early_p / np.maximum(late_p, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(np.log10(ratio_tan), bins=60)
    axes[0].axvline(0, color="red", linewidth=1, linestyle="--",
                    label="constant speed (log10 ratio = 0)")
    axes[0].set_xlabel("log10(early / late tangent speed)")
    axes[0].set_ylabel("# units")
    axes[0].set_title("Tangent angular speed: early/late ratio per unit")
    axes[0].legend()

    axes[1].hist(np.log10(ratio_pos), bins=60)
    axes[1].axvline(0, color="red", linewidth=1, linestyle="--",
                    label="constant speed (log10 ratio = 0)")
    axes[1].set_xlabel("log10(early / late position speed)")
    axes[1].set_ylabel("# units")
    axes[1].set_title("Position angular speed: early/late ratio per unit")
    axes[1].legend()

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

    out_dir = Path(f"angular_speed_out_{slugify(args.model)}")
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

    print("Computing angular signals ...")
    signals = compute_angular_signals(streams)

    print("Saving plots ...")
    plot_mean_angular_speed(
        signals, save_path=out_dir / "mean_angular_speed.png"
    )
    plot_mean_radius(
        signals, save_path=out_dir / "mean_radius.png"
    )
    plot_per_unit_traces(
        signals, save_path=out_dir / "per_unit_traces.png"
    )
    plot_speed_ratio_distribution(
        signals, save_path=out_dir / "speed_ratio_distribution.png"
    )

    # Summary
    ts = signals["tangent_speed"]
    ps = signals["position_speed"]
    rad = signals["radius"]
    ls = signals["linear_step"]

    n_sub_t = ts.shape[1]
    qt = max(1, n_sub_t // 4)
    n_sub_p = ps.shape[1]
    qp = max(1, n_sub_p // 4)

    summary = {
        "model": args.model,
        "n_samples": int(streams.shape[0]),
        "n_sublayers": int(streams.shape[1]),
        "d_model": int(d_model),
        "tangent_speed": {
            "early_quarter_mean": float(ts[:, :qt, :].mean()),
            "late_quarter_mean": float(ts[:, -qt:, :].mean()),
            "early_over_late_ratio_overall": float(
                ts[:, :qt, :].mean() / max(ts[:, -qt:, :].mean(), 1e-12)
            ),
            "per_sublayer_mean":
                ts.mean(axis=(0, 2)).tolist(),
        },
        "position_speed": {
            "early_quarter_mean": float(ps[:, :qp, :].mean()),
            "late_quarter_mean": float(ps[:, -qp:, :].mean()),
            "early_over_late_ratio_overall": float(
                ps[:, :qp, :].mean() / max(ps[:, -qp:, :].mean(), 1e-12)
            ),
            "per_sublayer_mean":
                ps.mean(axis=(0, 2)).tolist(),
        },
        "radius": {
            "first_sublayer_mean": float(rad[:, 0, :].mean()),
            "last_sublayer_mean": float(rad[:, -1, :].mean()),
            "ratio": float(
                rad[:, -1, :].mean() / max(rad[:, 0, :].mean(), 1e-12)
            ),
            "per_sublayer_mean": rad.mean(axis=(0, 2)).tolist(),
        },
        "linear_step": {
            "early_quarter_mean": float(ls[:, :qp, :].mean()),
            "late_quarter_mean": float(ls[:, -qp:, :].mean()),
            "ratio_late_over_early": float(
                ls[:, -qp:, :].mean() / max(ls[:, :qp, :].mean(), 1e-12)
            ),
        },
    }
    with open(out_dir / "angular_speed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(f"Tangent angular speed: early={summary['tangent_speed']['early_quarter_mean']:.4f}, "
          f"late={summary['tangent_speed']['late_quarter_mean']:.4f}, "
          f"ratio early/late={summary['tangent_speed']['early_over_late_ratio_overall']:.3f}")
    print(f"Position angular speed: early={summary['position_speed']['early_quarter_mean']:.4f}, "
          f"late={summary['position_speed']['late_quarter_mean']:.4f}, "
          f"ratio early/late={summary['position_speed']['early_over_late_ratio_overall']:.3f}")
    print(f"Radius: first sublayer={summary['radius']['first_sublayer_mean']:.4f}, "
          f"last={summary['radius']['last_sublayer_mean']:.4f}, "
          f"ratio={summary['radius']['ratio']:.3f}")
    print(f"Linear step size ratio (late/early): "
          f"{summary['linear_step']['ratio_late_over_early']:.3f}")
    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()