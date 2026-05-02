"""
Phase-space oscillator test across Pythia training checkpoints.

Loads Pythia at multiple training steps, runs the same phase-space
diagnostics from phase_space_llm.py at each, and produces trajectory
plots showing how the residual-stream oscillator structure evolves
over training. Step 0 is the released random-init checkpoint, so we
get the random-init baseline for free without needing a separate
config-only re-init.

Default model: Pythia-410M (24 layers -> 48 sublayers, d_model=1024).
Default checkpoints: [0, 128, 512, 2000, 8000, 32000, 64000, 143000].

Method matches phase_space_llm.py: hooks on the input-LN and
post-attention-LN of each transformer block, last-token snapshots
across 256 wikitext-2 validation texts. The four diagnostics:

  r       Pearson(x_l, Delta x_l) over the depth axis l
  A_norm  shoelace signed area of (x, Delta x), normalized
  R_PCA   small/large eigenvalue ratio after rescaling Delta x to match
          var(x). Frequency-invariant. Equals (1-|r|)/(1+|r|).
  dtheta  arg(Hilbert(Delta x)) - arg(Hilbert(x)), circular mean

Per-checkpoint we aggregate to per-unit (median across inputs) and
report population summaries. Across checkpoints we plot the population
median + IQR band as a function of training step.

Outputs in phase_space_pythia_<size>/:
  trajectory_combined.png        4-panel summary: dtheta dev, |r|,
                                 |A_norm|, resultant length vs step.
  dtheta_distributions.png       Per-checkpoint dtheta histograms,
                                 colored by training step (viridis).
  resultant_distributions.png    Per-checkpoint resultant-length
                                 histograms, same color convention.
  phase_space_trajectory.json    Full per-step numerical summary.

Usage:
  python phase_space_pythia_checkpoints.py
  python phase_space_pythia_checkpoints.py --model-size 70m
  python phase_space_pythia_checkpoints.py --model-size 410m \\
      --checkpoints 0 128 512 2000 8000 32000 64000 143000
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert as scipy_hilbert
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Pythia config (matches reference script)
# ---------------------------------------------------------------------------

PYTHIA_CONFIGS = {
    "70m":  "EleutherAI/pythia-70m",
    "160m": "EleutherAI/pythia-160m",
    "410m": "EleutherAI/pythia-410m",
    "1b":   "EleutherAI/pythia-1b",
    "1.4b": "EleutherAI/pythia-1.4b",
    "2.8b": "EleutherAI/pythia-2.8b",
}

DEFAULT_CHECKPOINTS = [0, 128, 512, 2000, 8000, 32000, 64000, 143000]

# Cache convention from the reference experiment_b script
_ORICO_CACHE = "/Volumes/ORICO/huggingface_cache"
DEFAULT_CACHE_DIR = _ORICO_CACHE if os.path.isdir("/Volumes/ORICO") else None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


# ---------------------------------------------------------------------------
# Stream collection
# ---------------------------------------------------------------------------

def get_hook_targets(model):
    """Locate LayerNorm modules in each transformer block.
    Supports GPT-2-style, GPT-NeoX (Pythia), and Llama/SmolLM-style."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
        attn_ln, mlp_ln = "ln_1", "ln_2"
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        blocks = model.gpt_neox.layers
        attn_ln, mlp_ln = "input_layernorm", "post_attention_layernorm"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
        attn_ln, mlp_ln = "input_layernorm", "post_attention_layernorm"
    else:
        raise RuntimeError(
            f"Could not locate transformer blocks in {type(model).__name__}."
        )
    targets = []
    for block in blocks:
        targets.append(getattr(block, attn_ln))
        targets.append(getattr(block, mlp_ln))
    return targets


def collect_streams(model, tokenizer, texts, seq_len):
    """Capture last-token residual stream snapshots across all sublayers.
    Returns (n_samples, n_sub, d_model) array."""
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
        hooks = [
            t.register_forward_hook(
                lambda m, i, o, c=captures: c.append(i[0].detach())
            )
            for t in targets
        ]
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
# Phase-space diagnostics (from phase_space_llm.py)
# ---------------------------------------------------------------------------

def pair_x_dx(x):
    dx = np.diff(x, axis=1)
    x_pair = x[:, :-1, :]
    return x_pair, dx


def per_traj_pearson(x, dx):
    x_c = x - x.mean(axis=1, keepdims=True)
    dx_c = dx - dx.mean(axis=1, keepdims=True)
    num = (x_c * dx_c).sum(axis=1)
    den = np.sqrt((x_c ** 2).sum(axis=1) * (dx_c ** 2).sum(axis=1)) + 1e-30
    return num / den


def per_traj_signed_area(x, dx):
    if x.shape[1] < 2:
        return np.zeros((x.shape[0], x.shape[2]))
    a = x[:, :-1, :] * dx[:, 1:, :]
    b = x[:, 1:, :] * dx[:, :-1, :]
    return 0.5 * (a - b).sum(axis=1)


def per_traj_pca_aspect(x, dx):
    """Rescaled PCA aspect ratio. Equals (1-|r|)/(1+|r|)."""
    x_c = x - x.mean(axis=1, keepdims=True)
    dx_c = dx - dx.mean(axis=1, keepdims=True)
    var_x = (x_c ** 2).mean(axis=1)
    var_dx = (dx_c ** 2).mean(axis=1)
    cov_xy = (x_c * dx_c).mean(axis=1)
    s = np.sqrt(var_x / (var_dx + 1e-30))
    cov_rescaled = s * cov_xy
    lam_max = var_x + np.abs(cov_rescaled)
    lam_min = np.maximum(var_x - np.abs(cov_rescaled), 0.0)
    return lam_min / (lam_max + 1e-30)


def per_traj_quadrature_phase(x, dx):
    z_x = scipy_hilbert(x, axis=1)
    z_dx = scipy_hilbert(dx, axis=1)
    ratio = z_dx / (z_x + 1e-30)
    return np.angle(ratio.mean(axis=1))


def collect_metrics(streams, trim=0):
    """Center, optionally trim, then compute per-unit diagnostics."""
    if trim > 0:
        if streams.shape[1] <= 2 * trim + 2:
            raise ValueError(
                f"Cannot trim {trim} from each end of {streams.shape[1]} sublayers."
            )
        streams = streams[:, trim:streams.shape[1] - trim, :]
    x = streams - streams.mean(axis=1, keepdims=True)
    x_pair, dx = pair_x_dx(x)

    r = per_traj_pearson(x_pair, dx)
    A = per_traj_signed_area(x_pair, dx)
    R = per_traj_pca_aspect(x_pair, dx)
    dtheta = per_traj_quadrature_phase(x_pair, dx)

    std_x = x_pair.std(axis=1)
    std_dx = dx.std(axis=1)
    L_pair = x_pair.shape[1]
    A_norm = A / ((std_x * std_dx * L_pair) + 1e-30)

    r_per_unit = np.median(r, axis=0)
    A_norm_per_unit = np.median(A_norm, axis=0)
    R_per_unit = np.median(R, axis=0)
    dtheta_complex = np.exp(1j * dtheta)
    dtheta_resultant = np.abs(dtheta_complex.mean(axis=0))
    dtheta_per_unit = np.angle(dtheta_complex.mean(axis=0))

    return {
        "shape": x.shape,
        "r_per_unit": r_per_unit,
        "A_norm_per_unit": A_norm_per_unit,
        "R_per_unit": R_per_unit,
        "dtheta_per_unit": dtheta_per_unit,
        "dtheta_resultant_per_unit": dtheta_resultant,
    }


# ---------------------------------------------------------------------------
# Pythia-specific loading and sweep
# ---------------------------------------------------------------------------

def load_pythia_at_step(model_name, step, cache_dir=None):
    """Load Pythia at a specific training step via revision='step{N}'."""
    revision = f"step{step}"
    print(f"  Loading {model_name} at {revision} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    ).to(DEVICE)
    model.eval()
    return model


def run_checkpoint_sweep(model_name, checkpoints, texts, tokenizer,
                          seq_len, trim, cache_dir=None):
    """Load and analyze each checkpoint sequentially. Drops model after
    each step to keep memory bounded; only diagnostic arrays persist."""
    results = []
    for step in checkpoints:
        print(f"\n=== {model_name} at step {step} ===")
        try:
            model = load_pythia_at_step(model_name, step, cache_dir=cache_dir)
        except Exception as e:
            print(f"  FAILED to load step {step}: {e}")
            continue

        try:
            print(f"  Collecting streams ...")
            streams = collect_streams(model, tokenizer, texts, seq_len)
            print(f"    shape (N, n_sub, d_model) = {streams.shape}")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        m = collect_metrics(streams, trim=trim)
        m["step"] = int(step)
        results.append(m)
        del streams

        # Quick per-step report
        dtheta_dev = np.abs(np.abs(m["dtheta_per_unit"]) - np.pi / 2)
        print(f"    median |r|={np.median(np.abs(m['r_per_unit'])):.3f}, "
              f"median dtheta_dev={np.median(dtheta_dev):.3f}, "
              f"median resultant={np.median(m['dtheta_resultant_per_unit']):.3f}")

    return results


# ---------------------------------------------------------------------------
# Trajectory plots
# ---------------------------------------------------------------------------

def _percentile_band(ax, steps, values, color, label, marker="o"):
    """Plot median + 25-75 IQR band on log-x axis."""
    median = np.median(values, axis=1)
    p25 = np.percentile(values, 25, axis=1)
    p75 = np.percentile(values, 75, axis=1)
    ax.fill_between(steps, p25, p75, color=color, alpha=0.25)
    ax.plot(steps, median, marker + "-", color=color,
            linewidth=2, markersize=6, label=label)


def plot_combined_trajectory(results, out_path, model_name):
    """4-panel summary across training."""
    steps = [m["step"] for m in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) dtheta deviation from pi/2
    ax = axes[0, 0]
    devs = np.array([np.abs(np.abs(m["dtheta_per_unit"]) - np.pi / 2)
                     for m in results])
    _percentile_band(ax, steps, devs, "C0", "median + IQR")
    ax.axhline(0, color="black", linestyle="--", alpha=0.4,
               label="ideal quadrature")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$||\Delta\theta| - \pi/2|$ (radians)")
    ax.set_title("(a) Quadrature deviation: lower = sharper oscillator")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)

    # (0,1) |r|
    ax = axes[0, 1]
    rs = np.array([np.abs(m["r_per_unit"]) for m in results])
    _percentile_band(ax, steps, rs, "C2", "median + IQR")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$|r|$")
    ax.set_title(r"(b) $|r|$: lower = $x$ and $\Delta x$ more orthogonal")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)

    # (1,0) |A_norm|
    ax = axes[1, 0]
    As = np.array([np.abs(m["A_norm_per_unit"]) for m in results])
    _percentile_band(ax, steps, As, "C3", "median + IQR")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$|A_{\mathrm{norm}}|$")
    ax.set_title("(c) Rotation strength: higher = stronger ellipse")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)

    # (1,1) Resultant length
    ax = axes[1, 1]
    Rs = np.array([m["dtheta_resultant_per_unit"] for m in results])
    _percentile_band(ax, steps, Rs, "C4", "median + IQR")
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.4,
               label="input-independent")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\Delta\theta$ resultant length")
    ax.set_title("(d) Input-locking: 1 = identical phase across inputs")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Phase-space dynamics across training: {model_name}",
        fontsize=14, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_dtheta_distributions(results, out_path, model_name):
    """Overlay dtheta histograms across checkpoints (viridis = step)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    n = len(results)
    cmap = plt.cm.viridis
    for i, m in enumerate(results):
        color = cmap(i / max(n - 1, 1))
        ax.hist(
            m["dtheta_per_unit"], bins=72, range=(-np.pi, np.pi),
            histtype="step", linewidth=1.5, density=True,
            color=color, label=f"step {m['step']}",
        )
    ax.axvline(np.pi / 2, color="black", linestyle="--", alpha=0.5,
               label=r"$+\pi/2$ (forward osc)")
    ax.axvline(-np.pi / 2, color="gray", linestyle=":", alpha=0.5,
               label=r"$-\pi/2$ (reverse osc)")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xlabel(r"$\Delta\theta$")
    ax.set_ylabel("density")
    ax.set_title(rf"$\Delta\theta$ distribution evolution: {model_name}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_resultant_distributions(results, out_path, model_name):
    """Overlay resultant-length histograms across checkpoints."""
    fig, ax = plt.subplots(figsize=(12, 6))
    n = len(results)
    cmap = plt.cm.viridis
    for i, m in enumerate(results):
        color = cmap(i / max(n - 1, 1))
        ax.hist(
            m["dtheta_resultant_per_unit"], bins=50, range=(0, 1),
            histtype="step", linewidth=1.5, density=True,
            color=color, label=f"step {m['step']}",
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("resultant length (input-locking)")
    ax.set_ylabel("density")
    ax.set_title(f"Resultant length distribution evolution: {model_name}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(results, model_name):
    summary = {
        "model": model_name,
        "checkpoints": [],
    }
    for m in results:
        D = m["shape"][2]
        L = m["shape"][1]
        N = m["shape"][0]
        dtheta_dev = np.abs(np.abs(m["dtheta_per_unit"]) - np.pi / 2)
        summary["checkpoints"].append({
            "step": int(m["step"]),
            "shape": {"N": int(N), "L": int(L), "D": int(D)},
            "r": {
                "abs_mean":   float(np.abs(m["r_per_unit"]).mean()),
                "abs_median": float(np.median(np.abs(m["r_per_unit"]))),
                "median":     float(np.median(m["r_per_unit"])),
            },
            "A_norm": {
                "abs_mean":   float(np.abs(m["A_norm_per_unit"]).mean()),
                "mean":       float(m["A_norm_per_unit"].mean()),
                "median":     float(np.median(m["A_norm_per_unit"])),
            },
            "R_PCA": {
                "mean":   float(m["R_per_unit"].mean()),
                "median": float(np.median(m["R_per_unit"])),
            },
            "dtheta": {
                "circular_mean": float(m["dtheta_per_unit"].mean()),
                "deviation_from_pi_over_2": {
                    "mean":   float(dtheta_dev.mean()),
                    "median": float(np.median(dtheta_dev)),
                },
            },
            "resultant_length": {
                "mean":   float(m["dtheta_resultant_per_unit"].mean()),
                "median": float(np.median(m["dtheta_resultant_per_unit"])),
            },
        })
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="410m",
                        choices=list(PYTHIA_CONFIGS.keys()))
    parser.add_argument("--checkpoints", type=int, nargs="+",
                        default=DEFAULT_CHECKPOINTS,
                        help="Pythia training steps to load.")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--trim-sublayers", type=int, default=2,
                        help="Drop this many sublayers from each end.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help="HF cache dir; default uses /Volumes/ORICO if mounted.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    model_name = PYTHIA_CONFIGS[args.model_size]
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(f"phase_space_pythia_{args.model_size}"))
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Model: {model_name}")
    print(f"Checkpoints: {args.checkpoints}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Output dir: {out_dir}")

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print(f"\nLoading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]

    print(f"Loading tokenizer for {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Sweep ----
    results = run_checkpoint_sweep(
        model_name, args.checkpoints, texts, tokenizer,
        args.seq_len, args.trim_sublayers, cache_dir=args.cache_dir,
    )

    if not results:
        raise SystemExit("No checkpoints loaded successfully.")

    # ---- Plots ----
    print(f"\nMaking plots ...")
    plot_combined_trajectory(
        results, out_dir / "trajectory_combined.png", model_name,
    )
    print(f"  trajectory_combined.png")

    plot_dtheta_distributions(
        results, out_dir / "dtheta_distributions.png", model_name,
    )
    print(f"  dtheta_distributions.png")

    plot_resultant_distributions(
        results, out_dir / "resultant_distributions.png", model_name,
    )
    print(f"  resultant_distributions.png")

    # ---- Summary ----
    summary = build_summary(results, model_name)
    with open(out_dir / "phase_space_trajectory.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  phase_space_trajectory.json")

    # ---- Console table ----
    print(f"\n--- Phase-space dynamics across training: {model_name} ---")
    print(f"{'step':>10} {'<|r|>':>8} {'<R>':>8} "
          f"{'<dtheta>':>10} {'<dt_dev>':>10} "
          f"{'<resultant>':>12} {'<|A|>':>8}")
    for s in summary["checkpoints"]:
        print(
            f"{s['step']:>10} "
            f"{s['r']['abs_mean']:>8.3f} "
            f"{s['R_PCA']['mean']:>8.3f} "
            f"{s['dtheta']['circular_mean']:>10.3f} "
            f"{s['dtheta']['deviation_from_pi_over_2']['mean']:>10.3f} "
            f"{s['resultant_length']['mean']:>12.3f} "
            f"{s['A_norm']['abs_mean']:>8.3f}"
        )

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
