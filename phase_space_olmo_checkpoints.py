"""
Phase-space oscillator test across OLMo-2-1B training checkpoints.

OLMo port of phase_space_pythia_checkpoints.py. Tests the prediction
from the dynamical reading of Bray (2026): SwiGLU models should show
a late-training drop in the dtheta resultant length, reflecting the
documented transfer of inter-layer coordination from static weights
to the input-dependent gate. Pythia GELU models do not show this
drop in our earlier sweep (resultant stays near 0.90), consistent
with their static communicative backbone persisting through training.

Architectural notes (the reason this script exists separately):
  OLMo-2 uses post-normalization, not pre-norm. Each layer has
  post_attention_layernorm and post_feedforward_layernorm but no
  input_layernorm. The forward is approximately:
    residual = x
    x = residual + post_attention_layernorm(self_attn(x))
    residual = x
    x = residual + post_feedforward_layernorm(mlp(x))

  Hooking the LayerNorms here would NOT give residual stream snapshots
  (it would give normalized sublayer outputs). Instead, we use
  forward_pre_hooks on self_attn and mlp to capture the residual
  stream as it enters each sublayer. This gives 2 snapshots per layer,
  matching the depth resolution we used for Pythia, GPT-2, and SmolLM.

  OLMo-2-1B has 16 layers, so n_sub=32 and L=28 after trim of 2 from
  each end. Lower depth resolution than Pythia-410M (44) but enough.

Loading (from reference experiment_b_olmo):
  - Step <= 37000: load from allenai/OLMo-2-0425-1B-early-training
  - Step >  37000: load from allenai/OLMo-2-0425-1B
  - Revision string: f"stage1-step{step}-tokens{tokens_b}B" with
    tokens_b = ceil(step * 2048 * 1024 / 1e9)

  The reference defaults to early-training only (steps 0-10000), but
  the SwiGLU decoherence phase the paper documents runs from step
  ~10K to 1M. The default here therefore extends to 1M to capture
  the decoherence window. Late-checkpoint revisions may not all
  exist; the loader handles missing revisions gracefully and
  continues to the next checkpoint.

Default checkpoints:
  [0, 1000, 2000, 3000, 5000, 10000,  -- backbone formation
   37000,                              -- early-training boundary
   100000, 500000, 1000000]            -- decoherence window

The prediction:
  - During backbone formation (steps 1000-10000): dtheta deviation
    minimum, resultant length high (matches Pythia early behavior).
  - From step 10000 onward (decoherence phase): resultant length
    drops as gate coordination takes over from weight coordination.
    By step 1M, the paper documents 46% pairwise alignment loss; we
    predict resultant length around 0.70-0.75 (vs 0.90 for Pythia).

Outputs in phase_space_olmo_2_1b/:
  trajectory_combined.png        4-panel summary: dtheta dev, |r|,
                                 |A_norm|, resultant length vs step.
  dtheta_distributions.png       Per-checkpoint dtheta histograms.
  resultant_distributions.png    Per-checkpoint resultant histograms.
  phase_space_trajectory.json    Full per-step numerical summary.

Usage:
  python phase_space_olmo_checkpoints.py
  python phase_space_olmo_checkpoints.py \\
      --checkpoints 0 1000 2000 3000 5000 10000 37000 100000 500000 1000000
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert as scipy_hilbert
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import scan_cache_dir


# ---------------------------------------------------------------------------
# OLMo-2-1B configuration (from reference experiment_b_olmo)
# ---------------------------------------------------------------------------

OLMO_MODEL = "allenai/OLMo-2-0425-1B"
EARLY_TRAINING_REPO = "allenai/OLMo-2-0425-1B-early-training"
EARLY_TRAINING_MAX_STEP = 37000

DEFAULT_CHECKPOINTS = [0, 1000, 2000, 3000, 5000, 10000, 100000, 1000000]

_ORICO_CACHE = "/Volumes/ORICO/huggingface_cache"
DEFAULT_CACHE_DIR = _ORICO_CACHE if os.path.isdir("/Volumes/ORICO") else None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def step_to_revision(step):
    """Convert a step number to the OLMo-2-1B revision string.
    Formula from reference experiment_b_olmo."""
    tokens_b = math.ceil(step * 2048 * 1024 / 1_000_000_000)
    return f"stage1-step{step}-tokens{tokens_b}B"


# ---------------------------------------------------------------------------
# OLMo-specific stream collection
# ---------------------------------------------------------------------------

def get_hook_targets(model):
    """For OLMo-2 (post-norm architecture), hook self_attn and mlp inputs.
    No input_layernorm exists; LayerNorms here are AFTER the sublayer."""
    blocks = model.model.layers
    targets = []
    for block in blocks:
        targets.append(block.self_attn)
        targets.append(block.mlp)
    return targets


def collect_streams(model, tokenizer, texts, seq_len):
    """Capture residual stream at each sublayer input via forward_pre_hooks.
    Returns (n_samples, n_sub, d_model) array of last-token snapshots.

    Each block contributes 2 snapshots: residual stream entering self_attn
    (= residual stream entering the layer), and residual stream entering
    mlp (= residual stream after the post-attention block has been added)."""
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
        # forward_pre_hook fires before the module's forward.
        # The hook signature is (module, inputs) where inputs is a tuple
        # of the positional args. inputs[0] is the residual stream tensor.
        hooks = [
            t.register_forward_pre_hook(
                lambda m, args, kwargs, c=captures: c.append(
                    (args[0] if args else kwargs["hidden_states"]).detach()
                ),
                with_kwargs=True,
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
# Phase-space diagnostics (identical to phase_space_pythia_checkpoints.py)
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
    """Rescaled PCA aspect ratio. Equals (1-|r|)/(1+|r|) by construction."""
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
# OLMo-specific loading and sweep
# ---------------------------------------------------------------------------

def load_olmo_at_step(step, cache_dir=None):
    """Load OLMo-2-1B at a specific training step. Routes to early-training
    repo for steps <= 37000, main repo for later steps."""
    repo = EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP else OLMO_MODEL
    revision = step_to_revision(step)
    print(f"  Loading {repo} at {revision} ...")
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    ).to(DEVICE)
    model.eval()
    return model


def _result_path(out_dir, step):
    return Path(out_dir) / f"step_{step}.npz"


def save_result(out_dir, result):
    """Save a single checkpoint result to an .npz file."""
    np.savez(
        _result_path(out_dir, result["step"]),
        r_per_unit=result["r_per_unit"],
        A_norm_per_unit=result["A_norm_per_unit"],
        R_per_unit=result["R_per_unit"],
        dtheta_per_unit=result["dtheta_per_unit"],
        dtheta_resultant_per_unit=result["dtheta_resultant_per_unit"],
        shape=np.array(result["shape"]),
        step=np.array(result["step"]),
    )


def load_result(path):
    """Load a checkpoint result from an .npz file."""
    d = np.load(path)
    return {
        "shape": tuple(d["shape"]),
        "r_per_unit": d["r_per_unit"],
        "A_norm_per_unit": d["A_norm_per_unit"],
        "R_per_unit": d["R_per_unit"],
        "dtheta_per_unit": d["dtheta_per_unit"],
        "dtheta_resultant_per_unit": d["dtheta_resultant_per_unit"],
        "step": int(d["step"]),
    }


def cleanup_checkpoint_cache(repo_id, revision, cache_dir=None):
    """Delete cached files for a specific revision to free disk space."""
    try:
        cache_info = scan_cache_dir(cache_dir)
        for repo_info in cache_info.repos:
            if repo_info.repo_id == repo_id:
                for rev_info in repo_info.revisions:
                    if revision in rev_info.refs:
                        strategy = cache_info.delete_revisions(rev_info.commit_hash)
                        print(f"  Cleaning up cache: freeing {strategy.expected_freed_size_str}")
                        strategy.execute()
                        return
        print(f"  No cached files found for {repo_id} @ {revision}")
    except Exception as e:
        print(f"  Cache cleanup failed (non-fatal): {e}")


def run_checkpoint_sweep(checkpoints, texts, tokenizer,
                          seq_len, trim, cache_dir=None, out_dir=None):
    """Load and analyze each checkpoint sequentially. Drops the model
    after each step to keep memory bounded; only diagnostic arrays
    persist. Skips checkpoints that fail to load (e.g., missing
    revisions for very late training steps). Deletes cached checkpoint
    files after each step to avoid filling up disk.

    If out_dir is set, saves per-step .npz files and writes the JSON
    summary incrementally. On restart, steps with existing .npz files
    are loaded from disk and skipped."""
    results = []
    for step in checkpoints:
        # ---- Resume: load from disk if already computed ----
        if out_dir is not None:
            rpath = _result_path(out_dir, step)
            if rpath.exists():
                print(f"\n=== OLMo-2-1B step {step}: loaded from cache ===")
                results.append(load_result(rpath))
                continue

        print(f"\n=== OLMo-2-1B at step {step} ===")
        repo = EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP else OLMO_MODEL
        revision = step_to_revision(step)
        try:
            model = load_olmo_at_step(step, cache_dir=cache_dir)
        except Exception as e:
            print(f"  FAILED to load step {step}: {e}")
            print(f"  Skipping. Other checkpoints will continue.")
            continue

        try:
            print(f"  Collecting streams ...")
            streams = collect_streams(model, tokenizer, texts, seq_len)
            print(f"    shape (N, n_sub, d_model) = {streams.shape}")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cleanup_checkpoint_cache(repo, revision, cache_dir=cache_dir)

        m = collect_metrics(streams, trim=trim)
        m["step"] = int(step)
        results.append(m)
        del streams

        # ---- Incremental save ----
        if out_dir is not None:
            save_result(out_dir, m)
            summary = build_summary(results, OLMO_MODEL)
            with open(out_dir / "phase_space_trajectory.json", "w") as f:
                json.dump(summary, f, indent=2)

        dtheta_dev = np.abs(np.abs(m["dtheta_per_unit"]) - np.pi / 2)
        print(f"    median |r|={np.median(np.abs(m['r_per_unit'])):.3f}, "
              f"median dtheta_dev={np.median(dtheta_dev):.3f}, "
              f"median resultant={np.median(m['dtheta_resultant_per_unit']):.3f}")

    return results


# ---------------------------------------------------------------------------
# Trajectory plots
# ---------------------------------------------------------------------------

def _percentile_band(ax, steps, values, color, label, marker="o"):
    median = np.median(values, axis=1)
    p25 = np.percentile(values, 25, axis=1)
    p75 = np.percentile(values, 75, axis=1)
    ax.fill_between(steps, p25, p75, color=color, alpha=0.25)
    ax.plot(steps, median, marker + "-", color=color,
            linewidth=2, markersize=6, label=label)


def plot_combined_trajectory(results, out_path, model_name):
    steps = [m["step"] for m in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) dtheta deviation from pi/2
    ax = axes[0, 0]
    devs = np.array([np.abs(np.abs(m["dtheta_per_unit"]) - np.pi / 2)
                     for m in results])
    _percentile_band(ax, steps, devs, "C0", "median + IQR")
    ax.axhline(0, color="black", linestyle="--", alpha=0.4,
               label="ideal quadrature")
    ax.set_xscale("symlog", linthresh=1000)
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
    ax.set_xscale("symlog", linthresh=1000)
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
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$|A_{\mathrm{norm}}|$")
    ax.set_title("(c) Rotation strength: higher = stronger ellipse")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)

    # (1,1) Resultant length (the key prediction)
    ax = axes[1, 1]
    Rs = np.array([m["dtheta_resultant_per_unit"] for m in results])
    _percentile_band(ax, steps, Rs, "C4", "median + IQR")
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.4,
               label="input-independent")
    ax.axhline(0.77, color="C2", linestyle="--", alpha=0.4,
               label="SmolLM2 trained ref (~0.77)")
    ax.axhline(0.90, color="C3", linestyle="--", alpha=0.4,
               label="Pythia trained ref (~0.90)")
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\Delta\theta$ resultant length")
    ax.set_title("(d) Input-locking: 1 = identical phase across inputs")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Phase-space dynamics across training: {model_name}",
        fontsize=14, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_dtheta_distributions(results, out_path, model_name):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=int, nargs="+",
                        default=DEFAULT_CHECKPOINTS,
                        help="OLMo-2-1B training steps to load.")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--trim-sublayers", type=int, default=2,
                        help="Drop this many sublayers from each end.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help="HF cache dir; defaults to $HF_HOME if set.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("phase_space_olmo_2_1b"))
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Model: {OLMO_MODEL}")
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

    print(f"Loading tokenizer for {OLMO_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        OLMO_MODEL,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Sweep ----
    results = run_checkpoint_sweep(
        args.checkpoints, texts, tokenizer,
        args.seq_len, args.trim_sublayers, cache_dir=args.cache_dir,
        out_dir=out_dir,
    )

    if not results:
        raise SystemExit("No checkpoints loaded successfully.")

    # ---- Plots ----
    print(f"\nMaking plots ...")
    plot_combined_trajectory(
        results, out_dir / "trajectory_combined.png", OLMO_MODEL,
    )
    print(f"  trajectory_combined.png")

    plot_dtheta_distributions(
        results, out_dir / "dtheta_distributions.png", OLMO_MODEL,
    )
    print(f"  dtheta_distributions.png")

    plot_resultant_distributions(
        results, out_dir / "resultant_distributions.png", OLMO_MODEL,
    )
    print(f"  resultant_distributions.png")

    # ---- Summary ----
    summary = build_summary(results, OLMO_MODEL)
    with open(out_dir / "phase_space_trajectory.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  phase_space_trajectory.json")

    # ---- Console table ----
    print(f"\n--- Phase-space dynamics across training: {OLMO_MODEL} ---")
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
