"""
Phase Portrait Analysis Across Training: OLMo-2-1B
===================================================

Replicates the phase portrait analysis from Fernando & Guitchounts (2025),
Section 2.6, at multiple OLMo-2-1B training checkpoints. Tracks how the
rotational dynamics of residual stream units evolve across training.

Default checkpoints: 1000 (during backbone formation) and 10000 (after
backbone formation, before gate takeover). Compare with the mature
(step ~1M) results from phase_portraits_olmo_bloom.py.

Checkpoint conventions (from Bray 2026):
  - Step <= 37000: allenai/OLMo-2-0425-1B-early-training
  - Else:         allenai/OLMo-2-0425-1B
  - Revision:     stage1-step{step}-tokens{tokens_b}B
    where tokens_b = ceil(step * 2048 * 1024 / 1e9)

Outputs (per checkpoint, into output_dir/):
  - olmo_step{N}_trajectories.npz       last-token RS trajectories (N, 2L, D)
  - olmo_step{N}_rotations.npz          per-unit rotations + nulls + local
  - olmo_step{N}_phase_portraits.png    8 units (Fig 2A style)
  - olmo_step{N}_rotation_hist.png      Fig 2B
  - olmo_step{N}_rotations_per_unit.png Fig 2C
  - olmo_step{N}_local_rot_density.png  sliding-window density vs depth

Usage:
    python phase_portraits_olmo_checkpoints.py
    python phase_portraits_olmo_checkpoints.py --checkpoints 1000 5000 10000
    python phase_portraits_olmo_checkpoints.py --n_inputs 1000 --n_shuffle 1000
"""

import os
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = os.environ.get("HF_HOME", None)

EARLY_TRAINING_REPO = "allenai/OLMo-2-0425-1B-early-training"
MATURE_REPO = "allenai/OLMo-2-0425-1B"
EARLY_TRAINING_MAX_STEP = 37000

DEFAULT_CHECKPOINTS = [100000]

# Paper-derived input filter
MIN_CHARS = 100
MAX_CHARS = 500
MAX_TOKENS = 512

# Defaults
DEFAULT_N_INPUTS = 200
DEFAULT_N_SHUFFLE = 200
DEFAULT_LOCAL_WINDOW = 8
N_PORTRAIT_UNITS = 8

SAMPLE_TEXTS = [
    "The development of quantum computing has accelerated in recent years, with several companies demonstrating systems that can perform calculations beyond the reach of classical supercomputers. These advances raise important questions about cryptography and security.",
    "In evolutionary biology, the concept of fitness landscapes provides a powerful framework for understanding how populations navigate the space of possible genotypes.",
    "The global supply chain disruptions of the early 2020s revealed deep vulnerabilities in just-in-time manufacturing.",
    "Neural network training involves navigating a high-dimensional loss landscape. The geometry of this landscape, including the presence of saddle points and flat minima, has significant implications for generalization.",
    "Archaeological evidence from the Indus Valley civilization suggests a remarkably sophisticated urban planning system, with standardized brick sizes and elaborate drainage systems.",
    "The interaction between the gut microbiome and the central nervous system, often called the gut-brain axis, has emerged as a major area of research.",
    "Monetary policy in the post-2008 era has been characterized by historically low interest rates and unconventional tools such as quantitative easing.",
    "The study of turbulence remains one of the great unsolved problems in classical physics.",
    "Recent advances in single-cell RNA sequencing have transformed our understanding of cellular heterogeneity within tissues previously thought to be homogeneous.",
    "The philosophy of language has grappled with the relationship between meaning and reference since Frege's distinction between sense and reference.",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wikitext_inputs(n_inputs):
    """Load WikiText-2 sequences filtered by character length, matching the paper."""
    try:
        from datasets import load_dataset
        print(f"  Loading WikiText-2 (filter {MIN_CHARS}-{MAX_CHARS} chars)...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                          cache_dir=CACHE_DIR)
        texts = []
        for ex in ds:
            t = ex["text"].strip()
            if MIN_CHARS < len(t) < MAX_CHARS:
                texts.append(t)
            if len(texts) >= n_inputs:
                break
        print(f"  Loaded {len(texts)} WikiText sequences")
        return texts
    except Exception as e:
        print(f"  WikiText load failed ({type(e).__name__}: {e})")
        print(f"  Falling back to SAMPLE_TEXTS (cycled)")
        cycles = (n_inputs // len(SAMPLE_TEXTS)) + 1
        return (SAMPLE_TEXTS * cycles)[:n_inputs]


# ---------------------------------------------------------------------------
# Checkpoint loading (matches Bray 2026 conventions)
# ---------------------------------------------------------------------------

def step_to_revision(step):
    tokens_b = math.ceil(step * 2048 * 1024 / 1_000_000_000)
    return f"stage1-step{step}-tokens{tokens_b}B"


def load_olmo_at_step(step, device):
    repo = EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP else MATURE_REPO
    revision = step_to_revision(step)
    print(f"  Loading {repo} at {revision}...")

    # Use float32 to match the precision used in Bray (2026) checkpoint
    # experiments and to avoid numerical noise in early-training weights.
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )
    model.to(device).eval()

    # Tokenizer pulled from the mature repo; same across all checkpoints.
    tokenizer = AutoTokenizer.from_pretrained(MATURE_REPO, cache_dir=CACHE_DIR)
    return model, tokenizer, repo, revision


# ---------------------------------------------------------------------------
# Cache cleanup
# ---------------------------------------------------------------------------

def delete_hf_revision(repo_id, revision):
    """
    Delete a specific cached revision from the HuggingFace Hub cache to free
    local disk space. Safe to call with a revision that isn't cached; in that
    case it prints a notice and returns None.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        print(f"    huggingface_hub unavailable; skipping cache cleanup")
        return None

    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        print(f"    scan_cache_dir failed ({type(e).__name__}: {e})")
        return None

    commit_hashes = []
    for repo in cache_info.repos:
        if repo.repo_id != repo_id:
            continue
        for rev in repo.revisions:
            if revision in rev.refs:
                commit_hashes.append(rev.commit_hash)

    if not commit_hashes:
        print(f"    No cached revision '{revision}' found for {repo_id}")
        return None

    strategy = cache_info.delete_revisions(*commit_hashes)
    freed = strategy.expected_freed_size_str
    print(f"    Deleting {repo_id} @ {revision}  (~{freed})")
    strategy.execute()
    return strategy.expected_freed_size


# ---------------------------------------------------------------------------
# Residual stream capture (with kwargs-safe hook)
# ---------------------------------------------------------------------------

class ResidualStreamCapture:
    """
    Captures the residual stream at 2L points per forward pass:
        h_attn[l]: residual stream entering attention sub-block at layer l
        h_mlp[l]:  residual stream entering MLP sub-block at layer l

    Both BEFORE any LayerNorm, matching the Dynamics paper. OLMo-2 applies
    LN after each sub-block, so self_attn and mlp inputs are the raw
    residual stream.

    Hooks registered with with_kwargs=True because HF transformers calls
    self_attn with hidden_states as a kwarg (empty positional args tuple).
    """

    def __init__(self, model):
        self.h_attn = {}
        self.h_mlp = {}
        self.hooks = []
        layers = model.model.layers
        for i, layer in enumerate(layers):
            self.hooks.append(layer.self_attn.register_forward_pre_hook(
                self._make_hook(self.h_attn, i), with_kwargs=True))
            self.hooks.append(layer.mlp.register_forward_pre_hook(
                self._make_hook(self.h_mlp, i), with_kwargs=True))

    @staticmethod
    def _make_hook(storage, idx):
        def hook_fn(module, args, kwargs):
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                x = args[0]
            elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                x = kwargs["hidden_states"]
            else:
                x = None
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor):
                        x = v
                        break
                if x is None:
                    raise RuntimeError(
                        f"Could not locate hidden_states for "
                        f"{type(module).__name__} at layer index {idx}; "
                        f"args types={[type(a).__name__ for a in args]}, "
                        f"kwargs keys={list(kwargs.keys())}")
            storage[idx] = x.detach().float().cpu()
        return hook_fn

    def clear(self):
        self.h_attn.clear()
        self.h_mlp.clear()

    def last_token_trajectory(self, n_layers):
        """Returns interleaved trajectory at last token: shape (2L, D)."""
        rows = []
        for l in range(n_layers):
            rows.append(self.h_attn[l][0, -1, :].numpy())
            rows.append(self.h_mlp[l][0, -1, :].numpy())
        return np.stack(rows, axis=0)

    def remove(self):
        for h in self.hooks:
            h.remove()


def collect_trajectories(model, tokenizer, n_layers, device, texts):
    """Run model on each text, capture last-token RS trajectory of shape (2L, D)."""
    capture = ResidualStreamCapture(model)
    trajectories = []
    for idx, text in enumerate(texts):
        capture.clear()
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=MAX_TOKENS)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            model(**tokens)
        trajectories.append(capture.last_token_trajectory(n_layers))
        if (idx + 1) % 50 == 0:
            print(f"    Processed {idx + 1}/{len(texts)} inputs")
    capture.remove()
    return np.stack(trajectories, axis=0)


# ---------------------------------------------------------------------------
# Phase portrait math
# ---------------------------------------------------------------------------

def compute_rotations_vec(traj):
    """
    Per-unit rotation counts.
    traj: (T, D); returns (D,).
    """
    a = np.asarray(traj, dtype=np.float64)
    g = np.gradient(a, axis=0)
    x = a - a[0:1, :]
    y = g - g[0:1, :]
    dx = np.diff(x, axis=0)
    dy = np.diff(y, axis=0)
    theta = np.arctan2(dy, dx)
    dtheta = np.diff(theta, axis=0)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    return np.sum(dtheta, axis=0) / (2 * np.pi)


def local_rotation_density_vec(traj, window):
    """
    Sliding-window rotation count across layers, all units.
    traj: (T, D); returns (T - window + 1, D).
    """
    a = np.asarray(traj, dtype=np.float64)
    g = np.gradient(a, axis=0)
    T, D = a.shape
    n_w = T - window + 1
    out = np.zeros((n_w, D))
    for s in range(n_w):
        e = s + window
        sub_a = a[s:e] - a[s:s + 1]
        sub_g = g[s:e] - g[s:s + 1]
        dx = np.diff(sub_a, axis=0)
        dy = np.diff(sub_g, axis=0)
        theta = np.arctan2(dy, dx)
        dtheta = np.diff(theta, axis=0)
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        out[s] = np.sum(dtheta, axis=0) / (2 * np.pi)
    return out


def shuffle_null(traj, n_shuffle, rng):
    """
    Null rotations under shared random permutations of the layer ordering.
    Returns (n_shuffle, D).
    """
    T, D = traj.shape
    out = np.empty((n_shuffle, D))
    for k in range(n_shuffle):
        perm = rng.permutation(T)
        out[k] = compute_rotations_vec(traj[perm])
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_phase_portraits(mean_traj, units, model_label, out_path):
    T, D = mean_traj.shape
    g_all = np.gradient(mean_traj, axis=0)
    cmap = plt.cm.viridis

    n_cols = 4
    n_rows = (len(units) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 3.2 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes).flatten()

    for k, u in enumerate(units):
        ax = axes[k]
        a = mean_traj[:, u]
        g = g_all[:, u]
        x = a - a[0]
        y = g - g[0]
        points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        layer_idx = np.arange(T - 1)
        norm = plt.Normalize(vmin=0, vmax=T - 1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.2)
        lc.set_array(layer_idx)
        ax.add_collection(lc)
        ax.scatter([0], [0], s=18, c="black", marker="o", zorder=5)
        m = max(abs(x).max(), abs(y).max()) * 1.1 + 1e-8
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        ax.set_xlabel("Activation (centered)")
        ax.set_ylabel("Gradient (centered)")
        R = compute_rotations_vec(mean_traj[:, u:u + 1])[0]
        ax.set_title(f"unit {u}  R={R:.2f}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    for k in range(len(units), len(axes)):
        axes[k].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, T - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), shrink=0.6, label="Effective layer")
    fig.suptitle(f"Phase portraits: {model_label}", fontsize=12)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"    Saved {out_path}")


def plot_rotation_histogram(rotations, null_means, model_label, out_path):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    lo = min(rotations.min(), null_means.min()) - 0.5
    hi = max(rotations.max(), null_means.max()) + 0.5
    bins = np.linspace(lo, hi, 50)
    ax.hist(null_means, bins=bins, alpha=0.6, label="Shuffled (per-unit mean)",
            color="#888888")
    ax.hist(rotations, bins=bins, alpha=0.6, label="Actual", color="#5BA9A6")
    ax.axvline(rotations.mean(), linestyle="--", color="#1a6f6b",
               label=f"Actual mean = {rotations.mean():.2f}")
    ax.axvline(null_means.mean(), linestyle="--", color="#444444",
               label=f"Null mean = {null_means.mean():.2f}")
    ax.set_xlabel("Number of rotations")
    ax.set_ylabel("Number of units")
    ax.set_title(f"Rotation distribution: {model_label}")
    ax.legend(fontsize=9)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"    Saved {out_path}")


def plot_rotations_per_unit(rotations, null_means, model_label, out_path):
    fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    idx = np.arange(len(rotations))
    ax.fill_between(idx, 0, rotations, color="#9BD3CE", alpha=0.6, label="Actual")
    ax.fill_between(idx, 0, null_means, color="#444444", alpha=0.5,
                    label="Shuffled (per-unit mean)")
    ax.set_xlabel("Unit")
    ax.set_ylabel("Number of rotations")
    ax.set_title(f"Rotations per unit: {model_label}")
    ax.legend(fontsize=9)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"    Saved {out_path}")


def plot_local_rotation_density(local_dens, window, model_label, out_path):
    n_w, D = local_dens.shape
    x = np.arange(n_w) + (window - 1) / 2.0
    abs_d = np.abs(local_dens)
    mean = abs_d.mean(axis=1)
    q25 = np.quantile(abs_d, 0.25, axis=1)
    q75 = np.quantile(abs_d, 0.75, axis=1)
    q05 = np.quantile(abs_d, 0.05, axis=1)
    q95 = np.quantile(abs_d, 0.95, axis=1)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.fill_between(x, q05, q95, color="#9BD3CE", alpha=0.3, label="5-95%")
    ax.fill_between(x, q25, q75, color="#5BA9A6", alpha=0.5, label="25-75%")
    ax.plot(x, mean, color="#1a6f6b", linewidth=2, label="Mean")
    ax.set_xlabel(f"Effective layer (window center, window size = {window})")
    ax.set_ylabel("|Local rotation density|")
    ax.set_title(f"Local rotation density across depth: {model_label}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"    Saved {out_path}")


# ---------------------------------------------------------------------------
# Per-checkpoint driver
# ---------------------------------------------------------------------------

def run_checkpoint(step, device, output_dir, n_inputs, n_shuffle,
                   local_window, seed, texts, cleanup_cache):
    label = f"OLMo-2-1B step {step}"
    file_prefix = f"olmo_step{step}"

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    model, tokenizer, repo, revision = load_olmo_at_step(step, device)
    n_layers = len(model.model.layers)
    n_layers_eff = 2 * n_layers
    print(f"  n_layers={n_layers}  (2L effective layers = {n_layers_eff})")

    # Collect trajectories
    print(f"  Collecting last-token trajectories for {len(texts)} inputs...")
    trajectories = collect_trajectories(model, tokenizer, n_layers, device, texts)
    print(f"  Trajectories shape: {trajectories.shape}  (N_inputs, 2L, D)")

    # Free model from memory; also free the on-disk cache if requested.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if cleanup_cache:
        print(f"  Cleaning HF cache for {repo} @ {revision}...")
        delete_hf_revision(repo, revision)

    # Mean trajectory for rotation analysis
    mean_traj = trajectories.mean(axis=0)
    D = mean_traj.shape[1]

    # Rotations
    print(f"  Computing rotations for {D} units...")
    rotations = compute_rotations_vec(mean_traj)

    # Shuffle nulls
    print(f"  Computing shuffle nulls ({n_shuffle} permutations)...")
    rng = np.random.default_rng(seed)
    null_full = shuffle_null(mean_traj, n_shuffle, rng)
    null_means = null_full.mean(axis=0)

    # Local rotation density
    print(f"  Computing local rotation density (window={local_window})...")
    local_dens = local_rotation_density_vec(mean_traj, local_window)

    # Portrait unit selection (spread across |R| quartiles)
    abs_R = np.abs(rotations)
    order = np.argsort(abs_R)[::-1]
    portrait_units = np.array([
        order[0], order[1],
        order[int(D * 0.25)], order[int(D * 0.25) + 1],
        order[int(D * 0.50)], order[int(D * 0.50) + 1],
        order[int(D * 0.75)], order[int(D * 0.75) + 1],
    ])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, f"{file_prefix}_trajectories.npz")
    np.savez_compressed(
        traj_path,
        trajectories=trajectories.astype(np.float32),
        mean_trajectory=mean_traj.astype(np.float32),
        model=label,
        arch="olmo",
        step=step,
        n_layers=n_layers,
        n_layers_effective=n_layers_eff,
        d_model=D,
        layer_kind=np.array(["attn", "mlp"] * n_layers),
    )
    print(f"  Saved {traj_path}")

    rot_path = os.path.join(output_dir, f"{file_prefix}_rotations.npz")
    np.savez_compressed(
        rot_path,
        rotations=rotations,
        null_means=null_means,
        null_full=null_full.astype(np.float32),
        local_rotation_density=local_dens.astype(np.float32),
        local_window=local_window,
        portrait_units=portrait_units,
        step=step,
    )
    print(f"  Saved {rot_path}")

    plot_phase_portraits(
        mean_traj, portrait_units.tolist(), label,
        os.path.join(output_dir, f"{file_prefix}_phase_portraits.png"))
    plot_rotation_histogram(
        rotations, null_means, label,
        os.path.join(output_dir, f"{file_prefix}_rotation_hist.png"))
    plot_rotations_per_unit(
        rotations, null_means, label,
        os.path.join(output_dir, f"{file_prefix}_rotations_per_unit.png"))
    plot_local_rotation_density(
        local_dens, local_window, label,
        os.path.join(output_dir, f"{file_prefix}_local_rot_density.png"))

    print(f"\n  Summary [{label}]:")
    print(f"    Mean |R| actual:    {abs_R.mean():.3f}")
    print(f"    Mean |R| shuffled:  {np.abs(null_means).mean():.3f}")
    print(f"    Mean R (signed):    {rotations.mean():.3f}")
    print(f"    Units with |R| > 1: {(abs_R > 1).sum()}/{D}")
    print(f"    Units with |R| > 2: {(abs_R > 2).sum()}/{D}")
    abs_d_mean = np.abs(local_dens).mean(axis=1)
    peak_window = int(np.argmax(abs_d_mean))
    peak_layer = peak_window + (local_window - 1) / 2.0
    print(f"    Local density peaks at effective layer {peak_layer:.1f} "
          f"(of {n_layers_eff})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Phase portrait analysis at OLMo-2-1B training checkpoints")
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="results/phase_portraits")
    p.add_argument("--checkpoints", nargs="+", type=int,
                   default=DEFAULT_CHECKPOINTS,
                   help="Training step numbers to evaluate")
    p.add_argument("--n_inputs", type=int, default=DEFAULT_N_INPUTS,
                   help="Number of WikiText inputs (paper uses 1000)")
    p.add_argument("--n_shuffle", type=int, default=DEFAULT_N_SHUFFLE,
                   help="Shuffle permutations")
    p.add_argument("--local_window", type=int, default=DEFAULT_LOCAL_WINDOW,
                   help="Window size for local rotation density (in 2L units)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep_checkpoints", action="store_true",
                   help="Keep downloaded checkpoints in the HF cache "
                        "(default: delete each one after use to save disk)")
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    print(f"Checkpoints: {args.checkpoints}")
    print(f"Config: n_inputs={args.n_inputs}  n_shuffle={args.n_shuffle}  "
          f"local_window={args.local_window}  seed={args.seed}")

    # Load inputs once; reuse across checkpoints for direct comparability.
    texts = load_wikitext_inputs(args.n_inputs)

    for step in args.checkpoints:
        run_checkpoint(step, device, args.output_dir, args.n_inputs,
                       args.n_shuffle, args.local_window, args.seed, texts,
                       cleanup_cache=not args.keep_checkpoints)

    print("\nDone.")


if __name__ == "__main__":
    main()
