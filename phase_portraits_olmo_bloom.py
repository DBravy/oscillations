"""
Phase Portrait Analysis: OLMo-2-1B and BLOOM-1b1
=================================================

Replicates the phase portrait analysis from Fernando & Guitchounts (2025),
"Transformer Dynamics: A neuroscientific approach to interpretability of
large language models", Section 2.6 and Figure 2.

For each residual stream unit at the last token position, we build a phase
portrait of (activation, layer-wise gradient) and count rotations in this
phase space. Comparison to a shuffle-control null follows the paper's
protocol.

Motivation: tests whether the rotation-to-drift transition in phase portraits
coincides with the depth gradient of gate coordination (OLMo, SwiGLU) or
static MLP weight alignment (BLOOM, GELU).

Method (from the paper, Section 2.6):
  * Capture residual stream at 2L points per forward pass:
      h_attn[l]: residual stream entering attention sub-block at layer l
      h_mlp[l]:  residual stream entering MLP sub-block at layer l
    Both BEFORE any LayerNorm.
  * Take last token position only.
  * Mean across N inputs gives the mean trajectory of shape (2L, D).
  * Phase portrait per unit u: x_l = a_l - a_0, y_l = grad(a)_l - grad(a)_0.
  * Rotation count R via cumulative tangent-angle change (wrap to [-pi, pi]).
  * Null: permute layer order; paper uses 1000 shuffles per unit.

Architecture hooks:
  OLMo-2 applies LN AFTER each sub-block (norm-reordered), so self_attn and
  mlp receive the raw residual stream. Pre-hook those modules.
  BLOOM uses pre-LN; the raw residual stream is the input to input_layernorm
  and post_attention_layernorm. Pre-hook those.

Outputs (per model, into output_dir/):
  - {arch}_trajectories.npz       last-token RS trajectories, shape (N, 2L, D)
  - {arch}_rotations.npz          per-unit rotations + nulls + local densities
  - {arch}_phase_portraits.png    8 units, matches Fig 2A
  - {arch}_rotation_hist.png      matches Fig 2B
  - {arch}_rotations_per_unit.png matches Fig 2C
  - {arch}_local_rot_density.png  sliding-window rotation density vs depth

Usage:
    python phase_portraits_olmo_bloom.py
    python phase_portraits_olmo_bloom.py --models olmo
    python phase_portraits_olmo_bloom.py --n_inputs 500 --n_shuffle 1000
"""

import os
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

OLMO_REPO = "allenai/OLMo-2-0425-1B"
BLOOM_REPO = "bigscience/bloom-1b1"

# Paper-derived input filter
MIN_CHARS = 100
MAX_CHARS = 500
MAX_TOKENS = 512

# Defaults (overrideable via CLI)
DEFAULT_N_INPUTS = 200       # paper uses 1000; 200 gives stable means and is fast
DEFAULT_N_SHUFFLE = 200      # paper uses 1000
DEFAULT_LOCAL_WINDOW = 8     # window size for local rotation density (in 2L units)
N_PORTRAIT_UNITS = 8         # match Fig 2A grid

# Fallback texts in case WikiText cannot be loaded
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
# Residual stream capture
# ---------------------------------------------------------------------------

class ResidualStreamCapture:
    """
    Captures the residual stream at 2L points per forward pass:
        h_attn[l]: residual stream entering attention sub-block at layer l
        h_mlp[l]:  residual stream entering MLP sub-block at layer l

    Both captured BEFORE any LayerNorm, matching the Dynamics paper:
    'captured before layernorm and the attention operation (pre-Attn) and
    before the MLP at each layer'.

    OLMo-2: norm-reordered (LN applied after each sub-block output), so
    self_attn and mlp inputs are the raw residual stream. Pre-hook them.

    BLOOM: pre-LN. The raw residual stream is the input to input_layernorm
    (pre-attn) and post_attention_layernorm (pre-MLP). Pre-hook those.
    """

    def __init__(self, model, arch):
        self.arch = arch
        self.h_attn = {}
        self.h_mlp = {}
        self.hooks = []
        self._register(model)

    def _register(self, model):
        # with_kwargs=True so the hook can find hidden_states whether it's
        # passed positionally or as a kwarg. Newer HF transformers (including
        # OLMo-2) call self_attn(hidden_states=..., ...) as kwargs, leaving
        # the positional args tuple empty.
        if self.arch == "olmo":
            layers = model.model.layers
            for i, layer in enumerate(layers):
                self.hooks.append(layer.self_attn.register_forward_pre_hook(
                    self._make_hook(self.h_attn, i), with_kwargs=True))
                self.hooks.append(layer.mlp.register_forward_pre_hook(
                    self._make_hook(self.h_mlp, i), with_kwargs=True))
        elif self.arch == "bloom":
            layers = model.transformer.h
            for i, layer in enumerate(layers):
                self.hooks.append(layer.input_layernorm.register_forward_pre_hook(
                    self._make_hook(self.h_attn, i), with_kwargs=True))
                self.hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
                    self._make_hook(self.h_mlp, i), with_kwargs=True))
        else:
            raise ValueError(f"Unknown arch: {self.arch}")

    @staticmethod
    def _make_hook(storage, idx):
        def hook_fn(module, args, kwargs):
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                x = args[0]
            elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                x = kwargs["hidden_states"]
            else:
                # Last-resort fallback: first tensor-valued kwarg.
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
        """
        Returns interleaved trajectory at last token position: shape (2L, D).
        Order: h_attn[0], h_mlp[0], h_attn[1], h_mlp[1], ...
        """
        rows = []
        for l in range(n_layers):
            rows.append(self.h_attn[l][0, -1, :].numpy())
            rows.append(self.h_mlp[l][0, -1, :].numpy())
        return np.stack(rows, axis=0)

    def remove(self):
        for h in self.hooks:
            h.remove()


def collect_trajectories(model, tokenizer, arch, n_layers, device, texts):
    """Run model on each text, capture last-token RS trajectory of shape (2L, D)."""
    capture = ResidualStreamCapture(model, arch)
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
# Phase portrait math (Section 2.6 of Dynamics paper)
# ---------------------------------------------------------------------------

def compute_rotations_vec(traj):
    """
    Per-unit rotation counts, vectorized.

    traj: shape (T, D) where T = 2L and D = units
    Returns: shape (D,)

    R_u = (1 / 2*pi) * sum_l wrap(theta_{l+1} - theta_l)
    where theta_l = arctan2(y_{l+1} - y_l, x_{l+1} - x_l),
    x_l = a_l - a_0, y_l = grad(a)_l - grad(a)_0, grad along layer axis.
    """
    a = np.asarray(traj, dtype=np.float64)             # (T, D)
    g = np.gradient(a, axis=0)                          # (T, D)
    x = a - a[0:1, :]
    y = g - g[0:1, :]
    dx = np.diff(x, axis=0)                             # (T-1, D)
    dy = np.diff(y, axis=0)
    theta = np.arctan2(dy, dx)                          # (T-1, D)
    dtheta = np.diff(theta, axis=0)                     # (T-2, D)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    return np.sum(dtheta, axis=0) / (2 * np.pi)         # (D,)


def local_rotation_density_vec(traj, window):
    """
    Sliding-window rotation count across layers, all units.

    traj: shape (T, D)
    window: window size (in 2L effective-layer units)
    Returns: shape (T - window + 1, D)

    For each window starting at s, computes the rotation count using
    sub_a = a[s:s+window] - a[s] and likewise for the gradient. The window
    center sits at s + (window - 1) / 2 on the effective-layer axis.
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
    Null rotations under random permutation of the layer ordering.

    Uses shared permutations across units (same permutation applied to all
    units for each shuffle iteration). This is statistically equivalent in
    expectation to the paper's per-unit independent shuffles for the
    purposes of estimating per-unit null means and the unit-wise histogram.
    Per-unit independence can be obtained by setting n_shuffle higher.

    Returns: shape (n_shuffle, D)
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
    """Fig 2A: 8 units in centered (activation, gradient) space, colored by layer."""
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
    """Fig 2B: distribution of rotations across units, actual vs shuffled."""
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
    """Fig 2C: rotations per unit, actual and shuffled."""
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
    """
    Local rotation density vs effective-layer position, aggregated across units.

    local_dens: shape (n_windows, D)
    Plots |rotation density| since the sign of rotation differs per unit.
    """
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
# Per-model driver
# ---------------------------------------------------------------------------

def run_model(model_name, arch, device, label, output_dir, n_inputs,
              n_shuffle, local_window, seed):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    # Load model
    dtype = torch.float32 if device == "cpu" else torch.float16
    kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    if CACHE_DIR:
        kwargs["cache_dir"] = CACHE_DIR
    if arch == "bloom":
        kwargs["use_safetensors"] = False

    print(f"  Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

    if arch == "olmo":
        n_layers = len(model.model.layers)
    elif arch == "bloom":
        n_layers = len(model.transformer.h)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    n_layers_eff = 2 * n_layers
    print(f"  n_layers={n_layers}  (2L effective layers = {n_layers_eff})")

    # Inputs
    texts = load_wikitext_inputs(n_inputs)

    # Collect trajectories
    print(f"  Collecting last-token trajectories for {len(texts)} inputs...")
    trajectories = collect_trajectories(model, tokenizer, arch, n_layers,
                                        device, texts)
    print(f"  Trajectories shape: {trajectories.shape}  (N_inputs, 2L, D)")

    # Free model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Mean trajectory: paper's basis for phase portrait analysis
    mean_traj = trajectories.mean(axis=0)            # (2L, D)
    D = mean_traj.shape[1]

    # Rotation counts
    print(f"  Computing rotations for {D} units...")
    rotations = compute_rotations_vec(mean_traj)     # (D,)

    # Shuffle nulls (shared permutations across units, vectorized)
    print(f"  Computing shuffle nulls ({n_shuffle} permutations)...")
    rng = np.random.default_rng(seed)
    null_full = shuffle_null(mean_traj, n_shuffle, rng)  # (n_shuffle, D)
    null_means = null_full.mean(axis=0)                  # (D,)

    # Local rotation density
    print(f"  Computing local rotation density (window={local_window})...")
    local_dens = local_rotation_density_vec(mean_traj, local_window)  # (n_w, D)

    # Pick units for portrait grid: spread across |R| quartiles
    abs_R = np.abs(rotations)
    order = np.argsort(abs_R)[::-1]
    quartile_picks = [
        order[0], order[1],
        order[int(D * 0.25)], order[int(D * 0.25) + 1],
        order[int(D * 0.50)], order[int(D * 0.50) + 1],
        order[int(D * 0.75)], order[int(D * 0.75) + 1],
    ]
    portrait_units = np.array(quartile_picks)

    # Save data files
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, f"{arch}_trajectories.npz")
    np.savez_compressed(
        traj_path,
        trajectories=trajectories.astype(np.float32),
        mean_trajectory=mean_traj.astype(np.float32),
        model=label,
        arch=arch,
        n_layers=n_layers,
        n_layers_effective=n_layers_eff,
        d_model=D,
        layer_kind=np.array(["attn", "mlp"] * n_layers),  # which sub-block each row is
    )
    print(f"  Saved {traj_path}")

    rot_path = os.path.join(output_dir, f"{arch}_rotations.npz")
    np.savez_compressed(
        rot_path,
        rotations=rotations,
        null_means=null_means,
        null_full=null_full.astype(np.float32),
        local_rotation_density=local_dens.astype(np.float32),
        local_window=local_window,
        portrait_units=portrait_units,
    )
    print(f"  Saved {rot_path}")

    # Plots
    plot_phase_portraits(
        mean_traj, portrait_units.tolist(), label,
        os.path.join(output_dir, f"{arch}_phase_portraits.png"))
    plot_rotation_histogram(
        rotations, null_means, label,
        os.path.join(output_dir, f"{arch}_rotation_hist.png"))
    plot_rotations_per_unit(
        rotations, null_means, label,
        os.path.join(output_dir, f"{arch}_rotations_per_unit.png"))
    plot_local_rotation_density(
        local_dens, local_window, label,
        os.path.join(output_dir, f"{arch}_local_rot_density.png"))

    # Summary
    print(f"\n  Summary [{label}]:")
    print(f"    Mean |R| actual:    {np.abs(rotations).mean():.3f}")
    print(f"    Mean |R| shuffled:  {np.abs(null_means).mean():.3f}")
    print(f"    Mean R (signed):    {rotations.mean():.3f}")
    print(f"    Units with |R| > 1: {(abs_R > 1).sum()}/{D}")
    print(f"    Units with |R| > 2: {(abs_R > 2).sum()}/{D}")
    # Where does local density peak on average?
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
        description="Phase portrait analysis: OLMo-2-1B and BLOOM-1b1")
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="results/phase_portraits")
    p.add_argument("--models", nargs="+", default=["olmo", "bloom"],
                   choices=["olmo", "bloom"])
    p.add_argument("--n_inputs", type=int, default=DEFAULT_N_INPUTS,
                   help="Number of WikiText inputs (paper uses 1000)")
    p.add_argument("--n_shuffle", type=int, default=DEFAULT_N_SHUFFLE,
                   help="Shuffle permutations (paper uses 1000 per unit)")
    p.add_argument("--local_window", type=int, default=DEFAULT_LOCAL_WINDOW,
                   help="Window size for local rotation density (in 2L units)")
    p.add_argument("--seed", type=int, default=0)
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
    print(f"Config: n_inputs={args.n_inputs}  n_shuffle={args.n_shuffle}  "
          f"local_window={args.local_window}  seed={args.seed}")

    if "olmo" in args.models:
        run_model(OLMO_REPO, "olmo", device, "OLMo-2-1B (SwiGLU)",
                  args.output_dir, args.n_inputs, args.n_shuffle,
                  args.local_window, args.seed)

    if "bloom" in args.models:
        run_model(BLOOM_REPO, "bloom", device, "BLOOM-1b1 (GELU)",
                  args.output_dir, args.n_inputs, args.n_shuffle,
                  args.local_window, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
