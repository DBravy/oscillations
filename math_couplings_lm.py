"""
Per-problem functional coupling structure for arithmetic problems.

Adapts the residual-stream coupling analysis from phase_couplings_lm.py:
instead of comparing token positions within a single generation, we
compare math problems. For each problem (e.g., "23 + 47 =") we run a
single forward pass and capture the residual stream at the last token
position, where the model is computing the next-token prediction (the
answer). The depth-axis cross-correlation among the d_model units at
that position defines a "coupling fingerprint" for the problem.

Goals:
  * Quantify within-operation variation: are all addition problems'
    fingerprints similar?
  * Quantify between-operation contrast: do addition and multiplication
    use systematically different couplings?

By default, problems are matched: the same (a, b) operand pair is used
for both "a + b =" and "a * b =", so any systematic between-operation
difference is attributable to the operation rather than the operands.

Pipeline:
  1. Sample n_per_op operand pairs (a, b).
  2. For each pair, build "a + b =" and "a * b =".
  3. Forward each through the model; capture residual stream at the
     last token position via LayerNorm-input hooks.
  4. Stack into (n_problems, n_sub, d_model), problems ordered
     [all addition, then all multiplication].
  5. Per-problem: depth-axis cross-correlation R among units, |R|
     Laplacian, Fiedler vector, top-k Laplacian eigenspace, spectral
     entropy, mean |R|, effective rank.
  6. Cross-problem: |cos(Fiedler)|, top-k eigenspace overlap, |R|
     Frobenius cosine. Each is an (n_problems x n_problems) matrix
     with a natural 2x2 block structure (add/add, add/mul, mul/mul).
  7. Within-operation vs between-operation similarity summary for each
     of the three measures.

Why the last position:
  In a causal transformer, the residual stream at the last token of a
  prompt is what produces the next-token prediction. For "23 + 47 =",
  the last position is where the model decides what number comes next,
  so it is the natural locus of the arithmetic computation.

Outputs in math_couplings_<model_slug>/:
  similarity_fiedler.png        Problem x problem |cos(Fiedler)|.
  similarity_eigenspace.png     Problem x problem top-k eigenspace overlap.
  similarity_corr.png           Problem x problem |R| Frobenius cosine.
  fiedler_heatmap.png           Fiedler value across problems, top-N units.
  per_problem_diagnostics.png   Effective rank, mean |R|, spectral entropy.
  within_between_summary.png    Within-op vs between-op similarity, 3 panels.
  summary.json                  Per-problem scalars + problem strings.

Usage:
  python math_couplings_lm.py
  python math_couplings_lm.py --n-per-op 32
  python math_couplings_lm.py --operand-low 1 --operand-high 9
  python math_couplings_lm.py --model HuggingFaceTB/SmolLM2-360M
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


# ---------------------------------------------------------------------------
# Stream collection (last token position only)
# ---------------------------------------------------------------------------

def get_hook_targets(model):
    """Locate sublayer-input hooks. LayerNorm-input hooks for
    Llama/SmolLM/Pythia/GPT-NeoX (input_layernorm + post_attention_layernorm),
    ln_1/ln_2 for GPT-2."""
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


def _capture_last_position(model, input_ids):
    """Single forward pass + hooks; return (n_sub, d_model) at the last token."""
    targets = get_hook_targets(model)
    captures = []
    hooks = [
        t.register_forward_hook(
            lambda m, i, o, c=captures: c.append(i[0].detach())
        )
        for t in targets
    ]
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()
    # Each captures[k] has shape (1, seq_len, d_model). Take last position.
    streams = torch.stack([c[0, -1, :] for c in captures], dim=0)
    return streams.float().cpu().numpy()  # (n_sub, d_model)


def capture_problems(model, tokenizer, problems, log_every=8):
    """Run one forward pass per problem, capture the last-position stream.

    Returns:
        streams:     (n_problems, n_sub, d_model)
        last_tokens: list of decoded last-token strings (for diagnostics).
    """
    model.eval()
    out = []
    last_tokens = []
    for k, problem in enumerate(problems):
        enc = tokenizer(problem, return_tensors="pt", truncation=False)
        ids = enc["input_ids"].to(DEVICE)
        out.append(_capture_last_position(model, ids))
        last_tokens.append(tokenizer.decode([int(ids[0, -1].item())]))
        if (k + 1) % log_every == 0 or k == len(problems) - 1:
            print(f"    captured {k+1}/{len(problems)}")
    return np.stack(out, axis=0), last_tokens


# ---------------------------------------------------------------------------
# Per-problem correlation coupling
# ---------------------------------------------------------------------------

def per_problem_correlation_coupling(streams, trim=2, k_eigvecs=10):
    """For each problem (axis 0), compute depth-axis cross-correlation
    among d_model units, build the |R| Laplacian, and extract spectral
    diagnostics. Identical math to per_position_correlation_coupling in
    the LM script; the analyzed axis is now problems instead of token
    positions.

    Parameters
    ----------
    streams : (n_problems, n_sub, d_model) float
    trim : int
        Drop this many sublayers from each end of the depth axis.
    k_eigvecs : int
        Number of low Laplacian eigenvectors to retain per problem.
    """
    if trim > 0:
        if streams.shape[1] <= 2 * trim + 2:
            raise ValueError(
                f"Cannot trim {trim} from each end of {streams.shape[1]} sublayers."
            )
        streams = streams[:, trim:streams.shape[1] - trim, :]
    n_problems, L, D = streams.shape

    # Center each (problem, unit) trajectory along the sublayer axis.
    streams_c = streams - streams.mean(axis=1, keepdims=True)

    abs_corr_matrices = np.zeros((n_problems, D, D), dtype=np.float32)
    fiedler_vecs = np.zeros((n_problems, D), dtype=np.float32)
    fiedler_eigvals = np.zeros(n_problems, dtype=np.float32)
    top_k_eigvecs = np.zeros((n_problems, D, k_eigvecs), dtype=np.float32)
    top_k_eigvals = np.zeros((n_problems, k_eigvecs), dtype=np.float32)
    spectral_entropy = np.zeros(n_problems, dtype=np.float32)
    mean_off_diag = np.zeros(n_problems, dtype=np.float32)
    effective_rank = np.zeros(n_problems, dtype=np.float32)

    for p in range(n_problems):
        X = streams_c[p]                              # (L, D)
        norms = np.linalg.norm(X, axis=0)             # (D,)
        norms = np.maximum(norms, 1e-12)
        # R[i, j] = (X[:, i] . X[:, j]) / (||X[:, i]|| * ||X[:, j]||)
        R = (X.T @ X) / np.outer(norms, norms)        # (D, D)
        affinity = np.abs(R).astype(np.float64)
        np.fill_diagonal(affinity, 0.0)
        abs_corr_matrices[p] = affinity.astype(np.float32)

        degree = affinity.sum(axis=1)
        laplacian = np.diag(degree) - affinity
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        # eigvals[0] is the trivial (~0) one corresponding to constant vec.
        fiedler_vecs[p] = eigvecs[:, 1]
        fiedler_eigvals[p] = eigvals[1]
        top_k_eigvals[p] = eigvals[1:k_eigvecs + 1]
        top_k_eigvecs[p] = eigvecs[:, 1:k_eigvecs + 1]

        # Spectral entropy of normalized non-trivial eigenvalues.
        nontriv = np.maximum(eigvals[1:], 0.0)
        s = nontriv.sum()
        if s > 0:
            ne = nontriv / s
            spectral_entropy[p] = float(-np.sum(ne * np.log(ne + 1e-30)))
        else:
            spectral_entropy[p] = 0.0

        mean_off_diag[p] = float(affinity.sum() / (D * (D - 1)))

        # Effective rank from singular values of the centered (L, D) trajectory.
        sv = np.linalg.svd(X, compute_uv=False)
        sv2 = sv ** 2
        s2 = sv2.sum()
        if s2 > 0:
            ne2 = sv2 / s2
            effective_rank[p] = float(np.exp(
                -np.sum(ne2 * np.log(ne2 + 1e-30))
            ))
        else:
            effective_rank[p] = 0.0

        if (p + 1) % 8 == 0 or p == n_problems - 1:
            print(f"    problem {p+1}/{n_problems}: "
                  f"fiedler_lambda={fiedler_eigvals[p]:.3f}, "
                  f"spectral_entropy={spectral_entropy[p]:.3f}, "
                  f"mean_|R|={mean_off_diag[p]:.3f}, "
                  f"eff_rank={effective_rank[p]:.1f}")

    return {
        "abs_corr_matrices": abs_corr_matrices,
        "fiedler_vecs": fiedler_vecs,
        "fiedler_eigvals": fiedler_eigvals,
        "top_k_eigvecs": top_k_eigvecs,
        "top_k_eigvals": top_k_eigvals,
        "spectral_entropy": spectral_entropy,
        "mean_off_diag": mean_off_diag,
        "effective_rank": effective_rank,
        "n_problems": n_problems,
        "L": L,
        "D": D,
    }


# ---------------------------------------------------------------------------
# Cross-problem similarities
# ---------------------------------------------------------------------------

def eigenspace_overlap(V1, V2):
    """Mean cosine of principal angles between two orthonormal subspaces."""
    M = V1.T @ V2
    cosines = np.linalg.svd(M, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    return float(np.mean(cosines))


def compute_similarities(per_pp):
    fiedler = per_pp["fiedler_vecs"]               # (n, D)
    top_k = per_pp["top_k_eigvecs"]                 # (n, D, K)
    affinity = per_pp["abs_corr_matrices"]          # (n, D, D)
    n = fiedler.shape[0]

    f_norm = fiedler / (np.linalg.norm(fiedler, axis=1, keepdims=True) + 1e-30)
    fiedler_sim = np.abs(f_norm @ f_norm.T)

    eig_sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        eig_sim[i, i] = 1.0
        for j in range(i + 1, n):
            v = eigenspace_overlap(top_k[i], top_k[j])
            eig_sim[i, j] = v
            eig_sim[j, i] = v

    aff_flat = affinity.reshape(n, -1)
    aff_norms = np.linalg.norm(aff_flat, axis=1) + 1e-30
    corr_sim = (aff_flat @ aff_flat.T) / np.outer(aff_norms, aff_norms)
    corr_sim = corr_sim.astype(np.float32)

    return {
        "fiedler_abs_cos": fiedler_sim,
        "eigenspace_topk_overlap": eig_sim,
        "corr_frobenius_cos": corr_sim,
    }


def within_between_split(sim, group_labels):
    """Return (within_add, within_mul, between) similarity arrays.

    within_*: upper-triangle pairs from a single group (excludes diagonal).
    between:  all (add, mul) pairs.
    """
    labels = np.asarray(group_labels)
    add_idx = np.where(labels == "add")[0]
    mul_idx = np.where(labels == "mul")[0]

    if len(add_idx) >= 2:
        iu = np.triu_indices(len(add_idx), k=1)
        within_add = sim[np.ix_(add_idx, add_idx)][iu]
    else:
        within_add = np.array([], dtype=sim.dtype)

    if len(mul_idx) >= 2:
        iu = np.triu_indices(len(mul_idx), k=1)
        within_mul = sim[np.ix_(mul_idx, mul_idx)][iu]
    else:
        within_mul = np.array([], dtype=sim.dtype)

    between = sim[np.ix_(add_idx, mul_idx)].ravel()
    return within_add, within_mul, between


# ---------------------------------------------------------------------------
# Sign alignment for Fiedler visualization
# ---------------------------------------------------------------------------

def align_fiedler_signs_within_groups(fiedler_vecs, group_labels):
    """Greedy sign alignment for visualization, restarted at each group
    boundary so we don't impose continuity across operations (where it
    would be misleading). Sign flips do NOT carry meaning; the |cos|
    similarity matrix is the unbiased place to compare partitions.
    """
    aligned = fiedler_vecs.copy()
    labels = np.asarray(group_labels)
    for p in range(1, len(aligned)):
        if labels[p] != labels[p - 1]:
            continue  # do not align across operation boundary
        if np.dot(aligned[p], aligned[p - 1]) < 0:
            aligned[p] = -aligned[p]
    return aligned


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _set_problem_xticks(ax, labels, fontsize=7):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)


def _draw_group_separator(ax, n_first_group, axis="both", **kwargs):
    style = dict(color="white", linewidth=1.0, alpha=0.7)
    style.update(kwargs)
    if axis in ("x", "both"):
        ax.axvline(n_first_group - 0.5, **style)
    if axis in ("y", "both"):
        ax.axhline(n_first_group - 0.5, **style)


def plot_similarity_matrix(sim, labels, n_first_group, title, out_path,
                           vmin=None, vmax=None, cmap="viridis"):
    n = len(labels)
    side = max(10, n * 0.22)
    fig, ax = plt.subplots(figsize=(side + 1, side))
    if vmin is None:
        vmin = float(np.nanmin(sim))
    if vmax is None:
        vmax = float(np.nanmax(sim))
    im = ax.imshow(sim, aspect="equal", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    _set_problem_xticks(ax, labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    _draw_group_separator(ax, n_first_group)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_within_between_summary(sims, group_labels, out_path):
    """One panel per similarity measure. Each panel: violin of within-add,
    within-mul, between, with means annotated."""
    measures = [
        ("fiedler_abs_cos", "|cos(Fiedler vectors)|"),
        ("eigenspace_topk_overlap", "Top-k eigenspace overlap"),
        ("corr_frobenius_cos", "|R| Frobenius cosine"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (key, title) in zip(axes, measures):
        wa, wm, bt = within_between_split(sims[key], group_labels)
        data = [wa, wm, bt]
        positions = [1, 2, 3]
        cat_labels = ["within-add", "within-mul", "between"]
        colors = ["C0", "C1", "C7"]

        # Only violin-plot non-empty distributions; empty ones get a marker only.
        viol_data = [d for d in data if len(d) > 0]
        viol_pos = [p for p, d in zip(positions, data) if len(d) > 0]
        if len(viol_data) > 0:
            parts = ax.violinplot(viol_data, positions=viol_pos,
                                  showmeans=False, showmedians=True,
                                  widths=0.7)
            for pc, idx in zip(parts["bodies"], viol_pos):
                pc.set_facecolor(colors[idx - 1])
                pc.set_edgecolor("black")
                pc.set_alpha(0.55)

        # Jittered points for transparency.
        rng_local = np.random.default_rng(0)
        for d, pos, c in zip(data, positions, colors):
            if len(d) == 0:
                continue
            jitter = rng_local.uniform(-0.10, 0.10, size=len(d))
            ax.scatter(pos + jitter, d, color=c, s=8, alpha=0.6,
                       edgecolor="black", linewidth=0.3, zorder=3)

        # Mean lines.
        for d, pos in zip(data, positions):
            if len(d) == 0:
                continue
            ax.hlines(np.mean(d), pos - 0.30, pos + 0.30,
                      colors="black", linewidth=1.5, zorder=4)

        ax.set_xticks(positions)
        ax.set_xticklabels(cat_labels)
        ax.set_ylabel("Similarity")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")

        means = [float(np.mean(d)) if len(d) > 0 else float("nan") for d in data]
        ax.text(
            0.02, 0.98,
            (f"means: {means[0]:.3f} / {means[1]:.3f} / {means[2]:.3f}\n"
             f"within-add  vs between: {means[0] - means[2]:+.3f}\n"
             f"within-mul  vs between: {means[1] - means[2]:+.3f}"),
            transform=ax.transAxes, fontsize=8, va="top",
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    fig.suptitle("Within-operation vs between-operation coupling similarity",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_problem_diagnostics(per_pp, group_labels, problem_labels, out_path):
    """Effective rank / mean |R| / spectral entropy per problem, by op."""
    labels = np.asarray(group_labels)
    n = len(labels)
    add_idx = np.where(labels == "add")[0]
    mul_idx = np.where(labels == "mul")[0]

    fig, axes = plt.subplots(3, 1, figsize=(max(12, n * 0.30), 10), sharex=True)

    metrics = [
        ("effective_rank", "Effective rank"),
        ("mean_off_diag", r"Mean off-diagonal $|R|$"),
        ("spectral_entropy", "Spectral entropy"),
    ]

    for ax, (key, ylabel) in zip(axes, metrics):
        vals = per_pp[key]
        ax.scatter(add_idx, vals[add_idx], c="C0", label="addition",
                   s=30, edgecolor="black", linewidth=0.3, zorder=3)
        ax.scatter(mul_idx, vals[mul_idx], c="C1", label="multiplication",
                   s=30, edgecolor="black", linewidth=0.3, zorder=3)
        if len(add_idx) > 0:
            ax.axhline(np.mean(vals[add_idx]), color="C0", linestyle="--",
                       alpha=0.5,
                       label=f"mean add = {np.mean(vals[add_idx]):.3f}")
        if len(mul_idx) > 0:
            ax.axhline(np.mean(vals[mul_idx]), color="C1", linestyle="--",
                       alpha=0.5,
                       label=f"mean mul = {np.mean(vals[mul_idx]):.3f}")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    _set_problem_xticks(axes[-1], problem_labels)
    axes[-1].set_xlabel("Problem")
    axes[0].set_title(
        "Per-problem coupling diagnostics, addition vs multiplication"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_fiedler_heatmap(per_pp, problem_labels, group_labels, n_first_group,
                          out_path, top_units=100):
    fiedler = align_fiedler_signs_within_groups(
        per_pp["fiedler_vecs"], group_labels
    )
    n_problems, D = fiedler.shape

    unit_strength = np.mean(np.abs(fiedler), axis=0)
    n_keep = int(min(top_units, D))
    top_idx = np.argsort(unit_strength)[-n_keep:]
    fiedler_top = fiedler[:, top_idx]

    distances = pdist(fiedler_top.T, metric="cosine")
    distances = np.nan_to_num(distances, nan=1.0)
    linkage_matrix = linkage(distances, method="average")
    unit_order = leaves_list(linkage_matrix)
    fiedler_ordered = fiedler_top.T[unit_order]

    width = max(12, n_problems * 0.30)
    fig, ax = plt.subplots(figsize=(width, 8))
    vmax = float(np.abs(fiedler_top).max())
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(
        fiedler_ordered, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    _set_problem_xticks(ax, problem_labels)
    ax.set_xlabel("Problem (left: addition, right: multiplication)")
    ax.set_ylabel(f"Unit (top {n_keep} of {D} by mean |Fiedler|, clustered)")
    ax.set_title(
        f"Fiedler vector (|R| Laplacian) across problems: top {n_keep} "
        f"most participating units, sign-aligned within each group"
    )
    _draw_group_separator(ax, n_first_group, axis="x",
                          color="black", linewidth=1.5, alpha=1.0)
    plt.colorbar(im, ax=ax, label="Fiedler value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-per-op", type=int, default=24,
                        help="Number of problems per operation.")
    parser.add_argument("--operand-low", type=int, default=10)
    parser.add_argument("--operand-high", type=int, default=99)
    parser.add_argument("--add-symbol", default="+")
    parser.add_argument("--mul-symbol", default="*")
    parser.add_argument("--trim-sublayers", type=int, default=2)
    parser.add_argument("--top-k-eigenvecs", type=int, default=10)
    parser.add_argument("--top-fiedler-units", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.n_per_op < 2:
        raise SystemExit("--n-per-op must be at least 2 to form within-op pairs.")

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(f"math_couplings_{slugify(args.model)}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:        {args.model}")
    print(f"n_per_op:     {args.n_per_op}")
    print(f"Operands:     [{args.operand_low}, {args.operand_high}]")
    print(f"Symbols:      add='{args.add_symbol}', mul='{args.mul_symbol}'")
    print(f"Out:          {out_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rng = np.random.default_rng(args.seed)
    a = rng.integers(args.operand_low, args.operand_high + 1, size=args.n_per_op)
    b = rng.integers(args.operand_low, args.operand_high + 1, size=args.n_per_op)
    operands = list(zip(a.tolist(), b.tolist()))
    add_problems = [f"{ai} {args.add_symbol} {bi} =" for ai, bi in operands]
    mul_problems = [f"{ai} {args.mul_symbol} {bi} =" for ai, bi in operands]

    print("\nExample problems:")
    print(f"  addition:        {add_problems[:3]}")
    print(f"  multiplication:  {mul_problems[:3]}")

    print("\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(DEVICE)

    print("\nCapturing residual streams (last token, addition) ...")
    add_streams, add_last_toks = capture_problems(model, tokenizer, add_problems)
    print("\nCapturing residual streams (last token, multiplication) ...")
    mul_streams, mul_last_toks = capture_problems(model, tokenizer, mul_problems)

    streams = np.concatenate([add_streams, mul_streams], axis=0)
    group_labels = ["add"] * args.n_per_op + ["mul"] * args.n_per_op
    problem_strings = add_problems + mul_problems
    problem_labels = (
        [f"{ai}+{bi}" for ai, bi in operands]
        + [f"{ai}*{bi}" for ai, bi in operands]
    )
    last_tokens = add_last_toks + mul_last_toks
    n_problems = streams.shape[0]
    print(f"\nStreams shape (n_problems, n_sub, d_model) = {streams.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nComputing per-problem correlation coupling "
          f"(trim={args.trim_sublayers}) ...")
    per_pp = per_problem_correlation_coupling(
        streams, trim=args.trim_sublayers, k_eigvecs=args.top_k_eigenvecs,
    )
    print(f"  L (after trim) = {per_pp['L']}")
    print(f"  D = {per_pp['D']}")

    print("\nComputing cross-problem similarities ...")
    sims = compute_similarities(per_pp)
    wb_means = {}
    print("  within / between summary (mean similarity):")
    for name in ("fiedler_abs_cos", "eigenspace_topk_overlap",
                 "corr_frobenius_cos"):
        wa, wm, bt = within_between_split(sims[name], group_labels)
        wb_means[name] = {
            "within_add": float(np.mean(wa)) if len(wa) else float("nan"),
            "within_mul": float(np.mean(wm)) if len(wm) else float("nan"),
            "between":    float(np.mean(bt)) if len(bt) else float("nan"),
        }
        print(f"    {name:<28} "
              f"within-add={wb_means[name]['within_add']:.3f}  "
              f"within-mul={wb_means[name]['within_mul']:.3f}  "
              f"between={wb_means[name]['between']:.3f}")

    print("\nMaking plots ...")
    plot_similarity_matrix(
        sims["fiedler_abs_cos"], problem_labels, args.n_per_op,
        "Problem x problem: |cos(Fiedler vectors)|",
        out_dir / "similarity_fiedler.png",
        vmin=0.0, vmax=1.0,
    )
    print("  similarity_fiedler.png")

    plot_similarity_matrix(
        sims["eigenspace_topk_overlap"], problem_labels, args.n_per_op,
        f"Problem x problem: top-{args.top_k_eigenvecs} Laplacian "
        f"eigenspace overlap (mean principal-angle cosine)",
        out_dir / "similarity_eigenspace.png",
        vmin=0.0, vmax=1.0,
    )
    print("  similarity_eigenspace.png")

    plot_similarity_matrix(
        sims["corr_frobenius_cos"], problem_labels, args.n_per_op,
        r"Problem x problem: $|R|$ matrix Frobenius cosine similarity",
        out_dir / "similarity_corr.png",
        vmin=float(np.nanmin(sims["corr_frobenius_cos"])),
        vmax=1.0,
    )
    print("  similarity_corr.png")

    plot_within_between_summary(
        sims, group_labels, out_dir / "within_between_summary.png",
    )
    print("  within_between_summary.png")

    plot_per_problem_diagnostics(
        per_pp, group_labels, problem_labels,
        out_dir / "per_problem_diagnostics.png",
    )
    print("  per_problem_diagnostics.png")

    plot_fiedler_heatmap(
        per_pp, problem_labels, group_labels, args.n_per_op,
        out_dir / "fiedler_heatmap.png",
        top_units=args.top_fiedler_units,
    )
    print("  fiedler_heatmap.png")

    summary = {
        "model": args.model,
        "n_per_op": args.n_per_op,
        "operand_range": [args.operand_low, args.operand_high],
        "symbols": {"add": args.add_symbol, "mul": args.mul_symbol},
        "matched_operands": True,
        "n_sub_used": int(per_pp["L"]),
        "d_model": int(per_pp["D"]),
        "top_k_eigvecs": int(args.top_k_eigenvecs),
        "problem_strings": problem_strings,
        "problem_labels": problem_labels,
        "group_labels": group_labels,
        "last_tokens_seen_by_model": last_tokens,
        "per_problem": {
            "fiedler_lambda":      per_pp["fiedler_eigvals"].tolist(),
            "top_k_eigvals":       per_pp["top_k_eigvals"].tolist(),
            "spectral_entropy":    per_pp["spectral_entropy"].tolist(),
            "mean_off_diag_abs_R": per_pp["mean_off_diag"].tolist(),
            "effective_rank":      per_pp["effective_rank"].tolist(),
        },
        "within_between_means": wb_means,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  summary.json")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
