"""
Per-position functional coupling structure during autoregressive generation.

Pipeline:
  1. Tokenize a short seed prompt.
  2. Generate N tokens autoregressively via sampling.
  3. Run a single forward pass on the full sequence (prompt + generated)
     and capture the residual stream at every (token_position, sublayer).
  4. By default, restrict analysis to the generated positions only;
     pass --include-prompt to keep the prompt positions too.

Why generate-then-prefill is correct:
  In a causal transformer, the residual stream at position i during a
  single prefill pass on a sequence S[0..K] is computed from S[0..i] and
  is therefore identical to the residual stream at position i during the
  autoregressive step that produced token S[i+1]. We get the residual
  streams of the model "deciding what comes next" at every generated
  position, exactly the autoregressive regime, in one prefill pass.

Why we use cross-correlation rather than PLV:
  The natural addition-toy analog (PLV across an input axis at fixed
  position) is unavailable here because we have one generation at a
  time, so per-position coupling has to be derived from the depth axis
  alone. PLV across the depth axis is uninformative in this regime:
  every unit oscillates at approximately the shared depth-axis carrier
  with a roughly constant phase offset (this is exactly what the
  (x, Delta x) quadrature result tells us), so |mean_l exp(i*(phi_li -
  phi_lj))| is approximately 1 for every (i, j). The PLV matrix
  collapses to near-all-ones, the Laplacian D*I - J becomes degenerate,
  and the Fiedler vector eigh() returns is essentially arbitrary.

  Depth-axis cross-correlation R[i, j] = corr(x[:, i], x[:, j]) is the
  standard functional-connectivity measure and remains informative when
  phases are locked. Two units sharing the carrier with different
  amplitude profiles get R < 1; functionally co-driven units get high
  R; independent units get low R. We use |R| as the affinity and the
  combinatorial Laplacian L = D_diag - |R| for spectral analysis.

Per-position diagnostics:
  R matrix:          D x D depth-axis cross-correlation among units.
                     Symmetric, diagonal = 1, off-diagonal in [-1, 1].
  |R| affinity:      |R| with diagonal zeroed; used for the Laplacian.
  Laplacian:         L = D_diag - |R|.
  Fiedler vector:    eigenvector of the second-smallest eigenvalue.
                     Encodes the dominant bipartition of units.
  Top-k eigenspace:  eigenvectors of the next k smallest eigenvalues.
                     Encodes higher-order partition structure.
  Spectral entropy:  -sum p_i log p_i over normalized non-trivial
                     eigenvalues. High = uniform spectrum.
  Mean |R|:          mean off-diagonal |R|. Overall coupling strength.
  Effective rank:    exp(-sum p_i log p_i) over normalized squared
                     singular values of the centered (L, D) trajectory.
                     The effective number of independent directions in
                     unit-space at this position. Low = position is
                     dominated by a few directions.

Cross-position diagnostics:
  Fiedler |cos|:        |cos| between Fiedler vectors. Sign-invariant.
  Top-k eigenspace:     mean principal-angle cosine between top-k
                        Laplacian eigenspaces.
  R Frobenius:          cosine similarity of flattened |R| matrices.

Sign convention for Fiedler heatmap:
  Fiedler vectors are sign-arbitrary. For visualization, we align each
  position's Fiedler to the previous position's by max correlation.
  Visual continuity along the position axis is enforced and should not
  be over-read; the |cos| similarity matrix is the unbiased place to
  look for partition stability.

Outputs in phase_couplings_<model_slug>/:
  fiedler_heatmap.png             Fiedler vector per position, top-N
                                  most participating units.
  participation_diagnostic.png    IPR per position + cumulative
                                  participation curve.
  spectrum_trajectory.png         Top-5 smallest non-trivial eigenvalues,
                                  spectral entropy, mean |R|, effective rank.
  similarity_fiedler.png          Position x position |cos(Fiedler)|.
  similarity_eigenspace.png       Position x position top-k eigenspace.
  similarity_corr.png             Position x position |R| Frobenius cosine.
  summary.json                    Per-position scalar summaries + tokens.

Usage:
  python phase_couplings_lm.py
  python phase_couplings_lm.py --prompt "Once upon a time" --n-generate 80
  python phase_couplings_lm.py --include-prompt --temperature 0.7
  python phase_couplings_lm.py --n-generate 0 --include-prompt   # prompt-only
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
DEFAULT_PROMPT = "Once upon a time,"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


# ---------------------------------------------------------------------------
# Stream collection
# ---------------------------------------------------------------------------

def get_hook_targets(model):
    """Locate sublayer-input hooks. We use LayerNorm-input hooks for
    architectures with input_layernorm + post_attention_layernorm
    (Llama/SmolLM/Pythia/GPT-NeoX), and ln_1/ln_2 for GPT-2."""
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


def _capture_streams_for_ids(model, input_ids):
    """Single forward pass + hooks; returns (seq_len, n_sub, d_model)."""
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
    streams = torch.stack([c[0] for c in captures], dim=0).float().cpu().numpy()
    return np.transpose(streams, (1, 0, 2))


def generate_then_capture(
    model, tokenizer, prompt, n_generate, temperature, top_p, seq_len_max,
):
    """Generate n_generate tokens autoregressively from prompt, then do a
    single forward pass on the full (prompt + generated) sequence to
    capture residual streams.

    Returns:
        streams:    (full_seq_len, n_sub, d_model) array
        tokens:     list of decoded token strings
        prompt_len: number of prompt tokens
    """
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt", truncation=False)
    prompt_ids = enc["input_ids"].to(DEVICE)
    prompt_len = int(prompt_ids.shape[1])

    if n_generate > 0:
        with torch.no_grad():
            full_ids = model.generate(
                prompt_ids,
                max_new_tokens=n_generate,
                do_sample=temperature > 0,
                temperature=max(float(temperature), 1e-5),
                top_p=float(top_p),
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
    else:
        full_ids = prompt_ids

    if full_ids.shape[1] > seq_len_max:
        full_ids = full_ids[:, :seq_len_max]

    streams = _capture_streams_for_ids(model, full_ids)
    tokens = [tokenizer.decode([t.item()]) for t in full_ids[0]]
    return streams, tokens, prompt_len


# ---------------------------------------------------------------------------
# Per-position correlation coupling
# ---------------------------------------------------------------------------

def per_position_correlation_coupling(streams, trim=2, k_eigvecs=10):
    """For each position, compute depth-axis cross-correlation among
    units, build the |R| Laplacian, and extract spectral diagnostics.

    Parameters
    ----------
    streams : (seq_len, n_sub, d_model) float
    trim : int
        Drop this many sublayers from each end of the depth axis.
    k_eigvecs : int
        Number of low Laplacian eigenvectors to retain per position.

    Returns dict with arrays of length seq_len plus the full |R|
    matrices for direct comparison.
    """
    if trim > 0:
        if streams.shape[1] <= 2 * trim + 2:
            raise ValueError(
                f"Cannot trim {trim} from each end of {streams.shape[1]} sublayers."
            )
        streams = streams[:, trim:streams.shape[1] - trim, :]
    seq_len, L, D = streams.shape

    # Center each (position, unit) trajectory along the sublayer axis.
    streams_c = streams - streams.mean(axis=1, keepdims=True)

    # Storage
    abs_corr_matrices = np.zeros((seq_len, D, D), dtype=np.float32)
    fiedler_vecs = np.zeros((seq_len, D), dtype=np.float32)
    fiedler_eigvals = np.zeros(seq_len, dtype=np.float32)
    top_k_eigvecs = np.zeros((seq_len, D, k_eigvecs), dtype=np.float32)
    top_k_eigvals = np.zeros((seq_len, k_eigvecs), dtype=np.float32)
    spectral_entropy = np.zeros(seq_len, dtype=np.float32)
    mean_off_diag = np.zeros(seq_len, dtype=np.float32)
    effective_rank = np.zeros(seq_len, dtype=np.float32)

    for p in range(seq_len):
        X = streams_c[p]                              # (L, D)
        # Per-unit norms along the depth axis. Add a floor for stability.
        norms = np.linalg.norm(X, axis=0)             # (D,)
        norms = np.maximum(norms, 1e-12)
        # Cross-correlation matrix R[i, j] = (X[:, i] . X[:, j]) /
        # (||X[:, i]|| * ||X[:, j]||).
        R = (X.T @ X) / np.outer(norms, norms)        # (D, D)
        affinity = np.abs(R).astype(np.float64)
        np.fill_diagonal(affinity, 0.0)
        abs_corr_matrices[p] = affinity.astype(np.float32)

        degree = affinity.sum(axis=1)
        laplacian = np.diag(degree) - affinity

        # Symmetric eigendecomposition; eigvals ascending.
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

        # Mean off-diagonal |R| (diagonal already zeroed).
        mean_off_diag[p] = float(affinity.sum() / (D * (D - 1)))

        # Effective rank from singular values of the centered (L, D)
        # trajectory.
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

        if (p + 1) % 16 == 0 or p == seq_len - 1:
            print(f"    position {p+1}/{seq_len}: "
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
        "seq_len": seq_len,
        "L": L,
        "D": D,
    }


# ---------------------------------------------------------------------------
# Cross-position similarities
# ---------------------------------------------------------------------------

def eigenspace_overlap(V1, V2):
    """Mean cosine of principal angles between two orthonormal subspaces.
    V1, V2: (D, k) matrices with orthonormal columns."""
    M = V1.T @ V2
    cosines = np.linalg.svd(M, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    return float(np.mean(cosines))


def compute_similarities(per_pos):
    fiedler = per_pos["fiedler_vecs"]              # (seq_len, D)
    top_k = per_pos["top_k_eigvecs"]                # (seq_len, D, K)
    affinity = per_pos["abs_corr_matrices"]         # (seq_len, D, D)
    seq_len = fiedler.shape[0]

    f_norm = fiedler / (np.linalg.norm(fiedler, axis=1, keepdims=True) + 1e-30)
    fiedler_sim = np.abs(f_norm @ f_norm.T)

    eig_sim = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        eig_sim[i, i] = 1.0
        for j in range(i + 1, seq_len):
            v = eigenspace_overlap(top_k[i], top_k[j])
            eig_sim[i, j] = v
            eig_sim[j, i] = v

    aff_flat = affinity.reshape(seq_len, -1)
    aff_norms = np.linalg.norm(aff_flat, axis=1) + 1e-30
    corr_sim = (aff_flat @ aff_flat.T) / np.outer(aff_norms, aff_norms)
    corr_sim = corr_sim.astype(np.float32)

    return {
        "fiedler_abs_cos": fiedler_sim,
        "eigenspace_topk_overlap": eig_sim,
        "corr_frobenius_cos": corr_sim,
    }


# ---------------------------------------------------------------------------
# Sign alignment for Fiedler visualization
# ---------------------------------------------------------------------------

def align_fiedler_signs(fiedler_vecs):
    """Greedy sign alignment for visualization. Sign flips do NOT carry
    meaning under this convention; they are imposed for visual continuity."""
    aligned = fiedler_vecs.copy()
    for p in range(1, len(aligned)):
        if np.dot(aligned[p], aligned[p - 1]) < 0:
            aligned[p] = -aligned[p]
    return aligned


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _set_token_xticks(ax, tokens, fontsize=7):
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(
        [t.replace("\n", "\\n") for t in tokens],
        rotation=90, fontsize=fontsize,
    )


def plot_fiedler_heatmap(per_pos, tokens, out_path, top_units=100):
    fiedler = align_fiedler_signs(per_pos["fiedler_vecs"])
    seq_len, D = fiedler.shape

    unit_strength = np.mean(np.abs(fiedler), axis=0)
    n_keep = int(min(top_units, D))
    top_idx = np.argsort(unit_strength)[-n_keep:]
    fiedler_top = fiedler[:, top_idx]

    distances = pdist(fiedler_top.T, metric="cosine")
    distances = np.nan_to_num(distances, nan=1.0)
    linkage_matrix = linkage(distances, method="average")
    unit_order = leaves_list(linkage_matrix)
    fiedler_ordered = fiedler_top.T[unit_order]

    kept_strength = float(np.median(unit_strength[top_idx]))
    if n_keep < D:
        dropped_idx = np.argsort(unit_strength)[:D - n_keep]
        dropped_strength = float(np.median(unit_strength[dropped_idx]))
    else:
        dropped_strength = float("nan")

    width = max(12, len(tokens) * 0.30)
    fig, ax = plt.subplots(figsize=(width, 8))
    vmax = float(np.abs(fiedler_top).max())
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(
        fiedler_ordered, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    _set_token_xticks(ax, tokens)
    ax.set_xlabel("Token position")
    ax.set_ylabel(f"Unit (top {n_keep} of {D} by mean |Fiedler|, clustered)")
    ax.set_title(
        f"Fiedler vector (|R| Laplacian) across positions: "
        f"top {n_keep} most participating units, sign-aligned\n"
        f"median |Fiedler| of kept units = {kept_strength:.4f}, "
        f"of dropped units = {dropped_strength:.4f}"
    )
    plt.colorbar(im, ax=ax, label="Fiedler value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_participation_diagnostic(per_pos, tokens, out_path):
    fiedler = per_pos["fiedler_vecs"]
    seq_len, D = fiedler.shape
    norms = np.linalg.norm(fiedler, axis=1, keepdims=True)
    f_unit = fiedler / (norms + 1e-30)

    sum_v4 = np.sum(f_unit ** 4, axis=1)
    ipr = 1.0 / (sum_v4 + 1e-30)

    sq_sorted = -np.sort(-(f_unit ** 2), axis=1)
    cum = np.cumsum(sq_sorted, axis=1)
    cum_mean = cum.mean(axis=0)

    width = max(12, len(tokens) * 0.30)
    fig, axes = plt.subplots(2, 1, figsize=(width, 8),
                              gridspec_kw={"height_ratios": [1, 1]})

    ax = axes[0]
    ax.plot(ipr, "o-", color="C4", markersize=3, linewidth=1.0,
            label="IPR")
    ax.axhline(D, color="black", linestyle=":", alpha=0.4,
               label=f"D = {D} (fully delocalized)")
    ax.set_xlabel("Token position")
    ax.set_ylabel("IPR")
    ax.set_title("Fiedler IPR per position")
    _set_token_xticks(ax, tokens)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    # Auto-scale rather than forcing y >= 0; if IPR is small relative to
    # D, autoscale lets it actually be visible.

    ax = axes[1]
    n_show = min(D, 400)
    ax.plot(np.arange(1, n_show + 1), cum_mean[:n_show],
            color="C2", linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6,
               label="50%")
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.6,
               label="90%")
    half_idx = int(np.searchsorted(cum_mean, 0.5)) + 1
    ninety_idx = int(np.searchsorted(cum_mean, 0.9)) + 1
    ax.axvline(half_idx, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(ninety_idx, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Number of top-ranked units (sorted descending by $|v_i|^2$)")
    ax.set_ylabel("Cumulative squared Fiedler norm")
    ax.set_title(
        f"Cumulative participation (mean over positions): "
        f"50% at unit {half_idx}, 90% at unit {ninety_idx}"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, n_show)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum_trajectory(per_pos, tokens, out_path):
    width = max(12, len(tokens) * 0.30)
    fig, axes = plt.subplots(4, 1, figsize=(width, 14), sharex=True)

    ax = axes[0]
    eigvals = per_pos["top_k_eigvals"]
    n_to_plot = min(5, eigvals.shape[1])
    cmap = plt.cm.viridis
    for i in range(n_to_plot):
        color = cmap(i / max(n_to_plot - 1, 1))
        ax.plot(eigvals[:, i], "o-", color=color,
                markersize=3, linewidth=1.0,
                label=fr"$\lambda_{{{i+2}}}$")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Smallest non-trivial Laplacian eigenvalues by position")
    ax.legend(fontsize=8, ncol=n_to_plot, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(per_pos["spectral_entropy"], "o-", color="C2",
            markersize=3, linewidth=1.0)
    ax.set_ylabel("Spectral entropy")
    ax.set_title("Laplacian spectral entropy")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(per_pos["mean_off_diag"], "o-", color="C3",
            markersize=3, linewidth=1.0)
    ax.set_ylabel(r"Mean $|R|$")
    ax.set_title(r"Mean off-diagonal $|R|$ (overall coupling strength)")
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(per_pos["effective_rank"], "o-", color="C5",
            markersize=3, linewidth=1.0)
    ax.set_ylabel("Effective rank")
    ax.set_xlabel("Token position")
    ax.set_title(
        "Effective rank of (L, D) trajectory "
        "(low = position uses few independent directions)"
    )
    _set_token_xticks(ax, tokens)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_matrix(sim, tokens, title, out_path,
                           vmin=None, vmax=None, cmap="viridis"):
    seq_len = len(tokens)
    side = max(10, seq_len * 0.22)
    fig, ax = plt.subplots(figsize=(side + 1, side))
    if vmin is None:
        vmin = float(np.nanmin(sim))
    if vmax is None:
        vmax = float(np.nanmax(sim))
    im = ax.imshow(sim, aspect="equal", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    _set_token_xticks(ax, tokens)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(
        [t.replace("\n", "\\n") for t in tokens],
        fontsize=7,
    )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="Seed prompt for generation.")
    parser.add_argument("--n-generate", type=int, default=80,
                        help="Number of tokens to generate autoregressively. "
                             "Set to 0 to skip (use with --include-prompt to "
                             "analyze the prompt only).")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--include-prompt", action="store_true",
                        help="Include prompt positions in analysis. "
                             "Default: analyze only generated positions.")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--trim-sublayers", type=int, default=2)
    parser.add_argument("--top-k-eigenvecs", type=int, default=10)
    parser.add_argument("--top-fiedler-units", type=int, default=100)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(f"phase_couplings_{slugify(args.model)}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:       {args.model}")
    print(f"Prompt:      {args.prompt!r}")
    print(f"n_generate:  {args.n_generate}  (temp={args.temperature}, "
          f"top_p={args.top_p})")
    print(f"Mode:        {'full sequence' if args.include_prompt else 'generated only'}")
    print(f"Out:         {out_dir}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(DEVICE)

    print("\nGenerate + capture residual streams ...")
    streams, tokens, prompt_len = generate_then_capture(
        model, tokenizer, args.prompt, args.n_generate,
        args.temperature, args.top_p, args.max_len,
    )
    print(f"  full sequence: {len(tokens)} tokens "
          f"(prompt={prompt_len}, generated={len(tokens) - prompt_len})")
    print(f"  prompt tokens:    {tokens[:prompt_len]}")
    print(f"  generated tokens: {tokens[prompt_len:]}")
    print(f"  streams shape (seq_len, n_sub, d_model) = {streams.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.include_prompt:
        if args.n_generate <= 0:
            raise SystemExit(
                "n_generate=0 with --include-prompt off would leave no "
                "positions to analyze."
            )
        streams = streams[prompt_len:]
        tokens = tokens[prompt_len:]
        print(f"  Restricted to {len(tokens)} generated positions.")

    print(f"\nComputing per-position correlation coupling "
          f"(trim={args.trim_sublayers}) ...")
    per_pos = per_position_correlation_coupling(
        streams, trim=args.trim_sublayers, k_eigvecs=args.top_k_eigenvecs,
    )
    print(f"  L (after trim) = {per_pos['L']}")
    print(f"  D = {per_pos['D']}")

    print("\nComputing cross-position similarities ...")
    sims = compute_similarities(per_pos)
    for name in ("fiedler_abs_cos", "eigenspace_topk_overlap",
                  "corr_frobenius_cos"):
        triu = sims[name][np.triu_indices_from(sims[name], 1)]
        print(f"  {name:<28} median off-diag = {np.median(triu):.3f}")

    print("\nMaking plots ...")
    plot_fiedler_heatmap(
        per_pos, tokens, out_dir / "fiedler_heatmap.png",
        top_units=args.top_fiedler_units,
    )
    print("  fiedler_heatmap.png")

    plot_participation_diagnostic(
        per_pos, tokens, out_dir / "participation_diagnostic.png",
    )
    print("  participation_diagnostic.png")

    plot_spectrum_trajectory(
        per_pos, tokens, out_dir / "spectrum_trajectory.png",
    )
    print("  spectrum_trajectory.png")

    plot_similarity_matrix(
        sims["fiedler_abs_cos"], tokens,
        "Position x position: |cos(Fiedler vectors)|",
        out_dir / "similarity_fiedler.png",
        vmin=0.0, vmax=1.0,
    )
    print("  similarity_fiedler.png")

    plot_similarity_matrix(
        sims["eigenspace_topk_overlap"], tokens,
        f"Position x position: top-{args.top_k_eigenvecs} Laplacian "
        f"eigenspace overlap (mean principal-angle cosine)",
        out_dir / "similarity_eigenspace.png",
        vmin=0.0, vmax=1.0,
    )
    print("  similarity_eigenspace.png")

    plot_similarity_matrix(
        sims["corr_frobenius_cos"], tokens,
        r"Position x position: $|R|$ matrix Frobenius cosine similarity",
        out_dir / "similarity_corr.png",
        vmin=float(np.nanmin(sims["corr_frobenius_cos"])),
        vmax=1.0,
    )
    print("  similarity_corr.png")

    summary = {
        "model": args.model,
        "prompt": args.prompt,
        "n_generate": args.n_generate,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "include_prompt_in_analysis": args.include_prompt,
        "n_sub_used": int(per_pos["L"]),
        "d_model": int(per_pos["D"]),
        "top_k_eigvecs": int(args.top_k_eigenvecs),
        "tokens_analyzed": tokens,
        "per_position": {
            "fiedler_lambda":      per_pos["fiedler_eigvals"].tolist(),
            "top_k_eigvals":       per_pos["top_k_eigvals"].tolist(),
            "spectral_entropy":    per_pos["spectral_entropy"].tolist(),
            "mean_off_diag_abs_R": per_pos["mean_off_diag"].tolist(),
            "effective_rank":      per_pos["effective_rank"].tolist(),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  summary.json")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
