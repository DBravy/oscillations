"""
Per-problem functional coupling structure for arithmetic problems (v2).

Adapts the residual-stream coupling analysis from phase_couplings_lm.py
to compare LM internal structure across math problems. v2 captures
residual streams at EVERY position (not just the last), giving two
complementary views of the coupling geometry:

  Per-position view
  -----------------
  For each token position separately, compute one coupling fingerprint
  per problem and a problem x problem similarity matrix. Then track how
  within-operation similarity and between-operation similarity evolve
  as the inference unfolds. This shows where in the forward pass the
  operator-specific signal actually lives.

  Whole-inference (aggregate) view
  ---------------------------------
  Per problem, stack the residual stream across (position, sublayer)
  into one extended trajectory, then compute a single |R| matrix per
  problem from this extended trajectory. This captures coupling that
  emerges over the whole inference rather than at one point.

By default, problems are matched: the same (a, b) operand pair is used
for both "a + b =" and "a * b =", so any systematic between-operation
difference is attributable to the operation rather than the operands.

Outputs in math_couplings_<model_slug>/:
  Per-position view:
    within_between_trajectory.png       Within-add / within-mul / between
                                        similarity vs token position, one
                                        panel per measure.
    similarity_corr_per_position.png    Row of |R| Frobenius cosine
                                        heatmaps, one per position.
    per_position_diagnostics.png        Effective rank, mean |R|,
                                        spectral entropy as a function of
                                        position, by operation.

  Whole-inference (aggregate) view:
    similarity_fiedler_aggregate.png
    similarity_eigenspace_aggregate.png
    similarity_corr_aggregate.png
    within_between_summary_aggregate.png
    per_problem_diagnostics_aggregate.png
    fiedler_heatmap_aggregate.png

  Top similar pairs:
    top_similar_pairs.md       Human-readable tables, annotated with the
                               model's predicted integer per problem and
                               whether it matches ground truth.
    top_similar_pairs.json     Same data as JSON.

  summary.json                 Per-position scalars + aggregate scalars
                               + within/between means at every position
                               + correctness records (predicted text,
                               parsed integer, ground truth, correct flag).

Usage:
  python math_couplings_lm.py
  python math_couplings_lm.py --n-per-op 32 --top-pairs 20
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

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
# Stream collection
# ---------------------------------------------------------------------------

def get_hook_targets(model):
    """LayerNorm-input hooks for Llama/SmolLM/Pythia/GPT-NeoX, ln_1/ln_2 for GPT-2."""
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


def _capture_full_streams(model, input_ids):
    """Single forward pass + hooks; return (seq_len, n_sub, d_model)."""
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
    return np.transpose(streams, (1, 0, 2))  # (seq_len, n_sub, d_model)


def parse_leading_int(text):
    """Extract an integer from generated text. Tries leading-numeric first,
    falls back to the first integer found anywhere. Tolerates thousands
    commas like '7,300'. Returns None if no integer is present.
    """
    s = text.lstrip()
    m = re.match(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)", s)
    if m is None:
        m = re.search(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)", s)
    if m is None:
        return None
    return int(m.group(0).replace(",", ""))


def evaluate_correctness(operands, operations, generated_texts):
    """Compute ground truth, parse the model's answer, mark correctness.

    Returns a list of per-problem dicts (one per problem, in the same order).
    """
    out = []
    for (a, b), op, gen in zip(operands, operations, generated_texts):
        truth = a + b if op == "add" else a * b
        pred = parse_leading_int(gen)
        out.append({
            "operands": [int(a), int(b)],
            "operation": op,
            "ground_truth": int(truth),
            "generated_text": gen,
            "predicted_int": pred,
            "correct": bool(pred == truth),
        })
    return out


def capture_full_problems(model, tokenizer, problems, n_generate=10, log_every=8):
    """Run forward+capture+greedy-generate per problem.

    For each problem:
      1. Forward pass with hooks on the prompt: capture residual streams
         at every position.
      2. Greedy generation (no hooks active) of n_generate tokens to
         obtain the model's actual answer for evaluation.

    Returns:
        streams:         (n_problems, seq_len, n_sub, d_model). Errors out
                         if problems tokenize to different sequence lengths.
        token_strs:      list of per-problem decoded prompt tokens.
        generated_texts: list of decoded generated text per problem (may
                         contain trailing junk past the answer; the parser
                         picks out the leading integer).
    """
    model.eval()
    out = []
    token_strs = []
    generated_texts = []
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    for k, problem in enumerate(problems):
        enc = tokenizer(problem, return_tensors="pt", truncation=False)
        ids = enc["input_ids"].to(DEVICE)
        prompt_len = int(ids.shape[1])

        # 1. Capture residuals via hooks (single forward pass on prompt).
        full = _capture_full_streams(model, ids)
        out.append(full)
        token_strs.append([tokenizer.decode([int(t)]) for t in ids[0]])

        # 2. Greedy generation of the answer (hooks no longer active).
        if n_generate > 0:
            with torch.no_grad():
                full_ids = model.generate(
                    ids,
                    max_new_tokens=n_generate,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            generated_texts.append(
                tokenizer.decode(full_ids[0, prompt_len:])
            )
        else:
            generated_texts.append("")

        if (k + 1) % log_every == 0 or k == len(problems) - 1:
            print(f"    {k+1}/{len(problems)}")

    lens = Counter(len(t) for t in token_strs)
    if len(lens) > 1:
        raise SystemExit(
            f"Problems tokenize to different sequence lengths: {dict(lens)}. "
            f"Position-aligned analysis requires uniform tokenization. "
            f"Try restricting --operand-low / --operand-high so all numbers "
            f"tokenize identically."
        )
    return np.stack(out, axis=0), token_strs, generated_texts


# ---------------------------------------------------------------------------
# Per-problem correlation coupling (works for either per-position or aggregate)
# ---------------------------------------------------------------------------

def per_problem_correlation_coupling(streams, trim=2, k_eigvecs=10, log_every=8):
    """For each problem (axis 0 of streams), compute depth-axis cross-correlation
    among d_model units, build the |R| Laplacian, extract spectral diagnostics.

    Used in two modes:
      - per-position: streams = (n_problems, n_sub, d_model)
      - aggregate:    streams = (n_problems, seq_len*n_sub, d_model) with trim=0.
    """
    if trim > 0:
        if streams.shape[1] <= 2 * trim + 2:
            raise ValueError(
                f"Cannot trim {trim} from each end of {streams.shape[1]} sublayers."
            )
        streams = streams[:, trim:streams.shape[1] - trim, :]
    n_problems, L, D = streams.shape

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
        X = streams_c[p]
        norms = np.linalg.norm(X, axis=0)
        norms = np.maximum(norms, 1e-12)
        R = (X.T @ X) / np.outer(norms, norms)
        affinity = np.abs(R).astype(np.float64)
        np.fill_diagonal(affinity, 0.0)
        abs_corr_matrices[p] = affinity.astype(np.float32)

        degree = affinity.sum(axis=1)
        laplacian = np.diag(degree) - affinity
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        fiedler_vecs[p] = eigvecs[:, 1]
        fiedler_eigvals[p] = eigvals[1]
        top_k_eigvals[p] = eigvals[1:k_eigvecs + 1]
        top_k_eigvecs[p] = eigvecs[:, 1:k_eigvecs + 1]

        nontriv = np.maximum(eigvals[1:], 0.0)
        s = nontriv.sum()
        if s > 0:
            ne = nontriv / s
            spectral_entropy[p] = float(-np.sum(ne * np.log(ne + 1e-30)))
        else:
            spectral_entropy[p] = 0.0
        mean_off_diag[p] = float(affinity.sum() / (D * (D - 1)))

        sv = np.linalg.svd(X, compute_uv=False)
        sv2 = sv ** 2
        s2 = sv2.sum()
        if s2 > 0:
            ne2 = sv2 / s2
            effective_rank[p] = float(np.exp(-np.sum(ne2 * np.log(ne2 + 1e-30))))
        else:
            effective_rank[p] = 0.0

        if log_every and ((p + 1) % log_every == 0 or p == n_problems - 1):
            print(f"      problem {p+1}/{n_problems}: "
                  f"fiedler_lambda={fiedler_eigvals[p]:.2f}, "
                  f"mean_|R|={mean_off_diag[p]:.3f}, "
                  f"eff_rank={effective_rank[p]:.2f}")

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


def per_position_full_analysis(streams, trim, k_eigvecs):
    """Per-(problem, position) coupling + per-position cross-problem similarities.

    Memory-conscious: |R| matrices are computed in-batch for one position,
    used for similarity computation, then discarded. Only compact diagnostics
    are retained across positions.

    Returns:
        per_pos_diag: list of per-position dicts (no abs_corr_matrices),
                      length seq_len.
        per_pos_sims: list of per-position similarity dicts, length seq_len.
    """
    n_problems, seq_len, n_sub, d_model = streams.shape
    per_pos_diag = []
    per_pos_sims = []
    for p in range(seq_len):
        print(f"  position {p+1}/{seq_len}")
        slice_streams = streams[:, p, :, :]
        per_pp = per_problem_correlation_coupling(
            slice_streams, trim=trim, k_eigvecs=k_eigvecs, log_every=0,
        )
        sims = compute_similarities(per_pp)
        per_pp_compact = {k: v for k, v in per_pp.items() if k != "abs_corr_matrices"}
        per_pos_diag.append(per_pp_compact)
        per_pos_sims.append(sims)
    return per_pos_diag, per_pos_sims


def whole_inference_coupling(streams, trim, k_eigvecs):
    """Aggregate fingerprint per problem.

    Stacks each problem's residual streams across (position, sublayer)
    into one extended depth-axis trajectory of length seq_len * n_sub_trimmed,
    then computes a single |R| matrix per problem from that.

    Note: this conflates layer-axis and position-axis variation. The |R|
    structure reflects coupling that holds across the whole forward pass,
    not coupling at any single point.

    streams: (n_problems, seq_len, n_sub, d_model)
    Returns: per_pp dict (same keys as per_problem_correlation_coupling).
    """
    n_problems, seq_len, n_sub, d_model = streams.shape
    if trim > 0:
        if n_sub <= 2 * trim + 2:
            raise ValueError(
                f"Cannot trim {trim} from each end of {n_sub} sublayers."
            )
        streams = streams[:, :, trim:n_sub - trim, :]
    streams_stacked = streams.reshape(n_problems, -1, d_model)
    print(f"  aggregate trajectory shape per problem: "
          f"{streams_stacked.shape[1]} samples x {d_model} units")
    return per_problem_correlation_coupling(
        streams_stacked, trim=0, k_eigvecs=k_eigvecs,
    )


# ---------------------------------------------------------------------------
# Cross-problem similarities
# ---------------------------------------------------------------------------

def eigenspace_overlap(V1, V2):
    M = V1.T @ V2
    cosines = np.linalg.svd(M, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    return float(np.mean(cosines))


def compute_similarities(per_pp):
    fiedler = per_pp["fiedler_vecs"]
    top_k = per_pp["top_k_eigvecs"]
    affinity = per_pp["abs_corr_matrices"]
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
    """Return (within_add, within_mul, between) similarity arrays."""
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
    aligned = fiedler_vecs.copy()
    labels = np.asarray(group_labels)
    for p in range(1, len(aligned)):
        if labels[p] != labels[p - 1]:
            continue
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


def plot_within_between_summary(sims, group_labels, out_path, suptitle=None):
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

        rng_local = np.random.default_rng(0)
        for d, pos, c in zip(data, positions, colors):
            if len(d) == 0:
                continue
            jitter = rng_local.uniform(-0.10, 0.10, size=len(d))
            ax.scatter(pos + jitter, d, color=c, s=8, alpha=0.6,
                       edgecolor="black", linewidth=0.3, zorder=3)
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

    fig.suptitle(suptitle or "Within-operation vs between-operation coupling similarity",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_problem_diagnostics(per_pp, group_labels, problem_labels, out_path,
                                  title_prefix=""):
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
        f"{title_prefix}Per-problem coupling diagnostics, addition vs multiplication"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_fiedler_heatmap(per_pp, problem_labels, group_labels, n_first_group,
                          out_path, top_units=100, title_prefix=""):
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
        f"{title_prefix}Fiedler vector across problems: top {n_keep} most "
        f"participating units, sign-aligned within each group"
    )
    _draw_group_separator(ax, n_first_group, axis="x",
                          color="black", linewidth=1.5, alpha=1.0)
    plt.colorbar(im, ax=ax, label="Fiedler value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_within_between_trajectory(per_pos_sims, group_labels, position_labels, out_path):
    measures = [
        ("fiedler_abs_cos", "|cos(Fiedler vectors)|"),
        ("eigenspace_topk_overlap", "Top-k eigenspace overlap"),
        ("corr_frobenius_cos", "|R| Frobenius cosine"),
    ]
    seq_len = len(per_pos_sims)
    fig, axes = plt.subplots(3, 1, figsize=(max(8, seq_len * 1.6), 10), sharex=True)
    for ax, (key, title) in zip(axes, measures):
        wa, wm, bt = [], [], []
        for p in range(seq_len):
            sa, sm, sb = within_between_split(per_pos_sims[p][key], group_labels)
            wa.append(np.mean(sa) if len(sa) else float("nan"))
            wm.append(np.mean(sm) if len(sm) else float("nan"))
            bt.append(np.mean(sb) if len(sb) else float("nan"))
        positions = np.arange(seq_len)
        ax.plot(positions, wa, "o-", label="within-add", color="C0", linewidth=2)
        ax.plot(positions, wm, "o-", label="within-mul", color="C1", linewidth=2)
        ax.plot(positions, bt, "o-", label="between", color="C7", linewidth=2)
        ax.set_ylabel("Mean similarity")
        ax.set_title(title)
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)
    axes[-1].set_xticks(np.arange(seq_len))
    axes[-1].set_xticklabels(position_labels, rotation=0)
    axes[-1].set_xlabel("Token position")
    fig.suptitle(
        "Within-operation vs between-operation similarity by token position",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_corr_per_position_grid(per_pos_sims, n_first_group, position_labels, out_path):
    seq_len = len(per_pos_sims)
    fig, axes = plt.subplots(1, seq_len, figsize=(4 * seq_len + 1, 5))
    if seq_len == 1:
        axes = [axes]
    all_sims = np.stack([per_pos_sims[p]["corr_frobenius_cos"] for p in range(seq_len)])
    vmin = float(np.nanmin(all_sims))
    vmax = 1.0
    last_im = None
    for p, ax in enumerate(axes):
        sim = per_pos_sims[p]["corr_frobenius_cos"]
        last_im = ax.imshow(sim, aspect="equal", cmap="viridis",
                            vmin=vmin, vmax=vmax)
        ax.set_title(position_labels[p], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axvline(n_first_group - 0.5, color="white", linewidth=1)
        ax.axhline(n_first_group - 0.5, color="white", linewidth=1)
    fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02,
                 label="|R| Frobenius cosine")
    fig.suptitle(
        "Per-position problem x problem |R| Frobenius cosine similarity "
        "(top-left quadrant: within-add; bottom-right: within-mul)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_position_diagnostics_traj(per_pos_diag, group_labels,
                                        position_labels, out_path):
    metrics = [
        ("effective_rank", "Effective rank"),
        ("mean_off_diag", r"Mean off-diagonal $|R|$"),
        ("spectral_entropy", "Spectral entropy"),
    ]
    seq_len = len(per_pos_diag)
    n_problems = len(group_labels)
    add_idx = [i for i, g in enumerate(group_labels) if g == "add"]
    mul_idx = [i for i, g in enumerate(group_labels) if g == "mul"]
    fig, axes = plt.subplots(3, 1, figsize=(max(8, seq_len * 1.6), 10), sharex=True)
    for ax, (key, ylabel) in zip(axes, metrics):
        traj = np.stack([per_pos_diag[p][key] for p in range(seq_len)], axis=0)
        for i in range(n_problems):
            color = "C0" if group_labels[i] == "add" else "C1"
            ax.plot(np.arange(seq_len), traj[:, i],
                    color=color, alpha=0.2, linewidth=0.8)
        ax.plot(np.arange(seq_len), traj[:, add_idx].mean(axis=1),
                color="C0", linewidth=2.5, marker="o",
                label="addition (mean)")
        ax.plot(np.arange(seq_len), traj[:, mul_idx].mean(axis=1),
                color="C1", linewidth=2.5, marker="o",
                label="multiplication (mean)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)
    axes[-1].set_xticks(np.arange(seq_len))
    axes[-1].set_xticklabels(position_labels, rotation=0)
    axes[-1].set_xlabel("Token position")
    fig.suptitle("Per-position coupling diagnostics by operation",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top similar pairs
# ---------------------------------------------------------------------------

def collect_pair_data(sim, problem_labels, problem_strings, group_labels,
                       correctness=None):
    """Build per-pair dicts. If correctness is provided, include each side's
    predicted integer, ground truth, and correct/incorrect flag."""
    n = sim.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            cat = (
                "within-add" if group_labels[i] == "add" and group_labels[j] == "add"
                else "within-mul" if group_labels[i] == "mul" and group_labels[j] == "mul"
                else "between"
            )
            entry = {
                "i": i,
                "j": j,
                "problem_a_label": problem_labels[i],
                "problem_b_label": problem_labels[j],
                "problem_a": problem_strings[i],
                "problem_b": problem_strings[j],
                "category": cat,
                "similarity": float(sim[i, j]),
            }
            if correctness is not None:
                entry.update({
                    "predicted_a": correctness[i]["predicted_int"],
                    "predicted_b": correctness[j]["predicted_int"],
                    "truth_a": correctness[i]["ground_truth"],
                    "truth_b": correctness[j]["ground_truth"],
                    "correct_a": correctness[i]["correct"],
                    "correct_b": correctness[j]["correct"],
                })
            pairs.append(entry)
    return pairs


def top_pairs_by_category(pairs, k):
    by_cat = {"within-add": [], "within-mul": [], "between": []}
    for p in pairs:
        by_cat[p["category"]].append(p)
    out = {}
    for cat in by_cat:
        sd = sorted(by_cat[cat], key=lambda p: -p["similarity"])
        out[f"top_{cat}"] = sd[:k]
        out[f"bottom_{cat}"] = sorted(sd[-k:], key=lambda p: p["similarity"])
    return out


def _format_pair_side(pair, side):
    """Render '`<problem> <pred>` <mark>' if correctness data is present,
    else just '`<problem>`'."""
    problem_str = pair[f"problem_{side}"]
    if f"predicted_{side}" not in pair:
        return f"`{problem_str}`"
    pred = pair[f"predicted_{side}"]
    truth = pair[f"truth_{side}"]
    correct = pair[f"correct_{side}"]
    if pred is None:
        return f"`{problem_str} ??` (truth {truth})"
    if correct:
        return f"`{problem_str} {pred}` \u2713"
    return f"`{problem_str} {pred}` \u2717 (truth {truth})"


def write_top_pairs_md_section(by_cat_top, k, fp):
    sections = [
        ("top_within-add",    f"Most similar within addition (top {k})"),
        ("top_within-mul",    f"Most similar within multiplication (top {k})"),
        ("top_between",       f"Most similar across operations (top {k})"),
        ("bottom_within-add", f"Least similar within addition (bottom {k})"),
        ("bottom_within-mul", f"Least similar within multiplication (bottom {k})"),
        ("bottom_between",    f"Least similar across operations (bottom {k})"),
    ]
    for key, header in sections:
        fp.write(f"\n## {header}\n\n")
        fp.write("| rank | problem A (model output) | problem B (model output) | similarity |\n")
        fp.write("|---:|:---|:---|---:|\n")
        for r, p in enumerate(by_cat_top[key], 1):
            fp.write(
                f"| {r} | {_format_pair_side(p, 'a')} | "
                f"{_format_pair_side(p, 'b')} | "
                f"{p['similarity']:.4f} |\n"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-per-op", type=int, default=24)
    parser.add_argument("--operand-low", type=int, default=1)
    parser.add_argument("--operand-high", type=int, default=9)
    parser.add_argument("--add-symbol", default="+")
    parser.add_argument("--mul-symbol", default="*")
    parser.add_argument("--n-generate", type=int, default=10,
                        help="Number of tokens to generate per problem to "
                             "capture the model's answer. Set to 0 to skip.")
    parser.add_argument("--trim-sublayers", type=int, default=2)
    parser.add_argument("--top-k-eigenvecs", type=int, default=10)
    parser.add_argument("--top-fiedler-units", type=int, default=100)
    parser.add_argument("--top-pairs", type=int, default=15,
                        help="Number of top/bottom pairs to list per category.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.n_per_op < 2:
        raise SystemExit("--n-per-op must be at least 2.")

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(f"math_couplings_{slugify(args.model)}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:       {args.model}")
    print(f"n_per_op:    {args.n_per_op}")
    print(f"Operands:    [{args.operand_low}, {args.operand_high}]")
    print(f"Symbols:     add='{args.add_symbol}' mul='{args.mul_symbol}'")
    print(f"Out:         {out_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_possible = (args.operand_high - args.operand_low + 1) ** 2
    if args.n_per_op > n_possible:
        raise SystemExit(
            f"--n-per-op ({args.n_per_op}) exceeds the {n_possible} possible "
            f"unique (a, b) pairs in [{args.operand_low}, {args.operand_high}]. "
            f"Reduce --n-per-op or widen the operand range."
        )
    rng = np.random.default_rng(args.seed)
    all_pairs = [
        (a, b)
        for a in range(args.operand_low, args.operand_high + 1)
        for b in range(args.operand_low, args.operand_high + 1)
    ]
    idx = rng.choice(len(all_pairs), size=args.n_per_op, replace=False)
    operands = [all_pairs[int(i)] for i in idx]
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

    print("\nCapturing residual streams + generating answers (addition) ...")
    add_streams, add_tokens, add_gens = capture_full_problems(
        model, tokenizer, add_problems, n_generate=args.n_generate,
    )
    print("\nCapturing residual streams + generating answers (multiplication) ...")
    mul_streams, mul_tokens, mul_gens = capture_full_problems(
        model, tokenizer, mul_problems, n_generate=args.n_generate,
    )

    streams = np.concatenate([add_streams, mul_streams], axis=0)
    group_labels = ["add"] * args.n_per_op + ["mul"] * args.n_per_op
    problem_strings = add_problems + mul_problems
    problem_labels = (
        [f"{ai}+{bi}" for ai, bi in operands]
        + [f"{ai}*{bi}" for ai, bi in operands]
    )
    token_strs = add_tokens + mul_tokens
    generated_texts = add_gens + mul_gens
    all_operands = operands + operands  # matched: same (a, b) in same order
    seq_len = streams.shape[1]
    print(f"\nStreams shape (n_problems, seq_len, n_sub, d_model) = {streams.shape}")
    print(f"Token sequence (from problem 0): {token_strs[0]}")

    # Evaluate correctness
    correctness = evaluate_correctness(all_operands, group_labels, generated_texts)
    n_add_correct = sum(1 for c in correctness[:args.n_per_op] if c["correct"])
    n_mul_correct = sum(1 for c in correctness[args.n_per_op:] if c["correct"])
    print(f"\nModel accuracy:  addition {n_add_correct}/{args.n_per_op} "
          f"({100*n_add_correct/args.n_per_op:.1f}%),  "
          f"multiplication {n_mul_correct}/{args.n_per_op} "
          f"({100*n_mul_correct/args.n_per_op:.1f}%)")
    print("\nPer-problem prediction (mark | a OP b = predicted (truth)):")
    for c in correctness:
        a, b = c["operands"]
        sym = "+" if c["operation"] == "add" else "*"
        mark = "\u2713" if c["correct"] else "\u2717"
        pred_str = (str(c["predicted_int"]) if c["predicted_int"] is not None
                    else "??")
        print(f"  {mark}  {a:>2} {sym} {b:<2} = {pred_str:<6} "
              f"(truth {c['ground_truth']})")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nPer-position coupling analysis (trim={args.trim_sublayers}) ...")
    per_pos_diag, per_pos_sims = per_position_full_analysis(
        streams, trim=args.trim_sublayers, k_eigvecs=args.top_k_eigenvecs,
    )

    print(f"\nWhole-inference (aggregate) coupling analysis ...")
    agg_pp = whole_inference_coupling(
        streams, trim=args.trim_sublayers, k_eigvecs=args.top_k_eigenvecs,
    )
    agg_sims = compute_similarities(agg_pp)

    # Print summaries
    print("\nWithin / between mean similarity at each position:")
    print(f"  {'pos':>3} {'token':<10} {'fied':>8} {'eig':>8} {'corr':>8}  "
          f"(within-add | within-mul | between for each)")
    for p in range(seq_len):
        row = f"  {p:>3} '{token_strs[0][p].strip():<8}'"
        for name in ("fiedler_abs_cos", "eigenspace_topk_overlap", "corr_frobenius_cos"):
            sa, sm, sb = within_between_split(per_pos_sims[p][name], group_labels)
            row += f"  {np.mean(sa):.2f}/{np.mean(sm):.2f}/{np.mean(sb):.2f}"
        print(row)

    print("\nWhole-inference aggregate within/between mean similarity:")
    for name in ("fiedler_abs_cos", "eigenspace_topk_overlap", "corr_frobenius_cos"):
        sa, sm, sb = within_between_split(agg_sims[name], group_labels)
        print(f"  {name:<28} within-add={np.mean(sa):.3f}  "
              f"within-mul={np.mean(sm):.3f}  between={np.mean(sb):.3f}")

    # Position labels for plots: combine index with token (stripped)
    position_labels = [
        f"pos {i}\n'{token_strs[0][i].strip()}'" for i in range(seq_len)
    ]

    print("\nMaking plots ...")
    plot_within_between_trajectory(
        per_pos_sims, group_labels, position_labels,
        out_dir / "within_between_trajectory.png",
    )
    print("  within_between_trajectory.png")

    plot_corr_per_position_grid(
        per_pos_sims, args.n_per_op, position_labels,
        out_dir / "similarity_corr_per_position.png",
    )
    print("  similarity_corr_per_position.png")

    plot_per_position_diagnostics_traj(
        per_pos_diag, group_labels, position_labels,
        out_dir / "per_position_diagnostics.png",
    )
    print("  per_position_diagnostics.png")

    # Aggregate-level plots
    plot_similarity_matrix(
        agg_sims["fiedler_abs_cos"], problem_labels, args.n_per_op,
        "Whole-inference: |cos(Fiedler vectors)|",
        out_dir / "similarity_fiedler_aggregate.png",
        vmin=0.0, vmax=1.0,
    )
    print("  similarity_fiedler_aggregate.png")

    plot_similarity_matrix(
        agg_sims["eigenspace_topk_overlap"], problem_labels, args.n_per_op,
        f"Whole-inference: top-{args.top_k_eigenvecs} eigenspace overlap",
        out_dir / "similarity_eigenspace_aggregate.png",
        vmin=0.0, vmax=1.0,
    )
    print("  similarity_eigenspace_aggregate.png")

    plot_similarity_matrix(
        agg_sims["corr_frobenius_cos"], problem_labels, args.n_per_op,
        r"Whole-inference: $|R|$ Frobenius cosine similarity",
        out_dir / "similarity_corr_aggregate.png",
        vmin=float(np.nanmin(agg_sims["corr_frobenius_cos"])),
        vmax=1.0,
    )
    print("  similarity_corr_aggregate.png")

    plot_within_between_summary(
        agg_sims, group_labels,
        out_dir / "within_between_summary_aggregate.png",
        suptitle="Whole-inference within-op vs between-op coupling similarity",
    )
    print("  within_between_summary_aggregate.png")

    plot_per_problem_diagnostics(
        agg_pp, group_labels, problem_labels,
        out_dir / "per_problem_diagnostics_aggregate.png",
        title_prefix="Whole-inference: ",
    )
    print("  per_problem_diagnostics_aggregate.png")

    plot_fiedler_heatmap(
        agg_pp, problem_labels, group_labels, args.n_per_op,
        out_dir / "fiedler_heatmap_aggregate.png",
        top_units=args.top_fiedler_units,
        title_prefix="Whole-inference: ",
    )
    print("  fiedler_heatmap_aggregate.png")

    # Top similar pairs (using whole-inference aggregate)
    print("\nCollecting top similar pairs (whole-inference aggregate) ...")
    pair_listings = {}
    for measure_key, measure_label in [
        ("corr_frobenius_cos", "Whole-inference |R| Frobenius cosine"),
        ("fiedler_abs_cos", "Whole-inference |cos(Fiedler)|"),
        ("eigenspace_topk_overlap", "Whole-inference top-k eigenspace overlap"),
    ]:
        pairs = collect_pair_data(
            agg_sims[measure_key], problem_labels, problem_strings, group_labels,
            correctness=correctness,
        )
        by_cat = top_pairs_by_category(pairs, args.top_pairs)
        pair_listings[measure_key] = {
            "label": measure_label,
            "by_category": by_cat,
        }

    with open(out_dir / "top_similar_pairs.json", "w") as f:
        json.dump(pair_listings, f, indent=2)
    print("  top_similar_pairs.json")

    with open(out_dir / "top_similar_pairs.md", "w") as f:
        f.write("# Top similar problem pairs (whole-inference aggregate)\n\n")
        f.write(
            f"**Model accuracy:**  addition {n_add_correct}/{args.n_per_op}, "
            f"multiplication {n_mul_correct}/{args.n_per_op}.\n\n"
            f"Each cell shows the problem prompt, the model's predicted "
            f"integer, and a check (\u2713 correct) or cross "
            f"(\u2717 wrong, with the true answer in parentheses). "
            f"Each section ranks the {args.top_pairs} most-similar (and "
            f"{args.top_pairs} least-similar) pairs in three categories: "
            f"within-addition, within-multiplication, and between operations.\n"
        )
        for measure_key, info in pair_listings.items():
            f.write("\n---\n\n")
            f.write(f"# {info['label']}\n")
            write_top_pairs_md_section(info["by_category"], args.top_pairs, f)
    print("  top_similar_pairs.md")

    # Print a teaser to stdout
    teaser = pair_listings["corr_frobenius_cos"]["by_category"]

    def _teaser_line(rank, p):
        def fmt(side):
            pred = p.get(f"predicted_{side}")
            mark = "\u2713" if p.get(f"correct_{side}") else "\u2717"
            pstr = str(pred) if pred is not None else "?"
            return f"{p[f'problem_{side}_label']}={pstr}{mark}"
        return (f"  {rank}. {fmt('a'):<18} vs  {fmt('b'):<18} "
                f"sim={p['similarity']:.4f}")

    print("\nTop 5 most similar within-addition pairs (|R| Frobenius cosine):")
    for r, p in enumerate(teaser["top_within-add"][:5], 1):
        print(_teaser_line(r, p))
    print("Top 5 most similar within-multiplication pairs:")
    for r, p in enumerate(teaser["top_within-mul"][:5], 1):
        print(_teaser_line(r, p))
    print("Top 5 most similar BETWEEN-op pairs:")
    for r, p in enumerate(teaser["top_between"][:5], 1):
        print(_teaser_line(r, p))

    # Summary JSON
    print("\nWriting summary.json ...")
    summary = {
        "model": args.model,
        "n_per_op": args.n_per_op,
        "operand_range": [args.operand_low, args.operand_high],
        "symbols": {"add": args.add_symbol, "mul": args.mul_symbol},
        "matched_operands": True,
        "n_generate": int(args.n_generate),
        "accuracy": {
            "addition_correct": int(n_add_correct),
            "addition_total": int(args.n_per_op),
            "multiplication_correct": int(n_mul_correct),
            "multiplication_total": int(args.n_per_op),
        },
        "correctness": correctness,
        "n_sub_used_per_position": int(per_pos_diag[0]["L"]),
        "n_positions": int(seq_len),
        "n_sub_used_aggregate": int(agg_pp["L"]),
        "d_model": int(agg_pp["D"]),
        "top_k_eigvecs": int(args.top_k_eigenvecs),
        "problem_strings": problem_strings,
        "problem_labels": problem_labels,
        "group_labels": group_labels,
        "tokens_per_problem": token_strs,
        "tokens_at_position_from_problem_0": token_strs[0],
        "per_position": [
            {
                "position": p,
                "token": token_strs[0][p],
                "fiedler_lambda":      per_pos_diag[p]["fiedler_eigvals"].tolist(),
                "spectral_entropy":    per_pos_diag[p]["spectral_entropy"].tolist(),
                "mean_off_diag_abs_R": per_pos_diag[p]["mean_off_diag"].tolist(),
                "effective_rank":      per_pos_diag[p]["effective_rank"].tolist(),
                "within_between_means": {
                    name: {
                        "within_add": float(np.mean(within_between_split(
                            per_pos_sims[p][name], group_labels)[0])),
                        "within_mul": float(np.mean(within_between_split(
                            per_pos_sims[p][name], group_labels)[1])),
                        "between": float(np.mean(within_between_split(
                            per_pos_sims[p][name], group_labels)[2])),
                    }
                    for name in ("fiedler_abs_cos", "eigenspace_topk_overlap",
                                 "corr_frobenius_cos")
                },
            }
            for p in range(seq_len)
        ],
        "aggregate": {
            "fiedler_lambda":      agg_pp["fiedler_eigvals"].tolist(),
            "spectral_entropy":    agg_pp["spectral_entropy"].tolist(),
            "mean_off_diag_abs_R": agg_pp["mean_off_diag"].tolist(),
            "effective_rank":      agg_pp["effective_rank"].tolist(),
            "within_between_means": {
                name: {
                    "within_add": float(np.mean(within_between_split(
                        agg_sims[name], group_labels)[0])),
                    "within_mul": float(np.mean(within_between_split(
                        agg_sims[name], group_labels)[1])),
                    "between": float(np.mean(within_between_split(
                        agg_sims[name], group_labels)[2])),
                }
                for name in ("fiedler_abs_cos", "eigenspace_topk_overlap",
                             "corr_frobenius_cos")
            },
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  summary.json")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
