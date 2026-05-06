"""
Harmonic phase-locking value (PLV) analysis on SmolLM2 360M residual streams.

Tests the frequency-multiplexed coupling hypothesis: whether the same set of
units carries multiple independent communication graphs at different harmonics
of the depth-axis oscillation.

For each unit u, take the FFT bin at harmonic h of the residual-stream
trajectory along the depth axis. Each input n gives a complex coefficient
X_u^(n)[h] with phase theta_u^(n,h). The harmonic-PLV between units i and j at
harmonic h is

    PLV_{ij}^(h) = | E_n exp(i (theta_i^(n,h) - theta_j^(n,h))) |

This is the consistency of the phase relationship between i and j at that
harmonic specifically. Distinct from the cross-bispectrum, which measures
phase coupling across different frequencies.

The diagnostic for frequency multiplexing is the joint distribution of
PLV^(h1) and PLV^(h2) across all unit pairs. Four regimes:

  HH: both high   -> coordinated on both channels
  HL: h1 only     -> fundamental-channel partners
  LH: h2 only     -> harmonic-channel partners
  LL: neither     -> uncoupled

If frequency multiplexing is real, all four regimes should be populated and
the off-diagonal corners (HL and LH) should hold non-trivial pair counts.
If not, the scatter lies on the diagonal.

Sister analysis to cross_bispectrum_smollm2.py. Same model, same hooks, same
WikiText-2 validation set, same skip-first / skip-last semantics.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


# ---------------------------------------------------------------------------
# Loading and stream collection (mirrors cross_bispectrum_smollm2.py)
# ---------------------------------------------------------------------------

def load_model():
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.float16
        if torch.cuda.is_available() or torch.backends.mps.is_available()
        else torch.float32
    )
    device_map = "auto" if (
        torch.cuda.is_available() or torch.backends.mps.is_available()
    ) else None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
        output_hidden_states=False,
    )
    model.eval()
    return model, tokenizer


def get_hook_targets(model):
    """Inputs to the pre-attn and pre-mlp layernorms of every block."""
    blocks = model.model.layers
    targets = []
    for block in blocks:
        targets.append(block.input_layernorm)
        targets.append(block.post_attention_layernorm)
    return targets


def collect_streams(model, tokenizer, texts, seq_len):
    """Forward each text, hook last-token vectors at every sublayer.
    Returns (n_samples, n_sublayers, d_model) float32."""
    targets = get_hook_targets(model)
    out = []
    for i, text in enumerate(texts):
        enc = tokenizer(
            text, return_tensors="pt",
            max_length=seq_len, truncation=True,
        )
        ids = enc["input_ids"].to(next(model.parameters()).device)
        if ids.shape[1] < 8:
            continue
        captures = []

        def make_hook(c):
            def fn(m, inp, out_):
                c.append(inp[0].detach())
            return fn

        hooks = [t.register_forward_hook(make_hook(captures))
                 for t in targets]
        try:
            with torch.no_grad():
                model(ids)
        finally:
            for h in hooks:
                h.remove()
        last = torch.stack(
            [c[0, -1, :].float() for c in captures], dim=0
        )
        out.append(last.cpu().numpy())
        if (i + 1) % 16 == 0:
            print(f"    {i + 1} / {len(texts)} samples ...")
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Spectrum and harmonic-PLV core
# ---------------------------------------------------------------------------

def streams_to_spec(streams):
    """(N, L, D) real -> (N, D, K) complex spectrum (mean-centered along
    the sublayer axis, then rFFT). K = L // 2 + 1."""
    centered = streams - streams.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(centered.astype(np.float64), axis=1)  # (N, K, D)
    return spec.transpose(0, 2, 1).astype(np.complex128)     # (N, D, K)


def per_harmonic_complex(spec, h):
    """spec: (N, D, K).  Returns (N, D) complex coefficients at bin h."""
    return spec[:, :, h]


def plv_matrix(Z):
    """Phase-only PLV.  Z: (N, D) complex.  Returns (D, D) complex M with
        M[i,j] = E_n[ exp(i (theta_i - theta_j)) ]
              = E_n[ Z_i / |Z_i| * conj(Z_j / |Z_j|) ]
    |M[i,j]| in [0,1].  arg(M[i,j]) is the typical phase offset of i over j.
    Diagonal is exactly 1."""
    N = Z.shape[0]
    Zp = Z / (np.abs(Z) + 1e-12)
    M = (Zp.T @ Zp.conj()) / N
    return M


def coherence_matrix(Z):
    """Magnitude-weighted version (classical complex coherence).
    Z: (N, D) complex.  Returns (D, D) complex C with |C[i,j]| in [0,1]."""
    N = Z.shape[0]
    cross = (Z.T @ Z.conj()) / N
    auto = np.mean(np.abs(Z) ** 2, axis=0)            # (D,)
    norm = np.sqrt(np.outer(auto, auto))
    return cross / (norm + 1e-12)


def plv_null_pool(Z, n_shuffle=10, rng=None):
    """Shuffle inputs of unit B; take off-diagonal magnitudes.
    Returns flat float32 pool."""
    if rng is None:
        rng = np.random.default_rng(0)
    N, D = Z.shape
    Zp = Z / (np.abs(Z) + 1e-12)
    pool = []
    for _ in range(n_shuffle):
        perm = rng.permutation(N)
        Zsh = Zp[perm]
        M = (Zp.T @ Zsh.conj()) / N
        mask = ~np.eye(D, dtype=bool)
        pool.append(np.abs(M[mask]).astype(np.float32))
    return np.concatenate(pool)


def degree_at_threshold(M_abs, threshold):
    """For each unit, count off-diagonal partners with PLV > threshold."""
    above = (M_abs > threshold).astype(np.int32)
    np.fill_diagonal(above, 0)
    return above.sum(axis=1)


def reorder_by_first_eigenvector(M_abs):
    """Return permutation that sorts units by leading eigenvector of |M|.
    Matches the convention in cross_bispectrum_smollm2.py."""
    A = M_abs.copy()
    np.fill_diagonal(A, 0)
    eigvals, eigvecs = np.linalg.eigh(A)
    v = eigvecs[:, -1]
    return np.argsort(-np.abs(v))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_plv_distributions(plv_off, null_off, harmonics, save_path):
    """Histogram of off-diagonal |M| at each harmonic, vs null."""
    n_h = len(harmonics)
    fig, axes = plt.subplots(
        1, n_h, figsize=(4.5 * n_h, 3.6), sharey=True,
    )
    if n_h == 1:
        axes = [axes]
    bins = np.linspace(0, 1, 80)
    for ax, h, p, np_ in zip(axes, harmonics, plv_off, null_off):
        ax.hist(np_, bins=bins, density=True, alpha=0.5,
                color="0.5", label="null (shuffled)")
        ax.hist(p, bins=bins, density=True, alpha=0.7,
                color="C0", label="real")
        p99_null = np.percentile(np_, 99)
        ax.axvline(p99_null, color="C3", linestyle="--",
                   linewidth=1, label=f"null p99 = {p99_null:.3f}")
        ax.set_yscale("log")
        ax.set_title(f"harmonic h = {h}", fontsize=10)
        ax.set_xlabel(r"$|M_{ij}|$  off-diagonal PLV")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density (log)")
    fig.suptitle(
        "Off-diagonal harmonic-PLV vs phase-shuffle null",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_plv_scatter(plv1_off, plv2_off, h1, h2,
                     thresholds, save_path):
    """The key diagnostic: scatter of PLV^(h1) vs PLV^(h2) for all pairs.
    Hexbin density.  Quadrant counts annotated.
    `thresholds` is a tuple (t1, t2) for the two harmonics."""
    fig, ax = plt.subplots(figsize=(6.4, 6))
    hb = ax.hexbin(
        plv1_off, plv2_off,
        gridsize=60, extent=(0, 1, 0, 1),
        bins="log", cmap="magma", mincnt=1,
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("log10 pair count")

    t1, t2 = thresholds
    ax.axvline(t1, color="C2", linestyle="--", linewidth=1)
    ax.axhline(t2, color="C2", linestyle="--", linewidth=1)
    ax.plot([0, 1], [0, 1], color="0.5", linestyle=":", linewidth=1)

    # Quadrant counts
    high1 = plv1_off > t1
    high2 = plv2_off > t2
    qHH = (high1 & high2).sum()
    qHL = (high1 & ~high2).sum()
    qLH = (~high1 & high2).sum()
    qLL = (~high1 & ~high2).sum()

    def fmt(n):
        if n >= 1e6:
            return f"{n/1e6:.2f}M"
        if n >= 1e3:
            return f"{n/1e3:.1f}k"
        return str(int(n))

    ax.text(0.97, 0.97, f"HH\n{fmt(qHH)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec="C2", alpha=0.9))
    ax.text(0.97, 0.03, f"HL (h{h1} only)\n{fmt(qHL)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec="C2", alpha=0.9))
    ax.text(0.03, 0.97, f"LH (h{h2} only)\n{fmt(qLH)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec="C2", alpha=0.9))
    ax.text(0.03, 0.03, f"LL\n{fmt(qLL)}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec="C2", alpha=0.9))

    ax.set_xlabel(rf"$\mathrm{{PLV}}^{{(h={h1})}}_{{ij}}$")
    ax.set_ylabel(rf"$\mathrm{{PLV}}^{{(h={h2})}}_{{ij}}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(
        "Joint distribution of harmonic-PLV across pairs\n"
        "(off-diagonal density = frequency multiplexing)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_plv_matrices(M_abs_list, harmonics, top_units,
                      shared_order, save_path):
    """Side-by-side D'xD' submatrices at each harmonic, sorted by a shared
    order (typically derived from the primary harmonic)."""
    n_h = len(harmonics)
    fig, axes = plt.subplots(
        1, n_h, figsize=(4.6 * n_h, 4.6), sharey=True,
    )
    if n_h == 1:
        axes = [axes]
    sub_idx = top_units[shared_order]
    for ax, h, M_abs in zip(axes, harmonics, M_abs_list):
        sub = M_abs[np.ix_(sub_idx, sub_idx)]
        np.fill_diagonal(sub, 0.0)
        im = ax.imshow(
            sub, cmap="magma", vmin=0, vmax=min(1, sub.max()),
            interpolation="nearest", aspect="equal",
        )
        ax.set_title(f"|M| at h = {h}", fontsize=10)
        ax.set_xlabel("unit (reordered)")
        if h == harmonics[0]:
            ax.set_ylabel("unit (reordered)")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="|PLV|")
    fig.suptitle(
        f"Harmonic-PLV submatrices, top {len(top_units)} units, "
        "shared community order",
        fontsize=11,
    )
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_hub_degree_scatter(deg1, deg2, h1, h2, threshold,
                             save_path, top_n=15):
    """Per-unit degree in G^(h1) vs G^(h2).  Highlight specialists."""
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.scatter(deg1, deg2, s=12, alpha=0.5, color="0.4")

    # Multi-channel hubs (high in both)
    multi_idx = np.argsort(-(deg1 + deg2))[:top_n]
    ax.scatter(deg1[multi_idx], deg2[multi_idx],
               s=40, color="C3", label=f"top-{top_n} multi-channel hubs",
               edgecolor="white", linewidth=0.5)
    for u in multi_idx[:5]:
        ax.annotate(str(u), (deg1[u], deg2[u]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")

    # Specialists: high one, low other (relative)
    diff = deg1 - deg2
    spec_h1 = np.argsort(-diff)[:top_n // 2]
    spec_h2 = np.argsort(diff)[:top_n // 2]
    ax.scatter(deg1[spec_h1], deg2[spec_h1],
               s=40, color="C0",
               label=f"top h{h1}-specialists",
               edgecolor="white", linewidth=0.5)
    ax.scatter(deg1[spec_h2], deg2[spec_h2],
               s=40, color="C2",
               label=f"top h{h2}-specialists",
               edgecolor="white", linewidth=0.5)
    for u in spec_h1[:3]:
        ax.annotate(str(u), (deg1[u], deg2[u]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    for u in spec_h2[:3]:
        ax.annotate(str(u), (deg1[u], deg2[u]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")

    lim = max(deg1.max(), deg2.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="0.7", linestyle=":", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(rf"degree in $G^{{({h1})}}$  "
                  rf"($\mathrm{{PLV}}>{threshold:.3f}$)")
    ax.set_ylabel(rf"degree in $G^{{({h2})}}$  "
                  rf"($\mathrm{{PLV}}>{threshold:.3f}$)")
    ax.set_title("Hub structure across harmonics", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_regime_counts(quadrant_counts, h1, h2, save_path):
    """Bar chart of HH / HL / LH / LL counts."""
    labels = [
        f"HH\n(both > p99)",
        f"HL\nh{h1} only",
        f"LH\nh{h2} only",
        f"LL\nneither",
    ]
    counts = [
        quadrant_counts["HH"],
        quadrant_counts["HL"],
        quadrant_counts["LH"],
        quadrant_counts["LL"],
    ]
    colors = ["C3", "C0", "C2", "0.6"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, counts, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("pair count (log)")
    ax.set_title(
        f"Pair regime counts at PLV null-p99 threshold (h={h1}, h={h2})",
        fontsize=11,
    )
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() * 1.05,
                f"{c}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str,
                        default="smollm2_harmonic_plv")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--harmonics", type=int, nargs="+", default=[1, 2, 3],
        help="FFT bins to analyze, relative to the trimmed sublayer range.",
    )
    parser.add_argument(
        "--scatter-pair", type=int, nargs=2, default=[1, 2],
        help="Two harmonics for the joint scatter plot.",
    )
    parser.add_argument(
        "--top-units-matrix", type=int, default=128,
        help="Number of units in the side-by-side PLV submatrix plot. "
             "Selected by sum of degrees across harmonics.",
    )
    parser.add_argument(
        "--top-pairs", type=int, default=30,
        help="How many top pairs per harmonic to record in JSON.",
    )
    parser.add_argument(
        "--n-shuffle", type=int, default=10,
        help="Permutations for the PLV null pool, per harmonic.",
    )
    parser.add_argument("--skip-first", type=int, default=0,
                        help="Drop the first N sublayers from the FFT input.")
    parser.add_argument("--skip-last", type=int, default=0,
                        help="Drop the last N sublayers from the FFT input.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.skip_first < 0 or args.skip_last < 0:
        raise ValueError("--skip-first and --skip-last must be >= 0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ------------- data -----------------
    print("Loading WikiText-2 ...")
    ds = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="validation",
    )
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]
    print(f"  {len(texts)} samples")

    model, tokenizer = load_model()

    print("\nCollecting streams ...")
    streams = collect_streams(model, tokenizer, texts, args.seq_len)
    print(f"  shape: {streams.shape}  (n_samples, n_sublayers, d_model)")

    L_full = streams.shape[1]
    skip_first = args.skip_first
    skip_last = args.skip_last
    if skip_first + skip_last >= L_full:
        raise ValueError(
            f"skip_first ({skip_first}) + skip_last ({skip_last}) >= "
            f"n_sublayers ({L_full}); nothing left to analyze."
        )
    if skip_first + skip_last > 0:
        end = L_full - skip_last if skip_last > 0 else L_full
        streams = streams[:, skip_first:end, :]
        print(
            f"  trimmed sublayers: kept indices "
            f"[{skip_first}:{end}] of original {L_full} "
            f"(skipped first={skip_first}, last={skip_last}, "
            f"new L={streams.shape[1]})"
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    N, L, D = streams.shape
    K = L // 2 + 1
    for h in args.harmonics:
        if h < 1 or h >= K:
            raise ValueError(
                f"Harmonic {h} is outside valid range [1, {K-1}] for L={L}."
            )

    print("\nComputing spectrum ...")
    spec = streams_to_spec(streams)        # (N, D, K)

    # ------------- per-harmonic PLV matrices -----------------
    print("\nComputing harmonic-PLV matrices ...")
    M_complex = {}
    M_abs = {}
    Coh_abs = {}
    for h in args.harmonics:
        Z = per_harmonic_complex(spec, h)            # (N, D)
        M = plv_matrix(Z)
        C = coherence_matrix(Z)
        M_complex[h] = M.astype(np.complex64)
        M_abs[h] = np.abs(M).astype(np.float32)
        Coh_abs[h] = np.abs(C).astype(np.float32)
        offdiag = np.abs(M[~np.eye(D, dtype=bool)])
        print(f"  h={h}:  off-diag mean = {offdiag.mean():.3f},  "
              f"p95 = {np.percentile(offdiag, 95):.3f},  "
              f"p99 = {np.percentile(offdiag, 99):.3f},  "
              f"max = {offdiag.max():.3f}")

    # ------------- nulls -----------------
    print("\nComputing PLV null pools ...")
    nulls = {}
    null_p95 = {}
    null_p99 = {}
    for h in args.harmonics:
        Z = per_harmonic_complex(spec, h)
        pool = plv_null_pool(Z, n_shuffle=args.n_shuffle, rng=rng)
        nulls[h] = pool
        null_p95[h] = float(np.percentile(pool, 95))
        null_p99[h] = float(np.percentile(pool, 99))
        print(f"  h={h}:  null p95 = {null_p95[h]:.3f},  "
              f"p99 = {null_p99[h]:.3f},  "
              f"max = {pool.max():.3f}")

    rayleigh_p99 = float(np.sqrt(-np.log(0.01) / N))
    print(f"\nAnalytical Rayleigh p99 (N={N}): {rayleigh_p99:.3f}")

    # ------------- per-unit degree per harmonic -----------------
    print("\nDegree per unit per harmonic ...")
    degrees = {}
    for h in args.harmonics:
        thr = null_p99[h]
        degrees[h] = degree_at_threshold(M_abs[h], thr)
        print(f"  h={h}:  threshold = {thr:.3f},  "
              f"mean degree = {degrees[h].mean():.1f},  "
              f"max degree = {degrees[h].max()}")

    # ------------- the joint scatter -----------------
    h1, h2 = args.scatter_pair
    if h1 not in args.harmonics or h2 not in args.harmonics:
        # add them in
        for h in [h1, h2]:
            if h not in M_abs:
                Z = per_harmonic_complex(spec, h)
                M_abs[h] = np.abs(plv_matrix(Z)).astype(np.float32)
                pool = plv_null_pool(
                    Z, n_shuffle=args.n_shuffle, rng=rng,
                )
                null_p99[h] = float(np.percentile(pool, 99))
                degrees[h] = degree_at_threshold(M_abs[h], null_p99[h])

    mask = ~np.eye(D, dtype=bool)
    plv1_off = M_abs[h1][mask]
    plv2_off = M_abs[h2][mask]

    t1 = null_p99[h1]
    t2 = null_p99[h2]
    high1 = plv1_off > t1
    high2 = plv2_off > t2
    quadrant_counts = {
        "HH": int((high1 & high2).sum()),
        "HL": int((high1 & ~high2).sum()),
        "LH": int((~high1 & high2).sum()),
        "LL": int((~high1 & ~high2).sum()),
        "threshold_h1": t1,
        "threshold_h2": t2,
    }
    print(f"\nQuadrant counts at null-p99 thresholds "
          f"(h{h1}={t1:.3f}, h{h2}={t2:.3f}):")
    for k in ("HH", "HL", "LH", "LL"):
        print(f"  {k}:  {quadrant_counts[k]}")

    # ------------- top pairs per harmonic -----------------
    top_pairs_per_h = {}
    ii_u, jj_u = np.where(np.triu(mask, k=1))  # upper triangle
    for h in args.harmonics:
        vals = M_abs[h][ii_u, jj_u]
        order = np.argsort(-vals)[:args.top_pairs]
        top_pairs_per_h[h] = [
            {
                "unit_A": int(ii_u[k]),
                "unit_B": int(jj_u[k]),
                "plv": float(vals[k]),
                "phase_offset_radians": float(
                    np.angle(M_complex[h][ii_u[k], jj_u[k]])
                    if h in M_complex else 0.0
                ),
                "ratio_to_null_p99": (
                    float(vals[k] / null_p99[h]) if null_p99[h] > 0 else None
                ),
            }
            for k in order
        ]

    # ------------- top hubs / specialists -----------------
    h_for_hubs = sorted(args.harmonics)[:2]
    if h1 not in h_for_hubs:
        h_for_hubs = [h1, h2]
    deg_a = degrees[h_for_hubs[0]]
    deg_b = degrees[h_for_hubs[1]]
    multi_idx = np.argsort(-(deg_a + deg_b))[:30]
    spec_a_idx = np.argsort(-(deg_a - deg_b))[:30]
    spec_b_idx = np.argsort(-(deg_b - deg_a))[:30]

    top_multi_hubs = [
        {
            "unit": int(u),
            f"degree_h{h_for_hubs[0]}": int(deg_a[u]),
            f"degree_h{h_for_hubs[1]}": int(deg_b[u]),
        }
        for u in multi_idx[:15]
    ]
    top_h1_specialists = [
        {
            "unit": int(u),
            f"degree_h{h_for_hubs[0]}": int(deg_a[u]),
            f"degree_h{h_for_hubs[1]}": int(deg_b[u]),
            "diff": int(deg_a[u] - deg_b[u]),
        }
        for u in spec_a_idx[:15]
    ]
    top_h2_specialists = [
        {
            "unit": int(u),
            f"degree_h{h_for_hubs[0]}": int(deg_a[u]),
            f"degree_h{h_for_hubs[1]}": int(deg_b[u]),
            "diff": int(deg_b[u] - deg_a[u]),
        }
        for u in spec_b_idx[:15]
    ]

    print(f"\nTop multi-channel hubs (high in both h={h_for_hubs[0]} "
          f"and h={h_for_hubs[1]}):")
    for r in top_multi_hubs[:10]:
        print(f"  unit {r['unit']:>4}:  "
              f"deg(h{h_for_hubs[0]}) = {r[f'degree_h{h_for_hubs[0]}']:>4}, "
              f"deg(h{h_for_hubs[1]}) = {r[f'degree_h{h_for_hubs[1]}']:>4}")

    print(f"\nTop h{h_for_hubs[0]}-specialists "
          f"(high in h{h_for_hubs[0]}, low in h{h_for_hubs[1]}):")
    for r in top_h1_specialists[:10]:
        print(f"  unit {r['unit']:>4}:  "
              f"deg(h{h_for_hubs[0]}) = {r[f'degree_h{h_for_hubs[0]}']:>4}, "
              f"deg(h{h_for_hubs[1]}) = {r[f'degree_h{h_for_hubs[1]}']:>4}, "
              f"diff = +{r['diff']}")

    print(f"\nTop h{h_for_hubs[1]}-specialists "
          f"(high in h{h_for_hubs[1]}, low in h{h_for_hubs[0]}):")
    for r in top_h2_specialists[:10]:
        print(f"  unit {r['unit']:>4}:  "
              f"deg(h{h_for_hubs[0]}) = {r[f'degree_h{h_for_hubs[0]}']:>4}, "
              f"deg(h{h_for_hubs[1]}) = {r[f'degree_h{h_for_hubs[1]}']:>4}, "
              f"diff = +{r['diff']}")

    # ------------- plots -----------------
    print("\nPlotting ...")
    plot_plv_distributions(
        plv_off=[M_abs[h][mask] for h in args.harmonics],
        null_off=[nulls[h] for h in args.harmonics],
        harmonics=args.harmonics,
        save_path=str(out_dir / "plv_distribution_per_harmonic.png"),
    )

    plot_plv_scatter(
        plv1_off=plv1_off, plv2_off=plv2_off,
        h1=h1, h2=h2,
        thresholds=(t1, t2),
        save_path=str(out_dir / "plv_scatter.png"),
    )

    # Side-by-side matrices: pick top-N units by total degree across both
    union_score = deg_a.astype(np.int64) + deg_b.astype(np.int64)
    top_units = np.argsort(-union_score)[:args.top_units_matrix]
    shared_order = reorder_by_first_eigenvector(
        M_abs[h1][np.ix_(top_units, top_units)],
    )
    plot_plv_matrices(
        M_abs_list=[M_abs[h1], M_abs[h2]],
        harmonics=[h1, h2],
        top_units=top_units,
        shared_order=shared_order,
        save_path=str(out_dir / "plv_matrices.png"),
    )

    plot_hub_degree_scatter(
        deg1=deg_a, deg2=deg_b,
        h1=h_for_hubs[0], h2=h_for_hubs[1],
        threshold=null_p99[h_for_hubs[0]],
        save_path=str(out_dir / "hub_degree_scatter.png"),
    )

    plot_regime_counts(
        quadrant_counts=quadrant_counts, h1=h1, h2=h2,
        save_path=str(out_dir / "regime_counts.png"),
    )

    # ------------- save data -----------------
    print("\nSaving data ...")
    np.savez_compressed(
        out_dir / "plv_data.npz",
        harmonics=np.array(args.harmonics),
        scatter_pair=np.array(args.scatter_pair),
        **{f"M_abs_h{h}":      M_abs[h]      for h in M_abs},
        **{f"M_real_h{h}":     M_complex[h].real
           for h in M_complex},
        **{f"M_imag_h{h}":     M_complex[h].imag
           for h in M_complex},
        **{f"coherence_h{h}":  Coh_abs[h]    for h in Coh_abs},
        **{f"degree_h{h}":     degrees[h].astype(np.int32)
           for h in degrees},
        **{f"null_pool_h{h}":  nulls[h].astype(np.float32)
           for h in nulls},
    )

    summary = {
        "model":             MODEL_NAME,
        "n_samples":         int(N),
        "n_sublayers":       int(L),
        "n_sublayers_full":  int(L_full),
        "skip_first":        int(skip_first),
        "skip_last":         int(skip_last),
        "d_model":           int(D),
        "fft_length_L":      int(L),
        "K_positive_freqs":  int(K),
        "harmonics":         list(args.harmonics),
        "scatter_pair":      list(args.scatter_pair),
        "analytical_rayleigh_p99": rayleigh_p99,
        "per_harmonic": {
            str(h): {
                "offdiag_mean":     float(M_abs[h][mask].mean()),
                "offdiag_p95":      float(np.percentile(M_abs[h][mask], 95)),
                "offdiag_p99":      float(np.percentile(M_abs[h][mask], 99)),
                "offdiag_max":      float(M_abs[h][mask].max()),
                "null_p95":         null_p95.get(h, None),
                "null_p99":         null_p99.get(h, None),
                "degree_threshold": null_p99[h],
                "mean_degree":      float(degrees[h].mean()),
                "max_degree":       int(degrees[h].max()),
                "fraction_pairs_above_null_p99": float(
                    (M_abs[h][mask] > null_p99[h]).mean()
                ),
            }
            for h in args.harmonics
        },
        "scatter_diagnostic": {
            "h1": int(h1),
            "h2": int(h2),
            "threshold_h1": t1,
            "threshold_h2": t2,
            "quadrant_counts": {
                k: quadrant_counts[k] for k in ("HH", "HL", "LH", "LL")
            },
            "n_total_pairs": int(mask.sum()),
            "fraction_HH": quadrant_counts["HH"] / int(mask.sum()),
            "fraction_HL": quadrant_counts["HL"] / int(mask.sum()),
            "fraction_LH": quadrant_counts["LH"] / int(mask.sum()),
            "fraction_LL": quadrant_counts["LL"] / int(mask.sum()),
        },
        "top_pairs_per_harmonic": {
            str(h): top_pairs_per_h[h] for h in args.harmonics
        },
        "top_multi_channel_hubs": top_multi_hubs,
        f"top_h{h_for_hubs[0]}_specialists": top_h1_specialists,
        f"top_h{h_for_hubs[1]}_specialists": top_h2_specialists,
    }

    with open(out_dir / "harmonic_plv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.  Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
