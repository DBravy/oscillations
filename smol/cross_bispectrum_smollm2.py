"""
Cross-bispectrum analysis on SmolLM2 360M residual streams.

Tests whether pairs of units share a "harmonic channel" along the depth
axis. For a unit u and input n, the residual stream trajectory across
sublayers s_u^(n)[l] has FFT coefficients X_u^(n)[k]. The bispectral
indicator at pair (k1, k2) is

    h_u^(n)(k1, k2) = X_u^(n)[k1] * X_u^(n)[k2] * conj(X_u^(n)[k1+k2]).

Its phase is the relative phase of the (k1+k2) component to the sum of
the (k1) and (k2) components. For (k1, k2) = (1, 1) this is exactly
the second-harmonic phase relative to the fundamental, the
"phi^(2)" of interest.

Three measurements per (k1, k2) pair:

  1. Auto-bicoherence (per unit, [0, 1]).
       b_u = |E_n[h_u]| / sqrt(E_n[|X[k1]X[k2]|^2] * E_n[|X[k1+k2]|^2]).
     Standard Hinich/Kim-Powers normalization. b_u close to 1 means
     unit u has a consistent harmonic phase across inputs. b_u close
     to 0 means harmonics are phase-randomized within the unit.

  2. Cross-bicoherence (per pair, complex, magnitude in [0, 1]).
       B_{AB} = E_n[h_A * conj(h_B)]
              / sqrt(E_n[|h_A|^2] * E_n[|h_B|^2]).
     Amplitude-weighted complex correlation between bispectral
     indicators. |B_{AB}| close to 1 means phi_A^(2) - phi_B^(2) is
     stable across inputs (shared harmonic channel).

  3. Cross-PLV (per pair, real, [0, 1]).
       |E_n[hat_h_A * conj(hat_h_B)]|, hat_h = h / |h|.
     Phase-only version, robust to inputs where one unit happens to
     have low harmonic amplitude. Reported on top pairs from B_{AB}.

Null: random permutation of input indices for unit B (preserves each
unit's own bispectrum, destroys inter-unit phase coupling). Pooled
across many cell pairs and shuffles to estimate the null distribution
of |B_{AB}| under H0 of independence. Analytical Rayleigh threshold
sqrt(-ln(p)/N) reported alongside as a sanity check.

Inputs: WikiText-2 validation, last-token snapshots at each
input-LN and post-attention-LN of every transformer block (so 64
sublayer captures for SmolLM2 360M).

Outputs in --out-dir:
  bicoh_auto_summary.png        max_bin x max_bin heatmap, mean over units
  bicoh_auto_per_unit.png       D x n_pairs heatmap, sorted by max bicoh
  z_trajectory_top_units.png    fundamental + harmonic in complex plane
                                  for the top-bicoherence units, several inputs
  bispectral_phase_polar.png    polar histograms of phi_u^(2) across inputs
  cross_bicoh_matrix.png        D x D heatmap at primary pair, reordered
                                  so phase-locked communities cluster
  cross_pair_examples.png       phi_A vs phi_B scatter on top pairs
  null_comparison.png           histograms: real vs shuffled |B_{AB}|
  cross_bispectrum_data.npz     all arrays
  cross_bispectrum_summary.json numerical summary, top findings

Usage:
  python cross_bispectrum_smollm2.py \\
      --out-dir smollm2_cross_bispectrum \\
      --n-samples 256 --seq-len 128 \\
      --pairs 1,1 1,2 2,2 1,3 2,3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


# ---------------------------------------------------------------------------
# Loading and stream collection (mirrors pac_probe_smollm2.py)
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
    """Return modules whose INPUTS are the residual stream snapshots:
    pre-attn layernorm and pre-mlp layernorm of every block."""
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
# Bispectrum core
# ---------------------------------------------------------------------------

def streams_to_spec(streams):
    """(N, L, D) real -> (N, D, K) complex spectrum, after centering
    the depth axis. K = L // 2 + 1."""
    centered = streams - streams.mean(axis=1, keepdims=True)
    # FFT along the depth axis L
    spec = np.fft.rfft(centered.astype(np.float64), axis=1)  # (N, K, D)
    return spec.transpose(0, 2, 1).astype(np.complex128)     # (N, D, K)


def bispectral_indicator(spec, k1, k2):
    """h_u^(n)(k1,k2) = X[k1] * X[k2] * conj(X[k1+k2]).
    spec: (N, D, K). Returns (N, D) complex."""
    return spec[:, :, k1] * spec[:, :, k2] * np.conj(spec[:, :, k1 + k2])


def auto_bicoherence(spec, k1, k2):
    """Hinich-style auto bicoherence per unit. Returns (D,) in [0, 1]."""
    h = bispectral_indicator(spec, k1, k2)                  # (N, D)
    num = np.abs(h.mean(axis=0))                            # (D,)
    p12 = spec[:, :, k1] * spec[:, :, k2]
    den1 = np.sqrt(np.mean(np.abs(p12) ** 2, axis=0))       # (D,)
    den2 = np.sqrt(np.mean(np.abs(spec[:, :, k1 + k2]) ** 2,
                           axis=0))                          # (D,)
    return num / (den1 * den2 + 1e-30)


def auto_plv(h):
    """Phase-only PLV per unit. h: (N, D) -> (D,) in [0, 1]."""
    h_hat = h / (np.abs(h) + 1e-30)
    return np.abs(h_hat.mean(axis=0))


def auto_resultant_phase(h):
    """Mean bispectral phase per unit. h: (N, D) -> (D,) angles."""
    h_hat = h / (np.abs(h) + 1e-30)
    return np.angle(h_hat.mean(axis=0))


def cross_bicoherence_matrix(h):
    """Complex amplitude-weighted cross-bicoherence at a fixed (k1, k2).
    h: (N, D) -> M (D, D) complex, |M_{AB}| in [0, 1].
    M_{AB} = E_n[h_A * conj(h_B)] / sqrt(E_n[|h_A|^2] * E_n[|h_B|^2]).
    With this convention, arg(M_{AB}) is the average bispectral-phase
    DIFFERENCE phi_A^(k1,k2) - phi_B^(k1,k2). Diagonal == 1."""
    N, D = h.shape
    cov = (h.T @ h.conj()) / N                              # (D, D)
    var = np.mean(np.abs(h) ** 2, axis=0)                   # (D,)
    den = np.sqrt(np.outer(var, var)) + 1e-30
    return cov / den


def cross_plv_matrix(h):
    """Phase-only cross PLV at fixed (k1, k2).
    M_{AB} = |E_n[hat_h_A * conj(hat_h_B)]| with hat_h = h / |h|.
    Returns (D, D) real in [0, 1]."""
    h_hat = h / (np.abs(h) + 1e-30)
    N = h_hat.shape[0]
    M = (h_hat.T @ h_hat.conj()) / N
    return np.abs(M).astype(np.float32)


def cross_bicoh_null_pool(h, n_shuffle, rng):
    """Pool null |B_{AB}| values from many shuffles of unit-B input
    indices. Each shuffle preserves each unit's own auto-bicoherence
    but breaks any across-unit phase coupling. Returns flat array of
    off-diagonal magnitudes pooled over shuffles."""
    N, D = h.shape
    var = np.mean(np.abs(h) ** 2, axis=0)
    den_outer = np.sqrt(np.outer(var, var)) + 1e-30
    iu = np.triu_indices(D, k=1)
    out = []
    for r in range(n_shuffle):
        perm = rng.permutation(N)
        h_shuf = h[perm]
        cov = (h.T @ h_shuf.conj()) / N
        mag = np.abs(cov / den_outer).astype(np.float32)
        out.append(mag[iu].copy())
    return np.concatenate(out)


def auto_bicoh_null_pool(spec, k1, k2, n_shuffle, rng):
    """For the auto bicoherence null, phase-randomize the spectrum at
    bins {k1, k2, k1+k2} independently per (input, unit). Returns
    (n_shuffle, D)."""
    N, D, K = spec.shape
    null = np.zeros((n_shuffle, D), dtype=np.float32)
    for r in range(n_shuffle):
        spec_shuf = spec.copy()
        for k in {k1, k2, k1 + k2}:
            mag = np.abs(spec_shuf[:, :, k])
            ph = rng.uniform(-np.pi, np.pi, size=(N, D))
            spec_shuf[:, :, k] = mag * np.exp(1j * ph)
        b = auto_bicoherence(spec_shuf, k1, k2)
        null[r] = b
    return null


# ---------------------------------------------------------------------------
# Trajectory reconstruction in the complex plane
# ---------------------------------------------------------------------------

def isolate_bin(spec_uk, k, L):
    """Analytic signal at one positive bin. spec_uk: (...,) complex coef
    at bin k. Returns z[L] complex via inverse using only bin k.
    Same convention as pac_probe_smollm2.isolate_bin_signal."""
    ls = np.arange(L)
    expand = spec_uk[..., None] * np.exp(2j * np.pi * k * ls / L)
    if k == 0 or (L % 2 == 0 and k == L // 2):
        return expand / L
    return 2 * expand / L


def reconstruct_z_from_bins(spec_unit_input, bins, L):
    """Sum analytic signals at the listed bins. Returns complex (L,)."""
    z = np.zeros(L, dtype=np.complex128)
    for k in bins:
        z = z + isolate_bin(spec_unit_input[k], k, L)
    return z


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_bicoh_auto_summary(per_unit_pair, pairs, max_bin, save_path):
    """per_unit_pair: (D, n_pairs). Mean over units, lay out in a
    (max_bin, max_bin) heatmap."""
    means = per_unit_pair.mean(axis=0)
    M = np.full((max_bin, max_bin), np.nan)
    for j, (a, b) in enumerate(pairs):
        M[a - 1, b - 1] = means[j]
        M[b - 1, a - 1] = means[j]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(
        M, origin="lower", aspect="equal", cmap="magma",
        extent=[0.5, max_bin + 0.5, 0.5, max_bin + 0.5],
        interpolation="nearest",
    )
    ax.set_xlabel(r"$k_1$")
    ax.set_ylabel(r"$k_2$")
    ax.set_title(
        "Mean auto-bicoherence across units (SmolLM2 360M)",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r"$\overline{b_u(k_1, k_2)}$")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_bicoh_auto_per_unit(per_unit_pair, pairs, save_path):
    D, n_pairs = per_unit_pair.shape
    row_order = np.argsort(-np.max(per_unit_pair, axis=1))
    sorted_table = per_unit_pair[row_order]
    fig, ax = plt.subplots(
        figsize=(min(0.5 * n_pairs + 5, 16), max(0.05 * D + 2, 8)),
    )
    vmax = max(per_unit_pair.max(), 1e-6)
    im = ax.imshow(
        sorted_table, aspect="auto", cmap="magma",
        vmin=0, vmax=vmax, interpolation="nearest",
    )
    pair_labels = [f"({a},{b})" for (a, b) in pairs]
    ax.set_xticks(np.arange(n_pairs))
    ax.set_xticklabels(pair_labels, rotation=45, fontsize=8, ha="right")
    ax.set_xlabel(r"$(k_1, k_2)$")
    ax.set_yticks([0, D - 1])
    ax.set_yticklabels(
        [f"unit {row_order[0]}", f"unit {row_order[-1]}"], fontsize=8,
    )
    ax.set_ylabel("units (sorted by max bicoherence)")
    ax.set_title("Per-unit auto-bicoherence, SmolLM2 360M", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=r"$b_u$")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_z_trajectories(streams, spec, top_units, primary_pair,
                         n_samples_show, save_path):
    """For each top-bicoherence unit, plot z_l in the complex plane
    where z is the sum of fundamental and harmonic analytic signals.
    A clean fundamental gives a circle. Adding the harmonic warps it
    into peanut / figure-eight, depending on relative phase."""
    k1, k2 = primary_pair
    bins = sorted({k1, k2, k1 + k2})
    N, L, D = streams.shape
    n_units = max(len(top_units), 1)
    n_cols = (n_units + 1) // 2
    fig, axes = plt.subplots(
        2, n_cols, figsize=(3.5 * n_cols, 7.5),
    )
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[len(top_units):]:
        ax.set_visible(False)
    for ax, u in zip(axes, top_units):
        for n in range(min(n_samples_show, N)):
            z = reconstruct_z_from_bins(spec[n, u, :], bins, L)
            ax.plot(z.real, z.imag, "-", linewidth=0.9, alpha=0.55)
            ax.plot(z.real[0], z.imag[0], "o", markersize=3,
                    color="black", alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.3, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.3, alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"unit {u}", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        f"$z_\\ell$ in the complex plane: "
        f"bins {bins} only, "
        f"{n_samples_show} sample(s) per unit. "
        f"Departures from circularity = harmonic content.",
        fontsize=10, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_bispectral_phase_polar(h_dict, top_units_by_pair, save_path):
    """For each pair (k1, k2), polar histogram of the per-input
    bispectral phase phi_u^(2,n) for top-bicoherence units. A tight
    peak means the unit's harmonic phase is consistent across inputs."""
    pairs = list(h_dict.keys())
    n_pairs = len(pairs)
    n_units_per_pair = len(next(iter(top_units_by_pair.values())))
    fig, axes = plt.subplots(
        n_pairs, n_units_per_pair,
        figsize=(2.6 * n_units_per_pair, 2.8 * n_pairs),
        subplot_kw=dict(projection="polar"),
    )
    if n_pairs == 1:
        axes = axes[None, :]
    if n_units_per_pair == 1:
        axes = axes[:, None]
    for r, (k1, k2) in enumerate(pairs):
        h = h_dict[(k1, k2)]                               # (N, D)
        units = top_units_by_pair[(k1, k2)]
        for c, u in enumerate(units):
            ax = axes[r, c]
            phases = np.angle(h[:, u])
            mag = np.abs(h[:, u])
            mag_norm = mag / (mag.max() + 1e-30)
            edges = np.linspace(-np.pi, np.pi, 25)
            counts, _ = np.histogram(phases, bins=edges, weights=mag_norm)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.bar(
                centers, counts,
                width=(edges[1] - edges[0]),
                bottom=0, alpha=0.75,
            )
            mean_phase = np.angle(np.mean(np.exp(1j * phases)))
            R = np.abs(np.mean(np.exp(1j * phases)))
            ax.plot([mean_phase, mean_phase], [0, counts.max() * 1.05],
                    "-", color="red", linewidth=1.4)
            ax.set_title(
                f"({k1},{k2}) unit {u}\n"
                f"PLV={R:.2f}",
                fontsize=8, pad=10,
            )
            ax.tick_params(labelsize=6)
            ax.set_yticklabels([])
    fig.suptitle(
        r"Bispectral phase $\phi_u^{(k_1,k_2)} ="
        r" \arg(X[k_1])+\arg(X[k_2])-\arg(X[k_1+k_2])$"
        " across inputs (amplitude-weighted)",
        fontsize=10, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def _community_reorder(M_abs, top_n):
    """Reorder rows/cols of (D, D) so units with strong off-diagonal
    coupling cluster together. Quick-and-dirty: take top_n units by
    sum-of-off-diagonal-coupling, sort by first eigenvector of the
    submatrix. Returns full ordering placing those units first."""
    D = M_abs.shape[0]
    Md = M_abs - np.diag(np.diag(M_abs))
    score = np.sum(Md, axis=1)
    top = np.argsort(-score)[:top_n]
    sub = Md[np.ix_(top, top)]
    try:
        w, v = np.linalg.eigh(sub + sub.T)
        proj = v[:, -1]
        sub_order = np.argsort(proj)
        top_sorted = top[sub_order]
    except np.linalg.LinAlgError:
        top_sorted = top
    rest = np.setdiff1d(np.arange(D), top_sorted, assume_unique=False)
    return np.concatenate([top_sorted, rest]), top_sorted


def plot_cross_bicoh_matrix(M_abs, primary_pair, save_path,
                              top_n=64, null_p99=None):
    """Heatmap of |B_{AB}| at the primary pair, with units reordered."""
    order, top = _community_reorder(M_abs, top_n=top_n)
    M_show = M_abs[np.ix_(top, top)]
    np.fill_diagonal(M_show, np.nan)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="black")
    im = ax.imshow(
        M_show, cmap=cmap, vmin=0,
        vmax=max(np.nanmax(M_show), 1e-6),
        interpolation="nearest",
    )
    ax.set_title(
        f"|B_{{AB}}| at $(k_1,k_2)=({primary_pair[0]},"
        f"{primary_pair[1]})$, top {top_n} units (community-ordered)\n"
        + (f"null p99 = {null_p99:.3f}" if null_p99 is not None else ""),
        fontsize=10,
    )
    ax.set_xlabel("unit B (reordered)")
    ax.set_ylabel("unit A (reordered)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r"$|B_{AB}|$")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_cross_pair_examples(h_primary, top_pairs, primary_pair,
                              save_path, n_show=6):
    """For top inter-unit pairs, scatter phi_A^(n) vs phi_B^(n)."""
    n = min(n_show, len(top_pairs))
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6))
    if n == 1:
        axes = [axes]
    k1, k2 = primary_pair
    for ax, (uA, uB, score) in zip(axes, top_pairs[:n]):
        phA = np.angle(h_primary[:, uA])
        phB = np.angle(h_primary[:, uB])
        ax.scatter(phA, phB, s=8, alpha=0.45, edgecolors="none")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel(rf"$\phi_A^{{({k1},{k2})}}$")
        ax.set_ylabel(rf"$\phi_B^{{({k1},{k2})}}$")
        ax.set_title(
            f"u{uA} vs u{uB}\n|B|={score:.2f}", fontsize=9,
        )
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")
        for d in (-np.pi, 0, np.pi):
            ax.plot([-np.pi, np.pi], [-np.pi + d, np.pi + d],
                    "-", color="red", linewidth=0.5, alpha=0.4)
    fig.suptitle(
        f"Top cross-bicoh pairs at $(k_1,k_2)=({k1},{k2})$. "
        "Tight diagonal band = phase-locked.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_null_comparison(real_pool, null_pool, primary_pair,
                          save_path, analytical_p95=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bins = np.linspace(0, 1, 60)
    ax.hist(null_pool, bins=bins, density=True, alpha=0.55,
            color="gray", label="null (input-shuffled)")
    ax.hist(real_pool, bins=bins, density=True, alpha=0.55,
            color="C0", label="real")
    p95 = np.percentile(null_pool, 95)
    p99 = np.percentile(null_pool, 99)
    ax.axvline(p95, color="black", linestyle="--", linewidth=1.0,
               label=f"null p95 = {p95:.3f}")
    ax.axvline(p99, color="black", linestyle=":", linewidth=1.0,
               label=f"null p99 = {p99:.3f}")
    if analytical_p95 is not None:
        ax.axvline(analytical_p95, color="red", linestyle="-",
                   linewidth=0.8, alpha=0.8,
                   label=f"Rayleigh p95 = {analytical_p95:.3f}")
    ax.set_xlabel(r"$|B_{AB}|$")
    ax.set_ylabel("density")
    ax.set_title(
        f"|cross-bicoh| at $(k_1,k_2)=({primary_pair[0]},"
        f"{primary_pair[1]})$: real vs null",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_pairs(pair_strs, max_k):
    pairs = []
    for s in pair_strs:
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError(f"Bad pair: {s}")
        a, b = int(parts[0]), int(parts[1])
        if a < 1 or b < 1:
            raise ValueError(f"Pair components must be >= 1: {s}")
        if a + b > max_k:
            raise ValueError(
                f"Pair {(a, b)} sums to {a + b}, "
                f"above max bin {max_k}"
            )
        pairs.append((min(a, b), max(a, b)))
    seen = set()
    out = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str,
                        default="smollm2_cross_bispectrum")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--pairs", nargs="+",
        default=["1,1", "1,2", "2,2", "1,3", "2,3", "3,3"],
        help="Bispectral pairs (k1,k2). Whitespace-separated.",
    )
    parser.add_argument("--primary-pair", type=str, default="1,1",
                        help="Pair used for the cross-bicoh matrix and "
                             "the null comparison plot.")
    parser.add_argument("--max-bin-display", type=int, default=6,
                        help="Used only for the auto bicoh "
                             "summary heatmap extent.")
    parser.add_argument("--top-units-traj", type=int, default=8,
                        help="Number of top units to plot z trajectories "
                             "and polar histograms for.")
    parser.add_argument("--top-pairs", type=int, default=20,
                        help="How many cross-pair findings to keep.")
    parser.add_argument("--top-units-cross", type=int, default=64,
                        help="Number of units to display in cross-bicoh "
                             "matrix plot.")
    parser.add_argument("--n-shuffle-cross", type=int, default=10,
                        help="Permutations for cross-bicoh null pool.")
    parser.add_argument("--n-shuffle-auto", type=int, default=20,
                        help="Phase-randomization runs for auto-bicoh null.")
    parser.add_argument("--n-traj-samples", type=int, default=4)
    parser.add_argument("--skip-first", type=int, default=0,
                        help="Drop the first N sublayers from the FFT input. "
                             "Use to exclude embedding-dominated early layers. "
                             "FFT bin numbering is then relative to the "
                             "trimmed sublayer range.")
    parser.add_argument("--skip-last", type=int, default=0,
                        help="Drop the last N sublayers from the FFT input. "
                             "Use to exclude unembedding-dominated late "
                             "layers. FFT bin numbering is then relative to "
                             "the trimmed sublayer range.")
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
    skip_last  = args.skip_last
    if skip_first + skip_last >= L_full:
        raise ValueError(
            f"skip_first ({skip_first}) + skip_last ({skip_last}) "
            f">= n_sublayers ({L_full}); nothing left to analyze."
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
        if streams.shape[1] < 8:
            print(
                f"  WARNING: only {streams.shape[1]} sublayers remain; "
                f"FFT bin resolution will be limited."
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    N, L, D = streams.shape
    K = L // 2 + 1
    max_pair_sum = K - 1

    pairs = parse_pairs(args.pairs, max_pair_sum)
    primary_pair = parse_pairs([args.primary_pair], max_pair_sum)[0]
    if primary_pair not in pairs:
        pairs = [primary_pair] + pairs
    print(f"\nBispectral pairs: {pairs}")
    print(f"Primary pair: {primary_pair}")
    print(f"FFT length L={L}, max valid k1+k2 = {max_pair_sum}")

    # ------------- spectrum -----------------
    print("\nComputing spectrum ...")
    spec = streams_to_spec(streams)                          # (N, D, K)
    print(f"  spec shape (complex): {spec.shape}")

    # ------------- per-pair bispectral indicators -----------------
    print("\nComputing per-pair bispectral indicators ...")
    h_dict = {}
    for (k1, k2) in pairs:
        h_dict[(k1, k2)] = bispectral_indicator(spec, k1, k2)  # (N, D)

    # ------------- auto-bicoherence per unit per pair -----------------
    print("Computing auto-bicoherence per unit ...")
    per_unit = np.zeros((D, len(pairs)), dtype=np.float64)
    auto_plv_pu = np.zeros((D, len(pairs)), dtype=np.float64)
    auto_phase_pu = np.zeros((D, len(pairs)), dtype=np.float64)
    for j, (k1, k2) in enumerate(pairs):
        per_unit[:, j] = auto_bicoherence(spec, k1, k2)
        auto_plv_pu[:, j] = auto_plv(h_dict[(k1, k2)])
        auto_phase_pu[:, j] = auto_resultant_phase(h_dict[(k1, k2)])

    # auto-bicoh null on the primary pair
    j_primary = pairs.index(primary_pair)
    print(f"\nAuto-bicoherence null at {primary_pair} "
          f"({args.n_shuffle_auto} shuffles) ...")
    auto_null = auto_bicoh_null_pool(
        spec, primary_pair[0], primary_pair[1],
        args.n_shuffle_auto, rng,
    )                                                        # (R, D)
    auto_null_p95 = float(np.percentile(auto_null, 95))
    auto_null_p99 = float(np.percentile(auto_null, 99))
    print(f"  auto null p95: {auto_null_p95:.4f}")
    print(f"  auto null p99: {auto_null_p99:.4f}")
    print(f"  real max bicoh: {per_unit[:, j_primary].max():.4f}")

    # ------------- cross-bicoherence at primary pair -----------------
    print(f"\nComputing cross-bicoherence matrix at {primary_pair} ...")
    h_primary = h_dict[primary_pair]
    M_complex = cross_bicoherence_matrix(h_primary)          # (D, D) complex
    M_abs = np.abs(M_complex).astype(np.float32)
    np.fill_diagonal(M_abs, 0.0)

    # cross PLV (phase-only) for sanity on top pairs
    M_plv = cross_plv_matrix(h_primary)
    np.fill_diagonal(M_plv, 0.0)

    # null pool
    print(f"Computing null cross-bicoh pool "
          f"({args.n_shuffle_cross} shuffles) ...")
    null_pool = cross_bicoh_null_pool(
        h_primary, args.n_shuffle_cross, np.random.default_rng(args.seed + 7),
    )
    null_p95 = float(np.percentile(null_pool, 95))
    null_p99 = float(np.percentile(null_pool, 99))
    null_p999 = float(np.percentile(null_pool, 99.9))
    analytical_p95 = float(np.sqrt(-np.log(0.05) / N))
    analytical_p99 = float(np.sqrt(-np.log(0.01) / N))
    print(f"  null p95   = {null_p95:.4f}  "
          f"(Rayleigh p95 = {analytical_p95:.4f})")
    print(f"  null p99   = {null_p99:.4f}  "
          f"(Rayleigh p99 = {analytical_p99:.4f})")
    print(f"  real off-diag max |B_{{AB}}| = {M_abs.max():.4f}")

    # real pool of off-diagonal magnitudes for histogram
    iu = np.triu_indices(D, k=1)
    real_pool = M_abs[iu]

    # ------------- top-pair findings -----------------
    flat_idx = np.argsort(-real_pool)[:args.top_pairs]
    top_pairs = []
    for idx in flat_idx:
        a = iu[0][idx]
        b = iu[1][idx]
        top_pairs.append((int(a), int(b), float(real_pool[idx])))

    # ------------- top units for trajectory and polar plots -----------------
    top_units_primary = list(np.argsort(-per_unit[:, j_primary])
                             [:args.top_units_traj])
    top_units_by_pair = {}
    for j, p in enumerate(pairs):
        top_units_by_pair[p] = list(np.argsort(-per_unit[:, j])
                                    [:max(4, args.top_units_traj // 2)])

    # ------------- plots -----------------
    print("\nMaking plots ...")
    plot_bicoh_auto_summary(
        per_unit, pairs, max(args.max_bin_display, max(p[1] for p in pairs)),
        out_dir / "bicoh_auto_summary.png",
    )
    plot_bicoh_auto_per_unit(
        per_unit, pairs, out_dir / "bicoh_auto_per_unit.png",
    )
    plot_z_trajectories(
        streams, spec, top_units_primary,
        primary_pair, args.n_traj_samples,
        out_dir / "z_trajectory_top_units.png",
    )
    plot_bispectral_phase_polar(
        h_dict, top_units_by_pair,
        out_dir / "bispectral_phase_polar.png",
    )
    plot_cross_bicoh_matrix(
        M_abs, primary_pair,
        out_dir / "cross_bicoh_matrix.png",
        top_n=args.top_units_cross, null_p99=null_p99,
    )
    plot_cross_pair_examples(
        h_primary, top_pairs, primary_pair,
        out_dir / "cross_pair_examples.png",
        n_show=min(6, len(top_pairs)),
    )
    plot_null_comparison(
        real_pool, null_pool, primary_pair,
        out_dir / "null_comparison.png",
        analytical_p95=analytical_p95,
    )

    # ------------- save arrays -----------------
    np.savez(
        out_dir / "cross_bispectrum_data.npz",
        per_unit=per_unit,
        auto_plv_pu=auto_plv_pu,
        auto_phase_pu=auto_phase_pu,
        pairs=np.array(pairs),
        primary_pair=np.array(primary_pair),
        M_abs=M_abs,
        M_plv=M_plv,
        # only complex data we save: full cross-bicoh at primary
        M_complex_real=M_complex.real.astype(np.float32),
        M_complex_imag=M_complex.imag.astype(np.float32),
        null_pool=null_pool.astype(np.float32),
        auto_null_pool=auto_null.astype(np.float32),
    )

    # ------------- summary -----------------
    summary = {
        "model":          MODEL_NAME,
        "n_samples":      int(N),
        "n_sublayers":    int(L),
        "n_sublayers_full": int(L_full),
        "skip_first":     int(skip_first),
        "skip_last":      int(skip_last),
        "d_model":        int(D),
        "fft_length_L":   int(L),
        "K_positive_freqs": int(K),
        "pairs":          [list(p) for p in pairs],
        "primary_pair":   list(primary_pair),
        "auto_bicoh": {
            "per_pair_means": {
                f"{a},{b}": float(per_unit[:, j].mean())
                for j, (a, b) in enumerate(pairs)
            },
            "per_pair_max": {
                f"{a},{b}": float(per_unit[:, j].max())
                for j, (a, b) in enumerate(pairs)
            },
            "primary_pair_null_p95": auto_null_p95,
            "primary_pair_null_p99": auto_null_p99,
            "n_units_above_null_p99_at_primary": int(
                (per_unit[:, j_primary] > auto_null_p99).sum()
            ),
        },
        "cross_bicoh_at_primary": {
            "real_max": float(M_abs.max()),
            "real_p95_offdiag": float(np.percentile(real_pool, 95)),
            "real_p99_offdiag": float(np.percentile(real_pool, 99)),
            "null_p95": null_p95,
            "null_p99": null_p99,
            "null_p999": null_p999,
            "analytical_rayleigh_p95": analytical_p95,
            "analytical_rayleigh_p99": analytical_p99,
            "n_pairs_above_null_p99": int(
                (real_pool > null_p99).sum()
            ),
            "n_pairs_above_null_p999": int(
                (real_pool > null_p999).sum()
            ),
            "fraction_pairs_above_null_p99": float(
                (real_pool > null_p99).mean()
            ),
        },
        "top_unit_pairs_at_primary": [
            {
                "unit_A": uA, "unit_B": uB,
                "abs_cross_bicoh": s,
                "cross_plv_phase_only": float(M_plv[uA, uB]),
                "auto_bicoh_A": float(per_unit[uA, j_primary]),
                "auto_bicoh_B": float(per_unit[uB, j_primary]),
                "ratio_to_null_p99": s / max(null_p99, 1e-6),
            }
            for (uA, uB, s) in top_pairs
        ],
        "top_auto_bicoh_units_per_pair": {
            f"{a},{b}": [
                {
                    "unit": int(u),
                    "bicoh": float(per_unit[u, j]),
                    "phase_locking_value": float(auto_plv_pu[u, j]),
                    "resultant_phase_radians": float(auto_phase_pu[u, j]),
                }
                for u in np.argsort(-per_unit[:, j])[:10]
            ]
            for j, (a, b) in enumerate(pairs)
        },
    }
    with open(out_dir / "cross_bispectrum_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------- console output -----------------
    print("\n--- Top 15 cross-bispectrum unit pairs at primary "
          f"({primary_pair[0]},{primary_pair[1]}) ---")
    print(f"{'rank':<5} {'A':<6} {'B':<6} "
          f"{'|B_AB|':>8} {'PLV':>8} "
          f"{'b_A':>7} {'b_B':>7} {'ratio_p99':>10}")
    for r, (uA, uB, s) in enumerate(top_pairs[:15], 1):
        ratio = s / max(null_p99, 1e-6)
        plv = float(M_plv[uA, uB])
        print(
            f"{r:<5} u{uA:<5} u{uB:<5} "
            f"{s:>8.4f} {plv:>8.4f} "
            f"{per_unit[uA, j_primary]:>7.3f} "
            f"{per_unit[uB, j_primary]:>7.3f} "
            f"{ratio:>10.2f}"
        )

    print("\n--- Top 10 auto-bicoherence units at primary pair ---")
    print(f"{'rank':<5} {'unit':<6} {'b_u':>8} {'PLV':>8} {'phi(rad)':>10}")
    top_auto = np.argsort(-per_unit[:, j_primary])[:10]
    for r, u in enumerate(top_auto, 1):
        print(
            f"{r:<5} u{u:<5} "
            f"{per_unit[u, j_primary]:>8.4f} "
            f"{auto_plv_pu[u, j_primary]:>8.4f} "
            f"{auto_phase_pu[u, j_primary]:>+10.3f}"
        )

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
