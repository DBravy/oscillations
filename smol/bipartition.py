"""
Test whether the real graph's community structure is a true bipartition,
and whether the two main communities differ in per-unit features.

Two analyses:
  1. Bipartition test: for the two largest communities A and B in the
     real graph, count edges within-A, within-B, and between A and B.
     A true bipartition has most edges between groups.
  2. Per-unit feature comparison: do A-units and B-units differ
     systematically in dominant frequency, loudness, or phase at
     the dominant FFT bin?

Outputs in bipartition_out_<model>/:
  - edge_breakdown.png         within-A, within-B, A-to-B edge counts
                               vs nulls
  - feature_comparison.png     per-unit feature distributions for
                               group A vs group B vs unassigned
  - phase_at_dom_bin.png       polar scatter of per-unit phases at
                               their own dominant FFT bin, colored
                               by group membership
  - bipartition_summary.json
"""

import argparse
import json
import re
from collections import Counter
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
N_PAIRS = 200_000
QUAD_BAND = np.pi / 6
HIGH_MAG_QUANTILE = 0.90
N_COMMUNITY_RUNS = 5  # multiple label-prop runs to check stability


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


def per_unit_spectra(streams):
    ms = streams - streams.mean(axis=1, keepdims=True)
    return np.fft.rfft(ms, axis=1)


def dominant_freq_per_unit(spectra):
    power = np.mean(np.abs(spectra) ** 2, axis=0)
    if power.shape[0] <= 1:
        return np.zeros(power.shape[1], dtype=int)
    return np.argmax(power[1:], axis=0) + 1


def per_unit_phase_at_dom_bin(spectra, dom_freq):
    """
    For each unit, compute the complex Fourier coefficient at its own
    dominant frequency, averaged across samples. Returns its angle.
    This is the per-unit "phase" relative to a global zero, NOT a
    pairwise phase difference.
    """
    n_samples, _, d_model = spectra.shape
    coeffs = np.zeros(d_model, dtype=complex)
    for s in range(n_samples):
        spec_s = spectra[s]
        for u in range(d_model):
            coeffs[u] += spec_s[dom_freq[u], u]
    coeffs /= n_samples
    return np.angle(coeffs), np.abs(coeffs)


def sample_pair_indices(d_model, n_pairs, rng):
    a = rng.integers(0, d_model, size=n_pairs * 2)
    b = rng.integers(0, d_model, size=n_pairs * 2)
    mask = a != b
    a = a[mask][:n_pairs]
    b = b[mask][:n_pairs]
    return np.minimum(a, b), np.maximum(a, b)


def trace_correlations(streams, u_idx, v_idx):
    n_samples = streams.shape[0]
    out = np.zeros(len(u_idx))
    for s in range(n_samples):
        sample = streams[s]
        sample_centered = sample - sample.mean(axis=0, keepdims=True)
        sample_std = sample_centered.std(axis=0) + 1e-8
        u_traces = sample_centered[:, u_idx] / sample_std[u_idx]
        v_traces = sample_centered[:, v_idx] / sample_std[v_idx]
        out += np.mean(u_traces * v_traces, axis=0)
    return out / n_samples


def pair_cross_spectral(spectra, freq_per_pair, u_idx, v_idx):
    n_samples = spectra.shape[0]
    coeffs = np.zeros(len(u_idx), dtype=complex)
    for s in range(n_samples):
        spec_s = spectra[s]
        u_vals = spec_s[freq_per_pair, u_idx]
        v_vals = spec_s[freq_per_pair, v_idx]
        coeffs += u_vals * np.conj(v_vals)
    coeffs /= n_samples
    return np.angle(coeffs), np.abs(coeffs)


# ---------------------------------------------------------------------------
# Graph + community detection
# ---------------------------------------------------------------------------

def build_adjacency(u_sel, v_sel, d_model):
    adj = {i: set() for i in range(d_model)}
    for u, v in zip(u_sel, v_sel):
        u, v = int(u), int(v)
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def label_propagation_communities(adj, d_model, max_iter=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    labels = np.arange(d_model)
    nodes = np.arange(d_model)
    for _ in range(max_iter):
        rng.shuffle(nodes)
        changed = 0
        for i in nodes:
            nbrs = adj[i]
            if not nbrs:
                continue
            cnt = Counter(labels[list(nbrs)])
            max_count = max(cnt.values())
            best = [lbl for lbl, c in cnt.items() if c == max_count]
            new_label = rng.choice(best)
            if new_label != labels[i]:
                labels[i] = new_label
                changed += 1
        if changed == 0:
            break
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def assign_AB_groups(labels):
    """
    Identify the two largest communities and assign each unit to A, B,
    or 'other' (singletons or smaller communities).
    """
    sizes = np.bincount(labels)
    top_two = np.argsort(sizes)[-2:][::-1]  # largest first
    a_label, b_label = int(top_two[0]), int(top_two[1])
    group = np.full_like(labels, fill_value=-1)
    group[labels == a_label] = 0  # A
    group[labels == b_label] = 1  # B
    return group, sizes[a_label], sizes[b_label]


def edge_breakdown(u_sel, v_sel, group):
    """
    Count edges within A, within B, and between A and B (and other).
    Returns dict.
    """
    counts = {"within_A": 0, "within_B": 0, "between_AB": 0,
              "involving_other": 0}
    for u, v in zip(u_sel, v_sel):
        gu, gv = int(group[u]), int(group[v])
        if gu == -1 or gv == -1:
            counts["involving_other"] += 1
        elif gu == gv == 0:
            counts["within_A"] += 1
        elif gu == gv == 1:
            counts["within_B"] += 1
        else:
            counts["between_AB"] += 1
    return counts


def expected_edge_breakdown(n_edges, group, d_model):
    """
    Under uniform random pair selection (with no self-loops and only
    units that are in A or B counted), what fraction of pairs would
    be within-A, within-B, between-AB?

    n_A = number of units in A, n_B = number of units in B,
    total relevant pairs C(n_A + n_B, 2).
    """
    n_A = int((group == 0).sum())
    n_B = int((group == 1).sum())
    n_total = n_A + n_B
    if n_total < 2:
        return None
    total_pairs = n_total * (n_total - 1) / 2
    aa = n_A * (n_A - 1) / 2 / total_pairs
    bb = n_B * (n_B - 1) / 2 / total_pairs
    ab = n_A * n_B / total_pairs
    return {"within_A_expected_frac": float(aa),
            "within_B_expected_frac": float(bb),
            "between_AB_expected_frac": float(ab)}


def community_stability(adj, d_model, n_runs):
    """
    Run label propagation multiple times with different seeds.
    For each run, identify the two largest communities. Then compute
    pairwise overlap (as a fraction of unit assignments matching)
    between runs.
    """
    runs = []
    for r in range(n_runs):
        labels = label_propagation_communities(
            adj, d_model, rng=np.random.default_rng(SEED + r * 7919)
        )
        group, _, _ = assign_AB_groups(labels)
        runs.append(group)

    n = len(runs)
    overlap_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mask = (runs[i] != -1) & (runs[j] != -1)
            if mask.sum() == 0:
                overlap_matrix[i, j] = np.nan
                continue
            same = (runs[i][mask] == runs[j][mask]).mean()
            # Also try with B/A swapped
            swapped = (runs[i][mask] == 1 - runs[j][mask]).mean()
            overlap_matrix[i, j] = max(same, swapped)
    return runs, overlap_matrix


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_edge_breakdown(counts, expected, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    total_in_AB = (counts["within_A"] + counts["within_B"] +
                   counts["between_AB"])
    if total_in_AB == 0:
        ax.text(0.5, 0.5, "No edges in A or B", ha="center")
    else:
        observed = {
            "within_A": counts["within_A"] / total_in_AB,
            "within_B": counts["within_B"] / total_in_AB,
            "between_AB": counts["between_AB"] / total_in_AB,
        }
        labels = ["within_A", "within_B", "between_AB"]
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2,
               [observed[k] for k in labels],
               width, label="observed")
        ax.bar(x + width/2,
               [expected[f"{k}_expected_frac"] for k in labels],
               width, label="expected (random within A∪B)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("fraction of edges")
        ax.set_title("Edge distribution: real graph vs random null "
                     "(restricted to A ∪ B)")
        ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_feature_comparison(group, dom_freq, mean_abs, per_unit_phase,
                            save_path=None):
    """
    For each per-unit feature, plot the distribution split by group A,
    group B, and other.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, values, title, kwargs in [
        (axes[0], dom_freq, "Dominant FFT bin",
         dict(bins=np.arange(0, dom_freq.max() + 2))),
        (axes[1], mean_abs, "Mean |activation| (log)",
         dict(bins=np.logspace(
             np.log10(max(mean_abs.min(), 1e-8)),
             np.log10(mean_abs.max() + 1e-12), 50,
         ))),
        (axes[2], per_unit_phase, "Per-unit phase at dom bin",
         dict(bins=np.linspace(-np.pi, np.pi, 50))),
    ]:
        for label, mask, color in [
            ("A", group == 0, "C0"),
            ("B", group == 1, "C1"),
            ("other", group == -1, "C2"),
        ]:
            if mask.sum() == 0:
                continue
            ax.hist(values[mask], histtype="step", linewidth=1.5,
                    density=True, label=f"{label} (n={mask.sum()})",
                    color=color, **kwargs)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        if title.startswith("Mean"):
            ax.set_xscale("log")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_polar_phases(group, per_unit_phase, per_unit_mag,
                      save_path=None):
    """
    Polar scatter: each unit at angle = its phase, radius = magnitude
    of its FFT coefficient at its own dominant bin. Colored by group.
    """
    fig, ax = plt.subplots(figsize=(7, 7),
                            subplot_kw={"projection": "polar"})
    for label, mask, color in [
        ("other", group == -1, "lightgray"),
        ("A", group == 0, "C0"),
        ("B", group == 1, "C1"),
    ]:
        if mask.sum() == 0:
            continue
        ax.scatter(per_unit_phase[mask], per_unit_mag[mask],
                   s=8, alpha=0.5, color=color, label=label)
    ax.set_title("Per-unit phase / magnitude at dominant bin")
    ax.legend(loc="upper right", fontsize=9)
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
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    args = parser.parse_args()

    out_dir = Path(f"bipartition_out_{slugify(args.model)}")
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
    norms = np.linalg.norm(streams, axis=2, keepdims=True)
    streams_n = streams / np.maximum(norms, 1e-8)
    mean_abs = np.mean(np.abs(streams_n), axis=(0, 1))

    print("Spectra and dominant freqs ...")
    spectra = per_unit_spectra(streams_n)
    dom_freq = dominant_freq_per_unit(spectra)
    per_unit_phase, per_unit_mag = per_unit_phase_at_dom_bin(
        spectra, dom_freq
    )

    print(f"Sampling {args.n_pairs} pairs ...")
    u_idx, v_idx = sample_pair_indices(d_model, args.n_pairs, rng)

    print("Trace correlations ...")
    trace_corr = trace_correlations(streams_n, u_idx, v_idx)

    print("Cross-spectral phase / mag ...")
    phase, mag = pair_cross_spectral(spectra, dom_freq[u_idx], u_idx, v_idx)

    prox = 1.0 - np.abs(trace_corr)
    edges = np.quantile(prox, [0, 1/3, 2/3, 1])
    furthest_mask = prox > edges[2]

    p_furthest = phase[furthest_mask]
    m_furthest = mag[furthest_mask]
    u_furthest = u_idx[furthest_mask]
    v_furthest = v_idx[furthest_mask]

    cutoff = np.quantile(m_furthest, HIGH_MAG_QUANTILE)
    high_mag = m_furthest >= cutoff
    near_quad = (
        (np.abs(p_furthest - np.pi / 2) < QUAD_BAND) |
        (np.abs(p_furthest + np.pi / 2) < QUAD_BAND)
    )
    selected = high_mag & near_quad
    u_sel = u_furthest[selected]
    v_sel = v_furthest[selected]
    n_edges = int(selected.sum())
    print(f"Selected {n_edges} edges")

    print("Building graph and detecting communities ...")
    adj_real = build_adjacency(u_sel, v_sel, d_model)

    # Stability check
    print(f"Running label propagation {N_COMMUNITY_RUNS} times ...")
    runs, overlap_matrix = community_stability(
        adj_real, d_model, N_COMMUNITY_RUNS
    )
    # Use the first run for the main analysis (after relabeling, the
    # results are equivalent up to A/B label swaps)
    group = runs[0]
    n_A = int((group == 0).sum())
    n_B = int((group == 1).sum())
    n_other = int((group == -1).sum())
    print(f"Group sizes: A={n_A}, B={n_B}, other={n_other}")
    print(f"Cross-run overlap matrix:\n{overlap_matrix}")

    print("Counting edges by group ...")
    counts = edge_breakdown(u_sel, v_sel, group)
    expected = expected_edge_breakdown(n_edges, group, d_model)
    print(f"Edge counts: {counts}")
    print(f"Expected fractions (random within A ∪ B): {expected}")

    print("Saving plots ...")
    plot_edge_breakdown(
        counts, expected,
        save_path=out_dir / "edge_breakdown.png",
    )
    plot_feature_comparison(
        group, dom_freq, mean_abs, per_unit_phase,
        save_path=out_dir / "feature_comparison.png",
    )
    plot_polar_phases(
        group, per_unit_phase, per_unit_mag,
        save_path=out_dir / "phase_at_dom_bin.png",
    )

    # Per-feature group comparisons (numeric)
    def feature_stats(group, values, label):
        out = {}
        for g_label, g_value in [("A", 0), ("B", 1), ("other", -1)]:
            mask = group == g_value
            if mask.sum() == 0:
                continue
            v = values[mask]
            out[g_label] = {
                "n": int(mask.sum()),
                "mean": float(v.mean()),
                "median": float(np.median(v)),
                "std": float(v.std()),
            }
        return {label: out}

    summary = {
        "model": args.model,
        "d_model": int(d_model),
        "n_edges": int(n_edges),
        "groups": {"A": n_A, "B": n_B, "other": n_other},
        "edge_breakdown": counts,
        "expected_breakdown": expected,
        "edge_breakdown_observed_fracs": {
            "within_A": (
                counts["within_A"] / max(
                    counts["within_A"] + counts["within_B"] +
                    counts["between_AB"], 1
                )
            ),
            "within_B": (
                counts["within_B"] / max(
                    counts["within_A"] + counts["within_B"] +
                    counts["between_AB"], 1
                )
            ),
            "between_AB": (
                counts["between_AB"] / max(
                    counts["within_A"] + counts["within_B"] +
                    counts["between_AB"], 1
                )
            ),
        },
        "community_stability": {
            "n_runs": N_COMMUNITY_RUNS,
            "min_pairwise_overlap": float(
                np.nanmin(
                    overlap_matrix[np.triu_indices(N_COMMUNITY_RUNS,
                                                   k=1)]
                )
            ) if N_COMMUNITY_RUNS > 1 else None,
            "mean_pairwise_overlap": float(
                np.nanmean(
                    overlap_matrix[np.triu_indices(N_COMMUNITY_RUNS,
                                                   k=1)]
                )
            ) if N_COMMUNITY_RUNS > 1 else None,
        },
        "feature_stats": {
            **feature_stats(group, dom_freq, "dominant_freq_bin"),
            **feature_stats(group, mean_abs, "mean_abs_activation"),
            **feature_stats(group, per_unit_phase,
                            "per_unit_phase_at_dom_bin"),
        },
    }

    # Bipartiteness scoring: 0 = perfectly bipartite (no within-group),
    # 1 = perfectly within-group. Use observed within-AB / total.
    total_AB = (counts["within_A"] + counts["within_B"] +
                counts["between_AB"])
    if total_AB > 0:
        within_frac = (counts["within_A"] + counts["within_B"]) / total_AB
        between_frac = counts["between_AB"] / total_AB
        summary["bipartite_score"] = {
            "within_AB_fraction": float(within_frac),
            "between_AB_fraction": float(between_frac),
            "expected_within_AB_fraction": (
                expected["within_A_expected_frac"] +
                expected["within_B_expected_frac"]
            ),
            "expected_between_AB_fraction":
                expected["between_AB_expected_frac"],
        }

    with open(out_dir / "bipartition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(f"Groups: A={n_A}, B={n_B}, other={n_other}")
    print(f"Cross-run community overlap: "
          f"min={summary['community_stability']['min_pairwise_overlap']}, "
          f"mean={summary['community_stability']['mean_pairwise_overlap']}")
    print(f"Edge breakdown: {counts}")
    if "bipartite_score" in summary:
        b = summary["bipartite_score"]
        print(f"Within A∪B edges: "
              f"between={b['between_AB_fraction']:.3f} "
              f"(expected {b['expected_between_AB_fraction']:.3f}), "
              f"within={b['within_AB_fraction']:.3f} "
              f"(expected {b['expected_within_AB_fraction']:.3f})")
    print(f"\nFeature stats by group:")
    for feat, stats in summary["feature_stats"].items():
        print(f"  {feat}:")
        for g_label, g_stats in stats.items():
            print(f"    {g_label}: mean={g_stats['mean']:.3f}, "
                  f"median={g_stats['median']:.3f}, n={g_stats['n']}")
    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()