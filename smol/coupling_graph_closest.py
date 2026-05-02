"""
Apply the same coupling-graph analysis to the closest regime.

In the closest regime, the phase-difference distribution peaks at 0
and at ±π. We select pairs near each peak separately and analyze the
resulting graphs:
  - "in-phase" graph: pairs with high cospec magnitude near phase 0
  - "anti-phase" graph: pairs with high cospec magnitude near phase ±π
  - "combined" graph: union of the two

For each graph we compute the same statistics as before:
degree distribution, clustering, hub subgraph density, communities,
and we compare to two null models (uniform and loudness-weighted).

Outputs in closest_graph_out_<model>/:
  - degree_distribution_in.png / _anti.png
  - hub_subgraph.png
  - community_sizes.png
  - edge_breakdown.png
  - feature_comparison.png
  - closest_graph_summary.json
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
PHASE_BAND = np.pi / 6
HIGH_MAG_QUANTILE = 0.90
N_COMMUNITY_RUNS = 5


# ---------------------------------------------------------------------------
# Boilerplate (same as before)
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
# Graph machinery (same as before)
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


def degree_array(adj, d_model):
    return np.array([len(adj[i]) for i in range(d_model)], dtype=int)


def clustering_coefficient(adj, d_model):
    cc = np.full(d_model, np.nan)
    for i in range(d_model):
        nbrs = adj[i]
        k = len(nbrs)
        if k < 2:
            continue
        nbrs_list = list(nbrs)
        edges_count = 0
        for j_idx in range(len(nbrs_list)):
            for k_idx in range(j_idx + 1, len(nbrs_list)):
                if nbrs_list[k_idx] in adj[nbrs_list[j_idx]]:
                    edges_count += 1
        possible = k * (k - 1) / 2
        cc[i] = edges_count / possible
    return cc


def hub_internal_density(adj, d_model, top_k):
    deg = degree_array(adj, d_model)
    top_nodes = np.argsort(deg)[-top_k:]
    top_set = set(int(x) for x in top_nodes)
    edges_in_top = 0
    for u in top_nodes:
        u = int(u)
        for v in adj[u]:
            if v in top_set and v > u:
                edges_in_top += 1
    possible_in_top = top_k * (top_k - 1) / 2
    density_in_top = edges_in_top / possible_in_top if possible_in_top else 0.0
    total_edges = sum(len(adj[i]) for i in range(d_model)) / 2
    possible_total = d_model * (d_model - 1) / 2
    density_overall = total_edges / possible_total if possible_total else 0.0
    return {
        "top_k": int(top_k),
        "edges_in_top": int(edges_in_top),
        "density_in_top": float(density_in_top),
        "density_overall": float(density_overall),
        "ratio": (
            float(density_in_top / density_overall)
            if density_overall > 0 else float("inf")
        ),
        "top_nodes": [int(x) for x in top_nodes],
    }


def connected_components(adj, d_model):
    seen = [False] * d_model
    comps = []
    for start in range(d_model):
        if seen[start]:
            continue
        comp = set()
        stack = [start]
        while stack:
            x = stack.pop()
            if seen[x]:
                continue
            seen[x] = True
            comp.add(x)
            for nb in adj[x]:
                if not seen[nb]:
                    stack.append(nb)
        comps.append(comp)
    return comps


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
    sizes = np.bincount(labels)
    if len(sizes) < 2:
        return np.full_like(labels, -1), 0, 0
    top_two = np.argsort(sizes)[-2:][::-1]
    a_label, b_label = int(top_two[0]), int(top_two[1])
    group = np.full_like(labels, fill_value=-1)
    group[labels == a_label] = 0
    group[labels == b_label] = 1
    return group, sizes[a_label], sizes[b_label]


def edge_breakdown(u_sel, v_sel, group):
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


def expected_edge_breakdown(group):
    n_A = int((group == 0).sum())
    n_B = int((group == 1).sum())
    n_total = n_A + n_B
    if n_total < 2:
        return None
    total_pairs = n_total * (n_total - 1) / 2
    return {
        "within_A_expected_frac": n_A * (n_A - 1) / 2 / total_pairs,
        "within_B_expected_frac": n_B * (n_B - 1) / 2 / total_pairs,
        "between_AB_expected_frac": n_A * n_B / total_pairs,
    }


def null_uniform_pairs(n_edges, d_model, rng):
    a = rng.integers(0, d_model, size=n_edges * 3)
    b = rng.integers(0, d_model, size=n_edges * 3)
    mask = a != b
    a = a[mask][:n_edges]
    b = b[mask][:n_edges]
    return np.minimum(a, b), np.maximum(a, b)


def null_weighted_pairs(n_edges, weights, rng):
    p = weights / weights.sum()
    d = len(weights)
    a = rng.choice(d, size=n_edges * 3, p=p)
    b = rng.choice(d, size=n_edges * 3, p=p)
    mask = a != b
    a = a[mask][:n_edges]
    b = b[mask][:n_edges]
    return np.minimum(a, b), np.maximum(a, b)


def overlap_score(group_a, group_b):
    """Best-match overlap between two community labelings."""
    mask = (group_a != -1) & (group_b != -1)
    if mask.sum() == 0:
        return np.nan
    same = (group_a[mask] == group_b[mask]).mean()
    swapped = (group_a[mask] == 1 - group_b[mask]).mean()
    return max(same, swapped)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_degree(deg_real, deg_uni, deg_wt, title, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, max(deg_real.max(), deg_uni.max(),
                             deg_wt.max()) + 2)
    for label, d in [("real", deg_real),
                     ("null: uniform", deg_uni),
                     ("null: loudness-weighted", deg_wt)]:
        ax.hist(d, bins=bins, histtype="step", linewidth=1.5,
                density=True, label=label)
    ax.set_xlabel("degree")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_hub_bars(stats_list, labels, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.25
    densities_top = [s["density_in_top"] for s in stats_list]
    densities_all = [s["density_overall"] for s in stats_list]
    ratios = [s["ratio"] for s in stats_list]
    ax.bar(x - width, densities_top, width, label="top-k subgraph density")
    ax.bar(x, densities_all, width, label="overall density")
    ax2 = ax.twinx()
    ax2.bar(x + width, ratios, width, label="ratio (right axis)",
            color="C2", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("edge density")
    ax2.set_ylabel("ratio (top-k / overall)")
    ax.set_title("Hub subgraph density across graphs and nulls")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_community_sizes(labels_dict, save_path=None):
    fig, axes = plt.subplots(1, len(labels_dict), figsize=(15, 4.5),
                             sharey=True)
    if len(labels_dict) == 1:
        axes = [axes]
    for ax, (name, labs) in zip(axes, labels_dict.items()):
        sizes = np.bincount(labs)
        sizes = np.sort(sizes)[::-1]
        ax.bar(np.arange(min(20, len(sizes))), sizes[:20])
        ax.set_yscale("log")
        ax.set_xlabel("community rank")
        ax.set_title(f"{name}\n({len(sizes)} communities)")
    axes[0].set_ylabel("community size")
    fig.suptitle("Community sizes (top 20 by rank)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_edge_breakdown(counts_dict, expected_dict, save_path=None):
    """One panel per graph, comparing observed vs expected breakdown."""
    fig, axes = plt.subplots(1, len(counts_dict), figsize=(15, 5),
                             sharey=True)
    if len(counts_dict) == 1:
        axes = [axes]
    for ax, (name, counts) in zip(axes, counts_dict.items()):
        expected = expected_dict[name]
        if expected is None:
            ax.set_title(f"{name}: no AB structure")
            continue
        total_in_AB = (counts["within_A"] + counts["within_B"] +
                       counts["between_AB"])
        if total_in_AB == 0:
            ax.set_title(f"{name}: empty AB")
            continue
        observed = {
            "within_A": counts["within_A"] / total_in_AB,
            "within_B": counts["within_B"] / total_in_AB,
            "between_AB": counts["between_AB"] / total_in_AB,
        }
        keys = ["within_A", "within_B", "between_AB"]
        x = np.arange(len(keys))
        width = 0.35
        ax.bar(x - width/2, [observed[k] for k in keys], width,
               label="observed")
        ax.bar(x + width/2,
               [expected[f"{k}_expected_frac"] for k in keys], width,
               label="expected (random)")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=20)
        ax.set_title(name)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("fraction of edges")
    fig.suptitle("Within / between edge fractions")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_feature_comparison(group_dict, dom_freq, mean_abs,
                            per_unit_phase, save_path=None):
    """For each graph variant, plot per-feature distributions."""
    fig, axes = plt.subplots(len(group_dict), 3,
                             figsize=(15, 3.5 * len(group_dict)))
    if len(group_dict) == 1:
        axes = axes.reshape(1, -1)
    for row, (name, group) in enumerate(group_dict.items()):
        feature_specs = [
            (dom_freq, "Dominant FFT bin",
             dict(bins=np.arange(0, dom_freq.max() + 2))),
            (mean_abs, "Mean |activation| (log)",
             dict(bins=np.logspace(
                 np.log10(max(mean_abs.min(), 1e-8)),
                 np.log10(mean_abs.max() + 1e-12), 50,
             ))),
            (per_unit_phase, "Per-unit phase at dom bin",
             dict(bins=np.linspace(-np.pi, np.pi, 50))),
        ]
        for col, (vals, title, kwargs) in enumerate(feature_specs):
            ax = axes[row, col]
            for label, mask, color in [
                ("A", group == 0, "C0"),
                ("B", group == 1, "C1"),
                ("other", group == -1, "C2"),
            ]:
                if mask.sum() == 0:
                    continue
                ax.hist(vals[mask], histtype="step", linewidth=1.5,
                        density=True, color=color,
                        label=f"{label} (n={mask.sum()})", **kwargs)
            if col == 0:
                ax.set_ylabel(name, fontsize=10)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7)
            if "log" in title:
                ax.set_xscale("log")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def select_pairs(phase, mag, mask, target_phase, band, mag_quantile):
    """Select pairs in `mask` whose phase is within `band` of
    `target_phase` (with circular distance), and whose magnitude is
    in the top fraction. target_phase=None means take all phases."""
    p = phase[mask]
    m = mag[mask]
    cutoff = np.quantile(m, mag_quantile)
    high_mag = m >= cutoff
    if target_phase is None:
        phase_ok = np.ones_like(high_mag, dtype=bool)
    elif target_phase == "anti":
        phase_ok = (np.abs(p - np.pi) < band) | (np.abs(p + np.pi) < band)
    else:
        phase_ok = np.abs(p - target_phase) < band
    return mask, high_mag & phase_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    args = parser.parse_args()

    out_dir = Path(f"closest_graph_out_{slugify(args.model)}")
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
    per_unit_phase, _ = per_unit_phase_at_dom_bin(spectra, dom_freq)

    print(f"Sampling {args.n_pairs} pairs ...")
    u_idx, v_idx = sample_pair_indices(d_model, args.n_pairs, rng)

    print("Trace correlations ...")
    trace_corr = trace_correlations(streams_n, u_idx, v_idx)

    print("Cross-spectral ...")
    phase, mag = pair_cross_spectral(spectra, dom_freq[u_idx], u_idx, v_idx)

    prox = 1.0 - np.abs(trace_corr)
    edges_q = np.quantile(prox, [0, 1/3, 2/3, 1])
    closest_mask = prox <= edges_q[1]

    # Three graphs: in-phase, anti-phase, and combined (both peaks)
    p_close = phase[closest_mask]
    m_close = mag[closest_mask]
    u_close = u_idx[closest_mask]
    v_close = v_idx[closest_mask]

    cutoff = np.quantile(m_close, HIGH_MAG_QUANTILE)
    high_mag = m_close >= cutoff

    near_zero = np.abs(p_close) < PHASE_BAND
    near_pi = (np.abs(p_close - np.pi) < PHASE_BAND) | \
              (np.abs(p_close + np.pi) < PHASE_BAND)

    selections = {
        "in_phase":   high_mag & near_zero,
        "anti_phase": high_mag & near_pi,
        "combined":   high_mag & (near_zero | near_pi),
    }

    print(f"Selected pair counts: "
          f"in_phase={selections['in_phase'].sum()}, "
          f"anti_phase={selections['anti_phase'].sum()}, "
          f"combined={selections['combined'].sum()}")

    # Build adjacencies and run analyses
    results = {}
    group_dict = {}
    counts_dict = {}
    expected_dict = {}

    for name, sel in selections.items():
        u_sel = u_close[sel]
        v_sel = v_close[sel]
        n_edges = len(u_sel)
        if n_edges < 10:
            print(f"Skipping {name}, only {n_edges} edges")
            continue

        adj_real = build_adjacency(u_sel, v_sel, d_model)
        deg_real = degree_array(adj_real, d_model)

        # Nulls
        u_uni, v_uni = null_uniform_pairs(n_edges, d_model, rng)
        adj_uni = build_adjacency(u_uni, v_uni, d_model)
        deg_uni = degree_array(adj_uni, d_model)

        u_wt, v_wt = null_weighted_pairs(
            n_edges, np.maximum(mean_abs, 1e-8), rng
        )
        adj_wt = build_adjacency(u_wt, v_wt, d_model)
        deg_wt = degree_array(adj_wt, d_model)

        # Clustering
        cc_real = clustering_coefficient(adj_real, d_model)
        cc_uni = clustering_coefficient(adj_uni, d_model)
        cc_wt = clustering_coefficient(adj_wt, d_model)

        # Hub subgraph
        top_k = max(50, int(0.05 * d_model))
        hub_real = hub_internal_density(adj_real, d_model, top_k)
        hub_uni = hub_internal_density(adj_uni, d_model, top_k)
        hub_wt = hub_internal_density(adj_wt, d_model, top_k)

        # Components
        comps = connected_components(adj_real, d_model)
        sizes_comp = sorted([len(c) for c in comps], reverse=True)

        # Communities (multiple runs)
        labels_runs = []
        for r in range(N_COMMUNITY_RUNS):
            labels = label_propagation_communities(
                adj_real, d_model,
                rng=np.random.default_rng(SEED + r * 7919),
            )
            labels_runs.append(labels)
        labels_real = labels_runs[0]
        group, n_A, n_B = assign_AB_groups(labels_real)

        # Stability across runs
        groups_runs = [assign_AB_groups(L)[0] for L in labels_runs]
        overlaps = []
        for i in range(N_COMMUNITY_RUNS):
            for j in range(i + 1, N_COMMUNITY_RUNS):
                overlaps.append(overlap_score(groups_runs[i],
                                              groups_runs[j]))

        # Edge breakdown
        counts = edge_breakdown(u_sel, v_sel, group)
        expected = expected_edge_breakdown(group)

        # Plot degree
        plot_degree(
            deg_real, deg_uni, deg_wt,
            title=f"Degree distribution ({name})",
            save_path=out_dir / f"degree_distribution_{name}.png",
        )

        # Save results
        results[name] = {
            "n_edges": int(n_edges),
            "n_isolated": int((deg_real == 0).sum()),
            "max_degree": int(deg_real.max()),
            "median_degree": float(np.median(deg_real)),
            "mean_degree": float(deg_real.mean()),
            "n_components": len(comps),
            "largest_component": int(sizes_comp[0]) if sizes_comp else 0,
            "mean_clustering_real": float(np.nanmean(cc_real)),
            "mean_clustering_uniform": float(np.nanmean(cc_uni)),
            "mean_clustering_weighted": float(np.nanmean(cc_wt)),
            "hub_real": hub_real,
            "hub_uniform": hub_uni,
            "hub_weighted": hub_wt,
            "n_communities": int(labels_real.max() + 1),
            "largest_community": int(np.bincount(labels_real).max()),
            "top_5_community_sizes": np.sort(
                np.bincount(labels_real)
            )[-5:][::-1].tolist(),
            "groups_AB": {"A": int(n_A), "B": int(n_B),
                          "other": int((group == -1).sum())},
            "edge_breakdown": counts,
            "expected_breakdown": expected,
            "community_stability_min": float(np.nanmin(overlaps))
                if overlaps else None,
            "community_stability_mean": float(np.nanmean(overlaps))
                if overlaps else None,
        }
        group_dict[name] = group
        counts_dict[name] = counts
        expected_dict[name] = expected

    # Cross-graph: how do in-phase, anti-phase, and furthest-quadrature
    # group assignments compare? We don't have the furthest assignment
    # here, but we can compare in-phase vs anti-phase.
    if "in_phase" in group_dict and "anti_phase" in group_dict:
        g_in = group_dict["in_phase"]
        g_anti = group_dict["anti_phase"]
        cross_overlap = overlap_score(g_in, g_anti)
        results["cross_in_vs_anti_overlap"] = float(cross_overlap)

    # Combined plots
    plot_hub_bars(
        [results[n]["hub_real"] for n in results
         if isinstance(results[n], dict) and "hub_real" in results[n]] +
        [results[n]["hub_weighted"] for n in results
         if isinstance(results[n], dict) and "hub_weighted" in results[n]],
        labels=[f"{n} real" for n in results
                if isinstance(results[n], dict) and "hub_real" in results[n]] +
               [f"{n} weighted" for n in results
                if isinstance(results[n], dict) and "hub_weighted" in results[n]],
        save_path=out_dir / "hub_subgraph.png",
    )
    plot_community_sizes(
        {name: label_propagation_communities(
            build_adjacency(
                u_close[selections[name]],
                v_close[selections[name]],
                d_model,
            ),
            d_model,
            rng=np.random.default_rng(SEED),
         ) for name in selections if selections[name].sum() >= 10},
        save_path=out_dir / "community_sizes.png",
    )
    plot_edge_breakdown(
        counts_dict, expected_dict,
        save_path=out_dir / "edge_breakdown.png",
    )
    plot_feature_comparison(
        group_dict, dom_freq, mean_abs, per_unit_phase,
        save_path=out_dir / "feature_comparison.png",
    )

    summary = {
        "model": args.model,
        "d_model": int(d_model),
        "regime": "closest",
        "phase_band_radians": float(PHASE_BAND),
        "magnitude_quantile": float(HIGH_MAG_QUANTILE),
        "graphs": results,
    }
    with open(out_dir / "closest_graph_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    for name in ["in_phase", "anti_phase", "combined"]:
        if name not in results or not isinstance(results[name], dict):
            continue
        r = results[name]
        print(f"\n[{name}]: {r['n_edges']} edges, "
              f"isolated={r['n_isolated']}, max_deg={r['max_degree']}, "
              f"largest_component={r['largest_component']}")
        print(f"  Clustering: real={r['mean_clustering_real']:.4f}, "
              f"uniform={r['mean_clustering_uniform']:.4f}, "
              f"weighted={r['mean_clustering_weighted']:.4f}")
        print(f"  Hub ratio: real={r['hub_real']['ratio']:.2f}, "
              f"weighted={r['hub_weighted']['ratio']:.2f}")
        print(f"  Communities: {r['n_communities']}, "
              f"largest={r['largest_community']}, "
              f"top5={r['top_5_community_sizes']}")
        print(f"  Groups A={r['groups_AB']['A']}, "
              f"B={r['groups_AB']['B']}, other={r['groups_AB']['other']}")
        eb = r["edge_breakdown"]
        ex = r["expected_breakdown"]
        if ex:
            total = (eb["within_A"] + eb["within_B"] + eb["between_AB"])
            if total > 0:
                print(f"  Edge fractions: within_A={eb['within_A']/total:.3f} "
                      f"(exp {ex['within_A_expected_frac']:.3f}), "
                      f"within_B={eb['within_B']/total:.3f} "
                      f"(exp {ex['within_B_expected_frac']:.3f}), "
                      f"between_AB={eb['between_AB']/total:.3f} "
                      f"(exp {ex['between_AB_expected_frac']:.3f})")
        print(f"  Stability: mean overlap "
              f"{r['community_stability_mean']:.3f}, "
              f"min {r['community_stability_min']:.3f}")
    if "cross_in_vs_anti_overlap" in results:
        print(f"\nCross-graph: in-phase A/B labels vs anti-phase A/B "
              f"labels overlap = {results['cross_in_vs_anti_overlap']:.3f}")
    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()