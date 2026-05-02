"""
Characterize the coupling graph defined by selected quadrature pairs.

Build a graph where nodes are residual units and edges are pairs that
were selected as 'coupled at quadrature' (high cospec magnitude in the
furthest regime, phase near ±π/2). Then ask:
  1. Is the graph dense, sparse, or somewhere in between?
  2. Do the heavy-participating units couple with each other (dense
     subgraph) or with random partners (hub-and-spoke)?
  3. Are there communities, and how strong are they?
  4. How do these compare to two null models:
       (a) uniform random pairs
       (b) loudness-weighted random pairs
     The second null tells us what's beyond the loudness effect.

Outputs in graph_out_<model>/:
  - degree_distribution.png   degree histogram with both nulls overlaid
  - clustering_distribution.png  per-node clustering coefficient
  - hub_subgraph.png          edge density among top-degree units vs others
  - community_sizes.png       sizes of detected communities
  - graph_summary.json
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
# Graph analysis
# ---------------------------------------------------------------------------

def build_adjacency(u_sel, v_sel, d_model):
    """
    Build a sparse symmetric adjacency matrix as a dict-of-sets.
    Multi-edges between the same pair are collapsed (we just track
    whether a pair was selected at least once).
    """
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
    """
    Local clustering coefficient: for each node, the fraction of pairs
    of neighbors that are themselves connected. Returns NaN for nodes
    with degree < 2.
    """
    cc = np.full(d_model, np.nan)
    for i in range(d_model):
        nbrs = adj[i]
        k = len(nbrs)
        if k < 2:
            continue
        # Count edges among neighbors
        nbrs_list = list(nbrs)
        edges = 0
        for j_idx in range(len(nbrs_list)):
            for k_idx in range(j_idx + 1, len(nbrs_list)):
                if nbrs_list[k_idx] in adj[nbrs_list[j_idx]]:
                    edges += 1
        possible = k * (k - 1) / 2
        cc[i] = edges / possible
    return cc


def hub_internal_density(adj, d_model, top_k):
    """
    Take the top_k highest-degree nodes. What fraction of all possible
    edges among them are present? Compare to overall edge density.
    """
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

    # Overall density: total edges / d_model choose 2
    total_edges = sum(len(adj[i]) for i in range(d_model)) / 2
    possible_total = d_model * (d_model - 1) / 2
    density_overall = total_edges / possible_total if possible_total else 0.0

    return {
        "top_k": int(top_k),
        "edges_in_top": int(edges_in_top),
        "possible_in_top": int(possible_in_top),
        "density_in_top": float(density_in_top),
        "density_overall": float(density_overall),
        "ratio": (
            float(density_in_top / density_overall)
            if density_overall > 0 else float("inf")
        ),
        "top_nodes": [int(x) for x in top_nodes],
    }


def connected_components(adj, d_model):
    """Return list of components (each a set of node indices)."""
    seen = [False] * d_model
    comps = []
    for start in range(d_model):
        if seen[start]:
            continue
        # BFS
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
    """
    Simple label propagation. Each node starts with its own label;
    repeatedly, each node adopts the most common label among its
    neighbors. Ties broken randomly. Iterate until stable or max_iter.

    This is a quick-and-dirty community detector. It's known to be
    unstable and to produce different results across runs. We treat
    its output as one possible community structure, not the truth.
    """
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
    # Renumber labels
    _, labels = np.unique(labels, return_inverse=True)
    return labels


# ---------------------------------------------------------------------------
# Null models
# ---------------------------------------------------------------------------

def null_uniform_pairs(n_edges, d_model, rng):
    """Sample n_edges pairs uniformly at random, no self-loops."""
    a = rng.integers(0, d_model, size=n_edges * 2)
    b = rng.integers(0, d_model, size=n_edges * 2)
    mask = a != b
    a = a[mask][:n_edges]
    b = b[mask][:n_edges]
    return np.minimum(a, b), np.maximum(a, b)


def null_weighted_pairs(n_edges, weights, rng):
    """
    Sample pairs with each unit picked proportional to weight.
    weights: (d_model,) non-negative array.
    """
    p = weights / weights.sum()
    d = len(weights)
    # Oversample to allow rejecting a==b
    factor = 3
    a = rng.choice(d, size=n_edges * factor, p=p)
    b = rng.choice(d, size=n_edges * factor, p=p)
    mask = a != b
    a = a[mask][:n_edges]
    b = b[mask][:n_edges]
    return np.minimum(a, b), np.maximum(a, b)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_degree_distribution(deg_real, deg_uniform, deg_weighted,
                             save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, max(deg_real.max(), deg_uniform.max(),
                             deg_weighted.max()) + 2)
    for label, d in [("real", deg_real),
                     ("null: uniform", deg_uniform),
                     ("null: loudness-weighted", deg_weighted)]:
        ax.hist(d, bins=bins, histtype="step", linewidth=1.5,
                density=True, label=label)
    ax.set_xlabel("degree (number of distinct partners per unit)")
    ax.set_ylabel("density")
    ax.set_title("Degree distribution: real graph vs nulls")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_clustering_distribution(cc_real, cc_uniform, cc_weighted,
                                 save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 0.5, 50)
    for label, cc in [("real", cc_real),
                      ("null: uniform", cc_uniform),
                      ("null: loudness-weighted", cc_weighted)]:
        finite = cc[~np.isnan(cc)]
        ax.hist(finite, bins=bins, histtype="step", linewidth=1.5,
                density=True, label=f"{label} (mean={finite.mean():.3f})")
    ax.set_xlabel("local clustering coefficient")
    ax.set_ylabel("density")
    ax.set_title("Clustering coefficient: real vs nulls")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_hub_subgraph(stats_real, stats_uniform, stats_weighted,
                      save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["real", "uniform", "loudness-weighted"]
    densities_top = [stats_real["density_in_top"],
                     stats_uniform["density_in_top"],
                     stats_weighted["density_in_top"]]
    densities_all = [stats_real["density_overall"],
                     stats_uniform["density_overall"],
                     stats_weighted["density_overall"]]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, densities_top, width, label="top-k subgraph")
    ax.bar(x + width/2, densities_all, width, label="full graph")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("edge density")
    ax.set_title(
        f"Edge density: top-{stats_real['top_k']} hub subgraph "
        f"vs whole graph"
    )
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_community_sizes(labels_real, labels_uniform, labels_weighted,
                         save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    titles = ["real", "null: uniform", "null: loudness-weighted"]
    for ax, labs, title in zip(axes,
                                [labels_real, labels_uniform,
                                 labels_weighted],
                                titles):
        sizes = np.bincount(labs)
        sizes = np.sort(sizes)[::-1]
        ax.bar(np.arange(len(sizes)), sizes)
        ax.set_xlabel("community rank")
        ax.set_yscale("log")
        ax.set_title(f"{title} ({len(sizes)} communities)")
    axes[0].set_ylabel("community size")
    fig.suptitle("Detected community sizes (label propagation)")
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

    out_dir = Path(f"graph_out_{slugify(args.model)}")
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
    n_layers = get_n_layers(model)
    print(f"d_model={d_model}, n_layers={n_layers}")

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

    print(f"Sampling {args.n_pairs} pairs ...")
    u_idx, v_idx = sample_pair_indices(d_model, args.n_pairs, rng)

    print("Trace correlations ...")
    trace_corr = trace_correlations(streams_n, u_idx, v_idx)

    print("Cross-spectral ...")
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
    print(f"Selected {n_edges} pairs to form the graph.")

    print("Building real adjacency ...")
    adj_real = build_adjacency(u_sel, v_sel, d_model)
    deg_real = degree_array(adj_real, d_model)

    print("Building null adjacencies ...")
    u_un, v_un = null_uniform_pairs(n_edges, d_model, rng)
    adj_uniform = build_adjacency(u_un, v_un, d_model)
    deg_uniform = degree_array(adj_uniform, d_model)

    u_wt, v_wt = null_weighted_pairs(
        n_edges, np.maximum(mean_abs, 1e-8), rng
    )
    adj_weighted = build_adjacency(u_wt, v_wt, d_model)
    deg_weighted = degree_array(adj_weighted, d_model)

    print("Computing clustering coefficients ...")
    cc_real = clustering_coefficient(adj_real, d_model)
    cc_uniform = clustering_coefficient(adj_uniform, d_model)
    cc_weighted = clustering_coefficient(adj_weighted, d_model)

    print("Hub subgraph density ...")
    top_k = max(50, int(0.05 * d_model))
    hub_real = hub_internal_density(adj_real, d_model, top_k)
    hub_uniform = hub_internal_density(adj_uniform, d_model, top_k)
    hub_weighted = hub_internal_density(adj_weighted, d_model, top_k)

    print("Connected components ...")
    comps_real = connected_components(adj_real, d_model)
    sizes_real = sorted([len(c) for c in comps_real], reverse=True)
    largest_comp_size = sizes_real[0] if sizes_real else 0

    print("Community detection (label propagation) ...")
    labels_real = label_propagation_communities(
        adj_real, d_model, rng=np.random.default_rng(SEED)
    )
    labels_uniform = label_propagation_communities(
        adj_uniform, d_model, rng=np.random.default_rng(SEED)
    )
    labels_weighted = label_propagation_communities(
        adj_weighted, d_model, rng=np.random.default_rng(SEED)
    )

    print("Saving plots ...")
    plot_degree_distribution(
        deg_real, deg_uniform, deg_weighted,
        save_path=out_dir / "degree_distribution.png",
    )
    plot_clustering_distribution(
        cc_real, cc_uniform, cc_weighted,
        save_path=out_dir / "clustering_distribution.png",
    )
    plot_hub_subgraph(
        hub_real, hub_uniform, hub_weighted,
        save_path=out_dir / "hub_subgraph.png",
    )
    plot_community_sizes(
        labels_real, labels_uniform, labels_weighted,
        save_path=out_dir / "community_sizes.png",
    )

    # Numerical summary
    summary = {
        "model": args.model,
        "d_model": int(d_model),
        "n_edges": int(n_edges),
        "graph": {
            "n_isolated_units": int((deg_real == 0).sum()),
            "max_degree": int(deg_real.max()),
            "median_degree": float(np.median(deg_real)),
            "mean_degree": float(deg_real.mean()),
            "n_connected_components": len(comps_real),
            "largest_component_size": int(largest_comp_size),
            "fraction_in_largest_component": float(
                largest_comp_size / d_model
            ),
            "mean_clustering_real": float(np.nanmean(cc_real)),
            "mean_clustering_uniform_null": float(np.nanmean(cc_uniform)),
            "mean_clustering_weighted_null": float(np.nanmean(cc_weighted)),
        },
        "hub_subgraph": {
            "real": hub_real,
            "uniform_null": hub_uniform,
            "weighted_null": hub_weighted,
        },
        "communities_label_prop": {
            "real_n_communities": int(labels_real.max() + 1),
            "uniform_null_n_communities": int(labels_uniform.max() + 1),
            "weighted_null_n_communities": int(labels_weighted.max() + 1),
            "real_largest_community": int(np.bincount(labels_real).max()),
            "real_top_5_community_sizes":
                np.sort(np.bincount(labels_real))[-5:][::-1].tolist(),
        },
    }
    with open(out_dir / "graph_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    g = summary["graph"]
    print(f"Edges: {n_edges}, isolated units: {g['n_isolated_units']}, "
          f"max degree: {g['max_degree']}")
    print(f"Connected components: {g['n_connected_components']}, "
          f"largest = {g['largest_component_size']} "
          f"({100 * g['fraction_in_largest_component']:.1f}% of units)")
    print(f"Mean clustering: real={g['mean_clustering_real']:.4f}, "
          f"uniform={g['mean_clustering_uniform_null']:.4f}, "
          f"weighted={g['mean_clustering_weighted_null']:.4f}")
    h = summary["hub_subgraph"]
    print(f"Top-{h['real']['top_k']} hub subgraph density: "
          f"real={h['real']['density_in_top']:.4f}, "
          f"uniform={h['uniform_null']['density_in_top']:.4f}, "
          f"weighted={h['weighted_null']['density_in_top']:.4f}")
    print(f"Hub-density / overall ratio: "
          f"real={h['real']['ratio']:.2f}, "
          f"uniform={h['uniform_null']['ratio']:.2f}, "
          f"weighted={h['weighted_null']['ratio']:.2f}")
    c = summary["communities_label_prop"]
    print(f"Communities (label prop): real={c['real_n_communities']}, "
          f"uniform={c['uniform_null_n_communities']}, "
          f"weighted={c['weighted_null_n_communities']}")
    print(f"Largest 5 real communities: {c['real_top_5_community_sizes']}")
    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()