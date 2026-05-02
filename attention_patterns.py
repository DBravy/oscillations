"""
Attention pattern analysis across all layers for p5, p6, p7.

Extracts per-input attention probabilities from every (layer, head),
then aggregates across the 256 (a, b) input pairs to produce both mean
and std views.

Question to be answered:

  Route 2: p7 directly retrieves c1's pre-computed value from p5 via
  attention. Signature: at some layer, p7 has unusually high attention
  weight on p5, and that attention is consistent across inputs.

  Route 1: p7 computes c0 locally using shared weights on its own local
  content (which already includes attention reads from earlier
  positions). Signature: ordinary attention patterns, p7 attending to
  wherever it normally would (e.g. p1, p4 directly for a0, b0).

Layout (8 token positions, indexed 0-7):
    pos 0: a1   pos 1: a0   pos 2: +
    pos 3: b1   pos 4: b0   pos 5: =
    pos 6: c2   pos 7: c1

Plots in --out-dir:
  attention_to_p5_curves.png       attention from each query (p6, p7) to
                                    p5 vs layer, per head; the headline
                                    Route 1 vs Route 2 plot
  mean_attention_query{N}.png      one figure per query position; per-head
                                    heatmaps (layer x key)
  attention_distribution_query{N}.png  stacked area: how attention is
                                    distributed across keys per layer
  input_dependence_query{N}.png    std across inputs of attention by
                                    (layer, key); high std = input-
                                    routing, low std = structural
  attention_summary.json           numerical summaries

Usage:
  python attention_patterns.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_attention_trained
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from toy_transformer_addition import (
    ToyTransformer,
    ModelConfig,
    AdditionDataset,
)


QUERY_POSITIONS = [5, 6, 7]
TOKEN_LABELS = {
    0: "a1", 1: "a0", 2: "+", 3: "b1",
    4: "b0", 5: "=", 6: "c2", 7: "c1",
}


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = ToyTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_attention_patterns(model, batch_size=64):
    """
    Run the model on all 256 (a, b) pairs, capturing attention probs at
    every (layer, head). Returns array of shape
        (N, n_layers, n_heads, T, T)
    where attn[n, l, h, q, k] is P(query q attends to key k) on input n
    in layer l, head h.
    """
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]
    ds = AdditionDataset(pairs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_per_layer = None      # list-of-lists; outer = batches
    for x, _, _ in loader:
        x = x.to(device)
        B, T = x.shape

        h = model.tok_emb(x) + model.pos_emb(model.pos_ids[:, :T])
        layer_patterns = []
        for block in model.blocks:
            attn = block.attn
            qkv = attn.W_qkv(h)
            qkv = qkv.view(B, T, 3, attn.n_heads, attn.d_head)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scores = torch.matmul(
                q, k.transpose(-2, -1)
            ) / math.sqrt(attn.d_head)
            scores = scores.masked_fill(
                ~attn.causal_mask[:T, :T], float("-inf"),
            )
            probs = F.softmax(scores, dim=-1)
            layer_patterns.append(probs.detach().cpu().numpy())

            # Continue forward pass to next block
            out = torch.matmul(probs, v)
            out = (out.transpose(1, 2).contiguous()
                      .view(B, T, attn.n_heads * attn.d_head))
            h = h + attn.W_o(out)
            h = h + block.mlp(h)

        # Stack into (n_layers, B, n_heads, T, T)
        batch_arr = np.stack(layer_patterns, axis=0)
        if all_per_layer is None:
            all_per_layer = [batch_arr]
        else:
            all_per_layer.append(batch_arr)

    # Concatenate along batch dim, then move batch to front
    full = np.concatenate(all_per_layer, axis=1)  # (L, N, H, T, T)
    return np.transpose(full, (1, 0, 2, 3, 4))    # (N, L, H, T, T)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def key_labels_for_query(q):
    return [
        f"{i}\n{TOKEN_LABELS[i]}"
        for i in range(q + 1)
    ]


def plot_attention_to_p5_curves(mean_attn, std_attn, save_path):
    """For each query position q in {5, 6, 7}, plot fraction of attn
    going to key=5, layer by layer, one line per head. Adds a uniform
    baseline 1/(q+1)."""
    n_layers, n_heads, T, _ = mean_attn.shape
    fig, axes = plt.subplots(1, len(QUERY_POSITIONS),
                              figsize=(6 * len(QUERY_POSITIONS), 5),
                              sharey=True)
    for ax, q in zip(axes, QUERY_POSITIONS):
        for h in range(n_heads):
            mean_curve = mean_attn[:, h, q, 5]    # (n_layers,)
            std_curve  = std_attn[:, h, q, 5]
            ax.plot(np.arange(n_layers), mean_curve, "o-",
                    label=f"head {h} (mean)",
                    color=f"C{h}")
            ax.fill_between(
                np.arange(n_layers),
                mean_curve - std_curve, mean_curve + std_curve,
                color=f"C{h}", alpha=0.15,
            )
        baseline = 1.0 / (q + 1)
        ax.axhline(baseline, color="gray", linewidth=0.6,
                   linestyle="--",
                   label=f"uniform: 1/(q+1) = {baseline:.3f}")
        ax.set_title(f"query: pos {q} ({TOKEN_LABELS[q]})", fontsize=11)
        ax.set_xlabel("layer")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('attention to p5 ("=")')
    fig.suptitle(
        "Attention from each query to p5 across layers "
        "(mean +/- std across 256 inputs)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_mean_attention_per_query(mean_attn, q, save_path):
    """For a given query position q, one figure with n_heads subplots,
    each a heatmap of (layer, key) showing mean attention."""
    n_layers, n_heads = mean_attn.shape[0], mean_attn.shape[1]
    fig, axes = plt.subplots(
        1, n_heads, figsize=(4.5 * n_heads, 5.5),
        squeeze=False, sharey=True,
    )
    for h in range(n_heads):
        ax = axes[0, h]
        # mean_attn[:, h, q, :q+1] -> (n_layers, q+1)
        M = mean_attn[:, h, q, : q + 1]
        im = ax.imshow(M, aspect="auto", cmap="viridis",
                        vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(q + 1))
        ax.set_xticklabels(key_labels_for_query(q), fontsize=8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{l}" for l in range(n_layers)],
                            fontsize=7)
        ax.set_xlabel("key position")
        ax.set_title(f"head {h}", fontsize=10)
        # Annotate: highlight high-attention cells
        for l in range(n_layers):
            for k in range(q + 1):
                val = M[l, k]
                if val > 0.4:
                    ax.text(
                        k, l, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=6,
                        color="white" if val < 0.7 else "black",
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel("layer")
    fig.suptitle(
        f"Mean attention from query pos {q} ({TOKEN_LABELS[q]}) "
        f"across all layers and heads",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_attention_distribution_per_query(mean_attn, q, save_path):
    """Stacked area plot showing how each head's attention is distributed
    across keys, layer by layer. One subplot per head."""
    n_layers, n_heads = mean_attn.shape[0], mean_attn.shape[1]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, n_heads, figsize=(6.5 * n_heads, 5),
                              squeeze=False, sharey=True)
    for h in range(n_heads):
        ax = axes[0, h]
        M = mean_attn[:, h, q, : q + 1]    # (n_layers, q+1)
        # Stack each key's attention as a band
        cumulative = np.zeros(n_layers)
        xs = np.arange(n_layers)
        for k in range(q + 1):
            top = cumulative + M[:, k]
            ax.fill_between(
                xs, cumulative, top,
                color=cmap(k % 10), alpha=0.85,
                label=f"key {k} ({TOKEN_LABELS[k]})",
            )
            cumulative = top
        ax.set_xlabel("layer")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(xs)
        ax.set_title(f"head {h}", fontsize=10)
        ax.legend(fontsize=7, ncol=2,
                   loc="lower right" if h == 0 else "lower left")
    axes[0, 0].set_ylabel(f"attention budget at query pos {q}")
    fig.suptitle(
        f"Attention distribution at query pos {q} ({TOKEN_LABELS[q]}) "
        f"across layers",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_input_dependence_per_query(std_attn, q, save_path):
    """Heatmap of std-across-inputs of attention from query q. High std
    cells indicate attention edges that depend on input. Low std cells
    indicate structural attention edges that the model uses
    consistently."""
    n_layers, n_heads = std_attn.shape[0], std_attn.shape[1]
    fig, axes = plt.subplots(1, n_heads, figsize=(4.5 * n_heads, 5.5),
                              squeeze=False, sharey=True)
    vmax = std_attn[:, :, q, : q + 1].max()
    for h in range(n_heads):
        ax = axes[0, h]
        S = std_attn[:, h, q, : q + 1]
        im = ax.imshow(S, aspect="auto", cmap="magma",
                        vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(q + 1))
        ax.set_xticklabels(key_labels_for_query(q), fontsize=8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{l}" for l in range(n_layers)],
                            fontsize=7)
        ax.set_xlabel("key position")
        ax.set_title(f"head {h}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel("layer")
    fig.suptitle(
        f"Input-dependence of attention from query pos {q} "
        f"({TOKEN_LABELS[q]}): std across 256 inputs",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="toy_attention")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, d_head={cfg.d_head}")

    print("\nExtracting attention patterns on all 256 (a, b) pairs ...")
    patterns = extract_attention_patterns(model)
    print(f"  shape: {patterns.shape}  "
          f"(N, n_layers, n_heads, T, T)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mean_attn = patterns.mean(axis=0)   # (L, H, T, T)
    std_attn  = patterns.std(axis=0)

    # ------------- Headline plot -------------
    print("\nMaking headline 'attention to p5' plot ...")
    plot_attention_to_p5_curves(
        mean_attn, std_attn,
        save_path=out_dir / "attention_to_p5_curves.png",
    )

    # ------------- Per-query detail plots -------------
    print("Making per-query detail plots ...")
    for q in QUERY_POSITIONS:
        plot_mean_attention_per_query(
            mean_attn, q,
            save_path=out_dir / f"mean_attention_query{q}.png",
        )
        plot_attention_distribution_per_query(
            mean_attn, q,
            save_path=out_dir / f"attention_distribution_query{q}.png",
        )
        plot_input_dependence_per_query(
            std_attn, q,
            save_path=out_dir / f"input_dependence_query{q}.png",
        )

    # ------------- Summaries -------------
    summary = {
        "checkpoint":   str(args.checkpoint),
        "n_inputs":     int(patterns.shape[0]),
        "n_layers":     int(cfg.n_layers),
        "n_heads":      int(cfg.n_heads),
        "seq_len":      int(patterns.shape[3]),
        "token_labels": TOKEN_LABELS,
        "per_query":    {},
    }

    for q in QUERY_POSITIONS:
        per_query_summary = {
            "label":          TOKEN_LABELS[q],
            "uniform_baseline": 1.0 / (q + 1),
            "max_attention_to_p5_across_layers_heads": {
                "value":      float(mean_attn[:, :, q, 5].max()),
                "layer":      int(np.unravel_index(
                                  mean_attn[:, :, q, 5].argmax(),
                                  mean_attn[:, :, q, 5].shape)[0]),
                "head":       int(np.unravel_index(
                                  mean_attn[:, :, q, 5].argmax(),
                                  mean_attn[:, :, q, 5].shape)[1]),
            },
            "attention_to_p5_per_layer": {
                f"layer_{l}": {
                    f"head_{h}": float(mean_attn[l, h, q, 5])
                    for h in range(cfg.n_heads)
                }
                for l in range(cfg.n_layers)
            },
            "attention_full_per_layer_head_summary": [
                {
                    "layer":          int(l),
                    "head":           int(h),
                    "top_3_keys":     [
                        {
                            "key":         int(k),
                            "key_label":   TOKEN_LABELS[k],
                            "mean_attn":   float(mean_attn[l, h, q, k]),
                            "std_attn":    float(std_attn[l, h, q, k]),
                        }
                        for k in np.argsort(
                            -mean_attn[l, h, q, : q + 1]
                        )[:3]
                    ],
                }
                for l in range(cfg.n_layers)
                for h in range(cfg.n_heads)
            ],
        }
        summary["per_query"][f"q{q}"] = per_query_summary

    with open(out_dir / "attention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save raw arrays
    np.savez(
        out_dir / "attention_data.npz",
        patterns=patterns,
        mean_attn=mean_attn,
        std_attn=std_attn,
    )

    # ------------- Console summary -------------
    print("\n--- Attention to p5 (mean across inputs) ---")
    print(f"{'query':<8} {'layer':<7} {'head':<6} {'mean':>10} {'std':>10}")
    for q in QUERY_POSITIONS:
        # Show top-5 layer/head combinations by attention to p5
        flat = mean_attn[:, :, q, 5].ravel()
        top5_flat_idx = np.argsort(-flat)[:5]
        for fi in top5_flat_idx:
            l, h = np.unravel_index(fi, mean_attn[:, :, q, 5].shape)
            print(f"q={q:<5} L{l:<6} h{h:<5} "
                  f"{mean_attn[l, h, q, 5]:>10.4f} "
                  f"{std_attn[l, h, q, 5]:>10.4f}")
        print()

    print(f"Outputs in {out_dir.resolve()}")
    print("  attention_to_p5_curves.png         (headline)")
    print("  mean_attention_query{5,6,7}.png    (heatmaps)")
    print("  attention_distribution_query{5,6,7}.png  (stacked area)")
    print("  input_dependence_query{5,6,7}.png  (std heatmaps)")
    print("  attention_data.npz")
    print("  attention_summary.json")


if __name__ == "__main__":
    main()
