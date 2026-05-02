"""
Plot phase portraits of the same residual unit across all four
trajectory variants (cumulative, combined_delta, attn_delta,
mlp_delta), to see how each unit's behavior decomposes.

Picks units that span the cumulative-trajectory R range, then plots
all four variants for each picked unit on one row.

Usage:
  python phase_portraits_per_unit.py
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


N_SAMPLES = 128
SEQ_LEN = 128
SEED = 0
N_UNITS_TO_PLOT = 6  # rows in the figure


def load_model():
    print("Loading TinyLlama ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()
    return model, tokenizer


def get_hook_targets(model):
    blocks = model.model.layers
    targets = []
    for block in blocks:
        targets.append(block.input_layernorm)
        targets.append(block.post_attention_layernorm)
    return targets


def collect_streams(model, tokenizer, texts, seq_len):
    targets = get_hook_targets(model)
    out = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt",
                        max_length=seq_len, truncation=True)
        ids = enc["input_ids"].to(next(model.parameters()).device)
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
        last = torch.stack(
            [c[0, -1, :].float() for c in captures], dim=0
        )
        out.append(last.cpu().numpy())
    return np.stack(out, axis=0)


def make_variants(streams):
    diffs = np.diff(streams, axis=1)
    return {
        "cumulative": streams,
        "combined_delta": diffs,
        "attn_delta": diffs[:, ::2, :],
        "mlp_delta": diffs[:, 1::2, :],
    }


def rotation_count_per_unit(streams):
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    theta = np.arctan2(dy, dx)
    dtheta = np.diff(theta, axis=1)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    return (dtheta.sum(axis=1) / (2 * np.pi)).mean(axis=0)


def plot_per_unit_portraits(variants, R_cumulative, units_to_plot,
                            sample_idx=0, save_path=None):
    n_units = len(units_to_plot)
    n_variants = len(variants)
    fig, axes = plt.subplots(n_units, n_variants,
                              figsize=(3.6 * n_variants,
                                       3.0 * n_units))
    if n_units == 1:
        axes = axes.reshape(1, -1)

    variant_order = ["cumulative", "combined_delta",
                     "attn_delta", "mlp_delta"]

    for row, u in enumerate(units_to_plot):
        for col, label in enumerate(variant_order):
            ax = axes[row, col]
            traj = variants[label][sample_idx, :, u]
            grad = np.gradient(traj)
            x = traj - traj[0]
            y = grad - grad[0]
            n_steps = len(traj)
            ax.plot(x, y, "-", linewidth=0.7, alpha=0.9)
            ax.scatter(x, y, c=np.arange(n_steps),
                       cmap="viridis", s=10)
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(
                    f"unit {u}\nR_cum={R_cumulative[u]:.2f}",
                    fontsize=10,
                )
            ax.set_xlabel("a - a_0", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--n-units", type=int, default=N_UNITS_TO_PLOT)
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path("tinyllama_out")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("Loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                      split="validation")
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]

    model, tokenizer = load_model()
    print("Collecting streams ...")
    streams = collect_streams(model, tokenizer, texts, args.seq_len)
    print(f"streams shape: {streams.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("Making variants ...")
    variants = make_variants(streams)

    print("Computing R for cumulative (to choose units) ...")
    R_cum = rotation_count_per_unit(variants["cumulative"])

    # Pick units across the |R_cumulative| range
    order = np.argsort(np.abs(R_cum))
    n = len(order)
    pick = [
        order[n // 20],          # very low
        order[n // 6],           # low
        order[n // 3],           # mid-low
        order[2 * n // 3],       # mid-high
        order[5 * n // 6],       # high
        order[-n // 20],         # very high
    ][:args.n_units]

    print(f"Selected units: {pick}")
    print(f"Their R_cumulative values: "
          f"{[float(R_cum[u]) for u in pick]}")

    plot_per_unit_portraits(
        variants, R_cum, pick,
        sample_idx=args.sample_idx,
        save_path=out_dir / "per_unit_portraits.png",
    )
    print(f"Saved per_unit_portraits.png to {out_dir}")


if __name__ == "__main__":
    main()