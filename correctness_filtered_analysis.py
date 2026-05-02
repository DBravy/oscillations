"""
Filter-to-correct diagnostic for the position 7 confound.

Question: Is position 7's low PLV (relative to position 5) caused by
the model failing more at position 7, or is it intrinsic to position 7's
mechanism even on samples the model handles correctly?

Approach: Partition the 256 (a, b) pairs by per-position correctness.
For each position p, we have:
  - correct_p: samples where preds[p] == target[p]
  - incorrect_p: complement

We compute PLV / per-unit power / cospec_mag on the all, correct, and
incorrect subsets at each position, then compare. The interpretive rule:

  If position 7's PLV on correct_7 is similar to position 7's PLV on
  all samples, the mechanism is real. Position 7 just genuinely has a
  different phase organization than position 5.

  If position 7's PLV on correct_7 jumps up toward position 5's PLV,
  the apparent difference was driven by failures. Use correct-only
  subsets going forward.

Outputs in --out-dir:
  - per_position_accuracy.txt          quick readout
  - subset_comparison_plv.png          PLV by (position, subset)
  - subset_comparison_power.png        per-unit bin-1 mag by (position, subset)
  - subset_comparison_cospec.png       cospec_mag p95 by (position, subset)
  - cross_position_correct_only.png    PLV / power / cospec across positions,
                                       computed on the "correct at all positions" subset
  - decomposition_top100_correct.png   the disentangle decomp restricted to
                                       all-correct subset
  - correctness_summary.json           numerical summary

Usage:
  python correctness_filtered_analysis.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_correctness_filter
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from toy_transformer_addition import (
    ToyTransformer, ModelConfig, capture_residual_streams,
    AdditionDataset,
)
from analyze_toy_residuals import (
    rotation_count_per_unit, per_unit_spectra, all_pair_indices, SEED,
)
from disentangle_cospec import (
    compute_metrics, classify_phase,
    POSITIONS, POSITION_LABELS, POSITION_COLORS,
)


# ---------------------------------------------------------------------------
# Stream + prediction collection
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = ToyTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def collect_streams_and_predictions(model, cfg, batch_size=64):
    """One pass for residuals (via the existing utility), plus a second
    pass to grab logits and per-position correctness."""
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]

    tokens, captures = capture_residual_streams(
        model, pairs, device, batch_size=batch_size
    )

    # Predictions
    ds = AdditionDataset(pairs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_targets.append(y)
    preds = torch.cat(all_preds, dim=0).numpy()       # (N, 8)
    targets = torch.cat(all_targets, dim=0).numpy()   # (N, 8)

    # Per-position correctness. preds[i, p] is the model's prediction
    # at input position p, which is meant to equal target tokens[p+1].
    correct_by_pos = {
        pos: (preds[:, pos] == targets[:, pos])
        for pos in POSITIONS
    }

    # Build cumulative streams at each position
    sublayer_meta = [(c["kind"], c["layer"]) for c in captures]
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    streams_by_pos = {}
    for pos in POSITIONS:
        at_pos = stacked[:, :, pos, :]                 # (n_sublayers, N, D)
        s = at_pos.permute(1, 0, 2).contiguous()       # (N, n_sublayers, D)
        streams_by_pos[pos] = s.numpy().astype(np.float32)

    return streams_by_pos, sublayer_meta, correct_by_pos, preds, targets


# ---------------------------------------------------------------------------
# Subset construction
# ---------------------------------------------------------------------------

def build_subsets(correct_by_pos):
    """Return a dict of {subset_name: bool mask of length N}."""
    n = len(correct_by_pos[POSITIONS[0]])
    correct_all = np.ones(n, dtype=bool)
    for pos in POSITIONS:
        correct_all &= correct_by_pos[pos]

    subsets = {
        "all":          np.ones(n, dtype=bool),
        "correct_5":    correct_by_pos[5],
        "incorrect_5":  ~correct_by_pos[5],
        "correct_6":    correct_by_pos[6],
        "incorrect_6":  ~correct_by_pos[6],
        "correct_7":    correct_by_pos[7],
        "incorrect_7":  ~correct_by_pos[7],
        "correct_all":  correct_all,
    }
    return subsets


def safe_compute_metrics(streams, mask):
    """Compute metrics on a subset; return None if subset is too small."""
    if mask.sum() < 4:
        return None
    return compute_metrics(streams[mask])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# For each position, which subsets to show
POS_SUBSETS = {
    5: [("all", "all"), ("correct_5", "correct"), ("incorrect_5", "incorrect")],
    6: [("all", "all"), ("correct_6", "correct"), ("incorrect_6", "incorrect")],
    7: [("all", "all"), ("correct_7", "correct"), ("incorrect_7", "incorrect")],
}
SUBSET_COLORS = {"all": "C7", "correct": "C2", "incorrect": "C3"}


def plot_subset_bar_metric(metrics_grid, value_fn, ylabel, title, save_path,
                            error_fn=None):
    """metrics_grid: {pos: {subset_label: metrics_dict_or_None}}.
    value_fn(metrics) returns the scalar to plot."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.27
    x_base = np.arange(len(POSITIONS))

    for i, (subset_key, subset_label) in enumerate(
        [("all", "all"), ("correct", "correct"), ("incorrect", "incorrect")]
    ):
        offset = (i - 1) * width
        vals, errs, ns = [], [], []
        for pos in POSITIONS:
            m = metrics_grid[pos].get(subset_label)
            if m is None:
                vals.append(np.nan)
                errs.append(0.0)
                ns.append(0)
            else:
                vals.append(value_fn(m))
                errs.append(error_fn(m) if error_fn else 0.0)
                ns.append(int(m["_subset_n"]))
        bars = ax.bar(x_base + offset, vals, width,
                       color=SUBSET_COLORS[subset_key],
                       label=subset_label,
                       yerr=errs if error_fn else None,
                       capsize=4)
        # Annotate with subset size
        for j, bar in enumerate(bars):
            if not np.isnan(vals[j]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"n={ns[j]}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x_base)
    ax.set_xticklabels([POSITION_LABELS[p] for p in POSITIONS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, title="subset")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_subset_comparison_plv(metrics_grid, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    width = 0.27
    x_base = np.arange(len(POSITIONS))

    for ax, (key, label) in zip(
        axes,
        [
            ("plv_all_mean",       "Mean PLV over all pairs"),
            ("plv_high_amp_mean",  "Mean PLV over high-amp pairs (top 25% amp_prod)"),
        ],
    ):
        for i, subset in enumerate(["all", "correct", "incorrect"]):
            offset = (i - 1) * width
            vals, ns = [], []
            for pos in POSITIONS:
                m = metrics_grid[pos].get(subset)
                if m is None:
                    vals.append(np.nan); ns.append(0)
                else:
                    vals.append(m[key]); ns.append(m["_subset_n"])
            bars = ax.bar(x_base + offset, vals, width,
                           color=SUBSET_COLORS[subset], label=subset)
            for j, b in enumerate(bars):
                if not np.isnan(vals[j]):
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_height(),
                            f"n={ns[j]}",
                            ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x_base)
        ax.set_xticklabels([POSITION_LABELS[p] for p in POSITIONS])
        ax.set_ylabel("mean PLV")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9, title="subset")

    fig.suptitle("PLV by position and correctness subset", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_subset_comparison_power(metrics_grid, save_path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.27
    x_base = np.arange(len(POSITIONS))
    for i, subset in enumerate(["all", "correct", "incorrect"]):
        offset = (i - 1) * width
        vals, ns = [], []
        for pos in POSITIONS:
            m = metrics_grid[pos].get(subset)
            if m is None:
                vals.append(np.nan); ns.append(0)
            else:
                vals.append(m["bin1_mag_mean"])
                ns.append(m["_subset_n"])
        bars = ax.bar(x_base + offset, vals, width,
                       color=SUBSET_COLORS[subset], label=subset)
        for j, b in enumerate(bars):
            if not np.isnan(vals[j]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"n={ns[j]}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_base)
    ax.set_xticklabels([POSITION_LABELS[p] for p in POSITIONS])
    ax.set_ylabel("mean per-unit bin-1 magnitude")
    ax.set_title("Per-unit bin-1 magnitude by position and correctness subset",
                  fontsize=11)
    ax.legend(fontsize=9, title="subset")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_subset_comparison_cospec(metrics_grid, save_path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.27
    x_base = np.arange(len(POSITIONS))
    for i, subset in enumerate(["all", "correct", "incorrect"]):
        offset = (i - 1) * width
        vals, ns = [], []
        for pos in POSITIONS:
            m = metrics_grid[pos].get(subset)
            if m is None:
                vals.append(np.nan); ns.append(0)
            else:
                vals.append(m["cospec_mag_p95"])
                ns.append(m["_subset_n"])
        bars = ax.bar(x_base + offset, vals, width,
                       color=SUBSET_COLORS[subset], label=subset)
        for j, b in enumerate(bars):
            if not np.isnan(vals[j]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"n={ns[j]}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_base)
    ax.set_xticklabels([POSITION_LABELS[p] for p in POSITIONS])
    ax.set_ylabel("cospec_mag p95")
    ax.set_title("Cospec magnitude (p95) by position and correctness subset",
                  fontsize=11)
    ax.legend(fontsize=9, title="subset")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_decomposition_top100_correct_only(
    metrics_correct_only_by_pos, top_n, save_path
):
    """The disentangle decomposition, but on the correct_all subset
    (only samples the model gets right at every position)."""
    pos5 = metrics_correct_only_by_pos[5]
    if pos5 is None:
        print("  correct_all subset too small; skipping decomposition plot")
        return

    top_idx = np.argsort(-pos5["cospec_mag"])[:top_n]
    metrics_to_show = [
        ("cospec_mag",            "cospec magnitude"),
        ("amp_prod",              "mean amplitude product"),
        ("plv",                   "PLV (phase consistency)"),
        ("weighted_consistency",  "weighted consistency"),
    ]
    fig, axes = plt.subplots(1, len(metrics_to_show),
                             figsize=(5 * len(metrics_to_show), 5))
    for ax, (key, title) in zip(axes, metrics_to_show):
        data_per_pos = [
            metrics_correct_only_by_pos[pos][key][top_idx]
            for pos in POSITIONS
        ]
        bp = ax.boxplot(data_per_pos,
                         tick_labels=[POSITION_LABELS[p] for p in POSITIONS],
                         showfliers=True, patch_artist=True)
        for patch, pos in zip(bp["boxes"], POSITIONS):
            patch.set_facecolor(POSITION_COLORS[pos])
            patch.set_alpha(0.5)
        for i, vals in enumerate(data_per_pos):
            ax.text(i + 1, np.median(vals),
                     f"med={np.median(vals):.3f}",
                     ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=11)
    fig.suptitle(
        f"Top-{top_n} pairs at pos 5 (correct_all subset only): "
        f"tracked across positions",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reduction: compute scalar summary metrics from a metrics dict
# ---------------------------------------------------------------------------

def reduce_metrics(metrics, n):
    """Reduce per-position metrics to scalars for plotting / JSON."""
    if metrics is None:
        return None
    plv = metrics["plv"]
    amp = metrics["amp_prod"]
    high = amp >= np.percentile(amp, 75)
    return {
        "_subset_n":            int(n),
        "bin1_mag_mean":        float(metrics["bin1_mag"].mean()),
        "bin1_power_mean":      float(metrics["bin1_power"].mean()),
        "plv_all_mean":         float(plv.mean()),
        "plv_all_median":       float(np.median(plv)),
        "plv_high_amp_mean":    float(plv[high].mean()) if high.any() else None,
        "plv_high_amp_median":  float(np.median(plv[high])) if high.any() else None,
        "cospec_mag_mean":      float(metrics["cospec_mag"].mean()),
        "cospec_mag_median":    float(np.median(metrics["cospec_mag"])),
        "cospec_mag_p95":       float(np.percentile(metrics["cospec_mag"], 95)),
        # Pass through arrays for the decomposition plot
        "cospec_mag":           metrics["cospec_mag"],
        "amp_prod":             metrics["amp_prod"],
        "plv":                  metrics["plv"],
        "weighted_consistency": metrics["weighted_consistency"],
        "cospec_phase":         metrics["cospec_phase"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="toy_correctness_filter")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("\nCollecting streams and predictions ...")
    (streams_by_pos, sublayer_meta,
     correct_by_pos, preds, targets) = collect_streams_and_predictions(
        model, cfg
    )
    n_total = preds.shape[0]
    print(f"Total samples: {n_total}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Per-position accuracy
    print("\nPer-position accuracy:")
    accuracy = {}
    for pos in POSITIONS:
        acc = float(correct_by_pos[pos].mean())
        accuracy[pos] = acc
        print(f"  {POSITION_LABELS[pos]:>14}: {acc:.4f} "
              f"({int(correct_by_pos[pos].sum())}/{n_total})")
    correct_all = np.ones(n_total, dtype=bool)
    for pos in POSITIONS:
        correct_all &= correct_by_pos[pos]
    print(f"  correct at all 3 positions: "
          f"{int(correct_all.sum())}/{n_total} "
          f"({correct_all.mean():.4f})")

    with open(out_dir / "per_position_accuracy.txt", "w") as f:
        for pos in POSITIONS:
            f.write(
                f"{POSITION_LABELS[pos]}: {accuracy[pos]:.4f} "
                f"({int(correct_by_pos[pos].sum())}/{n_total})\n"
            )
        f.write(
            f"correct at all 3 positions: "
            f"{int(correct_all.sum())}/{n_total}\n"
        )

    # Build subsets
    subsets = build_subsets(correct_by_pos)

    # Compute metrics on each (position, subset) pair
    print("\nComputing metrics by (position, subset) ...")
    metrics_grid = {pos: {} for pos in POSITIONS}
    for pos in POSITIONS:
        for subset_key, subset_label in POS_SUBSETS[pos]:
            mask = subsets[subset_key]
            n = int(mask.sum())
            print(f"  pos {pos}, {subset_label} (n={n}) ...")
            m = safe_compute_metrics(streams_by_pos[pos], mask)
            metrics_grid[pos][subset_label] = reduce_metrics(m, n)

    # Compute metrics for the "correct at all positions" subset across positions
    print("\nComputing metrics on 'correct at all 3 positions' subset ...")
    metrics_correct_all_by_pos = {}
    for pos in POSITIONS:
        n = int(correct_all.sum())
        print(f"  pos {pos} (n={n}) ...")
        metrics_correct_all_by_pos[pos] = safe_compute_metrics(
            streams_by_pos[pos], correct_all
        )

    # ------- Plots -------
    print("\nMaking plots ...")
    plot_subset_comparison_plv(
        metrics_grid, out_dir / "subset_comparison_plv.png"
    )
    plot_subset_comparison_power(
        metrics_grid, out_dir / "subset_comparison_power.png"
    )
    plot_subset_comparison_cospec(
        metrics_grid, out_dir / "subset_comparison_cospec.png"
    )
    plot_decomposition_top100_correct_only(
        metrics_correct_all_by_pos, args.top_n,
        out_dir / "decomposition_top100_correct.png",
    )

    # ------- Verdict -------
    print("\n--- Verdict ---")
    plv_all_pos5 = metrics_grid[5]["all"]["plv_all_mean"]
    plv_correct_pos5 = (metrics_grid[5]["correct"]["plv_all_mean"]
                         if metrics_grid[5].get("correct") else None)
    plv_all_pos7 = metrics_grid[7]["all"]["plv_all_mean"]
    plv_correct_pos7 = (metrics_grid[7]["correct"]["plv_all_mean"]
                         if metrics_grid[7].get("correct") else None)

    print(f"\nPLV all-pairs mean:")
    print(f"  pos 5 all:        {plv_all_pos5:.4f}")
    if plv_correct_pos5 is not None:
        print(f"  pos 5 correct:    {plv_correct_pos5:.4f}")
    print(f"  pos 7 all:        {plv_all_pos7:.4f}")
    if plv_correct_pos7 is not None:
        print(f"  pos 7 correct:    {plv_correct_pos7:.4f}")
        delta = plv_correct_pos7 - plv_all_pos7
        gap_correct = (plv_correct_pos5 - plv_correct_pos7
                        if plv_correct_pos5 is not None else None)
        gap_all = plv_all_pos5 - plv_all_pos7
        print(f"\n  pos 7 PLV change when filtering to correct: "
              f"{delta:+.4f}")
        print(f"  pos 5 - pos 7 PLV gap (all):     {gap_all:+.4f}")
        if gap_correct is not None:
            print(f"  pos 5 - pos 7 PLV gap (correct): "
                  f"{gap_correct:+.4f}")
            ratio = gap_correct / gap_all if gap_all != 0 else None
            if ratio is not None:
                print(f"  remaining gap / original gap: "
                      f"{ratio:.2%}")
                if ratio > 0.7:
                    print("\n  >> Most of the gap survives filtering. "
                          "Mechanism difference is real. No retraining needed "
                          "for the qualitative conclusion.")
                elif ratio < 0.3:
                    print("\n  >> Most of the gap closes when filtering to "
                          "correct samples. The original difference was "
                          "largely a failure-mode confound. Use correct-only "
                          "subsets in future analyses.")
                else:
                    print("\n  >> Filtering closes part of the gap. "
                          "Mixed picture; consider training longer or "
                          "with d_model = 96 to firm up the conclusion.")

    # ------- Summary JSON -------
    def metrics_to_json(m):
        if m is None:
            return None
        # Drop arrays
        return {k: v for k, v in m.items()
                if k in ("_subset_n", "bin1_mag_mean", "bin1_power_mean",
                         "plv_all_mean", "plv_all_median",
                         "plv_high_amp_mean", "plv_high_amp_median",
                         "cospec_mag_mean", "cospec_mag_median",
                         "cospec_mag_p95")}

    summary = {
        "checkpoint":    str(args.checkpoint),
        "n_total":       int(n_total),
        "accuracy":      {str(p): float(accuracy[p]) for p in POSITIONS},
        "correct_all_n": int(correct_all.sum()),
        "metrics_grid": {
            str(pos): {
                subset: metrics_to_json(metrics_grid[pos].get(subset))
                for subset in ("all", "correct", "incorrect")
            }
            for pos in POSITIONS
        },
        "metrics_correct_all_subset": {
            str(pos): metrics_to_json(reduce_metrics(
                metrics_correct_all_by_pos[pos],
                int(correct_all.sum()),
            ))
            for pos in POSITIONS
        },
    }
    with open(out_dir / "correctness_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
