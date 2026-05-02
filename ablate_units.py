"""
Ablate specific residual units (zero their dimensions throughout the
forward pass) and compare the phase structure of the remaining units
to the unablated baseline.

What "ablate" means here:

  Forward hook on tok_emb, pos_emb, and on each block.attn and block.mlp
  zeroes the targeted dimensions of the module's *output*. Because the
  residual stream is built by adding these outputs, zero outputs in those
  dims means the residual stream's value in those dims stays at exactly
  0 at every sublayer. Other sublayers can still *read* from those dims
  (they just read zeros), they cannot *write* to them.

What we measure:

  Baseline (no ablation) and Ablated (units 13, 29 zeroed) at
  positions 5, 6, 7. Then for every non-ablated unit u and pair (u, v)
  with u, v not in the ablated set:

  - per-unit R (rotation count) and bin-1 power
  - per-pair cospec magnitude, PLV, phase
  - top-100 pair sets: how many survive, how many appear, how many disappear
  - task accuracy at each output position

Outputs in --out-dir:
  - accuracy_summary.json
  - per_unit_changes_pos{5,6,7}.png
  - plv_distribution_pos{5,6,7}.png
  - phase_diff_distribution_pos{5,6,7}.png
  - top100_changes_pos{5,6,7}.csv
  - top100_movement_pos{5,6,7}.png
  - ablation_summary.json

Usage:
  python ablate_units.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --units-to-ablate 13 29 \
      --out-dir toy_ablation_13_29
"""

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from toy_transformer_addition import (
    ToyTransformer,
    ModelConfig,
    AdditionDataset,
    capture_residual_streams,
    SEQ_LEN,
)
from torch.utils.data import DataLoader

from analyze_toy_residuals import (
    rotation_count_per_unit,
    per_unit_spectra,
    SEED,
)
from disentangle_cospec import (
    compute_metrics,
    classify_phase,
    POSITIONS,
    POSITION_LABELS,
    POSITION_COLORS,
)


# ---------------------------------------------------------------------------
# Ablation context manager
# ---------------------------------------------------------------------------

@contextmanager
def ablated(model, units):
    """Within this context, zero the listed residual dimensions at every
    sublayer write, so the residual stream's value in those dims stays
    at 0 throughout the forward pass."""
    units_t = list(units)
    handles = []

    def zero_dims_hook(module, inputs, output):
        out = output.clone()
        out[..., units_t] = 0.0
        return out

    handles.append(model.tok_emb.register_forward_hook(zero_dims_hook))
    handles.append(model.pos_emb.register_forward_hook(zero_dims_hook))
    for block in model.blocks:
        handles.append(block.attn.register_forward_hook(zero_dims_hook))
        handles.append(block.mlp.register_forward_hook(zero_dims_hook))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Loading and evaluation
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = ToyTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def evaluate_per_position(model, device, batch_size=64):
    """Per-position accuracy across all 256 (a, b) pairs."""
    pairs = [(a, b) for a in range(16) for b in range(16)]
    ds = AdditionDataset(pairs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # The dataset returns (x, y, mask) where x has length SEQ_LEN - 1 (positions 0..7)
    # The mask says "score loss at output indices 5, 6, 7" (predicting tokens at 6, 7, 8).
    # We want per-position accuracy at each of those output indices.
    correct_at = {5: 0, 6: 0, 7: 0}
    total = 0
    full_correct = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            for pos in (5, 6, 7):
                correct_at[pos] += (preds[:, pos] == y[:, pos]).sum().item()
            seq_correct = (
                (preds[:, 5] == y[:, 5])
                & (preds[:, 6] == y[:, 6])
                & (preds[:, 7] == y[:, 7])
            )
            full_correct += seq_correct.sum().item()
            total += x.size(0)
    return {
        "n":             total,
        "acc_pos5":      correct_at[5] / total,
        "acc_pos6":      correct_at[6] / total,
        "acc_pos7":      correct_at[7] / total,
        "acc_full_seq":  full_correct / total,
    }


def collect_streams_at_positions(model, batch_size=64):
    """Capture cumulative residuals; return {pos: (N, n_sublayers, D)}."""
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]
    _, captures = capture_residual_streams(
        model, pairs, device, batch_size=batch_size
    )
    sublayer_meta = [(c["kind"], c["layer"]) for c in captures]
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    streams_by_pos = {}
    for pos in POSITIONS:
        s = stacked[:, :, pos, :].permute(1, 0, 2).contiguous()
        streams_by_pos[pos] = s.numpy().astype(np.float32)
    return streams_by_pos, sublayer_meta


# ---------------------------------------------------------------------------
# Filtering metrics to exclude ablated units
# ---------------------------------------------------------------------------

def filter_metrics_excluding(metrics, ablated_set):
    """Return a copy with per-unit and per-pair arrays restricted to
    units / pairs that exclude the ablated set."""
    u_idx = metrics["u_idx"]
    v_idx = metrics["v_idx"]
    keep_pair = np.array([
        (int(u) not in ablated_set) and (int(v) not in ablated_set)
        for u, v in zip(u_idx, v_idx)
    ])
    D = metrics["bin1_power"].shape[0]
    keep_unit = np.array([u not in ablated_set for u in range(D)])

    return {
        "freq_bin":             metrics["freq_bin"],
        "bin1_power":           metrics["bin1_power"][keep_unit],
        "bin1_mag":             metrics["bin1_mag"][keep_unit],
        "loudness":             metrics["loudness"][keep_unit],
        "R":                    metrics["R"][keep_unit],
        "unit_indices":         np.where(keep_unit)[0],
        "u_idx":                u_idx[keep_pair],
        "v_idx":                v_idx[keep_pair],
        "cospec_mag":           metrics["cospec_mag"][keep_pair],
        "cospec_phase":         metrics["cospec_phase"][keep_pair],
        "amp_prod":             metrics["amp_prod"][keep_pair],
        "plv":                  metrics["plv"][keep_pair],
        "weighted_consistency": metrics["weighted_consistency"][keep_pair],
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_per_unit_changes(filt_base, filt_abl, pos, save_path):
    """Scatter: baseline vs ablated for R, bin-1 power, loudness."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_to_plot = [
        ("R",          "rotation count R"),
        ("bin1_power", "per-unit bin-1 power"),
        ("loudness",   "mean |activation|"),
    ]
    color = POSITION_COLORS[pos]
    for ax, (key, label) in zip(axes, metrics_to_plot):
        x = filt_base[key]
        y = filt_abl[key]
        ax.scatter(x, y, s=18, alpha=0.7, c=color, edgecolors="none")
        # Diagonal
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.5)
        # Annotate top-shift units
        diffs = y - x
        order = np.argsort(-np.abs(diffs))
        unit_idx = filt_base["unit_indices"]
        for k in order[:5]:
            ax.annotate(
                f"{int(unit_idx[k])}", (x[k], y[k]),
                fontsize=7, alpha=0.8,
                xytext=(2, 2), textcoords="offset points",
            )
        ax.set_xlabel(f"baseline {label}")
        ax.set_ylabel(f"ablated {label}")
        ax.set_title(label, fontsize=10)
    fig.suptitle(
        f"Per-unit metrics, baseline vs ablated  ({POSITION_LABELS[pos]})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_plv_distribution(filt_base, filt_abl, pos, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for label, data, ls in [
        ("baseline", filt_base, "-"),
        ("ablated",  filt_abl,  "--"),
    ]:
        ax.hist(
            data["plv"], bins=40, range=(0, 1), density=True,
            histtype="step", linewidth=1.5, linestyle=ls,
            label=f"{label} (mean={data['plv'].mean():.3f})",
        )
    ax.set_xlabel("PLV")
    ax.set_ylabel("density")
    ax.set_title("All non-ablated pairs")
    ax.legend(fontsize=9)

    ax = axes[1]
    base_thresh = np.percentile(filt_base["amp_prod"], 75)
    base_keep = filt_base["amp_prod"] >= base_thresh
    abl_thresh = np.percentile(filt_abl["amp_prod"], 75)
    abl_keep = filt_abl["amp_prod"] >= abl_thresh
    for label, data, mask, ls in [
        ("baseline", filt_base, base_keep, "-"),
        ("ablated",  filt_abl,  abl_keep,  "--"),
    ]:
        ax.hist(
            data["plv"][mask], bins=40, range=(0, 1), density=True,
            histtype="step", linewidth=1.5, linestyle=ls,
            label=f"{label} (mean={data['plv'][mask].mean():.3f})",
        )
    ax.set_xlabel("PLV")
    ax.set_ylabel("density")
    ax.set_title("Pairs above own 75th-percentile amp_prod")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"PLV distribution, baseline vs ablated  ({POSITION_LABELS[pos]})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_distribution(filt_base, filt_abl, pos, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for label, data, ls in [
        ("baseline", filt_base, "-"),
        ("ablated",  filt_abl,  "--"),
    ]:
        ax.hist(
            data["cospec_phase"], bins=60, range=(-np.pi, np.pi),
            density=True, histtype="step", linewidth=1.5,
            linestyle=ls, label=label,
        )
    for x in (-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi):
        ax.axvline(x, color="gray", linewidth=0.4, linestyle=":")
    ax.set_xlabel("phase difference at bin 1")
    ax.set_ylabel("density")
    ax.set_title("All non-ablated pairs")
    ax.legend(fontsize=9)

    ax = axes[1]
    base_thresh = np.percentile(filt_base["amp_prod"], 75)
    abl_thresh = np.percentile(filt_abl["amp_prod"], 75)
    base_keep = filt_base["amp_prod"] >= base_thresh
    abl_keep = filt_abl["amp_prod"] >= abl_thresh
    for label, data, mask, ls in [
        ("baseline", filt_base, base_keep, "-"),
        ("ablated",  filt_abl,  abl_keep,  "--"),
    ]:
        ax.hist(
            data["cospec_phase"][mask], bins=60, range=(-np.pi, np.pi),
            density=True, histtype="step", linewidth=1.5,
            linestyle=ls, label=label,
        )
    for x in (-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi):
        ax.axvline(x, color="gray", linewidth=0.4, linestyle=":")
    ax.set_xlabel("phase difference at bin 1")
    ax.set_ylabel("density")
    ax.set_title("Pairs above own 75th-percentile amp_prod")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Phase difference distribution, baseline vs ablated  "
        f"({POSITION_LABELS[pos]})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def top_pair_changes(filt_base, filt_abl, top_n=100):
    """For each pair in either top-N, return its rank change and metric change."""
    base_order = np.argsort(-filt_base["cospec_mag"])
    abl_order = np.argsort(-filt_abl["cospec_mag"])
    # Map pair-index -> rank
    base_rank = np.empty(len(base_order), dtype=int)
    base_rank[base_order] = np.arange(len(base_order))
    abl_rank = np.empty(len(abl_order), dtype=int)
    abl_rank[abl_order] = np.arange(len(abl_order))

    base_top = set(base_order[:top_n].tolist())
    abl_top = set(abl_order[:top_n].tolist())
    union = sorted(base_top | abl_top)

    rows = []
    for idx in union:
        u = int(filt_base["u_idx"][idx])
        v = int(filt_base["v_idx"][idx])
        rows.append({
            "u": u, "v": v,
            "rank_base":         int(base_rank[idx]) + 1,
            "rank_abl":          int(abl_rank[idx]) + 1,
            "in_top_base":       idx in base_top,
            "in_top_abl":        idx in abl_top,
            "cospec_base":       float(filt_base["cospec_mag"][idx]),
            "cospec_abl":        float(filt_abl["cospec_mag"][idx]),
            "cospec_change":     float(filt_abl["cospec_mag"][idx]
                                       - filt_base["cospec_mag"][idx]),
            "plv_base":          float(filt_base["plv"][idx]),
            "plv_abl":           float(filt_abl["plv"][idx]),
            "phase_base":        float(filt_base["cospec_phase"][idx]),
            "phase_abl":         float(filt_abl["cospec_phase"][idx]),
            "cluster_base":      classify_phase(filt_base["cospec_phase"][idx]),
            "cluster_abl":       classify_phase(filt_abl["cospec_phase"][idx]),
        })
    rows.sort(key=lambda r: r["rank_base"])

    inter = base_top & abl_top
    only_base = base_top - abl_top
    only_abl = abl_top - base_top
    return rows, {
        "intersection": len(inter),
        "only_baseline": len(only_base),
        "only_ablated": len(only_abl),
        "jaccard": len(inter) / len(base_top | abl_top)
                    if (base_top | abl_top) else 0.0,
    }


def plot_top_pair_movement(rows, pos, save_path):
    """Plot rank_base vs rank_abl for pairs in either top-N."""
    fig, ax = plt.subplots(figsize=(7, 7))
    survived = [r for r in rows if r["in_top_base"] and r["in_top_abl"]]
    only_base = [r for r in rows if r["in_top_base"] and not r["in_top_abl"]]
    only_abl = [r for r in rows if not r["in_top_base"] and r["in_top_abl"]]

    if survived:
        ax.scatter(
            [r["rank_base"] for r in survived],
            [r["rank_abl"]  for r in survived],
            s=22, alpha=0.7, label=f"in both top-100 (n={len(survived)})",
        )
    if only_base:
        ax.scatter(
            [r["rank_base"] for r in only_base],
            [r["rank_abl"]  for r in only_base],
            s=22, alpha=0.6, marker="x",
            label=f"only baseline top-100 (n={len(only_base)})",
        )
    if only_abl:
        ax.scatter(
            [r["rank_base"] for r in only_abl],
            [r["rank_abl"]  for r in only_abl],
            s=22, alpha=0.6, marker="^",
            label=f"only ablated top-100 (n={len(only_abl)})",
        )

    max_rank = max(
        [r["rank_base"] for r in rows] + [r["rank_abl"] for r in rows]
    ) if rows else 1
    ax.plot([1, max_rank], [1, max_rank], "k--",
            linewidth=0.6, alpha=0.5)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(100, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("baseline cospec rank")
    ax.set_ylabel("ablated cospec rank")
    ax.set_title(
        f"Top-100 pair movement under ablation  ({POSITION_LABELS[pos]})",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="toy_ablation")
    parser.add_argument("--units-to-ablate", type=int, nargs="+",
                        required=True)
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
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}")
    print(f"  Ablating residual units: {args.units_to_ablate}")
    ablated_set = set(int(u) for u in args.units_to_ablate)
    if any(u >= cfg.d_model for u in ablated_set):
        raise ValueError(
            f"Unit index out of range; d_model={cfg.d_model}"
        )

    # ---------- Accuracy: baseline vs ablated ----------
    print("\nEvaluating accuracy ...")
    acc_baseline = evaluate_per_position(model, device)
    with ablated(model, ablated_set):
        acc_ablated = evaluate_per_position(model, device)
    accuracy_summary = {
        "baseline":     acc_baseline,
        "ablated":      acc_ablated,
        "ablated_units": sorted(ablated_set),
    }
    with open(out_dir / "accuracy_summary.json", "w") as f:
        json.dump(accuracy_summary, f, indent=2)

    print("\nAccuracy:")
    print(f"  {'metric':<14} {'baseline':>10} {'ablated':>10}")
    for key in ("acc_pos5", "acc_pos6", "acc_pos7", "acc_full_seq"):
        print(f"  {key:<14} {acc_baseline[key]:>10.3f} "
              f"{acc_ablated[key]:>10.3f}")

    # ---------- Capture streams: baseline vs ablated ----------
    print("\nCapturing baseline streams ...")
    streams_base, sublayer_meta = collect_streams_at_positions(model)
    print("Capturing ablated streams ...")
    with ablated(model, ablated_set):
        streams_abl, _ = collect_streams_at_positions(model)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Sanity check: ablated dims should be exactly zero
    for pos in POSITIONS:
        for u in ablated_set:
            mx = float(np.max(np.abs(streams_abl[pos][:, :, u])))
            assert mx < 1e-6, (
                f"Ablation failed at pos {pos}, unit {u}: max |x|={mx}"
            )
    print("Ablation sanity check passed: ablated dims are exactly zero.")

    # ---------- Per-position metric comparison ----------
    print("\nComputing metrics ...")
    summary_per_pos = {}
    for pos in POSITIONS:
        print(f"\n=== Position {pos} ({POSITION_LABELS[pos]}) ===")
        m_base = compute_metrics(streams_base[pos])
        m_abl  = compute_metrics(streams_abl[pos])
        filt_base = filter_metrics_excluding(m_base, ablated_set)
        filt_abl  = filter_metrics_excluding(m_abl,  ablated_set)

        plot_per_unit_changes(
            filt_base, filt_abl, pos,
            out_dir / f"per_unit_changes_pos{pos}.png",
        )
        plot_plv_distribution(
            filt_base, filt_abl, pos,
            out_dir / f"plv_distribution_pos{pos}.png",
        )
        plot_phase_distribution(
            filt_base, filt_abl, pos,
            out_dir / f"phase_diff_distribution_pos{pos}.png",
        )

        rows, top_overlap = top_pair_changes(
            filt_base, filt_abl, top_n=args.top_n
        )
        with open(out_dir / f"top{args.top_n}_changes_pos{pos}.csv", "w",
                  newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for row in rows:
                    w.writerow(row)
        plot_top_pair_movement(
            rows, pos,
            out_dir / f"top{args.top_n}_movement_pos{pos}.png",
        )

        # Aggregate stats
        def stats(arr):
            return {
                "mean":   float(arr.mean()),
                "median": float(np.median(arr)),
                "p95":    float(np.percentile(arr, 95)),
            }

        summary_per_pos[str(pos)] = {
            "label":            POSITION_LABELS[pos],
            "n_units_kept":     int(filt_base["bin1_power"].shape[0]),
            "n_pairs_kept":     int(filt_base["plv"].shape[0]),
            "baseline": {
                "bin1_mag":   stats(filt_base["bin1_mag"]),
                "bin1_power": stats(filt_base["bin1_power"]),
                "plv":        stats(filt_base["plv"]),
                "cospec_mag": stats(filt_base["cospec_mag"]),
                "abs_R":      stats(np.abs(filt_base["R"])),
            },
            "ablated": {
                "bin1_mag":   stats(filt_abl["bin1_mag"]),
                "bin1_power": stats(filt_abl["bin1_power"]),
                "plv":        stats(filt_abl["plv"]),
                "cospec_mag": stats(filt_abl["cospec_mag"]),
                "abs_R":      stats(np.abs(filt_abl["R"])),
            },
            "top_n_overlap": top_overlap,
        }

        print(f"  pairs kept (excluding ablated units): "
              f"{summary_per_pos[str(pos)]['n_pairs_kept']}")
        print(f"  baseline mean PLV={filt_base['plv'].mean():.3f}, "
              f"ablated mean PLV={filt_abl['plv'].mean():.3f}")
        print(f"  baseline median cospec_mag="
              f"{np.median(filt_base['cospec_mag']):.2f}, "
              f"ablated="
              f"{np.median(filt_abl['cospec_mag']):.2f}")
        print(f"  top-{args.top_n} overlap: "
              f"{top_overlap['intersection']}/{args.top_n} "
              f"(jaccard {top_overlap['jaccard']:.3f})")

    # ---------- Final summary ----------
    summary = {
        "checkpoint":     str(args.checkpoint),
        "ablated_units":  sorted(ablated_set),
        "d_model":        int(cfg.d_model),
        "n_pairs_total_per_position": int(
            cfg.d_model * (cfg.d_model - 1) // 2
        ),
        "accuracy":       accuracy_summary,
        "per_position":   summary_per_pos,
    }
    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
