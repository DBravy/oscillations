"""
Causal intervention to distinguish Route 1 (shared weights acting on
similar local content) from Route 2 (active transfer via attention) for
the cross-position Fiedler-vector overlap.

Three conditions:

  clean:        all 256 (a, b) pairs, normal tokens at every position
  corrupt_p7:   token at position 7 (c1) replaced with a random digit
                (0..3) per input, drawn independently. Stays
                in-distribution since c1 is normally a digit.
  corrupt_p5:   token at position 5 ("=") replaced with a random digit
                per input. OOD since "=" is the only training-time token
                at p5.

Predictions:

  Route 1   corrupt_p7: overlap p5<->p7 largely preserved
            corrupt_p5: overlap p5<->p7 largely preserved (p7 doing its
                        usual thing on its own content)

  Route 2   corrupt_p7: overlap p5<->p7 largely preserved
            corrupt_p5: overlap p5<->p7 collapses (the transfer source
                        is corrupted)

Both predict that corrupt_p7 looks normal. The discriminating signal is
whether corrupt_p5 collapses the p5<->p7 overlap or not.

The same coupling and Fiedler construction as sl_step3b is used:
  z_u(n, l) = (a_u(n,l) - a_u(n,0)) + i*(grad_u(n,l) - grad_u(n,0))
  C^p(l)[u, v] = | mean_n z_u^p(n, l) * conj(z_v^p(n, l)) |
  v_2 = Fiedler vector of L = D - C

Outputs in --out-dir:
  overlap_curves_by_condition.png    main; one panel per pair
  fiedler_stability_per_position.png |v_2_clean . v_2_corrupted| per
                                     position; how much does each
                                     position's Fiedler vector change
                                     under each corruption?
  final_layer_overlap_summary.png    bar chart of headline numbers
  intervention_summary.json          numerical results

Usage:
  python intervention_swap_inputs.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_intervention_trained
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from toy_transformer_addition import (
    ToyTransformer,
    ModelConfig,
    encode_example,
    DIGIT_TOKENS,
)


POSITIONS = [5, 6, 7]
POSITION_LABELS = {
    5: 'pos 5 ("=")',
    6: "pos 6 (c2)",
    7: "pos 7 (c1)",
}
POSITION_COLORS = {5: "C0", 6: "C1", 7: "C2"}

CONDITIONS = ["clean", "corrupt_p7", "corrupt_p5"]
CONDITION_LABELS = {
    "clean":      "clean",
    "corrupt_p7": "corrupt p7 (c1 -> rand digit)",
    "corrupt_p5": "corrupt p5 ('=' -> rand digit)",
}
CONDITION_COLORS = {
    "clean":      "tab:blue",
    "corrupt_p7": "tab:green",
    "corrupt_p5": "tab:red",
}
CONDITION_STYLES = {
    "clean":      "-",
    "corrupt_p7": "--",
    "corrupt_p5": ":",
}


# ---------------------------------------------------------------------------
# Loading and corruption
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = ToyTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def build_inputs(corrupt_position, seed=42):
    """Return (N, T) input tensor and a record of what was corrupted."""
    pairs = [(a, b) for a in range(16) for b in range(16)]
    rng = np.random.default_rng(seed)

    seqs = []
    corruption_log = []
    for (a, b) in pairs:
        toks = encode_example(a, b)
        original = toks[corrupt_position] if corrupt_position is not None else None
        if corrupt_position is not None:
            new_token = int(rng.choice(DIGIT_TOKENS))
            toks[corrupt_position] = new_token
            corruption_log.append({
                "a": a, "b": b,
                "original": original, "new": new_token,
            })
        # Model input is length 8 (positions 0..7)
        seqs.append(toks[:-1])
    x = torch.tensor(seqs, dtype=torch.long)
    return x, corruption_log


# ---------------------------------------------------------------------------
# Forward + residual capture
# ---------------------------------------------------------------------------

@torch.no_grad()
def capture_residuals_for_inputs(model, x, batch_size=64):
    """Run the model on x: (N, T), return per-sublayer residuals as a
    list of dicts, then a stacked (n_sublayers, N, T, D) array."""
    device = next(model.parameters()).device
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    captures_by_sublayer = None
    for (xb,) in loader:
        xb = xb.to(device)
        _, capture = model(xb, capture_residuals=True)
        if captures_by_sublayer is None:
            captures_by_sublayer = [[] for _ in capture]
        for i, (_, _, t) in enumerate(capture):
            captures_by_sublayer[i].append(t)

    stacked = []
    for chunks in captures_by_sublayer:
        stacked.append(torch.cat(chunks, dim=0))
    full = torch.stack(stacked, dim=0)              # (n_sublayers, N, T, D)
    return full.numpy().astype(np.float32)


def streams_at_position(full, pos):
    """full: (n_sublayers, N, T, D). Returns (N, n_sublayers, D)."""
    at_pos = full[:, :, pos, :]                     # (n_sublayers, N, D)
    return np.transpose(at_pos, (1, 0, 2))          # (N, n_sublayers, D)


# ---------------------------------------------------------------------------
# Fiedler machinery (same as step 3b)
# ---------------------------------------------------------------------------

def per_unit_z(streams):
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    return x + 1j * y


def coupling_matrix_at_layer(z_layer):
    cs = z_layer[:, :, None] * np.conj(z_layer[:, None, :])
    cs_mean = cs.mean(axis=0)
    C = np.abs(cs_mean)
    np.fill_diagonal(C, 0.0)
    return C


def laplacian_eigendecomp(C):
    deg = C.sum(axis=1)
    L = np.diag(deg) - C
    L = (L + L.T) / 2.0
    w, V = np.linalg.eigh(L)
    return w, V


def per_layer_fiedler(streams, n_eig=4):
    """streams: (N, L, D). Returns (L, D, n_eig) eigenvectors and
    (L, n_eig) eigenvalues, with NaN at layer 0 (degenerate)."""
    N, L, D = streams.shape
    z = per_unit_z(streams)
    eigvecs = np.full((L, D, n_eig), np.nan)
    eigvals = np.full((L, n_eig), np.nan)
    for l in range(L):
        if l == 0:
            continue
        C = coupling_matrix_at_layer(z[:, l, :])
        w, V = laplacian_eigendecomp(C)
        eigvals[l] = w[:n_eig]
        eigvecs[l] = V[:, :n_eig]
    return eigvals, eigvecs


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overlap_curves(overlaps_by_cond_pair, start, D, save_path):
    """One subplot per position-pair, three lines per subplot (one per
    condition)."""
    pairs = [(5, 6), (5, 7), (6, 7)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (p, q) in zip(axes, pairs):
        for cond in CONDITIONS:
            O = overlaps_by_cond_pair[cond][(p, q)]
            xs = np.arange(start, len(O))
            ax.plot(
                xs, O[start:],
                CONDITION_STYLES[cond],
                color=CONDITION_COLORS[cond],
                linewidth=1.8,
                label=(
                    f"{CONDITION_LABELS[cond]}: "
                    f"final={O[-1]:.3f}"
                ),
            )
        ax.axhline(2 / np.sqrt(np.pi * D), color="gray",
                   linewidth=0.6, linestyle="--",
                   label=f"random baseline (D={D})")
        ax.set_xlabel("sublayer l")
        ax.set_title(f"pos {p} <-> pos {q}", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Fiedler overlap |v_2^p . v_2^q|")
    fig.suptitle(
        "Cross-position Fiedler overlap under intervention",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_fiedler_stability(stability_by_pos_cond, start, D, save_path):
    """For each position p, plot |v_2_clean(p, l) . v_2_corrupt(p, l)|
    across layers, one line per corruption."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, p in zip(axes, POSITIONS):
        for cond in ["corrupt_p7", "corrupt_p5"]:
            S = stability_by_pos_cond[p][cond]
            xs = np.arange(start, len(S))
            ax.plot(
                xs, S[start:],
                CONDITION_STYLES[cond],
                color=CONDITION_COLORS[cond],
                linewidth=1.8,
                label=(
                    f"{CONDITION_LABELS[cond]}: "
                    f"final={S[-1]:.3f}"
                ),
            )
        ax.axhline(2 / np.sqrt(np.pi * D), color="gray",
                   linewidth=0.6, linestyle="--",
                   label=f"random baseline (D={D})")
        ax.set_xlabel("sublayer l")
        ax.set_title(POSITION_LABELS[p], fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel(
        "|v_2_clean(p, l) . v_2_corrupted(p, l)|"
    )
    fig.suptitle(
        "Fiedler stability per position: how much does v_2(p) change?",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_final_layer_summary(overlaps_by_cond_pair, save_path):
    """Bar chart of final-layer overlaps for each pair, grouped by
    condition. Quickest read of the headline numbers."""
    pairs = [(5, 6), (5, 7), (6, 7)]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.25
    x_base = np.arange(len(pairs))
    for i, cond in enumerate(CONDITIONS):
        vals = [
            float(overlaps_by_cond_pair[cond][(p, q)][-1])
            for (p, q) in pairs
        ]
        offset = (i - 1) * width
        ax.bar(
            x_base + offset, vals, width,
            color=CONDITION_COLORS[cond],
            label=CONDITION_LABELS[cond],
        )
        for j, v in enumerate(vals):
            ax.text(
                x_base[j] + offset, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8,
            )
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"p{p}<->p{q}" for (p, q) in pairs])
    ax.set_ylabel("Fiedler overlap at final sublayer")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Final-layer overlap by condition (headline)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str,
                        default="toy_intervention")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=1,
                        help="First sublayer to include in plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}")

    # ------------- Build inputs for each condition -------------
    print("\nBuilding inputs ...")
    inputs = {}
    corruption_logs = {}
    inputs["clean"], corruption_logs["clean"] = build_inputs(
        corrupt_position=None, seed=args.seed,
    )
    inputs["corrupt_p7"], corruption_logs["corrupt_p7"] = build_inputs(
        corrupt_position=7, seed=args.seed + 1,
    )
    inputs["corrupt_p5"], corruption_logs["corrupt_p5"] = build_inputs(
        corrupt_position=5, seed=args.seed + 2,
    )
    for cond in CONDITIONS:
        if cond == "clean":
            print(f"  {cond}: shape {inputs[cond].shape}")
        else:
            log = corruption_logs[cond]
            n_changed = sum(1 for r in log if r["original"] != r["new"])
            n_total = len(log)
            print(f"  {cond}: shape {inputs[cond].shape}; "
                  f"{n_changed}/{n_total} tokens changed by corruption")

    # ------------- Capture residuals -------------
    print("\nCapturing residuals for each condition ...")
    streams_by_cond_pos = {}
    for cond in CONDITIONS:
        print(f"  {cond} ...")
        full = capture_residuals_for_inputs(model, inputs[cond])
        streams_by_cond_pos[cond] = {
            pos: streams_at_position(full, pos) for pos in POSITIONS
        }
        print(
            f"    streams shape per position: "
            f"{streams_by_cond_pos[cond][5].shape}"
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------- Fiedler vectors per (cond, pos, layer) -------------
    print("\nComputing Fiedler vectors ...")
    fiedler_by_cond_pos = {}      # cond -> pos -> (L, D, n_eig)
    eigvals_by_cond_pos = {}
    for cond in CONDITIONS:
        fiedler_by_cond_pos[cond] = {}
        eigvals_by_cond_pos[cond] = {}
        for pos in POSITIONS:
            ev, V = per_layer_fiedler(streams_by_cond_pos[cond][pos])
            fiedler_by_cond_pos[cond][pos] = V       # (L, D, n_eig)
            eigvals_by_cond_pos[cond][pos] = ev

    # ------------- Cross-position overlaps -------------
    print("Computing cross-position overlaps ...")
    pairs = [(5, 6), (5, 7), (6, 7)]
    overlaps_by_cond_pair = {cond: {} for cond in CONDITIONS}
    for cond in CONDITIONS:
        for (p, q) in pairs:
            Vp = fiedler_by_cond_pos[cond][p][:, :, 1]    # (L, D)
            Vq = fiedler_by_cond_pos[cond][q][:, :, 1]
            O = np.abs((Vp * Vq).sum(axis=1))             # (L,)
            overlaps_by_cond_pair[cond][(p, q)] = O

    # ------------- Per-position Fiedler stability -------------
    # |v_2_clean(p, l) . v_2_corrupted(p, l)|, layer by layer
    print("Computing per-position Fiedler stability ...")
    stability_by_pos_cond = {pos: {} for pos in POSITIONS}
    for pos in POSITIONS:
        Vc = fiedler_by_cond_pos["clean"][pos][:, :, 1]
        for cond in ["corrupt_p7", "corrupt_p5"]:
            Vk = fiedler_by_cond_pos[cond][pos][:, :, 1]
            S = np.abs((Vc * Vk).sum(axis=1))
            stability_by_pos_cond[pos][cond] = S

    # ------------- Plots -------------
    print("\nMaking plots ...")
    D = cfg.d_model
    plot_overlap_curves(
        overlaps_by_cond_pair, args.start, D,
        out_dir / "overlap_curves_by_condition.png",
    )
    plot_fiedler_stability(
        stability_by_pos_cond, args.start, D,
        out_dir / "fiedler_stability_per_position.png",
    )
    plot_final_layer_summary(
        overlaps_by_cond_pair,
        out_dir / "final_layer_overlap_summary.png",
    )

    # ------------- Save & summary -------------
    save_dict = {}
    for cond in CONDITIONS:
        for pos in POSITIONS:
            save_dict[f"V_traj_{cond}_pos{pos}"] = (
                fiedler_by_cond_pos[cond][pos]
            )
        for (p, q) in pairs:
            save_dict[f"overlap_{cond}_pos{p}_pos{q}"] = (
                overlaps_by_cond_pair[cond][(p, q)]
            )
    for pos in POSITIONS:
        for cond in ["corrupt_p7", "corrupt_p5"]:
            save_dict[f"stability_{cond}_pos{pos}"] = (
                stability_by_pos_cond[pos][cond]
            )
    np.savez(out_dir / "intervention_data.npz", **save_dict)

    summary = {
        "checkpoint": str(args.checkpoint),
        "d_model":    int(cfg.d_model),
        "n_sublayers": int(streams_by_cond_pos["clean"][5].shape[1]),
        "seed":       args.seed,
        "random_baseline_2_over_sqrt_piD": float(
            2 / np.sqrt(np.pi * cfg.d_model)
        ),
        "final_layer_overlaps": {
            cond: {
                f"pos{p}_pos{q}": float(
                    overlaps_by_cond_pair[cond][(p, q)][-1]
                )
                for (p, q) in pairs
            }
            for cond in CONDITIONS
        },
        "final_layer_stability": {
            cond: {
                f"pos{pos}": float(
                    stability_by_pos_cond[pos][cond][-1]
                )
                for pos in POSITIONS
            }
            for cond in ["corrupt_p7", "corrupt_p5"]
        },
        "drop_relative_to_clean": {
            cond: {
                f"pos{p}_pos{q}": float(
                    overlaps_by_cond_pair["clean"][(p, q)][-1]
                    - overlaps_by_cond_pair[cond][(p, q)][-1]
                )
                for (p, q) in pairs
            }
            for cond in ["corrupt_p7", "corrupt_p5"]
        },
        "n_inputs_with_changed_token": {
            cond: sum(
                1 for r in corruption_logs[cond]
                if r["original"] != r["new"]
            )
            for cond in ["corrupt_p7", "corrupt_p5"]
        },
    }
    with open(out_dir / "intervention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------- Console summary -------------
    print("\n--- Final-layer overlaps by condition ---")
    print(f"{'pair':<14} "
          + "  ".join(f"{CONDITION_LABELS[c]:>26}" for c in CONDITIONS))
    for (p, q) in pairs:
        vals = [
            overlaps_by_cond_pair[cond][(p, q)][-1]
            for cond in CONDITIONS
        ]
        print(f"p{p}<->p{q:<10}  "
              + "  ".join(f"{v:>26.4f}" for v in vals))

    print("\n--- Drop relative to clean (positive = overlap reduced) ---")
    for cond in ["corrupt_p7", "corrupt_p5"]:
        print(f"\n{CONDITION_LABELS[cond]}:")
        for (p, q) in pairs:
            drop = (
                overlaps_by_cond_pair["clean"][(p, q)][-1]
                - overlaps_by_cond_pair[cond][(p, q)][-1]
            )
            print(f"  p{p}<->p{q}: drop={drop:+.4f}")

    print("\n--- Per-position Fiedler stability "
          "|v_2_clean . v_2_corrupted| (final sublayer) ---")
    for cond in ["corrupt_p7", "corrupt_p5"]:
        print(f"\n{CONDITION_LABELS[cond]}:")
        for pos in POSITIONS:
            s = stability_by_pos_cond[pos][cond][-1]
            print(f"  {POSITION_LABELS[pos]}: stability={s:.4f}")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
