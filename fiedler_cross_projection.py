"""
Cross-position Fiedler projection probe.

For each (probe_pos, target_pos) and each sublayer, use probe_pos's
Fiedler vector as the projection axis but project target_pos's residuals
onto it. Then ask: which task variable is best explained by this
cross-position projection?

The hypothesis being tested:

  In SwiGLU, the intervention test showed p6 is sensitive to corrupting
  p5 (Route 2 signature for the p5<->p6 channel). The within-position
  probe showed p6 has a1+b1 at eta^2 = 0.91. The within-position probe
  on p5 also shows strong a1+b1 encoding.

  Question: is the SAME representational direction shared? If we use
  p5's Fiedler vector to project p6's residuals, do we still recover
  a1+b1 with high eta^2?

  - If YES: p5 and p6 share the same encoding axis for a1+b1. p5 serves
    as a workbench whose contents p6 reads via attention.
  - If NO (cross-position projection drops eta^2 substantially relative
    to within-position): p5 and p6 each have their own Fiedler-aligned
    encoding direction; they don't share an axis even though both encode
    the same variable.

For each (probe_pos, target_pos, sublayer):
  - eta^2 of target's residuals projected onto probe's v_2 (1-D)
  - eta^2 of target's residuals projected onto probe's v_2..v_5 (4-D)

Outputs in --out-dir:
  cross_proj_heatmap_probe{P}_target{T}_{1D,4D}.png   per (probe, target)
  within_vs_cross_top.png                              best variable per
                                                        (probe, target),
                                                        within vs cross
  cross_proj_summary.json                              full table
  evidence_top.png                                     top-N findings,
                                                        proj distributions
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from toy_transformer_addition import (
    ToyTransformer,
    ModelConfig,
    capture_residual_streams,
)


POSITIONS = [5, 6, 7]
POSITION_LABELS = {
    5: 'pos 5 ("=")',
    6: "pos 6 (c2)",
    7: "pos 7 (c1)",
}
POSITION_COLORS = {5: "C0", 6: "C1", 7: "C2"}


# ---------------------------------------------------------------------------
# Task variables (same as fiedler_probe.py)
# ---------------------------------------------------------------------------

def task_variables_for_pair(a, b):
    a1, a0 = a // 4, a % 4
    b1, b0 = b // 4, b % 4
    c = a + b
    c0 = c % 4
    c1 = (c // 4) % 4
    c2 = c // 16
    carry_0_to_1 = 1 if (a0 + b0) >= 4 else 0
    carry_1_to_2 = 1 if (a1 + b1 + carry_0_to_1) >= 4 else 0
    return {
        "a": a, "b": b,
        "a0": a0, "a1": a1, "b0": b0, "b1": b1,
        "c0": c0, "c1": c1, "c2": c2,
        "carry_0_to_1":  carry_0_to_1,
        "carry_1_to_2":  carry_1_to_2,
        "a0+b0":         a0 + b0,
        "a1+b1":         a1 + b1,
    }


def build_task_table():
    pairs = [(a, b) for a in range(16) for b in range(16)]
    rows = [task_variables_for_pair(a, b) for (a, b) in pairs]
    keys = list(rows[0].keys())
    table = {k: np.array([r[k] for r in rows]) for k in keys}
    return table


# ---------------------------------------------------------------------------
# Loading and capture
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = ToyTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def collect_streams(model):
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]
    _, captures = capture_residual_streams(model, pairs, device)
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    streams_by_pos = {}
    for pos in POSITIONS:
        s = stacked[:, :, pos, :].permute(1, 0, 2).contiguous()
        streams_by_pos[pos] = s.numpy().astype(np.float32)
    return streams_by_pos


# ---------------------------------------------------------------------------
# Coupling, Laplacian, projection (re-implemented here for clarity)
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


def project_streams(stream_at_layer, basis):
    """stream: (N, D), basis: (D, k). Centered then projected."""
    centered = stream_at_layer - stream_at_layer.mean(axis=0, keepdims=True)
    return centered @ basis


def eta_squared(proj, values):
    if proj.ndim == 1:
        proj = proj[:, None]
    n_total = proj.shape[0]
    total_mean = proj.mean(axis=0)
    ss_total = np.sum((proj - total_mean) ** 2)
    if ss_total < 1e-12:
        return 0.0
    unique_vals = np.unique(values)
    if len(unique_vals) <= 8:
        ss_between = 0.0
        for val in unique_vals:
            mask = (values == val)
            if not mask.any():
                continue
            ss_between += int(mask.sum()) * np.sum(
                (proj[mask].mean(axis=0) - total_mean) ** 2
            )
        return float(np.clip(ss_between / ss_total, 0.0, 1.0))
    else:
        y = values.astype(float)
        ss_tot_y = np.sum((y - y.mean()) ** 2)
        if ss_tot_y < 1e-12:
            return 0.0
        X = np.hstack([proj, np.ones((n_total, 1))])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        ss_res = np.sum((y - X @ beta) ** 2)
        return float(np.clip(1.0 - ss_res / ss_tot_y, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_eta_heatmap(eta_table, var_names, probe_pos, target_pos,
                      probe_label, save_path, start=1):
    n_vars, n_layers = eta_table.shape
    row_order = np.argsort(-np.nanmax(eta_table, axis=1))
    sorted_table = eta_table[row_order]
    sorted_names = [var_names[i] for i in row_order]

    fig, ax = plt.subplots(
        figsize=(min(0.5 * n_layers + 4, 18), 0.4 * n_vars + 2)
    )
    im = ax.imshow(
        sorted_table[:, start:], aspect="auto", cmap="magma",
        vmin=0, vmax=1, interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xticks(np.arange(n_layers - start))
    ax.set_xticklabels(
        [str(l) for l in range(start, n_layers)], fontsize=7,
    )
    ax.set_xlabel("sublayer l")
    for i in range(sorted_table.shape[0]):
        for j in range(sorted_table.shape[1] - start):
            v = sorted_table[i, start + j]
            if v > 0.2:
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if v < 0.7 else "black",
                )
    ax.set_title(
        f"eta^2: {probe_label}, probe={POSITION_LABELS[probe_pos]}, "
        f"project on={POSITION_LABELS[target_pos]}",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_within_vs_cross_top(within_eta, cross_eta, var_names,
                              probe_label, save_path):
    """For each (probe, target) pair where probe != target, plot the
    max eta^2 achieved within (probe, probe) vs cross (probe, target)
    for each task variable, as grouped bars."""
    # Selected pairs: each non-trivial combination
    pair_list = [
        (p, t) for p in POSITIONS for t in POSITIONS if p != t
    ]
    n_vars = len(var_names)
    fig, axes = plt.subplots(
        len(pair_list), 1, figsize=(15, 3.2 * len(pair_list)),
        sharex=True,
    )
    if len(pair_list) == 1:
        axes = [axes]
    for ax, (p, t) in zip(axes, pair_list):
        within_max = np.nanmax(within_eta[(p, p)], axis=1)
        cross_max  = np.nanmax(cross_eta[(p, t)],  axis=1)
        x = np.arange(n_vars)
        width = 0.4
        ax.bar(x - width / 2, within_max, width,
               label=f"within ({POSITION_LABELS[p]} on itself)",
               color="C0")
        ax.bar(x + width / 2, cross_max, width,
               label=(f"cross (probe={POSITION_LABELS[p]}, "
                      f"target={POSITION_LABELS[t]})"),
               color="C3")
        ax.set_xticks(x)
        ax.set_xticklabels(var_names, rotation=30, ha="right",
                            fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("max eta^2 (across layers)")
        ax.set_title(
            f"probe pos {p} -> target pos {t} ({probe_label})",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Within-position vs cross-position projection ({probe_label})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_cross_minus_within(cross_eta, within_eta, var_names,
                             save_path):
    """Heatmap: cross_eta - within_eta. Negative = info lost moving from
    probe's own representation to target. Zero or near-zero = shared
    direction. Positive = target encodes the variable better than probe
    does (rare but possible)."""
    pair_list = [
        (p, t) for p in POSITIONS for t in POSITIONS if p != t
    ]
    n_vars = len(var_names)
    n_layers = next(iter(cross_eta.values())).shape[1]

    fig, axes = plt.subplots(
        len(pair_list), 1, figsize=(0.5 * n_layers + 4, 3.5 * len(pair_list)),
    )
    if len(pair_list) == 1:
        axes = [axes]
    for ax, (p, t) in zip(axes, pair_list):
        diff = cross_eta[(p, t)] - within_eta[(p, p)]
        # Sort rows by within max
        within_max = np.nanmax(within_eta[(p, p)], axis=1)
        order = np.argsort(-within_max)
        diff_sorted = diff[order]
        names_sorted = [var_names[i] for i in order]
        im = ax.imshow(
            diff_sorted, aspect="auto", cmap="RdBu_r",
            vmin=-1, vmax=1, interpolation="nearest",
        )
        ax.set_yticks(np.arange(n_vars))
        ax.set_yticklabels(names_sorted, fontsize=8)
        ax.set_xlabel("sublayer l")
        ax.set_title(
            f"cross_eta - within_eta  for probe={p}, target={t}",
            fontsize=10,
        )
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.suptitle(
        "Cross-projection minus within-projection eta^2",
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
    parser.add_argument("--out-dir",    type=str, default="toy_cross_probe")
    parser.add_argument("--start",      type=int, default=1)
    parser.add_argument("--n-eig-subspace", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}")

    streams_by_pos = collect_streams(model)
    L_total = streams_by_pos[5].shape[1]
    D = cfg.d_model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    task_table = build_task_table()
    var_names = list(task_table.keys())
    n_vars = len(var_names)

    # ----- Compute Fiedler eigenvectors per (pos, layer) -----
    print("\nComputing Fiedler eigenvectors per (position, layer) ...")
    eigvecs_by_pos = {pos: [] for pos in POSITIONS}
    for pos in POSITIONS:
        z = per_unit_z(streams_by_pos[pos])
        for l in range(L_total):
            if l == 0:
                eigvecs_by_pos[pos].append(
                    np.full((D, args.n_eig_subspace + 1), np.nan)
                )
                continue
            C = coupling_matrix_at_layer(z[:, l, :])
            _, V = laplacian_eigendecomp(C)
            eigvecs_by_pos[pos].append(V[:, : args.n_eig_subspace + 1])
        eigvecs_by_pos[pos] = np.stack(eigvecs_by_pos[pos], axis=0)
        # shape (L, D, n_eig+1)

    # ----- Compute eta^2 for all (probe_pos, target_pos, var, layer) -----
    print("\nComputing cross-projection eta^2 tables ...")
    eta_1d = {}     # (probe_pos, target_pos) -> (n_vars, L_total)
    eta_kd = {}
    for probe_pos in POSITIONS:
        for target_pos in POSITIONS:
            t1 = np.zeros((n_vars, L_total))
            tk = np.zeros((n_vars, L_total))
            for l in range(L_total):
                if l == 0:
                    t1[:, l] = np.nan
                    tk[:, l] = np.nan
                    continue
                v2     = eigvecs_by_pos[probe_pos][l, :, 1:2]
                v_subs = eigvecs_by_pos[probe_pos][l, :, 1:1 + args.n_eig_subspace]
                target_stream = streams_by_pos[target_pos][:, l, :]
                proj_1d = project_streams(target_stream, v2).flatten()
                proj_kd = project_streams(target_stream, v_subs)
                for i, name in enumerate(var_names):
                    t1[i, l] = eta_squared(proj_1d, task_table[name])
                    tk[i, l] = eta_squared(proj_kd, task_table[name])
            eta_1d[(probe_pos, target_pos)] = t1
            eta_kd[(probe_pos, target_pos)] = tk
            print(f"  probe={probe_pos} -> target={target_pos}: done")

    # ----- Findings: rank cross-projection eta^2 -----
    findings = []
    for (probe_pos, target_pos), table in eta_1d.items():
        for i, name in enumerate(var_names):
            for l in range(args.start, L_total):
                findings.append({
                    "probe_pos":     probe_pos,
                    "target_pos":    target_pos,
                    "layer":         l,
                    "task_variable": name,
                    "eta_squared":   float(table[i, l]),
                    "probe":         "1D",
                })
    for (probe_pos, target_pos), table in eta_kd.items():
        for i, name in enumerate(var_names):
            for l in range(args.start, L_total):
                findings.append({
                    "probe_pos":     probe_pos,
                    "target_pos":    target_pos,
                    "layer":         l,
                    "task_variable": name,
                    "eta_squared":   float(table[i, l]),
                    "probe":         f"{args.n_eig_subspace}D",
                })
    findings.sort(key=lambda f: -f["eta_squared"])

    # ----- Plots -----
    print("\nMaking plots ...")
    for probe_pos in POSITIONS:
        for target_pos in POSITIONS:
            if probe_pos == target_pos:
                continue
            plot_eta_heatmap(
                eta_1d[(probe_pos, target_pos)], var_names,
                probe_pos, target_pos, "1-D probe v_2",
                out_dir / f"cross_proj_heatmap_probe{probe_pos}_"
                          f"target{target_pos}_1D.png",
                start=args.start,
            )
            plot_eta_heatmap(
                eta_kd[(probe_pos, target_pos)], var_names,
                probe_pos, target_pos,
                f"{args.n_eig_subspace}-D probe v_2..v_{args.n_eig_subspace + 1}",
                out_dir / f"cross_proj_heatmap_probe{probe_pos}_"
                          f"target{target_pos}_4D.png",
                start=args.start,
            )

    plot_within_vs_cross_top(
        eta_1d, eta_1d, var_names, "1-D probe",
        out_dir / "within_vs_cross_top_1D.png",
    )
    plot_within_vs_cross_top(
        eta_kd, eta_kd, var_names,
        f"{args.n_eig_subspace}-D probe",
        out_dir / "within_vs_cross_top_4D.png",
    )

    plot_cross_minus_within(
        eta_1d,
        within_eta={
            (p, p): eta_1d[(p, p)] for p in POSITIONS
        },
        var_names=var_names,
        save_path=out_dir / "cross_minus_within_1D.png",
    )
    plot_cross_minus_within(
        eta_kd,
        within_eta={
            (p, p): eta_kd[(p, p)] for p in POSITIONS
        },
        var_names=var_names,
        save_path=out_dir / "cross_minus_within_4D.png",
    )

    # ----- Save & summary -----
    np.savez(
        out_dir / "cross_proj_data.npz",
        var_names=np.array(var_names),
        **{
            f"eta_1d_probe{p}_target{t}": eta_1d[(p, t)]
            for p in POSITIONS for t in POSITIONS
        },
        **{
            f"eta_kd_probe{p}_target{t}": eta_kd[(p, t)]
            for p in POSITIONS for t in POSITIONS
        },
    )

    summary = {
        "checkpoint":    str(args.checkpoint),
        "d_model":       int(D),
        "n_sublayers":   int(L_total),
        "n_eig_subspace": int(args.n_eig_subspace),
        "task_variables": var_names,
        "max_eta_per_pair_per_variable_1D": {
            f"probe{p}_target{t}": {
                name: {
                    "max_eta_squared": float(
                        np.nanmax(eta_1d[(p, t)][i])
                    ),
                    "argmax_layer": int(
                        np.nanargmax(eta_1d[(p, t)][i])
                    ),
                }
                for i, name in enumerate(var_names)
            }
            for p in POSITIONS for t in POSITIONS
        },
        "max_eta_per_pair_per_variable_4D": {
            f"probe{p}_target{t}": {
                name: {
                    "max_eta_squared": float(
                        np.nanmax(eta_kd[(p, t)][i])
                    ),
                    "argmax_layer": int(
                        np.nanargmax(eta_kd[(p, t)][i])
                    ),
                }
                for i, name in enumerate(var_names)
            }
            for p in POSITIONS for t in POSITIONS
        },
        "top_50_findings": [
            {
                "probe_pos":     f["probe_pos"],
                "target_pos":    f["target_pos"],
                "layer":         int(f["layer"]),
                "task_variable": f["task_variable"],
                "eta_squared":   round(f["eta_squared"], 4),
                "probe":         f["probe"],
            }
            for f in findings[:50]
        ],
    }
    with open(out_dir / "cross_proj_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ----- Console output -----
    print("\n--- Cross-projection summary (focus: probe != target) ---")
    print("\nFor each (probe, target) and each variable, max eta^2 "
          "across layers (1-D probe).\n"
          "Within-position rows are baseline; cross rows show how "
          "much carries to the other position's residuals.\n")
    for probe_pos in POSITIONS:
        print(f"\n--- probe = {POSITION_LABELS[probe_pos]} ---")
        within = eta_1d[(probe_pos, probe_pos)]
        for target_pos in POSITIONS:
            if target_pos == probe_pos:
                continue
            cross = eta_1d[(probe_pos, target_pos)]
            print(f"\n  target = {POSITION_LABELS[target_pos]}")
            print(f"    {'variable':<14} "
                  f"{'within max':>12} "
                  f"{'cross max':>12} "
                  f"{'cross/within':>14}")
            for i, name in enumerate(var_names):
                wm = float(np.nanmax(within[i]))
                cm = float(np.nanmax(cross[i]))
                ratio = (cm / wm) if wm > 1e-6 else float("nan")
                marker = ""
                if wm > 0.3:
                    if ratio > 0.7:
                        marker = "  <-- shared direction"
                    elif ratio < 0.3:
                        marker = "  <-- separate"
                print(f"    {name:<14} "
                      f"{wm:>12.3f} {cm:>12.3f} "
                      f"{ratio:>14.3f}{marker}")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
