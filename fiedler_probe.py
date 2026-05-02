"""
Fiedler-direction probe: what does the v_2 channel at each (position,
sublayer) encode about the addition task?

For a checkpoint, runs forward on all 256 (a, b) pairs, then for each
(position in {5, 6, 7}, sublayer >= 1):

  1. Computes the unnormalized-Laplacian eigendecomposition of the
     layer-localized cospec coupling matrix.
  2. Projects each input's residual stream at that (position, sublayer)
     onto v_2 (1-D probe) and onto the v_2..v_5 subspace (4-D probe).
     The streams are centered across inputs before projection.
  3. For each candidate task variable v, computes eta^2: the fraction of
     variance in the projection explained by grouping by v.

Task variables tested:
  Input digits:  a0, a1, b0, b1
  Output digits: c0, c1, c2
  Carries:       carry_0_to_1, carry_1_to_2
  Pre-mod sums:  a0+b0, a1+b1
  Totals:        a, b

Binary/categorical variables (<=8 unique values) get categorical eta^2.
Continuous variables (>8 unique values, e.g. a, b) get linear-regression
R^2 (1-D) or multivariate R^2 (4-D).

Outputs in --out-dir:
  eta2_heatmap_1D_pos{N}.png      heatmap of eta^2 (task_var x layer),
                                   1-D probe, one figure per position
  eta2_heatmap_4D_pos{N}.png      same for 4-D probe
  eta2_curves_pos{N}.png          line plot of eta^2 across layers,
                                   one line per task variable
  evidence_top_findings.png       for the top-N (pos, layer, task) cells
                                   by eta^2, plot the projection
                                   distribution split by class
  fiedler_probe_summary.json      ranked findings

Usage:
  python fiedler_probe.py \
      --checkpoint toy_transformer_run_swiglu/model_trained.pt \
      --out-dir toy_probe_swiglu
  python fiedler_probe.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_probe_gelu
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from toy_transformer_addition import (
    ToyTransformer,
    ModelConfig,
    AdditionDataset,
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
# Task variables
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
        "a":             a,
        "b":             b,
        "a0":            a0,
        "a1":            a1,
        "b0":            b0,
        "b1":            b1,
        "c0":            c0,
        "c1":            c1,
        "c2":            c2,
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
    return table, pairs


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
# Coupling, Laplacian, Fiedler subspace
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


# ---------------------------------------------------------------------------
# Projections and eta^2
# ---------------------------------------------------------------------------

def project_streams(stream_at_layer, basis):
    """
    stream_at_layer: (N, D) raw residuals at one (pos, layer)
    basis:           (D, k) eigenvectors v_2..v_{k+1}
    Returns:         (N, k) projection of the across-input-centered
                     residuals onto the basis.
    """
    centered = stream_at_layer - stream_at_layer.mean(axis=0, keepdims=True)
    return centered @ basis


def eta_squared(proj, values):
    """
    proj:   (N,) or (N, k)  numerical projection
    values: (N,)            task variable (categorical or continuous)
    Returns scalar in [0, 1] = fraction of variance in proj explained
    by the task variable.

    Categorical (<=8 unique values): standard one-way ANOVA eta^2.
    Continuous: linear regression R^2 of values on proj.
    """
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
            n_g = int(mask.sum())
            if n_g == 0:
                continue
            group_mean = proj[mask].mean(axis=0)
            ss_between += n_g * np.sum((group_mean - total_mean) ** 2)
        return float(np.clip(ss_between / ss_total, 0.0, 1.0))
    else:
        y = values.astype(float)
        ss_tot_y = np.sum((y - y.mean()) ** 2)
        if ss_tot_y < 1e-12:
            return 0.0
        X = np.hstack([proj, np.ones((n_total, 1))])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss_res = np.sum((y - pred) ** 2)
        return float(np.clip(1.0 - ss_res / ss_tot_y, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_eta_heatmap(eta_table, var_names, position, probe_label,
                      save_path, start=1):
    """eta_table: (n_vars, n_layers) array of eta^2 values."""
    n_vars, n_layers = eta_table.shape
    # Sort rows by max eta^2 across layers, descending
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
    layer_ticks = np.arange(n_layers - start)
    ax.set_xticks(layer_ticks)
    ax.set_xticklabels(
        [str(l) for l in range(start, n_layers)], fontsize=7,
    )
    ax.set_xlabel("sublayer l")
    # Annotate cells with eta^2 > 0.2
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
        f"eta^2: task variables vs Fiedler projection ({probe_label}) "
        f"at {POSITION_LABELS[position]}",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_eta_curves(eta_table_1d, eta_table_4d, var_names, position,
                     save_path, start=1):
    """One panel: lines for each task variable. Solid = 1D probe,
    dashed = 4D probe."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    n_vars, n_layers = eta_table_1d.shape
    cmap = plt.get_cmap("tab20")
    for ax, table, label in [
        (axes[0], eta_table_1d, "1-D probe (v_2 only)"),
        (axes[1], eta_table_4d, "4-D probe (v_2..v_5)"),
    ]:
        for i, name in enumerate(var_names):
            ax.plot(
                np.arange(start, n_layers),
                table[i, start:],
                "-o", markersize=3, linewidth=1.2,
                color=cmap(i % 20),
                label=name,
            )
        ax.set_xlabel("sublayer l")
        ax.set_ylim(0, 1.05)
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
    axes[0].set_ylabel("eta^2 (variance fraction explained)")
    fig.suptitle(
        f"Fiedler-projection probe at {POSITION_LABELS[position]}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_top_findings_evidence(top_findings, streams_by_pos,
                                eigvecs_cache, task_table,
                                save_path, n_show=6):
    """For the top-N findings by eta^2, plot the projection histogram
    split by task variable class."""
    rows_to_plot = top_findings[:n_show]
    if not rows_to_plot:
        return
    fig, axes = plt.subplots(
        n_show, 1, figsize=(11, 3 * n_show), sharex=False,
    )
    if n_show == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    for ax, finding in zip(axes, rows_to_plot):
        pos    = finding["position"]
        layer  = finding["layer"]
        var    = finding["task_variable"]
        eta    = finding["eta_squared"]
        probe  = finding["probe"]
        n_eig  = finding["n_eig"]
        # Recompute the 1D projection for plotting
        stream = streams_by_pos[pos][:, layer, :]
        V = eigvecs_cache[(pos, layer)]
        v2 = V[:, 1:1 + 1]
        proj = (stream - stream.mean(axis=0)) @ v2
        proj = proj.flatten()
        values = task_table[var]
        unique_vals = np.unique(values)
        for i, val in enumerate(unique_vals):
            mask = (values == val)
            ax.hist(
                proj[mask], bins=25, alpha=0.55,
                color=cmap(i / max(1, len(unique_vals) - 1)),
                label=f"{var}={val} (n={int(mask.sum())})",
            )
        ax.set_xlabel("v_2-projection of centered residual")
        ax.set_ylabel("count")
        ax.set_title(
            f"{POSITION_LABELS[pos]}, sublayer {layer}, "
            f"variable '{var}', "
            f"eta^2 ({probe}) = {eta:.3f}",
            fontsize=11,
        )
        ax.legend(fontsize=8, ncol=2 if len(unique_vals) > 4 else 1)
    fig.suptitle(
        "Top findings: projection distribution split by task variable",
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
    parser.add_argument("--out-dir", type=str, default="toy_probe")
    parser.add_argument("--start", type=int, default=1,
                        help="First sublayer included in plots")
    parser.add_argument("--n-eig-subspace", type=int, default=4,
                        help="Subspace size for the multi-D probe")
    parser.add_argument("--n-top-evidence", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}")

    print("\nCapturing streams ...")
    streams_by_pos = collect_streams(model)
    L_total = streams_by_pos[5].shape[1]
    D = cfg.d_model

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nBuilding task variable table ...")
    task_table, pairs = build_task_table()
    var_names = list(task_table.keys())
    print(f"  task variables: {var_names}")

    print("\nComputing Fiedler eigendecompositions and probes ...")
    eta_1d_by_pos = {pos: np.zeros((len(var_names), L_total))
                      for pos in POSITIONS}
    eta_4d_by_pos = {pos: np.zeros((len(var_names), L_total))
                      for pos in POSITIONS}
    eigvecs_cache = {}      # (pos, layer) -> (D, n_eig+2)

    for pos in POSITIONS:
        z = per_unit_z(streams_by_pos[pos])
        for l in range(L_total):
            if l == 0:
                continue
            C = coupling_matrix_at_layer(z[:, l, :])
            w, V = laplacian_eigendecomp(C)
            eigvecs_cache[(pos, l)] = V[:, : args.n_eig_subspace + 1]

            v2 = V[:, 1:2]                              # (D, 1)
            v_subspace = V[:, 1:1 + args.n_eig_subspace]  # (D, k)

            stream = streams_by_pos[pos][:, l, :]       # (N, D)
            proj_1d = project_streams(stream, v2)       # (N, 1)
            proj_kd = project_streams(stream, v_subspace)  # (N, k)

            for i, name in enumerate(var_names):
                eta_1d_by_pos[pos][i, l] = eta_squared(
                    proj_1d.flatten(), task_table[name],
                )
                eta_4d_by_pos[pos][i, l] = eta_squared(
                    proj_kd, task_table[name],
                )
        # Mark layer 0 as nan
        eta_1d_by_pos[pos][:, 0] = np.nan
        eta_4d_by_pos[pos][:, 0] = np.nan

    # ---------- Findings ranking ----------
    findings = []
    for pos in POSITIONS:
        for i, name in enumerate(var_names):
            for l in range(args.start, L_total):
                findings.append({
                    "position":      pos,
                    "layer":         l,
                    "task_variable": name,
                    "eta_squared":   float(eta_1d_by_pos[pos][i, l]),
                    "probe":         "1D",
                    "n_eig":         1,
                })
                findings.append({
                    "position":      pos,
                    "layer":         l,
                    "task_variable": name,
                    "eta_squared":   float(eta_4d_by_pos[pos][i, l]),
                    "probe":         f"{args.n_eig_subspace}D",
                    "n_eig":         args.n_eig_subspace,
                })
    findings.sort(key=lambda d: -d["eta_squared"])
    top_findings = findings[: args.n_top_evidence * 4]

    # ---------- Plots ----------
    print("\nMaking plots ...")
    for pos in POSITIONS:
        plot_eta_heatmap(
            eta_1d_by_pos[pos], var_names, pos,
            "1-D probe v_2",
            out_dir / f"eta2_heatmap_1D_pos{pos}.png",
            start=args.start,
        )
        plot_eta_heatmap(
            eta_4d_by_pos[pos], var_names, pos,
            f"{args.n_eig_subspace}-D probe v_2..v_{args.n_eig_subspace + 1}",
            out_dir / f"eta2_heatmap_4D_pos{pos}.png",
            start=args.start,
        )
        plot_eta_curves(
            eta_1d_by_pos[pos], eta_4d_by_pos[pos], var_names, pos,
            out_dir / f"eta2_curves_pos{pos}.png",
            start=args.start,
        )

    # Filter to 1D-only findings for evidence plot (cleanest histograms)
    one_d_findings = [f for f in findings if f["probe"] == "1D"]
    plot_top_findings_evidence(
        one_d_findings, streams_by_pos, eigvecs_cache, task_table,
        out_dir / "evidence_top_findings.png",
        n_show=args.n_top_evidence,
    )

    # ---------- Summary ----------
    summary = {
        "checkpoint":    str(args.checkpoint),
        "d_model":       int(D),
        "n_sublayers":   int(L_total),
        "task_variables": var_names,
        "top_50_findings": [
            {
                "position":      f["position"],
                "layer":         int(f["layer"]),
                "task_variable": f["task_variable"],
                "eta_squared":   round(f["eta_squared"], 4),
                "probe":         f["probe"],
            }
            for f in findings[:50]
        ],
        "max_eta_per_variable_per_position_1D": {
            f"pos{pos}": {
                name: {
                    "max_eta_squared": float(
                        np.nanmax(eta_1d_by_pos[pos][i])
                    ),
                    "argmax_layer": int(
                        np.nanargmax(eta_1d_by_pos[pos][i])
                    ),
                }
                for i, name in enumerate(var_names)
            }
            for pos in POSITIONS
        },
    }
    with open(out_dir / "fiedler_probe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        out_dir / "fiedler_probe_data.npz",
        **{f"eta_1d_pos{pos}": eta_1d_by_pos[pos] for pos in POSITIONS},
        **{f"eta_4d_pos{pos}": eta_4d_by_pos[pos] for pos in POSITIONS},
        var_names=np.array(var_names),
    )

    # ---------- Console summary ----------
    print("\n--- Top 15 findings (sorted by eta^2) ---")
    print(f"{'rank':<5} {'pos':<6} {'layer':<6} "
          f"{'task var':<14} {'probe':<6} {'eta^2':>8}")
    for rank, f in enumerate(findings[:15], 1):
        print(f"{rank:<5} p{f['position']:<5} "
              f"L{f['layer']:<5} {f['task_variable']:<14} "
              f"{f['probe']:<6} {f['eta_squared']:>8.3f}")

    print("\n--- Max eta^2 per variable per position (1-D probe) ---")
    for pos in POSITIONS:
        print(f"\n{POSITION_LABELS[pos]}:")
        for name in var_names:
            i = var_names.index(name)
            mx = np.nanmax(eta_1d_by_pos[pos][i])
            ml = int(np.nanargmax(eta_1d_by_pos[pos][i]))
            mark = "  <--" if mx >= 0.5 else ("    *" if mx >= 0.2 else "")
            print(f"  {name:<14}  max={mx:.3f} at L{ml:<3} {mark}")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
