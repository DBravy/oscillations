"""
Step 2b: cospec-based bootstrap correlation.

Same scatter as step 2 but using signed cospec instead of signed PLV
as the predictor of amplitude-growth correlation. PLV is amplitude-blind;
cospec includes both amplitude and consistency, which is closer to what
the SL coupling term K_ij * (z_j - z_i) actually does.

For each pair (u, v):

  cospec_uv = mean over (input n, sublayer l in pre-window) of
              z_u(n, l) * conj(z_v(n, l))
  where  z_u(n, l) = (a_u(n,l) - a_u(n,0)) + i * (grad_u(n,l) - grad_u(n,0))

  cospec_mag   = |cospec_uv|
  cospec_phase = angle(cospec_uv)
  signed_cospec = Re(cospec_uv) = cospec_mag * cos(cospec_phase)

For comparison, also computes:
  amp_prod_uv = mean |z_u| * |z_v|     (amplitude only, no consistency)
  signed_amp_prod = amp_prod * cos(cospec_phase)

  PLV (already implemented in step 2 for completeness, here too)

Each predictor is regressed against rho_uv (cross-input correlation of
radius growth) via the two Delta_r methods (delta and slope).

This three-predictor comparison apportions the bootstrap mechanism:
  - cospec works AND amp_prod works:  amplitude is the dominant factor
  - cospec works AND PLV works:       consistency matters too
  - cospec works, neither sub-factor: the joint product is what matters
  - nothing works:                    bootstrap mechanism not mediated
                                      by pre-explosion phase coupling

Outputs in --out-dir:
  bootstrap_cospec_scatter_{delta,slope}.png      cospec scatter
  bootstrap_amp_prod_scatter_{delta,slope}.png    amp_prod-only scatter
  predictor_comparison_{delta,slope}.png          slopes/R^2 side by side
  cluster_conditional_cospec_{delta,slope}.png    cluster boxplots
  bootstrap_step2b_summary.json

Usage:
  python sl_step2b_cospec.py \
      --radius-data toy_sl_step1_trained/radius_data.npz \
      --out-dir toy_sl_step2b_trained
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


POSITIONS = [5, 6, 7]
POSITION_LABELS = {
    5: 'pos 5 ("=")',
    6: "pos 6 (c2)",
    7: "pos 7 (c1)",
}
POSITION_COLORS = {5: "C0", 6: "C1", 7: "C2"}

QUAD_BAND = np.pi / 6


def cluster_of_phase(phi, band=QUAD_BAND):
    if abs(phi) < band:
        return "near_0"
    if abs(phi - np.pi / 2) < band:
        return "near_pi_over_2"
    if abs(phi + np.pi / 2) < band:
        return "near_minus_pi_over_2"
    if (abs(phi - np.pi) < band) or (abs(phi + np.pi) < band):
        return "near_pi"
    return "between"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_step1(npz_path):
    z = np.load(npz_path)
    streams = {pos: z[f"streams_pos{pos}"]   for pos in POSITIONS}
    radii   = {pos: z[f"r_per_unit_pos{pos}"] for pos in POSITIONS}
    return streams, radii


def all_pair_indices(D):
    u, v = np.triu_indices(D, k=1)
    return u, v


# ---------------------------------------------------------------------------
# z_u construction
# ---------------------------------------------------------------------------

def per_unit_z(streams):
    """
    streams: (N, L, D)
    Returns z: (N, L, D) complex, where
      z[n, l, u] = (a_u(n,l) - a_u(n,0)) + i * (grad_u(n,l) - grad_u(n,0))
    """
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    return x + 1j * y


# ---------------------------------------------------------------------------
# Per-pair couplings in the pre-explosion window
# ---------------------------------------------------------------------------

def compute_couplings(z, pre_start, pre_end):
    """
    z: (N, L, D) complex
    Returns dict with arrays of length n_pairs.
    """
    N, L, D = z.shape
    sub = z[:, pre_start:pre_end + 1, :]                  # (N, W, D)
    u_idx, v_idx = all_pair_indices(D)

    z_u = sub[..., u_idx]                                 # (N, W, n_pairs)
    z_v = sub[..., v_idx]
    cs  = z_u * np.conj(z_v)                              # complex

    # cospec: amplitude + consistency
    cospec        = cs.mean(axis=(0, 1))                  # complex (n_pairs,)
    cospec_mag    = np.abs(cospec)
    cospec_phase  = np.angle(cospec)
    signed_cospec = np.real(cospec)

    # amp_prod (amplitude only; phase ignored except for sign)
    amp_per_sample = np.abs(z_u) * np.abs(z_v)            # (N, W, n_pairs)
    amp_prod      = amp_per_sample.mean(axis=(0, 1))      # (n_pairs,)
    signed_amp_prod = amp_prod * np.cos(cospec_phase)

    # PLV (consistency only)
    eps = 1e-12
    cs_unit = cs / (np.abs(cs) + eps)
    plv = np.abs(cs_unit.mean(axis=(0, 1)))               # (n_pairs,)
    signed_plv = plv * np.cos(cospec_phase)

    # Per-unit pre-window amplitude (for the optional floor mask)
    per_unit_amp = np.abs(sub).mean(axis=(0, 1))          # (D,)

    return {
        "u_idx":            u_idx,
        "v_idx":            v_idx,
        "cospec_mag":       cospec_mag,
        "cospec_phase":     cospec_phase,
        "signed_cospec":    signed_cospec,
        "amp_prod":         amp_prod,
        "signed_amp_prod":  signed_amp_prod,
        "plv":              plv,
        "signed_plv":       signed_plv,
        "per_unit_amp":     per_unit_amp,
    }


# ---------------------------------------------------------------------------
# Post-explosion radius growth
# ---------------------------------------------------------------------------

def compute_delta_r_window(r, post_start, post_end):
    return r[:, post_end, :] - r[:, post_start, :]


def compute_slope_r_window(r, post_start, post_end):
    N, L, D = r.shape
    ls = np.arange(post_start, post_end + 1).astype(np.float32)
    sub = r[:, post_start:post_end + 1, :]
    ls_centered = ls - ls.mean()
    sub_centered = sub - sub.mean(axis=1, keepdims=True)
    num = (sub_centered * ls_centered[None, :, None]).sum(axis=1)
    den = (ls_centered ** 2).sum()
    return num / den


def correlate_over_inputs(delta_r, u_idx, v_idx):
    centered = delta_r - delta_r.mean(axis=0, keepdims=True)
    std = centered.std(axis=0) + 1e-12
    z = centered / std
    return (z[:, u_idx] * z[:, v_idx]).mean(axis=0)


def amplitude_floor_mask(per_unit_amp, u_idx, v_idx, percentile=25):
    threshold = np.percentile(per_unit_amp, percentile)
    keep_unit = per_unit_amp >= threshold
    return keep_unit[u_idx] & keep_unit[v_idx], threshold


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_predictor_scatter(rho_by_pos, predictor_by_pos, phase_by_pos,
                           keep_by_pos, predictor_name,
                           method_label, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
    cluster_colors = {
        "near_0":              "tab:green",
        "near_pi_over_2":      "tab:orange",
        "near_minus_pi_over_2": "tab:purple",
        "near_pi":             "tab:red",
        "between":             "lightgray",
    }
    for ax, pos in zip(axes, POSITIONS):
        rho   = rho_by_pos[pos]
        pred  = predictor_by_pos[pos]
        phase = phase_by_pos[pos]
        keep  = keep_by_pos[pos]

        clusters = np.array(
            [cluster_of_phase(p) for p in phase], dtype=object
        )

        ax.scatter(
            pred[~keep], rho[~keep], s=8, alpha=0.15,
            color="lightgray", edgecolors="none",
            label=f"low-amp (n={(~keep).sum()})",
        )
        for cname, ccolor in cluster_colors.items():
            mask = keep & (clusters == cname)
            if mask.any():
                ax.scatter(
                    pred[mask], rho[mask], s=22, alpha=0.6,
                    color=ccolor, edgecolors="none",
                    label=f"{cname} (n={int(mask.sum())})",
                )

        if keep.sum() >= 2:
            res = stats.linregress(pred[keep], rho[keep])
            xs = np.linspace(pred[keep].min(), pred[keep].max(), 50)
            ax.plot(
                xs, res.slope * xs + res.intercept,
                "k-", linewidth=1.4,
                label=(f"fit: slope={res.slope:.4g}, "
                       f"R^2={res.rvalue**2:.3f}, "
                       f"p={res.pvalue:.1e}"),
            )
        ax.axhline(0, color="gray", linewidth=0.4, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.4, linestyle=":")
        ax.set_xlabel(f"signed {predictor_name}")
        if ax is axes[0]:
            ax.set_ylabel("rho_uv (high-amp pairs)")
        ax.set_title(POSITION_LABELS[pos], fontsize=11)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"rho vs signed {predictor_name}  ({method_label})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_predictor_comparison_bars(fits_by_pred, method_label, save_path):
    """Side-by-side bar chart of slope, R^2, -log10(p) for the three
    predictors at each position (high-amp pairs only)."""
    metrics = [
        ("slope",          "slope"),
        ("r2",             "R^2"),
        ("neg_log10_p",    "-log10(p)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    width = 0.25
    x_base = np.arange(len(POSITIONS))
    predictors = list(fits_by_pred.keys())
    pred_colors = {
        "cospec":    "tab:blue",
        "amp_prod":  "tab:orange",
        "plv":       "tab:green",
    }
    for ax, (key, label) in zip(axes, metrics):
        for i, pred in enumerate(predictors):
            vals = []
            for pos in POSITIONS:
                f = fits_by_pred[pred][pos]
                if key == "neg_log10_p":
                    p = f["p_value"] if f["p_value"] else 1.0
                    vals.append(-np.log10(max(p, 1e-300)))
                else:
                    vals.append(f[key] if f[key] is not None else 0)
            offset = (i - 1) * width
            ax.bar(
                x_base + offset, vals, width,
                color=pred_colors[pred], label=pred,
            )
        ax.set_xticks(x_base)
        ax.set_xticklabels([POSITION_LABELS[p] for p in POSITIONS])
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.4)
    fig.suptitle(
        f"Three-predictor comparison  ({method_label}, high-amp pairs)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_cluster_conditional(rho_by_pos, phase_by_pos, keep_by_pos,
                              method_label, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    cluster_order = [
        "near_0", "near_pi_over_2", "near_minus_pi_over_2", "near_pi"
    ]
    cluster_short = {
        "near_0": "0",
        "near_pi_over_2": "+pi/2",
        "near_minus_pi_over_2": "-pi/2",
        "near_pi": "pi",
    }
    for ax, pos in zip(axes, POSITIONS):
        rho   = rho_by_pos[pos]
        phase = phase_by_pos[pos]
        keep  = keep_by_pos[pos]
        clusters = np.array(
            [cluster_of_phase(p) for p in phase], dtype=object
        )
        data = []
        labels = []
        for c in cluster_order:
            mask = keep & (clusters == c)
            if mask.any():
                data.append(rho[mask])
                labels.append(
                    f"{cluster_short[c]}\nn={int(mask.sum())}\n"
                    f"med={np.median(rho[mask]):+.3f}"
                )
            else:
                data.append(np.array([0.0]))
                labels.append(f"{cluster_short[c]}\nn=0")
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False,
                        patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(POSITION_COLORS[pos])
            patch.set_alpha(0.4)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(POSITION_LABELS[pos], fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("rho_uv (high-amp pairs)")
        ax.set_xlabel("phase cluster (cospec phase)")
    fig.suptitle(
        f"Cluster-conditional rho distributions  ({method_label})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fit helper
# ---------------------------------------------------------------------------

def fit(x, y, mask):
    if mask.sum() < 2:
        return {
            "n":         int(mask.sum()),
            "slope":     None,
            "intercept": None,
            "r2":        None,
            "p_value":   None,
        }
    res = stats.linregress(x[mask], y[mask])
    return {
        "n":         int(mask.sum()),
        "slope":     float(res.slope),
        "intercept": float(res.intercept),
        "r2":        float(res.rvalue ** 2),
        "p_value":   float(res.pvalue),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius-data", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="toy_sl_step2b")
    parser.add_argument("--pre-start",   type=int, default=10)
    parser.add_argument("--pre-end",     type=int, default=20)
    parser.add_argument("--post-start",  type=int, default=20)
    parser.add_argument("--post-end",    type=int, default=24)
    parser.add_argument("--amp-percentile", type=float, default=25.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.radius_data}")
    streams_by_pos, radii_by_pos = load_step1(args.radius_data)
    L = streams_by_pos[5].shape[1]
    print(f"  n_sublayers={L}")
    print(f"  pre = [{args.pre_start}, {args.pre_end}], "
          f"post = [{args.post_start}, {args.post_end}]")

    rho_delta_by_pos    = {}
    rho_slope_by_pos    = {}
    couplings_by_pos    = {}
    keep_by_pos         = {}

    for pos in POSITIONS:
        print(f"\n=== {POSITION_LABELS[pos]} ===")
        streams = streams_by_pos[pos]
        r       = radii_by_pos[pos]

        z = per_unit_z(streams)
        couplings = compute_couplings(z, args.pre_start, args.pre_end)
        u_idx = couplings["u_idx"]
        v_idx = couplings["v_idx"]

        keep_pair, threshold = amplitude_floor_mask(
            couplings["per_unit_amp"], u_idx, v_idx,
            percentile=args.amp_percentile,
        )
        couplings["amp_floor"] = float(threshold)

        delta_r = compute_delta_r_window(
            r, args.post_start, args.post_end
        )
        slope_r = compute_slope_r_window(
            r, args.post_start, args.post_end
        )
        rho_delta = correlate_over_inputs(delta_r, u_idx, v_idx)
        rho_slope = correlate_over_inputs(slope_r, u_idx, v_idx)

        rho_delta_by_pos[pos] = rho_delta
        rho_slope_by_pos[pos] = rho_slope
        couplings_by_pos[pos] = couplings
        keep_by_pos[pos]      = keep_pair

        print(f"  n_pairs total={len(rho_delta)}, "
              f"high-amp={keep_pair.sum()}")
        print(f"  cospec_mag (high-amp): "
              f"median={np.median(couplings['cospec_mag'][keep_pair]):.4g}, "
              f"max={couplings['cospec_mag'][keep_pair].max():.4g}")
        print(f"  amp_prod (high-amp): "
              f"median={np.median(couplings['amp_prod'][keep_pair]):.4g}")
        print(f"  PLV (high-amp): "
              f"mean={couplings['plv'][keep_pair].mean():.3f}")

    # ------------- Fits -------------
    fits_by_pred_method = {
        "delta": {
            "cospec":    {},
            "amp_prod":  {},
            "plv":       {},
        },
        "slope": {
            "cospec":    {},
            "amp_prod":  {},
            "plv":       {},
        },
    }
    for pos in POSITIONS:
        c = couplings_by_pos[pos]
        keep = keep_by_pos[pos]
        for method_name, rho_arr in [
            ("delta", rho_delta_by_pos[pos]),
            ("slope", rho_slope_by_pos[pos]),
        ]:
            fits_by_pred_method[method_name]["cospec"][pos] = \
                fit(c["signed_cospec"],   rho_arr, keep)
            fits_by_pred_method[method_name]["amp_prod"][pos] = \
                fit(c["signed_amp_prod"], rho_arr, keep)
            fits_by_pred_method[method_name]["plv"][pos] = \
                fit(c["signed_plv"],      rho_arr, keep)

    # ------------- Plots -------------
    print("\nMaking plots ...")

    for method_name, rho_by_pos in [
        ("delta", rho_delta_by_pos),
        ("slope", rho_slope_by_pos),
    ]:
        method_label = (
            "delta = r(post_end) - r(post_start)"
            if method_name == "delta"
            else "slope of r(l) over post window"
        )

        # cospec scatter
        plot_predictor_scatter(
            rho_by_pos,
            {pos: couplings_by_pos[pos]["signed_cospec"]
             for pos in POSITIONS},
            {pos: couplings_by_pos[pos]["cospec_phase"]
             for pos in POSITIONS},
            keep_by_pos,
            predictor_name="cospec = Re(<z_u conj(z_v)>)",
            method_label=method_label,
            save_path=out_dir / f"bootstrap_cospec_scatter_{method_name}.png",
        )

        # amp_prod scatter
        plot_predictor_scatter(
            rho_by_pos,
            {pos: couplings_by_pos[pos]["signed_amp_prod"]
             for pos in POSITIONS},
            {pos: couplings_by_pos[pos]["cospec_phase"]
             for pos in POSITIONS},
            keep_by_pos,
            predictor_name="amp_prod * cos(phi)",
            method_label=method_label,
            save_path=out_dir / f"bootstrap_amp_prod_scatter_{method_name}.png",
        )

        # Predictor comparison bars
        plot_predictor_comparison_bars(
            fits_by_pred_method[method_name], method_label,
            save_path=out_dir / f"predictor_comparison_{method_name}.png",
        )

        # Cluster-conditional (using cospec phase)
        plot_cluster_conditional(
            rho_by_pos,
            {pos: couplings_by_pos[pos]["cospec_phase"]
             for pos in POSITIONS},
            keep_by_pos,
            method_label=f"cospec phase clusters, {method_label}",
            save_path=out_dir / f"cluster_conditional_cospec_{method_name}.png",
        )

    # ------------- Save -------------
    save_dict = {}
    for pos in POSITIONS:
        c = couplings_by_pos[pos]
        save_dict[f"cospec_mag_pos{pos}"]      = c["cospec_mag"]
        save_dict[f"cospec_phase_pos{pos}"]    = c["cospec_phase"]
        save_dict[f"signed_cospec_pos{pos}"]   = c["signed_cospec"]
        save_dict[f"amp_prod_pos{pos}"]        = c["amp_prod"]
        save_dict[f"signed_amp_prod_pos{pos}"] = c["signed_amp_prod"]
        save_dict[f"plv_pos{pos}"]             = c["plv"]
        save_dict[f"keep_pos{pos}"]            = keep_by_pos[pos]
        save_dict[f"rho_delta_pos{pos}"]       = rho_delta_by_pos[pos]
        save_dict[f"rho_slope_pos{pos}"]       = rho_slope_by_pos[pos]
    np.savez(out_dir / "bootstrap_step2b_data.npz", **save_dict)

    # Build summary
    summary = {
        "radius_data":           args.radius_data,
        "pre_explosion_window":  [args.pre_start, args.pre_end],
        "post_explosion_window": [args.post_start, args.post_end],
        "amp_percentile":        args.amp_percentile,
        "fits": {
            method_name: {
                pred: {
                    str(pos): fit_result
                    for pos, fit_result in pos_fits.items()
                }
                for pred, pos_fits in pred_fits.items()
            }
            for method_name, pred_fits in fits_by_pred_method.items()
        },
        "per_position_pair_counts": {
            str(pos): {
                "total":    int(len(rho_delta_by_pos[pos])),
                "high_amp": int(keep_by_pos[pos].sum()),
            }
            for pos in POSITIONS
        },
    }
    with open(out_dir / "bootstrap_step2b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------- Console summary -------------
    print("\n--- Slope summary table (high-amp pairs) ---")
    for method_name in ("delta", "slope"):
        print(f"\nMethod: {method_name}")
        print(f"{'predictor':<10} {'pos':<14} "
              f"{'slope':>14} {'R^2':>10} {'p':>14} {'n':>8}")
        for pred in ("cospec", "amp_prod", "plv"):
            for pos in POSITIONS:
                f_ = fits_by_pred_method[method_name][pred][pos]
                slope_str = (f"{f_['slope']:.4g}"
                             if f_["slope"] is not None else "n/a")
                r2_str    = (f"{f_['r2']:.3f}"
                             if f_["r2"] is not None else "n/a")
                p_str     = (f"{f_['p_value']:.2e}"
                             if f_["p_value"] is not None else "n/a")
                print(f"{pred:<10} {POSITION_LABELS[pos]:<14} "
                      f"{slope_str:>14} {r2_str:>10} "
                      f"{p_str:>14} {f_['n']:>8}")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
