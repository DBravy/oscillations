"""
Step 2 of the Stuart-Landau bootstrap analysis: test whether high-PLV
pairs show coordinated amplitude growth across the explosion window.

Reads the radius_data.npz produced by step 1, recomputes per-unit phases
in the per-unit phase plane, then for each position:

  1. Computes pre-explosion PLV by averaging exp(i * (theta_u - theta_v))
     across input pairs AND across sublayers in [pre_start, pre_end].
     Magnitude = PLV; angle = mean phase difference.

  2. Computes post-explosion radius growth Delta_r per unit per input
     two ways (side by side):
       - delta:  r_u(n, post_end) - r_u(n, post_start)
       - slope:  linear-fit slope of r_u(n, l) for l in [post_start, post_end]

  3. Computes amplitude-growth correlation rho_uv = corr_n(Delta_r_u, Delta_r_v)
     across the 256 input pairs.

  4. Tests Stuart-Landau bootstrap prediction:
       rho_uv should track signed_coupling = PLV * cos(<theta_u - theta_v>)
       with positive slope across all pairs.

       In particular:
         in-phase pairs (cluster near 0)         -> rho > 0
         anti-phase pairs (cluster near pi)      -> rho < 0
         quadrature pairs (cluster near +/- pi/2) -> rho ~ 0

Outputs in --out-dir:
  bootstrap_scatter_{delta,slope}.png    main scatter, three positions
  cluster_conditional_{delta,slope}.png  rho per phase cluster at each position
  bootstrap_step2_summary.json           slopes, intercepts, R^2 per case

Usage:
  python sl_step2_bootstrap.py \
      --radius-data toy_sl_step1_trained/radius_data.npz \
      --out-dir toy_sl_step2_trained
  python sl_step2_bootstrap.py \
      --radius-data toy_sl_step1_random/radius_data.npz \
      --out-dir toy_sl_step2_random
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
# Load & basic computations
# ---------------------------------------------------------------------------

def load_step1(npz_path):
    z = np.load(npz_path)
    streams = {pos: z[f"streams_pos{pos}"]   for pos in POSITIONS}
    radii   = {pos: z[f"r_per_unit_pos{pos}"] for pos in POSITIONS}
    return streams, radii


def per_unit_phases(streams):
    """
    streams: (N, L, D)
    Returns theta: (N, L, D) where theta[n, l, u] is the angle in unit u's
    own phase plane: atan2(grad_u(l) - grad_u(0), a_u(l) - a_u(0)).
    Undefined at l=0 (both args zero); we set it to 0 there.
    """
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    theta = np.arctan2(y, x)
    theta[:, 0, :] = 0.0     # convention; never used since pre window starts > 0
    return theta


def all_pair_indices(D):
    u, v = np.triu_indices(D, k=1)
    return u, v


def compute_PLV_and_mean_phase(theta, pre_start, pre_end):
    """
    theta: (N, L, D)
    Returns:
      PLV:   (n_pairs,)
      phase: (n_pairs,)  the mean phase difference, in [-pi, pi]
    Computed by averaging exp(i*(theta_u - theta_v)) over (input, sublayer)
    in the pre-explosion window.
    """
    N, L, D = theta.shape
    sub = theta[:, pre_start:pre_end + 1, :]              # (N, W, D)
    u_idx, v_idx = all_pair_indices(D)
    # exp(i * (theta_u - theta_v)) = e^{i theta_u} * conj(e^{i theta_v})
    e_iu = np.exp(1j * sub[..., u_idx])                   # (N, W, n_pairs)
    e_iv = np.exp(1j * sub[..., v_idx])
    diff = e_iu * np.conj(e_iv)                           # (N, W, n_pairs)
    avg  = diff.mean(axis=(0, 1))                         # (n_pairs,)
    return np.abs(avg), np.angle(avg), u_idx, v_idx


def compute_delta_r_window(r, post_start, post_end):
    """r: (N, L, D). Returns delta_r: (N, D) = r[:, post_end, :] - r[:, post_start, :]."""
    return r[:, post_end, :] - r[:, post_start, :]


def compute_slope_r_window(r, post_start, post_end):
    """r: (N, L, D). Returns slope_r: (N, D) per-input per-unit slope of
    a linear fit to r[n, l, u] vs l for l in [post_start, post_end]."""
    N, L, D = r.shape
    ls = np.arange(post_start, post_end + 1).astype(np.float32)
    sub = r[:, post_start:post_end + 1, :]              # (N, W, D)
    # Slope = cov(l, sub) / var(l), per (n, u)
    ls_centered = ls - ls.mean()
    sub_centered = sub - sub.mean(axis=1, keepdims=True)
    num = (sub_centered * ls_centered[None, :, None]).sum(axis=1)
    den = (ls_centered ** 2).sum()
    return num / den                                    # (N, D)


def correlate_over_inputs(delta_r, u_idx, v_idx):
    """
    delta_r: (N, D)
    Returns rho: (n_pairs,) Pearson correlation across N inputs of
    delta_r[:, u] and delta_r[:, v].
    """
    centered = delta_r - delta_r.mean(axis=0, keepdims=True)
    std = centered.std(axis=0) + 1e-12
    z = centered / std
    return (z[:, u_idx] * z[:, v_idx]).mean(axis=0)


def amplitude_floor_mask(r, pre_start, pre_end, percentile=25):
    """
    r: (N, L, D). Returns boolean (n_pairs,) mask where both u and v
    have pre-explosion mean radius above the per-unit percentile.
    """
    pre_mean = r[:, pre_start:pre_end + 1, :].mean(axis=(0, 1))   # (D,)
    threshold = np.percentile(pre_mean, percentile)
    keep_unit = pre_mean >= threshold
    u_idx, v_idx = all_pair_indices(r.shape[2])
    keep_pair = keep_unit[u_idx] & keep_unit[v_idx]
    return keep_pair, threshold, pre_mean


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_bootstrap_scatter(rho_by_pos, plv_by_pos, phase_by_pos,
                           keep_by_pos, method_label, save_path):
    """Three panels, one per position. Scatter rho vs signed coupling
    (PLV * cos(mean phase)). Color by cluster. Annotate slope and R^2
    for the high-amplitude subset."""
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
        plv   = plv_by_pos[pos]
        phase = phase_by_pos[pos]
        keep  = keep_by_pos[pos]
        signed = plv * np.cos(phase)

        clusters = np.array(
            [cluster_of_phase(p) for p in phase],
            dtype=object,
        )
        # Plot all pairs faintly
        ax.scatter(
            signed[~keep], rho[~keep], s=8, alpha=0.15,
            color="lightgray", edgecolors="none",
            label=f"low-amp (n={(~keep).sum()})",
        )
        # Plot high-amp pairs colored by cluster
        for cname, ccolor in cluster_colors.items():
            mask = keep & (clusters == cname)
            if mask.any():
                ax.scatter(
                    signed[mask], rho[mask], s=22, alpha=0.6,
                    color=ccolor, edgecolors="none",
                    label=f"{cname} (n={int(mask.sum())})",
                )

        # Linear fit on high-amp pairs only
        if keep.sum() >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                signed[keep], rho[keep]
            )
            xs = np.linspace(signed[keep].min(), signed[keep].max(), 50)
            ax.plot(
                xs, slope * xs + intercept,
                "k-", linewidth=1.4,
                label=(f"fit: slope={slope:.3f}, "
                       f"R^2={r_value**2:.3f}, "
                       f"p={p_value:.1e}"),
            )
        ax.axhline(0, color="gray", linewidth=0.4, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.4, linestyle=":")
        ax.set_xlabel("PLV * cos(<theta_u - theta_v>)  (signed coupling)")
        if ax is axes[0]:
            ax.set_ylabel(
                "rho_uv = corr_n(Delta r_u, Delta r_v) (high-amp pairs)"
            )
        ax.set_title(POSITION_LABELS[pos], fontsize=11)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        f"Bootstrap test: rho vs signed coupling  ({method_label})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_cluster_conditional(rho_by_pos, plv_by_pos, phase_by_pos,
                              keep_by_pos, method_label, save_path):
    """For each position, four boxplots showing rho distribution split
    by phase cluster. Tests the cluster-specific predictions:
      near_0:    rho > 0
      near_pi:   rho < 0
      near +/- pi/2: rho ~ 0"""
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
        rho = rho_by_pos[pos]
        phase = phase_by_pos[pos]
        keep = keep_by_pos[pos]
        clusters = np.array(
            [cluster_of_phase(p) for p in phase],
            dtype=object,
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
                data.append(np.array([]))
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
        ax.set_xlabel("phase cluster")
    fig.suptitle(
        f"Cluster-conditional rho distributions  ({method_label})",
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
    parser.add_argument("--radius-data", type=str, required=True,
                        help="Path to radius_data.npz from step 1")
    parser.add_argument("--out-dir", type=str, default="toy_sl_step2")
    parser.add_argument("--pre-start", type=int, default=10)
    parser.add_argument("--pre-end",   type=int, default=20)
    parser.add_argument("--post-start", type=int, default=20)
    parser.add_argument("--post-end",   type=int, default=24)
    parser.add_argument("--amp-percentile", type=float, default=25.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading streams and radii from {args.radius_data} ...")
    streams_by_pos, radii_by_pos = load_step1(args.radius_data)
    L = streams_by_pos[5].shape[1]
    print(f"  n_sublayers = {L}")
    print(f"  pre-explosion window  = [{args.pre_start}, {args.pre_end}]")
    print(f"  post-explosion window = [{args.post_start}, {args.post_end}]")

    if args.pre_end >= L or args.post_end >= L:
        raise ValueError(
            f"Window bounds exceed n_sublayers={L}"
        )

    # Containers
    rho_delta_by_pos = {}
    rho_slope_by_pos = {}
    plv_by_pos       = {}
    phase_by_pos     = {}
    keep_by_pos      = {}
    summary_by_pos   = {}

    for pos in POSITIONS:
        print(f"\n=== {POSITION_LABELS[pos]} ===")
        streams = streams_by_pos[pos]
        r = radii_by_pos[pos]
        D = streams.shape[2]

        # Pre-explosion phases & PLV
        theta = per_unit_phases(streams)
        plv, mean_phase, u_idx, v_idx = compute_PLV_and_mean_phase(
            theta, args.pre_start, args.pre_end
        )

        # Post-explosion radius growth (two methods)
        delta_method = compute_delta_r_window(
            r, args.post_start, args.post_end
        )
        slope_method = compute_slope_r_window(
            r, args.post_start, args.post_end
        )

        rho_delta = correlate_over_inputs(delta_method, u_idx, v_idx)
        rho_slope = correlate_over_inputs(slope_method, u_idx, v_idx)

        # Amplitude floor mask
        keep_pair, threshold, per_unit_pre = amplitude_floor_mask(
            r, args.pre_start, args.pre_end,
            percentile=args.amp_percentile,
        )

        plv_by_pos[pos]       = plv
        phase_by_pos[pos]     = mean_phase
        rho_delta_by_pos[pos] = rho_delta
        rho_slope_by_pos[pos] = rho_slope
        keep_by_pos[pos]      = keep_pair

        # Per-position numerical summary
        signed = plv * np.cos(mean_phase)
        def fit(x, y, mask):
            if mask.sum() < 2:
                return {
                    "n": int(mask.sum()),
                    "slope": None, "intercept": None,
                    "r2": None, "p_value": None,
                }
            res = stats.linregress(x[mask], y[mask])
            return {
                "n":         int(mask.sum()),
                "slope":     float(res.slope),
                "intercept": float(res.intercept),
                "r2":        float(res.rvalue ** 2),
                "p_value":   float(res.pvalue),
            }

        all_mask = np.ones_like(keep_pair, dtype=bool)
        fits_delta = {
            "all_pairs":  fit(signed, rho_delta, all_mask),
            "high_amp":   fit(signed, rho_delta, keep_pair),
        }
        fits_slope = {
            "all_pairs":  fit(signed, rho_slope, all_mask),
            "high_amp":   fit(signed, rho_slope, keep_pair),
        }

        # Cluster-conditional medians
        clusters = np.array(
            [cluster_of_phase(p) for p in mean_phase], dtype=object
        )
        cluster_stats = {}
        for c in ["near_0", "near_pi_over_2",
                  "near_minus_pi_over_2", "near_pi"]:
            mask = keep_pair & (clusters == c)
            cluster_stats[c] = {
                "n":              int(mask.sum()),
                "rho_delta_med":  float(np.median(rho_delta[mask]))
                                  if mask.any() else None,
                "rho_slope_med":  float(np.median(rho_slope[mask]))
                                  if mask.any() else None,
                "plv_median":     float(np.median(plv[mask]))
                                  if mask.any() else None,
            }

        summary_by_pos[str(pos)] = {
            "label":             POSITION_LABELS[pos],
            "n_pairs_total":     int(len(plv)),
            "n_high_amp_pairs":  int(keep_pair.sum()),
            "amp_floor":         float(threshold),
            "pre_explosion_window":  [args.pre_start, args.pre_end],
            "post_explosion_window": [args.post_start, args.post_end],
            "fits_delta_method": fits_delta,
            "fits_slope_method": fits_slope,
            "cluster_stats":     cluster_stats,
            "plv_distribution": {
                "mean":   float(plv.mean()),
                "median": float(np.median(plv)),
                "p95":    float(np.percentile(plv, 95)),
            },
            "plv_high_amp_distribution": {
                "mean":   float(plv[keep_pair].mean())
                          if keep_pair.any() else None,
                "median": float(np.median(plv[keep_pair]))
                          if keep_pair.any() else None,
            },
        }

        print(
            f"  n_pairs total={len(plv)}, "
            f"high-amp={keep_pair.sum()}"
        )
        print(f"  PLV (high-amp): mean={plv[keep_pair].mean():.3f}, "
              f"median={np.median(plv[keep_pair]):.3f}")
        print(f"  delta-method fit (high-amp): "
              f"slope={fits_delta['high_amp']['slope']:.3f}, "
              f"R^2={fits_delta['high_amp']['r2']:.3f}, "
              f"p={fits_delta['high_amp']['p_value']:.2e}")
        print(f"  slope-method fit (high-amp): "
              f"slope={fits_slope['high_amp']['slope']:.3f}, "
              f"R^2={fits_slope['high_amp']['r2']:.3f}, "
              f"p={fits_slope['high_amp']['p_value']:.2e}")
        for c, st in cluster_stats.items():
            print(f"  cluster {c}: n={st['n']}, "
                  f"rho_delta_med={st['rho_delta_med']}, "
                  f"rho_slope_med={st['rho_slope_med']}")

    # ------------- Plots -------------
    print("\nMaking plots ...")
    plot_bootstrap_scatter(
        rho_delta_by_pos, plv_by_pos, phase_by_pos, keep_by_pos,
        method_label="delta = r(post_end) - r(post_start)",
        save_path=out_dir / "bootstrap_scatter_delta.png",
    )
    plot_bootstrap_scatter(
        rho_slope_by_pos, plv_by_pos, phase_by_pos, keep_by_pos,
        method_label="slope of r(l) over post-explosion window",
        save_path=out_dir / "bootstrap_scatter_slope.png",
    )
    plot_cluster_conditional(
        rho_delta_by_pos, plv_by_pos, phase_by_pos, keep_by_pos,
        method_label="delta method",
        save_path=out_dir / "cluster_conditional_delta.png",
    )
    plot_cluster_conditional(
        rho_slope_by_pos, plv_by_pos, phase_by_pos, keep_by_pos,
        method_label="slope method",
        save_path=out_dir / "cluster_conditional_slope.png",
    )

    # ------------- Save data -------------
    np.savez(
        out_dir / "bootstrap_data.npz",
        **{f"rho_delta_pos{p}": rho_delta_by_pos[p] for p in POSITIONS},
        **{f"rho_slope_pos{p}": rho_slope_by_pos[p] for p in POSITIONS},
        **{f"plv_pos{p}":        plv_by_pos[p]      for p in POSITIONS},
        **{f"phase_pos{p}":      phase_by_pos[p]    for p in POSITIONS},
        **{f"keep_pos{p}":       keep_by_pos[p]     for p in POSITIONS},
    )

    summary = {
        "radius_data":           args.radius_data,
        "n_sublayers":           int(L),
        "pre_explosion_window":  [args.pre_start, args.pre_end],
        "post_explosion_window": [args.post_start, args.post_end],
        "amp_percentile":        args.amp_percentile,
        "per_position":          summary_by_pos,
    }
    with open(out_dir / "bootstrap_step2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs in {out_dir.resolve()}")
    print("  bootstrap_scatter_delta.png")
    print("  bootstrap_scatter_slope.png")
    print("  cluster_conditional_delta.png")
    print("  cluster_conditional_slope.png")
    print("  bootstrap_data.npz")
    print("  bootstrap_step2_summary.json")


if __name__ == "__main__":
    main()
