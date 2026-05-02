"""
Phase-space oscillator test: x_ell vs Delta x_ell.

For a clean harmonic oscillator x(l) = A cos(omega*l + phi), the residual
update Delta x(l) = x(l+1) - x(l) is approximately -A*omega*sin(omega*l +
phi + omega/2). Plotting (x_l, Delta x_l) traces a closed ellipse with x
and Delta x in quadrature: 90 degrees out of phase. A unit whose
trajectory has merely been projected onto a plane that looks rotational
will fail this test, typically by showing a degenerate (linear) cloud or
a non-quadrature phase relationship.

This script tests, per (input, unit), how closely the (x, Delta x)
trajectory matches the oscillator picture by computing four diagnostics
on the L-1 paired points (x_l, Delta x_l) along the depth axis:

  r         Pearson correlation between x_l and Delta x_l along l.
            For an oscillator this is near 0 (x and dx are orthogonal).
            For a degenerate trajectory (line in phase space) it is +/- 1.

  A_norm    Signed area enclosed by the (x, Delta x) trajectory using the
            shoelace formula, normalized by std(x) * std(Delta x) * L so
            values are comparable across units of different amplitudes.
            Magnitude indicates rotational strength. Sign indicates the
            direction of rotation in phase space (forward-time oscillation
            corresponds to clockwise motion, i.e. NEGATIVE signed area).

  R_PCA     Ratio of smaller to larger eigenvalue of the 2x2 covariance
            matrix of the (x, Delta x) cloud, computed AFTER rescaling
            Delta x so var(rescaled Delta x) = var(x). This rescaling is
            essential: without it, an ideal oscillator at frequency omega
            would give R_PCA ~ omega^2, which is small for slow
            oscillations and would falsely flag genuine oscillators as
            degenerate. With the rescaling, an ideal oscillator at any
            frequency gives R_PCA = 1; a degenerate (linear) trajectory
            gives R_PCA = 0.

            Algebraic note: after the rescaling, R_PCA reduces exactly to
            (1 - |r|) / (1 + |r|). It is therefore a smooth transformation
            of |r| onto a [0, 1] "circularity" axis, not an independent
            test. It is reported because the [0, 1] scale is more
            interpretable than the [-1, 1] correlation, and because the
            redundancy provides a sanity check.

  delta_theta  Mean phase offset arg(Hilbert(Delta x)) - arg(Hilbert(x))
            per (input, unit), aggregated across inputs by circular mean.
            For a clean cosine oscillator delta_theta = +pi/2 (Delta x
            leads x by 90 degrees) for forward-rotating motion. Since
            this comes from the Hilbert transform of each component
            independently, it is genuinely independent of r.

The two scientifically independent diagnostics are r and delta_theta;
A_norm reports the sign and magnitude of rotation, R_PCA recasts r on a
more visual scale.

Aggregation: each diagnostic is computed per (input, unit), then taken
median (or circular mean for delta_theta) across inputs to give one
value per unit. The full per-input distribution is also retained for
plotting.

A unit is classified "oscillator" if it satisfies all of:
  |r|     < r_threshold              (default 0.30)
  R_PCA   > pca_threshold             (default 0.40, consistent with r=0.30
                                       under the rescaling identity above)
  ||delta_theta| - pi/2| < dtheta_thresh (default pi/4)

These thresholds are conservative; the diagnostics are also reported
as continuous distributions so you can apply your own cuts.

A unit is classified "degenerate" if R_PCA is below half the threshold
(very thin trajectory, regardless of correlation). This is the failure
mode of "rotational projection that isn't dynamical": the (x, Delta x)
cloud collapses to a line.

Otherwise the unit is "other" (intermediate cases).

Inputs: one or more analytic_signal.npz files (output of phase_step0).

Outputs:
  phase_portraits_{label}.png       9-panel grid of selected unit
                                    phase portraits, with all input
                                    trajectories overlaid as a colored
                                    point cloud (color = sublayer
                                    fraction, dark = early), one bold
                                    polyline showing input 0's actual
                                    trajectory, and a 2-sigma covariance
                                    ellipse overlaid in red.
  diagnostics_{label}.png           Histograms of r, A_norm, R_PCA,
                                    delta_theta across the D units.
  scatter_{label}.png               Scatter |r| vs R_PCA per unit, with
                                    classification regions shaded. Note:
                                    points should fall on the curve
                                    R_PCA = (1 - |r|) / (1 + |r|).
  phase_space_comparison.png        Cross-model summary: stacked bar of
                                    oscillator / degenerate / other
                                    fractions per checkpoint.
  phase_space_summary.json          Per-model numerical summary.

Usage:
  python phase_space_test.py \\
      --npz random.npz gelu.npz swiglu.npz \\
      --labels random gelu swiglu \\
      --position 5 \\
      --out-dir phase_space_pos5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal import hilbert as scipy_hilbert


PALETTE = ["C0", "C3", "C2", "C4", "C5", "C6", "C7", "C8"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_x_at_position(path, position):
    """Load the centered depth trajectory at the requested token position."""
    z = np.load(path)
    return z[f"x_used_pos{position}"].astype(np.float64)


# ---------------------------------------------------------------------------
# Per-trajectory diagnostics
# ---------------------------------------------------------------------------

def pair_x_dx(x):
    """Return (x_pair, dx) of shape (N, L-1, D) each.
    dx[:, l, :] = x[:, l+1, :] - x[:, l, :]   (forward difference)."""
    dx = np.diff(x, axis=1)
    x_pair = x[:, :-1, :]
    return x_pair, dx


def per_traj_pearson(x, dx):
    """Pearson(x, dx) over l, per (input, unit). x, dx shape (N, L-1, D)."""
    x_c  = x  - x.mean(axis=1, keepdims=True)
    dx_c = dx - dx.mean(axis=1, keepdims=True)
    num = (x_c * dx_c).sum(axis=1)
    den = np.sqrt((x_c ** 2).sum(axis=1) * (dx_c ** 2).sum(axis=1)) + 1e-30
    return num / den


def per_traj_signed_area(x, dx):
    """Shoelace signed area of the (x_l, dx_l) trajectory, per (input, unit).
    A = (1/2) sum_l (x_l * dx_{l+1} - x_{l+1} * dx_l).

    For forward-time harmonic oscillation (x = cos, dx = -sin) the trajectory
    goes (1, 0) -> (0, -omega) -> (-1, 0) -> (0, +omega) -> (1, 0),
    which is CLOCKWISE in (x, dx), giving NEGATIVE signed area."""
    if x.shape[1] < 2:
        return np.zeros((x.shape[0], x.shape[2]))
    a = x[:, :-1, :] * dx[:, 1:,  :]
    b = x[:, 1:,  :] * dx[:, :-1, :]
    return 0.5 * (a - b).sum(axis=1)


def per_traj_pca_aspect(x, dx):
    """Per-trajectory PCA aspect ratio of (x, dx) AFTER rescaling dx so
    var(rescaled dx) = var(x), per (input, unit).

    The rescaling makes the diagnostic frequency-invariant. Without it, a
    clean oscillator at frequency omega would give a non-isotropic ellipse
    with aspect ratio omega^2 (small for slow oscillations), which would
    falsely flag genuine oscillators as degenerate.

    After rescaling, the 2x2 covariance is [[v, c'], [c', v]] with v = var(x)
    and c' = sqrt(var(x) / var(dx)) * cov(x, dx). Eigenvalues are v +/- |c'|,
    and the ratio simplifies algebraically to (1 - |r|) / (1 + |r|) where
    r is the Pearson correlation. So this returns a transformation of |r|
    onto a [0, 1] circularity scale: 1 = circular, 0 = line."""
    x_c  = x  - x.mean(axis=1, keepdims=True)
    dx_c = dx - dx.mean(axis=1, keepdims=True)
    var_x  = (x_c  ** 2).mean(axis=1)
    var_dx = (dx_c ** 2).mean(axis=1)
    cov_xy = (x_c * dx_c).mean(axis=1)

    # Rescale dx so var(rescaled dx) = var(x). Scale factor s = sqrt(var_x/var_dx);
    # after scaling, cov becomes s * cov.
    s = np.sqrt(var_x / (var_dx + 1e-30))
    cov_rescaled = s * cov_xy

    # Eigenvalues of [[var_x, c'], [c', var_x]]:
    lam_max = var_x + np.abs(cov_rescaled)
    lam_min = np.maximum(var_x - np.abs(cov_rescaled), 0.0)
    return lam_min / (lam_max + 1e-30)


def per_traj_quadrature_phase(x, dx):
    """Mean phase offset arg(Z_dx / Z_x) over l, per (input, unit), where Z is
    the analytic signal computed by Hilbert transform along the depth axis.

    For a clean cosine oscillator x = cos(theta), dx ~ -sin(theta) =
    cos(theta + pi/2), so arg(Z_dx) - arg(Z_x) = +pi/2.

    Returned values are in (-pi, pi]."""
    z_x  = scipy_hilbert(x,  axis=1)
    z_dx = scipy_hilbert(dx, axis=1)
    ratio = z_dx / (z_x + 1e-30)
    return np.angle(ratio.mean(axis=1))


def collect_metrics(x):
    """Compute all phase-space diagnostics for a centered (N, L, D) trajectory."""
    x_pair, dx = pair_x_dx(x)

    r       = per_traj_pearson(x_pair, dx)            # (N, D)
    A       = per_traj_signed_area(x_pair, dx)         # (N, D)
    R       = per_traj_pca_aspect(x_pair, dx)          # (N, D)  (rescaled)
    dtheta  = per_traj_quadrature_phase(x_pair, dx)    # (N, D)

    # Scale-invariant signed area: divide by std(x) * std(dx) * L (per traj).
    std_x  = x_pair.std(axis=1)
    std_dx = dx.std(axis=1)
    L_pair = x_pair.shape[1]
    A_norm = A / ((std_x * std_dx * L_pair) + 1e-30)

    # Aggregate over inputs
    r_per_unit       = np.median(r, axis=0)
    A_norm_per_unit  = np.median(A_norm, axis=0)
    R_per_unit       = np.median(R, axis=0)

    # Circular mean for delta_theta
    dtheta_complex   = np.exp(1j * dtheta)
    dtheta_resultant = np.abs(dtheta_complex.mean(axis=0))
    dtheta_per_unit  = np.angle(dtheta_complex.mean(axis=0))

    return {
        "shape":                     x.shape,
        "x_pair":                    x_pair,
        "dx":                        dx,
        "r":                         r,
        "A":                         A,
        "A_norm":                    A_norm,
        "R":                         R,
        "dtheta":                    dtheta,
        "r_per_unit":                r_per_unit,
        "A_norm_per_unit":           A_norm_per_unit,
        "R_per_unit":                R_per_unit,
        "dtheta_per_unit":           dtheta_per_unit,
        "dtheta_resultant_per_unit": dtheta_resultant,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_units(m, r_thresh=0.3, pca_thresh=0.4, dtheta_thresh=np.pi / 4):
    """Three-way classification per unit: oscillator / degenerate / other.

    Note: with R_PCA computed on rescaled (x, dx) clouds, the criteria
    'R_PCA > pca_thresh' and '|r| < r_thresh' are linked by the identity
    R_PCA = (1 - |r|) / (1 + |r|). The default thresholds are set so the
    two are consistent: r_thresh = 0.30 corresponds to R_PCA = 0.538, so
    pca_thresh = 0.40 is a slightly looser secondary check."""
    r_abs   = np.abs(m["r_per_unit"])
    R       = m["R_per_unit"]
    dt_dev  = np.abs(np.abs(m["dtheta_per_unit"]) - np.pi / 2)

    is_osc        = (r_abs < r_thresh) & (R > pca_thresh) & (dt_dev < dtheta_thresh)
    is_degenerate = (R < pca_thresh / 2)
    is_other      = ~is_osc & ~is_degenerate
    return is_osc, is_degenerate, is_other


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _draw_cov_ellipse(ax, x_pts, y_pts, n_sigma=2.0,
                      color="red", alpha=0.7, lw=1.5):
    """Draw the n-sigma covariance ellipse of the (x_pts, y_pts) cloud."""
    if len(x_pts) < 3:
        return
    cov = np.cov(x_pts, y_pts)
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    semi_a = n_sigma * np.sqrt(max(eigvals[1], 0.0))
    semi_b = n_sigma * np.sqrt(max(eigvals[0], 0.0))
    e = Ellipse(
        xy=(x_pts.mean(), y_pts.mean()),
        width=2 * semi_a, height=2 * semi_b, angle=angle,
        fill=False, edgecolor=color, alpha=alpha, linewidth=lw,
    )
    ax.add_patch(e)


def plot_phase_portraits(m, label, out_path):
    """9-panel phase portrait grid: 3 highest, 3 median, 3 lowest |A_norm|."""
    A = np.abs(m["A_norm_per_unit"])
    order = np.argsort(A)
    D = len(order)
    if D < 9:
        chosen = list(order[-min(D, 9):])
        chosen_labels = [f"u{u}" for u in chosen]
    else:
        chosen = [
            order[-1], order[-2], order[-3],
            order[D // 2 - 1], order[D // 2], order[D // 2 + 1],
            order[0], order[1], order[2],
        ]
        chosen_labels = (
            ["high |A|"] * 3
            + ["median |A|"] * 3
            + ["low |A|"] * 3
        )

    x_pair = m["x_pair"]
    dx     = m["dx"]
    N, Lm1, _ = x_pair.shape
    layer_norm = np.arange(Lm1) / max(Lm1 - 1, 1)

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    for ax, u, lbl in zip(axes.flatten(), chosen, chosen_labels):
        xs  = x_pair[:, :, u].flatten()
        dys = dx[:, :, u].flatten()
        cs  = np.tile(layer_norm, N)
        ax.scatter(xs, dys, c=cs, cmap="viridis",
                   s=4, alpha=0.20, edgecolors="none")
        ax.plot(x_pair[0, :, u], dx[0, :, u],
                "-", color="black", linewidth=0.8, alpha=0.8)
        ax.scatter(x_pair[0, 0, u], dx[0, 0, u],
                   marker="o", s=40, facecolor="white",
                   edgecolor="black", linewidth=1.0, zorder=5)
        _draw_cov_ellipse(ax, xs, dys, n_sigma=2.0)

        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(r"$x_\ell$")
        ax.set_ylabel(r"$\Delta x_\ell$")
        ax.set_title(
            f"unit {u} ({lbl})\n"
            f"r={m['r_per_unit'][u]:+.2f}, "
            f"$A_{{\\mathrm{{norm}}}}$={m['A_norm_per_unit'][u]:+.3f}, "
            f"$R_{{\\mathrm{{PCA}}}}$={m['R_per_unit'][u]:.2f}, "
            f"$\\Delta\\theta$={m['dtheta_per_unit'][u]:+.2f}",
            fontsize=9,
        )

    fig.suptitle(
        f"Phase portrait $(x_\\ell, \\Delta x_\\ell)$: {label}\n"
        "color = sublayer fraction (dark = early, bright = late); "
        "white circle = trajectory start; red = 2$\\sigma$ cov ellipse",
        fontsize=12, y=0.998,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(m, label, out_path):
    """Four histograms across the D units: r, A_norm, R_PCA, delta_theta."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(m["r_per_unit"], bins=20, range=(-1, 1), color="C0", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0,
               label="oscillator ideal (r=0)")
    ax.set_xlabel(r"Pearson correlation $r$ between $x$ and $\Delta x$")
    ax.set_ylabel("number of units")
    ax.set_title("(a) correlation: 0 = oscillator, $\\pm 1$ = degenerate")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    A = m["A_norm_per_unit"]
    bound = max(np.abs(A).max(), 1e-6)
    ax.hist(A, bins=20, range=(-bound, bound), color="C2", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"normalized signed area $A_{\mathrm{norm}}$")
    ax.set_ylabel("number of units")
    ax.set_title("(b) signed area: nonzero magnitude = rotational")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.hist(m["R_per_unit"], bins=20, range=(0, 1), color="C3", alpha=0.85)
    ax.axvline(0.4, color="black", linestyle=":", linewidth=1.0,
               label="default threshold 0.4")
    ax.set_xlabel(r"PCA aspect ratio $R_{\mathrm{PCA}}$ (rescaled)")
    ax.set_ylabel("number of units")
    ax.set_title("(c) PCA aspect (rescaled): 1 = circular, 0 = line")
    ax.set_xlim(0, 1)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    dt = m["dtheta_per_unit"]
    ax.hist(dt, bins=24, range=(-np.pi, np.pi), color="C4", alpha=0.85)
    ax.axvline( np.pi / 2, color="black", linestyle="--", linewidth=1.0,
               label=r"$+\pi/2$ (forward osc)")
    ax.axvline(-np.pi / 2, color="black", linestyle=":",  linewidth=1.0,
               label=r"$-\pi/2$ (reverse osc)")
    ax.axvline(0, color="red", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$\Delta\theta = \arg(Z_{\Delta x}) - \arg(Z_x)$")
    ax.set_ylabel("number of units")
    ax.set_title(r"(d) quadrature phase: $\pm\pi/2$ = oscillator")
    ax.set_xlim(-np.pi, np.pi)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Phase-space diagnostics: {label}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(m, label, out_path,
                 r_thresh=0.3, pca_thresh=0.4, dtheta_thresh=np.pi / 4):
    """Scatter |r| vs R_PCA, colored by classification.

    With R_PCA on the rescaled (x, dx), all points should fall along the
    curve R_PCA = (1 - |r|) / (1 + |r|). The reference curve is overlaid."""
    is_osc, is_deg, is_other = classify_units(
        m, r_thresh=r_thresh, pca_thresh=pca_thresh, dtheta_thresh=dtheta_thresh,
    )
    r_abs = np.abs(m["r_per_unit"])
    R     = m["R_per_unit"]

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.add_patch(plt.Rectangle(
        (0, pca_thresh), r_thresh, 1 - pca_thresh,
        facecolor="C2", alpha=0.07, zorder=0,
    ))
    ax.add_patch(plt.Rectangle(
        (0, 0), 1, pca_thresh / 2,
        facecolor="C3", alpha=0.07, zorder=0,
    ))

    # Reference curve: R_PCA = (1 - |r|) / (1 + |r|)
    rr = np.linspace(0, 1, 200)
    ax.plot(rr, (1 - rr) / (1 + rr), color="black",
            linestyle="-", alpha=0.4, linewidth=1.2,
            label=r"$(1-|r|)/(1+|r|)$ (rescaled identity)")

    if is_other.any():
        ax.scatter(r_abs[is_other], R[is_other], s=40, alpha=0.6,
                   color="gray", edgecolors="black", linewidths=0.4,
                   label=f"other (n={is_other.sum()})")
    if is_deg.any():
        ax.scatter(r_abs[is_deg], R[is_deg], s=40, alpha=0.7,
                   color="C3", edgecolors="black", linewidths=0.4,
                   label=f"degenerate (n={is_deg.sum()})")
    if is_osc.any():
        ax.scatter(r_abs[is_osc], R[is_osc], s=70, alpha=0.85,
                   color="C2", marker="*",
                   edgecolors="black", linewidths=0.5,
                   label=f"oscillator (n={is_osc.sum()})")

    ax.axhline(pca_thresh, color="C2", linestyle=":", alpha=0.6)
    ax.axhline(pca_thresh / 2, color="C3", linestyle=":", alpha=0.6)
    ax.axvline(r_thresh, color="C2", linestyle=":", alpha=0.6)

    ax.set_xlabel(r"$|r|$  (correlation between $x$ and $\Delta x$)")
    ax.set_ylabel(r"$R_{\mathrm{PCA}}$ (rescaled)")
    ax.set_title(
        f"Phase-space classification: {label}\n"
        f"green zone = oscillator candidate, red shade = degenerate"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results, out_path, position,
                    r_thresh=0.3, pca_thresh=0.4, dtheta_thresh=np.pi / 4):
    """Stacked bar: fraction oscillator/degenerate/other per model."""
    labels = list(results.keys())
    osc_frac = []
    deg_frac = []
    oth_frac = []
    for label in labels:
        m = results[label]
        o, d, ot = classify_units(
            m, r_thresh=r_thresh, pca_thresh=pca_thresh,
            dtheta_thresh=dtheta_thresh,
        )
        D = len(o)
        osc_frac.append(o.sum() / D)
        deg_frac.append(d.sum() / D)
        oth_frac.append(ot.sum() / D)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    width = 0.7
    osc_arr = np.array(osc_frac)
    deg_arr = np.array(deg_frac)
    oth_arr = np.array(oth_frac)
    ax.bar(x, osc_arr, width, label="oscillator",  color="C2")
    ax.bar(x, deg_arr, width, bottom=osc_arr, label="degenerate", color="C3")
    ax.bar(x, oth_arr, width, bottom=osc_arr + deg_arr,
           label="other", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of units")
    ax.set_title(f"Phase-space classification per model (position {position})")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(results, position,
                  r_thresh, pca_thresh, dtheta_thresh):
    summary = {
        "position": position,
        "metrics_explained": {
            "r":      "Pearson(x_l, dx_l) over l per (input, unit). Median over inputs. ~0 = oscillator, +/-1 = degenerate.",
            "A_norm": "Shoelace signed area of (x, dx) trajectory per (input, unit), normalized by std(x)*std(dx)*L. Magnitude = rotation strength. Negative = forward (clockwise) oscillation.",
            "R_PCA":  "Smaller/larger eigenvalue of 2x2 cov of (x, dx) cloud computed AFTER rescaling dx so var(rescaled dx) = var(x). Frequency-invariant. 1 = circular oscillator at any frequency, 0 = line. Equals (1 - |r|) / (1 + |r|) by construction.",
            "dtheta": "Mean phase offset arg(Hilbert(dx)) - arg(Hilbert(x)) per (input, unit). Circular mean over inputs. +pi/2 = forward oscillator.",
        },
        "thresholds": {
            "r_threshold":      r_thresh,
            "pca_threshold":    pca_thresh,
            "dtheta_threshold_radians": dtheta_thresh,
        },
        "per_model": {},
    }
    for label, m in results.items():
        is_osc, is_deg, is_other = classify_units(
            m, r_thresh=r_thresh, pca_thresh=pca_thresh,
            dtheta_thresh=dtheta_thresh,
        )
        D = m["shape"][2]
        summary["per_model"][label] = {
            "shape": {"N": int(m["shape"][0]), "L": int(m["shape"][1]), "D": int(D)},
            "n_oscillator":         int(is_osc.sum()),
            "n_degenerate":         int(is_deg.sum()),
            "n_other":              int(is_other.sum()),
            "fraction_oscillator":  float(is_osc.sum() / D),
            "r_per_unit": {
                "mean":       float(m["r_per_unit"].mean()),
                "median":     float(np.median(m["r_per_unit"])),
                "abs_mean":   float(np.abs(m["r_per_unit"]).mean()),
                "abs_median": float(np.median(np.abs(m["r_per_unit"]))),
            },
            "A_norm_per_unit": {
                "mean":     float(m["A_norm_per_unit"].mean()),
                "abs_mean": float(np.abs(m["A_norm_per_unit"]).mean()),
                "median":   float(np.median(m["A_norm_per_unit"])),
            },
            "R_PCA_per_unit": {
                "mean":   float(m["R_per_unit"].mean()),
                "median": float(np.median(m["R_per_unit"])),
            },
            "dtheta_per_unit": {
                "circular_mean":  float(m["dtheta_per_unit"].mean()),
                "abs_mean":       float(np.abs(m["dtheta_per_unit"]).mean()),
                "deviation_from_pi_over_2_radians": {
                    "mean":   float(np.abs(np.abs(m["dtheta_per_unit"]) - np.pi / 2).mean()),
                    "median": float(np.median(np.abs(np.abs(m["dtheta_per_unit"]) - np.pi / 2))),
                },
                "resultant_length": {
                    "mean":   float(m["dtheta_resultant_per_unit"].mean()),
                    "median": float(np.median(m["dtheta_resultant_per_unit"])),
                },
            },
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, nargs="+", required=True,
                        help="One or more analytic_signal.npz files.")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Display labels per --npz (default: filenames).")
    parser.add_argument("--position", type=int, default=5)
    parser.add_argument("--out-dir",  type=str, required=True)
    parser.add_argument("--r-threshold",   type=float, default=0.30,
                        help="|r| below this counts as low correlation.")
    parser.add_argument("--pca-threshold", type=float, default=0.40,
                        help="R_PCA above this counts as non-degenerate. "
                             "Note: linked to r_threshold by the rescaling "
                             "identity R_PCA = (1-|r|)/(1+|r|).")
    parser.add_argument("--dtheta-threshold", type=float,
                        default=float(np.pi / 4),
                        help="||dtheta| - pi/2| below this counts as quadrature.")
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [Path(p).stem for p in args.npz]
    if len(args.labels) != len(args.npz):
        raise SystemExit("--labels must match --npz length")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {len(args.npz)} npz files at position {args.position} ...")
    results = {}
    for path, label in zip(args.npz, args.labels):
        print(f"  {label:<22} <- {path}")
        x = load_x_at_position(path, args.position)
        m = collect_metrics(x)
        results[label] = m
        print(f"    shape (N, L, D) = {m['shape']}")

    print(f"\nMaking plots ...")
    for label in args.labels:
        out = out_dir / f"phase_portraits_{label}.png"
        plot_phase_portraits(results[label], label, out)
        print(f"  {out.name}")
        out = out_dir / f"diagnostics_{label}.png"
        plot_diagnostics(results[label], label, out)
        print(f"  {out.name}")
        out = out_dir / f"scatter_{label}.png"
        plot_scatter(
            results[label], label, out,
            r_thresh=args.r_threshold,
            pca_thresh=args.pca_threshold,
            dtheta_thresh=args.dtheta_threshold,
        )
        print(f"  {out.name}")

    plot_comparison(
        results, out_dir / "phase_space_comparison.png", args.position,
        r_thresh=args.r_threshold,
        pca_thresh=args.pca_threshold,
        dtheta_thresh=args.dtheta_threshold,
    )
    print(f"  phase_space_comparison.png")

    summary = build_summary(
        results, args.position,
        r_thresh=args.r_threshold,
        pca_thresh=args.pca_threshold,
        dtheta_thresh=args.dtheta_threshold,
    )
    with open(out_dir / "phase_space_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  phase_space_summary.json")

    # ---------- Console table ----------
    print(f"\n--- Per-model phase-space classification ---")
    print(f"{'model':<22} {'D':>4} {'osc':>5} {'deg':>5} {'other':>6} "
          f"{'osc%':>7} {'<|r|>':>8} {'<R_PCA>':>9} {'<|dthe|/(pi/2)>':>17}")
    for label in args.labels:
        s = summary["per_model"][label]
        rel_dt = (s["dtheta_per_unit"]["abs_mean"] / (np.pi / 2))
        print(
            f"{label:<22} {s['shape']['D']:>4} "
            f"{s['n_oscillator']:>5} {s['n_degenerate']:>5} {s['n_other']:>6} "
            f"{100 * s['fraction_oscillator']:>6.1f}% "
            f"{s['r_per_unit']['abs_mean']:>8.3f} "
            f"{s['R_PCA_per_unit']['mean']:>9.3f} "
            f"{rel_dt:>17.3f}"
        )

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
