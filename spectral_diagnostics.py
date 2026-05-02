"""
Spectral diagnostics for residual-stream depth trajectories.

After phase_step0_analytic_signal.py establishes z_u(l), this script
asks WHICH of four spectral regimes each model is in:

  1. Multi-tone:           energy concentrated at one or more discrete
                           bins; clean separation between components;
                           stationary across depth.
  2. Frequency-modulated:  a single oscillation whose frequency drifts
                           systematically with l; energy spread across
                           a band even though only one component exists.
  3. Broadband + osc:      a clean oscillation sitting on top of a
                           broadband background; one peak, smooth tail.
  4. Multi-component NS:   different oscillatory components dominant at
                           different layers, possibly overlapping.

Diagnostics:

  (a) FFT magnitude spectrum on log scale (per unit, with population
      median and inter-quartile band).
      - Discrete peaks vs smooth decay: multi-tone vs broadband.
      - One peak + flat tail: broadband + osc.

  (b) STFT spectrogram (per unit, mean power averaged over inputs).
      - Horizontal stripes => stationary multi-tone.
      - Drifting curves    => FM.
      - Constant smear     => stationary broadband.
      - Different peaks in different time windows => multi-component NS.

  (c) Spectral entropy H / log(n_bins), per unit.
      Low (~0.3) => concentrated. High (~0.9) => spread (white-noise limit).

  (d) Spectral flatness (Wiener entropy): geometric / arithmetic mean
      of the power spectrum, in [0, 1].
      Close to 0 => pure tone. Close to 1 => white noise.

  (e) Frequency drift: linear slope of input-averaged omega_u(l) (from
      the Hilbert transform) over the interior of l. Magnitude > ~10%
      of the carrier 2*pi/L is FM-like.

  (f) STFT nonstationarity: peak frequency in early half of l vs late
      half of l. Large shift indicates non-stationary multi-component.

CAVEAT: With L=25 sublayers, the FFT has 13 bins and the STFT with
nperseg=8, noverlap=6 has 5 freq bins x ~14 time frames. Both
resolutions are fundamentally limited; these are population-level
measures (mean / median over inputs and units), not single-trajectory
spectroscopy.

Inputs: one or more analytic_signal.npz files (output of phase_step0).

Outputs:
  spectral_overview.png            FFT, STFT mean, entropy, flatness, drift
  spectrogram_examples_{label}.png 6 representative unit spectrograms
  scenario_fit.png                 scenario fit scores per model
  spectral_summary.json            per-model numerical summary

Usage:
  python spectral_diagnostics.py \
      --npz toy_phase_step0_random/analytic_signal.npz \
            toy_phase_step0_gelu/analytic_signal.npz \
            toy_phase_step0_swiglu/analytic_signal.npz \
      --labels random gelu_trained swiglu_trained \
      --position 5 \
      --out-dir spectral_diagnostics_pos5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft


EDGE_PAD = 2
DEFAULT_NPERSEG = 8
DEFAULT_NOVERLAP = 6   # hop = nperseg - noverlap = 2

PALETTE = ["C0", "C3", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_position_data(path, position):
    """Return the centered depth trajectory and instantaneous frequency."""
    z = np.load(path)
    return {
        "x":     z[f"x_used_pos{position}"].astype(np.float32),  # (N, L, D)
        "omega": z[f"omega_pos{position}"].astype(np.float32),   # (N, L, D)
    }


# ---------------------------------------------------------------------------
# Spectrum, entropy, flatness
# ---------------------------------------------------------------------------

def fft_power(x):
    """rFFT power spectrum along the depth axis. x: (N, L, D) -> (N, n_bins, D)."""
    F = np.fft.rfft(x, axis=1)
    return (np.abs(F) ** 2).astype(np.float64)


def _slice_axis(arr, axis, sl):
    """Return arr sliced along the given axis using slice object sl."""
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = sl
    return arr[tuple(slicer)]


def spectral_entropy_normalized(P, axis=-1, skip_dc=True):
    """Shannon entropy along axis, divided by log(n_bins). In [0, 1].

    skip_dc=True drops index 0 along axis. The depth trajectories used
    here are centered, so the DC bin is structurally zero and would
    otherwise leave entropy biased by the choice of epsilon."""
    if skip_dc:
        P = _slice_axis(P, axis, slice(1, None))
    p = P / (P.sum(axis=axis, keepdims=True) + 1e-30)
    H = -np.sum(p * np.log(p + 1e-30), axis=axis)
    return H / np.log(P.shape[axis])


def spectral_flatness(P, axis=-1, skip_dc=True):
    """Wiener entropy: geometric / arithmetic mean. In [0, 1].
    Close to 0 = pure tone. Close to 1 = white noise.

    skip_dc=True drops index 0 along axis. With a centered signal the
    DC bin is exactly zero and log(0) corrupts the geometric mean."""
    if skip_dc:
        P = _slice_axis(P, axis, slice(1, None))
    log_P = np.log(P + 1e-30)
    geo = np.exp(log_P.mean(axis=axis))
    arith = P.mean(axis=axis) + 1e-30
    return geo / arith


def count_peaks_above_threshold(p, height_ratio=0.10, min_separation=1):
    """Count strict local maxima above height_ratio * max(p)."""
    threshold = p.max() * height_ratio
    peaks = []
    for i in range(len(p)):
        is_peak = p[i] > threshold
        if i > 0 and p[i] <= p[i - 1]:
            is_peak = False
        if i < len(p) - 1 and p[i] <= p[i + 1]:
            is_peak = False
        if is_peak:
            if not peaks or i - peaks[-1] >= min_separation:
                peaks.append(i)
    return len(peaks), peaks


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------

def compute_stft_per_unit(x, nperseg, noverlap):
    """Mean STFT power across inputs per unit.
    x: (N, L, D). Returns f, t, S where S is (n_freq, n_time, D)."""
    # scipy.signal.stft on x with axis=1 returns Z of shape
    # (N, n_freq, D, n_time). Replace the L axis with frequency, then
    # append a new time axis at the end.
    f, t, Z = stft(
        x, nperseg=nperseg, noverlap=noverlap,
        boundary="zeros", padded=True, axis=1,
    )
    P = (np.abs(Z) ** 2).mean(axis=0)            # (n_freq, D, n_time)
    return f, t, P.transpose(0, 2, 1)            # (n_freq, n_time, D)


def spectral_centroid_per_frame(S, f):
    """Frequency centroid at each time frame, per unit.
    S: (n_freq, n_time, D), f: (n_freq,) -> (n_time, D)."""
    weighted = (f[:, None, None] * S).sum(axis=0)
    total = S.sum(axis=0) + 1e-30
    return weighted / total


def stft_nonstationarity(S, f):
    """Spectral nonstationarity diagnostics.

    Returns dict with:
      early_peak, late_peak: peak frequency in first / second half (D,)
      peak_shift:            |early - late|                       (D,)
      shape_distance:        cosine distance between early / late
                             half mean spectra, per unit          (D,)
      log_amplitude_growth:  log10(total power late / total power early), per unit (D,)
    """
    n_t = S.shape[1]
    early = S[:, :n_t // 2, :].mean(axis=1)      # (n_freq, D)
    late  = S[:, n_t // 2:, :].mean(axis=1)
    early_peak = f[np.argmax(early, axis=0)]
    late_peak  = f[np.argmax(late,  axis=0)]
    peak_shift = np.abs(early_peak - late_peak)

    early_norm = early / (np.linalg.norm(early, axis=0, keepdims=True) + 1e-30)
    late_norm  = late  / (np.linalg.norm(late,  axis=0, keepdims=True) + 1e-30)
    cos_sim = (early_norm * late_norm).sum(axis=0)
    shape_distance = 1 - cos_sim

    e_total = early.sum(axis=0) + 1e-30
    l_total = late.sum(axis=0) + 1e-30
    log_amplitude_growth = np.log10(l_total / e_total)

    return {
        "early_peak":           early_peak,
        "late_peak":            late_peak,
        "peak_shift":           peak_shift,
        "shape_distance":       shape_distance,
        "log_amplitude_growth": log_amplitude_growth,
    }


# ---------------------------------------------------------------------------
# Frequency drift (Hilbert-based)
# ---------------------------------------------------------------------------

def fit_linear_slope(y, axis):
    """Least-squares slope of y vs index along axis."""
    n = y.shape[axis]
    x = np.arange(n, dtype=np.float64)
    x_c = x - x.mean()
    var_x = (x_c ** 2).sum() + 1e-30
    shape = [1] * y.ndim
    shape[axis] = n
    return (x_c.reshape(shape) * y).sum(axis=axis) / var_x


def hilbert_omega_drift(omega, edge_pad=EDGE_PAD):
    """Linear slope of input-averaged omega(l) over interior l, per unit.
    omega: (N, L, D) -> (D,)."""
    L = omega.shape[1]
    if L <= 2 * edge_pad + 1:
        return np.zeros(omega.shape[2], dtype=np.float64)
    interior = omega[:, edge_pad:L - edge_pad, :].mean(axis=0)   # (L_int, D)
    return fit_linear_slope(interior, axis=0)


# ---------------------------------------------------------------------------
# Aggregate metrics + scenario scoring
# ---------------------------------------------------------------------------

def collect_metrics(model_data, nperseg, noverlap):
    x = model_data["x"]
    omega = model_data["omega"]
    N, L, D = x.shape

    # FFT
    P_full = fft_power(x)                                       # (N, n_bins, D)
    P_per_unit = P_full.mean(axis=0)                            # (n_bins, D)
    norm = P_per_unit.sum(axis=0, keepdims=True) + 1e-30
    P_per_unit_normed = P_per_unit / norm                       # (n_bins, D)
    n_bins = P_full.shape[1]

    # Entropy and flatness, computed on each unit's input-averaged spectrum
    H_per_unit = spectral_entropy_normalized(P_per_unit, axis=0)   # (D,)
    F_per_unit = spectral_flatness(P_per_unit, axis=0)             # (D,)

    # Peak counts and dominant-bin share
    peak_counts = np.array([
        count_peaks_above_threshold(P_per_unit[:, u])[0]
        for u in range(D)
    ])
    max_bin   = np.argmax(P_per_unit, axis=0)
    max_share = P_per_unit.max(axis=0) / norm.squeeze(0)

    # Hilbert drift
    drift_hilbert = hilbert_omega_drift(omega)
    carrier = 2 * np.pi / L
    drift_hilbert_rel = drift_hilbert / carrier

    # STFT
    f_stft, t_stft, S_stft = compute_stft_per_unit(x, nperseg, noverlap)
    ns = stft_nonstationarity(S_stft, f_stft)
    f_max = float(f_stft.max() + 1e-30)
    peak_shift_rel = ns["peak_shift"] / f_max

    centroid = spectral_centroid_per_frame(S_stft, f_stft)         # (n_time, D)
    centroid_drift = fit_linear_slope(centroid, axis=0) / (f_max + 1e-30)

    return {
        "shape":               {"N": N, "L": L, "D": D},
        "n_bins":              n_bins,
        "P_per_unit":          P_per_unit,
        "P_per_unit_normed":   P_per_unit_normed,
        "H_per_unit":          H_per_unit,
        "F_per_unit":          F_per_unit,
        "peak_counts":         peak_counts,
        "max_bin":             max_bin,
        "max_share":           max_share,
        "drift_hilbert":       drift_hilbert,
        "drift_hilbert_rel":   drift_hilbert_rel,
        "carrier":             carrier,
        "f_stft":              f_stft,
        "t_stft":              t_stft,
        "S_stft":              S_stft,
        "early_peak":          ns["early_peak"],
        "late_peak":            ns["late_peak"],
        "peak_shift":          ns["peak_shift"],
        "peak_shift_rel":      peak_shift_rel,
        "shape_distance":      ns["shape_distance"],
        "log_amplitude_growth": ns["log_amplitude_growth"],
        "centroid":            centroid,
        "centroid_drift_rel":  centroid_drift,
    }


def scenario_indicators(m):
    """Six indicator metrics. Descriptive, not exclusive.

    spectral_concentration   1 - <H/log(n_bins)>     (1 if spiky, 0 if uniform)
    spectral_peakiness       1 - <flatness>          (1 if tonal, 0 if white-noise)
    dominant_bin_share       <max bin share>         (in [0, 1])
    FM_drift                 |<Hilbert d omega/dl>| / carrier
    spectral_shape_NS        <cosine distance between early-half and late-half spectra>
                             (high if spectrum SHAPE changes across depth)
    amplitude_NS             |<log10(late_total / early_total)>|  (separate axis)
                             (high if amplitude grows / shrinks across depth, even
                              with constant spectral shape)
    """
    return {
        "spectral_concentration": float(1 - m["H_per_unit"].mean()),
        "spectral_peakiness":     float(1 - m["F_per_unit"].mean()),
        "dominant_bin_share":     float(m["max_share"].mean()),
        "FM_drift":               float(np.abs(m["drift_hilbert_rel"]).mean()),
        "spectral_shape_NS":      float(m["shape_distance"].mean()),
        "amplitude_NS":           float(np.abs(m["log_amplitude_growth"]).mean()),
    }


def scenario_fit_scores(ind):
    """Combine the indicators into a fit score for each scenario.

    These are heuristic combinations, not exclusive probabilities. Multiple
    can be high if a model has mixed structure. Scores in [0, 1].

    multi_tone:
        peakiness AND concentration AND not(drift) AND not(shape NS)
        (multiple discrete peaks, stationary spectrum)
    FM:
        peakiness AND drift
    broadband_plus_osc:
        peakiness LOW but dominant_bin_share HIGH
    multi_component_nonstationary:
        peakiness AND spectral_shape_NS
        (peaks exist but the spectrum shape changes across depth;
         amplitude growth alone does NOT count as multi-component)
    """
    pk = ind["spectral_peakiness"]
    co = ind["spectral_concentration"]
    db = ind["dominant_bin_share"]
    dr = min(ind["FM_drift"], 1.0)
    sns = min(ind["spectral_shape_NS"], 1.0)
    flat = 1 - pk

    return {
        "multi_tone":                    pk * co * (1 - dr) * (1 - sns),
        "FM":                            pk * dr,
        "broadband_plus_osc":            flat * db,
        "multi_component_nonstationary": pk * sns,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overview(results, out_path, position):
    labels = list(results.keys())
    n_models = len(labels)
    n_cols = max(n_models, 3)

    fig = plt.figure(figsize=(5 * n_cols, 14))
    gs = gridspec.GridSpec(3, n_cols, figure=fig, hspace=0.40, wspace=0.30)

    # ----- Row 0: FFT spectra per model -----
    for i, label in enumerate(labels):
        ax = fig.add_subplot(gs[0, i])
        m = results[label]
        n_bins = m["n_bins"]
        bins = np.arange(n_bins)
        P = m["P_per_unit_normed"]
        D = P.shape[1]
        for u in range(D):
            ax.semilogy(
                bins, P[:, u] + 1e-12,
                color=PALETTE[i], alpha=0.10, linewidth=0.6,
            )
        med = np.median(P, axis=1)
        p25 = np.percentile(P, 25, axis=1)
        p75 = np.percentile(P, 75, axis=1)
        ax.fill_between(bins, p25 + 1e-12, p75 + 1e-12,
                        color=PALETTE[i], alpha=0.25, label="25-75 pct")
        ax.semilogy(bins, med + 1e-12,
                    "o-", color=PALETTE[i], linewidth=2.0, markersize=4,
                    label="median")
        ax.set_xlabel("rFFT bin")
        ax.set_ylabel("normalized power (log)")
        ax.set_title(f"(a) FFT spectrum: {label}", fontsize=11)
        ax.set_ylim(1e-4, 1)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper right")

    # ----- Row 1: STFT mean (over units) per model, log power -----
    # Find common color scale across models for fair visual comparison.
    log_S_all = []
    for label in labels:
        S = results[label]["S_stft"].mean(axis=2)
        log_S_all.append(np.log10(S + 1e-30))
    vmin = min(arr.min() for arr in log_S_all)
    vmax = max(arr.max() for arr in log_S_all)

    for i, label in enumerate(labels):
        ax = fig.add_subplot(gs[1, i])
        m = results[label]
        log_S = log_S_all[i]
        im = ax.pcolormesh(
            m["t_stft"], m["f_stft"], log_S,
            cmap="viridis", shading="auto", vmin=vmin, vmax=vmax,
        )
        ax.set_xlabel("time frame center (sublayer)")
        ax.set_ylabel("frequency (cycles/sublayer)")
        ax.set_title(f"(b) STFT mean log10 power: {label}", fontsize=11)
        plt.colorbar(im, ax=ax, label="log10 power")

    # ----- Row 2: histograms -----
    ax = fig.add_subplot(gs[2, 0])
    for i, label in enumerate(labels):
        H = results[label]["H_per_unit"]
        ax.hist(H, bins=20, range=(0, 1), alpha=0.55,
                color=PALETTE[i], label=label)
    ax.set_xlabel(r"$H / \log(n_{\mathrm{bins}})$")
    ax.set_ylabel("number of units")
    ax.set_title("(c) Spectral entropy")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    for i, label in enumerate(labels):
        Fl = results[label]["F_per_unit"]
        ax.hist(Fl, bins=20, range=(0, 1), alpha=0.55,
                color=PALETTE[i], label=label)
    ax.set_xlabel("spectral flatness")
    ax.set_ylabel("number of units")
    ax.set_title("(d) Spectral flatness (0 = tone, 1 = white noise)")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 2])
    for i, label in enumerate(labels):
        d = results[label]["drift_hilbert_rel"]
        ax.hist(d, bins=20, alpha=0.55, color=PALETTE[i], label=label)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"Hilbert $\mathrm{d}\omega/\mathrm{d}\ell$ / carrier")
    ax.set_ylabel("number of units")
    ax.set_title("(e) Frequency drift (FM indicator)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(
        f"Spectral diagnostics, position {position}",
        fontsize=14, y=0.995,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_spectrogram_examples(metrics, label, out_path):
    f = metrics["f_stft"]
    t = metrics["t_stft"]
    S = metrics["S_stft"]
    D = S.shape[2]
    energy = S.sum(axis=(0, 1))
    order = np.argsort(energy)
    chosen = [
        order[-1], order[-2],
        order[D // 2 - 1], order[D // 2],
        order[0], order[1],
    ]
    chosen_labels = [
        "high energy", "high energy",
        "median energy", "median energy",
        "low energy", "low energy",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    log_S_chosen = [np.log10(S[:, :, u] + 1e-30) for u in chosen]
    vmin = min(a.min() for a in log_S_chosen)
    vmax = max(a.max() for a in log_S_chosen)
    for ax, u, lbl, log_S in zip(axes.flatten(), chosen, chosen_labels, log_S_chosen):
        im = ax.pcolormesh(t, f, log_S, cmap="viridis", shading="auto",
                           vmin=vmin, vmax=vmax)
        ax.set_title(f"unit {u} ({lbl})", fontsize=10)
        ax.set_xlabel("time frame")
        ax.set_ylabel("frequency")
        plt.colorbar(im, ax=ax)
    fig.suptitle(f"STFT examples: {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_fit(results, out_path):
    labels = list(results.keys())
    scenarios = [
        "multi_tone", "FM",
        "broadband_plus_osc", "multi_component_nonstationary",
    ]
    n_models = len(labels)
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(scenarios))

    fig, (ax_ind, ax_fit) = plt.subplots(1, 2, figsize=(16, 6))

    indicator_keys = [
        "spectral_concentration", "spectral_peakiness",
        "dominant_bin_share", "FM_drift",
        "spectral_shape_NS", "amplitude_NS",
    ]
    x_ind = np.arange(len(indicator_keys))
    for i, label in enumerate(labels):
        ind = scenario_indicators(results[label])
        vals = [ind[k] for k in indicator_keys]
        ax_ind.bar(
            x_ind + (i - (n_models - 1) / 2) * width, vals,
            width=width, color=PALETTE[i], label=label, alpha=0.85,
        )
    ax_ind.set_xticks(x_ind)
    ax_ind.set_xticklabels(
        [k.replace("_", "\n") for k in indicator_keys], fontsize=9,
    )
    ax_ind.set_ylabel("indicator value")
    ax_ind.set_title("Five spectral indicators")
    ax_ind.grid(alpha=0.3, axis="y")
    ax_ind.legend()
    ax_ind.set_ylim(0, 1.05)

    for i, label in enumerate(labels):
        ind = scenario_indicators(results[label])
        fit = scenario_fit_scores(ind)
        vals = [fit[s] for s in scenarios]
        ax_fit.bar(
            x + (i - (n_models - 1) / 2) * width, vals,
            width=width, color=PALETTE[i], label=label, alpha=0.85,
        )
    ax_fit.set_xticks(x)
    ax_fit.set_xticklabels(
        [s.replace("_", "\n") for s in scenarios], fontsize=9,
    )
    ax_fit.set_ylabel("scenario fit score")
    ax_fit.set_title("Scenario fit scores (descriptive, not mutually exclusive)")
    ax_fit.grid(alpha=0.3, axis="y")
    ax_fit.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def build_summary(results, position):
    summary = {
        "position": position,
        "indicators_explained": {
            "spectral_concentration":
                "1 - mean(H / log(n_bins-1)) per unit, computed on bins 1..end. "
                "1 if spectrum is spiky, 0 if uniform.",
            "spectral_peakiness":
                "1 - mean(spectral flatness) per unit, computed on bins 1..end. "
                "1 if tonal, 0 if white-noise.",
            "dominant_bin_share":
                "Mean over units of the fraction of energy in the largest FFT bin.",
            "FM_drift":
                "|mean Hilbert d omega / d l| over interior l, in carrier units (carrier = 2*pi/L).",
            "spectral_shape_NS":
                "Mean cosine distance between early-half and late-half mean STFT spectra. "
                "Captures changes in spectral SHAPE across depth.",
            "amplitude_NS":
                "Mean |log10(late total power / early total power)|. "
                "Captures amplitude growth or decay across depth, regardless of spectral shape.",
        },
        "per_model": {},
    }
    for label, m in results.items():
        ind = scenario_indicators(m)
        fit = scenario_fit_scores(ind)
        summary["per_model"][label] = {
            "shape":               m["shape"],
            "n_fft_bins":          int(m["n_bins"]),
            "carrier_freq":        float(m["carrier"]),
            "spectral_entropy": {
                "mean":   float(m["H_per_unit"].mean()),
                "median": float(np.median(m["H_per_unit"])),
                "std":    float(m["H_per_unit"].std()),
            },
            "spectral_flatness": {
                "mean":   float(m["F_per_unit"].mean()),
                "median": float(np.median(m["F_per_unit"])),
                "std":    float(m["F_per_unit"].std()),
            },
            "peak_counts": {
                "mean":   float(m["peak_counts"].mean()),
                "median": float(np.median(m["peak_counts"])),
                "histogram_0_to_5": [
                    int(np.sum(m["peak_counts"] == k)) for k in range(6)
                ],
            },
            "dominant_bin_distribution": [
                int(np.sum(m["max_bin"] == k))
                for k in range(int(m["n_bins"]))
            ],
            "max_share_per_unit": {
                "mean":   float(m["max_share"].mean()),
                "median": float(np.median(m["max_share"])),
            },
            "frequency_drift_hilbert": {
                "abs_mean":          float(np.abs(m["drift_hilbert"]).mean()),
                "abs_mean_relative": float(np.abs(m["drift_hilbert_rel"]).mean()),
            },
            "stft_peak_shift": {
                "mean":          float(m["peak_shift"].mean()),
                "mean_relative": float(m["peak_shift_rel"].mean()),
            },
            "stft_shape_distance": {
                "mean":   float(m["shape_distance"].mean()),
                "median": float(np.median(m["shape_distance"])),
            },
            "stft_log_amplitude_growth": {
                "mean":     float(m["log_amplitude_growth"].mean()),
                "abs_mean": float(np.abs(m["log_amplitude_growth"]).mean()),
                "median":   float(np.median(m["log_amplitude_growth"])),
            },
            "stft_centroid_drift_relative": {
                "abs_mean": float(np.abs(m["centroid_drift_rel"]).mean()),
            },
            "indicators":     ind,
            "scenario_fit":   fit,
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz", type=str, nargs="+", required=True,
        help="One or more analytic_signal.npz files (output of phase_step0).",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="Display labels per --npz file (default: derive from filenames).",
    )
    parser.add_argument("--position",  type=int, default=5)
    parser.add_argument("--out-dir",   type=str, required=True)
    parser.add_argument("--nperseg",   type=int, default=DEFAULT_NPERSEG)
    parser.add_argument("--noverlap",  type=int, default=DEFAULT_NOVERLAP)
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
        print(f"  {label:<20} <- {path}")
        data = load_position_data(path, args.position)
        m = collect_metrics(data, args.nperseg, args.noverlap)
        results[label] = m
        print(f"    shape={m['shape']}, n_fft_bins={m['n_bins']}, "
              f"n_stft_freq={len(m['f_stft'])}, n_stft_time={len(m['t_stft'])}")

    print(f"\nMaking plots ...")
    plot_overview(results, out_dir / "spectral_overview.png", args.position)
    print(f"  spectral_overview.png")

    for label in args.labels:
        out = out_dir / f"spectrogram_examples_{label}.png"
        plot_spectrogram_examples(results[label], label, out)
        print(f"  {out.name}")

    plot_scenario_fit(results, out_dir / "scenario_fit.png")
    print(f"  scenario_fit.png")

    summary = build_summary(results, args.position)
    with open(out_dir / "spectral_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  spectral_summary.json")

    # ------------- Console report -------------
    print(f"\n--- Indicators (mean over units) ---")
    keys = ["spectral_concentration", "spectral_peakiness",
            "dominant_bin_share", "FM_drift",
            "spectral_shape_NS", "amplitude_NS"]
    print(f"{'model':<22} " + " ".join(f"{k[:10]:>11}" for k in keys))
    for label in args.labels:
        ind = summary["per_model"][label]["indicators"]
        print(f"{label:<22} " +
              " ".join(f"{ind[k]:>11.3f}" for k in keys))

    print(f"\n--- Scenario fit scores ---")
    scenarios = [
        "multi_tone", "FM",
        "broadband_plus_osc", "multi_component_nonstationary",
    ]
    print(f"{'model':<22} " + " ".join(f"{s[:14]:>16}" for s in scenarios))
    for label in args.labels:
        fit = summary["per_model"][label]["scenario_fit"]
        print(f"{label:<22} " +
              " ".join(f"{fit[s]:>16.3f}" for s in scenarios))

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
