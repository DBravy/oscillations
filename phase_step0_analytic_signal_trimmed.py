"""
Step 0 of higher-order phase analysis: extract clean instantaneous
phase theta_j(l) and instantaneous amplitude r_j(l) for each unit j
at each sublayer l, using the discrete Hilbert transform.

For each input pair and token position, the depth trajectory of unit u
is x_u(l) for l = 0..L-1. We center it along depth (subtract the mean
over l per (input, unit), or optionally remove a linear trend) and form
the analytic signal

    z_u(l) = x_u(l) + i * H[x_u](l)        via scipy.signal.hilbert(axis=1)

From z we read off, per (input, layer, unit):

    r_u(l)     = |z_u(l)|              instantaneous amplitude
    theta_u(l) = arg z_u(l)            wrapped phase in (-pi, pi]
    phi_u(l)   = unwrap(theta_u(l))    cumulative phase along depth
    omega_u(l) = d phi_u / dl          instantaneous frequency (rad / sublayer)

These are the building blocks for higher-order phase analysis: phase-
locking values, cross-frequency coupling, Kuramoto-style order
parameters, etc.

The narrowband assumption (FFT energy concentrated in bin 1 along the
depth axis) is a precondition for the analytic signal to have a clean
interpretation. We check it explicitly with FFT energy concentration
and instantaneous-frequency stability diagnostics. We also report
per-unit unwrapped-phase monotonicity, restricted to trajectories
whose amplitude does not collapse to near zero anywhere along depth
(phase is ill-defined when |z| ~ 0).

Note on edge effects: scipy.signal.hilbert is FFT-based and assumes
the signal is periodic. With short depth trajectories, the first and
last few values of phi and omega can be biased. This version adds
--trim-sublayers so you can remove boundary sublayers *before* the
Hilbert transform is computed. We still report mean |z| and mean omega
over an "interior" window (excludes first / last 2 samples) alongside
the global statistics.

Outputs in --out-dir:
  analytic_signal.npz                   {Z, r, theta, phi, omega} per position,
                                        plus original streams and FFT energies
  fft_energy_per_bin.png                bin-wise energy fraction (narrowband test)
  amplitude_envelopes.png               r_u(l) per unit, mean over inputs
  phase_unwrapped.png                   phi_u(l) per unit vs 2*pi*l/L reference
  instantaneous_frequency.png           omega histogram with carrier reference
  polar_sample_units.png                z(l) traces in polar form for top units
  hilbert_step0_summary.json            numerical summary

Usage:
  python phase_step0_analytic_signal.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_phase_step0_trained

  python phase_step0_analytic_signal.py \
      --checkpoint toy_transformer_run/model_random_init.pt \
      --out-dir toy_phase_step0_random

  python phase_step0_analytic_signal.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_phase_step0_trained_pos8 \
      --positions 5 6 7 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from toy_transformer_addition import (
    ToyTransformer,
    ModelConfig,
    capture_residual_streams,
)


SEED = 0

POSITION_LABELS = {
    5: 'pos 5 ("=")',
    6: "pos 6 (c2)",
    7: "pos 7 (c1)",
    8: "pos 8 (c0)",
}
POSITION_COLORS = {5: "C0", 6: "C1", 7: "C2", 8: "C3"}

# Edge-effect window: ignore this many samples on each end when computing
# "interior" diagnostics. With L=17, default 2 leaves an interior of 13.
EDGE_PAD = 2

# Floor for "well-defined" phase: a (input, unit) trajectory's phase is
# considered well-defined if its minimum amplitude along depth is at least
# this fraction of its maximum amplitude. Below this, unwrap() produces
# noisy phase jumps.
AMP_FLOOR_FRAC = 0.05


# ---------------------------------------------------------------------------
# Stream collection (matches sl_step1_radius_dynamics.py conventions)
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = ToyTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def collect_streams(model, positions, batch_size=64):
    """One forward pass over all 256 (a, b) pairs; cumulative streams
    at every sublayer at the requested positions.

    Returns:
      streams_by_pos: dict of pos -> (N, L, D) float32
      sublayer_meta:  list of (kind, layer_idx) tuples, length L
    """
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]
    _, captures = capture_residual_streams(
        model, pairs, device, batch_size=batch_size
    )
    sublayer_meta = [(c["kind"], c["layer"]) for c in captures]
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    out = {}
    for pos in positions:
        s = stacked[:, :, pos, :].permute(1, 0, 2).contiguous()
        out[pos] = s.numpy().astype(np.float32)
    return out, sublayer_meta


def trim_streams_by_depth(streams_by_pos, sublayer_meta, trim):
    """Drop boundary sublayers before any depth-frequency/Hilbert analysis.

    trim=0 keeps the original stream. trim=k keeps indices k : L-k.
    This is intentionally applied before centering, FFT, Hilbert, PLV, or
    any other nonlocal depth-axis operation so boundary samples cannot leak
    into the analytic signal.
    """
    if trim < 0:
        raise ValueError("--trim-sublayers must be >= 0")

    any_pos = next(iter(streams_by_pos))
    L = streams_by_pos[any_pos].shape[1]
    if 2 * trim >= L:
        raise ValueError(
            f"--trim-sublayers={trim} removes all/too many sublayers "
            f"for L={L}. Need 2*trim < L."
        )

    if trim == 0:
        used_indices = list(range(L))
        return streams_by_pos, sublayer_meta, used_indices

    used_indices = list(range(trim, L - trim))
    trimmed = {
        pos: arr[:, trim:L - trim, :].copy()
        for pos, arr in streams_by_pos.items()
    }
    trimmed_meta = [sublayer_meta[i] for i in used_indices]
    return trimmed, trimmed_meta, used_indices



# ---------------------------------------------------------------------------
# Analytic-signal extraction
# ---------------------------------------------------------------------------

def detrend_along_depth(streams):
    """Per (input, unit) least-squares linear detrend along the depth axis.

    streams: (N, L, D) float
    Returns:
      x:         (N, L, D) detrended trajectory
      slope:     (N, D)
      intercept: (N, D)
    """
    L = streams.shape[1]
    l = np.arange(L, dtype=streams.dtype)
    mean_l = l.mean()
    centered_l = l - mean_l
    var_l = (centered_l ** 2).sum()

    slope_num = (centered_l[None, :, None] * streams).sum(axis=1)  # (N, D)
    slope = slope_num / var_l
    intercept = streams.mean(axis=1) - slope * mean_l               # (N, D)

    trend = (
        slope[:, None, :] * l[None, :, None]
        + intercept[:, None, :]
    )
    return streams - trend, slope, intercept


def analytic_signal(streams, mode="center"):
    """Compute the analytic signal along the depth axis.

    streams: (N, L, D) float
    mode: "center" subtracts the depth mean; "detrend" removes a per-
          (input, unit) linear trend. Hilbert assumes a zero-mean
          (and ideally trend-free) signal, so one of these is required.

    Returns dict with:
      x_used  (N, L, D) float    centered or detrended input to Hilbert
      Z       (N, L, D) complex  analytic signal
      r       (N, L, D) float    |Z|
      theta   (N, L, D) float    arg(Z) in (-pi, pi]
      phi     (N, L, D) float    unwrap(theta) along depth
      omega   (N, L, D) float    np.gradient(phi) along depth
    """
    if mode == "center":
        x = streams - streams.mean(axis=1, keepdims=True)
    elif mode == "detrend":
        x, _, _ = detrend_along_depth(streams)
    else:
        raise ValueError(f"unknown preprocessing mode: {mode}")

    Z = hilbert(x, axis=1)
    r = np.abs(Z)
    theta = np.angle(Z)
    phi = np.unwrap(theta, axis=1)
    omega = np.gradient(phi, axis=1)

    return {
        "x_used": x.astype(np.float32),
        "Z":      Z.astype(np.complex64),
        "r":      r.astype(np.float32),
        "theta":  theta.astype(np.float32),
        "phi":    phi.astype(np.float32),
        "omega":  omega.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def fft_energy_per_bin(streams):
    """Per-trajectory energy fraction in each rFFT bin along depth.

    streams: (N, L, D)
    Returns:
      mean_frac:     (n_freq,) mean energy fraction over (N, D)
      per_unit_frac: (D, n_freq) mean over N
      per_input_frac:(N, n_freq) mean over D
    """
    centered = streams - streams.mean(axis=1, keepdims=True)
    F = np.fft.rfft(centered, axis=1)
    P = np.abs(F) ** 2                              # (N, n_freq, D)
    total = P.sum(axis=1, keepdims=True) + 1e-30
    frac = P / total
    mean_frac = frac.mean(axis=(0, 2))              # (n_freq,)
    per_unit_frac = frac.mean(axis=0).T             # (D, n_freq)
    per_input_frac = frac.mean(axis=2)              # (N, n_freq)
    return mean_frac, per_unit_frac, per_input_frac


def phase_quality(r, phi, amp_floor_frac=AMP_FLOOR_FRAC):
    """Report how "clean" the unwrapped phase is.

    r, phi: (N, L, D)
    A trajectory (n, u) is well-defined if min_l r[n, l, u] >=
    amp_floor_frac * max_l r[n, l, u]. Within those, we report
    monotonicity of phi and the median instantaneous frequency.
    """
    N, L, D = phi.shape
    d = np.diff(phi, axis=1)                        # (N, L-1, D)
    monotonic_inc = (d > 0).all(axis=1)             # (N, D)
    monotonic_dec = (d < 0).all(axis=1)
    monotonic = monotonic_inc | monotonic_dec

    rmin = r.min(axis=1)                            # (N, D)
    rmax = r.max(axis=1)
    well_defined = rmin >= amp_floor_frac * (rmax + 1e-12)

    if well_defined.any():
        mono_well = float(monotonic[well_defined].mean())
        d_mask = np.broadcast_to(well_defined[:, None, :], d.shape)
        omega_well = d[d_mask]
        median_omega_well = float(np.median(omega_well))
        mean_abs_omega_well = float(np.mean(np.abs(omega_well)))
    else:
        mono_well = float("nan")
        median_omega_well = float("nan")
        mean_abs_omega_well = float("nan")

    return {
        "frac_monotonic_overall":      float(monotonic.mean()),
        "frac_well_defined":           float(well_defined.mean()),
        "frac_monotonic_well_defined": mono_well,
        "median_omega_well_defined":   median_omega_well,
        "mean_abs_omega_well_defined": mean_abs_omega_well,
    }


def edge_vs_interior(arr, pad=EDGE_PAD):
    """Compare statistics of an (N, L, D) array on the edges vs the interior."""
    L = arr.shape[1]
    if L <= 2 * pad + 1:
        return {"interior_mean": float(arr.mean()), "edge_mean": float("nan")}
    interior = arr[:, pad:L - pad, :]
    edge = np.concatenate([arr[:, :pad, :], arr[:, L - pad:, :]], axis=1)
    return {
        "interior_mean": float(interior.mean()),
        "interior_std":  float(interior.std()),
        "edge_mean":     float(edge.mean()),
        "edge_std":      float(edge.std()),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_fft_energy(streams_by_pos, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for pos in sorted(streams_by_pos):
        mean_frac, _, _ = fft_energy_per_bin(streams_by_pos[pos])
        bins = np.arange(len(mean_frac))
        ax.plot(
            bins, mean_frac, "o-",
            color=POSITION_COLORS.get(pos, None),
            label=POSITION_LABELS.get(pos, f"pos {pos}"),
        )
    ax.set_xlabel("rFFT bin (depth-axis frequency)")
    ax.set_ylabel("mean energy fraction")
    ax.set_title(
        "Per-trajectory energy along depth\n"
        "(narrowband test: bin 1 should dominate)"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def _grid_axes(n, subplot_kw=None):
    fig, axes = plt.subplots(
        1, n, figsize=(6 * n, 5), sharey=False,
        subplot_kw=subplot_kw,
    )
    if n == 1:
        axes = [axes]
    return fig, axes


def plot_amplitude_envelopes(results_by_pos, save_path):
    pos_list = sorted(results_by_pos)
    fig, axes = _grid_axes(len(pos_list))
    for ax, pos in zip(axes, pos_list):
        r = results_by_pos[pos]["r"]
        L, D = r.shape[1], r.shape[2]
        per_unit_mean = r.mean(axis=0)              # (L, D)
        for u in range(D):
            ax.plot(
                np.arange(L), per_unit_mean[:, u],
                color=POSITION_COLORS.get(pos, "C0"),
                alpha=0.10, linewidth=0.6,
            )
        ax.plot(
            np.arange(L), per_unit_mean.mean(axis=1),
            color="black", linewidth=2.5, label="mean over units",
        )
        ax.set_xlabel("sublayer index l")
        ax.set_ylabel(r"$|z_u(l)|$ (instantaneous amplitude)")
        ax.set_title(POSITION_LABELS.get(pos, f"pos {pos}"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(r"Amplitude envelope $r_u(\ell)$ (mean over input pairs)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_phase_unwrapped(results_by_pos, save_path):
    pos_list = sorted(results_by_pos)
    fig, axes = _grid_axes(len(pos_list))
    for ax, pos in zip(axes, pos_list):
        phi = results_by_pos[pos]["phi"]
        L, D = phi.shape[1], phi.shape[2]
        # Align each (input, unit) trajectory to start at phi=0 so the
        # plot focuses on the rotation rate, not the wrap-arbitrary offset.
        phi_aligned = phi - phi[:, 0:1, :]
        per_unit_mean = phi_aligned.mean(axis=0)    # (L, D)
        for u in range(D):
            ax.plot(
                np.arange(L), per_unit_mean[:, u],
                color=POSITION_COLORS.get(pos, "C0"),
                alpha=0.10, linewidth=0.6,
            )
        ax.plot(
            np.arange(L), 2 * np.pi * np.arange(L) / L,
            "k--", linewidth=1.2,
            label=r"$2\pi \ell / L$ reference",
        )
        ax.plot(
            np.arange(L), -2 * np.pi * np.arange(L) / L,
            color="gray", linestyle="--", linewidth=1.0,
            label=r"$-2\pi \ell / L$ reference",
        )
        ax.set_xlabel("sublayer index l")
        ax.set_ylabel(r"$\phi_u(\ell) - \phi_u(0)$ (rad)")
        ax.set_title(POSITION_LABELS.get(pos, f"pos {pos}"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Unwrapped phase (per-unit mean across input pairs)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_instantaneous_frequency(results_by_pos, save_path):
    pos_list = sorted(results_by_pos)
    fig, axes = _grid_axes(len(pos_list))
    for ax, pos in zip(axes, pos_list):
        omega = results_by_pos[pos]["omega"]
        L = omega.shape[1]
        ax.hist(
            omega.flatten(), bins=80, density=True,
            color=POSITION_COLORS.get(pos, "C0"), alpha=0.7,
        )
        carrier = 2 * np.pi / L
        ax.axvline(
            carrier, color="black", linestyle="--", linewidth=1.5,
            label=fr"$+2\pi/L$ = {carrier:.2f}",
        )
        ax.axvline(
            -carrier, color="gray", linestyle="--", linewidth=1.0,
            label=fr"$-2\pi/L$",
        )
        ax.axvline(0, color="red", linestyle=":", linewidth=0.8)
        ax.set_xlabel(r"$\omega_u(\ell)$ (rad / sublayer)")
        ax.set_ylabel("density")
        ax.set_title(POSITION_LABELS.get(pos, f"pos {pos}"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Instantaneous frequency distribution (narrowband consistency)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_polar_sample(results_by_pos, sample_input=0, n_units=6, save_path=None):
    """Polar trace of z_u(l) over depth for a few high-amplitude units, per position."""
    pos_list = sorted(results_by_pos)
    fig, axes = plt.subplots(
        len(pos_list), n_units,
        figsize=(2.6 * n_units, 2.6 * len(pos_list)),
        subplot_kw={"projection": "polar"},
    )
    if len(pos_list) == 1:
        axes = axes[None, :]
    for row, pos in enumerate(pos_list):
        Z = results_by_pos[pos]["Z"][sample_input]      # (L, D)
        L, D = Z.shape
        amp = np.abs(Z).mean(axis=0)
        chosen = np.argsort(-amp)[:n_units]
        for col, u in enumerate(chosen):
            ax = axes[row, col]
            theta = np.angle(Z[:, u])
            r = np.abs(Z[:, u])
            ax.plot(
                theta, r, "-", linewidth=0.8, alpha=0.8,
                color=POSITION_COLORS.get(pos, "C0"),
            )
            ax.scatter(theta, r, c=np.arange(L), cmap="viridis", s=12)
            ax.set_title(
                f"{POSITION_LABELS.get(pos, f'pos {pos}')} u={u}",
                fontsize=8,
            )
            ax.tick_params(labelsize=6)
    fig.suptitle(
        f"Analytic signal in polar form (input pair index {sample_input})",
        fontsize=11,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_position(streams, results):
    r = results["r"]
    phi = results["phi"]
    omega = results["omega"]
    Z = results["Z"]
    N, L, D = r.shape

    mean_frac, per_unit_frac, _ = fft_energy_per_bin(streams)
    quality = phase_quality(r, phi)

    r_edge = edge_vs_interior(r)
    omega_edge = edge_vs_interior(omega)

    return {
        "shape":          {"N": int(N), "L": int(L), "D": int(D)},
        "fft_energy_concentration": {
            "bin_means": [float(x) for x in mean_frac],
            "bin1_share":     float(mean_frac[1]) if len(mean_frac) > 1 else None,
            "bins_1to3_share": float(mean_frac[1:4].sum()) if len(mean_frac) > 1 else None,
            "carrier_freq_2pi_over_L": float(2 * np.pi / L),
        },
        "amplitude": {
            "mean":    float(r.mean()),
            "median":  float(np.median(r)),
            "std":     float(r.std()),
            "per_unit_mean_min": float(r.mean(axis=(0, 1)).min()),
            "per_unit_mean_max": float(r.mean(axis=(0, 1)).max()),
            **{f"edge_vs_interior_{k}": v for k, v in r_edge.items()},
        },
        "phase_quality":     quality,
        "instantaneous_frequency": {
            "mean":     float(omega.mean()),
            "median":   float(np.median(omega)),
            "std":      float(omega.std()),
            "abs_mean": float(np.abs(omega).mean()),
            **{f"edge_vs_interior_{k}": v for k, v in omega_edge.items()},
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir",    type=str, default="toy_phase_step0")
    parser.add_argument(
        "--positions", type=int, nargs="+", default=[5, 6, 7],
        help="Token positions to analyze (default: 5 6 7).",
    )
    parser.add_argument(
        "--mode", type=str, default="center",
        choices=["center", "detrend"],
        help=("Preprocessing along depth before Hilbert. 'center' "
              "subtracts the depth mean; 'detrend' removes a linear "
              "trend per (input, unit). Default: center."),
    )
    parser.add_argument(
        "--polar-input", type=int, default=0,
        help="Input pair index to use for polar sample plot.",
    )
    parser.add_argument(
        "--trim-sublayers", type=int, default=0, metavar="K",
        help=("Drop the first K and last K captured sublayers before any "
              "FFT/Hilbert analysis. This is the edge-effect control; "
              "default: 0."),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Loading: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}")

    print(f"\nCollecting cumulative residual streams at positions "
          f"{args.positions} ...")
    streams_by_pos, sublayer_meta = collect_streams(model, args.positions)
    for pos in args.positions:
        print(f"  pos {pos}: shape {streams_by_pos[pos].shape}")
    print(f"  sublayers: {sublayer_meta}")

    original_sublayer_meta = list(sublayer_meta)
    original_n_sublayers = streams_by_pos[args.positions[0]].shape[1]
    streams_by_pos, sublayer_meta, used_sublayer_indices = trim_streams_by_depth(
        streams_by_pos, sublayer_meta, args.trim_sublayers
    )
    if args.trim_sublayers:
        print(
            f"\nTrimmed first/last {args.trim_sublayers} sublayers before analysis: "
            f"L {original_n_sublayers} -> {streams_by_pos[args.positions[0]].shape[1]}"
        )
        print(f"  retained original sublayer indices: {used_sublayer_indices}")
        print(f"  retained sublayers: {sublayer_meta}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nComputing analytic signal (mode='{args.mode}') ...")
    results_by_pos = {}
    for pos in args.positions:
        results_by_pos[pos] = analytic_signal(
            streams_by_pos[pos], mode=args.mode
        )
        r = results_by_pos[pos]["r"]
        phi = results_by_pos[pos]["phi"]
        print(f"  pos {pos}: |z| mean={r.mean():.4f} "
              f"(min unit-mean {r.mean(axis=(0,1)).min():.4f}, "
              f"max unit-mean {r.mean(axis=(0,1)).max():.4f}); "
              f"phi span (mean over inputs/units) "
              f"{(phi[:,-1,:] - phi[:,0,:]).mean():.3f} rad")

    # ------------- Plots -------------
    print("\nMaking plots ...")
    plot_fft_energy(
        streams_by_pos, out_dir / "fft_energy_per_bin.png",
    )
    plot_amplitude_envelopes(
        results_by_pos, out_dir / "amplitude_envelopes.png",
    )
    plot_phase_unwrapped(
        results_by_pos, out_dir / "phase_unwrapped.png",
    )
    plot_instantaneous_frequency(
        results_by_pos, out_dir / "instantaneous_frequency.png",
    )
    plot_polar_sample(
        results_by_pos,
        sample_input=args.polar_input,
        save_path=out_dir / "polar_sample_units.png",
    )

    # ------------- Save derived data for higher-order steps -------------
    print("\nSaving npz ...")
    npz_payload = {
        "used_sublayer_indices": np.asarray(used_sublayer_indices, dtype=np.int64),
    }
    for pos in args.positions:
        npz_payload[f"streams_pos{pos}"] = streams_by_pos[pos]
        for key in ["x_used", "Z", "r", "theta", "phi", "omega"]:
            npz_payload[f"{key}_pos{pos}"] = results_by_pos[pos][key]
        mean_frac, per_unit_frac, per_input_frac = fft_energy_per_bin(
            streams_by_pos[pos]
        )
        npz_payload[f"fft_energy_mean_frac_pos{pos}"] = mean_frac
        npz_payload[f"fft_energy_per_unit_frac_pos{pos}"] = per_unit_frac
        npz_payload[f"fft_energy_per_input_frac_pos{pos}"] = per_input_frac
    np.savez(out_dir / "analytic_signal.npz", **npz_payload)

    # ------------- Summary -------------
    summary = {
        "checkpoint":     str(args.checkpoint),
        "preprocessing":  args.mode,
        "positions":      list(args.positions),
        "n_sublayers":    int(streams_by_pos[args.positions[0]].shape[1]),
        "original_n_sublayers": int(original_n_sublayers),
        "trim_sublayers": int(args.trim_sublayers),
        "used_sublayer_indices": [int(i) for i in used_sublayer_indices],
        "d_model":        int(streams_by_pos[args.positions[0]].shape[2]),
        "n_input_pairs":  int(streams_by_pos[args.positions[0]].shape[0]),
        "edge_pad":       EDGE_PAD,
        "amp_floor_frac": AMP_FLOOR_FRAC,
        "sublayer_meta":  [
            {"kind": k, "layer": l} for (k, l) in sublayer_meta
        ],
        "original_sublayer_meta": [
            {"kind": k, "layer": l} for (k, l) in original_sublayer_meta
        ],
        "per_position": {
            str(pos): summarize_position(
                streams_by_pos[pos], results_by_pos[pos]
            )
            for pos in args.positions
        },
    }
    with open(out_dir / "hilbert_step0_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------- Console report -------------
    print("\n--- Step 0 summary ---")
    print(f"\nNarrowband check (rFFT bin 1 share of total energy along depth):")
    print(f"{'position':<14} {'bin1':>8} {'bins 1-3':>10} {'2pi/L':>8}")
    for pos in args.positions:
        s = summary["per_position"][str(pos)]
        f = s["fft_energy_concentration"]
        print(
            f"{POSITION_LABELS.get(pos, f'pos {pos}'):<14} "
            f"{f['bin1_share']:>8.3f} "
            f"{f['bins_1to3_share']:>10.3f} "
            f"{f['carrier_freq_2pi_over_L']:>8.3f}"
        )

    print(f"\nPhase quality "
          f"(well-defined: min |z| >= {AMP_FLOOR_FRAC} * max |z|):")
    print(f"{'position':<14} "
          f"{'frac WD':>10} {'mono | WD':>12} "
          f"{'med omega WD':>14} {'<|omega|> WD':>14}")
    for pos in args.positions:
        q = summary["per_position"][str(pos)]["phase_quality"]
        print(
            f"{POSITION_LABELS.get(pos, f'pos {pos}'):<14} "
            f"{q['frac_well_defined']:>10.3f} "
            f"{q['frac_monotonic_well_defined']:>12.3f} "
            f"{q['median_omega_well_defined']:>14.3f} "
            f"{q['mean_abs_omega_well_defined']:>14.3f}"
        )

    print(f"\nAmplitude edge vs interior (interior excludes first/last "
          f"{EDGE_PAD} sublayers):")
    print(f"{'position':<14} {'edge mean':>10} {'interior mean':>14}")
    for pos in args.positions:
        a = summary["per_position"][str(pos)]["amplitude"]
        print(
            f"{POSITION_LABELS.get(pos, f'pos {pos}'):<14} "
            f"{a['edge_vs_interior_edge_mean']:>10.4f} "
            f"{a['edge_vs_interior_interior_mean']:>14.4f}"
        )

    if args.trim_sublayers:
        print(
            f"\nNOTE: all saved streams/results use the trimmed depth axis; "
            f"original indices are saved as used_sublayer_indices."
        )

    print(f"\nOutputs in {out_dir.resolve()}")
    print("  fft_energy_per_bin.png")
    print("  amplitude_envelopes.png")
    print("  phase_unwrapped.png")
    print("  instantaneous_frequency.png")
    print("  polar_sample_units.png")
    print("  analytic_signal.npz   (Z, r, theta, phi, omega per position)")
    print("  hilbert_step0_summary.json")


if __name__ == "__main__":
    main()
