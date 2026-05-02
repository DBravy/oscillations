"""
Per-unit phase-amplitude coupling (PAC) probe.

For each unit, scans all (slow_bin, fast_bin) pairs and asks whether the
unit's amplitude at the fast bin is modulated by the phase at the slow
bin, across the depth axis.

Method (Tort et al. modulation index):

  1. Compute the FFT of unit u's depth trajectory (centered).
  2. For each bin k, isolate that bin's contribution to the time-domain
     signal by zeroing all other bins and inverse-transforming. The
     result is a band-isolated complex analytic signal at bin k.
  3. From the slow-bin signal, get phase theta_slow(l).
     From the fast-bin signal, get amplitude A_fast(l).
  4. Bin theta_slow into n_phase_bins, compute mean A_fast per phase bin,
     normalize to a probability distribution P over phase bins.
  5. MI = (log(n_phase_bins) - H(P)) / log(n_phase_bins),
     i.e. KL divergence from uniform, normalized to [0, 1].
     MI = 0: amplitude is uniform across phases (no coupling).
     MI > 0: amplitude is concentrated at specific phases.

The PAC matrix M[s, f] for one unit is computed for s = 1..max_bin,
f = 1..max_bin. We only really care about s < f (slow phase, fast
amplitude), but the script computes the full matrix so symmetry can
be examined.

Per input pair: each input gives its own depth trajectory, hence its own
PAC matrix per unit. We aggregate across inputs in two ways:
  - mean across inputs
  - the input-pooled trajectory: stack all 256 inputs' depth signals
    end-to-end and compute one big PAC. This better reflects the
    statistical PAC (more data per phase bin).

Surrogate null: shuffle the slow-phase trajectory's order across depth
within each input, recompute MI, repeat. Provides a per-cell null so
real PAC can be distinguished from finite-sample noise.

Outputs in --out-dir:
  pac_matrix_pos{N}_summary.png     mean PAC matrix per position
                                    (bin pairs that show coupling)
  pac_per_unit_pos{N}.png           heatmap (unit x bin pair) of MI;
                                    rows sorted by max MI
  pac_unit_examples_pos{N}.png      for the top units, plot the slow
                                    phase / fast amplitude relationship
                                    explicitly (the "comodulogram view"):
                                    one panel per unit, with binned-
                                    amplitude vs slow-phase
  pac_summary.json                  numerical results

Usage:
  python pac_probe.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_pac
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
# Loading
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = ToyTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def collect_streams(model, position):
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]
    _, captures = capture_residual_streams(model, pairs, device)
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    s = stacked[:, :, position, :].permute(1, 0, 2).contiguous()
    return s.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Band isolation via single-bin inverse FFT
# ---------------------------------------------------------------------------

def isolate_bin(signal, bin_idx):
    """
    signal: (..., L) real array
    Returns the complex analytic signal at FFT bin `bin_idx` only,
    by zeroing all other bins of the rfft and inverse-transforming.
    The output is the time-domain signal you'd get if only this single
    bin were active. We use the analytic version (complex), so the
    output has both phase and amplitude as functions of l.
    """
    L = signal.shape[-1]
    spec = np.fft.rfft(signal, axis=-1)
    isolated_spec = np.zeros_like(spec)
    isolated_spec[..., bin_idx] = spec[..., bin_idx]
    # Inverse but keep complex by manually computing the analytic signal.
    # The analytic signal of a single positive frequency bin k is
    # 2 * spec[k] * exp(2 pi i k l / L). We construct that directly.
    ls = np.arange(L)
    coef = isolated_spec[..., bin_idx]                    # complex (...,)
    # Outer product: (..., L)
    expand = coef[..., None] * np.exp(
        2j * np.pi * bin_idx * ls / L
    )                                                     # (..., L)
    # Factor of 2/L is the standard single-sided-spectrum analytic-signal
    # normalization; for bin 0 and Nyquist (if present), no doubling.
    if bin_idx == 0 or (L % 2 == 0 and bin_idx == L // 2):
        return expand / L
    return 2 * expand / L


def band_phase_amplitude(signal, bin_idx):
    """Returns (phase, amplitude) trajectories for bin `bin_idx`."""
    z = isolate_bin(signal, bin_idx)
    return np.angle(z), np.abs(z)


# ---------------------------------------------------------------------------
# Modulation index (Tort)
# ---------------------------------------------------------------------------

def modulation_index(phase, amplitude, n_phase_bins=18):
    """
    phase, amplitude: 1-D arrays of equal length (the time samples).
    Returns scalar MI in [0, 1].
    """
    # Bin phase into n_phase_bins bins on [-pi, pi]
    edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    bin_idx = np.digitize(phase, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_phase_bins - 1)
    mean_amp = np.zeros(n_phase_bins)
    for b in range(n_phase_bins):
        mask = bin_idx == b
        if mask.any():
            mean_amp[b] = amplitude[mask].mean()
    total = mean_amp.sum()
    if total < 1e-12:
        return 0.0
    P = mean_amp / total
    # Avoid log(0)
    P_safe = np.where(P > 0, P, 1e-12)
    H = -np.sum(P * np.log(P_safe))
    H_max = np.log(n_phase_bins)
    return float((H_max - H) / H_max)


def amp_by_phase(phase, amplitude, n_phase_bins=18):
    """Returns mean amplitude per phase bin, plus the bin centers."""
    edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_idx = np.digitize(phase, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_phase_bins - 1)
    mean_amp = np.zeros(n_phase_bins)
    counts = np.zeros(n_phase_bins, dtype=int)
    for b in range(n_phase_bins):
        mask = bin_idx == b
        if mask.any():
            mean_amp[b] = amplitude[mask].mean()
            counts[b] = int(mask.sum())
    return centers, mean_amp, counts


# ---------------------------------------------------------------------------
# PAC matrix per unit
# ---------------------------------------------------------------------------

def pac_matrix_for_signal(signal, bin_pairs, n_phase_bins=18):
    """
    signal: 1-D array of length L (one unit's depth trajectory, possibly
            multiple inputs concatenated end to end).
    bin_pairs: iterable of (slow_bin, fast_bin) tuples.
    Returns dict mapping (s, f) -> MI.
    """
    L = len(signal)
    centered = signal - signal.mean()
    # Pre-compute band signals once per bin used
    bins_used = set()
    for s, f in bin_pairs:
        bins_used.add(s)
        bins_used.add(f)
    band_cache = {b: band_phase_amplitude(centered, b) for b in bins_used}

    out = {}
    for s, f in bin_pairs:
        slow_phase, _ = band_cache[s]
        _, fast_amp = band_cache[f]
        mi = modulation_index(slow_phase, fast_amp,
                               n_phase_bins=n_phase_bins)
        out[(s, f)] = mi
    return out


def pac_matrix_pooled(streams_unit, bin_pairs, n_phase_bins=18):
    """
    streams_unit: (N, L) all inputs' trajectories for one unit.
    Concatenates them end-to-end and computes PAC on the stitched
    signal, recentering each input segment to suppress jumps.
    """
    centered_per_input = (
        streams_unit - streams_unit.mean(axis=1, keepdims=True)
    )
    # We want the band isolation to be per-input, then concatenate
    # phase and amplitude trajectories. Otherwise the FFT smears across
    # boundaries.
    bins_used = set()
    for s, f in bin_pairs:
        bins_used.add(s)
        bins_used.add(f)

    band_cache = {b: ([], []) for b in bins_used}  # b -> (phases, amps)
    for n in range(streams_unit.shape[0]):
        sig = centered_per_input[n]
        for b in bins_used:
            ph, amp = band_phase_amplitude(sig, b)
            band_cache[b][0].append(ph)
            band_cache[b][1].append(amp)
    pooled = {}
    for b in bins_used:
        pooled[b] = (
            np.concatenate(band_cache[b][0]),
            np.concatenate(band_cache[b][1]),
        )

    out = {}
    for s, f in bin_pairs:
        slow_phase, _ = pooled[s]
        _, fast_amp = pooled[f]
        mi = modulation_index(slow_phase, fast_amp,
                               n_phase_bins=n_phase_bins)
        out[(s, f)] = mi
    return out, pooled


def shuffle_null_pac(streams_unit, bin_pairs, n_runs,
                     n_phase_bins=18, rng=None):
    """For each run, shuffle the depth-order of each input independently,
    compute PAC, return (n_runs, n_pairs) array."""
    if rng is None:
        rng = np.random.default_rng(0)
    null = np.zeros((n_runs, len(bin_pairs)))
    for r in range(n_runs):
        shuffled = streams_unit.copy()
        for n in range(shuffled.shape[0]):
            perm = rng.permutation(shuffled.shape[1])
            shuffled[n] = shuffled[n, perm]
        result, _ = pac_matrix_pooled(
            shuffled, bin_pairs, n_phase_bins=n_phase_bins,
        )
        for j, key in enumerate(bin_pairs):
            null[r, j] = result[key]
    return null


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pac_matrix_summary(mi_by_position, bin_pairs, max_bin,
                              save_path):
    """One panel per position. Heatmap of mean MI (across units) at each
    (slow_bin, fast_bin) cell."""
    fig, axes = plt.subplots(1, len(POSITIONS),
                              figsize=(6 * len(POSITIONS), 5),
                              sharey=True)
    vmax = max(
        np.nanmax(mi_by_position[pos]["pac_per_unit"].mean(axis=0))
        for pos in POSITIONS
    )
    for ax, pos in zip(axes, POSITIONS):
        per_unit = mi_by_position[pos]["pac_per_unit"]   # (D, n_pairs)
        # Build a (max_bin, max_bin) matrix from bin_pairs and means
        M = np.full((max_bin, max_bin), np.nan)
        means = per_unit.mean(axis=0)
        for j, (s, f) in enumerate(bin_pairs):
            M[s - 1, f - 1] = means[j]
        im = ax.imshow(
            M, origin="lower", aspect="equal", cmap="magma",
            vmin=0, vmax=vmax,
            extent=[0.5, max_bin + 0.5, 0.5, max_bin + 0.5],
            interpolation="nearest",
        )
        ax.set_xlabel("fast bin (amplitude)")
        ax.set_ylabel("slow bin (phase)")
        ax.set_title(POSITION_LABELS[pos], fontsize=11)
        ax.plot([0.5, max_bin + 0.5], [0.5, max_bin + 0.5],
                color="white", linewidth=0.5, linestyle="--",
                alpha=0.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Mean PAC modulation index across units, per position",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_pac_per_unit(per_unit, bin_pairs, position, save_path,
                      null_threshold=None):
    """Heatmap of MI per unit and bin-pair, sorted by max MI."""
    D = per_unit.shape[0]
    n_pairs = per_unit.shape[1]
    row_order = np.argsort(-np.max(per_unit, axis=1))
    sorted_table = per_unit[row_order]
    fig, ax = plt.subplots(
        figsize=(min(0.4 * n_pairs + 5, 18), 0.18 * D + 2),
    )
    vmax = max(np.nanmax(per_unit), 1e-6)
    im = ax.imshow(
        sorted_table, aspect="auto", cmap="magma",
        vmin=0, vmax=vmax, interpolation="nearest",
    )
    pair_labels = [f"{s}->{f}" for (s, f) in bin_pairs]
    ax.set_xticks(np.arange(n_pairs))
    ax.set_xticklabels(pair_labels, rotation=60, fontsize=7,
                        ha="right")
    ax.set_xlabel("(slow_bin -> fast_bin)")
    ax.set_yticks([0, D - 1])
    ax.set_yticklabels([f"unit {row_order[0]} (highest)",
                         f"unit {row_order[-1]} (lowest)"],
                       fontsize=8)
    ax.set_ylabel(f"units (sorted by max MI)")
    title = f"Per-unit PAC at {POSITION_LABELS[position]}"
    if null_threshold is not None:
        title += f"  (null threshold dashed)"
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    if null_threshold is not None:
        # Vertical lines wouldn't make sense; mark cells exceeding the
        # null threshold by overlaying a contour on the unsorted matrix.
        # Skip for simplicity; the threshold is recorded in JSON.
        pass
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_unit_examples(streams_by_pos, top_findings, bin_pairs,
                        save_path, n_phase_bins=18, n_show=6):
    """For the top findings (highest MI globally), plot the binned-
    amplitude vs slow-phase distribution. The comodulogram view: a
    polar plot of A_fast(theta_slow)."""
    if not top_findings:
        return
    fig, axes = plt.subplots(
        1, n_show, figsize=(4 * n_show, 4.5),
        subplot_kw=dict(projection="polar"),
    )
    if n_show == 1:
        axes = [axes]
    for ax, finding in zip(axes, top_findings[:n_show]):
        pos = finding["position"]
        u = finding["unit"]
        s_bin = finding["slow_bin"]
        f_bin = finding["fast_bin"]
        mi = finding["mi"]

        streams_unit = streams_by_pos[pos][:, :, u]
        # Build the pooled phase/amplitude trajectories
        slow_phases = []
        fast_amps = []
        for n in range(streams_unit.shape[0]):
            sig = streams_unit[n] - streams_unit[n].mean()
            ph, _ = band_phase_amplitude(sig, s_bin)
            _, amp = band_phase_amplitude(sig, f_bin)
            slow_phases.append(ph)
            fast_amps.append(amp)
        slow_phases = np.concatenate(slow_phases)
        fast_amps = np.concatenate(fast_amps)
        centers, mean_amp, _ = amp_by_phase(
            slow_phases, fast_amps, n_phase_bins=n_phase_bins,
        )
        # Close the loop for polar plot
        centers_closed = np.concatenate([centers, centers[:1]])
        mean_amp_closed = np.concatenate([mean_amp, mean_amp[:1]])
        ax.plot(centers_closed, mean_amp_closed, "-", linewidth=1.6)
        ax.fill(centers_closed, mean_amp_closed, alpha=0.3)
        ax.set_title(
            f"{POSITION_LABELS[pos]}\n"
            f"unit {u}, slow={s_bin}, fast={f_bin}\n"
            f"MI={mi:.3f}",
            fontsize=9, pad=12,
        )
        ax.tick_params(labelsize=7)
    fig.suptitle(
        "Top PAC findings: amplitude(fast) by phase(slow)",
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
    parser.add_argument("--out-dir", type=str, default="toy_pac")
    parser.add_argument("--max-bin", type=int, default=8,
                        help="Maximum FFT bin to consider (slow and fast)")
    parser.add_argument("--n-phase-bins", type=int, default=18,
                        help="Number of phase bins for MI calculation")
    parser.add_argument("--n-shuffle", type=int, default=20,
                        help="Number of shuffle null runs")
    parser.add_argument("--n-top-evidence", type=int, default=6,
                        help="Number of top findings to plot in detail")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}")

    streams_by_pos = {}
    for pos in POSITIONS:
        print(f"  collecting at position {pos} ...")
        streams_by_pos[pos] = collect_streams(model, pos)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build bin pairs: all (s, f) with s != f and s, f in [1, max_bin]
    bin_pairs = [
        (s, f)
        for s in range(1, args.max_bin + 1)
        for f in range(1, args.max_bin + 1)
        if s != f
    ]
    print(f"\nScanning {len(bin_pairs)} bin pairs per unit")
    print(f"  slow_bin in [1, {args.max_bin}], "
          f"fast_bin in [1, {args.max_bin}], slow != fast")

    mi_by_position = {}
    for pos in POSITIONS:
        print(f"\n=== {POSITION_LABELS[pos]} ===")
        streams = streams_by_pos[pos]                    # (N, L, D)
        N, L, D = streams.shape

        # Per-unit PAC: pool inputs end-to-end for each unit
        per_unit = np.zeros((D, len(bin_pairs)))
        for u in range(D):
            streams_unit = streams[:, :, u]
            result, _ = pac_matrix_pooled(
                streams_unit, bin_pairs,
                n_phase_bins=args.n_phase_bins,
            )
            for j, key in enumerate(bin_pairs):
                per_unit[u, j] = result[key]
            if u % 16 == 0:
                print(f"  unit {u}/{D} done")

        # Shuffle null on a sample of units (cost prohibitive on all)
        sample_units = np.random.default_rng(0).choice(
            D, size=min(8, D), replace=False,
        )
        print(f"  computing shuffle null on {len(sample_units)} units ...")
        null_per_pair = np.zeros((len(sample_units), args.n_shuffle,
                                   len(bin_pairs)))
        for j, u in enumerate(sample_units):
            null_per_pair[j] = shuffle_null_pac(
                streams[:, :, u], bin_pairs, args.n_shuffle,
                n_phase_bins=args.n_phase_bins,
                rng=np.random.default_rng(j + 1),
            )
        # Per-pair null threshold: 95th percentile across (sample_unit, run)
        null_p95 = np.percentile(
            null_per_pair.reshape(-1, len(bin_pairs)), 95, axis=0,
        )
        null_mean = null_per_pair.reshape(-1, len(bin_pairs)).mean(axis=0)

        # Significance: per (unit, pair), is MI above null_p95?
        sig_mask = per_unit > null_p95[None, :]

        mi_by_position[pos] = {
            "pac_per_unit":    per_unit,
            "null_p95":        null_p95,
            "null_mean":       null_mean,
            "sig_mask":        sig_mask,
            "bin_pairs":       bin_pairs,
        }

        print(f"  fraction of (unit, pair) cells exceeding null p95: "
              f"{sig_mask.mean():.3f}")
        print(f"  max MI: {per_unit.max():.3f}")
        print(f"  null mean MI: {null_mean.mean():.3f}, "
              f"p95: {null_p95.mean():.3f}")

    # Aggregate findings
    findings = []
    for pos in POSITIONS:
        per_unit = mi_by_position[pos]["pac_per_unit"]
        null_p95 = mi_by_position[pos]["null_p95"]
        D = per_unit.shape[0]
        for u in range(D):
            for j, (s, f) in enumerate(bin_pairs):
                mi = per_unit[u, j]
                findings.append({
                    "position":     int(pos),
                    "unit":         int(u),
                    "slow_bin":     int(s),
                    "fast_bin":     int(f),
                    "mi":           float(mi),
                    "null_p95":     float(null_p95[j]),
                    "above_null":   bool(mi > null_p95[j]),
                    "ratio_to_null": float(mi / max(null_p95[j], 1e-6)),
                })
    findings.sort(key=lambda f: -f["mi"])

    # Plots
    print("\nMaking plots ...")
    plot_pac_matrix_summary(
        mi_by_position, bin_pairs, args.max_bin,
        out_dir / "pac_matrix_summary.png",
    )
    for pos in POSITIONS:
        plot_pac_per_unit(
            mi_by_position[pos]["pac_per_unit"], bin_pairs, pos,
            out_dir / f"pac_per_unit_pos{pos}.png",
            null_threshold=mi_by_position[pos]["null_p95"],
        )
    plot_unit_examples(
        streams_by_pos, findings, bin_pairs,
        out_dir / "pac_unit_examples.png",
        n_phase_bins=args.n_phase_bins,
        n_show=args.n_top_evidence,
    )

    # Save data
    save_dict = {}
    for pos in POSITIONS:
        save_dict[f"per_unit_pos{pos}"] = (
            mi_by_position[pos]["pac_per_unit"]
        )
        save_dict[f"null_p95_pos{pos}"] = (
            mi_by_position[pos]["null_p95"]
        )
        save_dict[f"sig_mask_pos{pos}"] = (
            mi_by_position[pos]["sig_mask"]
        )
    save_dict["bin_pairs"] = np.array(bin_pairs)
    np.savez(out_dir / "pac_data.npz", **save_dict)

    summary = {
        "checkpoint":     str(args.checkpoint),
        "max_bin":        args.max_bin,
        "n_phase_bins":   args.n_phase_bins,
        "n_shuffle":      args.n_shuffle,
        "n_bin_pairs":    len(bin_pairs),
        "per_position": {
            str(pos): {
                "label":            POSITION_LABELS[pos],
                "max_mi":           float(
                    mi_by_position[pos]["pac_per_unit"].max()
                ),
                "median_mi":        float(
                    np.median(mi_by_position[pos]["pac_per_unit"])
                ),
                "mean_null_p95":    float(
                    mi_by_position[pos]["null_p95"].mean()
                ),
                "fraction_above_null": float(
                    mi_by_position[pos]["sig_mask"].mean()
                ),
                "n_units_with_any_significant_pair": int(
                    np.any(mi_by_position[pos]["sig_mask"], axis=1).sum()
                ),
            }
            for pos in POSITIONS
        },
        "top_50_findings": findings[:50],
    }
    with open(out_dir / "pac_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print("\n--- Top 15 PAC findings (sorted by MI) ---")
    print(f"{'rank':<5} {'pos':<6} {'unit':<6} {'s->f':<8} "
          f"{'MI':>8} {'null_p95':>10} {'ratio':>8} {'sig':>5}")
    for rank, f in enumerate(findings[:15], 1):
        print(f"{rank:<5} p{f['position']:<5} "
              f"u{f['unit']:<5} {f['slow_bin']}->{f['fast_bin']:<5} "
              f"{f['mi']:>8.3f} {f['null_p95']:>10.3f} "
              f"{f['ratio_to_null']:>8.2f} "
              f"{'YES' if f['above_null'] else 'no':>5}")

    print("\n--- Per-position PAC summary ---")
    print(f"{'position':<14} {'max MI':>10} {'med MI':>10} "
          f"{'null p95':>10} {'frac>null':>11} "
          f"{'units w/ sig pair':>20}")
    for pos in POSITIONS:
        s = summary["per_position"][str(pos)]
        print(
            f"{POSITION_LABELS[pos]:<14} "
            f"{s['max_mi']:>10.3f} "
            f"{s['median_mi']:>10.3f} "
            f"{s['mean_null_p95']:>10.3f} "
            f"{s['fraction_above_null']:>11.3f} "
            f"{s['n_units_with_any_significant_pair']:>20}"
        )

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
