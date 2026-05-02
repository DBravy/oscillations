"""
Per-unit phase-amplitude coupling (PAC) probe for SmolLM2 360M.

Adapted from pac_probe.py for the toy transformer. Key differences:

  - Captures residual streams from SmolLM2 360M via layernorm input hooks
    (same approach as the earlier llama_start.py reference). Last-token
    position only.
  - WikiText-2 validation samples as the input distribution.
  - Phase-randomized surrogate null (Tort et al. style, IAAFT-flavored)
    rather than order shuffle. This preserves the per-bin amplitude
    envelope, including any depth-localized events like the radius
    explosion, but destroys phase relationships between bins. PAC
    computed against this null isolates genuine cross-bin phase
    coupling from amplitude-envelope artifacts.
  - 60+ sublayers gives enough depth that genuinely nested rhythms
    could be detected if they exist.

Method: Tort modulation index, with phase binned into n_phase_bins.
Pools across samples by computing PAC on the concatenation of
per-sample band-isolated trajectories.

Outputs in --out-dir:
  pac_matrix_smollm2.png        mean PAC matrix across units (slow x fast)
  pac_per_unit_smollm2.png      heatmap (unit x bin pair) of MI
  pac_unit_examples.png         polar plots of top findings
  pac_data.npz                  arrays
  pac_summary.json              numerical summary

Usage:
  python pac_probe_smollm2.py \
      --out-dir smollm2_pac \
      --n-samples 256 --seq-len 128 \
      --max-bin 12

Notes:
  - SmolLM2 360M has 32 transformer layers, so 64 sublayers + embedding
    = 65 capture points per sample.
  - The model is loaded in fp16 with float32 captures for FFT stability.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


# ---------------------------------------------------------------------------
# Loading and stream collection
# ---------------------------------------------------------------------------

def load_model():
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.float16
        if torch.cuda.is_available() or torch.backends.mps.is_available()
        else torch.float32
    )
    device_map = "auto" if (
        torch.cuda.is_available() or torch.backends.mps.is_available()
    ) else None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
        output_hidden_states=False,
    )
    model.eval()
    return model, tokenizer


def get_hook_targets(model):
    """Return the modules whose INPUTS we capture: pre-attn layernorm and
    pre-mlp layernorm of every transformer block. The input to each
    layernorm is the residual stream at that sublayer position.
    """
    blocks = model.model.layers
    targets = []
    for block in blocks:
        targets.append(block.input_layernorm)
        targets.append(block.post_attention_layernorm)
    return targets


def collect_streams(model, tokenizer, texts, seq_len):
    """Run forward on each text, capture the residual stream at each
    layernorm input, take the last token's vector. Returns array of
    shape (n_samples, n_sublayers, d_model) in float32."""
    targets = get_hook_targets(model)
    out = []
    for i, text in enumerate(texts):
        enc = tokenizer(
            text, return_tensors="pt",
            max_length=seq_len, truncation=True,
        )
        ids = enc["input_ids"].to(next(model.parameters()).device)
        if ids.shape[1] < 8:
            continue
        captures = []

        def make_hook(c):
            def fn(m, inp, out_):
                c.append(inp[0].detach())
            return fn

        hooks = [t.register_forward_hook(make_hook(captures))
                 for t in targets]
        try:
            with torch.no_grad():
                model(ids)
        finally:
            for h in hooks:
                h.remove()
        # captures: list of (1, T, D) tensors. Take last token of each.
        last = torch.stack(
            [c[0, -1, :].float() for c in captures], dim=0
        )                                          # (n_sublayers, D)
        out.append(last.cpu().numpy())
        if (i + 1) % 16 == 0:
            print(f"    {i + 1} / {len(texts)} samples ...")
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Band isolation, modulation index (same as toy version)
# ---------------------------------------------------------------------------

def isolate_bin_signal(signal, bin_idx):
    """signal: (..., L) real. Returns analytic complex signal at this
    bin only."""
    L = signal.shape[-1]
    spec = np.fft.rfft(signal, axis=-1)
    coef = spec[..., bin_idx]
    ls = np.arange(L)
    expand = coef[..., None] * np.exp(2j * np.pi * bin_idx * ls / L)
    if bin_idx == 0 or (L % 2 == 0 and bin_idx == L // 2):
        return expand / L
    return 2 * expand / L


def band_phase_amplitude(signal, bin_idx):
    z = isolate_bin_signal(signal, bin_idx)
    return np.angle(z), np.abs(z)


def modulation_index(phase, amplitude, n_phase_bins=18):
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
    P_safe = np.where(P > 0, P, 1e-12)
    H = -np.sum(P * np.log(P_safe))
    H_max = np.log(n_phase_bins)
    return float((H_max - H) / H_max)


def amp_by_phase(phase, amplitude, n_phase_bins=18):
    edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_idx = np.digitize(phase, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_phase_bins - 1)
    mean_amp = np.zeros(n_phase_bins)
    for b in range(n_phase_bins):
        mask = bin_idx == b
        if mask.any():
            mean_amp[b] = amplitude[mask].mean()
    return centers, mean_amp


# ---------------------------------------------------------------------------
# Phase-randomized null (preserves amplitude envelope per bin)
# ---------------------------------------------------------------------------

def phase_randomize_signal(signal, rng):
    """signal: (L,). Returns a real-valued signal with the same power
    spectrum but randomized phases on non-DC, non-Nyquist bins.
    Preserves the AMPLITUDE structure of every band (so the explosion
    is preserved per bin) but destroys phase relationships across bins."""
    L = len(signal)
    spec = np.fft.rfft(signal)
    n_pos = len(spec)
    new_spec = np.copy(spec)
    # Keep DC and (if even L) Nyquist real; randomize phases of others
    for k in range(1, n_pos):
        if L % 2 == 0 and k == n_pos - 1:
            continue
        mag = np.abs(spec[k])
        new_phase = rng.uniform(-np.pi, np.pi)
        new_spec[k] = mag * np.exp(1j * new_phase)
    return np.fft.irfft(new_spec, n=L)


def pac_for_signal_pooled(streams_unit, bin_pairs, n_phase_bins=18):
    """streams_unit: (N, L) trajectories for one unit, pre-centered or not.
    Returns dict pair -> MI."""
    centered = streams_unit - streams_unit.mean(axis=1, keepdims=True)
    bins_used = set()
    for s, f in bin_pairs:
        bins_used.add(s)
        bins_used.add(f)
    band = {b: ([], []) for b in bins_used}
    for n in range(centered.shape[0]):
        sig = centered[n]
        for b in bins_used:
            ph, amp = band_phase_amplitude(sig, b)
            band[b][0].append(ph)
            band[b][1].append(amp)
    pooled = {b: (np.concatenate(band[b][0]),
                  np.concatenate(band[b][1]))
              for b in bins_used}
    out = {}
    for s, f in bin_pairs:
        slow_phase, _ = pooled[s]
        _, fast_amp = pooled[f]
        out[(s, f)] = modulation_index(
            slow_phase, fast_amp, n_phase_bins=n_phase_bins,
        )
    return out


def phase_randomized_null(streams_unit, bin_pairs, n_runs,
                           n_phase_bins=18, rng=None):
    """For each run, phase-randomize each input's signal (preserving its
    per-bin amplitude structure but randomizing inter-bin phases),
    recompute PAC, return (n_runs, n_pairs)."""
    if rng is None:
        rng = np.random.default_rng(0)
    null = np.zeros((n_runs, len(bin_pairs)))
    for r in range(n_runs):
        randomized = np.zeros_like(streams_unit)
        for n in range(streams_unit.shape[0]):
            randomized[n] = phase_randomize_signal(
                streams_unit[n], rng
            )
        result = pac_for_signal_pooled(
            randomized, bin_pairs, n_phase_bins=n_phase_bins,
        )
        for j, key in enumerate(bin_pairs):
            null[r, j] = result[key]
    return null


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pac_matrix_summary(per_unit, bin_pairs, max_bin, save_path):
    means = per_unit.mean(axis=0)
    M = np.full((max_bin, max_bin), np.nan)
    for j, (s, f) in enumerate(bin_pairs):
        M[s - 1, f - 1] = means[j]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        M, origin="lower", aspect="equal", cmap="magma",
        extent=[0.5, max_bin + 0.5, 0.5, max_bin + 0.5],
        interpolation="nearest",
    )
    ax.set_xlabel("fast bin (amplitude)")
    ax.set_ylabel("slow bin (phase)")
    ax.set_title(
        "Mean PAC modulation index across units (SmolLM2 360M)",
        fontsize=11,
    )
    ax.plot([0.5, max_bin + 0.5], [0.5, max_bin + 0.5],
            color="white", linewidth=0.5, linestyle="--", alpha=0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_pac_per_unit(per_unit, bin_pairs, save_path):
    D, n_pairs = per_unit.shape
    row_order = np.argsort(-np.max(per_unit, axis=1))
    sorted_table = per_unit[row_order]
    fig, ax = plt.subplots(
        figsize=(min(0.3 * n_pairs + 5, 20), max(0.05 * D + 2, 8)),
    )
    vmax = max(per_unit.max(), 1e-6)
    im = ax.imshow(
        sorted_table, aspect="auto", cmap="magma",
        vmin=0, vmax=vmax, interpolation="nearest",
    )
    pair_labels = [f"{s}->{f}" for (s, f) in bin_pairs]
    ax.set_xticks(np.arange(n_pairs))
    ax.set_xticklabels(pair_labels, rotation=60, fontsize=6,
                        ha="right")
    ax.set_xlabel("(slow_bin -> fast_bin)")
    ax.set_yticks([0, D - 1])
    ax.set_yticklabels(
        [f"unit {row_order[0]}", f"unit {row_order[-1]}"],
        fontsize=8,
    )
    ax.set_ylabel("units (sorted by max MI)")
    ax.set_title("Per-unit PAC, SmolLM2 360M", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_unit_examples(streams, top_findings, n_show, n_phase_bins,
                        save_path):
    if not top_findings:
        return
    fig, axes = plt.subplots(
        1, n_show, figsize=(4 * n_show, 4.5),
        subplot_kw=dict(projection="polar"),
    )
    if n_show == 1:
        axes = [axes]
    for ax, finding in zip(axes, top_findings[:n_show]):
        u = finding["unit"]
        s_bin = finding["slow_bin"]
        f_bin = finding["fast_bin"]
        mi = finding["mi"]
        slow_phases = []
        fast_amps = []
        for n in range(streams.shape[0]):
            sig = streams[n, :, u] - streams[n, :, u].mean()
            ph, _ = band_phase_amplitude(sig, s_bin)
            _, amp = band_phase_amplitude(sig, f_bin)
            slow_phases.append(ph)
            fast_amps.append(amp)
        slow_phases = np.concatenate(slow_phases)
        fast_amps = np.concatenate(fast_amps)
        centers, mean_amp = amp_by_phase(
            slow_phases, fast_amps, n_phase_bins=n_phase_bins,
        )
        cc = np.concatenate([centers, centers[:1]])
        mc = np.concatenate([mean_amp, mean_amp[:1]])
        ax.plot(cc, mc, "-", linewidth=1.6)
        ax.fill(cc, mc, alpha=0.3)
        ax.set_title(
            f"unit {u}, slow={s_bin}, fast={f_bin}\n"
            f"MI={mi:.3f}, ratio_to_null={finding['ratio_to_null']:.1f}",
            fontsize=9, pad=12,
        )
        ax.tick_params(labelsize=7)
    fig.suptitle("Top PAC findings: amplitude(fast) by phase(slow)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="smollm2_pac")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-bin", type=int, default=12,
                        help="Maximum FFT bin to consider")
    parser.add_argument("--n-phase-bins", type=int, default=18)
    parser.add_argument("--n-shuffle", type=int, default=20,
                        help="Phase-randomized null runs per sample-unit")
    parser.add_argument("--n-null-units", type=int, default=8,
                        help="Number of units to compute null on")
    parser.add_argument("--n-top-evidence", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print("Loading WikiText-2 ...")
    ds = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="validation",
    )
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]
    print(f"  {len(texts)} samples")

    model, tokenizer = load_model()

    print("\nCollecting streams ...")
    streams = collect_streams(model, tokenizer, texts, args.seq_len)
    print(f"  shape: {streams.shape}  (n_samples, n_sublayers, d_model)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    N, L, D = streams.shape

    bin_pairs = [
        (s, f)
        for s in range(1, args.max_bin + 1)
        for f in range(1, args.max_bin + 1)
        if s != f
    ]
    print(f"\nScanning {len(bin_pairs)} bin pairs per unit "
          f"(max_bin={args.max_bin})")

    # Per-unit PAC matrix
    print(f"\nComputing PAC for all {D} units ...")
    per_unit = np.zeros((D, len(bin_pairs)))
    for u in range(D):
        result = pac_for_signal_pooled(
            streams[:, :, u], bin_pairs,
            n_phase_bins=args.n_phase_bins,
        )
        for j, key in enumerate(bin_pairs):
            per_unit[u, j] = result[key]
        if (u + 1) % 64 == 0:
            print(f"  {u + 1} / {D} units")

    # Phase-randomized null on a sample of units
    sample_units = np.random.default_rng(args.seed + 1).choice(
        D, size=min(args.n_null_units, D), replace=False,
    )
    print(f"\nComputing phase-randomized null on "
          f"{len(sample_units)} units ...")
    null_per_pair = np.zeros((
        len(sample_units), args.n_shuffle, len(bin_pairs),
    ))
    for i, u in enumerate(sample_units):
        null_per_pair[i] = phase_randomized_null(
            streams[:, :, u], bin_pairs, args.n_shuffle,
            n_phase_bins=args.n_phase_bins,
            rng=np.random.default_rng(args.seed + 100 + i),
        )
        print(f"  null unit {i + 1} / {len(sample_units)} done")

    null_flat = null_per_pair.reshape(-1, len(bin_pairs))
    null_p95 = np.percentile(null_flat, 95, axis=0)
    null_mean = null_flat.mean(axis=0)
    null_p99 = np.percentile(null_flat, 99, axis=0)
    print(f"  null mean MI:  {null_mean.mean():.4f}")
    print(f"  null p95 mean: {null_p95.mean():.4f}")
    print(f"  null p99 mean: {null_p99.mean():.4f}")
    print(f"  real max MI:   {per_unit.max():.4f}")

    sig_mask = per_unit > null_p95[None, :]
    sig_strict = per_unit > null_p99[None, :]

    # Findings
    findings = []
    for u in range(D):
        for j, (s, f) in enumerate(bin_pairs):
            mi = per_unit[u, j]
            findings.append({
                "unit":         int(u),
                "slow_bin":     int(s),
                "fast_bin":     int(f),
                "mi":           float(mi),
                "null_p95":     float(null_p95[j]),
                "null_p99":     float(null_p99[j]),
                "above_null95": bool(mi > null_p95[j]),
                "above_null99": bool(mi > null_p99[j]),
                "ratio_to_null": float(mi / max(null_p95[j], 1e-6)),
            })
    findings.sort(key=lambda f: -f["mi"])

    # Plots
    print("\nMaking plots ...")
    plot_pac_matrix_summary(
        per_unit, bin_pairs, args.max_bin,
        out_dir / "pac_matrix_smollm2.png",
    )
    plot_pac_per_unit(
        per_unit, bin_pairs, out_dir / "pac_per_unit_smollm2.png",
    )
    plot_unit_examples(
        streams, findings, args.n_top_evidence,
        args.n_phase_bins,
        out_dir / "pac_unit_examples.png",
    )

    np.savez(
        out_dir / "pac_data.npz",
        per_unit=per_unit,
        null_p95=null_p95,
        null_p99=null_p99,
        null_mean=null_mean,
        sig_mask=sig_mask,
        sig_strict=sig_strict,
        bin_pairs=np.array(bin_pairs),
        sample_units=sample_units,
    )

    summary = {
        "model":        MODEL_NAME,
        "n_samples":    int(N),
        "n_sublayers":  int(L),
        "d_model":      int(D),
        "max_bin":      args.max_bin,
        "n_phase_bins": args.n_phase_bins,
        "n_shuffle":    args.n_shuffle,
        "n_null_units": int(len(sample_units)),
        "max_mi":       float(per_unit.max()),
        "median_mi":    float(np.median(per_unit)),
        "null_mean_mi": float(null_mean.mean()),
        "null_p95_mean": float(null_p95.mean()),
        "null_p99_mean": float(null_p99.mean()),
        "fraction_above_null95": float(sig_mask.mean()),
        "fraction_above_null99": float(sig_strict.mean()),
        "n_units_with_any_significant_pair_p95": int(
            np.any(sig_mask, axis=1).sum()
        ),
        "n_units_with_any_significant_pair_p99": int(
            np.any(sig_strict, axis=1).sum()
        ),
        "top_50_findings": findings[:50],
    }
    with open(out_dir / "pac_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Top 15 PAC findings ---")
    print(f"{'rank':<5} {'unit':<6} {'s->f':<8} "
          f"{'MI':>8} {'null_p95':>10} {'ratio':>8} "
          f"{'sig95':>6} {'sig99':>6}")
    for rank, f in enumerate(findings[:15], 1):
        print(f"{rank:<5} u{f['unit']:<5} "
              f"{f['slow_bin']}->{f['fast_bin']:<5} "
              f"{f['mi']:>8.4f} {f['null_p95']:>10.4f} "
              f"{f['ratio_to_null']:>8.2f} "
              f"{'YES' if f['above_null95'] else 'no':>6} "
              f"{'YES' if f['above_null99'] else 'no':>6}")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
