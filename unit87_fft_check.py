"""
FFT power spectrum check for unit 87 (and other top PAC units) in
SmolLM2-360M's residual stream.

Question: is unit 87's high PAC at slow_bin=16 because (A) bin 16 is so
dominant in unit 87's depth spectrum that the unit IS the carrier, or
(B) unit 87 has substantial power at many bins with the bin-16 phase
gating the others' amplitudes?

Method:
  - Recollect the same residual streams used in pac_probe_smollm2.py
    (256 wikitext-2 texts, last-token, 64 sublayers via the same hooks).
  - Center each per-sample depth trajectory.
  - Compute |FFT|^2 per sample across the depth axis (length-64), then
    average across samples to get per-bin mean power for each unit.
  - Normalize so per-unit bins sum to 1, giving a probability mass
    function over depth-axis frequencies.
  - For each diagnostic unit, report:
      * fraction of power at bin 16
      * top-5 bins by power
      * full spectrum

Diagnostic units:
  - 87, 212, 78, 612, 642, 40   (top PAC units, all peak at slow=16)
  - 295, 547, 753, 939          (median-MI units, controls)

Output: a single matplotlib figure with one panel per unit, plus a
console summary.

Usage:
  python unit87_fft_check.py [--out-path unit87_fft_spectrum.png]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
N_SAMPLES = 256
SEQ_LEN = 128
SEED = 0

# Top PAC units (from pac_data.npz analysis) and bulk-control units.
TOP_PAC_UNITS = [87, 212, 78, 612, 642, 40]
CONTROL_UNITS = [295, 547, 753, 939]
ALL_UNITS = TOP_PAC_UNITS + CONTROL_UNITS


# ---------------------------------------------------------------------------
# Stream collection (matches pac_probe_smollm2.py exactly)
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
    blocks = model.model.layers
    targets = []
    for block in blocks:
        targets.append(block.input_layernorm)
        targets.append(block.post_attention_layernorm)
    return targets


def collect_streams(model, tokenizer, texts, seq_len):
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
        last = torch.stack(
            [c[0, -1, :].float() for c in captures], dim=0
        )
        out.append(last.cpu().numpy())
        if (i + 1) % 32 == 0:
            print(f"  {i + 1} / {len(texts)} samples ...")
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Spectrum computation
# ---------------------------------------------------------------------------

def per_unit_spectrum(streams, unit):
    """Return mean power spectrum across samples for a single unit.

    streams: (N, L, D) array of residual stream snapshots, depth axis = L.
    Returns: (n_bins,) array, where n_bins = L//2 + 1, summing to 1."""
    sig = streams[:, :, unit]                  # (N, L)
    sig = sig - sig.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(sig, axis=1)            # (N, n_bins)
    power = np.abs(spec) ** 2                   # (N, n_bins)
    mean_power = power.mean(axis=0)             # (n_bins,)
    total = mean_power.sum()
    if total < 1e-30:
        return mean_power
    return mean_power / total


# ---------------------------------------------------------------------------
# Plot + report
# ---------------------------------------------------------------------------

def make_figure(streams, units, n_top, out_path):
    L = streams.shape[1]
    n_bins = L // 2 + 1
    bin_axis = np.arange(n_bins)

    n_panels = len(units)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3 * n_rows),
                              squeeze=False)
    summary_rows = []

    for idx, u in enumerate(units):
        spec = per_unit_spectrum(streams, u)
        ax = axes[idx // n_cols, idx % n_cols]

        ax.bar(bin_axis, spec, color="C0", alpha=0.85)
        ax.axvline(16, color="red", linestyle=":", linewidth=1.5,
                    alpha=0.7, label="bin 16 (carrier)")
        ax.axvline(8,  color="orange", linestyle=":", linewidth=1.0,
                    alpha=0.6, label="bin 8 (harmonic)")
        ax.axvline(32, color="gray", linestyle=":", linewidth=1.0,
                    alpha=0.5, label="bin 32 (Nyquist)")
        ax.set_xlim(0, n_bins - 1)
        ax.set_xlabel("FFT bin (depth axis)")
        ax.set_ylabel("normalized power")
        is_top = u in TOP_PAC_UNITS
        marker = " [TOP-PAC]" if is_top else " [control]"
        ax.set_title(f"unit {u}{marker}: bin-16 share = "
                     f"{spec[16]*100:.1f}%, "
                     f"bin-8 share = {spec[8]*100:.1f}%",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Sort bins by power for top-N summary
        order = np.argsort(-spec)
        top_bins = order[:n_top]
        top_str = ", ".join([f"bin {b}: {spec[b]*100:.1f}%"
                              for b in top_bins])
        summary_rows.append({
            "unit":  u,
            "top":   is_top,
            "bin16_share": float(spec[16]),
            "bin8_share":  float(spec[8]),
            "bin32_share": float(spec[32]) if 32 < n_bins else 0.0,
            "top_bins":    top_str,
        })

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.suptitle(
        f"Depth-axis power spectrum per unit ({MODEL_NAME}, "
        f"L={L} sublayers, N=256 samples)\n"
        "Top-PAC units should show strong bin-16 power if reading A "
        "(unit IS the carrier); flat distribution if reading B (unit "
        "carries multiple modes)",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return summary_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", default="unit87_fft_spectrum.png")
    parser.add_argument("--n-top-bins", type=int, default=5,
                        help="How many top bins to list per unit.")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("Loading WikiText-2 ...")
    ds = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="validation",
    )
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:N_SAMPLES]
    print(f"  {len(texts)} samples")

    model, tokenizer = load_model()

    print("\nCollecting streams ...")
    streams = collect_streams(model, tokenizer, texts, SEQ_LEN)
    print(f"  shape: {streams.shape}  (N, L, D)")
    N, L, D = streams.shape

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"\nComputing FFT spectra for {len(ALL_UNITS)} units ...")
    rows = make_figure(streams, ALL_UNITS, args.n_top_bins, args.out_path)
    print(f"Saved figure to {Path(args.out_path).resolve()}")

    # ---- Console summary ----
    print()
    print(f"=== Per-unit spectrum summary (N={N}, L={L}) ===")
    print(f"{'unit':>6} {'class':>10} "
          f"{'bin16 %':>10} {'bin8 %':>10} {'bin32 %':>10}  top-5 bins")
    for r in rows:
        cls = "TOP-PAC" if r["top"] else "control"
        print(f"{r['unit']:>6} {cls:>10} "
              f"{r['bin16_share']*100:>9.2f}% "
              f"{r['bin8_share']*100:>9.2f}% "
              f"{r['bin32_share']*100:>9.2f}%  {r['top_bins']}")

    # ---- Verdict ----
    top_bin16 = [r["bin16_share"] for r in rows if r["top"]]
    ctl_bin16 = [r["bin16_share"] for r in rows if not r["top"]]
    print()
    print(f"=== Verdict ===")
    print(f"Top-PAC units mean bin-16 share: {np.mean(top_bin16)*100:.1f}%")
    print(f"Control units mean bin-16 share: {np.mean(ctl_bin16)*100:.1f}%")
    if np.mean(top_bin16) > 0.50:
        print("Reading A: top-PAC units' depth signal is dominated by bin 16.")
        print("  PAC is largely a self-modulation effect of bin-16 dominance.")
    elif np.mean(top_bin16) > 0.20:
        print("Reading B: bin 16 leads but does not dominate.")
        print("  Genuine multi-mode multiplexing structure.")
    else:
        print("Mixed: bin 16 not dominant; PAC may live in projection rather "
              "than basis-aligned spectrum. Worth checking PCA-projected "
              "spectrum.")


if __name__ == "__main__":
    main()
