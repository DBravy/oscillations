"""
Find where the bin-16 PAC carrier lives in SmolLM2-360M's residual stream.

The basis-aligned FFT spectrum (unit87_fft_check.py) showed bin-16 share
under 1% in every diagnostic unit, despite the PAC measurement finding
ratio-to-null of 110 for unit 87 at slow_bin=16. This means the bin-16
phase information lives somewhere the per-unit FFT can't see. Two
candidate explanations, and this script tests both.

  Hypothesis 1 (detrended basis spectrum): the bin-16 carrier is in
    the basis-aligned units, but the per-sample bin-1 ramp (residual
    norm growth across depth) is so dominant that it crowds out
    everything else in the normalized power distribution. Subtracting
    a low-order polynomial trend per sample before the FFT removes
    the ramp, and a real bin-16 component would then emerge as a
    substantial share of what remains.

  Hypothesis 2 (projection-direction carrier): the bin-16 component
    is small in any single basis unit but coherent across many units,
    so it accumulates only when projected along the right direction.
    The principal directions of the residual stream's depth-axis
    variation should reveal the carrier as a single dominant component.

Method: collect the same residual streams as pac_probe_smollm2.py
(SmolLM2-360M, 256 wikitext-2 texts, last-token, 64 sublayers).

  Test 1: per-sample, fit a polynomial of degree --detrend-order to
  the depth trajectory of each unit, subtract, then FFT. Average
  power across samples and report bin-16 share for top-PAC and control
  units.

  Test 2: per-sample, form (L=64, D=960) centered matrix, SVD it
  across the unit axis to get the dominant depth-axis directions for
  that sample. Project the per-sample trajectory onto each of the top
  K directions, FFT, average power across samples (sign-aligning per
  sample so the projections add coherently in mean spectrum). Report
  bin-16 share for each top PC.

  Both tests share data collection.

Output: figure with both diagnostics side by side, plus console summary.

Usage:
  python locate_pac_carrier.py
  python locate_pac_carrier.py --detrend-order 3 --n-pcs 10
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

TOP_PAC_UNITS = [87, 212, 78, 612, 642, 40]
CONTROL_UNITS = [295, 547, 753, 939]


# ---------------------------------------------------------------------------
# Stream collection
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
# Test 1: detrended per-unit FFT
# ---------------------------------------------------------------------------

def detrend_polynomial(sig, order):
    """sig: (L,) or (..., L). Subtract a polynomial of given order from
    each independent trajectory along the last axis."""
    L = sig.shape[-1]
    l = np.arange(L)
    flat = sig.reshape(-1, L)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        coefs = np.polyfit(l, flat[i], order)
        trend = np.polyval(coefs, l)
        out[i] = flat[i] - trend
    return out.reshape(sig.shape)


def detrended_unit_spectrum(streams, unit, detrend_order):
    """Return mean detrended power spectrum for one unit, normalized."""
    sig = streams[:, :, unit]                              # (N, L)
    sig_dt = detrend_polynomial(sig, detrend_order)        # (N, L)
    spec = np.fft.rfft(sig_dt, axis=1)
    power = np.abs(spec) ** 2
    mean_power = power.mean(axis=0)
    total = mean_power.sum()
    return mean_power / max(total, 1e-30)


# ---------------------------------------------------------------------------
# Test 2: SVD-projected spectrum
# ---------------------------------------------------------------------------

def per_sample_pcs(streams, k):
    """For each sample, compute the top-k PCs of the (L, D) depth-by-unit
    matrix (after centering across depth). Returns:
      pc_signals  (N, k, L) -- the projection of each sample's trajectory
                              onto its top-k PCs
      pc_var_frac (N, k)   -- fraction of total variance per PC per sample
      pc_dirs     (N, k, D) -- the unit-space direction of each PC

    Sign convention: each PC's sign is set so its first nonzero value
    along depth is positive, for stable averaging across samples.
    """
    N, L, D = streams.shape
    pc_signals  = np.empty((N, k, L))
    pc_var_frac = np.empty((N, k))
    pc_dirs     = np.empty((N, k, D))
    for n in range(N):
        x = streams[n]                                       # (L, D)
        x = x - x.mean(axis=0, keepdims=True)
        # SVD: x = U S Vt, with U (L, r), S (r,), Vt (r, D)
        U, S, Vt = np.linalg.svd(x, full_matrices=False)
        # Top-k components
        for j in range(k):
            sig = U[:, j] * S[j]
            # Sign-align: first sublayer positive
            if sig[0] < 0:
                sig = -sig
                pc_dirs[n, j] = -Vt[j]
            else:
                pc_dirs[n, j] = Vt[j]
            pc_signals[n, j] = sig
            pc_var_frac[n, j] = (S[j] ** 2) / (S ** 2).sum()
    return pc_signals, pc_var_frac, pc_dirs


def pc_mean_spectrum(pc_signals_one_pc):
    """pc_signals_one_pc: (N, L). Return normalized mean power spectrum."""
    spec = np.fft.rfft(pc_signals_one_pc, axis=1)
    power = np.abs(spec) ** 2
    mean_power = power.mean(axis=0)
    total = mean_power.sum()
    return mean_power / max(total, 1e-30)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(streams, detrend_order, n_pcs, out_path):
    L = streams.shape[1]
    n_bins = L // 2 + 1
    bin_axis = np.arange(n_bins)

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

    # ---- Test 1: detrended per-unit spectra ----
    ax_top = fig.add_subplot(gs[0, 0])
    ax_ctl = fig.add_subplot(gs[0, 1])

    detrended_top = {}
    detrended_ctl = {}
    for u in TOP_PAC_UNITS:
        detrended_top[u] = detrended_unit_spectrum(streams, u, detrend_order)
    for u in CONTROL_UNITS:
        detrended_ctl[u] = detrended_unit_spectrum(streams, u, detrend_order)

    for u, spec in detrended_top.items():
        ax_top.plot(bin_axis, spec, "-o", markersize=3,
                     linewidth=1.2, alpha=0.8, label=f"u{u}")
    ax_top.axvline(16, color="red", linestyle=":", linewidth=1.5,
                    alpha=0.5, label="bin 16")
    ax_top.set_xlim(0, n_bins - 1)
    ax_top.set_xlabel("FFT bin")
    ax_top.set_ylabel("normalized power (detrended)")
    ax_top.set_title(f"Test 1a: TOP-PAC units, detrended spectrum "
                     f"(poly order {detrend_order})")
    ax_top.legend(fontsize=8, ncol=2)
    ax_top.grid(alpha=0.3)

    for u, spec in detrended_ctl.items():
        ax_ctl.plot(bin_axis, spec, "-o", markersize=3,
                     linewidth=1.2, alpha=0.8, label=f"u{u}")
    ax_ctl.axvline(16, color="red", linestyle=":", linewidth=1.5,
                    alpha=0.5, label="bin 16")
    ax_ctl.set_xlim(0, n_bins - 1)
    ax_ctl.set_xlabel("FFT bin")
    ax_ctl.set_ylabel("normalized power (detrended)")
    ax_ctl.set_title(f"Test 1b: CONTROL units, detrended spectrum")
    ax_ctl.legend(fontsize=8, ncol=2)
    ax_ctl.grid(alpha=0.3)

    # ---- Test 2: SVD-projected spectra ----
    print(f"  Computing per-sample SVD with top-{n_pcs} PCs ...")
    pc_signals, pc_var_frac, pc_dirs = per_sample_pcs(streams, n_pcs)
    pc_specs = np.array([pc_mean_spectrum(pc_signals[:, j])
                          for j in range(n_pcs)])

    # PC variance fractions across samples
    ax_var = fig.add_subplot(gs[1, 0])
    var_means = pc_var_frac.mean(axis=0)
    var_stds  = pc_var_frac.std(axis=0)
    ax_var.bar(range(1, n_pcs + 1), var_means * 100,
                yerr=var_stds * 100, capsize=3,
                color="C2", alpha=0.85)
    ax_var.set_xlabel("PC index")
    ax_var.set_ylabel("variance fraction (%)")
    ax_var.set_title(f"Test 2a: top-{n_pcs} PCs of per-sample (L, D) "
                     f"matrix, mean variance fraction")
    ax_var.set_xticks(range(1, n_pcs + 1))
    ax_var.grid(alpha=0.3, axis="y")

    # PC bin-16 share
    ax_bin16 = fig.add_subplot(gs[1, 1])
    ax_bin16.bar(range(1, n_pcs + 1), pc_specs[:, 16] * 100,
                  color="C3", alpha=0.85, label="bin 16")
    ax_bin16.bar(range(1, n_pcs + 1), pc_specs[:, 8] * 100,
                  color="C1", alpha=0.55, label="bin 8")
    ax_bin16.set_xlabel("PC index")
    ax_bin16.set_ylabel("share (%) of PC's total power")
    ax_bin16.set_title(f"Test 2b: bin-16 and bin-8 share of each PC")
    ax_bin16.set_xticks(range(1, n_pcs + 1))
    ax_bin16.legend(fontsize=9)
    ax_bin16.grid(alpha=0.3, axis="y")

    # Spectra of top PCs
    ax_pcspec = fig.add_subplot(gs[2, :])
    n_show = min(n_pcs, 6)
    for j in range(n_show):
        ax_pcspec.plot(bin_axis, pc_specs[j], "-o", markersize=3,
                        linewidth=1.3, alpha=0.85,
                        label=f"PC {j+1} ({var_means[j]*100:.1f}% var)")
    ax_pcspec.axvline(16, color="red", linestyle=":", linewidth=1.5,
                       alpha=0.5, label="bin 16")
    ax_pcspec.axvline(8, color="orange", linestyle=":", linewidth=1.0,
                       alpha=0.5, label="bin 8")
    ax_pcspec.set_xlim(0, n_bins - 1)
    ax_pcspec.set_xlabel("FFT bin")
    ax_pcspec.set_ylabel("normalized power")
    ax_pcspec.set_title(f"Test 2c: full spectrum of top-{n_show} PCs "
                        f"(averaged across samples)")
    ax_pcspec.legend(fontsize=8, ncol=2)
    ax_pcspec.grid(alpha=0.3)

    fig.suptitle(
        f"Locating the bin-16 carrier in {MODEL_NAME}\n"
        "Test 1: does detrending reveal bin-16 in basis-aligned units?  "
        "Test 2: does the carrier live in a PC direction?",
        fontsize=12, y=0.997,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return detrended_top, detrended_ctl, pc_specs, var_means, pc_dirs


# ---------------------------------------------------------------------------
# Per-PC alignment with top-PAC units (which units load each PC?)
# ---------------------------------------------------------------------------

def report_pc_unit_alignment(pc_dirs, top_units, n_pcs_show=4, n_units_show=15):
    """For each top PC direction, report which basis units load most heavily.
    pc_dirs: (N, n_pcs, D) per-sample directions.
    Average |loading| across samples gives a robust per-unit weight.
    """
    N, n_pcs, D = pc_dirs.shape
    # Average absolute loading across samples
    mean_abs = np.abs(pc_dirs).mean(axis=0)         # (n_pcs, D)
    print()
    print(f"=== PC unit-alignment: top-{n_units_show} basis units per PC ===")
    print(f"(mean |loading| across {N} samples; checking if top-PAC units "
          f"are concentrated in any PC)")
    for j in range(min(n_pcs_show, n_pcs)):
        order = np.argsort(-mean_abs[j])
        top = order[:n_units_show]
        marker = ["*" if u in top_units else " " for u in top]
        items = [f"{m}u{u}({mean_abs[j, u]:.3f})"
                 for u, m in zip(top, marker)]
        print(f"PC {j+1}: " + ", ".join(items))
        # How many of TOP_PAC_UNITS appear in this PC's top-50 loadings
        pac_in_top50 = sum(u in order[:50] for u in TOP_PAC_UNITS)
        print(f"        TOP_PAC_UNITS in this PC's top-50: "
              f"{pac_in_top50}/{len(TOP_PAC_UNITS)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", default="locate_pac_carrier.png")
    parser.add_argument("--detrend-order", type=int, default=3,
                        help="Polynomial order for per-sample detrending.")
    parser.add_argument("--n-pcs", type=int, default=10)
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

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"\nRunning Test 1 (detrend, order {args.detrend_order}) "
          f"and Test 2 (SVD, n_pcs={args.n_pcs}) ...")
    detrended_top, detrended_ctl, pc_specs, var_means, pc_dirs = plot_results(
        streams, args.detrend_order, args.n_pcs, args.out_path,
    )
    print(f"\nFigure: {Path(args.out_path).resolve()}")

    # ---- Console summary ----
    L = streams.shape[1]
    print()
    print(f"=== Test 1: detrended bin-16 share (poly order "
          f"{args.detrend_order}) ===")
    print(f"{'unit':>6} {'class':>10} "
          f"{'bin16 %':>10} {'bin8 %':>10}  top-3 bins")
    rows_top = []
    rows_ctl = []
    for u, spec in detrended_top.items():
        order = np.argsort(-spec)[:3]
        top_str = ", ".join([f"bin {b}: {spec[b]*100:.1f}%" for b in order])
        print(f"{u:>6} {'TOP-PAC':>10} "
              f"{spec[16]*100:>9.2f}% "
              f"{spec[8]*100:>9.2f}%  {top_str}")
        rows_top.append(spec[16])
    for u, spec in detrended_ctl.items():
        order = np.argsort(-spec)[:3]
        top_str = ", ".join([f"bin {b}: {spec[b]*100:.1f}%" for b in order])
        print(f"{u:>6} {'control':>10} "
              f"{spec[16]*100:>9.2f}% "
              f"{spec[8]*100:>9.2f}%  {top_str}")
        rows_ctl.append(spec[16])

    print()
    print(f"Mean bin-16 share, TOP-PAC: {np.mean(rows_top)*100:.2f}%")
    print(f"Mean bin-16 share, control: {np.mean(rows_ctl)*100:.2f}%")

    print()
    print(f"=== Test 2: per-PC bin-16 share ===")
    print(f"{'PC':>4} {'var %':>8} {'bin16 %':>10} {'bin8 %':>10} "
          f"{'bin1 %':>10}  top-3 bins")
    for j in range(args.n_pcs):
        spec = pc_specs[j]
        order = np.argsort(-spec)[:3]
        top_str = ", ".join([f"bin {b}: {spec[b]*100:.1f}%" for b in order])
        print(f"{j+1:>4} {var_means[j]*100:>7.1f}% "
              f"{spec[16]*100:>9.2f}% "
              f"{spec[8]*100:>9.2f}% "
              f"{spec[1]*100:>9.2f}%  {top_str}")

    # PC-unit alignment
    report_pc_unit_alignment(pc_dirs, set(TOP_PAC_UNITS),
                              n_pcs_show=4, n_units_show=15)

    # ---- Verdict ----
    print()
    print("=== Verdict ===")
    bin16_pcs = pc_specs[:, 16] * 100
    pc_with_bin16 = np.where(bin16_pcs > 5.0)[0]
    if np.mean(rows_top) > 0.05:
        print(f"Test 1: detrending revealed bin-16 share "
              f"{np.mean(rows_top)*100:.1f}% in TOP-PAC units. The carrier "
              f"WAS in basis-aligned units, just buried under bin-1 ramp.")
    else:
        print(f"Test 1: even after detrending, bin-16 share is "
              f"{np.mean(rows_top)*100:.2f}% in TOP-PAC units. The carrier "
              f"is NOT basis-aligned.")
    if len(pc_with_bin16) > 0:
        pcs_str = ", ".join([f"PC{j+1}: {bin16_pcs[j]:.1f}%"
                              for j in pc_with_bin16])
        print(f"Test 2: bin-16 concentrated in {pcs_str}. The carrier "
              f"lives in projection space.")
    else:
        print(f"Test 2: no PC has substantial bin-16 share. The carrier "
              f"is more diffuse than top-{args.n_pcs} PCs.")


if __name__ == "__main__":
    main()
