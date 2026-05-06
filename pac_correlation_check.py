"""
Diagnose what the bin-16 PAC at slow=16/fast=anything is actually measuring.

The PAC method as implemented in pac_probe_smollm2.py cannot detect
within-signal temporal multiplexing of pure Fourier components, because
each bin's amplitude is constant within a single sample. So whatever
makes real PAC > null PAC must operate across the 256 samples. There
are two non-exclusive mechanisms that could drive the 110x ratio
observed for unit 87 at slow=16:

H1: Cross-sample correlation. The bin-16 phase coefficient phi_n and
    the bin-fast magnitude coefficient A_n are correlated across the
    256 samples. Test: circular-linear r^2 between A and phi at 1-fold
    and 4-fold symmetry. With L=64 and k=16, bin-16 phase visits 4
    distinct values per sample {phi_n + j*pi/2 for j=0..3}, so the
    PAC discretization is sensitive to 4-fold modulation; r^2_4 should
    be the tighter correspondence to the original MI.

H2: Phase concentration. The bin-16 phase coefficient phi_n is
    concentrated around some mean value across samples instead of
    being uniform on [0, 2pi). If concentrated, real samples visit
    similar 4 phase bins each, while phase-randomized null samples
    visit random bins. This alone produces real MI > null MI even
    with no correlation between phi and A.
    Test: circular variance of (4*phi_n) across samples. Uniform
    distribution gives variance close to 1; a delta gives 0.

Either hypothesis or their combination can explain the PAC without
implying genuine within-signal temporal multiplexing.

Diagnostic units:
  TOP_PAC: 87, 212, 78, 612, 642, 40   (peaked at slow=16 in PAC scan)
  CONTROL: 295, 547, 753, 939          (median MI; baseline)

Outputs:
  pac_correlation_scatter.png    -- scatter (4*phi mod 2pi, A) per unit
  pac_correlation_summary.json   -- per-unit numerical results

Usage:
  python pac_correlation_check.py
  python pac_correlation_check.py --fast-bin 5
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
N_SAMPLES = 256
SEQ_LEN = 128
SEED = 0

TOP_PAC_UNITS = [87, 212, 78, 612, 642, 40]
CONTROL_UNITS = [295, 547, 753, 939]


# ---------------------------------------------------------------------------
# Stream collection (matches pac_probe_smollm2.py)
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
            def fn(m, inp, _):
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
# Per-sample Fourier extraction
# ---------------------------------------------------------------------------

def per_sample_fourier(streams, unit, slow_bin, fast_bin):
    """For each sample, extract phi_slow, A_fast, M_slow for one unit.
    Returns dict of arrays of shape (N,)."""
    sig = streams[:, :, unit]                       # (N, L)
    sig = sig - sig.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(sig, axis=1)                 # (N, n_bins)
    return {
        "phi": np.angle(spec[:, slow_bin]),
        "A":   np.abs(spec[:, fast_bin]),
        "M":   np.abs(spec[:, slow_bin]),
    }


# ---------------------------------------------------------------------------
# Circular statistics
# ---------------------------------------------------------------------------

def circular_linear_r2(x, theta, k=1):
    """Circular-linear r^2 between linear x and circular theta (multiplied
    by k to capture k-fold symmetric structure).

    Standard formula (Mardia & Jupp): proportion of variance in x
    explained by the optimal sinusoid in (cos(k*theta), sin(k*theta))."""
    if len(x) < 3 or x.std() < 1e-12:
        return 0.0
    c = np.cos(k * theta)
    s = np.sin(k * theta)
    if c.std() < 1e-12 or s.std() < 1e-12:
        return 0.0
    r_xc = np.corrcoef(x, c)[0, 1]
    r_xs = np.corrcoef(x, s)[0, 1]
    r_cs = np.corrcoef(c, s)[0, 1]
    denom = 1.0 - r_cs ** 2
    if abs(denom) < 1e-12:
        return 0.0
    val = (r_xc ** 2 + r_xs ** 2 - 2 * r_xc * r_xs * r_cs) / denom
    return float(np.clip(val, 0.0, 1.0))


def circular_variance(theta, k=1):
    """Circular variance of (k*theta). 0 = perfect concentration,
    1 = uniform distribution. Defined as 1 - mean resultant length."""
    z = np.exp(1j * k * theta)
    return float(1.0 - np.abs(z.mean()))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_unit_scatters(streams, all_units, classes, out_path,
                        slow_bin, fast_bin):
    n = len(all_units)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3.7 * n_rows),
                              squeeze=False)

    rows = []
    for i, (u, cls) in enumerate(zip(all_units, classes)):
        ax = axes.flat[i]
        f = per_sample_fourier(streams, u, slow_bin, fast_bin)
        phi, A, M = f["phi"], f["A"], f["M"]

        r2_1 = circular_linear_r2(A, phi, k=1)
        r2_4 = circular_linear_r2(A, phi, k=4)
        cv_4 = circular_variance(phi, k=4)
        cv_1 = circular_variance(phi, k=1)

        # 4-fold scatter: x = (4*phi) mod 2pi
        x = (4 * phi) % (2 * np.pi)
        color = "C0" if cls == "TOP_PAC" else "gray"
        ax.scatter(x, A, s=12, alpha=0.6, c=color, edgecolors="none")

        # Binned mean curve
        n_pbins = 12
        bin_edges = np.linspace(0, 2 * np.pi, n_pbins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_idx = np.clip(np.digitize(x, bin_edges) - 1, 0, n_pbins - 1)
        bin_means = np.array([
            A[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
            for b in range(n_pbins)
        ])
        ax.plot(bin_centers, bin_means, "-o", color="red",
                markersize=5, linewidth=1.5, alpha=0.85,
                label="binned mean")

        marker = " *" if cls == "TOP_PAC" else ""
        ax.set_title(
            f"u{u}{marker}  r$^2_4$={r2_4:.3f}  r$^2_1$={r2_1:.3f}\n"
            f"circ.var(4$\\varphi$)={cv_4:.3f} (1=unif, 0=delta)",
            fontsize=9,
        )
        ax.set_xlabel(r"$(4\varphi_{16}) \; \mathrm{mod} \; 2\pi$")
        ax.set_ylabel(f"|coef[{fast_bin}]|")
        ax.set_xlim(0, 2 * np.pi)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

        rows.append({
            "unit":   int(u),
            "class":  cls,
            "r2_1":   r2_1,
            "r2_4":   r2_4,
            "circ_var_phi_1fold": cv_1,
            "circ_var_phi_4fold": cv_4,
            "M_mean": float(M.mean()),
            "M_std":  float(M.std()),
            "A_mean": float(A.mean()),
            "A_std":  float(A.std()),
        })

    for i in range(n, n_rows * n_cols):
        axes.flat[i].axis("off")

    fig.suptitle(
        f"Cross-sample analysis: A=|coef[{fast_bin}]| vs "
        rf"$\varphi_{{16}}$=arg(coef[{slow_bin}])"
        "\n  H1 (correlation): high $r^2_4$ means A varies systematically "
        "with 4-fold reduction of $\\varphi$"
        "\n  H2 (concentration): low circ.var(4$\\varphi$) means $\\varphi$ "
        "is concentrated mod $\\pi$/2 across samples",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow-bin", type=int, default=16)
    parser.add_argument("--fast-bin", type=int, default=1)
    parser.add_argument("--out-path", default="pac_correlation_scatter.png")
    parser.add_argument("--summary-path",
                        default="pac_correlation_summary.json")
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
    print(f"  shape: {streams.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    all_units = TOP_PAC_UNITS + CONTROL_UNITS
    classes = (["TOP_PAC"] * len(TOP_PAC_UNITS)
                + ["control"] * len(CONTROL_UNITS))

    print(f"\nComputing diagnostics: slow={args.slow_bin}, "
          f"fast={args.fast_bin}")
    rows = plot_unit_scatters(
        streams, all_units, classes, args.out_path,
        args.slow_bin, args.fast_bin,
    )
    print(f"\nFigure: {Path(args.out_path).resolve()}")

    # ---- Per-unit table ----
    print()
    print(f"=== Per-unit diagnostics: A=|coef[{args.fast_bin}]|, "
          f"phi=arg(coef[{args.slow_bin}]) ===")
    print(f"{'unit':>5} {'class':>9} {'r²_1':>7} {'r²_4':>7} "
          f"{'cvar(φ)':>9} {'cvar(4φ)':>10} {'⟨|coef[16]|⟩':>14}")
    for r in rows:
        print(f"{r['unit']:>5} {r['class']:>9} "
              f"{r['r2_1']:>7.3f} {r['r2_4']:>7.3f} "
              f"{r['circ_var_phi_1fold']:>9.3f} "
              f"{r['circ_var_phi_4fold']:>10.3f} "
              f"{r['M_mean']:>14.4f}")

    # ---- Aggregate verdict ----
    top = [r for r in rows if r["class"] == "TOP_PAC"]
    ctl = [r for r in rows if r["class"] == "control"]

    top_r2_4   = np.mean([r["r2_4"] for r in top])
    ctl_r2_4   = np.mean([r["r2_4"] for r in ctl])
    top_cvar4  = np.mean([r["circ_var_phi_4fold"] for r in top])
    ctl_cvar4  = np.mean([r["circ_var_phi_4fold"] for r in ctl])

    print()
    print(f"=== Aggregate: TOP_PAC vs control ===")
    print(f"  Mean r²_4:           TOP={top_r2_4:.3f}  ctl={ctl_r2_4:.3f}")
    print(f"  Mean circ.var(4φ):   TOP={top_cvar4:.3f}  ctl={ctl_cvar4:.3f}")
    print(f"    (1.0 = uniform on [0, 2π); 0 = delta)")

    print()
    print("=== Verdict ===")
    h1_evidence = top_r2_4 > 0.05 and top_r2_4 > 3 * ctl_r2_4
    h2_evidence = top_cvar4 < 0.7 and top_cvar4 < ctl_cvar4 - 0.05
    if h1_evidence and h2_evidence:
        print("BOTH H1 and H2 are supported. PAC reflects a combination of")
        print("(a) cross-sample correlation between phi_16 and |coef[fast]|")
        print("(b) concentration of phi_16 mod pi/2 in TOP_PAC units.")
    elif h1_evidence:
        print("H1 supported (cross-sample correlation), H2 not.")
        print("PAC reflects: TOP_PAC units have phi_16 phase coefficient")
        print("that systematically tracks |coef[fast]| across inputs.")
    elif h2_evidence:
        print("H2 supported (phase concentration), H1 not.")
        print("PAC reflects: TOP_PAC units have phi_16 concentrated around")
        print("a mean value mod pi/2 across inputs. The histogram is")
        print("concentrated even without correlation between phi and A.")
    else:
        print("Neither hypothesis cleanly supported by these numbers.")
        print("The PAC must be driven by something else; recheck the")
        print("methodology of the original PAC measurement.")

    # ---- Multi-fast-bin sweep for unit 87 ----
    print()
    print(f"=== Unit 87: r²_4 across fast bins (slow={args.slow_bin}) ===")
    print(f"{'fast':>6} {'r²_1':>8} {'r²_4':>8} {'⟨|coef[fast]|⟩':>16}")
    for fb in [1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 17, 19]:
        f = per_sample_fourier(streams, 87, args.slow_bin, fb)
        r1 = circular_linear_r2(f["A"], f["phi"], k=1)
        r4 = circular_linear_r2(f["A"], f["phi"], k=4)
        print(f"{fb:>6} {r1:>8.3f} {r4:>8.3f} {f['A'].mean():>16.4f}")

    # ---- Save summary ----
    summary = {
        "model": MODEL_NAME,
        "slow_bin": args.slow_bin,
        "fast_bin": args.fast_bin,
        "n_samples": int(streams.shape[0]),
        "n_sublayers": int(streams.shape[1]),
        "per_unit": rows,
        "aggregate": {
            "mean_r2_4_TOP_PAC":    float(top_r2_4),
            "mean_r2_4_control":    float(ctl_r2_4),
            "mean_cvar4_TOP_PAC":   float(top_cvar4),
            "mean_cvar4_control":   float(ctl_cvar4),
        },
    }
    with open(args.summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {Path(args.summary_path).resolve()}")


if __name__ == "__main__":
    main()
