"""
Linearization-and-eigenvalue analysis of transformer sublayer Jacobians.

Tests the substrate-maintenance prediction directly at the parameter level.
For each transformer block we have two sublayers, each contributing an
additive update to the residual stream:

    x_{ell+1} = x_ell + f_sublayer(x_ell)

with f_attn(x) = self_attn(LN1(x), context) and f_mlp(x) = mlp(LN2(x)).
The substrate-maintenance hypothesis predicts that, at typical operating
points, the Jacobian J = df/dx of each sublayer should have a dominant
pair of complex eigenvalues with imaginary parts close to +-omega and
real parts close to 0, where omega is the depth-axis carrier frequency
in radians per sublayer measured from the residual stream itself. This
is the parameter-level signature of the rotational substrate that the
activation-level diagnostics already characterize.

omega is estimated from the FFT of each unit's depth-axis trajectory
across the (n_sub, d_model) snapshots collected during the same forward
passes, mirroring the convention of the phase_space_llm and
harmonic_plv scripts. We summarize as the median dominant frequency
across units.

Per-sublayer Jacobian
---------------------
Attention. Earlier tokens are treated as fixed context. We differentiate
the attention output at the LAST token w.r.t. its own input vector,
holding all prior tokens' inputs fixed. Mechanically: one forward pass,
hooks capture (a) the residual stream entering each sublayer and (b) the
positional and keyword arguments going into self_attn. We then replay
the LN1 -> self_attn call with only the last position's residual stream
substituted by a leaf tensor of shape (d_model,), and feed that scalar
function to torch.autograd.functional.jacobian.

MLP. Per-token, no context dependence. We differentiate
mlp(LN2(x_last)) directly.

Each Jacobian is a real (d_model x d_model) matrix; numpy.linalg.eigvals
returns its complex spectrum.

Diagnostics
-----------
For each (sublayer type, block index):
  - Eigenvalue scatter in the complex plane, per block.
  - Top eigenvalues by |Im|, aggregated across samples and blocks.
  - Reference: imag(top) ~ omega, real(top) ~ 0.

Comparison: trained checkpoint vs random-init checkpoint of the same
config. The trained model should show top eigenvalues clustered at
+-i*omega with small real part. The random-init model should show a
generic random-matrix spectrum, with no preferred imaginary value and
nontrivial real-part spread. The ratio (trained-clustering /
random-clustering) is the direct parameter-level signature of substrate
formation by training.

Outputs in linearization_eigvals_<model_slug>/:
  eigvalue_scatter_attn_trained.png
  eigvalue_scatter_attn_random_init.png
  eigvalue_scatter_mlp_trained.png
  eigvalue_scatter_mlp_random_init.png
  imag_distribution.png
  real_distribution.png
  per_layer_summary.png
  eigvals_summary.json

Usage:
  python linearization_eigvals_lm.py --model HuggingFaceTB/SmolLM2-360M
  python linearization_eigvals_lm.py --model gpt2 --n-samples 8 \
      --n-block-subset 6
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


# ---------------------------------------------------------------------------
# Block / sublayer discovery (matches phase_space_llm.py convention)
# ---------------------------------------------------------------------------

def get_block_info(model):
    """
    Return (blocks, attn_ln_attr, mlp_ln_attr, attn_attr, mlp_attr).
    Supports HF GPT-2 style and Llama / SmolLM2 style.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return (model.transformer.h, "ln_1", "ln_2", "attn", "mlp")
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return (
            model.model.layers,
            "input_layernorm", "post_attention_layernorm",
            "self_attn", "mlp",
        )
    raise RuntimeError("Could not locate transformer blocks.")


# ---------------------------------------------------------------------------
# Per-sample capture: residual stream entering each sublayer + self_attn args
# ---------------------------------------------------------------------------

def capture_per_sample(model, tokenizer, text, seq_len):
    """
    Run one forward pass and capture, per block:
      x_attn_in    : residual stream entering input_layernorm (attn sublayer in).
      x_mlp_in     : residual stream entering post_attention_layernorm (mlp sublayer in).
      attn_args    : positional args passed into self_attn.
      attn_kwargs  : keyword args passed into self_attn.
    """
    blocks, ln1_name, ln2_name, attn_name, _ = get_block_info(model)

    enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
    ids = enc["input_ids"].to(DEVICE)
    if ids.shape[1] < 8:
        return None

    block_data = [
        {"x_attn_in": None, "x_mlp_in": None,
         "attn_args": None, "attn_kwargs": None}
        for _ in blocks
    ]

    handles = []
    for i, blk in enumerate(blocks):
        ln1 = getattr(blk, ln1_name)
        ln2 = getattr(blk, ln2_name)
        attn = getattr(blk, attn_name)

        def make_ln_in_hook(idx, key):
            def hook(module, inp):
                # inp is a tuple, hidden state is inp[0]
                if isinstance(inp, tuple) and len(inp) > 0 and torch.is_tensor(inp[0]):
                    block_data[idx][key] = inp[0].detach()
            return hook

        def make_attn_hook(idx):
            def hook(module, args, kwargs):
                block_data[idx]["attn_args"] = args
                block_data[idx]["attn_kwargs"] = kwargs
                return None
            return hook

        handles.append(ln1.register_forward_pre_hook(make_ln_in_hook(i, "x_attn_in")))
        handles.append(ln2.register_forward_pre_hook(make_ln_in_hook(i, "x_mlp_in")))
        handles.append(attn.register_forward_pre_hook(
            make_attn_hook(i), with_kwargs=True,
        ))

    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()

    return {"input_ids": ids, "blocks": block_data}


# ---------------------------------------------------------------------------
# Per-sublayer Jacobian at the last token
# ---------------------------------------------------------------------------

def _build_attn_call(captured, x_norm):
    """
    Reconstruct the self_attn call with hidden_states substituted by x_norm.
    Returns (args, kwargs) tuple ready to splat into attn(...).
    Sanitizes cache-related kwargs that should not be reused on replay.
    """
    raw_args = captured.get("attn_args") or ()
    raw_kwargs = dict(captured.get("attn_kwargs") or {})

    # Disable caching / attention output on the replay so signatures stay
    # well-defined and we don't mutate any cache object.
    for k, v in {
        "past_key_value": None,
        "past_key_values": None,
        "use_cache": False,
        "output_attentions": False,
    }.items():
        if k in raw_kwargs:
            raw_kwargs[k] = v

    # Identify where hidden_states lives in the original call. Llama-style
    # blocks pass it via kwarg; some implementations pass it positionally.
    hs_in_args = (
        len(raw_args) > 0
        and torch.is_tensor(raw_args[0])
        and raw_args[0].dim() == 3
    )
    if hs_in_args:
        new_args = (x_norm,) + tuple(raw_args[1:])
        return new_args, raw_kwargs
    else:
        new_kwargs = dict(raw_kwargs)
        new_kwargs["hidden_states"] = x_norm
        return tuple(raw_args), new_kwargs


def compute_attn_jacobian(block, captured, ln1_name, attn_name):
    """
    J = d(attn_out_at_last_token) / d(x_at_last_token), other tokens fixed.
    Returns numpy (D, D) float32, or None if capture is missing.
    """
    ln1 = getattr(block, ln1_name)
    attn = getattr(block, attn_name)

    x_full = captured["x_attn_in"]
    if x_full is None or x_full.dim() != 3 or x_full.shape[0] != 1:
        return None

    # Promote to float32 for stable Jacobian computation regardless of model dtype.
    x_full_f32 = x_full.to(torch.float32)
    x_last_init = x_full_f32[0, -1, :].detach().clone()

    # Pre-compute the part we don't need to differentiate through. Note we
    # only sub in x_last; other positions of x_full are constants for J.

    def f(x_last):
        # In-place index assignment is differentiable through the assigned
        # values. clone() detaches us from the captured tensor's graph (it
        # has no grad anyway) and gives us a writable buffer.
        x = x_full_f32.clone()
        x[0, -1, :] = x_last
        x_norm = ln1(x)
        # Attention may run in lower precision internally; cast inputs to
        # match the module's expected dtype if needed.
        target_dtype = next(attn.parameters()).dtype
        if x_norm.dtype != target_dtype:
            x_norm = x_norm.to(target_dtype)
        args, kwargs = _build_attn_call(captured, x_norm)
        out = attn(*args, **kwargs)
        attn_out = out[0] if isinstance(out, tuple) else out
        return attn_out[0, -1, :].to(torch.float32)

    J = torch.autograd.functional.jacobian(
        f, x_last_init, vectorize=False, create_graph=False,
    )
    return J.detach().cpu().numpy().astype(np.float32)


def compute_mlp_jacobian(block, captured, ln2_name, mlp_name):
    """
    J = d(mlp_out_at_last_token) / d(x_at_last_token). Per-token, no context.
    """
    ln2 = getattr(block, ln2_name)
    mlp = getattr(block, mlp_name)

    x_full = captured["x_mlp_in"]
    if x_full is None or x_full.dim() != 3:
        return None

    x_last_init = x_full[0, -1, :].detach().clone().to(torch.float32)
    target_dtype = next(mlp.parameters()).dtype

    def f(x_last):
        x = x_last.unsqueeze(0).unsqueeze(0)  # (1, 1, D), float32
        x_norm = ln2(x)
        if x_norm.dtype != target_dtype:
            x_norm = x_norm.to(target_dtype)
        out = mlp(x_norm)
        return out[0, 0, :].to(torch.float32)

    J = torch.autograd.functional.jacobian(
        f, x_last_init, vectorize=False, create_graph=False,
    )
    return J.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Driver: capture + Jacobians + eigenvalues for one checkpoint
# ---------------------------------------------------------------------------

def analyze_checkpoint(model, tokenizer, texts, seq_len, n_block_subset=None,
                      verbose=True):
    """
    Run capture + per-sublayer Jacobian + eigenvalue extraction across
    samples and blocks. Returns a dict with eigenvalues per (sublayer, block)
    plus the residual streams for omega estimation.
    """
    model.eval()
    blocks, ln1_name, ln2_name, attn_name, mlp_name = get_block_info(model)
    n_blocks = len(blocks)

    if n_block_subset is None or n_block_subset >= n_blocks:
        block_indices = list(range(n_blocks))
    else:
        block_indices = np.linspace(
            0, n_blocks - 1, n_block_subset, dtype=int,
        ).tolist()
        block_indices = sorted(set(block_indices))

    eigvals_attn = {bi: [] for bi in block_indices}
    eigvals_mlp = {bi: [] for bi in block_indices}
    streams_for_omega = []

    t0 = time.time()
    n_done = 0
    for s_idx, text in enumerate(texts):
        captured = capture_per_sample(model, tokenizer, text, seq_len)
        if captured is None:
            continue

        # Build (n_sub, D) stream at the last token for this sample.
        sample_stream = []
        for d in captured["blocks"]:
            if d["x_attn_in"] is not None:
                sample_stream.append(
                    d["x_attn_in"][0, -1, :].float().cpu().numpy()
                )
            if d["x_mlp_in"] is not None:
                sample_stream.append(
                    d["x_mlp_in"][0, -1, :].float().cpu().numpy()
                )
        if len(sample_stream) > 0:
            streams_for_omega.append(np.stack(sample_stream, axis=0))

        # Per-block Jacobians
        for bi in block_indices:
            blk = blocks[bi]
            cb = captured["blocks"][bi]

            J_a = compute_attn_jacobian(blk, cb, ln1_name, attn_name)
            if J_a is not None:
                eigvals_attn[bi].append(np.linalg.eigvals(J_a))

            J_m = compute_mlp_jacobian(blk, cb, ln2_name, mlp_name)
            if J_m is not None:
                eigvals_mlp[bi].append(np.linalg.eigvals(J_m))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        n_done += 1
        if verbose:
            elapsed = time.time() - t0
            avg = elapsed / max(n_done, 1)
            remaining = avg * (len(texts) - n_done)
            print(
                f"  sample {n_done}/{len(texts)}  "
                f"elapsed={elapsed:6.1f}s  "
                f"avg/sample={avg:5.1f}s  "
                f"eta={remaining:6.1f}s"
            )

    streams = (
        np.stack(streams_for_omega, axis=0)
        if len(streams_for_omega) > 0 else None
    )
    return {
        "block_indices": block_indices,
        "eigvals_attn": eigvals_attn,
        "eigvals_mlp": eigvals_mlp,
        "streams": streams,
    }


# ---------------------------------------------------------------------------
# omega estimation (FFT along the depth axis)
# ---------------------------------------------------------------------------

def estimate_omega(streams, trim=2):
    """
    streams: (N, n_sub, D) residual stream snapshots at the last token.
    Returns (omega_per_unit, omega_median, omega_mode), all in radians per
    sublayer. omega_per_unit is the dominant non-DC FFT bin per unit,
    averaged over samples by argmax of summed power.
    """
    if trim > 0 and streams.shape[1] > 2 * trim + 2:
        streams = streams[:, trim:streams.shape[1] - trim, :]
    s = streams - streams.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(s, axis=1)
    power = (np.abs(spec) ** 2).sum(axis=0)  # (n_freqs, D)
    if power.shape[0] > 1:
        power[0, :] = 0  # drop DC
    L = streams.shape[1]
    bins = power.argmax(axis=0)
    omega_per_unit = 2 * np.pi * bins / L
    omega_med = float(np.median(omega_per_unit))

    # modal omega across units (binned histogram)
    if len(omega_per_unit) > 0:
        unique_bins, counts = np.unique(bins, return_counts=True)
        mode_bin = unique_bins[np.argmax(counts)]
        omega_mode = float(2 * np.pi * mode_bin / L)
    else:
        omega_mode = float("nan")
    return omega_per_unit, omega_med, omega_mode


# ---------------------------------------------------------------------------
# Spectrum aggregation
# ---------------------------------------------------------------------------

def top_eigs(ev, k=2):
    """Top-k eigenvalues by |Im|. Returns shape (k,) complex."""
    if len(ev) == 0:
        return np.zeros(k, dtype=complex)
    idx = np.argsort(-np.abs(ev.imag))[:k]
    return ev[idx]


def aggregate_top(eig_dict, k=2):
    """Return shape (n_total, k) complex array of top eigenvalues."""
    rows = []
    for bi in sorted(eig_dict.keys()):
        for ev in eig_dict[bi]:
            rows.append(top_eigs(ev, k=k))
    if len(rows) == 0:
        return np.zeros((0, k), dtype=complex)
    return np.stack(rows, axis=0)


def per_layer_top(eig_dict, k=2):
    """
    Per-block summaries of top |Im| and |Re|. Returns three np arrays:
    block indices, median |Im(top)| per block, median |Re(top)| per block.
    """
    bis = sorted(eig_dict.keys())
    im_med = []
    re_med = []
    for bi in bis:
        if len(eig_dict[bi]) == 0:
            im_med.append(np.nan)
            re_med.append(np.nan)
            continue
        tops = np.stack(
            [top_eigs(ev, k=k) for ev in eig_dict[bi]], axis=0,
        )
        im_med.append(float(np.median(np.abs(tops.imag))))
        re_med.append(float(np.median(np.abs(tops.real))))
    return np.array(bis), np.array(im_med), np.array(re_med)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_eig_scatter(eig_dict, omega_pred, label, sublayer_name, out_path,
                     max_blocks=8):
    """Per-block eigenvalue scatter in the complex plane."""
    block_indices = sorted(eig_dict.keys())
    if len(block_indices) > max_blocks:
        idx = np.linspace(0, len(block_indices) - 1, max_blocks, dtype=int)
        block_indices = [block_indices[i] for i in idx]

    n = len(block_indices)
    if n == 0:
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False,
    )

    for ax, bi in zip(axes.ravel(), block_indices):
        if len(eig_dict[bi]) == 0:
            ax.set_title(f"Block {bi} (no data)")
            continue
        evs = np.concatenate(eig_dict[bi])
        ax.scatter(evs.real, evs.imag, s=2, alpha=0.25, color="C0")
        ax.axhline(omega_pred, color="red", linestyle="--", alpha=0.6,
                   linewidth=1.0)
        ax.axhline(-omega_pred, color="red", linestyle="--", alpha=0.6,
                   linewidth=1.0)
        ax.axvline(0, color="black", linestyle=":", alpha=0.4, linewidth=1.0)
        ax.axhline(0, color="black", linestyle=":", alpha=0.4, linewidth=1.0)
        ax.set_title(f"Block {bi}")
        ax.set_xlabel("Re(eigval)")
        ax.set_ylabel("Im(eigval)")

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(
        f"{sublayer_name} sublayer eigenvalues, {label}\n"
        f"red dashed: predicted +-omega = +-{omega_pred:.3f} rad/sublayer"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_imag_dist(res_trained, res_init, omega_pred_trained,
                   omega_pred_init, out_path):
    """
    |Im(top eigenvalue)| histograms, attention vs MLP, trained vs init.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in [(axes[0], "eigvals_attn", "Attention sublayer"),
                            (axes[1], "eigvals_mlp", "MLP sublayer")]:
        top_t = aggregate_top(res_trained[key], k=2)
        top_i = aggregate_top(res_init[key], k=2)

        if top_t.shape[0] > 0:
            im_t = np.abs(top_t.imag).ravel()
            ax.hist(im_t, bins=60, alpha=0.55, density=True,
                    color="C0", label="trained")
        if top_i.shape[0] > 0:
            im_i = np.abs(top_i.imag).ravel()
            ax.hist(im_i, bins=60, alpha=0.55, density=True,
                    color="C3", label="random init")

        ax.axvline(omega_pred_trained, color="C0", linestyle="--",
                   linewidth=1.2,
                   label=f"omega trained = {omega_pred_trained:.3f}")
        if not np.isnan(omega_pred_init):
            ax.axvline(omega_pred_init, color="C3", linestyle="--",
                       linewidth=1.2,
                       label=f"omega init = {omega_pred_init:.3f}")
        ax.set_xlabel("|Im(top eigenvalue)|  (rad/sublayer)")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "|Im| of top-by-|Im| eigenvalues, all blocks pooled\n"
        "substrate hypothesis: trained mass concentrated near omega"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_real_dist(res_trained, res_init, out_path):
    """Re(top) distribution; substrate predicts trained near 0."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in [(axes[0], "eigvals_attn", "Attention sublayer"),
                            (axes[1], "eigvals_mlp", "MLP sublayer")]:
        top_t = aggregate_top(res_trained[key], k=2)
        top_i = aggregate_top(res_init[key], k=2)

        if top_t.shape[0] > 0:
            re_t = top_t.real.ravel()
            ax.hist(re_t, bins=60, alpha=0.55, density=True,
                    color="C0", label="trained")
        if top_i.shape[0] > 0:
            re_i = top_i.real.ravel()
            ax.hist(re_i, bins=60, alpha=0.55, density=True,
                    color="C3", label="random init")
        ax.axvline(0, color="black", linestyle=":", alpha=0.7)
        ax.set_xlabel("Re(top eigenvalue)")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Re of top eigenvalues\n"
        "substrate hypothesis: trained mass concentrated near 0"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_layer_summary(res_trained, res_init,
                           omega_pred_trained, omega_pred_init, out_path):
    """
    Per-block median |Im(top)| and median |Re(top)|, both sublayers,
    both checkpoints.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, key, title in [(0, "eigvals_attn", "Attention"),
                             (1, "eigvals_mlp", "MLP")]:
        bi_t, im_t, re_t = per_layer_top(res_trained[key], k=2)
        bi_i, im_i, re_i = per_layer_top(res_init[key], k=2)

        ax = axes[0, col]
        ax.plot(bi_t, im_t, "-o", color="C0", label="trained", markersize=4)
        ax.plot(bi_i, im_i, "-o", color="C3", label="random init",
                markersize=4)
        ax.axhline(omega_pred_trained, color="C0", linestyle="--",
                   linewidth=1.0, alpha=0.7,
                   label=f"omega trained = {omega_pred_trained:.3f}")
        if not np.isnan(omega_pred_init):
            ax.axhline(omega_pred_init, color="C3", linestyle="--",
                       linewidth=1.0, alpha=0.7,
                       label=f"omega init = {omega_pred_init:.3f}")
        ax.set_ylabel("median |Im(top)|")
        ax.set_title(f"{title}: |Im(top)| vs depth")
        ax.legend(fontsize=8, loc="best")

        ax = axes[1, col]
        ax.plot(bi_t, re_t, "-o", color="C0", label="trained", markersize=4)
        ax.plot(bi_i, re_i, "-o", color="C3", label="random init",
                markersize=4)
        ax.axhline(0, color="black", linestyle=":", alpha=0.5)
        ax.set_xlabel("block index")
        ax.set_ylabel("median |Re(top)|")
        ax.set_title(f"{title}: |Re(top)| vs depth")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Per-block top-eigenvalue summary, trained vs random init"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(res_trained, res_init, omega_t, omega_i, model_name,
                  n_block_subset, n_samples_used):
    def per_check(res, omega_pred):
        out = {}
        for key in ("eigvals_attn", "eigvals_mlp"):
            top = aggregate_top(res[key], k=2)
            if top.shape[0] == 0:
                out[key] = {"n": 0}
                continue
            im_top = np.abs(top.imag).ravel()
            re_top = top.real.ravel()
            n_in_band = int(((im_top > 0.5 * omega_pred)
                             & (im_top < 1.5 * omega_pred)).sum())
            out[key] = {
                "n_top_eigvals": int(im_top.size),
                "abs_imag_top": {
                    "mean": float(im_top.mean()),
                    "median": float(np.median(im_top)),
                    "std": float(im_top.std()),
                },
                "real_top": {
                    "mean": float(re_top.mean()),
                    "median": float(np.median(re_top)),
                    "abs_mean": float(np.abs(re_top).mean()),
                    "abs_median": float(np.median(np.abs(re_top))),
                    "std": float(re_top.std()),
                },
                "fraction_in_omega_band": (
                    n_in_band / im_top.size
                ),
            }
        return out

    return {
        "model": model_name,
        "n_block_subset": n_block_subset,
        "n_samples_used": int(n_samples_used),
        "omega_predicted": {
            "trained_median_rad_per_sublayer": float(omega_t),
            "random_init_median_rad_per_sublayer": float(omega_i),
        },
        "per_checkpoint": {
            "trained": per_check(res_trained, omega_t),
            "random_init": per_check(res_init, omega_t),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model id.")
    parser.add_argument("--n-samples", type=int, default=16,
                        help="Number of texts (Jacobians are expensive).")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Max tokens per text.")
    parser.add_argument("--n-block-subset", type=int, default=None,
                        help="Analyze only this many evenly-spaced blocks. "
                             "Default: all.")
    parser.add_argument("--trim-sublayers", type=int, default=2,
                        help="Trim this many sublayers from each end of the "
                             "depth-axis stream when estimating omega.")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--skip-init", action="store_true",
                        help="Skip the random-init baseline (faster smoke test).")
    args = parser.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(f"linearization_eigvals_{slugify(args.model)}"))
    out_dir.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("Loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]

    print(f"Loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Trained ----
    print(f"\n=== TRAINED ({args.model}) ===")
    trained_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(DEVICE)
    res_trained = analyze_checkpoint(
        trained_model, tokenizer, texts, args.seq_len,
        n_block_subset=args.n_block_subset,
    )
    del trained_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if res_trained["streams"] is None:
        raise RuntimeError("No streams collected from trained model.")
    omega_t_per_unit, omega_t_med, omega_t_mode = estimate_omega(
        res_trained["streams"], trim=args.trim_sublayers,
    )
    print(f"\n[trained] median omega = {omega_t_med:.4f} rad/sublayer "
          f"(mode {omega_t_mode:.4f})")

    # ---- Random init ----
    if args.skip_init:
        res_init = {
            "block_indices": res_trained["block_indices"],
            "eigvals_attn": {bi: [] for bi in res_trained["block_indices"]},
            "eigvals_mlp": {bi: [] for bi in res_trained["block_indices"]},
            "streams": None,
        }
        omega_i_med = float("nan")
    else:
        print(f"\n=== RANDOM INIT ===")
        config = AutoConfig.from_pretrained(args.model)
        init_model = AutoModelForCausalLM.from_config(config).to(DEVICE).float()
        res_init = analyze_checkpoint(
            init_model, tokenizer, texts, args.seq_len,
            n_block_subset=args.n_block_subset,
        )
        del init_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if res_init["streams"] is not None:
            _, omega_i_med, _ = estimate_omega(
                res_init["streams"], trim=args.trim_sublayers,
            )
        else:
            omega_i_med = float("nan")
        print(f"\n[random init] median omega = {omega_i_med:.4f} rad/sublayer")

    # ---- Plots ----
    print("\nMaking plots ...")
    plot_eig_scatter(
        res_trained["eigvals_attn"], omega_t_med, "trained",
        "attention", out_dir / "eigvalue_scatter_attn_trained.png",
    )
    plot_eig_scatter(
        res_trained["eigvals_mlp"], omega_t_med, "trained",
        "MLP", out_dir / "eigvalue_scatter_mlp_trained.png",
    )
    if not args.skip_init:
        plot_eig_scatter(
            res_init["eigvals_attn"],
            omega_i_med if not np.isnan(omega_i_med) else omega_t_med,
            "random init", "attention",
            out_dir / "eigvalue_scatter_attn_random_init.png",
        )
        plot_eig_scatter(
            res_init["eigvals_mlp"],
            omega_i_med if not np.isnan(omega_i_med) else omega_t_med,
            "random init", "MLP",
            out_dir / "eigvalue_scatter_mlp_random_init.png",
        )
    plot_imag_dist(res_trained, res_init, omega_t_med, omega_i_med,
                   out_dir / "imag_distribution.png")
    plot_real_dist(res_trained, res_init,
                   out_dir / "real_distribution.png")
    plot_per_layer_summary(res_trained, res_init,
                           omega_t_med, omega_i_med,
                           out_dir / "per_layer_summary.png")

    # ---- Summary ----
    summary = build_summary(
        res_trained, res_init, omega_t_med, omega_i_med,
        args.model, args.n_block_subset, len(texts),
    )
    with open(out_dir / "eigvals_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Console table ----
    print("\n--- Substrate eigenvalue signature ---")
    print(f"{'checkpoint':<14} {'sublayer':<6} "
          f"{'<|Im(top)|>':>14} {'<|Re(top)|>':>14} "
          f"{'frac in omega band':>22}")
    for label, m in (("trained", summary["per_checkpoint"]["trained"]),
                     ("random_init", summary["per_checkpoint"]["random_init"])):
        for sub_key, sub_name in (("eigvals_attn", "attn"),
                                   ("eigvals_mlp", "mlp")):
            entry = m.get(sub_key, {})
            if entry.get("n_top_eigvals", 0) == 0:
                continue
            print(
                f"{label:<14} {sub_name:<6} "
                f"{entry['abs_imag_top']['mean']:>14.4f} "
                f"{entry['real_top']['abs_mean']:>14.4f} "
                f"{entry['fraction_in_omega_band']:>22.3f}"
            )

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
