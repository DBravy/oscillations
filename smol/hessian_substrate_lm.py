"""
Hessian-and-substrate analysis for one transformer block.

Tests the framework's prediction that high-curvature directions in the
loss landscape correspond to substrate-disrupting parameter perturbations.

Pipeline (per block analyzed):

  1. Capture residual stream + self_attn args at one operating point.
  2. Compute J_attn = d(attn_out_at_last_token)/d(x_at_last_token)
     using vectorized autograd Jacobian (fast).
  3. Eigendecompose J_attn. Identify the substrate eigenvalue: the
     eigenvalue with largest |Im| inside the band [0.5*omega, 1.5*omega].
     Extract its right eigenvector v_0 and left eigenvector u_0
     (left = right of J^T at the same eigenvalue), normalized so
     u_0^T v_0 = 1.
  4. Build a small batch cross-entropy loss. Define a Hessian-vector
     product (HVP) operator restricted to the block's attention weights
     (q_proj, k_proj, v_proj, o_proj) via Pearlmutter's trick.
  5. Run scipy Lanczos (eigsh) for top-k Hessian eigenpairs.
  6. For each Hessian eigenvector and each random unit direction,
     perturb theta -> theta + eps * v, compute lambda_perturbed via
     first-order tracking:
         lambda_pert ~ u_0^T J(theta + eps v) v_0
                     = u_0^T jvp(f_attn, x_0, v_0)
     (one forward pass, no full Jacobian needed). Record |delta lambda|.
  7. Plot Hessian eigenvalue vs |delta lambda|, with random directions
     overlaid. Save raw arrays to JSON + npz for reanalysis.

Speed (SmolLM2-360M, one block, 4-sequence batch):
  capture + J_attn      ~1 sec
  Lanczos top-20        ~15 sec
  40 perturbation evals ~5 sec
  total per block       ~25 sec

Usage:
  python hessian_substrate_lm.py --model HuggingFaceTB/SmolLM2-360M \\
      --block-idx 16 --omega 0.105

  # Sweep multiple blocks:
  python hessian_substrate_lm.py --block-idx 4,12,20,28
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigsh
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


# ---------------------------------------------------------------------------
# Block discovery (matches phase_space_llm.py / linearization_eigvals_lm.py)
# ---------------------------------------------------------------------------

def get_block_info(model):
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
# Capture residual stream + attention call args at one operating point
# ---------------------------------------------------------------------------

def capture_one_sample(model, tokenizer, text, seq_len, target_block_idx):
    """Forward pass with hooks; capture x_attn_in and self_attn args/kwargs
    for the target block. Returns dict or None if too short."""
    blocks, ln1_name, _ln2_name, attn_name, _ = get_block_info(model)

    enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
    ids = enc["input_ids"].to(DEVICE)
    if ids.shape[1] < 8:
        return None

    captured = {"x_attn_in": None, "attn_args": None, "attn_kwargs": None}
    blk = blocks[target_block_idx]
    ln1 = getattr(blk, ln1_name)
    attn = getattr(blk, attn_name)

    handles = []

    def ln_hook(module, inp):
        if isinstance(inp, tuple) and len(inp) > 0 and torch.is_tensor(inp[0]):
            captured["x_attn_in"] = inp[0].detach().clone()
    handles.append(ln1.register_forward_pre_hook(ln_hook))

    def attn_hook(module, args, kwargs):
        captured["attn_args"] = args
        captured["attn_kwargs"] = kwargs
        return None
    handles.append(attn.register_forward_pre_hook(attn_hook, with_kwargs=True))

    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()

    if captured["x_attn_in"] is None:
        return None
    return {"input_ids": ids, **captured}


def _build_attn_call(captured, x_norm):
    """Reconstruct self_attn call with hidden_states substituted by x_norm."""
    raw_args = captured.get("attn_args") or ()
    raw_kwargs = dict(captured.get("attn_kwargs") or {})
    for k in ("past_key_value", "past_key_values"):
        if k in raw_kwargs:
            raw_kwargs[k] = None
    for k, v in (("use_cache", False), ("output_attentions", False)):
        if k in raw_kwargs:
            raw_kwargs[k] = v

    hs_in_args = (len(raw_args) > 0
                  and torch.is_tensor(raw_args[0])
                  and raw_args[0].dim() == 3)
    if hs_in_args:
        return (x_norm,) + tuple(raw_args[1:]), raw_kwargs
    raw_kwargs["hidden_states"] = x_norm
    return tuple(raw_args), raw_kwargs


# ---------------------------------------------------------------------------
# Substrate eigenvector extraction
# ---------------------------------------------------------------------------

def attn_output_last(block, captured, ln1_name, attn_name, x_last):
    """f_attn at the last token, given a (D,) input vector for the last position
    and frozen earlier-token context. Returns (D,) attention output vector."""
    ln1 = getattr(block, ln1_name)
    attn = getattr(block, attn_name)

    x_full_f32 = captured["x_attn_in"].to(torch.float32)
    x = x_full_f32.clone()
    x[0, -1, :] = x_last
    x_norm = ln1(x)

    target_dtype = next(attn.parameters()).dtype
    if x_norm.dtype != target_dtype:
        x_norm = x_norm.to(target_dtype)
    args, kwargs = _build_attn_call(captured, x_norm)
    out = attn(*args, **kwargs)
    attn_out = out[0] if isinstance(out, tuple) else out
    return attn_out[0, -1, :].to(torch.float32)


def compute_J_attn(block, captured, ln1_name, attn_name):
    """Vectorized Jacobian of f_attn w.r.t. last-token input. Falls back to
    non-vectorized if vectorize=True fails."""
    x_last_init = captured["x_attn_in"][0, -1, :].detach().clone().to(torch.float32)

    def f(x_last):
        return attn_output_last(block, captured, ln1_name, attn_name, x_last)

    try:
        J = torch.autograd.functional.jacobian(
            f, x_last_init, vectorize=True, create_graph=False,
        )
    except Exception as e:
        print(f"  vectorize=True failed ({type(e).__name__}); falling back.")
        J = torch.autograd.functional.jacobian(
            f, x_last_init, vectorize=False, create_graph=False,
        )
    return J.detach().cpu().numpy().astype(np.float64)


def find_substrate_eigenpair(J, omega_target):
    """Pick the eigenvalue with largest |Im| inside [0.5*omega, 1.5*omega].
    Return (lambda, v_right, v_left) with biorthogonal normalization
    v_left^T v_right = 1. v_right and v_left are complex (D,)."""
    eigvals, eigvecs_R = np.linalg.eig(J)        # right eigenvectors
    eigvals_T, eigvecs_L_T = np.linalg.eig(J.T)  # left = right of J^T

    im_abs = np.abs(eigvals.imag)
    in_band = (im_abs > 0.5 * omega_target) & (im_abs < 1.5 * omega_target)
    if not in_band.any():
        # fall back to overall top by |Im|
        idx_R = int(np.argmax(im_abs))
        print(f"  warning: no eigenvalue in [{0.5*omega_target:.3f}, "
              f"{1.5*omega_target:.3f}]; using overall top by |Im|.")
    else:
        candidates = np.where(in_band)[0]
        idx_R = candidates[np.argmax(im_abs[candidates])]

    lam = eigvals[idx_R]
    vR = eigvecs_R[:, idx_R]

    # Match to the corresponding left eigenvalue (closest match)
    diffs = np.abs(eigvals_T - lam)
    idx_L = int(np.argmin(diffs))
    vL = eigvecs_L_T[:, idx_L]

    # Biorthogonal normalization: vL^T vR = 1
    inner = vL @ vR
    if abs(inner) < 1e-12:
        raise RuntimeError("Left/right eigenvector inner product near zero; "
                           "eigenvalue may be defective.")
    vL = vL / inner
    return lam, vR, vL


# ---------------------------------------------------------------------------
# Loss + HVP for one block's attention parameters
# ---------------------------------------------------------------------------

def collect_loss_batch(model, tokenizer, texts, seq_len, batch_size):
    """Tokenize a small batch and stack into (B, T) of equal length T (padded)."""
    rows = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt",
                        max_length=seq_len, truncation=True)
        ids = enc["input_ids"][0]
        if ids.numel() < 8:
            continue
        rows.append(ids[:seq_len])
        if len(rows) >= batch_size:
            break
    if len(rows) == 0:
        raise RuntimeError("No batch could be built.")
    T = max(r.numel() for r in rows)
    pad_id = tokenizer.pad_token_id
    batch = torch.full((len(rows), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), T), dtype=torch.bool)
    for i, r in enumerate(rows):
        batch[i, :r.numel()] = r
        mask[i, :r.numel()] = True
    return batch.to(DEVICE), mask.to(DEVICE)


def block_attn_params(block, attn_name):
    """Return the list of weight tensors we differentiate the Hessian over.
    We use only the .weight of q/k/v/o projections."""
    attn = getattr(block, attn_name)
    names = ["q_proj", "k_proj", "v_proj", "o_proj"]  # llama-style
    if not hasattr(attn, names[0]):
        # GPT-2 style: c_attn (qkv combined) and c_proj
        names = ["c_attn", "c_proj"]
    params = []
    param_names = []
    for n in names:
        if not hasattr(attn, n):
            continue
        sub = getattr(attn, n)
        if hasattr(sub, "weight") and sub.weight is not None:
            params.append(sub.weight)
            param_names.append(f"{n}.weight")
    if len(params) == 0:
        raise RuntimeError("No attention weights found.")
    return params, param_names


def make_loss_fn(model, batch_ids, mask):
    """Cross-entropy on next-token prediction averaged over masked positions."""
    def loss_fn():
        # Shifted CE: predict batch_ids[:, 1:] from batch_ids[:, :-1].
        out = model(batch_ids).logits  # (B, T, V)
        logits = out[:, :-1, :].contiguous()
        targets = batch_ids[:, 1:].contiguous()
        m = mask[:, 1:].contiguous()
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        loss = (nll * m.float()).sum() / m.float().sum().clamp(min=1.0)
        return loss
    return loss_fn


def make_hvp(loss_fn, params):
    """Pearlmutter HVP. Returns flat numpy callable for scipy."""
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]
    total = int(sum(sizes))

    def hvp_flat(v_np):
        v_t = torch.from_numpy(v_np.astype(np.float32)).to(DEVICE)
        v_list = []
        i = 0
        for sz, sh in zip(sizes, shapes):
            v_list.append(v_t[i:i + sz].view(sh))
            i += sz

        for p in params:
            p.requires_grad_(True)

        # Use MATH backend because flash-attention CPU kernel lacks higher-order grad
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            loss = loss_fn()
            g = torch.autograd.grad(loss, params, create_graph=True)
            g_dot_v = sum((gi * vi).sum() for gi, vi in zip(g, v_list))
            Hv = torch.autograd.grad(g_dot_v, params, retain_graph=False)

        out = torch.cat([h.reshape(-1) for h in Hv]).detach().cpu().numpy()
        return out.astype(np.float64)

    return hvp_flat, total, shapes, sizes


# ---------------------------------------------------------------------------
# First-order eigenvalue tracking
# ---------------------------------------------------------------------------

def lambda_first_order(block, captured, ln1_name, attn_name, vR, vL):
    """
    Return current u_0^T J(theta) v_0 evaluated via two real jvps.

    vR = vR_re + i vR_im  (complex right eigenvector)
    vL = vL_re + i vL_im  (complex left eigenvector)

    For real J:
      u^T J v = (uR + i uI)^T (J vR + i J vI)
             = uR^T J vR - uI^T J vI + i (uR^T J vI + uI^T J vR)
    """
    x_last_init = captured["x_attn_in"][0, -1, :].detach().clone().to(torch.float32)
    vR_re = torch.from_numpy(np.ascontiguousarray(vR.real)).to(
        DEVICE, dtype=torch.float32)
    vR_im = torch.from_numpy(np.ascontiguousarray(vR.imag)).to(
        DEVICE, dtype=torch.float32)
    vL_re = torch.from_numpy(np.ascontiguousarray(vL.real)).to(
        DEVICE, dtype=torch.float32)
    vL_im = torch.from_numpy(np.ascontiguousarray(vL.imag)).to(
        DEVICE, dtype=torch.float32)

    def f(x_last):
        return attn_output_last(block, captured, ln1_name, attn_name, x_last)

    # jvp returns (output, jvp_value)
    # Use MATH backend because flash-attention CPU kernel lacks higher-order grad support
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        _, JvR = torch.autograd.functional.jvp(f, x_last_init, vR_re,
                                                create_graph=False, strict=False)
        _, JvI = torch.autograd.functional.jvp(f, x_last_init, vR_im,
                                                create_graph=False, strict=False)
    JvR = JvR.detach()
    JvI = JvI.detach()

    re = (vL_re @ JvR).item() - (vL_im @ JvI).item()
    im = (vL_re @ JvI).item() + (vL_im @ JvR).item()
    return complex(re, im)


# ---------------------------------------------------------------------------
# Per-block analysis
# ---------------------------------------------------------------------------

def analyze_block(model, tokenizer, texts, target_block_idx,
                  seq_len, batch_size, omega_target, n_hess, n_random,
                  eps_list, hvp_lanczos_iters):
    blocks, ln1_name, _ln2_name, attn_name, _ = get_block_info(model)
    block = blocks[target_block_idx]

    # ---- 1. Capture operating point ----
    print(f"  [block {target_block_idx}] capturing operating point ...")
    captured = None
    for text in texts:
        cap = capture_one_sample(model, tokenizer, text, seq_len, target_block_idx)
        if cap is not None:
            captured = cap
            break
    if captured is None:
        raise RuntimeError("No usable text for capture.")

    # ---- 2. Compute J_attn ----
    t0 = time.time()
    J = compute_J_attn(block, captured, ln1_name, attn_name)
    print(f"  [block {target_block_idx}] J_attn shape {J.shape}, "
          f"computed in {time.time()-t0:.2f}s")

    # ---- 3. Substrate eigenpair ----
    lam0, vR, vL = find_substrate_eigenpair(J, omega_target)
    print(f"  [block {target_block_idx}] substrate eigenvalue: "
          f"{lam0.real:+.4f} {lam0.imag:+.4f}j   "
          f"(target |Im|={omega_target:.4f})")

    # Sanity check via direct evaluation:
    lam0_check = lambda_first_order(block, captured, ln1_name, attn_name, vR, vL)
    print(f"  [block {target_block_idx}] u^T J v (via jvp) = "
          f"{lam0_check.real:+.4f} {lam0_check.imag:+.4f}j   "
          f"(should match within ~1e-3)")

    # ---- 4. Loss + HVP ----
    print(f"  [block {target_block_idx}] building loss batch ...")
    batch_ids, mask = collect_loss_batch(
        model, tokenizer, texts, seq_len, batch_size,
    )
    loss_fn = make_loss_fn(model, batch_ids, mask)

    # Freeze everything; unfreeze only the block's attn weights
    for p in model.parameters():
        p.requires_grad_(False)
    params, param_names = block_attn_params(block, attn_name)
    for p in params:
        p.requires_grad_(True)
    n_params_block = sum(p.numel() for p in params)
    print(f"  [block {target_block_idx}] HVP over {n_params_block:,} "
          f"params: {param_names}")

    hvp_flat, total, shapes, sizes = make_hvp(loss_fn, params)

    # ---- 5. Lanczos top-k ----
    print(f"  [block {target_block_idx}] Lanczos top-{n_hess} ...")
    op = LinearOperator((total, total), matvec=hvp_flat, dtype=np.float64)
    t0 = time.time()
    eigvals_H, eigvecs_H = eigsh(
        op, k=n_hess, which="LA", tol=1e-3,
        maxiter=max(hvp_lanczos_iters, 4 * n_hess),
    )
    # Sort descending
    order = np.argsort(-eigvals_H)
    eigvals_H = eigvals_H[order]
    eigvecs_H = eigvecs_H[:, order]
    print(f"  [block {target_block_idx}] Lanczos done in {time.time()-t0:.1f}s; "
          f"top eig: {eigvals_H[0]:.3e}, smallest of top-{n_hess}: "
          f"{eigvals_H[-1]:.3e}")

    # ---- 6. Random directions ----
    rng = np.random.default_rng(SEED + target_block_idx)
    rand_dirs = rng.standard_normal((n_random, total)).astype(np.float64)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)

    # ---- 7. Perturbation sweep ----
    print(f"  [block {target_block_idx}] perturbation sweep ...")
    flat_params0 = [p.detach().clone() for p in params]

    def apply_perturbation(direction_flat, eps):
        i = 0
        with torch.no_grad():
            for p, sz, sh in zip(params, sizes, shapes):
                d = torch.from_numpy(
                    direction_flat[i:i + sz].astype(np.float32),
                ).to(DEVICE).view(sh)
                p.add_(eps * d)
                i += sz

    def restore_params():
        with torch.no_grad():
            for p, p0 in zip(params, flat_params0):
                p.copy_(p0)

    results = {
        "hessian": {"eigvals": eigvals_H.tolist(), "delta_lambda": {}},
        "random": {"delta_lambda": {}},
    }

    for eps in eps_list:
        d_lam_H = []
        for k in range(n_hess):
            apply_perturbation(eigvecs_H[:, k], eps)
            lam_p = lambda_first_order(
                block, captured, ln1_name, attn_name, vR, vL,
            )
            d_lam_H.append(lam_p - lam0)
            restore_params()
        d_lam_R = []
        for k in range(n_random):
            apply_perturbation(rand_dirs[k], eps)
            lam_p = lambda_first_order(
                block, captured, ln1_name, attn_name, vR, vL,
            )
            d_lam_R.append(lam_p - lam0)
            restore_params()

        results["hessian"]["delta_lambda"][f"eps={eps:.0e}"] = [
            (z.real, z.imag) for z in d_lam_H
        ]
        results["random"]["delta_lambda"][f"eps={eps:.0e}"] = [
            (z.real, z.imag) for z in d_lam_R
        ]

    # ---- Save raw J eigenvalues for boundary reanalysis ----
    raw_eigvals = np.linalg.eigvals(J)

    return {
        "block_idx": target_block_idx,
        "n_params_block": n_params_block,
        "param_names": param_names,
        "lambda_0": (lam0.real, lam0.imag),
        "lambda_0_jvp_check": (lam0_check.real, lam0_check.imag),
        "hessian_eigvals": eigvals_H.tolist(),
        "perturbations": results,
        "eps_list": list(eps_list),
        "J_raw_eigvals_real": raw_eigvals.real.tolist(),
        "J_raw_eigvals_imag": raw_eigvals.imag.tolist(),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_one_block(res, omega_target, out_path):
    bi = res["block_idx"]
    eigvals_H = np.array(res["hessian_eigvals"])
    n_hess = len(eigvals_H)
    eps_list = res["eps_list"]
    perts = res["perturbations"]

    # Pick the smallest eps for the linear-regime plot.
    eps_use = eps_list[0]
    key = f"eps={eps_use:.0e}"
    d_H = np.array(perts["hessian"]["delta_lambda"][key])
    d_R = np.array(perts["random"]["delta_lambda"][key])
    abs_dH = np.sqrt(d_H[:, 0] ** 2 + d_H[:, 1] ** 2)
    abs_dR = np.sqrt(d_R[:, 0] ** 2 + d_R[:, 1] ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # (1) Hessian eigvalue vs |delta lambda|
    ax = axes[0]
    ax.scatter(np.arange(n_hess), abs_dH / eps_use, color="C0",
               label="top-k Hessian")
    ax.scatter(np.arange(len(abs_dR)), abs_dR / eps_use,
               color="C3", marker="x", label="random unit dirs")
    ax.set_yscale("log")
    ax.set_xlabel("direction index (Hessian: largest -> smallest)")
    ax.set_ylabel("|delta lambda| / eps")
    ax.set_title(f"Block {bi}: substrate disruption per direction")
    ax.legend(fontsize=9, loc="best")

    # (2) Hessian eigvals vs |delta lambda| / eps
    ax = axes[1]
    ax.loglog(eigvals_H, abs_dH / eps_use, "o", color="C0")
    if eigvals_H.min() > 0 and abs_dH.min() > 0:
        # rough power-law guide line
        x_ref = np.array([eigvals_H.min(), eigvals_H.max()])
        # fit log-log
        lx = np.log(eigvals_H)
        ly = np.log((abs_dH / eps_use).clip(min=1e-30))
        slope, intercept = np.polyfit(lx, ly, 1)
        ax.loglog(x_ref, np.exp(intercept) * x_ref ** slope, "--",
                  color="gray", linewidth=1.0,
                  label=f"slope = {slope:.2f}")
        ax.legend(fontsize=9, loc="best")
    ax.set_xlabel("Hessian eigenvalue")
    ax.set_ylabel("|delta lambda| / eps")
    ax.set_title(f"Block {bi}: curvature vs disruption")

    # (3) Linearity check across eps values (median ratio).
    ax = axes[2]
    for eps in eps_list:
        key = f"eps={eps:.0e}"
        d_H = np.array(perts["hessian"]["delta_lambda"][key])
        abs_dH_e = np.sqrt(d_H[:, 0] ** 2 + d_H[:, 1] ** 2) / eps
        ax.plot(np.arange(n_hess), abs_dH_e, "-o", markersize=3,
                label=f"eps={eps:.0e}")
    ax.set_yscale("log")
    ax.set_xlabel("Hessian direction index")
    ax.set_ylabel("|delta lambda| / eps")
    ax.set_title(f"Block {bi}: linearity check")
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"Hessian-substrate analysis, block {bi}\n"
        f"lambda_0 = {res['lambda_0'][0]:+.4f} + {res['lambda_0'][1]:+.4f}j   "
        f"(omega target = {omega_target:.4f})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_summary_across_blocks(all_res, omega_target, out_path):
    """One-panel summary: median Hessian-eigvec |delta lambda| vs median
    random |delta lambda| per block."""
    blocks = sorted(r["block_idx"] for r in all_res)
    by_idx = {r["block_idx"]: r for r in all_res}
    med_H, med_R = [], []
    for bi in blocks:
        r = by_idx[bi]
        eps_use = r["eps_list"][0]
        key = f"eps={eps_use:.0e}"
        d_H = np.array(r["perturbations"]["hessian"]["delta_lambda"][key])
        d_R = np.array(r["perturbations"]["random"]["delta_lambda"][key])
        abs_dH = np.sqrt(d_H[:, 0] ** 2 + d_H[:, 1] ** 2) / eps_use
        abs_dR = np.sqrt(d_R[:, 0] ** 2 + d_R[:, 1] ** 2) / eps_use
        med_H.append(np.median(abs_dH))
        med_R.append(np.median(abs_dR))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(blocks, med_H, "-o", color="C0", label="median |dl|/eps, top Hessian")
    ax.plot(blocks, med_R, "-o", color="C3", label="median |dl|/eps, random")
    ax.set_yscale("log")
    ax.set_xlabel("block index")
    ax.set_ylabel("|delta lambda| / eps")
    ax.set_title(
        "Substrate disruption: top-Hessian vs random directions\n"
        "ratio = strength of the curvature/substrate alignment"
    )
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--block-idx", type=str, default=None,
                        help="Comma-separated list of block indices to analyze. "
                             "Default: middle block.")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Texts to consider when picking a capture sample "
                             "and building the loss batch.")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for the loss / HVP.")
    parser.add_argument("--omega", type=float, default=0.105,
                        help="Target carrier frequency (rad/sublayer). For "
                             "SmolLM2-360M the previous run gave ~0.105.")
    parser.add_argument("--n-hess", type=int, default=20,
                        help="Top-k Hessian eigenvalues via Lanczos.")
    parser.add_argument("--n-random", type=int, default=20,
                        help="Random unit-norm directions for comparison.")
    parser.add_argument("--eps", type=str, default="1e-4,1e-3,1e-2",
                        help="Comma-separated perturbation magnitudes.")
    parser.add_argument("--lanczos-iters", type=int, default=80)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(f"hessian_substrate_{slugify(args.model)}"))
    out_dir.mkdir(exist_ok=True, parents=True)

    eps_list = [float(s) for s in args.eps.split(",")]

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

    print(f"Loading model ({args.model}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(DEVICE)
    model.eval()
    blocks, _, _, _, _ = get_block_info(model)
    n_blocks = len(blocks)

    if args.block_idx is None:
        block_indices = [n_blocks // 2]
    else:
        block_indices = [int(s) for s in args.block_idx.split(",")]
    print(f"Analyzing blocks {block_indices} of {n_blocks}.")

    all_res = []
    t_total = time.time()
    for bi in block_indices:
        if bi < 0 or bi >= n_blocks:
            print(f"  block {bi} out of range; skipping.")
            continue
        t_b = time.time()
        try:
            res = analyze_block(
                model, tokenizer, texts, bi,
                seq_len=args.seq_len, batch_size=args.batch_size,
                omega_target=args.omega, n_hess=args.n_hess,
                n_random=args.n_random, eps_list=eps_list,
                hvp_lanczos_iters=args.lanczos_iters,
            )
        except Exception as e:
            print(f"  block {bi} failed: {type(e).__name__}: {e}")
            raise
        all_res.append(res)
        print(f"  [block {bi}] done in {time.time()-t_b:.1f}s")
        plot_one_block(res, args.omega,
                       out_dir / f"hessian_substrate_block_{bi:02d}.png")

    if len(all_res) > 1:
        plot_summary_across_blocks(
            all_res, args.omega, out_dir / "summary_across_blocks.png",
        )

    # JSON: drop the bulky per-eigenvalue raw arrays into a separate npz.
    summary = {
        "model": args.model,
        "n_blocks_total": n_blocks,
        "block_indices_analyzed": [r["block_idx"] for r in all_res],
        "omega_target": args.omega,
        "eps_list": eps_list,
        "n_hess": args.n_hess,
        "n_random": args.n_random,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "results": [
            {k: v for k, v in r.items()
             if k not in ("J_raw_eigvals_real", "J_raw_eigvals_imag")}
            for r in all_res
        ],
    }
    with open(out_dir / "hessian_substrate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        out_dir / "raw_J_eigvals.npz",
        **{
            f"block_{r['block_idx']:02d}": np.array(r["J_raw_eigvals_real"])
                + 1j * np.array(r["J_raw_eigvals_imag"])
            for r in all_res
        },
    )

    # Console table
    print("\n--- Substrate disruption summary ---")
    eps_use = eps_list[0]
    key = f"eps={eps_use:.0e}"
    print(f"{'block':>5} {'lambda_0':>22} "
          f"{'med |dl|/e (Hess)':>20} {'med |dl|/e (rand)':>20} "
          f"{'ratio':>8}")
    for r in all_res:
        d_H = np.array(r["perturbations"]["hessian"]["delta_lambda"][key])
        d_R = np.array(r["perturbations"]["random"]["delta_lambda"][key])
        abs_dH = np.sqrt(d_H[:, 0] ** 2 + d_H[:, 1] ** 2) / eps_use
        abs_dR = np.sqrt(d_R[:, 0] ** 2 + d_R[:, 1] ** 2) / eps_use
        med_H = float(np.median(abs_dH))
        med_R = float(np.median(abs_dR))
        ratio = med_H / max(med_R, 1e-30)
        lam_str = (f"{r['lambda_0'][0]:+.3f}{r['lambda_0'][1]:+.3f}j")
        print(f"{r['block_idx']:>5} {lam_str:>22} "
              f"{med_H:>20.3e} {med_R:>20.3e} {ratio:>8.2f}")

    print(f"\nElapsed: {time.time()-t_total:.1f}s")
    print(f"Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
