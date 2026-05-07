"""
Hessian-and-substrate analysis, v2.

Extends hessian_substrate_lm.py with two follow-up tests:

(1) Bottom-Hessian eigenvectors (which='SA' in Lanczos). The framework
    predicts that flat directions are substrate-preserving. This is the
    direct test, where random directions in the v1 script were a proxy.
    We expect: |delta lambda| for bottom-Hess << |delta lambda| for
    top-Hess. If bottom-Hess directions disrupt as much as top-Hess
    directions, the framework's "curvature = substrate-disrupting"
    prediction is wrong at both ends. If bottom-Hess matches the random
    baseline, the framework's prediction is fully supported.

    Caveat. For an indefinite Hessian, which='SA' returns the smallest
    algebraic (most negative). For NN Hessians at trained models the
    spectrum is mostly small-magnitude with a few large-positive
    eigenvalues; 'SA' typically returns near-zero or mildly negative
    eigenvalues. We report the actual eigenvalues found so the
    interpretation is unambiguous.

(2) Direct substrate gradient. Compute g = (g_re, g_im) where
    g_re = d Re(lambda_0) / d W,  g_im = d Im(lambda_0) / d W
    via two backward passes through the chain
        lambda_0 = u_left^T (df_attn / dx)|_{x_0} v_right.
    Project g onto the top-k and bottom-k Hessian subspaces. The
    framework's prediction is that g lives mostly in the top-Hessian
    subspace.

    Once we have g, we can also predict |delta lambda| for any
    perturbation v as
        |delta lambda|^2 ~ (g_re . v)^2 + (g_im . v)^2
    and compare to the measured |delta lambda|. Agreement validates
    the first-order picture and lets us decompose substrate disruption
    into "in the (g_re, g_im) plane" vs "out of plane".

Outputs (extends v1):
  hessian_substrate_block_<bi>.png      Per-block plot, now 6 panels.
  summary_across_blocks.png             Same as v1 (if multiple blocks).
  hessian_substrate_v2_summary.json     Numerical summary including
                                        projection statistics.
  raw_J_eigvals.npz                     Raw J eigenvalues per block.

Usage:
  python hessian_substrate_v2_lm.py --block-idx 16
  python hessian_substrate_v2_lm.py --block-idx 4,12,16,20,28
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
# Block discovery
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
# Capture (one sample), attention-call rebuild, J_attn (vectorized)
# ---------------------------------------------------------------------------

def capture_one_sample(model, tokenizer, text, seq_len, target_block_idx):
    blocks, ln1_name, _, attn_name, _ = get_block_info(model)
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

    return None if captured["x_attn_in"] is None else {
        "input_ids": ids, **captured,
    }


def _build_attn_call(captured, x_norm):
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


def attn_output_last(block, captured, ln1_name, attn_name, x_last):
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
    eigvals, eigvecs_R = np.linalg.eig(J)
    eigvals_T, eigvecs_L_T = np.linalg.eig(J.T)

    im_abs = np.abs(eigvals.imag)
    in_band = (im_abs > 0.5 * omega_target) & (im_abs < 1.5 * omega_target)
    if not in_band.any():
        idx_R = int(np.argmax(im_abs))
        print(f"  no eigenvalue in [{0.5*omega_target:.3f}, "
              f"{1.5*omega_target:.3f}]; using overall top by |Im|.")
    else:
        candidates = np.where(in_band)[0]
        idx_R = candidates[np.argmax(im_abs[candidates])]

    lam = eigvals[idx_R]
    vR = eigvecs_R[:, idx_R]
    diffs = np.abs(eigvals_T - lam)
    idx_L = int(np.argmin(diffs))
    vL = eigvecs_L_T[:, idx_L]
    inner = vL @ vR
    if abs(inner) < 1e-12:
        raise RuntimeError("Defective eigenvalue.")
    vL = vL / inner
    return lam, vR, vL


# ---------------------------------------------------------------------------
# Loss + HVP
# ---------------------------------------------------------------------------

def collect_loss_batch(model, tokenizer, texts, seq_len, batch_size):
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
    attn = getattr(block, attn_name)
    names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if not hasattr(attn, names[0]):
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
    def loss_fn():
        out = model(batch_ids).logits
        logits = out[:, :-1, :].contiguous()
        targets = batch_ids[:, 1:].contiguous()
        m = mask[:, 1:].contiguous()
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return (nll * m.float()).sum() / m.float().sum().clamp(min=1.0)
    return loss_fn


def make_hvp(loss_fn, params):
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
# First-order eigenvalue tracking (re-used from v1)
# ---------------------------------------------------------------------------

def lambda_first_order(block, captured, ln1_name, attn_name, vR, vL):
    x_last_init = captured["x_attn_in"][0, -1, :].detach().clone().to(torch.float32)
    vR_re = torch.from_numpy(np.ascontiguousarray(vR.real)).to(DEVICE, dtype=torch.float32)
    vR_im = torch.from_numpy(np.ascontiguousarray(vR.imag)).to(DEVICE, dtype=torch.float32)
    vL_re = torch.from_numpy(np.ascontiguousarray(vL.real)).to(DEVICE, dtype=torch.float32)
    vL_im = torch.from_numpy(np.ascontiguousarray(vL.imag)).to(DEVICE, dtype=torch.float32)

    def f(x_last):
        return attn_output_last(block, captured, ln1_name, attn_name, x_last)

    # Use MATH backend because flash-attention CPU kernel lacks higher-order grad
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
# Substrate gradient: d lambda_0 / d W
# ---------------------------------------------------------------------------

def compute_substrate_gradient(block, captured, ln1_name, attn_name,
                                vR, vL, params, sizes):
    """
    lambda_0 = u^T J(W) v, where J = df/dx at x_0.
    Use the identity u^T J v = (du f / dx) . v with the scalar field
    h(x) = u^T f(x). Two backward passes (one per real/imag component
    of u), gradient through the inner grad gives d lambda_0 / d W.

    Returns g_re, g_im as flat float64 numpy arrays of size sum(sizes).
    """
    vR_re_t = torch.from_numpy(np.ascontiguousarray(vR.real)).to(DEVICE, dtype=torch.float32)
    vR_im_t = torch.from_numpy(np.ascontiguousarray(vR.imag)).to(DEVICE, dtype=torch.float32)
    vL_re_t = torch.from_numpy(np.ascontiguousarray(vL.real)).to(DEVICE, dtype=torch.float32)
    vL_im_t = torch.from_numpy(np.ascontiguousarray(vL.imag)).to(DEVICE, dtype=torch.float32)

    x_last_init = captured["x_attn_in"][0, -1, :].detach().clone().to(torch.float32)
    x_in = x_last_init.detach().clone().requires_grad_(True)

    for p in params:
        p.requires_grad_(True)

    # Use MATH backend because flash-attention CPU kernel lacks higher-order grad
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        fx = attn_output_last(block, captured, ln1_name, attn_name, x_in)

        h_re = vL_re_t @ fx
        dh_re_dx = torch.autograd.grad(
            h_re, x_in, create_graph=True, retain_graph=True,
        )[0]

        h_im = vL_im_t @ fx
        dh_im_dx = torch.autograd.grad(
            h_im, x_in, create_graph=True, retain_graph=True,
        )[0]

        re_lambda = (dh_re_dx @ vR_re_t) - (dh_im_dx @ vR_im_t)
        im_lambda = (dh_re_dx @ vR_im_t) + (dh_im_dx @ vR_re_t)

        g_re_list = torch.autograd.grad(re_lambda, params, retain_graph=True)
        g_im_list = torch.autograd.grad(im_lambda, params, retain_graph=False)

    g_re = torch.cat([g.reshape(-1) for g in g_re_list]).detach().cpu().numpy()
    g_im = torch.cat([g.reshape(-1) for g in g_im_list]).detach().cpu().numpy()
    return g_re.astype(np.float64), g_im.astype(np.float64)


# ---------------------------------------------------------------------------
# Per-block analysis
# ---------------------------------------------------------------------------

def perturb_and_track(block, captured, ln1_name, attn_name,
                      params, shapes, sizes, lam0, vR, vL,
                      directions, eps_list):
    """For each direction (rows of `directions`, shape (k, total)) and each
    eps, return a dict eps -> list of complex delta lambda."""
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

    out = {}
    for eps in eps_list:
        d_lam = []
        for k in range(directions.shape[0]):
            apply_perturbation(directions[k], eps)
            lam_p = lambda_first_order(
                block, captured, ln1_name, attn_name, vR, vL,
            )
            d_lam.append(lam_p - lam0)
            restore_params()
        out[f"eps={eps:.0e}"] = [(z.real, z.imag) for z in d_lam]
    return out


def analyze_block(model, tokenizer, texts, target_block_idx,
                  seq_len, batch_size, omega_target, n_hess, n_random,
                  eps_list, lanczos_iters, do_bottom, do_grad):
    blocks, ln1_name, _, attn_name, _ = get_block_info(model)
    block = blocks[target_block_idx]

    # Capture
    print(f"  [block {target_block_idx}] capturing operating point ...")
    captured = None
    for text in texts:
        cap = capture_one_sample(model, tokenizer, text, seq_len, target_block_idx)
        if cap is not None:
            captured = cap
            break
    if captured is None:
        raise RuntimeError("No usable text for capture.")

    # J_attn + substrate eigenpair
    t0 = time.time()
    J = compute_J_attn(block, captured, ln1_name, attn_name)
    print(f"  [block {target_block_idx}] J_attn shape {J.shape}, "
          f"computed in {time.time()-t0:.2f}s")
    lam0, vR, vL = find_substrate_eigenpair(J, omega_target)
    print(f"  [block {target_block_idx}] lambda_0 = "
          f"{lam0.real:+.4f} {lam0.imag:+.4f}j   "
          f"(target |Im|={omega_target:.4f})")

    # Loss + HVP
    print(f"  [block {target_block_idx}] building loss batch ...")
    batch_ids, mask = collect_loss_batch(
        model, tokenizer, texts, seq_len, batch_size,
    )
    loss_fn = make_loss_fn(model, batch_ids, mask)

    for p in model.parameters():
        p.requires_grad_(False)
    params, param_names = block_attn_params(block, attn_name)
    for p in params:
        p.requires_grad_(True)
    n_params_block = sum(p.numel() for p in params)
    print(f"  [block {target_block_idx}] HVP over {n_params_block:,} "
          f"params: {param_names}")

    hvp_flat, total, shapes, sizes = make_hvp(loss_fn, params)
    op = LinearOperator((total, total), matvec=hvp_flat, dtype=np.float64)

    # Lanczos: top-k
    print(f"  [block {target_block_idx}] Lanczos top-{n_hess} (which='LA') ...")
    t0 = time.time()
    eigvals_T, eigvecs_T = eigsh(
        op, k=n_hess, which="LA", tol=1e-3,
        maxiter=max(lanczos_iters, 4 * n_hess),
    )
    order = np.argsort(-eigvals_T)
    eigvals_T = eigvals_T[order]
    eigvecs_T = eigvecs_T[:, order]
    print(f"  [block {target_block_idx}] top Lanczos: {time.time()-t0:.1f}s   "
          f"range [{eigvals_T[-1]:.3e}, {eigvals_T[0]:.3e}]")

    # Lanczos: bottom-k
    eigvals_B, eigvecs_B = None, None
    if do_bottom:
        print(f"  [block {target_block_idx}] Lanczos bottom-{n_hess} (which='SA') ...")
        t0 = time.time()
        eigvals_B, eigvecs_B = eigsh(
            op, k=n_hess, which="SA", tol=1e-3,
            maxiter=max(lanczos_iters, 4 * n_hess),
        )
        order = np.argsort(eigvals_B)  # ascending
        eigvals_B = eigvals_B[order]
        eigvecs_B = eigvecs_B[:, order]
        print(f"  [block {target_block_idx}] bottom Lanczos: {time.time()-t0:.1f}s   "
              f"range [{eigvals_B[0]:.3e}, {eigvals_B[-1]:.3e}]")

    # Random
    rng = np.random.default_rng(SEED + target_block_idx)
    rand_dirs = rng.standard_normal((n_random, total)).astype(np.float64)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)

    # Substrate gradient
    g_re = g_im = None
    grad_stats = None
    if do_grad:
        print(f"  [block {target_block_idx}] computing d lambda / d W ...")
        t0 = time.time()
        g_re, g_im = compute_substrate_gradient(
            block, captured, ln1_name, attn_name, vR, vL, params, sizes,
        )
        norm_re = float(np.linalg.norm(g_re))
        norm_im = float(np.linalg.norm(g_im))
        cos_re_im = float(g_re @ g_im / (norm_re * norm_im + 1e-30))
        print(f"  [block {target_block_idx}] grad done in {time.time()-t0:.2f}s   "
              f"|g_re|={norm_re:.3e}  |g_im|={norm_im:.3e}  "
              f"cos(g_re,g_im)={cos_re_im:+.3f}")

        # Orthonormalize the (g_re, g_im) plane via Gram-Schmidt
        e1 = g_re / max(norm_re, 1e-30)
        proj = e1 @ g_im
        e2 = g_im - proj * e1
        norm_e2 = float(np.linalg.norm(e2))
        if norm_e2 > 1e-30:
            e2 = e2 / norm_e2
        else:
            e2 = np.zeros_like(e1)

        # Per-eigenvector overlaps with the (g_re, g_im) plane.
        # |proj_v_onto_plane|^2 = (e1 . v)^2 + (e2 . v)^2
        def plane_overlap(eigvecs_matrix):
            if eigvecs_matrix is None:
                return None
            o1 = eigvecs_matrix.T @ e1
            o2 = eigvecs_matrix.T @ e2
            return (o1 ** 2 + o2 ** 2)  # shape (k,)

        ov_top = plane_overlap(eigvecs_T)
        ov_bot = plane_overlap(eigvecs_B) if eigvecs_B is not None else None
        ov_rand = plane_overlap(rand_dirs.T)

        grad_stats = {
            "norm_g_re": norm_re,
            "norm_g_im": norm_im,
            "cos_g_re_g_im": cos_re_im,
            "frac_g_in_top_subspace": float(ov_top.sum()),
            "frac_g_in_bottom_subspace": (
                float(ov_bot.sum()) if ov_bot is not None else None
            ),
            "frac_g_in_random_subspace": float(ov_rand.sum()),
            "per_top_overlap_sq": ov_top.tolist(),
            "per_bottom_overlap_sq": (
                ov_bot.tolist() if ov_bot is not None else None
            ),
            "per_random_overlap_sq": ov_rand.tolist(),
        }
        print(f"  [block {target_block_idx}] g energy in subspaces: "
              f"top={grad_stats['frac_g_in_top_subspace']:.3f}, "
              f"bottom={grad_stats['frac_g_in_bottom_subspace']}, "
              f"random={grad_stats['frac_g_in_random_subspace']:.3f}")

    # Perturbation tracking
    print(f"  [block {target_block_idx}] perturbation tracking ...")
    pert = {"top": perturb_and_track(
        block, captured, ln1_name, attn_name, params, shapes, sizes,
        lam0, vR, vL, eigvecs_T.T, eps_list,
    )}
    if eigvecs_B is not None:
        pert["bottom"] = perturb_and_track(
            block, captured, ln1_name, attn_name, params, shapes, sizes,
            lam0, vR, vL, eigvecs_B.T, eps_list,
        )
    pert["random"] = perturb_and_track(
        block, captured, ln1_name, attn_name, params, shapes, sizes,
        lam0, vR, vL, rand_dirs, eps_list,
    )

    raw_eigvals = np.linalg.eigvals(J)

    return {
        "block_idx": target_block_idx,
        "n_params_block": n_params_block,
        "param_names": param_names,
        "lambda_0": (lam0.real, lam0.imag),
        "hessian_top_eigvals": eigvals_T.tolist(),
        "hessian_bottom_eigvals": (
            eigvals_B.tolist() if eigvals_B is not None else None
        ),
        "perturbations": pert,
        "eps_list": list(eps_list),
        "grad_stats": grad_stats,
        "J_raw_eigvals_real": raw_eigvals.real.tolist(),
        "J_raw_eigvals_imag": raw_eigvals.imag.tolist(),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _abs_dl_per_eps(perts_dict, eps):
    key = f"eps={eps:.0e}"
    arr = np.array(perts_dict[key])
    return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2) / eps


def plot_one_block(res, omega_target, out_path):
    bi = res["block_idx"]
    eps_use = res["eps_list"][1] if len(res["eps_list"]) > 1 else res["eps_list"][0]
    H_top = np.array(res["hessian_top_eigvals"])
    H_bot = (np.array(res["hessian_bottom_eigvals"])
             if res["hessian_bottom_eigvals"] is not None else None)
    perts = res["perturbations"]
    abs_dT = _abs_dl_per_eps(perts["top"], eps_use)
    abs_dR = _abs_dl_per_eps(perts["random"], eps_use)
    abs_dB = _abs_dl_per_eps(perts["bottom"], eps_use) if "bottom" in perts else None

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    # (1, 1) Disruption per direction
    ax = axes[0, 0]
    ax.scatter(np.arange(len(abs_dT)), abs_dT, color="C0", label="top Hess")
    if abs_dB is not None:
        ax.scatter(np.arange(len(abs_dB)), abs_dB, color="C2",
                   marker="s", label="bottom Hess")
    ax.scatter(np.arange(len(abs_dR)), abs_dR, color="C3",
               marker="x", label="random")
    ax.set_yscale("log")
    ax.set_xlabel("direction index")
    ax.set_ylabel("|delta lambda| / eps")
    ax.set_title(f"Block {bi}: disruption per direction (eps={eps_use:.0e})")
    ax.legend(fontsize=8, loc="best")

    # (1, 2) Hess eigval vs disruption (loglog where defined)
    ax = axes[0, 1]
    # Top: eigvals positive
    pos_T = H_top > 0
    if pos_T.any():
        ax.loglog(H_top[pos_T], abs_dT[pos_T], "o", color="C0", label="top Hess")
    if abs_dB is not None:
        pos_B = H_bot > 0
        if pos_B.any():
            ax.loglog(H_bot[pos_B], abs_dB[pos_B], "s", color="C2",
                      label="bottom Hess (+)")
        neg_B = H_bot < 0
        if neg_B.any():
            ax.loglog(np.abs(H_bot[neg_B]), abs_dB[neg_B], "s", color="C4",
                      label="bottom Hess (-, |H|)")
    ax.set_xlabel("|Hessian eigenvalue|")
    ax.set_ylabel("|delta lambda| / eps")
    ax.set_title(f"Block {bi}: curvature vs disruption")
    ax.legend(fontsize=8, loc="best")

    # (1, 3) Linearity check
    ax = axes[0, 2]
    for eps in res["eps_list"]:
        rates = _abs_dl_per_eps(perts["top"], eps)
        ax.plot(np.arange(len(rates)), rates, "-o", markersize=3,
                label=f"eps={eps:.0e}")
    ax.set_yscale("log")
    ax.set_xlabel("top-Hess direction index")
    ax.set_ylabel("|delta lambda| / eps")
    ax.set_title(f"Block {bi}: linearity check")
    ax.legend(fontsize=8, loc="best")

    # (2, 1) Substrate gradient overlap per direction
    gs = res.get("grad_stats")
    ax = axes[1, 0]
    if gs is not None:
        ax.plot(gs["per_top_overlap_sq"], "-o", color="C0",
                label="top Hess", markersize=4)
        if gs["per_bottom_overlap_sq"] is not None:
            ax.plot(gs["per_bottom_overlap_sq"], "-s", color="C2",
                    label="bottom Hess", markersize=4)
        ax.plot(gs["per_random_overlap_sq"], "-x", color="C3",
                label="random", markersize=4)
        ax.set_xlabel("direction index")
        ax.set_ylabel("|projection onto (g_re, g_im) plane|^2")
        ax.set_title(f"Block {bi}: substrate-gradient overlap")
        ax.set_yscale("log")
        ax.legend(fontsize=8, loc="best")
    else:
        ax.axis("off")

    # (2, 2) predicted vs measured |delta lambda|
    ax = axes[1, 1]
    if gs is not None:
        # For each top-Hess direction k, predicted disruption is:
        # |g_re|^2 (e1 . v_k)^2 + |g_im|^2 (e2 . v_k)^2 + cross.
        # Easier exact formula: |delta|^2 = (g_re . v)^2 + (g_im . v)^2.
        # We don't have g_re, g_im stored here; use the overlap-based
        # reconstruction with norms:
        norm_re = gs["norm_g_re"]
        norm_im = gs["norm_g_im"]
        # |proj on plane| measured == sum of (e1.v)^2 + (e2.v)^2.
        # Predicted |delta|/eps for unit v (using e1 = g_re / |g_re|):
        # delta_re = g_re . v = |g_re| (e1 . v)
        # delta_im = g_im . v = |g_im| (cos.. e1 . v + sin.. e2 . v)
        # We don't have separated (e1.v) vs (e2.v) stored, so plot
        # predicted = sqrt(|g_re|^2 * overlap_sq) as a rough upper bound.
        # The cleaner thing: just show |delta_lambda| vs |projection on plane|.
        ovT = np.array(gs["per_top_overlap_sq"])
        ax.loglog(np.sqrt(ovT) + 1e-30, abs_dT, "o", color="C0",
                  label="top Hess")
        if gs["per_bottom_overlap_sq"] is not None and abs_dB is not None:
            ovB = np.array(gs["per_bottom_overlap_sq"])
            ax.loglog(np.sqrt(ovB) + 1e-30, abs_dB, "s", color="C2",
                      label="bottom Hess")
        ovR = np.array(gs["per_random_overlap_sq"])
        ax.loglog(np.sqrt(ovR) + 1e-30, abs_dR, "x", color="C3",
                  label="random")
        ax.set_xlabel("|projection of v onto (g_re, g_im) plane|")
        ax.set_ylabel("|delta lambda| / eps")
        ax.set_title(f"Block {bi}: alignment vs disruption")
        ax.legend(fontsize=8, loc="best")
    else:
        ax.axis("off")

    # (2, 3) Spectrum overview
    ax = axes[1, 2]
    if H_bot is not None:
        ax.plot(np.sort(H_bot), "s-", color="C2", label="bottom-k", markersize=3)
    ax.plot(np.sort(H_top), "o-", color="C0", label="top-k", markersize=3)
    ax.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("rank within set (ascending)")
    ax.set_ylabel("Hessian eigenvalue")
    ax.set_title(f"Block {bi}: Hessian spectrum")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Hessian-substrate v2, block {bi}    "
        f"lambda_0 = {res['lambda_0'][0]:+.4f}{res['lambda_0'][1]:+.4f}j   "
        f"(omega target = {omega_target:.4f})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_summary_across_blocks(all_res, out_path):
    blocks = sorted(r["block_idx"] for r in all_res)
    by_idx = {r["block_idx"]: r for r in all_res}
    med_T, med_R, med_B = [], [], []
    frac_top, frac_bot = [], []
    for bi in blocks:
        r = by_idx[bi]
        eps_use = r["eps_list"][1] if len(r["eps_list"]) > 1 else r["eps_list"][0]
        med_T.append(np.median(_abs_dl_per_eps(r["perturbations"]["top"], eps_use)))
        med_R.append(np.median(_abs_dl_per_eps(r["perturbations"]["random"], eps_use)))
        if "bottom" in r["perturbations"]:
            med_B.append(np.median(_abs_dl_per_eps(
                r["perturbations"]["bottom"], eps_use,
            )))
        else:
            med_B.append(np.nan)
        gs = r.get("grad_stats")
        if gs:
            frac_top.append(gs["frac_g_in_top_subspace"])
            frac_bot.append(
                gs["frac_g_in_bottom_subspace"]
                if gs["frac_g_in_bottom_subspace"] is not None else np.nan
            )
        else:
            frac_top.append(np.nan)
            frac_bot.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(blocks, med_T, "-o", color="C0", label="top Hess")
    ax.plot(blocks, med_B, "-s", color="C2", label="bottom Hess")
    ax.plot(blocks, med_R, "-x", color="C3", label="random")
    ax.set_yscale("log")
    ax.set_xlabel("block index")
    ax.set_ylabel("median |delta lambda| / eps")
    ax.set_title("Substrate disruption by direction class")
    ax.legend(fontsize=9, loc="best")

    ax = axes[1]
    ax.plot(blocks, frac_top, "-o", color="C0", label="top-k subspace")
    ax.plot(blocks, frac_bot, "-s", color="C2", label="bottom-k subspace")
    ax.set_xlabel("block index")
    ax.set_ylabel("fraction of substrate-gradient energy")
    ax.set_title("Substrate gradient subspace alignment")
    ax.legend(fontsize=9, loc="best")
    ax.set_ylim([0, 1])

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
    parser.add_argument("--block-idx", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--omega", type=float, default=0.105)
    parser.add_argument("--n-hess", type=int, default=20)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--eps", type=str, default="1e-4,1e-3,1e-2")
    parser.add_argument("--lanczos-iters", type=int, default=200)
    parser.add_argument("--skip-bottom", action="store_true")
    parser.add_argument("--skip-grad", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(f"hessian_substrate_v2_{slugify(args.model)}"))
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
                lanczos_iters=args.lanczos_iters,
                do_bottom=(not args.skip_bottom),
                do_grad=(not args.skip_grad),
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
            all_res, out_dir / "summary_across_blocks.png",
        )

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
    with open(out_dir / "hessian_substrate_v2_summary.json", "w") as f:
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
    print("\n--- v2 summary ---")
    eps_use = eps_list[1] if len(eps_list) > 1 else eps_list[0]
    header = (f"{'block':>5} {'lam0':>20} "
              f"{'med dl/e top':>14} "
              f"{'med dl/e bot':>14} "
              f"{'med dl/e rnd':>14} "
              f"{'top/rnd':>9} {'bot/rnd':>9} "
              f"{'g_top':>7} {'g_bot':>7}")
    print(header)
    for r in all_res:
        med_T = np.median(_abs_dl_per_eps(r["perturbations"]["top"], eps_use))
        med_R = np.median(_abs_dl_per_eps(r["perturbations"]["random"], eps_use))
        if "bottom" in r["perturbations"]:
            med_B = np.median(_abs_dl_per_eps(
                r["perturbations"]["bottom"], eps_use,
            ))
        else:
            med_B = float("nan")
        ratio_T = med_T / max(med_R, 1e-30)
        ratio_B = med_B / max(med_R, 1e-30)
        gs = r.get("grad_stats") or {}
        g_top = gs.get("frac_g_in_top_subspace", float("nan"))
        g_bot = gs.get("frac_g_in_bottom_subspace")
        if g_bot is None:
            g_bot = float("nan")
        lam_str = f"{r['lambda_0'][0]:+.3f}{r['lambda_0'][1]:+.3f}j"
        print(f"{r['block_idx']:>5} {lam_str:>20} "
              f"{med_T:>14.3e} {med_B:>14.3e} {med_R:>14.3e} "
              f"{ratio_T:>9.1f} {ratio_B:>9.1f} "
              f"{g_top:>7.3f} {g_bot:>7.3f}")

    print(f"\nElapsed: {time.time()-t_total:.1f}s")
    print(f"Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
