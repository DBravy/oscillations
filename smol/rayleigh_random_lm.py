"""
Rayleigh quotient analysis for random directions, addon to v2.

For each random direction r used in v2's perturbation tracking, compute
v^T H v / v^T v (the Rayleigh quotient) by one HVP per direction.
Combine with v2's stored |delta lambda| measurements to test the
prediction:

    within random directions, those with naturally higher |Rayleigh|
    should have higher |delta lambda|.

If yes: |H| (curvature magnitude) genuinely predicts substrate
disruption everywhere, not just at the spectral edges.

If no: random-direction |delta lambda| variance is dominated by
chance projection on the substrate gradient g, with no relation to
local curvature. The substrate-edge alignment we saw in v2 is then a
property of the eigenvectors specifically, not a generic feature of
"curvature predicts disruption".

The plot also overlays top-Hess and bottom-Hess data (whose Rayleigh
quotient equals their Lanczos eigenvalue by construction). This puts
random directions on the same axes as the spectral edges, testing
whether all three classes lie on a unified |H| -> |delta lambda|
relationship.

Cost: roughly 20 HVPs per block, ~6 seconds. Plus model loading.

Usage:
  python rayleigh_random_lm.py --v2-dir hessian_substrate_v2_HuggingFaceTB_SmolLM2-360M
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


# ---------------------------------------------------------------------------
# Reuse from v2 (kept verbatim for batch reproducibility)
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


def block_attn_params(block, attn_name):
    attn = getattr(block, attn_name)
    names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if not hasattr(attn, names[0]):
        names = ["c_attn", "c_proj"]
    params, param_names = [], []
    for n in names:
        if hasattr(attn, n):
            sub = getattr(attn, n)
            if hasattr(sub, "weight") and sub.weight is not None:
                params.append(sub.weight)
                param_names.append(f"{n}.weight")
    return params, param_names


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
        # Use MATH backend to ensure autograd graph is built for higher-order grad
        with torch.enable_grad(), \
             torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            loss = loss_fn()
            g = torch.autograd.grad(loss, params, create_graph=True)
            g_dot_v = sum((gi * vi).sum() for gi, vi in zip(g, v_list))
            Hv = torch.autograd.grad(g_dot_v, params, retain_graph=False)
        out = torch.cat([h.reshape(-1) for h in Hv]).detach().cpu().numpy()
        return out.astype(np.float64)

    return hvp_flat, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", type=str, required=True,
                        help="Directory containing hessian_substrate_v2_summary.json")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Must match v2 run; default 8 matches v2 default.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Default: same as --v2-dir.")
    args = parser.parse_args()

    v2_dir = Path(args.v2_dir)
    summary_path = v2_dir / "hessian_substrate_v2_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find {summary_path}")
    with open(summary_path) as f:
        v2 = json.load(f)

    out_dir = Path(args.out_dir) if args.out_dir else v2_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    model_name = v2["model"]
    seq_len = v2["seq_len"]
    batch_size = v2["batch_size"]
    n_random = v2["n_random"]
    eps_list = v2["eps_list"]
    eps_use = eps_list[1] if len(eps_list) > 1 else eps_list[0]
    eps_key = f"eps={eps_use:.0e}"

    # Reproduce text shuffling from v2 (same seed)
    print("Loading wikitext-2 ...")
    rng = np.random.default_rng(SEED)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]

    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    ).to(DEVICE)
    model.eval()
    blocks, _, _, attn_name, _ = get_block_info(model)

    rayleigh_results = {}

    for r in v2["results"]:
        bi = r["block_idx"]
        block = blocks[bi]
        print(f"\n=== Block {bi} ===")

        # Build batch and HVP for this block
        batch_ids, mask = collect_loss_batch(
            model, tokenizer, texts, seq_len, batch_size,
        )
        loss_fn = make_loss_fn(model, batch_ids, mask)
        for p in model.parameters():
            p.requires_grad_(False)
        params, _ = block_attn_params(block, attn_name)
        for p in params:
            p.requires_grad_(True)
        hvp_flat, total = make_hvp(loss_fn, params)

        # Reproduce v2's random directions (same seed)
        rng_block = np.random.default_rng(SEED + bi)
        rand_dirs = rng_block.standard_normal((n_random, total)).astype(np.float64)
        rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)

        # Rayleigh quotient for each random direction
        print(f"  computing {n_random} HVPs ...")
        t0 = time.time()
        rayleigh_rand = np.zeros(n_random)
        for k in range(n_random):
            Hv = hvp_flat(rand_dirs[k])
            rayleigh_rand[k] = float(rand_dirs[k] @ Hv)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s ({elapsed/n_random:.2f}s/HVP)")
        print(f"  random Rayleigh: mean={rayleigh_rand.mean():+.3e}  "
              f"std={rayleigh_rand.std():.3e}  "
              f"range=[{rayleigh_rand.min():+.3e}, {rayleigh_rand.max():+.3e}]")

        # Pull v2 measurements
        H_top = np.array(r["hessian_top_eigvals"])
        H_bot = np.array(r["hessian_bottom_eigvals"])
        d_T = np.array(r["perturbations"]["top"][eps_key])
        d_B = np.array(r["perturbations"]["bottom"][eps_key])
        d_R = np.array(r["perturbations"]["random"][eps_key])
        abs_dT = np.sqrt(d_T[:, 0] ** 2 + d_T[:, 1] ** 2) / eps_use
        abs_dB = np.sqrt(d_B[:, 0] ** 2 + d_B[:, 1] ** 2) / eps_use
        abs_dR = np.sqrt(d_R[:, 0] ** 2 + d_R[:, 1] ** 2) / eps_use

        # Correlations within random
        if n_random > 3 and rayleigh_rand.std() > 0:
            pearson_abs = float(np.corrcoef(np.abs(rayleigh_rand), abs_dR)[0, 1])
            spearman_abs = float(np.corrcoef(
                np.argsort(np.argsort(np.abs(rayleigh_rand))),
                np.argsort(np.argsort(abs_dR)),
            )[0, 1])
            pearson_signed = float(np.corrcoef(rayleigh_rand, abs_dR)[0, 1])
            print(f"  Pearson(|Rayleigh|, |dl|/eps) random: {pearson_abs:+.3f}")
            print(f"  Spearman rank corr:                   {spearman_abs:+.3f}")
            print(f"  Pearson(Rayleigh signed, |dl|/eps):   {pearson_signed:+.3f}")
        else:
            pearson_abs = spearman_abs = pearson_signed = float("nan")

        # Pooled regression: combine all three classes on |H| vs |dl|
        all_x = np.concatenate([np.abs(H_top), np.abs(H_bot),
                                 np.abs(rayleigh_rand)])
        all_y = np.concatenate([abs_dT, abs_dB, abs_dR])
        pos_mask = (all_x > 0) & (all_y > 0)
        if pos_mask.sum() > 5:
            slope, intercept = np.polyfit(
                np.log(all_x[pos_mask]), np.log(all_y[pos_mask]), 1,
            )
            print(f"  pooled log-log slope (|H| vs |dl|): {slope:+.3f}")
        else:
            slope, intercept = float("nan"), float("nan")

        rayleigh_results[str(bi)] = {
            "rayleigh_random": rayleigh_rand.tolist(),
            "abs_dl_random": abs_dR.tolist(),
            "rayleigh_top": H_top.tolist(),
            "abs_dl_top": abs_dT.tolist(),
            "rayleigh_bot": H_bot.tolist(),
            "abs_dl_bot": abs_dB.tolist(),
            "pearson_abs_random": pearson_abs,
            "spearman_random": spearman_abs,
            "pearson_signed_random": pearson_signed,
            "pooled_loglog_slope": float(slope),
            "pooled_loglog_intercept": float(intercept),
        }

        # Per-block plot
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        ax = axes[0]
        ax.scatter(np.abs(H_top), abs_dT, color="C0", s=60,
                   label="top Hess (eigval)",
                   edgecolors="k", linewidth=0.4)
        ax.scatter(np.abs(H_bot), abs_dB, color="C2", s=60, marker="s",
                   label="bottom Hess (|eigval|)",
                   edgecolors="k", linewidth=0.4)
        ax.scatter(np.abs(rayleigh_rand), abs_dR, color="C3", s=45,
                   marker="x", linewidth=1.4,
                   label="random (|Rayleigh|)")
        if not np.isnan(slope):
            x_ref = np.array([all_x[pos_mask].min(), all_x[pos_mask].max()])
            ax.plot(x_ref, np.exp(intercept) * x_ref ** slope, "--",
                    color="gray", linewidth=1.1,
                    label=f"pooled fit, slope={slope:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|v^T H v|  (curvature magnitude)")
        ax.set_ylabel("|delta lambda| / eps")
        ax.set_title(f"Block {bi}: |H| vs substrate disruption, all classes")
        ax.legend(fontsize=9, loc="best")

        ax = axes[1]
        # Random-only zoom: shows the within-class correlation
        ax.scatter(np.abs(rayleigh_rand), abs_dR, color="C3", s=55,
                   marker="x", linewidth=1.5)
        if rayleigh_rand.std() > 0 and n_random > 3:
            # Linear fit for visual
            xs = np.abs(rayleigh_rand)
            ys = abs_dR
            sl, ic = np.polyfit(xs, ys, 1)
            xr = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(xr, sl * xr + ic, "--", color="gray", linewidth=1.0,
                    label=f"linear fit  Pearson={pearson_abs:+.2f}")
            ax.legend(fontsize=9, loc="best")
        ax.set_xlabel("|Rayleigh| of random direction")
        ax.set_ylabel("|delta lambda| / eps")
        ax.set_title(f"Block {bi}: random directions only")

        fig.tight_layout()
        fig.savefig(out_dir / f"rayleigh_block_{bi:02d}.png", dpi=120)
        plt.close(fig)

    with open(out_dir / "rayleigh_random_summary.json", "w") as f:
        json.dump(rayleigh_results, f, indent=2)

    # Console summary
    print("\n--- Rayleigh-vs-disruption summary ---")
    print(f"{'block':>5} {'<|R|> rand':>12} {'std(R) rand':>14} "
          f"{'pearson':>9} {'spearman':>9} {'slope':>7}")
    for bi_str, rr in rayleigh_results.items():
        rays = np.array(rr["rayleigh_random"])
        print(f"{bi_str:>5} {np.mean(np.abs(rays)):>12.3e} "
              f"{np.std(rays):>14.3e} "
              f"{rr['pearson_abs_random']:>+9.3f} "
              f"{rr['spearman_random']:>+9.3f} "
              f"{rr['pooled_loglog_slope']:>+7.3f}")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
