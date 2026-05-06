"""
Loss curve sweep across OLMo-2-1B training checkpoints.

OLMo port companion to phase_space_olmo_checkpoints.py. Computes mean
cross-entropy loss on a held-out corpus at each checkpoint in the
training run, producing a loss / perplexity curve over training steps.

Mirrors the checkpoint-handling conventions from the phase-space script:
  - Step <= 37000: load from allenai/OLMo-2-0425-1B-early-training
  - Step >  37000: load from allenai/OLMo-2-0425-1B
  - Revision string: f"stage1-step{step}-tokens{tokens_b}B" with
    tokens_b = ceil(step * 2048 * 1024 / 1e9)
  - Per-step .npz files for resume on restart
  - HF cache cleanup after each checkpoint to keep disk bounded
  - Skips checkpoints that fail to load (e.g. missing late revisions)

Eval pipeline (standard packed-CE recipe):
  Concatenate the dataset texts with EOS separators, tokenize once,
  chunk into fixed seq_len blocks, drop the last partial block. Each
  block contributes seq_len-1 next-token predictions. Total NLL and
  total token count are accumulated, and reported as nats/token plus
  perplexity. This is the cleanest single-scalar-per-checkpoint number
  for plotting a loss curve.

Default eval data is the wikitext-2 validation split. Swap in another
HF text dataset via --dataset / --dataset-config / --dataset-split if
you want a stronger eval (e.g. wikitext-103 test, c4 validation).

Default checkpoints span the full training run. The early-training
repo has all checkpoints at every 1000 steps from 0 to 37000. The
main repo has thousands of stage-1 revisions but the spacing is
irregular, so late steps may not all exist; the loader skips and
continues. Inspect available revisions with:
  from huggingface_hub import list_repo_refs
  [b.name for b in list_repo_refs("allenai/OLMo-2-0425-1B").branches]

Outputs in olmo_loss_curve/:
  step_<step>.npz   Per-checkpoint scalar (loss, total_tokens).
  loss_curve.json   Aggregated curve, written incrementally.
  loss_curve.png    Two-panel plot: loss and perplexity vs step.

Usage:
  python loss_curve_olmo_checkpoints.py
  python loss_curve_olmo_checkpoints.py --checkpoints 0 1000 5000 10000 37000
  python loss_curve_olmo_checkpoints.py --seq-len 2048 --batch-size 8
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from huggingface_hub import scan_cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# OLMo-2-1B configuration (matches phase_space_olmo_checkpoints.py)
# ---------------------------------------------------------------------------

OLMO_MODEL = "allenai/OLMo-2-0425-1B"
EARLY_TRAINING_REPO = "allenai/OLMo-2-0425-1B-early-training"
EARLY_TRAINING_MAX_STEP = 37000

DEFAULT_CHECKPOINTS = [0, 1000, 2000, 3000, 5000, 10000, 100000, 1000000]

_ORICO_CACHE = "/Volumes/ORICO/huggingface_cache"
DEFAULT_CACHE_DIR = _ORICO_CACHE if os.path.isdir("/Volumes/ORICO") else None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def step_to_revision(step):
    """Convert a step number to the OLMo-2-1B revision string."""
    tokens_b = math.ceil(step * 2048 * 1024 / 1_000_000_000)
    return f"stage1-step{step}-tokens{tokens_b}B"


# ---------------------------------------------------------------------------
# Loading and cache management (lifted from phase_space_olmo_checkpoints.py)
# ---------------------------------------------------------------------------

def load_olmo_at_step(step, dtype, cache_dir=None):
    """Load OLMo-2-1B at a specific training step. Routes to early-training
    repo for steps <= 37000, main repo for later steps."""
    repo = EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP else OLMO_MODEL
    revision = step_to_revision(step)
    print(f"  Loading {repo} at {revision} ...")
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    ).to(DEVICE)
    model.eval()
    return model


def cleanup_checkpoint_cache(repo_id, revision, cache_dir=None):
    """Delete cached files for a specific revision to free disk space."""
    try:
        cache_info = scan_cache_dir(cache_dir)
        for repo_info in cache_info.repos:
            if repo_info.repo_id == repo_id:
                for rev_info in repo_info.revisions:
                    if revision in rev_info.refs:
                        strategy = cache_info.delete_revisions(rev_info.commit_hash)
                        print(f"  Cleaning up cache: freeing "
                              f"{strategy.expected_freed_size_str}")
                        strategy.execute()
                        return
        print(f"  No cached files found for {repo_id} @ {revision}")
    except Exception as e:
        print(f"  Cache cleanup failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Eval pipeline
# ---------------------------------------------------------------------------

def build_eval_blocks(tokenizer, dataset, seq_len, max_tokens, rng):
    """Concatenate text, tokenize, chunk into seq_len blocks.
    Standard packed-eval pipeline. Returns (n_blocks, seq_len) int64 tensor.

    Texts are shuffled with the global RNG so that the eval sample is
    deterministic given SEED but not always taken from the start of the
    dataset (which in wikitext is biased toward a single article).
    """
    texts = [x["text"] for x in dataset if x["text"].strip()]
    rng.shuffle(texts)

    eos = tokenizer.eos_token_id
    ids = []
    for t in texts:
        toks = tokenizer.encode(t, add_special_tokens=False)
        ids.extend(toks)
        if eos is not None:
            ids.append(eos)
        if len(ids) >= max_tokens + seq_len:
            break

    # Trim to a multiple of seq_len, capped at max_tokens.
    cap = (max_tokens // seq_len) * seq_len if max_tokens else len(ids)
    n_full = min(len(ids), cap) // seq_len
    if n_full == 0:
        raise ValueError(
            f"Eval corpus is too small: got {len(ids)} tokens, need "
            f"at least seq_len={seq_len}."
        )
    ids = ids[: n_full * seq_len]
    arr = np.array(ids, dtype=np.int64).reshape(n_full, seq_len)
    return torch.from_numpy(arr)


@torch.no_grad()
def evaluate_loss(model, blocks, batch_size):
    """Packed cross-entropy eval. Returns (mean_loss_in_nats, total_tokens).

    Each block of length T contributes T-1 next-token predictions. We
    accumulate total NLL (sum, not mean) and total token count, then
    divide once at the end. This is more numerically faithful than
    averaging per-block means, and handles the last-batch case correctly
    even if it is smaller than batch_size.
    """
    total_nll = 0.0
    total_tokens = 0
    n_blocks = blocks.shape[0]
    for i in range(0, n_blocks, batch_size):
        batch = blocks[i:i + batch_size].to(DEVICE)
        logits = model(batch).logits  # (B, T, V)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        # Cast logits to fp32 for the CE reduction. Critical when running
        # the forward pass in bf16: the log-softmax is the only spot
        # where bf16 loses meaningful precision for eval numbers.
        loss_sum = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += loss_sum.item()
        total_tokens += shift_labels.numel()
        del logits, shift_logits, shift_labels, loss_sum
    return total_nll / total_tokens, total_tokens


# ---------------------------------------------------------------------------
# Per-step result IO (resume support)
# ---------------------------------------------------------------------------

def _result_path(out_dir, step):
    return Path(out_dir) / f"step_{step}.npz"


def save_result(out_dir, step, mean_loss, total_tokens):
    np.savez(
        _result_path(out_dir, step),
        step=np.array(step),
        mean_loss=np.array(mean_loss),
        total_tokens=np.array(total_tokens),
    )


def load_result(path):
    d = np.load(path)
    return {
        "step": int(d["step"]),
        "mean_loss": float(d["mean_loss"]),
        "total_tokens": int(d["total_tokens"]),
    }


def write_summary_json(out_dir, results, eval_meta):
    payload = {
        "model": OLMO_MODEL,
        "eval": eval_meta,
        "checkpoints": sorted(results, key=lambda r: r["step"]),
    }
    with open(Path(out_dir) / "loss_curve.json", "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_loss_sweep(checkpoints, blocks, dtype, batch_size,
                    cache_dir, out_dir, eval_meta):
    """Load and evaluate each checkpoint sequentially. Drops the model
    after each step; deletes the on-disk checkpoint cache after each
    step. Resumes from per-step .npz files on restart."""
    results = []
    for step in checkpoints:
        # ---- Resume from disk if already computed ----
        rpath = _result_path(out_dir, step)
        if rpath.exists():
            print(f"\n=== OLMo-2-1B step {step}: loaded from cache ===")
            r = load_result(rpath)
            print(f"    mean_loss = {r['mean_loss']:.4f}, "
                  f"ppl = {math.exp(r['mean_loss']):.2f}, "
                  f"tokens = {r['total_tokens']}")
            results.append(r)
            continue

        print(f"\n=== OLMo-2-1B at step {step} ===")
        repo = (EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP
                else OLMO_MODEL)
        revision = step_to_revision(step)
        try:
            model = load_olmo_at_step(step, dtype, cache_dir=cache_dir)
        except Exception as e:
            print(f"  FAILED to load step {step}: {e}")
            print(f"  Skipping. Other checkpoints will continue.")
            continue

        try:
            print(f"  Evaluating on {blocks.shape[0]} blocks of "
                  f"{blocks.shape[1]} tokens (batch_size={batch_size}) ...")
            mean_loss, total_tokens = evaluate_loss(model, blocks, batch_size)
            ppl = math.exp(mean_loss)
            print(f"    mean_loss = {mean_loss:.4f}, "
                  f"ppl = {ppl:.2f}, tokens = {total_tokens}")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cleanup_checkpoint_cache(repo, revision, cache_dir=cache_dir)

        save_result(out_dir, step, mean_loss, total_tokens)
        results.append({
            "step": int(step),
            "mean_loss": float(mean_loss),
            "total_tokens": int(total_tokens),
        })
        write_summary_json(out_dir, results, eval_meta)

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_loss_curve(results, out_path, model_name, eval_meta):
    results = sorted(results, key=lambda r: r["step"])
    steps = np.array([r["step"] for r in results])
    losses = np.array([r["mean_loss"] for r in results])
    ppls = np.exp(losses)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(steps, losses, "o-", linewidth=2, markersize=6, color="C0")
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss (nats / token)")
    ax.set_title("(a) Loss")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(steps, ppls, "o-", linewidth=2, markersize=6, color="C3")
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Perplexity")
    ax.set_title("(b) Perplexity")
    ax.grid(alpha=0.3, which="both")

    eval_blurb = (f"{eval_meta['dataset']}/{eval_meta['dataset_config']}"
                  f"[{eval_meta['dataset_split']}], "
                  f"seq_len={eval_meta['seq_len']}, "
                  f"{eval_meta['n_blocks']} blocks")
    fig.suptitle(f"Loss curve: {model_name}\n{eval_blurb}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=int, nargs="+",
                        default=DEFAULT_CHECKPOINTS,
                        help="OLMo-2-1B training steps to evaluate.")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Block length. OLMo-2 was pretrained at 4096; "
                             "2048 is a fine speed/fidelity tradeoff.")
    parser.add_argument("--max-tokens", type=int, default=200_000,
                        help="Cap on total eval tokens per checkpoint.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                        default="bfloat16")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help="HF cache dir; defaults to $HF_HOME if unset.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("olmo_loss_curve"))
    out_dir.mkdir(exist_ok=True, parents=True)

    dtype = {"float32": torch.float32,
             "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    print(f"Model: {OLMO_MODEL}")
    print(f"Checkpoints: {args.checkpoints}")
    print(f"Eval data: {args.dataset}/{args.dataset_config}"
          f"[{args.dataset_split}]")
    print(f"seq_len={args.seq_len}, max_tokens={args.max_tokens}, "
          f"batch_size={args.batch_size}, dtype={args.dtype}")
    print(f"Device: {DEVICE}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Output dir: {out_dir}")

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print(f"\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        OLMO_MODEL, cache_dir=args.cache_dir,
    )

    print(f"Loading {args.dataset}/{args.dataset_config}"
          f"[{args.dataset_split}] ...")
    ds = load_dataset(args.dataset, args.dataset_config,
                      split=args.dataset_split)
    blocks = build_eval_blocks(tokenizer, ds, args.seq_len,
                                args.max_tokens, rng)
    print(f"Built {blocks.shape[0]} blocks of {blocks.shape[1]} tokens "
          f"({blocks.numel()} total tokens).")

    eval_meta = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "seq_len": int(args.seq_len),
        "n_blocks": int(blocks.shape[0]),
        "total_tokens": int(blocks.numel()),
        "dtype": args.dtype,
        "seed": SEED,
    }

    # ---- Sweep ----
    results = run_loss_sweep(
        args.checkpoints, blocks, dtype, args.batch_size,
        args.cache_dir, out_dir, eval_meta,
    )

    if not results:
        raise SystemExit("No checkpoints loaded successfully.")

    results.sort(key=lambda r: r["step"])

    # ---- Plot ----
    print(f"\nMaking plots ...")
    plot_loss_curve(results, out_dir / "loss_curve.png",
                    OLMO_MODEL, eval_meta)
    print(f"  loss_curve.png")

    # ---- Final JSON write ----
    write_summary_json(out_dir, results, eval_meta)
    print(f"  loss_curve.json")

    # ---- Console table ----
    print(f"\n--- Loss curve: {OLMO_MODEL} ---")
    print(f"{'step':>10} {'loss':>10} {'ppl':>12} {'tokens':>12}")
    for r in results:
        print(f"{r['step']:>10} "
              f"{r['mean_loss']:>10.4f} "
              f"{math.exp(r['mean_loss']):>12.2f} "
              f"{r['total_tokens']:>12}")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
