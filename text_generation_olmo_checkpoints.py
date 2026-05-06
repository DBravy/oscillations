"""
Generation sweep across OLMo-2-1B training checkpoints.

OLMo port companion to phase_space_olmo_checkpoints.py. At each
checkpoint, generates continuations for a fixed cross-domain prompt
set covering factual retrieval, arithmetic, code, reasoning, and
open-ended completion. The point is to read off how each capability
emerges (or fails to emerge) as a function of training step.

Mirrors checkpoint-handling conventions from the phase-space script:
  - Step <= 37000: load from allenai/OLMo-2-0425-1B-early-training
  - Step >  37000: load from allenai/OLMo-2-0425-1B
  - Revision string: f"stage1-step{step}-tokens{tokens_b}B" with
    tokens_b = ceil(step * 2048 * 1024 / 1e9)
  - Per-step JSON files for resume on restart
  - HF cache cleanup after each checkpoint to keep disk bounded
  - Skips checkpoints that fail to load

Generation strategy:
  Greedy by default. Greedy decoding is deterministic given (model,
  prompt), which makes cross-checkpoint comparisons cleanly readable
  (any difference is the model, not the seed). Sampling is available
  via --temperature > 0 for runs where you want to see the
  distribution rather than a single completion. Sampling is seeded so
  that the same prompt produces the same text given the same model.

Prompt categories:
  fact         Closed-form factual retrieval.
  math         Arithmetic and word problems with a single answer.
  definition   Should-be-canonical short explanations.
  code         Code completion in Python.
  reasoning    Multi-step problems where chain of thought helps.
  creative     Open-ended; quality judged on coherence and prose.
  completion   Short linguistic completions (general fluency).

Outputs in olmo_generations/:
  step_<step>.json      Per-checkpoint generations (all prompts).
  generations.json      Aggregated, written incrementally.
  generations.md        Human-readable Markdown grouped by prompt,
                        showing each checkpoint's continuation
                        underneath the prompt for easy scrolling.

Usage:
  python text_generation_olmo_checkpoints.py
  python text_generation_olmo_checkpoints.py --temperature 0.7
  python text_generation_olmo_checkpoints.py \\
      --checkpoints 0 1000 5000 10000 37000 100000 1000000 \\
      --max-new-tokens 120
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
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
# Prompt set
# ---------------------------------------------------------------------------
#
# Mix of closed-form (clear right answer) and open-ended prompts. Kept
# deliberately short so the prompt itself is not doing the heavy
# lifting; what we want to see is what the model adds.
#
# Categories:
#   fact, math, definition, code, reasoning, creative, completion
#
# The (category, id, prompt) triple is stable; if you want to tweak
# the set, append rather than reorder so old per-step JSONs remain
# diff-able.

PROMPTS = [
    # ---- Factual retrieval (closed) ----
    {"id": "fact_capital", "category": "fact",
     "prompt": "The capital of France is"},
    {"id": "fact_planet", "category": "fact",
     "prompt": "The largest planet in our solar system is"},
    {"id": "fact_freeze", "category": "fact",
     "prompt": "Water freezes at a temperature of"},
    {"id": "fact_gold", "category": "fact",
     "prompt": "The chemical symbol for gold is"},
    {"id": "fact_moon", "category": "fact",
     "prompt": "The first person to walk on the moon was"},
    {"id": "fact_shakespeare", "category": "fact",
     "prompt": "Romeo and Juliet is a tragedy written by"},

    # ---- Math (closed) ----
    {"id": "math_basic", "category": "math",
     "prompt": "2 + 2 ="},
    {"id": "math_sqrt", "category": "math",
     "prompt": "The square root of 144 is"},
    {"id": "math_word", "category": "math",
     "prompt": "Q: What is 17 times 6?\nA:"},

    # ---- Definitions ----
    {"id": "def_photosynthesis", "category": "definition",
     "prompt": "Photosynthesis is the process by which"},
    {"id": "def_democracy", "category": "definition",
     "prompt": "Democracy is a system of government in which"},

    # ---- Code ----
    {"id": "code_fib", "category": "code",
     "prompt": "def fibonacci(n):\n    "},
    {"id": "code_reverse", "category": "code",
     "prompt": "# Python function to reverse a string\n"
               "def reverse_string(s):\n    "},

    # ---- Reasoning ----
    {"id": "reasoning_simple", "category": "reasoning",
     "prompt": "If Alice has 3 apples and Bob gives her 5 more apples, "
               "then Alice has"},
    {"id": "reasoning_cot", "category": "reasoning",
     "prompt": ("Q: Roger has 5 tennis balls. He buys 2 more cans of "
                "tennis balls. Each can has 3 tennis balls. How many "
                "tennis balls does he have now?\n"
                "A: Let's think step by step.")},

    # ---- Creative / open-ended ----
    {"id": "creative_story", "category": "creative",
     "prompt": "Once upon a time, in a forest far away,"},
    {"id": "creative_sensory", "category": "creative",
     "prompt": "The smell of rain on warm asphalt"},
    {"id": "creative_poem", "category": "creative",
     "prompt": "A short poem about autumn:\n"},

    # ---- Short fluency completion ----
    {"id": "compl_door", "category": "completion",
     "prompt": "She opened the door and"},
    {"id": "compl_lesson", "category": "completion",
     "prompt": "The most important lesson I have ever learned is"},
]


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
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_one(model, tokenizer, prompt, max_new_tokens,
                  temperature, top_p, repetition_penalty):
    """Generate a single continuation. Greedy if temperature == 0,
    otherwise nucleus sampling at the given temperature.

    Returns the continuation string only (not prompt + continuation),
    so the markdown / JSON output is easier to scan."""
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    prompt_len = input_ids.shape[1]

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )
    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_generations_for_step(model, tokenizer, prompts, args):
    """Run all prompts against a loaded model. Returns a list of dicts."""
    out = []
    for p in prompts:
        # Re-seed before each prompt so sampling stays deterministic
        # per (step, prompt) pair regardless of prompt order or how
        # many tokens the previous generation drew.
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        completion = generate_one(
            model, tokenizer, p["prompt"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        out.append({
            "id": p["id"],
            "category": p["category"],
            "prompt": p["prompt"],
            "completion": completion,
        })
    return out


# ---------------------------------------------------------------------------
# Per-step result IO (resume support)
# ---------------------------------------------------------------------------

def _result_path(out_dir, step):
    return Path(out_dir) / f"step_{step}.json"


def save_result(out_dir, step, generations, gen_meta):
    payload = {
        "step": int(step),
        "gen_meta": gen_meta,
        "generations": generations,
    }
    with open(_result_path(out_dir, step), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_result(path):
    with open(path) as f:
        return json.load(f)


def write_summary_json(out_dir, results, gen_meta):
    payload = {
        "model": OLMO_MODEL,
        "gen_meta": gen_meta,
        "checkpoints": sorted(results, key=lambda r: r["step"]),
    }
    with open(Path(out_dir) / "generations.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Markdown report (prompt-major: one section per prompt, all steps inside)
# ---------------------------------------------------------------------------

def write_markdown_report(out_dir, results, gen_meta):
    """Markdown report grouped by prompt. Reads top-down as: here is
    prompt X, watch how the continuation evolves with training step."""
    results = sorted(results, key=lambda r: r["step"])

    # Index: prompt id -> list of (step, completion) ordered by step.
    by_prompt = {}
    prompt_meta = {}
    for r in results:
        for g in r["generations"]:
            by_prompt.setdefault(g["id"], []).append(
                (r["step"], g["completion"])
            )
            prompt_meta[g["id"]] = (g["category"], g["prompt"])

    lines = []
    lines.append(f"# OLMo-2-1B generation sweep")
    lines.append("")
    lines.append(f"Model: `{OLMO_MODEL}`")
    lines.append(f"Decoding: temperature={gen_meta['temperature']}, "
                 f"top_p={gen_meta['top_p']}, "
                 f"max_new_tokens={gen_meta['max_new_tokens']}, "
                 f"repetition_penalty={gen_meta['repetition_penalty']}")
    lines.append(f"Steps evaluated: "
                 f"{[r['step'] for r in results]}")
    lines.append("")

    # Group prompts by category in stable order from PROMPTS.
    cat_order = []
    seen = set()
    for p in PROMPTS:
        if p["category"] not in seen:
            cat_order.append(p["category"])
            seen.add(p["category"])

    for cat in cat_order:
        lines.append(f"## Category: {cat}")
        lines.append("")
        for p in PROMPTS:
            if p["category"] != cat:
                continue
            if p["id"] not in by_prompt:
                continue
            cat_, prompt_text = prompt_meta[p["id"]]
            lines.append(f"### `{p['id']}`")
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append("```")
            lines.append(prompt_text)
            lines.append("```")
            lines.append("")
            for step, completion in by_prompt[p["id"]]:
                lines.append(f"**step {step}:**")
                lines.append("")
                lines.append("```")
                lines.append(completion.rstrip())
                lines.append("```")
                lines.append("")

    with open(Path(out_dir) / "generations.md", "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_generation_sweep(checkpoints, tokenizer, args, out_dir, gen_meta):
    """Load and generate at each checkpoint sequentially. Drops the
    model after each step; deletes the on-disk checkpoint cache after
    each step. Resumes from per-step JSON files on restart."""
    dtype = {"float32": torch.float32,
             "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    results = []
    for step in checkpoints:
        # ---- Resume from disk if already computed ----
        rpath = _result_path(out_dir, step)
        if rpath.exists():
            print(f"\n=== OLMo-2-1B step {step}: loaded from cache ===")
            results.append(load_result(rpath))
            continue

        print(f"\n=== OLMo-2-1B at step {step} ===")
        repo = (EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP
                else OLMO_MODEL)
        revision = step_to_revision(step)
        try:
            model = load_olmo_at_step(step, dtype, cache_dir=args.cache_dir)
        except Exception as e:
            print(f"  FAILED to load step {step}: {e}")
            print(f"  Skipping. Other checkpoints will continue.")
            continue

        try:
            print(f"  Generating {len(PROMPTS)} prompts ...")
            gens = run_generations_for_step(model, tokenizer, PROMPTS, args)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cleanup_checkpoint_cache(repo, revision, cache_dir=args.cache_dir)

        save_result(out_dir, step, gens, gen_meta)
        results.append({
            "step": int(step),
            "gen_meta": gen_meta,
            "generations": gens,
        })
        write_summary_json(out_dir, results, gen_meta)
        write_markdown_report(out_dir, results, gen_meta)

        # ---- Mini console preview ----
        # Pick three prompts to print so progress is visible without
        # flooding the terminal. The choice is fixed across runs.
        preview_ids = {"fact_capital", "math_basic", "creative_story"}
        for g in gens:
            if g["id"] in preview_ids:
                preview = g["completion"].strip().splitlines()[0][:120]
                print(f"    [{g['id']}] {preview!r}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=int, nargs="+",
                        default=DEFAULT_CHECKPOINTS,
                        help="OLMo-2-1B training steps to evaluate.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy. > 0 enables nucleus sampling.")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="1.0 = off. Untrained checkpoints often "
                             "loop; 1.1-1.2 helps but biases comparisons.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                        default="bfloat16")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help="HF cache dir; defaults to $HF_HOME if unset.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("olmo_generations"))
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Model: {OLMO_MODEL}")
    print(f"Checkpoints: {args.checkpoints}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"max_new_tokens={args.max_new_tokens}, "
          f"temperature={args.temperature}, top_p={args.top_p}, "
          f"repetition_penalty={args.repetition_penalty}")
    print(f"Device: {DEVICE}, dtype={args.dtype}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Output dir: {out_dir}")

    print(f"\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        OLMO_MODEL, cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_meta = {
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "repetition_penalty": float(args.repetition_penalty),
        "dtype": args.dtype,
        "seed": SEED,
    }

    # ---- Sweep ----
    results = run_generation_sweep(
        args.checkpoints, tokenizer, args, out_dir, gen_meta,
    )

    if not results:
        raise SystemExit("No checkpoints loaded successfully.")

    results.sort(key=lambda r: r["step"])

    # ---- Final writes ----
    print(f"\nWriting outputs ...")
    write_summary_json(out_dir, results, gen_meta)
    print(f"  generations.json")
    write_markdown_report(out_dir, results, gen_meta)
    print(f"  generations.md")

    print(f"\nDone. {len(results)} checkpoints x {len(PROMPTS)} prompts.")
    print(f"Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
