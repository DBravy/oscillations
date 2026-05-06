"""
Per-unit phase-amplitude coupling (PAC) probe with checkpoint-sweep
support, for Pythia (GPT-NeoX), OLMo-2, and LLaMA-style HF models.

Sweep orchestration is the main change vs the single-checkpoint version
of pac_probe_general.py. The PAC math (Tort modulation index,
phase-randomized null) is unchanged.

Output layout
-------------
A single invocation processes one model across one or many revisions.
Each checkpoint gets its own subdirectory:

  {out_dir}/{model_slug}/{label}/
    pac_data.npz
    pac_summary.json
    pac_matrix.png
    pac_per_unit.png
    pac_unit_examples.png

After all checkpoints are processed (or skipped), a sweep roll-up is
written using whatever pac_summary.json files currently exist on disk:

  {out_dir}/{model_slug}/sweep_summary.json
  {out_dir}/{model_slug}/sweep_trajectory.png

Resumability
------------
A checkpoint is "done" iff its pac_summary.json exists. On rerun, done
checkpoints are skipped without loading the model. To force redo, delete
the relevant pac_summary.json. JSON writes are atomic (tmp + rename) so
a crash mid-write cannot leave a partial summary that would later be
mistaken for completion. A checkpoint that crashes mid-PAC-computation
is reprocessed from scratch on the next run; in-memory progress within
a single checkpoint is not persisted.

Cache cleanup
-------------
After each checkpoint completes (or is skipped), the HF cache entry for
its specific revision is deleted via huggingface_hub.scan_cache_dir.
Cleanup is per (repo, revision), so other revisions of the same repo
are preserved. Pass --no-cleanup to keep everything cached.

OLMo-2-1B repo dispatch
-----------------------
When --model is allenai/OLMo-2-0425-1B and --checkpoints includes any
step <= 37000, those revisions are loaded from
allenai/OLMo-2-0425-1B-early-training automatically. Always pass the
main repo as --model.

Checkpoint specification
------------------------
Three mutually exclusive ways to pick revisions:

  --checkpoints 1000 5000 10000 ...
      Integer training steps. Auto-translated to revisions for Pythia
      (step{N}) and OLMo-2-1B (stage1-step{N}-tokens{M}B with repo
      dispatch). Errors for other models.

  --revisions step1000 step5000 ...
      Explicit revision strings, used as-is on the given --model.

  --revision step1000   (singular, back-compat)
      Single revision.

  (none)
      Use the model's default revision (main).

Architecture handling (unchanged)
---------------------------------
LLaMA-style pre-norm: hook input_layernorm and post_attention_layernorm
of every block; 2 snapshots/block.
GPT-2: same pattern with ln_1, ln_2.
Pythia parallel residual (default): 1 snapshot/block via forward_pre on
the layer module. Falls back to LayerNorms for serial Pythia configs.
OLMo-2 (post-norm): forward_pre on self_attn and mlp; 2 snapshots/block.

max_bin clamping
----------------
For each checkpoint, --max-bin is clamped to (n_sublayers // 2) - 1 if
larger. This means a single sweep can mix architectures with different
n_sublayers safely.

Usage
-----
  # Pythia-410M, six checkpoints, max_bin 10
  python pac_probe_general.py \\
      --model EleutherAI/pythia-410m \\
      --checkpoints 1000 5000 10000 50000 100000 143000 \\
      --max-bin 10 --out-dir pac_sweep

  # Resume the same sweep (already-done checkpoints are skipped)
  python pac_probe_general.py \\
      --model EleutherAI/pythia-410m \\
      --checkpoints 1000 5000 10000 50000 100000 143000 \\
      --max-bin 10 --out-dir pac_sweep

  # OLMo-2-1B, with auto repo dispatch for early steps
  python pac_probe_general.py \\
      --model allenai/OLMo-2-0425-1B \\
      --checkpoints 1000 5000 10000 100000 1000000 \\
      --max-bin 14 --out-dir pac_sweep
"""

import argparse
import json
import math
import os
import re
import traceback
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLMO2_1B_MAIN = "allenai/OLMo-2-0425-1B"
OLMO2_1B_EARLY = "allenai/OLMo-2-0425-1B-early-training"
OLMO2_1B_EARLY_MAX_STEP = 37000


# ---------------------------------------------------------------------------
# Naming and dispatch
# ---------------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name))


def is_pythia(model_name):
    return "pythia" in model_name.lower()


def is_olmo2_1b(model_name):
    return model_name in (OLMO2_1B_MAIN, OLMO2_1B_EARLY)


def step_to_revision_olmo2(step):
    """Returns (effective_repo, revision_string) for OLMo-2-1B."""
    tokens_b = math.ceil(step * 2048 * 1024 / 1_000_000_000)
    rev = f"stage1-step{step}-tokens{tokens_b}B"
    if step <= OLMO2_1B_EARLY_MAX_STEP:
        return OLMO2_1B_EARLY, rev
    return OLMO2_1B_MAIN, rev


def expand_checkpoints(model_name, checkpoints):
    """For each step, return a list of (effective_model, revision, label)."""
    out = []
    if is_pythia(model_name):
        for step in checkpoints:
            out.append((model_name, f"step{step}", f"step{step}"))
    elif is_olmo2_1b(model_name):
        for step in checkpoints:
            effective, rev = step_to_revision_olmo2(step)
            out.append((effective, rev, f"step{step}"))
    else:
        raise ValueError(
            f"Don't know how to translate --checkpoints to revisions for "
            f"model {model_name!r}. Use --revisions with explicit "
            f"revision strings instead."
        )
    return out


# ---------------------------------------------------------------------------
# HF cache cleanup
# ---------------------------------------------------------------------------

def fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def delete_cached_revision(repo_id, revision, cache_dir=None):
    """Delete one revision of one repo from the HF cache. Returns bytes
    freed (0 if nothing was deleted or the call failed)."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return 0
    try:
        cache_info = scan_cache_dir(cache_dir=cache_dir)
    except Exception as e:
        print(f"    [cleanup] scan_cache_dir failed: {e}")
        return 0

    target_hashes = []
    wanted = revision if revision is not None else "main"
    for repo in cache_info.repos:
        if repo.repo_id != repo_id:
            continue
        for rev in repo.revisions:
            refs = set(rev.refs)
            if wanted in refs or rev.commit_hash == wanted:
                target_hashes.append(rev.commit_hash)

    if not target_hashes:
        return 0
    try:
        strategy = cache_info.delete_revisions(*target_hashes)
        freed = strategy.expected_freed_size
        strategy.execute()
        return freed
    except Exception as e:
        print(f"    [cleanup] delete failed for {repo_id}@{wanted}: {e}")
        return 0


# ---------------------------------------------------------------------------
# Atomic JSON
# ---------------------------------------------------------------------------

def write_atomic_json(path, data):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(model_name, revision=None, cache_dir=None):
    msg = f"  Loading {model_name}"
    if revision:
        msg += f" @ {revision}"
    msg += " ..."
    print(msg)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_accel = (
        torch.cuda.is_available() or torch.backends.mps.is_available()
    )
    dtype = torch.float16 if use_accel else torch.float32
    device_map = "auto" if use_accel else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map=device_map,
        output_hidden_states=False,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Architecture detection and capture plan
# ---------------------------------------------------------------------------

def detect_architecture(model):
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return "pythia", model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "gpt2", model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        block0 = model.model.layers[0]
        if hasattr(block0, "post_feedforward_layernorm"):
            return "olmo2", model.model.layers
        if (hasattr(block0, "input_layernorm")
                and hasattr(block0, "post_attention_layernorm")):
            return "llama", model.model.layers
    raise RuntimeError(
        f"Unsupported architecture: {type(model).__name__}."
    )


def build_capture_plan(model):
    arch, blocks = detect_architecture(model)
    plan = []
    if arch == "llama":
        for b in blocks:
            plan.append((b.input_layernorm, "fwd"))
            plan.append((b.post_attention_layernorm, "fwd"))
    elif arch == "gpt2":
        for b in blocks:
            plan.append((b.ln_1, "fwd"))
            plan.append((b.ln_2, "fwd"))
    elif arch == "olmo2":
        for b in blocks:
            plan.append((b.self_attn, "fwd_pre"))
            plan.append((b.mlp, "fwd_pre"))
    elif arch == "pythia":
        use_parallel = getattr(
            model.config, "use_parallel_residual", True
        )
        if use_parallel:
            for b in blocks:
                plan.append((b, "fwd_pre"))
        else:
            for b in blocks:
                plan.append((b.input_layernorm, "fwd"))
                plan.append((b.post_attention_layernorm, "fwd"))
    return arch, plan


def describe_capture(arch, n_layers, n_sublayers, use_parallel=None):
    parts = [f"arch={arch}", f"n_layers={n_layers}",
             f"n_sublayers={n_sublayers}"]
    if arch == "pythia" and use_parallel is not None:
        parts.append(f"use_parallel_residual={use_parallel}")
    return ", ".join(parts)


def collect_streams(model, tokenizer, texts, seq_len):
    arch, plan = build_capture_plan(model)
    device = next(model.parameters()).device
    out = []
    skipped = 0
    for i, text in enumerate(texts):
        enc = tokenizer(
            text, return_tensors="pt",
            max_length=seq_len, truncation=True,
        )
        ids = enc["input_ids"].to(device)
        if ids.shape[1] < 8:
            skipped += 1
            continue

        captures = []
        hooks = []

        def make_fwd(c):
            def fn(m, inp, out_):
                c.append(inp[0].detach())
            return fn

        def make_pre(c):
            def fn(m, args, kwargs):
                if args:
                    c.append(args[0].detach())
                else:
                    c.append(kwargs["hidden_states"].detach())
            return fn

        for module, kind in plan:
            if kind == "fwd":
                hooks.append(
                    module.register_forward_hook(make_fwd(captures))
                )
            else:
                hooks.append(
                    module.register_forward_pre_hook(
                        make_pre(captures), with_kwargs=True,
                    )
                )

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
            print(f"    {i + 1} / {len(texts)} samples")

    if skipped:
        print(f"    ({skipped} sample(s) skipped, < 8 tokens)")
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Band isolation, modulation index (unchanged from pac_probe_smollm2.py)
# ---------------------------------------------------------------------------

def isolate_bin_signal(signal, bin_idx):
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


def phase_randomize_signal(signal, rng):
    L = len(signal)
    spec = np.fft.rfft(signal)
    n_pos = len(spec)
    new_spec = np.copy(spec)
    for k in range(1, n_pos):
        if L % 2 == 0 and k == n_pos - 1:
            continue
        mag = np.abs(spec[k])
        new_phase = rng.uniform(-np.pi, np.pi)
        new_spec[k] = mag * np.exp(1j * new_phase)
    return np.fft.irfft(new_spec, n=L)


def pac_for_signal_pooled(streams_unit, bin_pairs, n_phase_bins=18):
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
    if rng is None:
        rng = np.random.default_rng(0)
    null = np.zeros((n_runs, len(bin_pairs)))
    for r in range(n_runs):
        randomized = np.zeros_like(streams_unit)
        for n in range(streams_unit.shape[0]):
            randomized[n] = phase_randomize_signal(streams_unit[n], rng)
        result = pac_for_signal_pooled(
            randomized, bin_pairs, n_phase_bins=n_phase_bins,
        )
        for j, key in enumerate(bin_pairs):
            null[r, j] = result[key]
    return null


# ---------------------------------------------------------------------------
# Per-checkpoint plots
# ---------------------------------------------------------------------------

def plot_pac_matrix_summary(per_unit, bin_pairs, max_bin, save_path,
                             model_name):
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
        f"Mean PAC modulation index across units ({model_name})",
        fontsize=11,
    )
    ax.plot([0.5, max_bin + 0.5], [0.5, max_bin + 0.5],
            color="white", linewidth=0.5, linestyle="--", alpha=0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_pac_per_unit(per_unit, bin_pairs, save_path, model_name):
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
    ax.set_title(f"Per-unit PAC, {model_name}", fontsize=11)
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
# Per-checkpoint pipeline
# ---------------------------------------------------------------------------

def pac_for_one_checkpoint(model_name, revision, label, ckpt_dir,
                            texts, args):
    """Run PAC analysis on one checkpoint. Writes outputs into ckpt_dir.

    Returns one of: 'done', 'skipped', 'load_failed', 'error'."""
    summary_path = ckpt_dir / "pac_summary.json"
    if summary_path.exists():
        print(f"  [skip] {label}: pac_summary.json exists")
        return "skipped"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        model, tokenizer = load_model(
            model_name, revision=revision, cache_dir=args.cache_dir,
        )
    except Exception as e:
        print(f"  [load failed] {label}: {e}")
        return "load_failed"

    try:
        arch, plan = build_capture_plan(model)
        if arch == "pythia":
            n_layers = len(model.gpt_neox.layers)
            use_parallel = getattr(
                model.config, "use_parallel_residual", True
            )
        elif arch == "gpt2":
            n_layers = len(model.transformer.h)
            use_parallel = None
        else:
            n_layers = len(model.model.layers)
            use_parallel = None
        print(
            "  Architecture: "
            + describe_capture(arch, n_layers, len(plan), use_parallel)
        )

        print("  Collecting streams ...")
        streams = collect_streams(
            model, tokenizer, texts, args.seq_len,
        )
        print(f"    shape: {streams.shape}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        N, L, D = streams.shape

        max_bin_cap = max(L // 2 - 1, 1)
        if args.max_bin > max_bin_cap:
            print(
                f"  Clamping max_bin {args.max_bin} -> {max_bin_cap} "
                f"(L={L}, Nyquist bin={L // 2})"
            )
            max_bin = max_bin_cap
        else:
            max_bin = args.max_bin

        bin_pairs = [
            (s, f)
            for s in range(1, max_bin + 1)
            for f in range(1, max_bin + 1)
            if s != f
        ]
        print(
            f"  Scanning {len(bin_pairs)} bin pairs per unit "
            f"(max_bin={max_bin})"
        )

        per_unit = np.zeros((D, len(bin_pairs)))
        print(f"  Computing PAC across {D} units ...")
        for u in range(D):
            result = pac_for_signal_pooled(
                streams[:, :, u], bin_pairs,
                n_phase_bins=args.n_phase_bins,
            )
            for j, key in enumerate(bin_pairs):
                per_unit[u, j] = result[key]
            if (u + 1) % 256 == 0:
                print(f"    {u + 1} / {D}")

        sample_units = np.random.default_rng(args.seed + 1).choice(
            D, size=min(args.n_null_units, D), replace=False,
        )
        print(f"  Computing null on {len(sample_units)} units ...")
        null_per_pair = np.zeros((
            len(sample_units), args.n_shuffle, len(bin_pairs),
        ))
        for i, u in enumerate(sample_units):
            null_per_pair[i] = phase_randomized_null(
                streams[:, :, u], bin_pairs, args.n_shuffle,
                n_phase_bins=args.n_phase_bins,
                rng=np.random.default_rng(args.seed + 100 + i),
            )

        null_flat = null_per_pair.reshape(-1, len(bin_pairs))
        null_p95 = np.percentile(null_flat, 95, axis=0)
        null_p99 = np.percentile(null_flat, 99, axis=0)
        null_mean = null_flat.mean(axis=0)

        sig_mask = per_unit > null_p95[None, :]
        sig_strict = per_unit > null_p99[None, :]

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
                    "ratio_to_null":
                        float(mi / max(null_p95[j], 1e-6)),
                })
        findings.sort(key=lambda f: -f["mi"])

        display_name = f"{model_name.split('/')[-1]}@{label}"

        plot_pac_matrix_summary(
            per_unit, bin_pairs, max_bin,
            ckpt_dir / "pac_matrix.png", display_name,
        )
        plot_pac_per_unit(
            per_unit, bin_pairs,
            ckpt_dir / "pac_per_unit.png", display_name,
        )
        plot_unit_examples(
            streams, findings, args.n_top_evidence,
            args.n_phase_bins,
            ckpt_dir / "pac_unit_examples.png",
        )

        np.savez(
            ckpt_dir / "pac_data.npz",
            per_unit=per_unit,
            null_p95=null_p95,
            null_p99=null_p99,
            null_mean=null_mean,
            sig_mask=sig_mask,
            sig_strict=sig_strict,
            bin_pairs=np.array(bin_pairs),
            sample_units=sample_units,
        )

        step = None
        m = re.match(r"step(\d+)", label)
        if m:
            step = int(m.group(1))

        summary = {
            "model":          model_name,
            "revision":       revision,
            "label":          label,
            "step":           step,
            "architecture":   arch,
            "n_layers":       int(n_layers),
            "n_samples":      int(N),
            "n_sublayers":    int(L),
            "d_model":        int(D),
            "max_bin":        int(max_bin),
            "max_bin_requested": int(args.max_bin),
            "n_phase_bins":   args.n_phase_bins,
            "n_shuffle":      args.n_shuffle,
            "n_null_units":   int(len(sample_units)),
            "max_mi":         float(per_unit.max()),
            "median_mi":      float(np.median(per_unit)),
            "null_mean_mi":   float(null_mean.mean()),
            "null_p95_mean":  float(null_p95.mean()),
            "null_p99_mean":  float(null_p99.mean()),
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
        write_atomic_json(summary_path, summary)
        print(
            f"  [done] {label}: max_mi={per_unit.max():.4f}, "
            f"frac>null95={sig_mask.mean():.3f}"
        )
        return "done"
    except Exception as e:
        print(f"  [error] {label}: {e}")
        traceback.print_exc()
        return "error"


# ---------------------------------------------------------------------------
# Sweep roll-up: trajectory plot and sweep_summary.json
# ---------------------------------------------------------------------------

def collect_completed_summaries(model_dir):
    summaries = []
    if not model_dir.exists():
        return summaries
    for sub in sorted(model_dir.iterdir()):
        if not sub.is_dir():
            continue
        sf = sub / "pac_summary.json"
        if not sf.exists():
            continue
        try:
            with open(sf) as f:
                summaries.append(json.load(f))
        except Exception:
            continue
    summaries.sort(key=lambda s: (
        s.get("step") is None, s.get("step") or 0, s.get("label") or "",
    ))
    return summaries


def plot_sweep_trajectory(summaries, model_name, save_path):
    if len(summaries) < 1:
        return
    has_steps = all(s.get("step") is not None for s in summaries)
    xs = [s["step"] if has_steps else i
          for i, s in enumerate(summaries)]
    xlabel = "training step" if has_steps else "checkpoint index"
    use_symlog = has_steps and any(x and x >= 1000 for x in xs)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(xs, [s["max_mi"] for s in summaries], "o-",
            color="C0", label="max real MI")
    ax.plot(xs, [s["null_p95_mean"] for s in summaries], "k--",
            alpha=0.6, label="null p95 (mean over pairs)")
    ax.plot(xs, [s["null_p99_mean"] for s in summaries], "k:",
            alpha=0.6, label="null p99 (mean over pairs)")
    if use_symlog:
        ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("modulation index")
    ax.set_title("Peak PAC vs null")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(xs, [s["median_mi"] for s in summaries], "o-", color="C2")
    if use_symlog:
        ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("median MI across (unit, pair)")
    ax.set_title("Bulk PAC level")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(xs, [s["fraction_above_null95"] for s in summaries],
            "o-", label="above p95")
    ax.plot(xs, [s["fraction_above_null99"] for s in summaries],
            "s-", label="above p99")
    if use_symlog:
        ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of (unit, pair) entries")
    ax.set_title("Significant entries fraction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    fracs95 = [
        s["n_units_with_any_significant_pair_p95"] / s["d_model"]
        for s in summaries
    ]
    fracs99 = [
        s["n_units_with_any_significant_pair_p99"] / s["d_model"]
        for s in summaries
    ]
    ax.plot(xs, fracs95, "o-", label="any pair > p95")
    ax.plot(xs, fracs99, "s-", label="any pair > p99")
    if use_symlog:
        ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of units")
    ax.set_title("PAC coverage across units")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"PAC trajectory across training: {model_name}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_sweep_summary(model_dir, model_name, summaries):
    rolled = {
        "model": model_name,
        "n_completed_checkpoints": len(summaries),
        "checkpoints": [
            {
                "label":          s.get("label"),
                "step":           s.get("step"),
                "revision":       s.get("revision"),
                "n_sublayers":    s.get("n_sublayers"),
                "max_bin":        s.get("max_bin"),
                "max_mi":         s.get("max_mi"),
                "median_mi":      s.get("median_mi"),
                "null_p95_mean":  s.get("null_p95_mean"),
                "null_p99_mean":  s.get("null_p99_mean"),
                "fraction_above_null95": s.get("fraction_above_null95"),
                "fraction_above_null99": s.get("fraction_above_null99"),
                "n_units_with_any_significant_pair_p95":
                    s.get("n_units_with_any_significant_pair_p95"),
                "n_units_with_any_significant_pair_p99":
                    s.get("n_units_with_any_significant_pair_p99"),
            }
            for s in summaries
        ],
    }
    write_atomic_json(model_dir / "sweep_summary.json", rolled)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None,
                        help="Single revision (back-compat shortcut).")
    parser.add_argument("--revisions", type=str, nargs="+", default=None,
                        help="Explicit list of revision strings.")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None,
                        help="Training step numbers; auto-translated to "
                             "revisions for Pythia and OLMo-2-1B.")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="pac_sweep")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep HF cache after each checkpoint. "
                             "Default: cleanup is ON.")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-bin", type=int, default=20)
    parser.add_argument("--n-phase-bins", type=int, default=18)
    parser.add_argument("--n-shuffle", type=int, default=20)
    parser.add_argument("--n-null-units", type=int, default=8)
    parser.add_argument("--n-top-evidence", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Resolve target list
    if args.checkpoints:
        targets = expand_checkpoints(args.model, args.checkpoints)
    elif args.revisions:
        targets = [(args.model, r, slugify(r)) for r in args.revisions]
    elif args.revision is not None:
        targets = [(args.model, args.revision, slugify(args.revision))]
    else:
        targets = [(args.model, None, "main")]

    out_dir = Path(args.out_dir)
    model_dir = out_dir / slugify(args.model)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:      {args.model}")
    print(f"Output dir: {model_dir.resolve()}")
    print(f"Cache dir:  {args.cache_dir or '(default HF cache)'}")
    print(f"Cleanup:    {'OFF' if args.no_cleanup else 'ON'}")
    print(f"Targets:    {len(targets)} checkpoint(s)")
    for em, rv, lb in targets:
        print(f"  {lb:<14}  {em} @ {rv}")
    print()

    # Wikitext-2 loaded once for the whole sweep
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    print("Loading wikitext-2 ...")
    ds = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="validation",
    )
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]
    print(f"  {len(texts)} samples\n")

    n_done = 0
    n_skipped = 0
    n_load_failed = 0
    n_error = 0
    total_freed = 0

    for idx, (effective_model, revision, label) in enumerate(targets, 1):
        print(f"=== [{idx}/{len(targets)}] {label} "
              f"({effective_model} @ {revision}) ===")
        ckpt_dir = model_dir / label
        status = pac_for_one_checkpoint(
            effective_model, revision, label, ckpt_dir, texts, args,
        )
        if status == "done":
            n_done += 1
        elif status == "skipped":
            n_skipped += 1
        elif status == "load_failed":
            n_load_failed += 1
        else:
            n_error += 1

        # Cleanup is per (repo, revision). Skip cleanup if the load
        # failed (cache may not have been populated, and trying anyway
        # just adds noise to the log).
        if not args.no_cleanup and status != "load_failed":
            freed = delete_cached_revision(
                effective_model, revision, args.cache_dir,
            )
            if freed > 0:
                print(f"  [cleanup] freed {fmt_bytes(freed)} from "
                      f"{effective_model}@{revision}")
                total_freed += freed
        print()

    # Roll-up across whatever's currently on disk
    summaries = collect_completed_summaries(model_dir)
    if summaries:
        write_sweep_summary(model_dir, args.model, summaries)
        plot_sweep_trajectory(
            summaries, args.model.split("/")[-1],
            model_dir / "sweep_trajectory.png",
        )
        print(f"Sweep roll-up: {len(summaries)} completed checkpoints "
              f"in {model_dir}")

    print(
        f"\nSweep done. "
        f"Processed: {n_done}, Skipped: {n_skipped}, "
        f"Load failed: {n_load_failed}, Errors: {n_error}, "
        f"Cache freed: {fmt_bytes(total_freed)}"
    )


if __name__ == "__main__":
    main()
