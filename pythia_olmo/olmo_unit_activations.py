"""
Visualize residual stream depth-trajectories for individual units at
OLMo-2-1B checkpoints.

Loads OLMo-2-1B at multiple training steps, captures residual streams
at each sublayer using forward_pre_hooks on self_attn and mlp (matching
phase_space_olmo_checkpoints.py), then plots the depth-trajectory of
selected units across checkpoints. The same unit indices are used for
each checkpoint, since d_model is fixed across training. The same
wikitext samples are also used at all checkpoints (same RNG seed), so
highlighted input k at step 1000 corresponds to the same text as
highlighted input k at step 1000000.

Default checkpoints: 1000 (during backbone formation),
                     100000 (at dtheta_dev floor in earlier sweep),
                     1000000 (late training, after dtheta_dev rebound).

The motivation: the phase-space diagnostics confirm "this depth-trajectory
is sinusoidal" but do not pin down the dominant frequency. Plotting the
actual activations shows what waveform is actually being executed and
whether it matches the 4-sublayer-period hypothesis suggested by phase-
amplitude coupling results elsewhere.

Outputs in olmo_activation_visuals/:
  activations_centered.png   Depth-axis centered activations.
                             Rows = selected units, cols = checkpoints.
                             Gray cloud = all 256 inputs at low alpha.
                             Colored lines = highlighted inputs (same
                             input indices across all panels).
  activations_raw.png        Same data without depth-axis centering;
                             reveals DC offset variability that the
                             diagnostics suppress.
  activations_fft.png        Magnitude FFT of each unit's depth-trajectory,
                             averaged over inputs. Red dashed line marks
                             the bin corresponding to a 4-sublayer period
                             (L_used / 4). A peak there confirms the fast-
                             oscillation hypothesis.
  activation_data.npz        Selected units' streams saved per checkpoint
                             for re-plotting without re-running the model.

Usage:
  python olmo_unit_activations.py
  python olmo_unit_activations.py --n-units 12 --n-inputs-highlight 6
  python olmo_unit_activations.py --unit-indices 100 250 500 1000 1500 2000
  python olmo_unit_activations.py --checkpoints 1000 10000 100000 1000000
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# OLMo-2-1B configuration (from phase_space_olmo_checkpoints.py)
# ---------------------------------------------------------------------------

OLMO_MODEL = "allenai/OLMo-2-0425-1B"
EARLY_TRAINING_REPO = "allenai/OLMo-2-0425-1B-early-training"
EARLY_TRAINING_MAX_STEP = 37000

DEFAULT_CHECKPOINTS = [0, 1000, 100000, 1000000]

DEFAULT_CACHE_DIR = os.environ.get("HF_HOME", None)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def step_to_revision(step):
    tokens_b = math.ceil(step * 2048 * 1024 / 1_000_000_000)
    return f"stage1-step{step}-tokens{tokens_b}B"


# ---------------------------------------------------------------------------
# Stream collection (matches phase_space_olmo_checkpoints.py)
# ---------------------------------------------------------------------------

def get_hook_targets(model):
    """OLMo-2 is post-norm. Hook self_attn and mlp inputs to capture the
    residual stream as it enters each sublayer."""
    blocks = model.model.layers
    targets = []
    for block in blocks:
        targets.append(block.self_attn)
        targets.append(block.mlp)
    return targets


def collect_streams(model, tokenizer, texts, seq_len):
    """Capture last-token residual stream snapshot at each sublayer input.
    Returns (n_samples, n_sub, d_model) array."""
    model.eval()
    targets = get_hook_targets(model)
    out = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt",
                        max_length=seq_len, truncation=True)
        ids = enc["input_ids"].to(DEVICE)
        if ids.shape[1] < 8:
            continue
        captures = []
        def _capture_hook(m, args, kwargs, c=captures):
            if args:
                c.append(args[0].detach())
            else:
                c.append(kwargs["hidden_states"].detach())
        hooks = [
            t.register_forward_pre_hook(_capture_hook, with_kwargs=True)
            for t in targets
        ]
        try:
            with torch.no_grad():
                model(ids)
        finally:
            for h in hooks:
                h.remove()
        last = torch.stack([c[0, -1, :].float() for c in captures], dim=0)
        out.append(last.cpu().numpy())
    return np.stack(out, axis=0)


def load_olmo_at_step(step, cache_dir=None):
    repo = EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP else OLMO_MODEL
    revision = step_to_revision(step)
    print(f"  Loading {repo} at {revision} ...")
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    ).to(DEVICE)
    model.eval()
    return model


def _repo_and_revision_for_step(step):
    """Return (repo_id, revision_string) for a given training step."""
    repo = EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP else OLMO_MODEL
    return repo, step_to_revision(step)


# ---------------------------------------------------------------------------
# HF cache cleanup (from pac_probe_general.py)
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
# Unit selection via |A_norm| ranking
# ---------------------------------------------------------------------------

def compute_a_norm_per_unit(streams, trim=2):
    """Per-unit |A_norm|: shoelace signed area of (x, dx) trajectory,
    normalized, taken median across inputs. Returns (D,) array."""
    if trim > 0:
        if streams.shape[1] <= 2 * trim + 2:
            raise ValueError(
                f"Cannot trim {trim} from each end of {streams.shape[1]} sublayers."
            )
        streams = streams[:, trim:streams.shape[1] - trim, :]
    x = streams - streams.mean(axis=1, keepdims=True)
    dx = np.diff(x, axis=1)
    x_pair = x[:, :-1, :]

    a = x_pair[:, :-1, :] * dx[:, 1:, :]
    b = x_pair[:, 1:, :] * dx[:, :-1, :]
    A = 0.5 * (a - b).sum(axis=1)

    std_x = x_pair.std(axis=1)
    std_dx = dx.std(axis=1)
    L_pair = x_pair.shape[1]
    A_norm = A / ((std_x * std_dx * L_pair) + 1e-30)

    return np.median(A_norm, axis=0)


def select_units(streams, n_units=9, trim=2):
    """Pick units in three tiers by |A_norm|: high, median, low.
    Returns indices in the d_model dimension."""
    a_norm = np.abs(compute_a_norm_per_unit(streams, trim=trim))
    order = np.argsort(a_norm)
    D = len(order)
    if D < n_units:
        return list(order[-D:])

    n_per = n_units // 3
    n_remainder = n_units - 3 * n_per
    chosen = []
    chosen.extend(reversed(order[-n_per:].tolist()))  # highest |A_norm| first
    mid = D // 2
    half = n_per // 2
    chosen.extend(order[mid - half:mid - half + n_per].tolist())
    chosen.extend(order[:n_per + n_remainder].tolist())
    return chosen, a_norm


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _trimmed_x(streams, trim, centered):
    if trim > 0:
        x = streams[:, trim:streams.shape[1] - trim, :]
    else:
        x = streams
    if centered:
        x = x - x.mean(axis=1, keepdims=True)
    return x


def plot_activations(streams_by_step, unit_indices, out_path,
                     centered=True, trim=2, n_inputs_highlight=4,
                     a_norm_per_step=None):
    """Rows = units, cols = checkpoints. Gray cloud + highlighted inputs."""
    steps = sorted(streams_by_step.keys())
    n_units = len(unit_indices)
    n_steps = len(steps)

    fig, axes = plt.subplots(
        n_units, n_steps,
        figsize=(4.2 * n_steps, 2.4 * n_units),
        squeeze=False, sharex=True,
    )

    highlight_colors = plt.cm.tab10(np.linspace(0, 1, max(n_inputs_highlight, 4)))

    # Determine y-limits per row (per unit) so within a row you can compare
    # across checkpoints at the same scale, but rows can differ. This makes
    # 4-sublayer oscillation visible at the right scale per unit.
    row_ymax = []
    for u in unit_indices:
        vals = []
        for step in steps:
            x = _trimmed_x(streams_by_step[step], trim, centered)
            vals.append(np.abs(x[:, :, u]).max())
        row_ymax.append(max(vals) * 1.1)

    for j, step in enumerate(steps):
        x = _trimmed_x(streams_by_step[step], trim, centered)
        N, L, D = x.shape
        sublayer_idx = np.arange(L)

        for i, u in enumerate(unit_indices):
            ax = axes[i, j]

            # Background cloud: all inputs at low alpha
            for k in range(N):
                ax.plot(sublayer_idx, x[k, :, u],
                        color="gray", alpha=0.06, linewidth=0.5)

            # Highlighted inputs (same input indices across all panels)
            for k in range(min(n_inputs_highlight, N)):
                ax.plot(sublayer_idx, x[k, :, u],
                        color=highlight_colors[k], alpha=0.9,
                        linewidth=1.3, label=f"input {k}" if (i == 0 and j == 0) else None)

            ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
            ax.grid(alpha=0.2)
            ax.set_ylim(-row_ymax[i], row_ymax[i])

            if i == 0:
                ax.set_title(f"step {step}", fontsize=11)

            label = f"unit {u}"
            if a_norm_per_step is not None and step in a_norm_per_step:
                a_val = a_norm_per_step[step][u]
                label += f"\n|A_norm|={abs(a_val):.2f}"
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
            if i == n_units - 1:
                ax.set_xlabel("sublayer index (after trim)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right",
               bbox_to_anchor=(0.99, 0.995), fontsize=9, ncol=1)

    fig.suptitle(
        f"OLMo-2-1B residual stream depth-trajectories "
        f"({'depth-centered' if centered else 'raw'})\n"
        f"gray cloud = 256 inputs; colored lines = same {n_inputs_highlight} "
        f"highlighted texts across all panels",
        fontsize=13, y=0.997,
    )
    fig.tight_layout(rect=[0, 0, 0.95, 0.99])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_fft_spectra(streams_by_step, unit_indices, out_path, trim=2):
    """Magnitude FFT across depth, averaged over inputs. Marks the bin
    corresponding to a 4-sublayer period (L/4) for direct comparison."""
    steps = sorted(streams_by_step.keys())
    n_units = len(unit_indices)
    n_steps = len(steps)

    fig, axes = plt.subplots(
        n_units, n_steps,
        figsize=(4.2 * n_steps, 2.0 * n_units),
        squeeze=False, sharex=True,
    )

    for j, step in enumerate(steps):
        x = _trimmed_x(streams_by_step[step], trim, centered=True)
        N, L, D = x.shape
        target_bin = L / 4.0

        spectra = np.abs(np.fft.rfft(x, axis=1))  # (N, L//2 + 1, D)
        n_bins = spectra.shape[1]

        for i, u in enumerate(unit_indices):
            ax = axes[i, j]
            mean_spec = spectra[:, :, u].mean(axis=0)
            std_spec = spectra[:, :, u].std(axis=0)
            freqs = np.arange(n_bins)

            ax.bar(freqs, mean_spec, width=0.8,
                   color="C0", alpha=0.75, edgecolor="none")
            ax.errorbar(freqs, mean_spec, yerr=std_spec,
                        fmt="none", ecolor="black", alpha=0.4,
                        capsize=1.5, linewidth=0.6)
            ax.axvline(target_bin, color="red", linestyle="--",
                       alpha=0.8, linewidth=1.4,
                       label=f"4-sublayer bin (={target_bin:.1f})")
            ax.set_xlim(-0.5, n_bins - 0.5)
            ax.grid(alpha=0.3)

            if i == 0:
                ax.set_title(f"step {step}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"unit {u}\nFFT magnitude", fontsize=9)
            if i == n_units - 1:
                ax.set_xlabel("FFT bin (cycles per L)")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        "Depth-axis FFT magnitude per unit (mean over inputs, error bars = std)\n"
        "Red dashed = bin for 4-sublayer period; "
        "a peak there matches the fast-oscillator hypothesis.",
        fontsize=13, y=0.997,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=int, nargs="+",
                        default=DEFAULT_CHECKPOINTS,
                        help="Training steps to load.")
    parser.add_argument("--n-samples", type=int, default=256,
                        help="Number of texts from wikitext-2.")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--trim-sublayers", type=int, default=2,
                        help="Drop this many sublayers from each end.")
    parser.add_argument("--n-units", type=int, default=9,
                        help="Number of units to visualize.")
    parser.add_argument("--n-inputs-highlight", type=int, default=4,
                        help="Number of distinct-color highlighted input lines.")
    parser.add_argument("--unit-selection-step", type=int, default=None,
                        help="Which checkpoint to use for |A_norm|-based unit "
                             "selection. Default: last successfully loaded.")
    parser.add_argument("--unit-indices", type=int, nargs="+", default=None,
                        help="Override automatic selection with explicit indices.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep HF cache after each checkpoint. "
                             "Default: cleanup is ON.")
    parser.add_argument("--out-dir", default="olmo_activation_visuals")
    parser.add_argument("--save-full-streams", action="store_true",
                        help="Save full (N, L, D) tensors per checkpoint. "
                             "Default: only save the selected units' data.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("Loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]

    print(f"Loading tokenizer for {OLMO_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        OLMO_MODEL, cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Collect and save streams per checkpoint ----
    print(f"Cleanup: {'OFF' if args.no_cleanup else 'ON'}")
    total_freed = 0
    completed_steps = []

    for idx, step in enumerate(args.checkpoints, 1):
        streams_path = out_dir / f"streams_step{step}.npz"
        print(f"\n=== [{idx}/{len(args.checkpoints)}] OLMo-2-1B at step {step} ===")

        # Resumability: skip if streams already saved to disk
        if streams_path.exists():
            print(f"  [skip] streams already on disk: {streams_path.name}")
            completed_steps.append(step)
            continue

        repo, revision = _repo_and_revision_for_step(step)
        try:
            model = load_olmo_at_step(step, cache_dir=args.cache_dir)
        except Exception as e:
            print(f"  [load failed] step {step}: {e}")
            # Cleanup even on load failure (partial download may exist)
            if not args.no_cleanup:
                freed = delete_cached_revision(repo, revision, args.cache_dir)
                if freed > 0:
                    print(f"  [cleanup] freed {fmt_bytes(freed)}")
                    total_freed += freed
            continue

        try:
            print(f"  Collecting streams ...")
            streams = collect_streams(model, tokenizer, texts, args.seq_len)
            print(f"    shape (N, n_sub, d_model) = {streams.shape}")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save immediately so work survives crashes
        np.savez_compressed(streams_path, streams=streams)
        print(f"  [saved] {streams_path.name}")
        completed_steps.append(step)
        del streams

        # Cleanup HF cache for this revision
        if not args.no_cleanup:
            freed = delete_cached_revision(repo, revision, args.cache_dir)
            if freed > 0:
                print(f"  [cleanup] freed {fmt_bytes(freed)} "
                      f"from {repo}@{revision}")
                total_freed += freed

    if total_freed > 0:
        print(f"\nTotal cache freed: {fmt_bytes(total_freed)}")

    # ---- Reload saved streams for analysis ----
    if not completed_steps:
        raise SystemExit("No checkpoints loaded successfully.")

    print(f"\nLoading saved streams for {len(completed_steps)} checkpoints ...")
    streams_by_step = {}
    for step in completed_steps:
        streams_path = out_dir / f"streams_step{step}.npz"
        streams_by_step[step] = np.load(streams_path)["streams"]

    # ---- Select units ----
    if args.unit_indices is not None:
        unit_indices = list(args.unit_indices)
        print(f"\nUsing user-specified unit indices: {unit_indices}")
        a_norm_per_step = {
            s: compute_a_norm_per_unit(streams_by_step[s], trim=args.trim_sublayers)
            for s in streams_by_step
        }
    else:
        sel_step = args.unit_selection_step
        if sel_step is None or sel_step not in streams_by_step:
            sel_step = max(streams_by_step.keys())
        print(f"\nSelecting {args.n_units} units by |A_norm| at step {sel_step}")
        unit_indices, _ = select_units(
            streams_by_step[sel_step],
            n_units=args.n_units,
            trim=args.trim_sublayers,
        )
        a_norm_per_step = {
            s: compute_a_norm_per_unit(streams_by_step[s], trim=args.trim_sublayers)
            for s in streams_by_step
        }
        print(f"Selected unit indices: {unit_indices}")
        for u in unit_indices:
            tier = "high" if abs(a_norm_per_step[sel_step][u]) > 0.20 else (
                "median" if abs(a_norm_per_step[sel_step][u]) > 0.05 else "low"
            )
            print(f"  unit {u:>4}: |A_norm|={abs(a_norm_per_step[sel_step][u]):.3f} ({tier})")

    # ---- Save combined data ----
    save_dict = {"unit_indices": np.array(unit_indices)}
    for s, streams in streams_by_step.items():
        if args.save_full_streams:
            save_dict[f"step{s}_full"] = streams
        save_dict[f"step{s}_selected"] = streams[:, :, unit_indices]
    np.savez_compressed(out_dir / "activation_data.npz", **save_dict)
    print(f"\nSaved data to {out_dir / 'activation_data.npz'}")

    # ---- Plots ----
    print("\nMaking plots ...")
    plot_activations(
        streams_by_step, unit_indices,
        out_dir / "activations_centered.png",
        centered=True, trim=args.trim_sublayers,
        n_inputs_highlight=args.n_inputs_highlight,
        a_norm_per_step=a_norm_per_step,
    )
    print(f"  activations_centered.png")

    plot_activations(
        streams_by_step, unit_indices,
        out_dir / "activations_raw.png",
        centered=False, trim=args.trim_sublayers,
        n_inputs_highlight=args.n_inputs_highlight,
        a_norm_per_step=a_norm_per_step,
    )
    print(f"  activations_raw.png")

    plot_fft_spectra(
        streams_by_step, unit_indices,
        out_dir / "activations_fft.png",
        trim=args.trim_sublayers,
    )
    print(f"  activations_fft.png")

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
