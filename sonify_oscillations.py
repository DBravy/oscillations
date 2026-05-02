"""
Sonify residual stream oscillations.

Each unit's trajectory across depth is a 25-sample 1-D signal. We turn
it into audible sound by treating depth as time at a fast playback rate
and stitching trajectories from many inputs together.

Modes:

  single_unit:   plays ONE unit's trajectory, looped across all 256 inputs.
                 Each input's 25-sample trajectory is upsampled (cubic
                 interpolation) to a chunk of audio.
  unit_chord:    plays SEVERAL units simultaneously (one per output channel
                 if --stereo, otherwise mixed mono). Lets you hear coupled
                 units together.
  fiedler_proj:  plays the Fiedler-vector projection of the residual stream,
                 i.e. the scalar v_2 . h(l) trajectory. Gives a one-dim
                 audio signal that summarizes the dominant rotational mode.
  pair_stereo:   plays one unit on the left channel and another on the
                 right channel. Lets you spatially hear phase relationships
                 between two units.

Per-input scaling: each input's chunk is normalized to unit RMS so that
loud and quiet inputs all sound roughly equal. Use --no-normalize to
hear actual amplitude differences (mostly the explosion).

Outputs in --out-dir:
  single_unit_{u}.wav            single-unit sonification, all 256 inputs
  unit_chord.wav                 several units mixed
  fiedler_proj_pos{N}.wav        Fiedler projection, one file per position
  pair_stereo_{u}_{v}.wav        stereo pair sonification
  spectrogram_{name}.png         visual spectrogram of each output
  oscillation_metadata.json      sample rate, mappings used

Usage:
  python sonify_oscillations.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_audio \
      --mode single_unit --unit 13 --position 5

  python sonify_oscillations.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_audio \
      --mode pair_stereo --unit 13 --unit2 29 --position 5

  python sonify_oscillations.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --out-dir toy_audio \
      --mode fiedler_proj --position 5
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import torch

from toy_transformer_addition import (
    ToyTransformer,
    ModelConfig,
    capture_residual_streams,
)


SAMPLE_RATE = 44100


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = ToyTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def collect_streams(model, position):
    device = next(model.parameters()).device
    pairs = [(a, b) for a in range(16) for b in range(16)]
    _, captures = capture_residual_streams(model, pairs, device)
    stacked = torch.stack([c["residual"] for c in captures], dim=0)
    s = stacked[:, :, position, :].permute(1, 0, 2).contiguous()
    return s.numpy().astype(np.float32), pairs


# ---------------------------------------------------------------------------
# Trajectory -> audio
# ---------------------------------------------------------------------------

def upsample_trajectory(trajectory, n_out):
    """trajectory: (L,) array. Returns (n_out,) array via cubic spline."""
    L = len(trajectory)
    xs_in  = np.linspace(0, 1, L)
    xs_out = np.linspace(0, 1, n_out)
    spline = CubicSpline(xs_in, trajectory)
    return spline(xs_out)


def normalize_chunk(chunk, target_rms=0.2):
    """Scale chunk to target RMS. Skip if already silent."""
    rms = np.sqrt(np.mean(chunk ** 2))
    if rms < 1e-9:
        return chunk
    return chunk * (target_rms / rms)


def trajectories_to_audio(
    trajectories, samples_per_input, fade_samples=128,
    normalize_per_input=True,
):
    """
    trajectories: (N, L) per-input scalar trajectories.
    Returns:      (N * samples_per_input,) audio array.

    Each input becomes one chunk; chunks are concatenated. A short
    crossfade is applied between chunks so transitions don't pop.
    """
    N, L = trajectories.shape
    chunks = []
    for n in range(N):
        chunk = upsample_trajectory(trajectories[n], samples_per_input)
        if normalize_per_input:
            chunk = normalize_chunk(chunk)
        # Apply small fade at chunk edges to suppress clicks at boundaries
        if fade_samples > 0:
            fade = np.linspace(0, 1, fade_samples)
            chunk[:fade_samples]  *= fade
            chunk[-fade_samples:] *= fade[::-1]
        chunks.append(chunk)
    return np.concatenate(chunks)


def to_int16(audio, headroom=0.92):
    """Clip and convert float audio to int16, with headroom to avoid
    distortion."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (headroom / peak)
    return np.clip(audio * 32767.0, -32767, 32767).astype(np.int16)


def write_wav(filepath, audio_int16, n_channels=1,
              sample_rate=SAMPLE_RATE):
    """audio_int16 must be (n_samples,) for mono or (n_samples, n_channels)."""
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)         # 16-bit
        wf.setframerate(sample_rate)
        if n_channels == 1:
            wf.writeframes(audio_int16.tobytes())
        else:
            interleaved = audio_int16.astype(np.int16).tobytes()
            wf.writeframes(interleaved)


# ---------------------------------------------------------------------------
# Plot helpers (visual companion to audio)
# ---------------------------------------------------------------------------

def plot_spectrogram(audio, sample_rate, save_path, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    Pxx, freqs, bins, im = ax.specgram(
        audio, NFFT=2048, Fs=sample_rate, noverlap=1024,
        cmap="magma",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_yscale("symlog", linthresh=20)
    ax.set_ylim(20, sample_rate / 2)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_trajectory_overlay(trajectories, save_path, title,
                             max_to_plot=64):
    """Plot the first max_to_plot trajectories overlaid in a single panel."""
    fig, ax = plt.subplots(figsize=(11, 5))
    n_show = min(max_to_plot, trajectories.shape[0])
    for n in range(n_show):
        ax.plot(trajectories[n], "-", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("sublayer l")
    ax.set_ylabel("scalar value")
    ax.set_title(f"{title} (first {n_show} inputs overlaid)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_single_unit(streams, args, out_dir, metadata):
    u = args.unit
    if u is None or u >= streams.shape[2]:
        raise ValueError(f"--unit must be set and < {streams.shape[2]}")
    trajectories = streams[:, :, u]
    audio = trajectories_to_audio(
        trajectories, args.samples_per_input,
        fade_samples=args.fade_samples,
        normalize_per_input=not args.no_normalize,
    )
    audio_int = to_int16(audio)
    name = f"single_unit_u{u}_pos{args.position}"
    write_wav(out_dir / f"{name}.wav", audio_int)
    plot_spectrogram(
        audio.astype(np.float32), SAMPLE_RATE,
        out_dir / f"spectrogram_{name}.png",
        f"single unit u={u}, position {args.position}",
    )
    plot_trajectory_overlay(
        trajectories, out_dir / f"trajectories_{name}.png",
        f"unit {u} at position {args.position}",
    )
    metadata["files"].append({
        "name":    f"{name}.wav",
        "mode":    "single_unit",
        "unit":    int(u),
        "position": int(args.position),
        "duration_s": len(audio) / SAMPLE_RATE,
    })


def mode_unit_chord(streams, args, out_dir, metadata):
    units = args.units or [0, 8, 16, 24, 32, 40, 48, 56]
    units = [u for u in units if u < streams.shape[2]]
    if not units:
        raise ValueError("No valid units in --units")
    chunks = []
    for u in units:
        traj = streams[:, :, u]
        audio = trajectories_to_audio(
            traj, args.samples_per_input,
            fade_samples=args.fade_samples,
            normalize_per_input=not args.no_normalize,
        )
        chunks.append(audio)
    mixed = np.mean(np.stack(chunks, axis=0), axis=0)
    audio_int = to_int16(mixed)
    units_str = "_".join(str(u) for u in units)
    name = f"unit_chord_pos{args.position}_units_{units_str}"
    write_wav(out_dir / f"{name}.wav", audio_int)
    plot_spectrogram(
        mixed.astype(np.float32), SAMPLE_RATE,
        out_dir / f"spectrogram_{name}.png",
        f"chord of units {units} at position {args.position}",
    )
    metadata["files"].append({
        "name":     f"{name}.wav",
        "mode":     "unit_chord",
        "units":    [int(u) for u in units],
        "position": int(args.position),
        "duration_s": len(mixed) / SAMPLE_RATE,
    })


def mode_fiedler_proj(streams, args, out_dir, metadata):
    """Project residual stream onto its own Fiedler v_2 at each layer.

    For each input, build a scalar trajectory:
      proj(n, l) = v_2(l) . (h(n, l) - mean_n h(n, l))
    Note: v_2 is computed at each layer using the across-input cospec
    coupling, so the projection is well-defined per layer.
    """
    N, L, D = streams.shape
    # Build per-unit z and per-layer Fiedler vectors
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    z = x + 1j * y                                  # (N, L, D), complex
    proj_traj = np.zeros((N, L), dtype=np.float32)
    for l in range(L):
        if l == 0:
            continue
        cs = z[:, l, :, None] * np.conj(z[:, l, None, :])
        cs_mean = cs.mean(axis=0)
        C = np.abs(cs_mean)
        np.fill_diagonal(C, 0.0)
        deg = C.sum(axis=1)
        Lap = np.diag(deg) - C
        Lap = (Lap + Lap.T) / 2.0
        w, V = np.linalg.eigh(Lap)
        v2 = V[:, 1]
        # Project across-input-centered residuals onto v2
        centered = streams[:, l, :] - streams[:, l, :].mean(axis=0)
        proj_traj[:, l] = centered @ v2
    audio = trajectories_to_audio(
        proj_traj, args.samples_per_input,
        fade_samples=args.fade_samples,
        normalize_per_input=not args.no_normalize,
    )
    audio_int = to_int16(audio)
    name = f"fiedler_proj_pos{args.position}"
    write_wav(out_dir / f"{name}.wav", audio_int)
    plot_spectrogram(
        audio.astype(np.float32), SAMPLE_RATE,
        out_dir / f"spectrogram_{name}.png",
        f"Fiedler projection at position {args.position}",
    )
    plot_trajectory_overlay(
        proj_traj, out_dir / f"trajectories_{name}.png",
        f"Fiedler projection at position {args.position}",
    )
    metadata["files"].append({
        "name":     f"{name}.wav",
        "mode":     "fiedler_proj",
        "position": int(args.position),
        "duration_s": len(audio) / SAMPLE_RATE,
    })


def mode_pair_stereo(streams, args, out_dir, metadata):
    u, v = args.unit, args.unit2
    if u is None or v is None:
        raise ValueError(
            "--unit and --unit2 are required for pair_stereo mode"
        )
    if u >= streams.shape[2] or v >= streams.shape[2]:
        raise ValueError(f"unit indices must be < {streams.shape[2]}")
    traj_u = streams[:, :, u]
    traj_v = streams[:, :, v]
    audio_u = trajectories_to_audio(
        traj_u, args.samples_per_input,
        fade_samples=args.fade_samples,
        normalize_per_input=not args.no_normalize,
    )
    audio_v = trajectories_to_audio(
        traj_v, args.samples_per_input,
        fade_samples=args.fade_samples,
        normalize_per_input=not args.no_normalize,
    )
    # Match lengths just in case
    n = min(len(audio_u), len(audio_v))
    audio_u = audio_u[:n]
    audio_v = audio_v[:n]
    # Build interleaved stereo
    peak = max(np.max(np.abs(audio_u)), np.max(np.abs(audio_v)), 1e-9)
    audio_u_int = to_int16(audio_u / peak)
    audio_v_int = to_int16(audio_v / peak)
    interleaved = np.empty((n, 2), dtype=np.int16)
    interleaved[:, 0] = audio_u_int
    interleaved[:, 1] = audio_v_int
    name = f"pair_stereo_u{u}_v{v}_pos{args.position}"
    write_wav(out_dir / f"{name}.wav", interleaved, n_channels=2)
    plot_spectrogram(
        ((audio_u + audio_v) / 2).astype(np.float32), SAMPLE_RATE,
        out_dir / f"spectrogram_{name}.png",
        f"pair (u={u}, v={v}) at position {args.position} (mono mix)",
    )
    metadata["files"].append({
        "name":     f"{name}.wav",
        "mode":     "pair_stereo",
        "unit_left":  int(u),
        "unit_right": int(v),
        "position":   int(args.position),
        "duration_s": n / SAMPLE_RATE,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="toy_audio")
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["single_unit", "unit_chord",
                 "fiedler_proj", "pair_stereo"],
    )
    parser.add_argument("--position", type=int, default=5,
                        choices=[5, 6, 7])
    parser.add_argument("--unit", type=int, default=None,
                        help="single unit index (modes single_unit, "
                             "pair_stereo)")
    parser.add_argument("--unit2", type=int, default=None,
                        help="second unit (mode pair_stereo)")
    parser.add_argument("--units", type=int, nargs="+", default=None,
                        help="list of units (mode unit_chord)")
    parser.add_argument(
        "--samples-per-input", type=int, default=22050,
        help=("Audio samples generated per input pair. "
              "22050 at 44.1kHz = 0.5s per input. "
              "256 inputs * 0.5s = 128s total."),
    )
    parser.add_argument("--fade-samples", type=int, default=256,
                        help="Crossfade samples between input chunks")
    parser.add_argument("--no-normalize", action="store_true",
                        help=("Do NOT per-input-normalize. Hear actual "
                              "amplitude differences (the explosion)."))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  d_model={cfg.d_model}")

    print(f"\nCollecting streams at position {args.position} ...")
    streams, pairs = collect_streams(model, args.position)
    print(f"  shape: {streams.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metadata = {
        "checkpoint": str(args.checkpoint),
        "position":   int(args.position),
        "n_inputs":   int(streams.shape[0]),
        "n_sublayers": int(streams.shape[1]),
        "d_model":    int(streams.shape[2]),
        "samples_per_input": int(args.samples_per_input),
        "sample_rate":       SAMPLE_RATE,
        "normalize_per_input": not args.no_normalize,
        "files":      [],
    }

    if args.mode == "single_unit":
        mode_single_unit(streams, args, out_dir, metadata)
    elif args.mode == "unit_chord":
        mode_unit_chord(streams, args, out_dir, metadata)
    elif args.mode == "fiedler_proj":
        mode_fiedler_proj(streams, args, out_dir, metadata)
    elif args.mode == "pair_stereo":
        mode_pair_stereo(streams, args, out_dir, metadata)

    with open(out_dir / "oscillation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nFiles written:")
    for entry in metadata["files"]:
        print(f"  {entry['name']}  ({entry['duration_s']:.1f}s)")
    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
