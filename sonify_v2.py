"""
Sonify residual stream oscillations as continuous waves.

The depth trajectory of a unit is treated as one cycle of a periodic
waveform. Looping it at a chosen frequency makes it an audible tone.

For a unit u on input n, the trajectory is r_u(n, l) for l = 0..L-1.
We loop this waveform at base_freq Hz to produce sound for duration_s
seconds, by sampling the waveform with a continuous read-pointer that
advances at base_freq * L samples per second.

The fundamental frequency of the result is base_freq Hz. Higher harmonics
arise from the non-sinusoidal shape of the actual trajectory.

Modes:

  unit_tone:        single unit, single input, one continuous tone
  unit_sweep:       single unit, all 256 inputs played in sequence
                    (each input becomes a brief tone)
  unit_chord:       several units played simultaneously, all at the same
                    base frequency, on a single input
  unit_pair_stereo: two units on left and right channels, single input,
                    same base frequency (so phase relationships are audible)
  fiedler_tone:     Fiedler-direction projection of residual stream as a
                    tone, one input

Outputs in --out-dir:
  *.wav                          audio file(s)
  waveform_*.png                 plot of the actual waveform that's being
                                  looped (one cycle)
  spectrogram_*.png              spectrogram of the result

Usage:
  python sonify_v2.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --mode unit_tone --unit 13 --position 5 --input-idx 0 \
      --base-freq 220 --duration 3 --out-dir toy_audio

  python sonify_v2.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --mode unit_sweep --unit 13 --position 5 \
      --base-freq 220 --tone-duration 0.4 --out-dir toy_audio

  python sonify_v2.py \
      --checkpoint toy_transformer_run/model_trained.pt \
      --mode unit_pair_stereo --unit 13 --unit2 29 \
      --position 5 --input-idx 0 \
      --base-freq 220 --duration 5 --out-dir toy_audio
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
# The core operation: loop a single-cycle waveform at a frequency
# ---------------------------------------------------------------------------

def loop_waveform_at_frequency(
    cycle, base_freq, duration_s, sample_rate=SAMPLE_RATE,
    detrend=True, normalize=True,
):
    """
    Treat `cycle` as one period of a waveform. Generate `duration_s`
    seconds of audio at `sample_rate` by reading the cycle continuously
    at `base_freq` Hz, with linear interpolation between samples.

    cycle: (L,) array. The waveform's shape across one period.
    base_freq: fundamental frequency of the output tone (Hz).
    duration_s: output duration in seconds.

    Returns: audio array of length int(duration_s * sample_rate).
    """
    L = len(cycle)
    cycle = np.asarray(cycle, dtype=np.float64)
    if detrend:
        # Remove DC so the tone doesn't have a constant offset
        cycle = cycle - cycle.mean()
    if normalize:
        peak = np.max(np.abs(cycle))
        if peak > 0:
            cycle = cycle / peak

    n_out = int(duration_s * sample_rate)
    # The read-pointer advances by (base_freq * L) samples per second.
    # In samples-of-output, that's (base_freq * L / sample_rate) cycle-
    # samples per output-sample.
    step_per_sample = base_freq * L / sample_rate
    # Float positions of the read pointer (modulo L)
    positions = (np.arange(n_out) * step_per_sample) % L
    lo = np.floor(positions).astype(int)
    hi = (lo + 1) % L
    frac = positions - lo
    # Linear interpolation between cycle[lo] and cycle[hi]
    audio = cycle[lo] * (1.0 - frac) + cycle[hi] * frac
    return audio


def envelope(n_samples, attack=0.01, release=0.05,
              sample_rate=SAMPLE_RATE):
    """Apply a simple attack-release envelope to suppress click at start
    and end. Returns multiplicative envelope of shape (n_samples,)."""
    env = np.ones(n_samples)
    a = int(attack * sample_rate)
    r = int(release * sample_rate)
    a = min(a, n_samples // 2)
    r = min(r, n_samples // 2)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if r > 0:
        env[-r:] = np.linspace(1, 0, r)
    return env


def to_int16(audio, headroom=0.92):
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (headroom / peak)
    return np.clip(audio * 32767.0, -32767, 32767).astype(np.int16)


def write_wav(filepath, audio_int16, n_channels=1,
              sample_rate=SAMPLE_RATE):
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        if n_channels == 1:
            wf.writeframes(audio_int16.tobytes())
        else:
            wf.writeframes(audio_int16.tobytes())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_waveform(cycle, save_path, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(cycle)), cycle, "o-", linewidth=1.5)
    ax.axhline(cycle.mean(), color="gray", linewidth=0.5, linestyle=":",
               label=f"mean = {cycle.mean():.3f}")
    ax.set_xlabel("sublayer l (one period)")
    ax.set_ylabel("unit value")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_spectrogram(audio, sample_rate, save_path, title):
    fig, ax = plt.subplots(figsize=(11, 5))
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


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_unit_tone(streams, args, out_dir, metadata):
    u = args.unit
    n = args.input_idx
    cycle = streams[n, :, u]
    audio = loop_waveform_at_frequency(
        cycle, args.base_freq, args.duration,
    )
    audio = audio * envelope(len(audio))
    name = f"unit_tone_u{u}_pos{args.position}_input{n}_f{args.base_freq:.0f}"
    write_wav(out_dir / f"{name}.wav", to_int16(audio))
    plot_waveform(
        cycle, out_dir / f"waveform_{name}.png",
        f"unit {u} at pos {args.position}, input {n} (one cycle)",
    )
    plot_spectrogram(
        audio.astype(np.float32), SAMPLE_RATE,
        out_dir / f"spectrogram_{name}.png",
        f"unit {u} looped at {args.base_freq} Hz",
    )
    metadata["files"].append({
        "name":       f"{name}.wav",
        "mode":       "unit_tone",
        "unit":       int(u),
        "position":   int(args.position),
        "input_idx":  int(n),
        "base_freq":  float(args.base_freq),
        "duration_s": float(args.duration),
    })


def mode_unit_sweep(streams, args, out_dir, metadata):
    """Each input becomes a brief tone using that input's cycle."""
    u = args.unit
    N = streams.shape[0]
    chunks = []
    tone_samples = int(args.tone_duration * SAMPLE_RATE)
    for n in range(N):
        cycle = streams[n, :, u]
        audio = loop_waveform_at_frequency(
            cycle, args.base_freq, args.tone_duration,
        )
        audio = audio * envelope(len(audio))
        chunks.append(audio)
    full = np.concatenate(chunks)
    name = f"unit_sweep_u{u}_pos{args.position}_f{args.base_freq:.0f}"
    write_wav(out_dir / f"{name}.wav", to_int16(full))
    plot_spectrogram(
        full.astype(np.float32), SAMPLE_RATE,
        out_dir / f"spectrogram_{name}.png",
        (f"unit {u} sweep across {N} inputs "
         f"@ {args.base_freq} Hz, {args.tone_duration}s each"),
    )
    metadata["files"].append({
        "name":          f"{name}.wav",
        "mode":          "unit_sweep",
        "unit":          int(u),
        "position":      int(args.position),
        "tone_duration": float(args.tone_duration),
        "base_freq":     float(args.base_freq),
        "n_inputs":      int(N),
        "duration_s":    float(len(full) / SAMPLE_RATE),
    })


def mode_unit_chord(streams, args, out_dir, metadata):
    units = args.units or [0, 8, 16, 24, 32, 40, 48, 56]
    n = args.input_idx
    voices = []
    for u in units:
        cycle = streams[n, :, u]
        audio = loop_waveform_at_frequency(
            cycle, args.base_freq, args.duration,
        )
        voices.append(audio)
    mixed = np.mean(np.stack(voices, axis=0), axis=0)
    mixed = mixed * envelope(len(mixed))
    units_str = "_".join(str(u) for u in units)
    name = (f"unit_chord_pos{args.position}_input{n}"
            f"_units_{units_str}_f{args.base_freq:.0f}")
    write_wav(out_dir / f"{name}.wav", to_int16(mixed))
    plot_spectrogram(
        mixed.astype(np.float32), SAMPLE_RATE,
        out_dir / f"spectrogram_{name}.png",
        f"chord of units {units} on input {n} @ {args.base_freq} Hz",
    )
    metadata["files"].append({
        "name":       f"{name}.wav",
        "mode":       "unit_chord",
        "units":      [int(u) for u in units],
        "position":   int(args.position),
        "input_idx":  int(n),
        "base_freq":  float(args.base_freq),
        "duration_s": float(args.duration),
    })


def mode_unit_pair_stereo(streams, args, out_dir, metadata):
    u, v = args.unit, args.unit2
    if u is None or v is None:
        raise ValueError(
            "--unit and --unit2 are required for unit_pair_stereo"
        )
    n = args.input_idx
    cycle_u = streams[n, :, u]
    cycle_v = streams[n, :, v]
    audio_u = loop_waveform_at_frequency(
        cycle_u, args.base_freq, args.duration,
    )
    audio_v = loop_waveform_at_frequency(
        cycle_v, args.base_freq, args.duration,
    )
    env = envelope(len(audio_u))
    audio_u = audio_u * env
    audio_v = audio_v * env
    # Equalize peaks across the two channels
    peak = max(np.max(np.abs(audio_u)), np.max(np.abs(audio_v)), 1e-9)
    int_u = to_int16(audio_u / peak)
    int_v = to_int16(audio_v / peak)
    interleaved = np.empty((len(int_u), 2), dtype=np.int16)
    interleaved[:, 0] = int_u
    interleaved[:, 1] = int_v
    name = (f"unit_pair_stereo_u{u}_v{v}_pos{args.position}"
            f"_input{n}_f{args.base_freq:.0f}")
    write_wav(out_dir / f"{name}.wav", interleaved, n_channels=2)
    plot_spectrogram(
        ((audio_u + audio_v) / 2).astype(np.float32),
        SAMPLE_RATE,
        out_dir / f"spectrogram_{name}.png",
        (f"pair stereo: u={u} (L) + v={v} (R) "
         f"on input {n} @ {args.base_freq} Hz"),
    )
    metadata["files"].append({
        "name":       f"{name}.wav",
        "mode":       "unit_pair_stereo",
        "unit_left":  int(u),
        "unit_right": int(v),
        "position":   int(args.position),
        "input_idx":  int(n),
        "base_freq":  float(args.base_freq),
        "duration_s": float(args.duration),
    })


def mode_fiedler_tone(streams, args, out_dir, metadata):
    """Build the Fiedler-projection trajectory for one input and play
    it as a tone."""
    n = args.input_idx
    N, L, D = streams.shape

    # Per-input z and global Fiedler over (positions=just one) -- we
    # can't compute Fiedler from a single input because cospec requires
    # cross-input averaging. So we use the across-input Fiedler at each
    # layer and apply it to this single input.
    grad = np.gradient(streams, axis=1)
    x = streams - streams[:, 0:1, :]
    y = grad - grad[:, 0:1, :]
    z = x + 1j * y
    proj_traj = np.zeros(L, dtype=np.float32)
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
        centered = streams[n, l, :] - streams[:, l, :].mean(axis=0)
        proj_traj[l] = float(centered @ v2)

    audio = loop_waveform_at_frequency(
        proj_traj, args.base_freq, args.duration,
    )
    audio = audio * envelope(len(audio))
    name = (f"fiedler_tone_pos{args.position}_input{n}"
            f"_f{args.base_freq:.0f}")
    write_wav(out_dir / f"{name}.wav", to_int16(audio))
    plot_waveform(
        proj_traj, out_dir / f"waveform_{name}.png",
        f"Fiedler projection at pos {args.position}, input {n}",
    )
    plot_spectrogram(
        audio.astype(np.float32), SAMPLE_RATE,
        out_dir / f"spectrogram_{name}.png",
        f"Fiedler projection looped at {args.base_freq} Hz",
    )
    metadata["files"].append({
        "name":       f"{name}.wav",
        "mode":       "fiedler_tone",
        "position":   int(args.position),
        "input_idx":  int(n),
        "base_freq":  float(args.base_freq),
        "duration_s": float(args.duration),
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
        choices=["unit_tone", "unit_sweep", "unit_chord",
                 "unit_pair_stereo", "fiedler_tone"],
    )
    parser.add_argument("--position", type=int, default=5,
                        choices=[5, 6, 7])
    parser.add_argument("--unit", type=int, default=None)
    parser.add_argument("--unit2", type=int, default=None)
    parser.add_argument("--units", type=int, nargs="+", default=None)
    parser.add_argument("--input-idx", type=int, default=0,
                        help="Which (a, b) pair index to use")
    parser.add_argument(
        "--base-freq", type=float, default=220.0,
        help="Fundamental frequency of the looped tone (Hz)",
    )
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Output tone duration in seconds")
    parser.add_argument("--tone-duration", type=float, default=0.4,
                        help="Per-input tone duration for unit_sweep mode")
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
        "checkpoint":  str(args.checkpoint),
        "position":    int(args.position),
        "n_inputs":    int(streams.shape[0]),
        "n_sublayers": int(streams.shape[1]),
        "d_model":     int(streams.shape[2]),
        "sample_rate": SAMPLE_RATE,
        "files":       [],
    }

    if args.mode == "unit_tone":
        mode_unit_tone(streams, args, out_dir, metadata)
    elif args.mode == "unit_sweep":
        mode_unit_sweep(streams, args, out_dir, metadata)
    elif args.mode == "unit_chord":
        mode_unit_chord(streams, args, out_dir, metadata)
    elif args.mode == "unit_pair_stereo":
        mode_unit_pair_stereo(streams, args, out_dir, metadata)
    elif args.mode == "fiedler_tone":
        mode_fiedler_tone(streams, args, out_dir, metadata)

    with open(out_dir / "sonify_v2_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nFiles written:")
    for entry in metadata["files"]:
        print(f"  {entry['name']}")
    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
