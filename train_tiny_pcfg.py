"""
Train a tiny transformer (d_model=3, 32 layers) on a small PCFG, report a
quadrature score per epoch, and save the residual-stream tensor of a
single fixed inference 3 times per epoch (plus once at random init) for
later inspection.

Two output files per run, named by --mlp:

    residuals_{mlp_type}.npz
        residuals: float64, shape (S, L, T, D)
                       S = number of captures
                       L = 2 * n_layers + 1   (one snapshot per sublayer)
                       T = sequence length
                       D = d_model
        epoch:     int,     shape (S,)   epoch index (0 for the init capture)
        fraction:  float,   shape (S,)   fraction through the epoch
                                         (0.0, 1/3, 2/3, or 1.0)
        tokens:    int,     shape (T,)   input token ids of the inference
        vocab:     str,     shape (V,)   vocab list for decoding tokens

    residuals_{mlp_type}.csv
        Long format, one row per (capture, position, layer). Columns:
        capture_idx, epoch, fraction, position, token, layer,
        unit_0, unit_1, unit_2
        Rows are ordered by (capture_idx, position, layer), so for any
        choice of capture and position the consecutive rows trace the
        depth trajectory through the residual stream.

Quadrature score formula:
    score = 1 - (2/pi) * mean_over_units(||delta_theta| - pi/2|)
where delta_theta is the Hilbert phase offset between x(l) and dx(l).
1 = ideal quadrature on every unit, 0 = no quadrature anywhere.

MLP variant is selectable via --mlp:
    --mlp gelu      Standard 2-matrix MLP with GELU (default).
    --mlp swiglu    SwiGLU: 3-matrix gated MLP, bias-free (LLaMA-style).
"""

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert as scipy_hilbert
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# Vocabulary and grammar
# -----------------------------------------------------------------------------

VOCAB = ["<pad>", "<bos>", "alice", "bob", "carol", "dave",
         "sees", "likes", "and", "."]
STOI = {tok: i for i, tok in enumerate(VOCAB)}
ITOS = {i: tok for i, tok in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PAD_ID = STOI["<pad>"]
BOS_ID = STOI["<bos>"]

NOUNS = ["alice", "bob", "carol", "dave"]
VERBS = ["sees", "likes"]
P_CONJ = 0.4

MAX_LEN = 10  # input length after [:-1] is 9, exactly the max sentence length


def sample_np(rng):
    if rng.random() < P_CONJ:
        return [rng.choice(NOUNS), "and", rng.choice(NOUNS)]
    return [rng.choice(NOUNS)]


def sample_sentence(rng):
    subj = sample_np(rng)
    verb = rng.choice(VERBS)
    obj = sample_np(rng)
    return ["<bos>"] + subj + [verb] + obj + ["."]


def sample_max_sentence(rng):
    """Always-maximum-structure: <bos> N and N V N and N ."""
    return [
        "<bos>",
        rng.choice(NOUNS), "and", rng.choice(NOUNS),
        rng.choice(VERBS),
        rng.choice(NOUNS), "and", rng.choice(NOUNS),
        ".",
    ]


def encode(tokens):
    return [STOI[t] for t in tokens]


def decode(ids):
    return [ITOS[int(i)] for i in ids]


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

class PCFGDataset(Dataset):
    def __init__(self, num_examples, seed=0, max_len=MAX_LEN):
        rng = random.Random(seed)
        self.max_len = max_len
        self.examples = []
        for _ in range(num_examples):
            toks = encode(sample_sentence(rng))
            assert len(toks) <= max_len, f"sentence too long ({len(toks)})"
            toks = toks + [PAD_ID] * (max_len - len(toks))
            self.examples.append(torch.tensor(toks, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq = self.examples[idx]
        return seq[:-1], seq[1:]


def make_eval_batch(num_examples, seed):
    """Fixed batch of max-structure sentences (used for the quadrature score)."""
    rng = random.Random(seed)
    seqs = []
    for _ in range(num_examples):
        toks = encode(sample_max_sentence(rng))
        toks = toks + [PAD_ID] * (MAX_LEN - len(toks))
        seqs.append(toks)
    full = torch.tensor(seqs, dtype=torch.long)
    return full[:, :-1]


def make_inference_input(seed):
    """Single fixed input sequence for residual-stream capture."""
    rng = random.Random(seed)
    toks = encode(sample_max_sentence(rng))
    return torch.tensor(toks, dtype=torch.long)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class MLP(nn.Module):
    """Standard 2-matrix MLP with GELU."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SwiGLU(nn.Module):
    """SwiGLU: gated MLP with SiLU on the gate branch, bias-free."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def make_mlp(mlp_type, d_model, d_ff):
    if mlp_type == "gelu":
        return MLP(d_model, d_ff)
    if mlp_type == "swiglu":
        return SwiGLU(d_model, d_ff)
    raise ValueError(f"unknown mlp_type {mlp_type!r} (use 'gelu' or 'swiglu')")


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, mlp_type="gelu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = make_mlp(mlp_type, d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=3, n_layers=32, n_heads=1,
                 d_ff=12, max_len=MAX_LEN, mlp_type="gelu"):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, mlp_type=mlp_type)
             for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# -----------------------------------------------------------------------------
# Residual capture
# -----------------------------------------------------------------------------

@torch.no_grad()
def _capture_residuals(model, batch, device):
    """Forward pass recording residual stream at every sublayer.
    Returns (L, N, T, D) array with L = 2 * n_layers + 1."""
    model.eval()
    x = batch.to(device)
    B, T = x.shape
    pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    h = model.tok_emb(x) + model.pos_emb(pos)
    snaps = [h.detach().cpu().numpy()]
    for block in model.blocks:
        h = h + block.attn(block.ln1(h))
        snaps.append(h.detach().cpu().numpy())
        h = h + block.mlp(block.ln2(h))
        snaps.append(h.detach().cpu().numpy())
    return np.stack(snaps, axis=0)


def capture_inference(model, single_input, device):
    """Run a single inference and return residual stream of shape (L, T, D)."""
    batch = single_input.unsqueeze(0)
    snaps = _capture_residuals(model, batch, device)
    return snaps[:, 0, :, :].astype(np.float64)


# -----------------------------------------------------------------------------
# Quadrature score
# -----------------------------------------------------------------------------

def quadrature_score(model, batch, device, position=4):
    """Single scalar in [0, 1]; see module docstring."""
    snaps = _capture_residuals(model, batch, device)
    x = snaps[:, :, position, :].transpose(1, 0, 2).astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    dx = np.diff(x, axis=1)
    x_pair = x[:, :-1, :]

    z_x  = scipy_hilbert(x_pair, axis=1)
    z_dx = scipy_hilbert(dx,     axis=1)
    ratio = z_dx / (z_x + 1e-30)
    dtheta_per_input = np.angle(ratio.mean(axis=1))

    z = np.exp(1j * dtheta_per_input)
    dtheta_per_unit = np.angle(z.mean(axis=0))

    deviation = np.abs(np.abs(dtheta_per_unit) - np.pi / 2)
    return float(1.0 - (2.0 / np.pi) * deviation.mean())


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------

def save_npz(out_path, residuals, epochs, fractions, tokens):
    np.savez(
        out_path,
        residuals=residuals,
        epoch=epochs.astype(np.int32),
        fraction=fractions.astype(np.float64),
        tokens=tokens.astype(np.int32),
        vocab=np.array(VOCAB),
    )


def save_csv(out_path, residuals, epochs, fractions, tokens):
    """Long-format CSV. Rows = (capture, position, layer), columns = unit values.

    Within a fixed (capture_idx, position) the consecutive rows trace the
    depth trajectory through the residual stream, so reading down the
    layer column for that filter shows changes across depth for one pass.
    """
    S, L, T, D = residuals.shape
    fieldnames = ["capture_idx", "epoch", "fraction",
                  "position", "token", "layer"]
    fieldnames += [f"unit_{d}" for d in range(D)]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for s in range(S):
            ep = int(epochs[s])
            fr = float(fractions[s])
            for p in range(T):
                tok_str = VOCAB[int(tokens[p])]
                for l in range(L):
                    row = [s, ep, fr, p, tok_str, l]
                    row.extend(float(residuals[s, l, p, d]) for d in range(D))
                    writer.writerow(row)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(
    num_train=20000,
    num_val=2000,
    num_eval=64,
    batch_size=128,
    epochs=100,
    lr=3e-4,
    seed=0,
    device=None,
    eval_position=4,
    mlp_type="gelu",
    out_path=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    train_ds = PCFGDataset(num_train, seed=seed)
    val_ds = PCFGDataset(num_val, seed=seed + 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    eval_batch = make_eval_batch(num_eval, seed=seed + 100)
    inference_input = make_inference_input(seed=seed + 200)
    inference_tokens = decode(inference_input.tolist())

    model = TinyTransformer(VOCAB_SIZE, mlp_type=mlp_type).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}  params: {n_params:,}  "
          f"mlp: {mlp_type}  quadrature position: {eval_position}")
    print(f"inference input: {' '.join(inference_tokens)}")

    cap_residuals = []   # list of (L, T, D) arrays
    cap_epoch = []
    cap_fraction = []

    def record(epoch, fraction):
        cap_residuals.append(capture_inference(model, inference_input, device))
        cap_epoch.append(epoch)
        cap_fraction.append(fraction)

    record(epoch=0, fraction=0.0)
    qscore = quadrature_score(model, eval_batch, device, position=eval_position)
    print(f"epoch   0  (init)                                quad {qscore:.3f}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    n_batches = len(train_loader)
    capture_indices = {
        n_batches // 3 - 1:     1.0 / 3,
        2 * n_batches // 3 - 1: 2.0 / 3,
        n_batches - 1:          1.0,
    }

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
                ignore_index=PAD_ID,
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            n += 1
            if batch_idx in capture_indices:
                record(epoch=epoch, fraction=capture_indices[batch_idx])
                model.train()  # restore train mode after capture
        train_loss = total / n

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    y.reshape(-1),
                    ignore_index=PAD_ID,
                )
                total += loss.item()
                n += 1
        val_loss = total / n

        qscore = quadrature_score(
            model, eval_batch, device, position=eval_position
        )
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  "
              f"val {val_loss:.4f}  quad {qscore:.3f}")

    # Save outputs.
    if out_path is None:
        out_path = f"residuals_{mlp_type}.npz"
    npz_path = Path(out_path)
    csv_path = npz_path.with_suffix(".csv")
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    residuals = np.stack(cap_residuals, axis=0)        # (S, L, T, D)
    epochs_arr = np.array(cap_epoch, dtype=np.int32)
    fractions_arr = np.array(cap_fraction, dtype=np.float64)
    tokens_arr = inference_input.numpy()

    save_npz(npz_path, residuals, epochs_arr, fractions_arr, tokens_arr)
    save_csv(csv_path, residuals, epochs_arr, fractions_arr, tokens_arr)
    print(f"\nsaved {len(cap_residuals)} residual captures to:")
    print(f"  {npz_path}")
    print(f"  {csv_path}")

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp", choices=["gelu", "swiglu"], default="gelu",
                        help="MLP variant in each block (default: gelu).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-position", type=int, default=4,
                        help="Token position used to compute the quadrature "
                             "score (0..8). Default 4 = the verb.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for residual captures npz "
                             "(.csv will share the stem). "
                             "Default: residuals_{mlp_type}.npz")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        mlp_type=args.mlp,
        epochs=args.epochs,
        seed=args.seed,
        eval_position=args.eval_position,
        out_path=args.out,
    )
