"""
Toy transformer for MSB-first base-4 two-digit addition.

Task format:
  Input:  a1 a0 + b1 b0 =
  Output: c2 c1 c0   (most significant digit first)

Where a, b in [0, 15] (two base-4 digits each), sum c in [0, 30] (three base-4 digits).
MSB-first output forces the model to compute the full answer and store
the unwritten suffix across autoregressive steps.

Architecture:
  - d_model = 8
  - n_layers = 8
  - n_heads = 2, d_head = 4
  - mlp_hidden = 32 (4x ratio)
  - No layernorm (so residual stream is exactly the trajectory we analyze)
  - Learned positional embeddings
  - Tied embedding/unembedding

Hooks:
  Captures the residual stream after every sublayer (attention write and mlp write).
  With 8 layers, this gives 17 captures total (embedding + 16 sublayer outputs).
  Stored in float32 for FFT/arctan2 stability even if the model runs in lower precision.
"""

import math
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Vocabulary and data
# ---------------------------------------------------------------------------

# Tokens: 0,1,2,3 (digits), '+', '=', PAD
DIGIT_TOKENS = [0, 1, 2, 3]
PLUS_TOKEN = 4
EQ_TOKEN = 5
PAD_TOKEN = 6
VOCAB_SIZE = 7

# Sequence layout (length 8):
#   pos 0: a1   (high digit of a)
#   pos 1: a0   (low digit of a)
#   pos 2: '+'
#   pos 3: b1
#   pos 4: b0
#   pos 5: '='
#   pos 6: c2   (high digit of sum)
#   pos 7: c1
#   pos 8: c0   (low digit of sum)
# That's 9 tokens. We'll use seq_len = 9.
SEQ_LEN = 9

# Positions where the model produces output (predicts next token).
# At position 5 ('='), the model must predict c2.
# At position 6 (c2), it must predict c1.
# At position 7 (c1), it must predict c0.
ANSWER_INPUT_POSITIONS = [5, 6, 7]
ANSWER_TARGET_POSITIONS = [6, 7, 8]


def encode_example(a: int, b: int):
    """Build the token sequence for a + b = c with MSB-first output."""
    assert 0 <= a < 16 and 0 <= b < 16
    a1, a0 = a // 4, a % 4
    b1, b0 = b // 4, b % 4
    c = a + b
    c2 = c // 16
    c1 = (c // 4) % 4
    c0 = c % 4
    tokens = [a1, a0, PLUS_TOKEN, b1, b0, EQ_TOKEN, c2, c1, c0]
    assert len(tokens) == SEQ_LEN
    return tokens


class AdditionDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        tokens = encode_example(a, b)
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        # Mask: only score loss at the three answer-output positions.
        # tokens[1:] aligns indices: target at output index t is tokens[t+1].
        # Output index 5 -> predicts tokens[6] = c2 (target position in y is index 5)
        # Output index 6 -> predicts tokens[7] = c1 (y index 6)
        # Output index 7 -> predicts tokens[8] = c0 (y index 7)
        loss_mask = torch.zeros(SEQ_LEN - 1, dtype=torch.bool)
        loss_mask[5] = True
        loss_mask[6] = True
        loss_mask[7] = True
        return x, y, loss_mask


def make_train_test_split(seed=0, train_frac=0.8):
    """Split the 256 (a, b) pairs into train and test."""
    pairs = [(a, b) for a in range(16) for b in range(16)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_train = int(len(pairs) * train_frac)
    return pairs[:n_train], pairs[n_train:]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len: int = SEQ_LEN
    d_model: int = 64
    n_layers: int = 12
    n_heads: int = 2
    d_head: int = 32
    mlp_hidden: int = 256
    init_scale: float = 0.02
    mlp_type: str = "gelu"  # "gelu" or "swiglu"


class Attention(nn.Module):
    """Single-block multi-head self-attention with causal mask. No layernorm."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model
        inner = cfg.n_heads * cfg.d_head
        self.W_qkv = nn.Linear(cfg.d_model, 3 * inner, bias=False)
        self.W_o = nn.Linear(inner, cfg.d_model, bias=False)
        # Causal mask (1 = keep, 0 = mask out)
        mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len))
        self.register_buffer("causal_mask", mask.bool())

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.W_qkv(x)  # (B, T, 3 * n_heads * d_head)
        qkv = qkv.view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        # (B, n_heads, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.fc2 = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_val = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_out = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_out(F.silu(self.w_gate(x)) * self.w_val(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = SwiGLUMLP(cfg) if cfg.mlp_type == "swiglu" else MLP(cfg)

    def forward(self, x, capture=None, layer_idx=None):
        """If capture is not None, append residual after attn and after mlp."""
        x = x + self.attn(x)
        if capture is not None:
            capture.append(("attn", layer_idx, x.detach().to(torch.float32).cpu()))
        x = x + self.mlp(x)
        if capture is not None:
            capture.append(("mlp", layer_idx, x.detach().to(torch.float32).cpu()))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        # Tied unembedding: use tok_emb.weight transpose for output
        self.register_buffer("pos_ids", torch.arange(cfg.seq_len).unsqueeze(0))
        self._init_weights()

    def _init_weights(self):
        scale = self.cfg.init_scale
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=scale)
            else:
                nn.init.zeros_(p)

    def forward(self, x, capture_residuals=False):
        """
        x: (B, T) long tokens
        Returns logits (B, T, vocab_size).
        If capture_residuals, also returns a list of captures:
          [("embed", None, tensor), ("attn", 0, tensor), ("mlp", 0, tensor), ...]
        Each tensor is (B, T, d_model) on CPU in float32.
        """
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb(self.pos_ids[:, :T])
        capture = [] if capture_residuals else None
        if capture is not None:
            capture.append(("embed", None, h.detach().to(torch.float32).cpu()))
        for i, block in enumerate(self.blocks):
            h = block(h, capture=capture, layer_idx=i)
        logits = h @ self.tok_emb.weight.t()
        if capture_residuals:
            return logits, capture
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    total_correct_tokens = 0
    total_tokens = 0
    total_correct_seqs = 0
    total_seqs = 0
    with torch.no_grad():
        for x, y, loss_mask in loader:
            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct = (preds == y) & loss_mask
            total_correct_tokens += correct.sum().item()
            total_tokens += loss_mask.sum().item()
            seq_correct = (correct.sum(dim=1) == loss_mask.sum(dim=1))
            total_correct_seqs += seq_correct.sum().item()
            total_seqs += x.size(0)
    return {
        "token_acc": total_correct_tokens / max(1, total_tokens),
        "sequence_acc": total_correct_seqs / max(1, total_seqs),
    }


def train(
    out_dir="toy_transformer_run_swiglu200k",
    seed=0,
    n_steps=200000,
    batch_size=64,
    lr=1e-3,
    weight_decay=0.99,
    log_every=500,
    eval_every=10000,
    save_random_init=True,
    cfg=None,
):
    torch.manual_seed(seed)
    random.seed(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg is None:
        cfg = ModelConfig()

    # Build datasets
    train_pairs, test_pairs = make_train_test_split(seed=seed, train_frac=0.8)
    train_ds = AdditionDataset(train_pairs)
    test_ds = AdditionDataset(test_pairs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = ToyTransformer(cfg).to(device)

    # Save the random-init weights for trained vs random comparison.
    if save_random_init:
        torch.save(
            {"state_dict": model.state_dict(), "config": asdict(cfg)},
            out_path / "model_random_init.pt",
        )

    # AdamW with high weight decay encourages grokking on modular tasks.
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )

    history = []
    step = 0
    train_iter = iter(train_loader)
    while step < n_steps:
        try:
            x, y, loss_mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y, loss_mask = next(train_iter)
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)

        model.train()
        logits = model(x)
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1),
            reduction="none",
        ).reshape(y.shape)
        loss = (loss_per_token * loss_mask).sum() / loss_mask.sum()

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % log_every == 0:
            print(f"[step {step:6d}] loss={loss.item():.4f}")

        if step % eval_every == 0 or step == n_steps - 1:
            train_metrics = evaluate(model, train_loader, device)
            test_metrics = evaluate(model, test_loader, device)
            entry = {
                "step": step,
                "loss": loss.item(),
                "train_token_acc": train_metrics["token_acc"],
                "train_seq_acc": train_metrics["sequence_acc"],
                "test_token_acc": test_metrics["token_acc"],
                "test_seq_acc": test_metrics["sequence_acc"],
            }
            history.append(entry)
            print(
                f"  eval: train_seq_acc={train_metrics['sequence_acc']:.3f}  "
                f"test_seq_acc={test_metrics['sequence_acc']:.3f}"
            )

        step += 1

    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(cfg)},
        out_path / "model_trained.pt",
    )
    with open(out_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, cfg, history


# ---------------------------------------------------------------------------
# Residual capture utility for downstream analysis
# ---------------------------------------------------------------------------

def capture_residual_streams(model, pairs, device, batch_size=64):
    """
    Run the model on a list of (a, b) pairs, capturing the residual stream
    at every sublayer.

    Returns:
      tokens: (N, seq_len) long
      captures: list of dicts, one per sublayer, each containing:
        - kind: 'embed' | 'attn' | 'mlp'
        - layer: int or None
        - residual: (N, seq_len, d_model) float32
      The list is in depth order: embed, attn_0, mlp_0, attn_1, mlp_1, ...
    """
    model.eval()
    ds = AdditionDataset(pairs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_tokens = []
    # captures_by_sublayer[i] accumulates (B, T, D) tensors for sublayer i
    captures_by_sublayer = None
    sublayer_meta = None

    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            _, capture = model(x, capture_residuals=True)
            if captures_by_sublayer is None:
                captures_by_sublayer = [[] for _ in capture]
                sublayer_meta = [(kind, layer) for (kind, layer, _) in capture]
            for i, (_, _, t) in enumerate(capture):
                captures_by_sublayer[i].append(t)
            all_tokens.append(x.cpu())

    tokens = torch.cat(all_tokens, dim=0)
    captures = []
    for i, (kind, layer) in enumerate(sublayer_meta):
        residual = torch.cat(captures_by_sublayer[i], dim=0)
        captures.append({"kind": kind, "layer": layer, "residual": residual})
    return tokens, captures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-type", choices=["gelu", "swiglu"], default="gelu")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or f"toy_transformer_run{'_' + args.mlp_type if args.mlp_type != 'gelu' else ''}"
    cfg = ModelConfig(mlp_type=args.mlp_type)
    model, cfg, history = train(out_dir=out_dir, cfg=cfg)
    print("Done training. Final entry:", history[-1])
    print(f"Model and history saved to ./{out_dir}/")
