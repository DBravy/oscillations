"""
Train three architecturally ablated variants on MSB-first base-4 addition,
all with matched hyperparameters and seeds.

  full:       attention and MLP both learned (the original model)
  attn_only:  attention learned, MLP layers replaced with zero contribution
              (so block is x = x + attn(x))
  mlp_only:   MLP learned, attention replaced with fixed uniform causal
              averaging (no learnable attention parameters)

In all three, the residual capture interface is identical and produces
the same 17-sublayer trajectory, so downstream phase analysis runs the
same way. For attn_only the captured "mlp" sublayer equals the captured
"attn" sublayer (the MLP added zero), so mlp_delta is zero by construction.
For mlp_only the "attn" delta is the fixed uniform mixer's contribution
rather than learned attention.

Outputs (in --out-root):
  ablation_run/full/model_trained.pt        + training_history.json + model_random_init.pt
  ablation_run/attn_only/model_trained.pt   + training_history.json + model_random_init.pt
  ablation_run/mlp_only/model_trained.pt    + training_history.json + model_random_init.pt

Usage:
  python train_ablations.py --out-root ablation_run --n-steps 20000
  python train_ablations.py --modes full attn_only       (subset)
  python train_ablations.py --skip-existing              (don't retrain modes
                                                         whose checkpoint exists)
"""

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse the dataset and tokenization machinery
from toy_transformer_addition import (
    VOCAB_SIZE,
    SEQ_LEN,
    AdditionDataset,
    make_train_test_split,
    Attention,
    MLP,
)


ALL_MODES = ["full", "attn_only", "mlp_only"]


# ---------------------------------------------------------------------------
# Model config and ablated transformer
# ---------------------------------------------------------------------------

@dataclass
class AblatedConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len:    int = SEQ_LEN
    d_model:    int = 64
    n_layers:   int = 8
    n_heads:    int = 2
    d_head:     int = 32
    mlp_hidden: int = 256
    init_scale: float = 0.02
    mode: str = "full"   # one of ALL_MODES


def fixed_uniform_causal_attn(x):
    """Causal uniform averaging across sequence positions.

    For each position i, output[i] is the mean of x[0..i] (inclusive).
    No learnable parameters. Matches the residual stream's expected shape
    so the rest of the architecture is unchanged.
    """
    _, T, _ = x.shape
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype))
    mask = mask / mask.sum(dim=-1, keepdim=True)
    return torch.einsum("ij,bjd->bid", mask, x)


class AblatedBlock(nn.Module):
    def __init__(self, cfg: AblatedConfig):
        super().__init__()
        self.mode = cfg.mode
        # Always allocate the learnable submodule so that param count and
        # initialization match across modes (we just don't use the unused
        # ones in forward).
        if self.mode in ("full", "attn_only"):
            self.attn = Attention(cfg)
        else:
            self.attn = None
        if self.mode in ("full", "mlp_only"):
            self.mlp = MLP(cfg)
        else:
            self.mlp = None

    def forward(self, x, capture=None, layer_idx=None):
        # Attention sublayer
        if self.mode == "mlp_only":
            attn_out = fixed_uniform_causal_attn(x)
        else:
            attn_out = self.attn(x)
        x = x + attn_out
        if capture is not None:
            capture.append(
                ("attn", layer_idx, x.detach().to(torch.float32).cpu())
            )

        # MLP sublayer
        if self.mode == "attn_only":
            mlp_out = torch.zeros_like(x)
        else:
            mlp_out = self.mlp(x)
        x = x + mlp_out
        if capture is not None:
            capture.append(
                ("mlp", layer_idx, x.detach().to(torch.float32).cpu())
            )
        return x


class AblatedTransformer(nn.Module):
    def __init__(self, cfg: AblatedConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [AblatedBlock(cfg) for _ in range(cfg.n_layers)]
        )
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
        _, T = x.shape
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
    """Return overall token / sequence accuracy and per-position token accuracy."""
    model.eval()
    total_correct_tokens = 0
    total_tokens = 0
    total_correct_seqs = 0
    total_seqs = 0
    # Per-position counts for the three answer-output positions.
    # In x of length 8, answer-output positions are indices 5, 6, 7.
    per_pos_correct = {5: 0, 6: 0, 7: 0}
    per_pos_total = {5: 0, 6: 0, 7: 0}
    with torch.no_grad():
        for x, y, loss_mask in loader:
            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct = (preds == y) & loss_mask
            total_correct_tokens += correct.sum().item()
            total_tokens += loss_mask.sum().item()
            seq_correct = correct.sum(dim=1) == loss_mask.sum(dim=1)
            total_correct_seqs += seq_correct.sum().item()
            total_seqs += x.size(0)
            for pos in (5, 6, 7):
                per_pos_correct[pos] += correct[:, pos].sum().item()
                per_pos_total[pos] += loss_mask[:, pos].sum().item()
    return {
        "token_acc":    total_correct_tokens / max(1, total_tokens),
        "sequence_acc": total_correct_seqs / max(1, total_seqs),
        "per_pos_acc": {
            pos: per_pos_correct[pos] / max(1, per_pos_total[pos])
            for pos in (5, 6, 7)
        },
    }


def train_one_mode(
    mode,
    out_dir,
    seed,
    n_steps,
    batch_size,
    lr,
    weight_decay,
    log_every,
    eval_every,
    cfg_kwargs,
):
    torch.manual_seed(seed)
    random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AblatedConfig(mode=mode, **cfg_kwargs)
    print(f"\n=== Training mode = {mode} ===")
    print(f"  config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, mlp_hidden={cfg.mlp_hidden}")
    print(f"  device: {device}")

    train_pairs, test_pairs = make_train_test_split(seed=seed, train_frac=0.8)
    train_loader = DataLoader(
        AdditionDataset(train_pairs),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        AdditionDataset(test_pairs),
        batch_size=batch_size,
        shuffle=False,
    )
    full_loader = DataLoader(
        AdditionDataset(train_pairs + test_pairs),
        batch_size=batch_size,
        shuffle=False,
    )

    model = AblatedTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params}")

    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(cfg)},
        out_dir / "model_random_init.pt",
    )

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
            print(f"  [step {step:6d}] loss={loss.item():.4f}")

        if step % eval_every == 0 or step == n_steps - 1:
            train_metrics = evaluate(model, train_loader, device)
            test_metrics = evaluate(model, test_loader, device)
            entry = {
                "step":          step,
                "loss":          loss.item(),
                "train_token_acc": train_metrics["token_acc"],
                "train_seq_acc":   train_metrics["sequence_acc"],
                "train_per_pos":   train_metrics["per_pos_acc"],
                "test_token_acc":  test_metrics["token_acc"],
                "test_seq_acc":    test_metrics["sequence_acc"],
                "test_per_pos":    test_metrics["per_pos_acc"],
            }
            history.append(entry)
            print(
                f"    eval: train_seq={train_metrics['sequence_acc']:.3f}  "
                f"test_seq={test_metrics['sequence_acc']:.3f}  "
                f"per_pos(test)={ {p: round(v, 3) for p, v in test_metrics['per_pos_acc'].items()} }"
            )

        step += 1

    # Final eval on the full set so the comparison script can read it directly
    full_metrics = evaluate(model, full_loader, device)

    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(cfg)},
        out_dir / "model_trained.pt",
    )
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(
            {
                "history":      history,
                "final_full":   full_metrics,
                "config":       asdict(cfg),
                "n_params":     n_params,
                "n_steps":      n_steps,
                "seed":         seed,
            },
            f,
            indent=2,
        )

    print(f"  done; final test_seq_acc = {history[-1]['test_seq_acc']:.3f}, "
          f"full per_pos = {full_metrics['per_pos_acc']}")
    print(f"  saved to {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="ablation_run")
    parser.add_argument("--modes", type=str, nargs="+", default=ALL_MODES,
                        choices=ALL_MODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--d-head", type=int, default=32)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a mode if its model_trained.pt already exists")
    args = parser.parse_args()

    cfg_kwargs = dict(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_head=args.d_head,
        mlp_hidden=args.mlp_hidden,
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        out_dir = out_root / mode
        ckpt = out_dir / "model_trained.pt"
        if args.skip_existing and ckpt.exists():
            print(f"\n=== Skipping mode = {mode} (checkpoint exists) ===")
            continue
        train_one_mode(
            mode=mode,
            out_dir=out_dir,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            log_every=args.log_every,
            eval_every=args.eval_every,
            cfg_kwargs=cfg_kwargs,
        )


if __name__ == "__main__":
    main()
