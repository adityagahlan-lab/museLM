import torch
import pandas as pd
import ast
import re
import math
torch.set_num_threads(4)
from model import MuseLM, embedding_dim, block_size, num_heads

# ---- Device setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---- Load tokenizer ----
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(
    "musellm_tokenizer-vocab.json",
    "musellm_tokenizer-merges.txt"
)

vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size (BPE, fixed): {vocab_size}")

def encode(s):
    return tokenizer.encode(s).ids

def decode(ids):
    return tokenizer.decode(ids)

# ---- Load preprocessed tokens ----

print("Loading tokenized dataset...")

data = torch.load("tokens.pt")

print(f"Loaded {len(data):,} tokens.")

# ---- Train/val split ----
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens: {len(val_data):,}")

# ---- Batch loading ----
batch_size = 16

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ---- Model setup ----
num_layers = 6
model = MuseLM(vocab_size, embedding_dim, block_size, num_heads, num_layers)
model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# ---- LR schedule ----
max_iters = 75000
warmup_iters = 500
eval_interval = 1000

def get_lr(it):
    if it < warmup_iters:
        return 3e-4 * (it + 1) / warmup_iters
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return 3e-4 * 0.5 * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(20)
        for k in range(20):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

best_val_loss = float('inf')
patience = 8
no_improve_count = 0

for iter in range(max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            no_improve_count = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": iter,
                "best_val_loss": best_val_loss,
                }, "musellm.pt")
            print(f"  -> new best model saved (val loss {best_val_loss:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at step {iter} — no improvement for {patience} evals")
                break

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if (iter + 1) % 10000 == 0:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": iter,
            "best_val_loss": best_val_loss,
        }, f"checkpoint_{iter+1}.pt")
        
        print(f"Checkpoint saved: checkpoint_{iter+1}.pt")

print(f"\nBest val loss achieved: {best_val_loss:.4f}")
print("Best model saved to musellm.pt")