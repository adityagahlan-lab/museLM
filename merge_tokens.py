import glob
import torch
import re

files = sorted(
    glob.glob("token_chunks/tokens_chunk_*.pt"),
    key=lambda x: int(re.search(r"tokens_chunk_(\d+)\.pt", x).group(1))
)

all_tokens = []

for f in files:
    print("Loading", f)
    all_tokens.append(torch.load(f))

tokens = torch.cat(all_tokens)

torch.save(tokens, "tokens.pt")

print("Done!")

print(f"Total tokens: {len(tokens):,}")