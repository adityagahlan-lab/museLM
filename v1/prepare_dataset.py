import torch
from tokenizers import ByteLevelBPETokenizer

print("Loading tokenizer...")

tokenizer = ByteLevelBPETokenizer(
    "musellm_tokenizer-vocab.json",
    "musellm_tokenizer-merges.txt"
)

tokens = []

print("Tokenizing dataset...")

with open("dataset.txt", "r", encoding="utf-8") as f:

    chunk_size = 1024 * 1024  # 1 MB

    while True:
        chunk = f.read(chunk_size)

        if not chunk:
            break

        ids = tokenizer.encode(chunk).ids
        tokens.extend(ids)

        print(f"Tokens so far: {len(tokens):,}")

print("Saving tokenized dataset...")

torch.save(torch.tensor(tokens, dtype=torch.long), "tokens.pt")

print("Done!")
print(f"Total tokens: {len(tokens):,}")