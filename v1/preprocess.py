import pandas as pd
import ast
import re
import torch
from tokenizers import ByteLevelBPETokenizer
import os

SAVE_EVERY = 5000
chunk = 1
chunk_ids = []
print("Loading tokenizer...")

tokenizer = ByteLevelBPETokenizer(
    "musellm_tokenizer-vocab.json",
    "musellm_tokenizer-merges.txt"
)

BAD_PATTERNS = re.compile(
    r"<html|<body|<div|<script|<style|<input|<button|<head|"
    r"def |class |import |function\(|const |var |let |==>|```",
    re.IGNORECASE
)

all_ids = []

# ---------------- UltraChat ---------------- #

print("Loading UltraChat...")

df = pd.read_csv("train_sft.csv")

kept = 0
skipped = 0

for row in df["messages"]:

    try:
        fixed = re.sub(r"}\s*{", "}, {", str(row))
        conversation = ast.literal_eval(fixed)

        full_text = " ".join(msg["content"] for msg in conversation)

        if BAD_PATTERNS.search(full_text):
            skipped += 1
            continue

        turns = []

        for msg in conversation:

            role = msg["role"]
            text = msg["content"].replace("\n", " ").strip()

            if role == "user":
                turns.append(f"<user> {text}")

            elif role == "assistant":
                turns.append(f"<bot> {text}")

        turns.append("<eos>")

        ids = tokenizer.encode(" ".join(turns)).ids

        chunk_ids.extend(ids)

        kept += 1

        if kept % SAVE_EVERY == 0:
            print(f"Saving chunk {chunk}...")
            
            os.makedirs("token_chunks", exist_ok=True)
            torch.save(
                torch.tensor(chunk_ids, dtype=torch.long),
                f"token_chunks/tokens_chunk_{chunk}.pt"
            )
            print(f"Saved tokens_chunk_{chunk}.pt")
            chunk += 1
            chunk_ids = []

        if kept % 1000 == 0:
            print(f"UltraChat: {kept:,} conversations")

    except Exception:
        skipped += 1

print(f"UltraChat kept: {kept:,}")
print(f"UltraChat skipped: {skipped:,}")

# ---------------- OASST ---------------- #

print("\nLoading OpenAssistant...")

df = pd.read_csv("oasst1-train.csv")

kept = 0
skipped = 0

for _, row in df.iterrows():

    try:

        text = str(row.get("text", "")).replace("\n", " ").strip()

        role = str(row.get("role", "")).strip()

        if BAD_PATTERNS.search(text):
            skipped += 1
            continue

        if role == "prompter":
            ids = tokenizer.encode(f"<user> {text}").ids

        elif role == "assistant":
            ids = tokenizer.encode(f"<bot> {text} <eos>").ids

        else:
            continue

        chunk_ids.extend(ids)

        kept += 1
        
        if kept % SAVE_EVERY == 0:
            print(f"Saving chunk {chunk}...")
            
            os.makedirs("token_chunks", exist_ok=True)
            torch.save(
                torch.tensor(chunk_ids, dtype=torch.long),
                f"token_chunks/tokens_chunk_{chunk}.pt"
            )
            print(f"Saved tokens_chunk_{chunk}.pt")
            chunk += 1
            chunk_ids = []

        if kept % 10000 == 0:
            print(f"OASST: {kept:,} rows")

    except Exception:
        skipped += 1

print(f"OASST kept: {kept:,}")
print(f"OASST skipped: {skipped:,}")

# ---------------- Save ---------------- #

print("\nSaving tokenized dataset...")

# Save remaining tokens

if len(chunk_ids) > 0:
    
    os.makedirs("token_chunks", exist_ok=True)
    torch.save(
        torch.tensor(chunk_ids, dtype=torch.long),
        f"token_chunks/tokens_chunk_{chunk}.pt"
    )

import glob

print(f"Finished preprocessing.")
print(f"Created {len(glob.glob('token_chunks/tokens_chunk_*.pt'))} token chunks.")