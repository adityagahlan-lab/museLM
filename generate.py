import torch
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

# ---- Load trained model ----
num_layers = 6
model = MuseLM(vocab_size, embedding_dim, block_size, num_heads, num_layers)
checkpoint = torch.load("musellm.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

print("\nChatLM ready! Type your message. Type 'quit' to exit.\n")

# ---- Multi-turn chat loop (Option A: in-context memory) ----
conversation_ids = []  # full conversation history as token ids

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    if not user_input:
        continue

    # Format and encode user turn
    user_turn = f"<user> {user_input} <bot> "
    conversation_ids += encode(user_turn)

    # Sliding window — trim if exceeding block_size
    if len(conversation_ids) > block_size - 50:
        conversation_ids = conversation_ids[-(block_size - 50):]

    # Generate bot response using model.generate()
    idx = torch.tensor([conversation_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        idx,
        max_new_tokens=150,
        temperature=0.7,
        top_k=20
    )

    # Extract only the newly generated tokens
    response_ids = output_ids[0][len(conversation_ids):].tolist()

    # Decode and clean special tokens from response
    response_text = decode(response_ids)
    for tag in ["<eos>", "<user>", "<bot>"]:
        response_text = response_text.replace(tag, "")
    response_text = response_text.strip()

    print(f"Bot: {response_text}\n")

    # Append bot response to history
    conversation_ids += response_ids

    # Trim again after bot response is added
    if len(conversation_ids) > block_size - 50:
        conversation_ids = conversation_ids[-(block_size - 50):]