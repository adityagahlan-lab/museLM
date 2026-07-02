import torch
import streamlit as st
from model import MuseLM, embedding_dim, block_size, num_heads
from tokenizers import ByteLevelBPETokenizer

# ---- Page config ----
st.set_page_config(
    page_title="MuseLM",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 MuseLM")
st.caption("A conversational language model built from scratch.")

# ---- Load model and tokenizer (cached so it only loads once) ----
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = ByteLevelBPETokenizer(
        "musellm_tokenizer-vocab.json",
        "musellm_tokenizer-merges.txt"
    )

    num_layers = 6
    vocab_size = tokenizer.get_vocab_size()
    model = MuseLM(vocab_size, embedding_dim, block_size, num_heads, num_layers)
    checkpoint = torch.load("musellm.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    return model, tokenizer, device

model, tokenizer, device = load_model()

def encode(s):
    return tokenizer.encode(s).ids

def decode(ids):
    return tokenizer.decode(ids)

# ---- Session state for conversation history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_ids" not in st.session_state:
    st.session_state.conversation_ids = []

# ---- Display chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat input ----
user_input = st.chat_input("Type a message...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Format and encode user turn
    user_turn = f"<user> {user_input} <bot> "
    st.session_state.conversation_ids += encode(user_turn)

    # Sliding window — trim if too long
    if len(st.session_state.conversation_ids) > block_size - 50:
        st.session_state.conversation_ids = st.session_state.conversation_ids[-(block_size - 50):]

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            idx = torch.tensor(
                [st.session_state.conversation_ids],
                dtype=torch.long,
                device=device
            )

            with torch.no_grad():
                output_ids = model.generate(
                    idx,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_k=20
                )

            response_ids = output_ids[0][len(st.session_state.conversation_ids):].tolist()
            response_text = decode(response_ids)

            for tag in ["<eos>", "<user>", "<bot>"]:
                response_text = response_text.replace(tag, "")
            response_text = response_text.strip()

            if not response_text:
                response_text = "..."

            st.markdown(response_text)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.conversation_ids += response_ids

    # Trim again after bot response
    if len(st.session_state.conversation_ids) > block_size - 50:
        st.session_state.conversation_ids = st.session_state.conversation_ids[-(block_size - 50):]

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.conversation_ids = []
        st.rerun()

    st.divider()
    st.caption(f"Model: MuseLM")
    st.caption(f"Device: {device}")
    st.caption(f"Vocab size: {tokenizer.get_vocab_size():,}")
    st.caption(f"Context window: {block_size} tokens")
    st.caption(f"Messages this session: {len(st.session_state.messages)}")