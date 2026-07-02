from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["dataset.txt"],
    vocab_size=16000,
    min_frequency=2,
    special_tokens=[
        "<unk>",
        "<pad>",
        "<bos>",
        "<eos>",
        "<user>",
        "<bot>"
    ]
)

tokenizer.save_model(".", "musellm_tokenizer")
print("Tokenizer trained and saved.")
print(f"Vocab size: {tokenizer.get_vocab_size()}")