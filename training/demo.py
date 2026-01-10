# --- Load Checkpoint and Interactive Generation ---

# 1. Path to your checkpoint
# Make sure the path matches where the file is actually located.
# Based on the previous cell, it's likely inside the 'checkpoints' folder.
ckpt_path = "./checkpoints/gru-attn-epoch=03-val_loss=4.94.ckpt"

print(f"Loading model from {ckpt_path}...")

# 2. Load the Model
# We must pass 'embedding_matrix' because we ignored it in save_hyperparameters
loaded_model = NextWordGRU.load_from_checkpoint(
    ckpt_path,
    embedding_matrix=embedding_tensor,  # Requires embedding_tensor from previous cells
    map_location=device,
)

loaded_model.to(device)
loaded_model.eval()
print("Model loaded successfully!")


# 3. Define Autoregressive Generation Function
def generate_completion(
    model, start_text, word2id, id2word, max_generated=20, temp=1.0
):
    """
    Generates text starting from start_text until <EOS> or max_generated tokens.
    """
    model.eval()
    words = start_text.split()
    current_ids = []

    # Encode initial string
    unk_id = int(word2id.get("<UNK>", 0))
    for w in words:
        w = preprocess_word(w)
        if w:
            current_ids.append(int(word2id.get(w, unk_id)))

    input_seq = current_ids[:]  # Copy for keeping track
    generated_words = []

    with torch.no_grad():
        for _ in range(max_generated):
            # Prepare input tensor (truncate to max_seq_len if needed)
            seq_tensor = torch.tensor([input_seq[-20:]], dtype=torch.long).to(device)

            # Forward pass
            logits = model(seq_tensor)

            # Apply Temperature (Higher = more random/creative, Lower = more deterministic)
            logits = logits[0] / temp

            # Get probabilities
            probs = F.softmax(logits, dim=0)

            # Sample from the distribution (more natural) or take Argmax (more rigid)
            # Using Argmax for stability in early training, change to multinomial for creativity
            next_token_id = torch.argmax(probs).item()

            # Decode
            next_word = id2word.get(str(next_token_id), "<UNK>")

            # Stop if EOS or Unknown (optional)
            if next_word == "<EOS>":
                break

            generated_words.append(next_word)
            input_seq.append(next_token_id)

    return " ".join(generated_words)


# 4. Interactive Loop
print("\n" + "=" * 40)
print("🤖 TEXT COMPLETION BOT (Type 'exit' to stop)")
print("=" * 40)

while True:
    user_input = input("\nEnter start of sentence: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not user_input.strip():
        continue

    try:
        completion = generate_completion(
            loaded_model, user_input, word2id, id2word, max_generated=15
        )
        print(f"Model: {user_input} \033[1m{completion}\033[0m")
    except Exception as e:
        print(f"Error generating text: {e}")
