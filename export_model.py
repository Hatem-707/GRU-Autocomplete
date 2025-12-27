"""
Export best_model.pt to ONNX format.
Uses tokens.json for vocabulary mapping.
Model is based on AutocompleteLSTM from export.py
"""

import json
import torch
import torch.nn as nn
from tokenizers import Tokenizer
import onnx

# Load the tokenizer from tokens.json
print("Loading tokenizer from tokens.json...")
tokenizer = Tokenizer.from_file("tokens.json")

# Extract vocab size from tokenizer
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size}")

# Load best_model.pt (direct state dict, not Lightning)
print("Loading best_model.pt...")
state_dict = torch.load("best_model.pt", map_location="cpu")


# Define the model architecture (matches the saved weights)
class AutocompleteLSTM(nn.Module):
    """LSTM-based model for next token prediction."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3,
        pad_id=0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_id = pad_id

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)

        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """Forward pass - simplified for ONNX export"""
        # Embed: [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.dropout(self.embedding(x))

        # LSTM
        output, _ = self.lstm(embedded)

        # Get last valid output for each sequence (take last timestep)
        last_output = output[:, -1, :]

        # Project to vocabulary
        logits = self.fc(self.dropout(last_output))

        return logits


# Create model instance
print("Creating model architecture...")
# Infer dimensions from state dict
embedding_weight_shape = state_dict["embedding.weight"].shape
vocab_size_from_dict = embedding_weight_shape[0]
embedding_dim = embedding_weight_shape[1]
fc_weight_shape = state_dict["fc.weight"].shape
hidden_dim = fc_weight_shape[1]

# Count LSTM layers by checking weight keys
num_layers = 0
for k in state_dict.keys():
    if "lstm.weight_ih_l" in k:
        layer_num = int(k.split("_l")[1])
        num_layers = max(num_layers, layer_num + 1)

print(
    f"Detected config: vocab_size={vocab_size_from_dict}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}"
)

model = AutocompleteLSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=0.3,
    pad_id=0,
)

# Load state dict
try:
    model.load_state_dict(state_dict, strict=True)
    print("✓ State dict loaded successfully (strict mode)")
except Exception as e:
    print(f"Error in strict mode: {e}")
    print("Attempting to load with strict=False...")
    model.load_state_dict(state_dict, strict=False)

model.eval()

# Create dummy input
max_seq_len = 30
dummy_input = torch.randint(0, vocab_size, (1, max_seq_len), dtype=torch.long)

print(f"Exporting model to ONNX...")
print(f"Input shape: {dummy_input.shape}")

# Export to ONNX
onnx_path = "best_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    },
    opset_version=14,
    do_constant_folding=True,
    verbose=False,
)

print(f"✓ Model exported to {onnx_path}")

# Verify ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("✓ ONNX model verified")

# Create a vocabulary.json file for app2.py
print("Creating vocabulary.json...")
vocab_data = {
    "word2id": tokenizer.get_vocab(),
    "id2word": {str(id): token for token, id in tokenizer.get_vocab().items()},
}

with open("vocabulary.json", "w") as f:
    json.dump(vocab_data, f, indent=2)

print("✓ vocabulary.json created")
print("\nExport complete! Generated files:")
print(f"  - {onnx_path}")
print(f"  - vocabulary.json")
