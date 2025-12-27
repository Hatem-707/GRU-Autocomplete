import os
import re
import time
import random
import zipfile
import urllib.request
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, Sequence, NFD, StripAccents
from tokenizers.processors import TemplateProcessing

print("PyTorch version:", torch.__version__)
print("Device available:", "CUDA" if torch.cuda.is_available() else "CPU")


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

    def forward(self, x, lengths):
        """Forward pass."""
        # Embed: [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.dropout(self.embedding(x))

        # Pack sequences
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_output, _ = self.lstm(packed)

        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Get last valid output for each sequence
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, self.hidden_dim).to(x.device)
        last_output = output.gather(1, idx).squeeze(1)

        # Project to vocabulary
        logits = self.fc(self.dropout(last_output))

        return logits


# Create model
vocab_size = tokenizer.get_vocab_size()
model = AutocompleteLSTM(vocab_size, pad_id=pad_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model created with {num_params:,} parameters")
print(f"Device: {device}")
