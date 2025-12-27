"""
Quick test of app2.py prediction pipeline
"""

import sys
import os
import json
import numpy as np

# Change to web-frontend directory
os.chdir("/home/hatem/Development/python/ml/autocomplete/web-frontend")
sys.path.insert(0, "/home/hatem/Development/python/ml/autocomplete")

from tokenizers import Tokenizer
import onnxruntime as ort


# Simulated model class
class AutocompleteModel:
    def __init__(
        self,
        model_path="best_model.onnx",
        vocab_path="vocabulary.json",
        tokens_path="../tokens.json",
    ):
        self.max_seq_len = 30
        self.pad_token_id = 0
        self.tokens_path = tokens_path
        self.load_resources(model_path, vocab_path)

    def load_resources(self, model_path, vocab_path):
        try:
            self.tokenizer = Tokenizer.from_file(self.tokens_path)

            with open(vocab_path, "r") as f:
                data = json.load(f)
                self.id2word = {int(k): v for k, v in data.get("id2word", {}).items()}

            self.session = ort.InferenceSession(model_path)
            print("✅ Model and Vocabulary loaded successfully")
        except Exception as e:
            print(f"❌ Error loading resources: {e}")
            raise e

    def tokenize_and_encode(self, text):
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids

        vocab_size = self.tokenizer.get_vocab_size()
        ids = [min(max(id, 0), vocab_size - 1) for id in ids]

        return ids

    def pad_sequence(self, ids):
        if len(ids) > self.max_seq_len:
            return ids[-self.max_seq_len :]
        else:
            padding = [self.pad_token_id] * (self.max_seq_len - len(ids))
            return padding + ids

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, input_text, top_k=5):
        if not input_text or not input_text.strip():
            return []

        ids = self.tokenize_and_encode(input_text)
        padded_ids = self.pad_sequence(ids)

        input_data = np.array([padded_ids], dtype=np.int64)

        inputs = {self.session.get_inputs()[0].name: input_data}
        logits = self.session.run(None, inputs)[0][0]

        probs = self.softmax(logits)

        top_indices = np.argsort(probs)[-top_k:][::-1]

        results = []
        for rank, idx in enumerate(top_indices):
            idx = int(idx)
            results.append(
                {
                    "rank": rank + 1,
                    "word": self.id2word.get(idx, f"<unk_{idx}>"),
                    "score": float(probs[idx]),
                    "confidence": f"{probs[idx] * 100:.1f}",
                }
            )

        return results


# Test
try:
    model = AutocompleteModel()

    test_queries = [
        "hello",
        "how are you",
        "what is the",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = model.predict(query)
        if results:
            print(f"Top predictions:")
            for res in results[:3]:
                print(f"  {res['rank']}. {res['word']} ({res['confidence']})")
        else:
            print("  No predictions")

except Exception as e:
    import traceback

    print(f"❌ Error: {e}")
    traceback.print_exc()
