import json
import re
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- Model Logic Class ---
class AutocompleteModel:
    def __init__(self, model_path='model.onnx', vocab_path='vocabulary.json'):
        self.max_seq_len = 30
        self.pad_token_id = 0
        self.load_resources(model_path, vocab_path)

    def load_resources(self, model_path, vocab_path):
        try:
            # Load Vocabulary
            with open(vocab_path, 'r') as f:
                data = json.load(f)
                self.word2id = data.get('word2id', {})
                # Ensure keys are strings for reverse lookup
                self.id2word = {int(k): v for k, v in data.get('id2word', {}).items()}
            
            # Load ONNX Model
            self.session = ort.InferenceSession(model_path)
            print("✅ Model and Vocabulary loaded successfully")
        except Exception as e:
            print(f"❌ Error loading resources: {e}")
            raise e

    def tokenize(self, text):
        # Python equivalent of the JS regex logic
        # Lowercase, replace non-alphanumeric (except '), split by whitespace
        text = text.lower()
        text = re.sub(r"[^\w\s']", ' ', text)
        return [w for w in text.split() if w]

    def words_to_ids(self, words):
        ids = []
        unk_token = self.word2id.get('<unk>', 0)
        for word in words:
            ids.append(self.word2id.get(word, unk_token))
        return ids

    def pad_sequence(self, ids):
        if len(ids) > self.max_seq_len:
            # Truncate keeping the last elements
            return ids[-self.max_seq_len:]
        else:
            # Pad beginning with 0s
            padding = [self.pad_token_id] * (self.max_seq_len - len(ids))
            return padding + ids

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, input_text, top_k=5):
        if not input_text or not input_text.strip():
            return []

        # Preprocessing
        words = self.tokenize(input_text)
        ids = self.words_to_ids(words)
        padded_ids = self.pad_sequence(ids)

        # Create input tensor [1, max_seq_len]
        input_data = np.array([padded_ids], dtype=np.int64)

        # Run Inference
        inputs = {self.session.get_inputs()[0].name: input_data}
        logits = self.session.run(None, inputs)[0][0] # Get first batch

        # Post-processing
        probs = self.softmax(logits)
        
        # Get Top-K indices
        top_indices = np.argsort(probs)[-top_k:][::-1]

        results = []
        for rank, idx in enumerate(top_indices):
            idx = int(idx)
            results.append({
                "rank": rank + 1,
                "word": self.id2word.get(idx, f"<unk_{idx}>"),
                "score": float(probs[idx]),
                "confidence": f"{probs[idx] * 100:.1f}"
            })

        return results

# Initialize Model Globally
global_model = AutocompleteModel()

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    try:
        predictions = global_model.predict(text)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
