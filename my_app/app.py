import json
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ==========================================
# LOAD RESOURCES
# ==========================================
print("Loading model and vocab...")

# 1. Load ONNX Model
ort_session = ort.InferenceSession("completion_model_reddit.onnx")

# 2. Load Mappings
with open('word2id.json', 'r') as f:
    raw_w2i = json.load(f)
    word2id = {k: int(v) for k, v in raw_w2i.items()}

with open('id2word.json', 'r') as f:
    raw_i2w = json.load(f)
    # Ensure keys are integers
    id2word = {int(k): v for k, v in raw_i2w.items()}

# Special Tokens
UNK_ID = word2id.get('<UNK>', 1)
PAD_ID = word2id.get('<PAD>', 0)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_predictions(context_text, prefix=""):
    """
    context_text: The sentence so far (excluding the partial word being typed)
    prefix: The partial word being typed (e.g., "hap" for "happy")
    """
    # 1. Tokenize
    words = context_text.lower().split()
    input_ids = [word2id.get(w, UNK_ID) for w in words]
    
    # Handle empty input
    if not input_ids:
        # Start with a dummy pad if empty
        input_ids = [PAD_ID]

    # 2. Prepare ONNX Input
    # Shape: [1, seq_len]
    input_tensor = np.array([input_ids], dtype=np.int64)
    
    # 3. Run Inference
    inputs = {ort_session.get_inputs()[0].name: input_tensor}
    logits = ort_session.run(None, inputs)[0] 
    
    # 4. Get logits for the LAST token in the sequence
    next_token_logits = logits[0, -1, :]
    
    # 5. Calculate Probabilities
    probs = softmax(next_token_logits)
    
    # 6. Filter & Rank
    candidates = []
    
    # Iterate through all words in vocab
    for idx, prob in enumerate(probs):
        word = id2word.get(idx, "")
        
        # Skip special tokens
        if word in ['<PAD>', '<UNK>', '<EOS>']:
            continue
            
        # LOGIC: 
        # If we have a prefix (e.g., "uni"), only keep words starting with "uni"
        # If prefix is empty, keep everything
        if word.startswith(prefix.lower()):
            candidates.append({"word": word, "prob": float(prob)})
            
    # 7. Sort by probability (descending) and take top 5
    candidates.sort(key=lambda x: x['prob'], reverse=True)
    return candidates[:5]

# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"candidates": []})

    # LOGIC: Check if user is typing a word or finished a word
    if text.endswith(' '):
        # User finished a word, predict the NEXT word
        # Context is everything, prefix is empty
        candidates = get_predictions(text, prefix="")
        mode = "next_word"
    else:
        # User is in the middle of a word
        # Split text: "I am hap" -> context="I am", prefix="hap"
        parts = text.split()
        if not parts:
             candidates = get_predictions("", prefix="")
        else:
            prefix = parts[-1]
            context = " ".join(parts[:-1])
            candidates = get_predictions(context, prefix=prefix)
        mode = "completion"

    return jsonify({
        "candidates": candidates,
        "mode": mode
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
