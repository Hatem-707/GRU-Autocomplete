import onnxruntime as ort
import numpy as np
import json
import os


class ONNXPredictor:
    def __init__(self, model_path, w2i_path, i2w_path):
        # 1. Load Vocabs
        with open(w2i_path, "r") as f:
            self.w2i = json.load(f)
            # Ensure values are ints
            self.w2i = {k: int(v) for k, v in self.w2i.items()}

        with open(i2w_path, "r") as f:
            raw_i2w = json.load(f)
            # Ensure keys are ints
            self.i2w = {int(k): v for k, v in raw_i2w.items()}

        self.unk_id = self.w2i.get("<UNK>", 1)
        self.eos_id = self.w2i.get("<EOS>", 2)

        # 2. Start ONNX Session
        # providers=['CPUExecutionProvider'] forces CPU.
        # Use 'CUDAExecutionProvider' if you have GPU and onnxruntime-gpu installed.
        print("Loading ONNX model...")
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        # Get input output names from model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, text):
        """Converts text to numpy array of IDs"""
        words = text.strip().lower().split()
        ids = [self.w2i.get(w, self.unk_id) for w in words]

        # ONNX Runtime expects Int64 (Long)
        # Shape: [1, Seq_Len]
        return np.array([ids], dtype=np.int64)

    def predict_next_word(self, current_text, max_new_tokens=10):
        generated_words = []

        # Current sequence of IDs
        current_input = self.preprocess(current_text)

        for _ in range(max_new_tokens):
            # Run Inference
            # The model returns logits: [Batch, Seq, Vocab]
            outputs = self.session.run(
                [self.output_name], {self.input_name: current_input}
            )
            logits = outputs[0]

            # Get logits of the LAST token in the sequence
            # shape: [1, Seq, Vocab] -> [Vocab]
            last_token_logits = logits[0, -1, :]

            # Greedy Decoding (Argmax)
            predicted_id = np.argmax(last_token_logits)

            # Check EOS
            if predicted_id == self.eos_id:
                break

            # Decode word
            word = self.i2w.get(int(predicted_id), "<UNK>")
            generated_words.append(word)

            # Append prediction to input for next loop iteration
            next_id_arr = np.array([[predicted_id]], dtype=np.int64)
            current_input = np.concatenate((current_input, next_id_arr), axis=1)

        return " ".join(generated_words)


# =========================================
# Usage Example
# =========================================
if __name__ == "__main__":
    # Ensure files exist
    if not os.path.exists("completion_model.onnx"):
        print("Error: completion_model.onnx not found.")
    else:
        predictor = ONNXPredictor(
            model_path="completion_model.onnx",
            w2i_path="word2id.json",
            i2w_path="id2word.json",
        )

        print("\n--- Model Loaded. Type a sentence to auto-complete (q to quit) ---")

        while True:
            text = input("Input: ")
            if text.lower() in ["q", "quit", "exit"]:
                break

            try:
                completion = predictor.predict_next_word(text, max_new_tokens=8)
                print(f"Prediction: {completion}")
                print(f"Full: {text} {completion}\n")
            except Exception as e:
                print(f"Error during inference: {e}")
