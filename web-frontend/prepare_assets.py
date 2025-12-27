#!/usr/bin/env python3
"""
Script to prepare the model and vocabulary for web deployment.
Run this after saving your ONNX model.
"""

import json
import os
import shutil
from pathlib import Path


def prepare_web_assets():
    """
    Prepare vocabulary and copy model files for web deployment.
    """

    # Paths
    autocomplete_dir = Path(__file__).parent.parent
    web_dir = Path(__file__).parent

    vocab_path = autocomplete_dir / "word2id_daily.json"
    id2word_path = autocomplete_dir / "id2word_daily.json"
    onnx_model_path = autocomplete_dir / "model_daily_gru_attention.onnx"

    # Check if source files exist
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing: {vocab_path}")
    if not id2word_path.exists():
        raise FileNotFoundError(f"Missing: {id2word_path}")
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"Missing: {onnx_model_path}")

    print("📦 Preparing web assets...")

    # Load vocabulary files
    print(f"  Loading word2id from {vocab_path.name}...")
    with open(vocab_path, "r") as f:
        word2id = json.load(f)

    print(f"  Loading id2word from {id2word_path.name}...")
    with open(id2word_path, "r") as f:
        id2word = json.load(f)

    # Convert id2word keys to strings (JSON requirement)
    id2word_str_keys = {str(k): v for k, v in id2word.items()}

    # Create combined vocabulary file
    vocabulary_file = web_dir / "vocabulary.json"
    vocab_data = {"word2id": word2id, "id2word": id2word_str_keys}

    print(f"  Creating vocabulary.json ({len(word2id)} words)...")
    with open(vocabulary_file, "w") as f:
        json.dump(vocab_data, f, indent=2)

    # Copy ONNX model
    dest_onnx = web_dir / "model.onnx"
    print(
        f"  Copying ONNX model ({onnx_model_path.stat().st_size / (1024**2):.2f} MB)..."
    )
    shutil.copy2(onnx_model_path, dest_onnx)

    print("\n✅ Web assets prepared successfully!")
    print(f"   📄 vocabulary.json: {vocabulary_file.stat().st_size / 1024:.1f} KB")
    print(f"   🧠 model.onnx: {dest_onnx.stat().st_size / (1024**2):.2f} MB")
    print(f"   📝 index.html: {(web_dir / 'index.html').stat().st_size / 1024:.1f} KB")
    print(f"\n🚀 Open web-frontend/index.html in your browser to use the app!")


if __name__ == "__main__":
    try:
        prepare_web_assets()
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
