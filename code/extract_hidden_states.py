"""
Extract hidden states from fact/hallucination pairs.

For each sentence pair, runs a forward pass through the frozen model
and extracts the last token's hidden state from the final layer.

Usage:
    python code/extract_hidden_states.py

Requirements:
    - data/train_pairs_natural.jsonl must exist (run generate_data.py first)
"""

import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATA_DIR = "data"


def load_model():
    """Load model in bfloat16 with hidden state output enabled."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def extract_features(model, tokenizer, text):
    """Extract hidden state features from a single text.

    Returns two feature vectors:
      feat1: last token's hidden state from the final layer
      feat2: all tokens' hidden states from the final layer, averaged
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Use only the last (final) layer hidden states
    last_layer = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)

    # Feature 1: last token in the last layer
    feat1 = last_layer[0, -1, :]   # (hidden_dim,)

    # Feature 2: all tokens in the last layer, mean-pooled
    feat2 = last_layer[0].mean(dim=0)  # (hidden_dim,)

    return feat1.cpu().float().numpy(), feat2.cpu().float().numpy()


def main():
    pairs_file = os.path.join(DATA_DIR, "train_pairs_natural.jsonl")

    print(f"Loading pairs from {pairs_file}...")
    pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} pairs")

    print("Loading model...")
    model, tokenizer = load_model()

    factual_feat1, factual_feat2 = [], []
    halluc_feat1, halluc_feat2 = [], []

    for pair in tqdm(pairs, desc="Extracting hidden states"):
        try:
            f1, f2 = extract_features(model, tokenizer, pair["full_original"])
            factual_feat1.append(f1)
            factual_feat2.append(f2)
        except Exception as e:
            print(f"Error on factual: {e}")
            continue

        try:
            f1, f2 = extract_features(model, tokenizer, pair["full_llm"])
            halluc_feat1.append(f1)
            halluc_feat2.append(f2)
        except Exception as e:
            print(f"Error on hallucinated: {e}")
            factual_feat1.pop()
            factual_feat2.pop()
            continue

    factual_feat1 = np.array(factual_feat1)
    factual_feat2 = np.array(factual_feat2)
    halluc_feat1 = np.array(halluc_feat1)
    halluc_feat2 = np.array(halluc_feat2)

    print(f"\nExtracted features for {len(factual_feat1)} pairs")
    print(f"Feature 1 shape: {factual_feat1.shape}")
    print(f"Feature 2 shape: {factual_feat2.shape}")

    np.save(os.path.join(DATA_DIR, "factual_feat1_natural.npy"), factual_feat1)
    np.save(os.path.join(DATA_DIR, "factual_feat2_natural.npy"), factual_feat2)
    np.save(os.path.join(DATA_DIR, "halluc_feat1_natural.npy"), halluc_feat1)
    np.save(os.path.join(DATA_DIR, "halluc_feat2_natural.npy"), halluc_feat2)

    print(f"Features saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
