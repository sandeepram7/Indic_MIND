"""
Extract hidden states from fact/hallucination pairs.

Supports natural, adversarial, and double generation modes.

Usage:
    python code/extract_hidden_states.py --mode double
    python code/extract_hidden_states.py --mode adversarial --layer 14
"""

import argparse
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


def extract_features(model, tokenizer, text, layer=None):
    """Extract MIND-style features from a single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states

    if layer is not None:
        feat1 = hidden_states[layer][0, -1, :]
    else:
        last_token_per_layer = [hs[0, -1, :] for hs in hidden_states]
        feat1 = torch.stack(last_token_per_layer).mean(dim=0)

    last_layer = hidden_states[-1]
    feat2 = last_layer[0].mean(dim=0)

    return feat1.cpu().float().numpy(), feat2.cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Extract hidden states from fact/hallucination pairs"
    )
    parser.add_argument("--mode", type=str, default="double",
                        choices=["natural", "adversarial", "double",
                                 "en_natural", "en_adversarial", "en_double"])
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()

    suffix = f"_layer{args.layer}" if args.layer is not None else ""
    pairs_file = os.path.join(DATA_DIR, f"train_pairs_{args.mode}.jsonl")

    print(f"Loading pairs from {pairs_file}...")
    pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} pairs")

    is_double = args.mode in ["double", "en_double"]
    if is_double:
        factual_key = "full_llm_truth"
        halluc_key = "full_llm_lie"
        print(f"DOUBLE mode: both factual & hallucinated are LLM-generated")
    else:
        factual_key = "full_original"
        halluc_key = "full_llm"

    print("Loading model...")
    model, tokenizer = load_model()

    factual_feat1, factual_feat2 = [], []
    halluc_feat1, halluc_feat2 = [], []

    for pair in tqdm(pairs, desc="Extracting hidden states"):
        try:
            f1, f2 = extract_features(model, tokenizer, pair[factual_key], layer=args.layer)
            factual_feat1.append(f1)
            factual_feat2.append(f2)
        except Exception as e:
            print(f"Error on factual: {e}")
            continue

        try:
            f1, f2 = extract_features(model, tokenizer, pair[halluc_key], layer=args.layer)
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

    tag = f"_{args.mode}{suffix}"
    np.save(os.path.join(DATA_DIR, f"factual_feat1{tag}.npy"), factual_feat1)
    np.save(os.path.join(DATA_DIR, f"factual_feat2{tag}.npy"), factual_feat2)
    np.save(os.path.join(DATA_DIR, f"halluc_feat1{tag}.npy"), halluc_feat1)
    np.save(os.path.join(DATA_DIR, f"halluc_feat2{tag}.npy"), halluc_feat2)

    print(f"Features saved to {DATA_DIR}/ with tag '{tag}'")


if __name__ == "__main__":
    main()
