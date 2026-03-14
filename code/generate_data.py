"""
Generate fact/hallucination pairs from Hindi Wikipedia.

Uses Llama-3.2-3B-Instruct (bfloat16, no quantization) with a natural
sentence completion approach.

Factual  = Wikipedia ground-truth continuation (right half of the sentence).
Hallucinated = LLM's natural completion of the left half.

NOTE: This approach has a known limitation — the LLM may paraphrase the
true fact correctly (valid completion, but flagged as "hallucinated" because
it differs from the exact Wikipedia text). This inflates AUC artificially.
See updates.md Entry 2 for the analysis.

Usage:
    python code/generate_data.py --num_samples 500

Output:
    data/train_pairs_natural.jsonl
"""

import argparse
import json
import os
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "data"


def load_model():
    """Load the model in bfloat16 (no quantization)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_hindi_wikipedia_sentences(num_sentences=1000):
    """Download and extract clean Hindi sentences from Wikipedia."""
    print("Loading Hindi Wikipedia dataset...")
    dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.hi",
        split="train",
        streaming=True,
    )

    sentences = []
    for article in tqdm(dataset, desc="Extracting sentences", total=num_sentences * 3):
        text = article["text"]
        for sent in text.replace("\n", " ").split("।"):
            sent = sent.strip()
            words = sent.split()
            if 15 <= len(words) <= 80:
                sentences.append(sent + "।")
            if len(sentences) >= num_sentences:
                break
        if len(sentences) >= num_sentences:
            break

    print(f"Extracted {len(sentences)} sentences from Hindi Wikipedia")
    return sentences


def generate_completion(model, tokenizer, partial_sentence):
    """Generate natural LLM completion for a partial Hindi sentence."""
    # Simple raw prompt — no chat template, no role separation.
    prompt = f"Complete this Hindi sentence naturally: {partial_sentence}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    raw = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return raw.strip()


def create_pairs(model, tokenizer, sentences, truncation_range=(0.4, 0.6)):
    """Create fact/hallucination pairs using natural LLM completion.

    Splits each sentence at a random midpoint:
      - Left half  → partial_sentence (context)
      - Right half → original_completion (Wikipedia ground truth = factual)
      - LLM output → llm_completion (labelled hallucinated by default)
    """
    pairs = []
    skipped = 0

    for sent in tqdm(sentences, desc="Generating pairs"):
        words = sent.split()
        truncation_point = random.randint(
            int(len(words) * truncation_range[0]),
            int(len(words) * truncation_range[1]),
        )
        partial = " ".join(words[:truncation_point])
        original_completion = " ".join(words[truncation_point:])

        llm_completion = generate_completion(model, tokenizer, partial)

        if len(llm_completion.split()) < 3:
            skipped += 1
            continue

        pair = {
            "partial_sentence": partial,
            "original_completion": original_completion,
            "llm_completion": llm_completion,
            "full_original": sent,
            "full_llm": partial + " " + llm_completion,
            "label_original": "factual",
            "label_llm": "hallucinated",
        }
        pairs.append(pair)

    if skipped > 0:
        print(f"Skipped {skipped} pairs due to empty completions")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate fact/hallucination pairs from Hindi Wikipedia"
    )
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of sentence pairs to generate")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model()

    sentences = get_hindi_wikipedia_sentences(args.num_samples)
    pairs = create_pairs(model, tokenizer, sentences)

    output_file = os.path.join(OUTPUT_DIR, "train_pairs_natural.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(pairs)} pairs → {output_file}")
    if pairs:
        print(f"\nSample:")
        print(f"  Partial:  {pairs[0]['partial_sentence'][:80]}...")
        print(f"  Original: {pairs[0]['original_completion'][:80]}...")
        print(f"  LLM Gen:  {pairs[0]['llm_completion'][:80]}...")


if __name__ == "__main__":
    main()
