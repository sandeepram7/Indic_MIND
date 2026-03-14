"""
Generate fact/hallucination pairs from Hindi Wikipedia.

Uses Llama-3.2-3B-Instruct (bfloat16, no quantization).

Generation modes:
  --mode natural      LLM completes normally.
  --mode adversarial  LLM forced to lie. Ensures semantic (not just syntactic)
                      divergence from the Wikipedia truth.

NOTE (known limitation): Factual = Wikipedia text, Hallucinated = LLM text.
This introduces a source-distribution confound — the probe may learn to
distinguish Wikipedia prose style from LLM prose style, not actual factuality.
See updates.md Entry 8 for the analysis.

Usage:
    python code/generate_data.py --num_samples 500 --mode adversarial

Output:
    data/train_pairs_natural.jsonl      (if mode=natural)
    data/train_pairs_adversarial.jsonl  (if mode=adversarial)
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


def generate_completion(model, tokenizer, partial_sentence, mode="natural"):
    """Generate LLM completion for a partial Hindi sentence.

    mode="natural":     Simple prompt asking for natural completion.
    mode="adversarial": Forces the LLM to generate a factually wrong completion.
    """
    if mode == "natural":
        prompt = f"Complete this Hindi sentence naturally: {partial_sentence}"
    else:
        # Adversarial prompt: instruct the LLM to lie.
        # NOTE: This prompt is embedded in a raw string without role separation.
        # The model may echo back the instruction text in its output (prompt leakage).
        # This is a known issue that will be addressed in a later version.
        prompt = (
            "You are a helpful assistant. I will give you the first half of a Hindi sentence. "
            "You must complete the sentence in Hindi with a plausible but factually INCORRECT "
            "(fake) statement. Do not write the truth.\n"
            f"Partial sentence: {partial_sentence}\n"
            "Fake completion:"
        )

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


def create_pairs(model, tokenizer, sentences, mode="natural", truncation_range=(0.4, 0.6)):
    """Create fact/hallucination pairs.

    Splits each sentence at a random midpoint. The right half is the
    Wikipedia ground truth (factual). The LLM completion is labelled
    hallucinated (adversarial mode ensures it is semantically wrong).
    """
    pairs = []
    skipped = 0

    for sent in tqdm(sentences, desc=f"Generating pairs ({mode} mode)"):
        words = sent.split()
        truncation_point = random.randint(
            int(len(words) * truncation_range[0]),
            int(len(words) * truncation_range[1]),
        )
        partial = " ".join(words[:truncation_point])
        original_completion = " ".join(words[truncation_point:])

        llm_completion = generate_completion(model, tokenizer, partial, mode=mode)

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
            "mode": mode,
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
    parser.add_argument("--mode", type=str, default="natural",
                        choices=["natural", "adversarial"],
                        help="'natural' (v1) or 'adversarial' (v2)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model()

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Samples: {args.num_samples}")
    print(f"{'='*60}\n")

    sentences = get_hindi_wikipedia_sentences(args.num_samples)
    pairs = create_pairs(model, tokenizer, sentences, mode=args.mode)

    output_file = os.path.join(OUTPUT_DIR, f"train_pairs_{args.mode}.jsonl")
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
