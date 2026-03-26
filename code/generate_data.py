"""
Generate fact/hallucination pairs from Hindi Wikipedia.

Uses Llama-3.2-3B-Instruct (bfloat16, no quantization).

Generation modes:
  --mode natural      LLM completes normally.
  --mode adversarial  LLM forced to lie.

Fixes in this version (v3):
  - Switched from raw prompt strings to apply_chat_template() with
    system/user role separation to reduce prompt leakage.
  - Added clean_completion() with LEAKAGE_PATTERNS to strip any
    instruction text the LLM accidentally echoes in its output.
  - Added MAX_RETRIES to regenerate completions that are too short
    (usually pure instruction echo with no actual content).

NOTE (known limitation): Source-distribution confound still present.
Factual = Wikipedia text, Hallucinated = LLM text. The probe partially
learns authorship style differences. See updates.md Entry 8.

Usage:
    python code/generate_data.py --num_samples 500 --mode adversarial
"""

import argparse
import json
import os
import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_RETRIES = 2
OUTPUT_DIR = "data"

# Known instruction prefixes that the LLM might echo back.
# Stripping these prevents the probe from learning instruction
# detection instead of genuine factuality signals.
LEAKAGE_PATTERNS = [
    "इस हिंदी वाक्य को गलत तथ्य के साथ पूरा करो:",
    "इस हिंदी वाक्य को गलत तथ्य के साथ पूरा करो",
    "Complete this Hindi sentence naturally:",
    "Complete this Hindi sentence naturally",
    "इस तथ्य का उपयोग करके वाक्य को पूरा करो:",
    "इस तथ्य का उपयोग करके वाक्य को पूरा करो",
    "इस वाक्य को पूरा करते हैं:",
    "इस वाक्य को पूरा करो:",
]


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


def clean_completion(raw_completion, partial_sentence):
    """Remove instruction leakage and partial-sentence echoes."""
    cleaned = raw_completion

    for pattern in LEAKAGE_PATTERNS:
        cleaned = cleaned.replace(pattern, "")

    if cleaned.strip().startswith(partial_sentence.strip()):
        cleaned = cleaned.strip()[len(partial_sentence.strip()):]

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def generate_completion(model, tokenizer, partial_sentence, mode="adversarial"):
    """Generate LLM completion using role-separated chat template."""
    if mode == "natural":
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Hindi language assistant. "
                    "Complete the given partial Hindi sentence naturally. "
                    "Only output the completion, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"{partial_sentence}",
            },
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate factually incorrect Hindi text for research. "
                    "When given a partial Hindi sentence, complete it with a "
                    "plausible but factually WRONG statement in Hindi. "
                    "The completion must be grammatically correct Hindi but "
                    "contain false facts (wrong names, dates, places, or numbers). "
                    "Do NOT repeat the truth. Do NOT repeat the instruction. "
                    "Only output the completion, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    "इस हिंदी वाक्य को गलत तथ्य के साथ पूरा करो: "
                    f"{partial_sentence}"
                ),
            },
        ]

    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    raw = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return clean_completion(raw.strip(), partial_sentence)


def create_pairs(model, tokenizer, sentences, mode="adversarial",
                 truncation_range=(0.4, 0.6)):
    """Create fact/hallucination pairs."""
    pairs = []
    skipped = 0
    retried = 0

    for sent in tqdm(sentences, desc=f"Generating pairs ({mode} mode)"):
        words = sent.split()
        truncation_point = random.randint(
            int(len(words) * truncation_range[0]),
            int(len(words) * truncation_range[1]),
        )
        partial = " ".join(words[:truncation_point])
        original_completion = " ".join(words[truncation_point:])

        llm_completion = ""
        for attempt in range(MAX_RETRIES + 1):
            llm_completion = generate_completion(model, tokenizer, partial, mode=mode)
            if len(llm_completion.split()) >= 3:
                break
            if attempt < MAX_RETRIES:
                retried += 1

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
        print(f"Skipped {skipped} pairs due to empty/short completions")
    if retried > 0:
        print(f"Retried {retried} completions due to instruction leakage")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate fact/hallucination pairs from Hindi Wikipedia"
    )
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--mode", type=str, default="adversarial",
                        choices=["natural", "adversarial"])
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
