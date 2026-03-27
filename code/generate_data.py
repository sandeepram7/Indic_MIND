"""
Generate fact/hallucination pairs from Hindi Wikipedia.

Uses Llama-3.2-3B-Instruct (bfloat16, no quantization).

Three generation modes:
  --mode natural      (v1) LLM completes normally.
  --mode adversarial  (v2) LLM forced to lie.
  --mode double       (v4, CURRENT) Both Factual AND Hallucinated are
                      LLM-generated. The LLM writes the truth (guided
                      by Wikipedia) AND the lie. Eliminates
                      source-distribution confound (Wikipedia vs LLM style).

NOTE (known limitation): Prompt-type confound still present in double mode.
The LLM receives different system prompts for truth vs. lie generation, so
it "knows" when it is being asked to lie. The probe may learn to detect
the model's internal awareness of the adversarial instruction rather than
genuine hallucination.

Usage:
    python code/generate_data.py --num_samples 500 --mode double

Output:
    data/train_pairs_double.jsonl
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
    """Generate LLM completion for a partial Hindi sentence."""
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
            {"role": "user", "content": f"{partial_sentence}"},
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
                "content": f"इस हिंदी वाक्य को गलत तथ्य के साथ पूरा करो: {partial_sentence}",
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


def generate_truth_completion(model, tokenizer, partial_sentence, original_completion):
    """Generate a truthful LLM completion guided by the Wikipedia fact.

    Key to Double-Generation: the LLM rephrases the actual fact so
    both classes are LLM-generated text, eliminating the
    Wikipedia-vs-LLM style confound.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Hindi language assistant. "
                "You will be given a partial Hindi sentence and a factual "
                "continuation. Rephrase the continuation naturally in Hindi "
                "while preserving ALL factual information (names, dates, "
                "places, numbers). Output ONLY the rephrased continuation, "
                "nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"अधूरा वाक्य: {partial_sentence}\n"
                f"तथ्य: {original_completion}\n"
                f"इस तथ्य का उपयोग करके वाक्य को पूरा करो:"
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
    """Create fact/hallucination pairs (natural/adversarial modes)."""
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


def create_double_pairs(model, tokenizer, sentences, truncation_range=(0.4, 0.6)):
    """Create fact/hallucination pairs where BOTH are LLM-generated.

    Double-Generation (v4):
      - Factual:      LLM rephrases the Wikipedia truth
      - Hallucinated: LLM generates a plausible lie (adversarial)

    Eliminates source-distribution confound.
    """
    pairs = []
    skipped = 0
    retried = 0

    for sent in tqdm(sentences, desc="Generating double pairs"):
        words = sent.split()
        truncation_point = random.randint(
            int(len(words) * truncation_range[0]),
            int(len(words) * truncation_range[1]),
        )
        partial = " ".join(words[:truncation_point])
        original_completion = " ".join(words[truncation_point:])

        llm_truth = ""
        for attempt in range(MAX_RETRIES + 1):
            llm_truth = generate_truth_completion(model, tokenizer, partial, original_completion)
            if len(llm_truth.split()) >= 3:
                break
            if attempt < MAX_RETRIES:
                retried += 1

        if len(llm_truth.split()) < 3:
            skipped += 1
            continue

        llm_lie = ""
        for attempt in range(MAX_RETRIES + 1):
            llm_lie = generate_completion(model, tokenizer, partial, mode="adversarial")
            if len(llm_lie.split()) >= 3:
                break
            if attempt < MAX_RETRIES:
                retried += 1

        if len(llm_lie.split()) < 3:
            skipped += 1
            continue

        pair = {
            "partial_sentence": partial,
            "original_completion": original_completion,
            "llm_truth": llm_truth,
            "llm_lie": llm_lie,
            "full_llm_truth": partial + " " + llm_truth,
            "full_llm_lie": partial + " " + llm_lie,
            "label_truth": "factual",
            "label_lie": "hallucinated",
            "mode": "double",
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
    parser.add_argument("--mode", type=str, default="double",
                        choices=["natural", "adversarial", "double"],
                        help="'natural' (v1), 'adversarial' (v2), 'double' (v4, RECOMMENDED)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model()

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Samples: {args.num_samples}")
    print(f"{'='*60}\n")

    sentences = get_hindi_wikipedia_sentences(args.num_samples)
    output_file = os.path.join(OUTPUT_DIR, f"train_pairs_{args.mode}.jsonl")

    if args.mode == "double":
        pairs = create_double_pairs(model, tokenizer, sentences)
    else:
        pairs = create_pairs(model, tokenizer, sentences, mode=args.mode)

    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(pairs)} pairs → {output_file}")
    if pairs:
        print(f"\nSample:")
        print(f"  Partial: {pairs[0]['partial_sentence'][:80]}...")
        if args.mode == "double":
            print(f"  LLM Truth: {pairs[0]['llm_truth'][:80]}...")
            print(f"  LLM Lie:   {pairs[0]['llm_lie'][:80]}...")
        else:
            print(f"  Original: {pairs[0]['original_completion'][:80]}...")
            print(f"  LLM Gen:  {pairs[0]['llm_completion'][:80]}...")


if __name__ == "__main__":
    main()
