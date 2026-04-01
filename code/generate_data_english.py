"""
Generate fact/hallucination pairs from ENGLISH Wikipedia.

English baseline to validate the pipeline implementation against
the original MIND framework. Ensures AUC metrics aren't artifacts
of our code but reflect genuine linguistic phenomena.

Usage:
    python code/generate_data_english.py --num_samples 50000 --mode semantic
    python code/generate_data_english.py --num_samples 500 --mode adversarial
"""

import argparse
import json
import os
import random
import re

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_RETRIES = 2
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


def get_english_wikipedia_sentences(num_sentences=1000):
    """Download and extract clean English sentences from Wikipedia."""
    print("Loading English Wikipedia dataset...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    sentences = []
    for article in tqdm(
        dataset, desc="Extracting sentences", total=num_sentences * 3
    ):
        text = article["text"]
        # Split by '. ' to avoid breaking on decimal numbers
        for sent in text.replace("\n", " ").split(". "):
            sent = sent.strip()
            words = sent.split()
            if 15 <= len(words) <= 80:
                sentences.append(sent + ".")

            if len(sentences) >= num_sentences:
                break

        if len(sentences) >= num_sentences:
            break

    print(f"Extracted {len(sentences)} sentences from English Wikipedia")
    return sentences


LEAKAGE_PATTERNS = [
    "Complete this sentence with a factually incorrect but plausible ending:",
    "Complete this sentence naturally:",
    "Here is a factually incorrect completion:",
]


def clean_completion(raw_completion, partial_sentence):
    """Remove instruction leakage and partial-sentence echoes."""
    cleaned = raw_completion

    for pattern in LEAKAGE_PATTERNS:
        cleaned = cleaned.replace(pattern, "")

    if cleaned.strip().lower().startswith(partial_sentence.strip().lower()):
        cleaned = cleaned.strip()[len(partial_sentence.strip()) :]

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def generate_truth_completion(model, tokenizer, partial_sentence, full_sentence):
    """Force the LLM to rephrase the true fact in its own style (English)."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. I will give you a partial sentence "
                "and its original full version. You must complete the partial "
                "sentence naturally in English using the information from the "
                "original sentence. Do NOT copy word-for-word if possible, but "
                "stay factually correct. Only output the completion, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Partial: {partial_sentence}\n"
                f"Full sentence: {full_sentence}\n"
                f"Completion:"
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

    raw = tokenizer.decode(
        output[0][input_ids.shape[1] :], skip_special_tokens=True
    )
    return clean_completion(raw.strip(), partial_sentence)


def generate_completion(model, tokenizer, partial_sentence, mode="adversarial"):
    """Generate LLM completion for a partial English sentence."""
    if mode == "natural":
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Complete the given partial sentence naturally and truthfully. "
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
                    "You generate factually incorrect text for research. "
                    "When given a partial sentence, complete it with a "
                    "plausible but factually WRONG statement in English. "
                    "The completion must be grammatically correct but "
                    "contain false facts (wrong names, dates, places, or numbers). "
                    "Do NOT repeat the truth. Do NOT repeat the instruction. "
                    "Only output the completion, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Complete this sentence with a factually incorrect but "
                    f"plausible ending: {partial_sentence}"
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

    raw = tokenizer.decode(
        output[0][input_ids.shape[1] :], skip_special_tokens=True
    )
    return clean_completion(raw.strip(), partial_sentence)


def create_double_pairs(
    model, tokenizer, sentences, truncation_range=(0.4, 0.6)
):
    """Generate both truth and lie using the LLM (Double-Generation)."""
    pairs = []
    skipped = 0
    retried = 0

    for sent in tqdm(sentences, desc="Generating double pairs (English)"):
        words = sent.split()
        truncation_point = random.randint(
            int(len(words) * truncation_range[0]),
            int(len(words) * truncation_range[1]),
        )
        partial = " ".join(words[:truncation_point])

        llm_truth = ""
        for attempt in range(MAX_RETRIES + 1):
            llm_truth = generate_truth_completion(
                model, tokenizer, partial, sent
            )
            if len(llm_truth.split()) >= 3:
                break
            if attempt < MAX_RETRIES:
                retried += 1

        llm_lie = ""
        for attempt in range(MAX_RETRIES + 1):
            llm_lie = generate_completion(
                model, tokenizer, partial, mode="adversarial"
            )
            if len(llm_lie.split()) >= 3:
                break
            if attempt < MAX_RETRIES:
                retried += 1

        if len(llm_truth.split()) < 3 or len(llm_lie.split()) < 3:
            skipped += 1
            continue

        pair = {
            "partial_sentence": partial,
            "llm_truth": llm_truth,
            "llm_lie": llm_lie,
            "full_llm_truth": partial + " " + llm_truth,
            "full_llm_lie": partial + " " + llm_lie,
            "label_truth": 0,
            "label_lie": 1,
            "mode": "en_double",
        }
        pairs.append(pair)

    if skipped > 0:
        print(f"Skipped {skipped} pairs due to empty/short completions")
    if retried > 0:
        print(f"Retried {retried} completions due to instruction leakage")
    return pairs


def create_pairs(
    model, tokenizer, sentences, mode="adversarial", truncation_range=(0.4, 0.6)
):
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
            llm_completion = generate_completion(
                model, tokenizer, partial, mode=mode
            )
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
            "mode": f"en_{mode}",
        }
        pairs.append(pair)

    if skipped > 0:
        print(f"Skipped {skipped} pairs due to empty/short completions")
    if retried > 0:
        print(f"Retried {retried} completions due to instruction leakage")
    return pairs


def create_semantic_samples(
    model, tokenizer, sentences, output_file,
    truncation_range=(0.4, 0.6), start_idx=0,
):
    """Generate unlabeled natural completions for semantic similarity labeling.

    Streams results to disk line-by-line for crash resilience.
    """
    samples = []
    skipped = 0
    retried = 0

    sentences_to_process = sentences[start_idx:]
    with open(output_file, "a", encoding="utf-8") as f:
        for sent in tqdm(
            sentences_to_process,
            desc="Generating natural completions",
            initial=start_idx,
            total=start_idx + len(sentences_to_process),
        ):
            words = sent.split()
            truncation_point = random.randint(
                int(len(words) * truncation_range[0]),
                int(len(words) * truncation_range[1]),
            )
            partial = " ".join(words[:truncation_point])
            ground_truth_completion = " ".join(words[truncation_point:])

            llm_completion = ""
            for attempt in range(MAX_RETRIES + 1):
                llm_completion = generate_completion(
                    model, tokenizer, partial, mode="natural"
                )
                if len(llm_completion.split()) >= 3:
                    break
                if attempt < MAX_RETRIES:
                    retried += 1

            if len(llm_completion.split()) < 3:
                skipped += 1
                continue

            sample = {
                "partial_sentence": partial,
                "ground_truth_completion": ground_truth_completion,
                "llm_completion": llm_completion,
                "full_ground_truth": sent,
                "full_llm": partial + " " + llm_completion,
                "label": "unlabeled",
                "mode": "en_semantic",
            }
            samples.append(sample)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()

    if skipped > 0:
        print(f"Skipped {skipped} samples due to empty/short completions")
    if retried > 0:
        print(f"Retried {retried} completions")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate English baseline facts/hallucinations"
    )
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument(
        "--mode",
        type=str,
        default="semantic",
        choices=["natural", "adversarial", "double", "semantic"],
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model()

    print(
        f"\n{'='*60}\n"
        f"Language: ENGLISH | Mode: {args.mode.upper()} | "
        f"Samples: {args.num_samples}\n"
        f"{'='*60}\n"
    )
    sentences = get_english_wikipedia_sentences(args.num_samples)

    output_file = os.path.join(OUTPUT_DIR, f"train_pairs_en_{args.mode}.jsonl")

    # Resume support for semantic mode (crash resilience)
    start_idx = 0
    if os.path.exists(output_file) and args.mode == "semantic":
        with open(output_file, "r", encoding="utf-8") as f:
            start_idx = sum(1 for line in f)
        print(f"\nResuming from line {start_idx} in {output_file}...")

    if args.mode == "semantic":
        pairs = create_semantic_samples(
            model, tokenizer, sentences, output_file, start_idx=start_idx
        )
    elif args.mode == "double":
        pairs = create_double_pairs(model, tokenizer, sentences)
        with open(output_file, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    else:
        pairs = create_pairs(model, tokenizer, sentences, mode=args.mode)
        with open(output_file, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nFinished processing. Total new samples this run: {len(pairs)}")


if __name__ == "__main__":
    main()
