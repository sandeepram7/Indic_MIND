"""
Label unlabeled semantic-mode data using sentence similarity.

Takes the raw output from generate_data.py --mode semantic and assigns
labels (factual / hallucinated / discarded) based on cosine similarity
between the LLM completion and the Wikipedia ground truth.

Model used:
  Hindi:   l3cube-pune/hindi-sentence-similarity-sbert

Thresholds:
  similarity >= 0.85  →  factual
  similarity <  0.50  →  hallucinated
  0.50 <= sim < 0.85  →  discarded (ambiguous)

Usage:
    python code/label_data.py
    python code/label_data.py --inspect
"""

import argparse
import json
import os
import random
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

DATA_DIR = "data"

# Hindi sentence similarity model (fine-tuned for STS)
MODEL_NAME = "l3cube-pune/hindi-sentence-similarity-sbert"


def load_samples(input_file):
    """Load JSONL samples from disk."""
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def compute_similarities(model, samples):
    """Compute cosine similarity between ground truth and LLM completion."""
    ground_truths = [s["full_ground_truth"] for s in samples]
    llm_outputs = [s["full_llm"] for s in samples]

    print("Encoding ground truths...")
    gt_embeddings = model.encode(ground_truths, show_progress_bar=True,
                                  batch_size=32, convert_to_tensor=True)
    print("Encoding LLM completions...")
    llm_embeddings = model.encode(llm_outputs, show_progress_bar=True,
                                   batch_size=32, convert_to_tensor=True)

    similarities = []
    for i in range(len(samples)):
        sim = util.cos_sim(gt_embeddings[i], llm_embeddings[i]).item()
        similarities.append(sim)

    return np.array(similarities)


def inspect_distribution(similarities):
    """Print similarity distribution stats for threshold tuning."""
    print(f"\n{'='*60}")
    print(f"Similarity Distribution")
    print(f"{'='*60}")
    print(f"  Count:  {len(similarities)}")
    print(f"  Mean:   {similarities.mean():.4f}")
    print(f"  Std:    {similarities.std():.4f}")
    print(f"  Min:    {similarities.min():.4f}")
    print(f"  Max:    {similarities.max():.4f}")

    buckets = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
    print(f"\nBucket Distribution:")
    for lo, hi in buckets:
        count = ((similarities >= lo) & (similarities < hi)).sum()
        pct = count / len(similarities) * 100
        label = " ← hallucinated" if hi <= 0.5 else (" ← factual" if lo >= 0.85 else " ← discard")
        print(f"  [{lo:.2f}, {hi:.2f}): {count:4d} ({pct:5.1f}%){label}")


def label_samples(samples, similarities, high_thresh=0.85, low_thresh=0.50):
    """Assign labels based on cosine similarity thresholds."""
    factual, hallucinated, discarded = [], [], []

    for sample, sim in zip(samples, similarities):
        sample["similarity"] = float(sim)
        if sim >= high_thresh:
            sample["label"] = "factual"
            factual.append(sample)
        elif sim < low_thresh:
            sample["label"] = "hallucinated"
            hallucinated.append(sample)
        else:
            sample["label"] = "discarded"
            discarded.append(sample)

    return factual, hallucinated, discarded


def save_labeled_data(factual, hallucinated, output_file):
    """Save balanced paired data for extract_hidden_states.py."""
    n = min(len(factual), len(hallucinated))

    random.seed(42)
    random.shuffle(factual)
    random.shuffle(hallucinated)

    pairs = []
    for i in range(n):
        pair = {
            "full_factual": factual[i]["full_llm"],
            "full_hallucinated": hallucinated[i]["full_llm"],
            "factual_similarity": factual[i]["similarity"],
            "hallucinated_similarity": hallucinated[i]["similarity"],
            "mode": "semantic",
        }
        pairs.append(pair)

    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Label semantic-mode data using sentence similarity"
    )
    parser.add_argument("--high_thresh", type=float, default=0.85)
    parser.add_argument("--low_thresh", type=float, default=0.50)
    parser.add_argument("--inspect", action="store_true",
                        help="Only print similarity distribution, don't label")
    args = parser.parse_args()

    input_file = os.path.join(DATA_DIR, "train_pairs_semantic.jsonl")
    print(f"Loading samples from {input_file}...")
    samples = load_samples(input_file)
    print(f"Loaded {len(samples)} samples")

    print(f"\nLoading similarity model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("\nComputing similarities...")
    similarities = compute_similarities(model, samples)
    inspect_distribution(similarities)

    if args.inspect:
        print("\n[--inspect mode] Exiting without labeling.")
        return

    print(f"\nLabeling: factual >= {args.high_thresh}, hallucinated < {args.low_thresh}")
    factual, hallucinated, discarded = label_samples(
        samples, similarities, args.high_thresh, args.low_thresh
    )

    print(f"\n{'='*60}")
    print(f"  Factual:      {len(factual)}")
    print(f"  Hallucinated: {len(hallucinated)}")
    print(f"  Discarded:    {len(discarded)}")
    print(f"{'='*60}")

    output_file = os.path.join(DATA_DIR, "train_pairs_semantic_labeled.jsonl")
    pairs = save_labeled_data(factual, hallucinated, output_file)

    print(f"\nSaved {len(pairs)} balanced pairs to {output_file}")


if __name__ == "__main__":
    main()
