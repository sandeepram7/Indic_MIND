"""
Label unlabeled semantic-mode data using sentence similarity.

Takes the raw output from generate_data.py --mode semantic and assigns
labels (factual / hallucinated / discarded) based on cosine similarity
between the LLM completion and the Wikipedia ground truth.

Models used:
  Hindi:   l3cube-pune/hindi-sentence-similarity-sbert  (fine-tuned Hindi STS)
  English: paraphrase-multilingual-MiniLM-L12-v2        (fast multilingual)

Usage:
    python code/label_data.py --lang hi
    python code/label_data.py --lang en
    python code/label_data.py --high_thresh 0.85 --low_thresh 0.50
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

MODELS = {
    "hi": "l3cube-pune/hindi-sentence-similarity-sbert",
    "en": "paraphrase-multilingual-MiniLM-L12-v2",
}


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
    gt_embeddings = model.encode(
        ground_truths, show_progress_bar=True, batch_size=32, convert_to_tensor=True
    )
    print("Encoding LLM completions...")
    llm_embeddings = model.encode(
        llm_outputs, show_progress_bar=True, batch_size=32, convert_to_tensor=True
    )

    similarities = []
    for i in range(len(samples)):
        sim = util.cos_sim(gt_embeddings[i], llm_embeddings[i]).item()
        similarities.append(sim)

    return np.array(similarities)


def inspect_distribution(similarities):
    """Print similarity distribution stats for threshold tuning."""
    print(f"\n{'='*60}")
    print("Similarity Distribution Analysis")
    print(f"{'='*60}")
    print(f"  Count:  {len(similarities)}")
    print(f"  Mean:   {similarities.mean():.4f}")
    print(f"  Std:    {similarities.std():.4f}")
    print(f"  Min:    {similarities.min():.4f}")
    print(f"  Max:    {similarities.max():.4f}")
    print(f"  Median: {np.median(similarities):.4f}")

    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p:2d}:    {np.percentile(similarities, p):.4f}")

    print("\nBucket Distribution:")
    buckets = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
    for lo, hi in buckets:
        count = ((similarities >= lo) & (similarities < hi)).sum()
        pct = count / len(similarities) * 100
        if hi <= 0.5:
            label = " <- hallucinated"
        elif lo >= 0.85:
            label = " <- factual"
        else:
            label = " <- discard zone"
        print(f"  [{lo:.2f}, {hi:.2f}): {count:4d} ({pct:5.1f}%){label}")


def label_samples(samples, similarities, high_thresh, low_thresh):
    """Assign labels based on similarity thresholds.

    Rules:
      similarity >= high_thresh  ->  factual   (semantically equivalent)
      similarity <  low_thresh   ->  hallucinated (semantically divergent)
      otherwise                  ->  discarded (ambiguous)
    """
    factual = []
    hallucinated = []
    discarded = []

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
    """Save labeled samples paired by index, truncated to the smaller class.

    Output format matches what extract_hidden_states.py expects:
      {"full_factual": ..., "full_hallucinated": ..., ...}
    """
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
            "factual_ground_truth": factual[i]["full_ground_truth"],
            "hallucinated_ground_truth": hallucinated[i]["full_ground_truth"],
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
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        choices=["hi", "en"],
        help="Language: 'hi' for Hindi, 'en' for English",
    )
    parser.add_argument(
        "--high_thresh",
        type=float,
        default=0.85,
        help="Similarity >= this -> factual (default: 0.85)",
    )
    parser.add_argument(
        "--low_thresh",
        type=float,
        default=0.50,
        help="Similarity < this -> hallucinated (default: 0.50)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only print similarity distribution, don't label",
    )
    parser.add_argument(
        "--input_file", type=str, default=None, help="Override input file path"
    )
    args = parser.parse_args()

    if args.input_file:
        input_file = args.input_file
    elif args.lang == "en":
        input_file = os.path.join(DATA_DIR, "train_pairs_en_semantic.jsonl")
    else:
        input_file = os.path.join(DATA_DIR, "train_pairs_semantic.jsonl")

    print(f"Loading samples from {input_file}...")
    samples = load_samples(input_file)
    print(f"Loaded {len(samples)} samples")

    model_name = MODELS[args.lang]
    print(f"\nLoading similarity model: {model_name}")
    model = SentenceTransformer(model_name)

    print("\nComputing similarities...")
    similarities = compute_similarities(model, samples)

    inspect_distribution(similarities)

    if args.inspect:
        print("\n[--inspect mode] Exiting without labeling.")
        return

    print(
        f"\nLabeling with thresholds: factual >= {args.high_thresh}, "
        f"hallucinated < {args.low_thresh}"
    )
    factual, hallucinated, discarded = label_samples(
        samples, similarities, args.high_thresh, args.low_thresh
    )

    print(f"\n{'='*60}")
    print("Results:")
    print(f"  Factual:      {len(factual):4d}")
    print(f"  Hallucinated: {len(hallucinated):4d}")
    print(f"  Discarded:    {len(discarded):4d}")
    print(f"{'='*60}")

    n_pairs = min(len(factual), len(hallucinated))
    if n_pairs == 0:
        print("\nERROR: No balanced pairs possible! Adjust thresholds.")
        print("  Try: --inspect to see the distribution first")
        return

    lang_prefix = "en_" if args.lang == "en" else ""
    output_file = os.path.join(
        DATA_DIR, f"train_pairs_{lang_prefix}semantic_labeled.jsonl"
    )
    pairs = save_labeled_data(factual, hallucinated, output_file)

    print(f"\nSaved {len(pairs)} balanced pairs to {output_file}")
    print(f"  (Truncated to min({len(factual)}, {len(hallucinated)}) = {n_pairs})")

    if pairs:
        print("\nSample pair:")
        print(f"  Factual (sim={pairs[0]['factual_similarity']:.3f}):")
        print(f"    {pairs[0]['full_factual'][:100]}...")
        print(f"  Hallucinated (sim={pairs[0]['hallucinated_similarity']:.3f}):")
        print(f"    {pairs[0]['full_hallucinated'][:100]}...")


if __name__ == "__main__":
    main()
