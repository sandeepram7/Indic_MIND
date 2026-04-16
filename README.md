# Indic-MIND: Unsupervised Real-Time Hallucination Detection for Hindi via Internal State Probing

This repository contains the implementation of **Indic-MIND**, an adaptation and improvement of the [MIND framework](https://github.com/oneal2000/MIND) (Su et al., ACL 2024 Findings) for real-time hallucination detection in Hindi Large Language Models.

> **Paper:** Indic-MIND: Unsupervised Real-Time Hallucination Detection for Hindi via Internal State Probing
> **Base Model:** `meta-llama/Llama-3.2-3B-Instruct` (bfloat16, no quantization)
> **Data Source:** Hindi Wikipedia (~160,000 articles) and English Wikipedia (for baseline)

## Overview

Large Language Models hallucinate - they generate fluent, grammatically correct text that is factually wrong. This is a well-documented problem in English, but almost zero research exists on detecting hallucinations in Indic languages in real-time. Hindi Wikipedia has ~160,000 articles compared to English's ~6.8 million (a 42x gap), and Llama-3's pre-training data uses less than 0.1% Hindi text. This means the model's internal representations for Hindi are significantly sparser and noisier, making hallucination detection both harder and more scientifically interesting.

MIND (Su et al., 2024) demonstrated that when an LLM generates factually correct text, its internal hidden states (the intermediate vector representations computed at each Transformer layer) are mathematically distinguishable from when it hallucinates. They trained a lightweight MLP classifier on these hidden states to perform real-time hallucination detection with less than 1ms per-token overhead. However, their work was validated exclusively on English data using six different LLMs (LLaMA-2, Falcon, OPT, GPT-J).

**Indic-MIND is the first effort to bring real-time internal-state hallucination probing to a low-resource Indic language.**

## What We Changed from MIND

The original MIND framework has a fundamental weakness in its data generation pipeline that we identified and fixed.

### The Problem with MIND's NER-Based Labeling

MIND generates training data by:

1. Taking a Wikipedia sentence and identifying a Named Entity using SpaCy NER.
2. Truncating the sentence at that entity.
3. Asking the LLM to complete the sentence.
4. Checking if the LLM outputs the **exact same entity string**. If yes → Factual. If no → Hallucinated.

This produces noisy labels. If the LLM correctly paraphrases the fact (e.g., generates "the capital of France" instead of "Paris"), MIND incorrectly flags it as a hallucination. Hindi NER tools are also significantly weaker than English SpaCy, making this approach even more unreliable for our use case.

### Our Solution: Natural Generation + Semantic Similarity Labeling

We completely replaced MIND's NER-masking with a two-stage pipeline:

**Stage 1 — Natural Generation:** We truncate Hindi Wikipedia sentences at a random midpoint (40-60% of tokens) and prompt Llama-3.2-3B to complete them using a purely neutral prompt. The model is never instructed to hallucinate. This eliminates the "prompt-type confound" — in our earlier iterations, when we used an adversarial prompt forcing the model to lie, the probe learned to detect the model's internal awareness of its adversarial instructions rather than genuine hallucination.

**Stage 2 — Semantic Similarity Labeling:** We decouple labeling from generation entirely. A dedicated Hindi Sentence Transformer model ([`l3cube-pune/hindi-sentence-similarity-sbert`](https://huggingface.co/l3cube-pune/hindi-sentence-similarity-sbert)) computes the cosine similarity between the LLM's natural completion and the Wikipedia ground truth:

- **Similarity ≥ 0.85** → Factual (the LLM reproduced the correct information)
- **Similarity < 0.50** → Hallucinated (the LLM deviated significantly from the truth)
- **0.50 – 0.85** → Discarded (ambiguous, could go either way)

This gives the MLP probe a mathematically clean, unambiguous training signal with zero label noise from paraphrasing.

## Results

### Hindi (Primary, 50,000 samples generated)

| Metric               | Value                |
| -------------------- | -------------------- |
| Test AUC             | **0.9163**     |
| Test Accuracy        | 82.4%                |
| Hallucination Recall | **0.89**       |
| Factual Recall       | 0.76                 |
| Training Samples     | 5,464 balanced pairs |

### English Baseline (Control, 50,000 samples generated)

| Metric        | Value            |
| ------------- | ---------------- |
| Test AUC      | **0.9588** |
| Test Accuracy | 86.1%            |

The English baseline was run under identical conditions (same model, same thresholds, same probe architecture) to serve as a scientific control. The ~4.2% AUC gap directly measures the impact of Hindi's sparse pre-training data on the detectability of internal factuality signals. This gap is our core finding: **real-time hallucination detection works for Hindi, but the sparser internal representations make it measurably harder than English.**

### Why Our AUC is Higher than MIND's (0.72–0.80)

The original MIND paper reports sentence-level AUC of 0.72–0.80 across six LLMs. Our numbers are higher for three verifiable reasons:

1. **Cleaner training labels.** MIND's NER-based labeling produces noisy labels (paraphrases wrongly labeled as hallucinations). Our semantic similarity pipeline with explicit ambiguity filtering produces a mathematically clean dataset.
2. **Better base model.** MIND used LLaMA-2-7B/13B, OPT-6.7B, Falcon-40B, and GPT-J-6B. We use Llama-3.2-3B-Instruct, which has significantly denser and more refined internal representations despite being smaller.
3. **Optimized training.** MIND used a 4-layer MLP with 20% dropout and no early stopping. We use a 3-layer MLP with 50% dropout, L2 regularization, weighted BCE loss for class asymmetry correction, and early stopping with patience=15.

## Pipeline

The pipeline has four stages:

```
Wikipedia Sentences
        |
        v
[1] generate_data.py          -- Extracts Hindi Wikipedia sentences, truncates them, and prompts Llama-3.2-3B to complete naturally.
        |
        v
[2] label_data.py              -- Computes cosine similarity between each LLM completion and the Wikipedia ground truth using Hindi-SBERT. Labels samples as factual/hallucinated/discarded based on thresholds.
        |
        v
[3] extract_hidden_states.py   -- Extract 6144-dim vectors from frozen Llama-3.2-3B
        |
        v
[4] train_probe.py             -- Train 3-layer MLP on the extracted hidden states with weighted BCE loss
```

## Setup

### Requirements

- Python 3.9+
- CUDA-capable GPU (tested on RTX 4070 Ti, 12GB VRAM — the 3B model in bfloat16 uses ~6GB, leaving room for hidden state extraction)
- Hugging Face access to `meta-llama/Llama-3.2-3B-Instruct`

### Installation

```bash
pip install torch transformers accelerate datasets
pip install sentence-transformers scikit-learn numpy tqdm
```

### Running the Full Pipeline

#### Hindi (Primary)

```bash
# Step 1: Generate 50,000 natural completions (streams to disk, auto-resumes)
python code/generate_data.py --num_samples 50000 --mode semantic

# Step 2: Label using semantic similarity
python code/label_data.py --lang hi --high_thresh 0.85 --low_thresh 0.50

# Step 3: Extract hidden states from Llama-3.2-3B
python code/extract_hidden_states.py --mode semantic_labeled

# Step 4: Train the MLP probe
python code/train_probe.py --tag _semantic_labeled --epochs 100
```

#### English Baseline

```bash
python code/generate_data_english.py --num_samples 50000 --mode semantic
python code/label_data.py --lang en --high_thresh 0.85 --low_thresh 0.50
python code/extract_hidden_states.py --mode en_semantic_labeled
python code/train_probe.py --tag _en_semantic_labeled --epochs 100
```

### Inspect Similarity Distribution (Before Labeling)

```bash
python code/label_data.py --lang hi --inspect
```

This prints the full similarity distribution (mean, std, percentiles, bucket counts) without actually labeling, so you can tune thresholds before committing.

## Repository Structure

```
code/
├── generate_data.py            # Hindi Wikipedia sentence completion generation
├── generate_data_english.py    # English baseline generation (identical architecture)
├── label_data.py               # Semantic similarity labeling (Hindi-SBERT / MiniLM)
├── extract_hidden_states.py    # Hidden state extraction from frozen Llama-3.2-3B
└── train_probe.py              # MLP probe training with weighted loss and early stopping
```

## Technical Details

### Feature Extraction

For each labeled sentence, two feature vectors are extracted from the frozen Llama-3.2-3B model:

- **Feature 1:** Hidden state of the last token, averaged across all 28 Transformer layers. Captures the model's cumulative semantic representation at the point of generation.
- **Feature 2:** Last layer activations, averaged across all tokens. Captures the model's final-layer understanding of the full sentence.

These are concatenated into a single 6,144-dimensional vector per sample.

### Probe Architecture

- 3-layer MLP: `Linear(6144, 256) → ReLU → Dropout(0.5) → Linear(256, 64) → ReLU → Dropout(0.5) → Linear(64, 1)`
- Loss: `BCEWithLogitsLoss(pos_weight=2.0)` — penalizes missed hallucinations 2x more heavily than missed facts. This was critical for resolving a class asymmetry issue where the initial probe achieved 93% factual recall but only 70% hallucination recall.
- Optimizer: Adam with `lr=1e-3`, `weight_decay=1e-4`
- Early stopping: patience=15 epochs monitoring validation AUC
- Data split: 72% train / 13% validation / 15% test (stratified)

### Crash Resilience

The generation scripts (`generate_data.py`, `generate_data_english.py`) write each sample to disk line-by-line via JSONL streaming. If the process is interrupted (GPU OOM, SSH disconnect, etc.), re-running the same command automatically detects existing output and resumes from the exact line where it stopped.

## Future Work

- **BHRAM-IL Evaluation:** Evaluate the trained probe against the Hindi subset of BHRAM-IL (Thakur et al., 2025) for external validity.
- **Layer Ablation:** Systematic ablation across individual Transformer layers to identify which layers encode the strongest factuality signal.
- **Cross-lingual Transfer:** Test whether a probe trained on Hindi hidden states transfers to other Indic languages (Marathi, Gujarati) without retraining.
- **Gradio Demo:** Real-time demonstration interface showing per-token hallucination probabilities during generation.

## Citation

This work builds on the MIND framework:

```bibtex
@inproceedings{su2024mind,
  title={Unsupervised Real-Time Hallucination Detection based on the Internal States of Large Language Models},
  author={Su, Weihang and Wang, Changyue and Ai, Qingyao and Hu, Yiran and Wu, Zhijing and Zhou, Yujia and Liu, Yiqun},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2024},
  year={2024}
}
```
