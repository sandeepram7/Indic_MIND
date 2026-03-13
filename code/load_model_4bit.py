"""
Step 1: Load Llama-3.2-3B-Instruct (unquantized bfloat16) and test Hindi generation.

Usage:
    python code/load_model_4bit.py

Requirements:
    - RTX 4070 Ti (12GB VRAM) or better
    - pip install torch transformers accelerate
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "outputs"

def load_model():
    """Load Llama-3.2-3B-Instruct in bfloat16 (no quantization)."""
    print(f"Loading {MODEL_NAME} in bfloat16 (unquantized)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded. Device map: {model.hf_device_map}")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model, tokenizer


def test_hindi_generation(model, tokenizer):
    """Test the model with Hindi prompts to verify it generates readable Hindi."""

    hindi_prompts = [
        "Dolo-650 ki recommended dose kya hai? Hindi mein batayein.",
        "Bharat ki rajdhani kya hai? Vistaar se batayein.",
        "Diabetes ke lakshan kya hote hain? Hindi mein samjhayein.",
        "Mahatma Gandhi ka janam kab hua tha?",
        "Python programming language kya hai? Hindi mein batayein.",
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    for i, prompt in enumerate(hindi_prompts):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Input: {prompt}")

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Output: {response[:200]}")
        results.append(f"Prompt: {prompt}\nResponse: {response}\n{'='*80}\n")

    output_file = os.path.join(OUTPUT_DIR, "hindi_generation_test.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(results)

    print(f"\nResults saved to {output_file}")
    return results


if __name__ == "__main__":
    model, tokenizer = load_model()
    test_hindi_generation(model, tokenizer)
