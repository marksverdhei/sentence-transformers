"""
Evaluate Word Sense Disambiguation accuracy for any SentenceTransformer model.

Usage:
    python eval_wsd.py                           # Evaluate distilbert-base-uncased
    python eval_wsd.py --model BAAI/bge-large-en-v1.5
    python eval_wsd.py --model ./output/my-trained-model/final
"""

import argparse
import logging
import torch
from collections import defaultdict
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def load_polysemous_words():
    """Load WordNet dataset and extract polysemous words."""
    ds = load_dataset("marksverdhei/wordnet-definitions-en-2021", split="train")

    polysemous = defaultdict(list)
    for row in ds:
        example = row.get("Example")
        if example:
            polysemous[row["Word"]].append({
                "example": example,
                "definition": row["Definition"]
            })

    # Keep only words with multiple senses
    polysemous = {w: exs for w, exs in polysemous.items() if len(exs) >= 2}
    return polysemous


def evaluate_wsd(model: SentenceTransformer, polysemous: dict, use_word_prefix: bool = True):
    """
    Evaluate word sense disambiguation accuracy.

    For each polysemous word, check if the model correctly matches each example
    to its definition among all definitions of that word.
    """
    # Collect all texts first
    all_anchors = []
    all_definitions = []
    word_indices = []  # (word, n_senses, start_idx)

    idx = 0
    for word, examples in polysemous.items():
        n_senses = len(examples)
        if use_word_prefix:
            anchors = [f"'{word}': {ex['example']}" for ex in examples]
            definitions = [f"'{word}': {ex['definition']}" for ex in examples]
        else:
            anchors = [ex["example"] for ex in examples]
            definitions = [ex["definition"] for ex in examples]

        all_anchors.extend(anchors)
        all_definitions.extend(definitions)
        word_indices.append((word, n_senses, idx))
        idx += n_senses

    logging.info(f"Encoding {len(all_anchors)} anchors...")
    anchor_embs = model.encode(all_anchors, convert_to_tensor=True, normalize_embeddings=True,
                                show_progress_bar=True, batch_size=128)
    logging.info(f"Encoding {len(all_definitions)} definitions...")
    def_embs = model.encode(all_definitions, convert_to_tensor=True, normalize_embeddings=True,
                             show_progress_bar=True, batch_size=128)

    # Evaluate per word
    correct = 0
    total = 0
    correct_by_num_senses = defaultdict(lambda: [0, 0])

    for word, n_senses, start_idx in word_indices:
        end_idx = start_idx + n_senses

        # Get embeddings for this word
        word_anchor_embs = anchor_embs[start_idx:end_idx]
        word_def_embs = def_embs[start_idx:end_idx]

        # Compute similarity matrix for this word
        sims = torch.mm(word_anchor_embs, word_def_embs.T)

        # Check if each example retrieves its correct definition
        for i in range(n_senses):
            top_idx = sims[i].argmax().item()
            if top_idx == i:
                correct += 1
                correct_by_num_senses[n_senses][0] += 1
            total += 1
            correct_by_num_senses[n_senses][1] += 1

    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_num_senses": dict(correct_by_num_senses)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate WSD accuracy for a SentenceTransformer model")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                        help="Model name or path to evaluate")
    parser.add_argument("--no-word-prefix", action="store_true",
                        help="Don't use word prefix in inputs")
    args = parser.parse_args()

    logging.info(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    logging.info("Loading WordNet dataset...")
    polysemous = load_polysemous_words()
    logging.info(f"Polysemous words: {len(polysemous)}")

    logging.info("Evaluating WSD...")
    results = evaluate_wsd(model, polysemous, use_word_prefix=not args.no_word_prefix)

    logging.info(f"\n{'='*60}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Word prefix: {not args.no_word_prefix}")
    logging.info(f"{'='*60}")
    logging.info(f"Overall Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    logging.info(f"\nBreakdown by number of senses:")
    for n_senses in sorted(results["by_num_senses"].keys())[:10]:
        c, t = results["by_num_senses"][n_senses]
        logging.info(f"  {n_senses} senses: {c/t:.1%} ({c}/{t})")


if __name__ == "__main__":
    main()
