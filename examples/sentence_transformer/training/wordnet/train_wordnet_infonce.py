"""
Train a sentence embedding model using InfoNCE on WordNet definitions.

Key features:
1. Word-prefixed inputs: Both example and definition have "'<word>': " prefix
2. Word-token pooling: Mean pool only tokens of the target word (not full sentence)
3. Inter-word negatives: Other definitions of same word are hard negatives for disambiguation

The model learns word-sense embeddings that disambiguate polysemous words.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


class WordPooling(nn.Module):
    """
    Pooling layer that computes mean pooling only over the tokens of the target word.

    Input format expected: "'<word>': <text>"
    The pooling is done on the tokens that make up <word>.
    """

    def __init__(self, word_embedding_dimension: int):
        super().__init__()
        self.word_embedding_dimension = word_embedding_dimension

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        token_embeddings = features["token_embeddings"]  # [batch, seq_len, dim]
        attention_mask = features["attention_mask"]  # [batch, seq_len]

        # Get word masks if provided, otherwise fall back to full mean pooling
        if "word_mask" in features:
            word_mask = features["word_mask"]  # [batch, seq_len]
        else:
            # Fall back to using attention mask (full sentence pooling)
            word_mask = attention_mask

        # Expand mask for broadcasting
        word_mask_expanded = word_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings where word_mask is 1
        sum_embeddings = torch.sum(token_embeddings * word_mask_expanded, dim=1)

        # Count non-zero positions (avoid division by zero)
        sum_mask = word_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Mean pooling
        sentence_embeddings = sum_embeddings / sum_mask

        features["sentence_embedding"] = sentence_embeddings
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.word_embedding_dimension

    def get_config_dict(self) -> dict:
        return {"word_embedding_dimension": self.word_embedding_dimension}


class WordSenseTransformer(SentenceTransformer):
    """
    Custom SentenceTransformer that:
    1. Identifies word tokens in the input
    2. Pools only those tokens for the embedding
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        # Initialize with empty modules first
        super().__init__(modules=[])

        # Load transformer and tokenizer
        self.transformer = Transformer(model_name)
        self.tokenizer = self.transformer.tokenizer

        # Create word pooling layer
        self.pooling = WordPooling(self.transformer.get_word_embedding_dimension())

        # Add modules
        self._modules["0"] = self.transformer
        self._modules["1"] = self.pooling

    def tokenize(self, texts: list[str], **kwargs) -> dict[str, Tensor]:
        """
        Tokenize and create word masks.

        Expected format: "'<word>': <text>"
        """
        # Standard tokenization
        batch_encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # Create word masks
        word_masks = []
        offset_mapping = batch_encoding.pop("offset_mapping")

        for idx, text in enumerate(texts):
            # Parse the word from the text format "'<word>': <text>"
            word_mask = torch.zeros(batch_encoding["input_ids"].shape[1], dtype=torch.long)

            if text.startswith("'") and "': " in text:
                # Extract word between quotes
                end_quote = text.index("': ")
                word = text[1:end_quote]
                word_start_char = 1  # After opening quote
                word_end_char = end_quote  # Before closing quote

                # Find tokens that overlap with the word position
                offsets = offset_mapping[idx]
                for token_idx, (start, end) in enumerate(offsets):
                    if start == 0 and end == 0:
                        continue  # Special token
                    # Check if token overlaps with word position
                    if start < word_end_char and end > word_start_char:
                        word_mask[token_idx] = 1
            else:
                # Fallback: use all non-special tokens
                word_mask = batch_encoding["attention_mask"][idx].clone()

            # Ensure at least one token is selected
            if word_mask.sum() == 0:
                word_mask = batch_encoding["attention_mask"][idx].clone()

            word_masks.append(word_mask)

        batch_encoding["word_mask"] = torch.stack(word_masks)
        return batch_encoding

    def forward(self, features: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Forward pass through transformer and word pooling."""
        # Pass through transformer
        trans_features = self.transformer(features)

        # Add word_mask to features if present
        if "word_mask" in features:
            trans_features["word_mask"] = features["word_mask"]

        # Pass through pooling
        return self.pooling(trans_features)


def load_and_prepare_dataset():
    """
    Load WordNet dataset and prepare with inter-word negatives.

    Returns:
        - train_dataset with columns: anchor, positive, negative_0, ..., negative_k
        - val/test datasets for evaluation
        - word_to_indices mapping
    """
    logging.info("Loading WordNet definitions dataset...")
    dataset = load_dataset("marksverdhei/wordnet-definitions-en-2021")

    # Group by word to find polysemous words
    word_to_examples = defaultdict(list)

    for idx, example in enumerate(dataset["train"]):
        word = example["Word"]
        word_to_examples[word].append({
            "idx": idx,
            "word": word,
            "example": example["Example"],
            "definition": example["Definition"],
        })

    # Count polysemous words
    polysemous = {w: exs for w, exs in word_to_examples.items() if len(exs) > 1}
    logging.info(f"Total unique words: {len(word_to_examples)}")
    logging.info(f"Polysemous words (multiple definitions): {len(polysemous)}")
    logging.info(f"Max definitions per word: {max(len(v) for v in word_to_examples.values())}")

    # Prepare training data with inter-word negatives
    train_data = {
        "anchor": [],
        "positive": [],
    }

    # Add columns for inter-word negatives (up to 4)
    max_inter_negatives = 4
    for i in range(max_inter_negatives):
        train_data[f"negative_{i}"] = []

    for word, examples in word_to_examples.items():
        for i, ex in enumerate(examples):
            # Format: "'<word>': <text>"
            anchor = f"'{word}': {ex['example']}"
            positive = f"'{word}': {ex['definition']}"

            train_data["anchor"].append(anchor)
            train_data["positive"].append(positive)

            # Inter-word negatives: other definitions of the same word
            other_defs = [e for j, e in enumerate(examples) if j != i]

            for neg_idx in range(max_inter_negatives):
                if neg_idx < len(other_defs):
                    # Use another definition of same word as negative
                    neg_def = other_defs[neg_idx]["definition"]
                    train_data[f"negative_{neg_idx}"].append(f"'{word}': {neg_def}")
                else:
                    # No more inter-word negatives, use empty string (will be filtered)
                    train_data[f"negative_{neg_idx}"].append("")

    # Create dataset
    train_dataset = Dataset.from_dict(train_data)

    # Prepare val/test datasets
    def format_split(split):
        anchors = []
        positives = []
        for ex in dataset[split]:
            word = ex["Word"]
            anchors.append(f"'{word}': {ex['Example']}")
            positives.append(f"'{word}': {ex['Definition']}")
        return anchors, positives

    val_anchors, val_positives = format_split("validation")
    test_anchors, test_positives = format_split("test")

    return train_dataset, val_anchors, val_positives, test_anchors, test_positives, polysemous


class InterWordNegativeLoss(nn.Module):
    """
    Custom loss that:
    1. Uses MultipleNegativesRankingLoss as base
    2. Properly handles inter-word negatives (other definitions of same word)
    3. Falls back to in-batch negatives when inter-word negatives are empty
    """

    def __init__(self, model: SentenceTransformer, scale: float = 20.0):
        super().__init__()
        self.model = model
        self.scale = scale
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, sentence_features: list[dict[str, Tensor]], labels: Tensor = None) -> Tensor:
        # sentence_features: [anchor_features, positive_features, neg0_features, ...]

        # Encode all
        embeddings = []
        for features in sentence_features:
            emb = self.model(features)["sentence_embedding"]
            embeddings.append(emb)

        anchor_emb = embeddings[0]  # [batch, dim]
        positive_emb = embeddings[1]  # [batch, dim]

        # Normalize
        anchor_emb = torch.nn.functional.normalize(anchor_emb, p=2, dim=1)
        positive_emb = torch.nn.functional.normalize(positive_emb, p=2, dim=1)

        # Compute similarity with positives (in-batch negatives)
        scores = torch.mm(anchor_emb, positive_emb.T) * self.scale  # [batch, batch]

        # Add inter-word negatives if present
        if len(embeddings) > 2:
            for neg_emb in embeddings[2:]:
                neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=1)
                # Each anchor's inter-word negative
                inter_neg_scores = (anchor_emb * neg_emb).sum(dim=1, keepdim=True) * self.scale
                scores = torch.cat([scores, inter_neg_scores], dim=1)

        # Labels: positive is always at index i for anchor i
        labels = torch.arange(anchor_emb.size(0), device=anchor_emb.device)

        return self.cross_entropy(scores, labels)


def create_ir_evaluator(
    model: SentenceTransformer,
    anchors: list[str],
    definitions: list[str],
    name: str = "wordnet",
) -> InformationRetrievalEvaluator:
    """Create IR evaluator for word sense retrieval."""
    queries = {str(i): anchor for i, anchor in enumerate(anchors)}
    corpus = {str(i): definition for i, definition in enumerate(definitions)}
    relevant_docs = {str(i): {str(i)} for i in range(len(anchors))}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        mrr_at_k=[1, 5, 10],
        ndcg_at_k=[1, 5, 10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10],
        show_progress_bar=True,
    )


def show_polysemy_examples(polysemous: dict, n_words: int = 5):
    """Show examples of polysemous words."""
    logging.info(f"\n📚 Examples of polysemous words (words with multiple definitions):")

    for i, (word, examples) in enumerate(list(polysemous.items())[:n_words]):
        logging.info(f"\n  '{word}' ({len(examples)} senses):")
        for j, ex in enumerate(examples[:3]):
            logging.info(f"    [{j+1}] {ex['definition'][:60]}...")
            logging.info(f"        Example: {ex['example'][:50]}...")


def main():
    # Configuration
    model_name = "distilbert-base-uncased"
    num_train_epochs = 3
    batch_size = 64
    learning_rate = 2e-5
    warmup_ratio = 0.1

    output_dir = f"output/wordnet-wordsense-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Load data with inter-word negatives
    train_dataset, val_anchors, val_positives, test_anchors, test_positives, polysemous = \
        load_and_prepare_dataset()

    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_anchors)}")
    logging.info(f"Test samples: {len(test_anchors)}")

    # Show polysemy examples
    show_polysemy_examples(polysemous)

    # Show training example
    logging.info("\n📝 Training example:")
    ex = train_dataset[0]
    logging.info(f"  Anchor:   {ex['anchor']}")
    logging.info(f"  Positive: {ex['positive']}")
    for i in range(4):
        neg = ex.get(f'negative_{i}', '')
        if neg:
            logging.info(f"  Neg {i}:    {neg}")

    # Create model with word pooling
    logging.info(f"\nCreating WordSenseTransformer from {model_name}...")
    model = WordSenseTransformer(model_name)

    # Test tokenization with word mask
    logging.info("\n🔍 Testing word-token identification:")
    test_texts = [
        "'split up': My friend and I split up",
        "'bank': I went to the bank",
        "'running': The running water was cold",
    ]
    for text in test_texts:
        tokens = model.tokenize([text])
        word_mask = tokens["word_mask"][0]
        input_ids = tokens["input_ids"][0]
        token_strs = model.tokenizer.convert_ids_to_tokens(input_ids)
        word_tokens = [t for t, m in zip(token_strs, word_mask) if m == 1]
        logging.info(f"  '{text[:40]}...' → word tokens: {word_tokens}")

    # Create validation dataset
    val_dataset = Dataset.from_dict({
        "anchor": val_anchors,
        "positive": val_positives,
    })

    # Create loss function with inter-word negatives
    train_loss = InterWordNegativeLoss(model=model, scale=20.0)

    # Create evaluator
    evaluator = create_ir_evaluator(
        model=model,
        anchors=val_anchors,
        definitions=val_positives,
        name="wordnet-validation",
    )

    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=True,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="wordnet-validation_ndcg@10",
        greater_is_better=True,
        run_name="wordnet-wordsense",
    )

    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

    # Evaluate before training
    logging.info("\n📊 Evaluating model before training...")
    evaluator(model, epoch=0, steps=0)

    # Train
    logging.info("\n🚀 Starting training...")
    trainer.train()

    # Save model
    final_output_dir = f"{output_dir}/final"
    model.save(final_output_dir)
    logging.info(f"Model saved to {final_output_dir}")

    # Final evaluation
    logging.info("\n📊 Final evaluation on test set...")
    test_evaluator = create_ir_evaluator(
        model=model,
        anchors=test_anchors,
        definitions=test_positives,
        name="wordnet-test",
    )
    test_results = test_evaluator(model)

    logging.info("\nTest Results:")
    for metric, value in test_results.items():
        logging.info(f"  {metric}: {value:.4f}")

    # Show disambiguation examples
    logging.info("\n🎯 Word Sense Disambiguation Examples:")
    show_disambiguation_examples(model, polysemous)


def show_disambiguation_examples(model: WordSenseTransformer, polysemous: dict, n_words: int = 5):
    """Show how well the model disambiguates word senses."""

    for word, examples in list(polysemous.items())[:n_words]:
        if len(examples) < 2:
            continue

        logging.info(f"\n  Word: '{word}' ({len(examples)} senses)")

        # Encode all examples and definitions for this word
        anchors = [f"'{word}': {ex['example']}" for ex in examples]
        definitions = [f"'{word}': {ex['definition']}" for ex in examples]

        anchor_embs = model.encode(anchors, convert_to_tensor=True, normalize_embeddings=True)
        def_embs = model.encode(definitions, convert_to_tensor=True, normalize_embeddings=True)

        # Compute similarity matrix
        sims = torch.mm(anchor_embs, def_embs.T)

        # Check if each example retrieves its correct definition
        for i, ex in enumerate(examples[:3]):
            top_idx = sims[i].argmax().item()
            correct = "✅" if top_idx == i else f"❌ (got def {top_idx+1})"
            logging.info(f"    Example {i+1}: '{ex['example'][:40]}...'")
            logging.info(f"      → {correct} Definition: '{ex['definition'][:40]}...'")


if __name__ == "__main__":
    main()
