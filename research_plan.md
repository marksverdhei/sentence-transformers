# Research Plan: Let's Embed Word Sense

This plan outlines the steps to implement and verify the research proposed in the abstract, focusing on embedding word senses using contrastive learning with dictionary definitions and extending this to Sense Sequence Models (SSM).

## Phase 1: Data & Methodology Setup

- [ ] **Data Preparation: WordNet**
    - [ ] Extract WordNet entries (lemmas, definitions, usage examples).
    - [ ] Format data into pairs: `(Target Word + Context, Definition)`.
    - [ ] Implement "Intra-word" negative mining (identify other senses of the same word lemma).
    - [ ] Implement "Hard" negative mining (identify similar but distinct words/definitions).

- [ ] **Core Model Architecture**
    - [ ] Implement `WordPooling` layer:
        - [ ] Input: Token embeddings + Word mask (indicating target word positions).
        - [ ] Operation: Mean pool *only* the tokens belonging to the target word.
    - [ ] Encoder Backbone: Select and initialize a standard transformer (e.g., BERT, RoBERTa, or DistilBERT).
    - [ ] Definition Encoder: Decide if sharing weights with the context encoder or using a separate encoder.

- [ ] **Loss Function Implementation**
    - [ ] Implement Contrastive Loss (InfoNCE/MultipleNegativesRankingLoss).
    - [ ] Integrate intra-word negatives: Ensure other senses of the anchor word are treated as hard negatives in the batch or explicitly added.

## Phase 2: Training & Verification

- [ ] **Training Loop**
    - [ ] Setup training pipeline (likely using `sentence-transformers` or Hugging Face `Trainer`).
    - [ ] Train on the prepared WordNet dataset.
    - [ ] Monitor validation loss on held-out word senses.

- [ ] **Qualitative Analysis**
    - [ ] Visualize embeddings: Check if different senses of polysemous words (e.g., "bank") form distinct clusters.
    - [ ] Nearest Neighbors: Query the embedding space with a word in context and retrieve the correct definition.

## Phase 3: Core Experiments (WSD & WSI)

- [ ] **Word Sense Disambiguation (WSD)**
    - [ ] Select benchmarks (e.g., WSD Framework, Raganato et al. datasets).
    - [ ] Implement evaluation script:
        1. Embed target word in context.
        2. Embed all candidate definitions from WordNet.
        3. Predict sense via nearest neighbor search (cosine similarity).
    - [ ] Compare against baselines (standard BERT, static embeddings, Most Frequent Sense).

- [ ] **Word Sense Induction (WSI)**
    - [ ] Select WSI datasets (e.g., SemEval-2010 Task 14, SemEval-2013 Task 13).
    - [ ] Implement clustering approach on the learned sense embeddings.
    - [ ] Measure Adjusted Rand Index (ARI) or V-Measure.

## Phase 4: Advanced Extensions

- [ ] **Auxiliary RL Objectives**
    - [ ] Design the RL setup for Autoregressive (AR) Language Models.
    - [ ] Define the reward function based on sense embedding coherence.
    - [ ] Run experiments to see if this improves generation quality or consistency.

- [ ] **Sense Sequence Model (SSM)**
    - [ ] Architecture Design: Create a model that predicts a sequence of *sense vectors* rather than token IDs.
    - [ ] Training: Train on a corpus annotated with sense IDs (or silver-labeled data).
    - [ ] Analysis: Investigate properties of sense-level generation vs. token-level generation.

## Phase 5: Paper Writing & Documentation

- [ ] **Drafting**
    - [ ] Fill in Methodology section with specific mathematical formulations of the loss and pooling.
    - [ ] Create tables for WSD and WSI results.
    - [ ] Produce t-SNE/PCA plots for embedding visualizations.
- [ ] **Review**
    - [ ] Check against the Abstract claims.
    - [ ] Finalize the "Limitations" and "Ethics Statement" sections.
