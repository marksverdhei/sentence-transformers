# Training Word-Sense Embeddings with InfoNCE and WordNet

In this post, we explore a specialized method for training sentence embeddings that focuses on **word sense disambiguation**. By leveraging the rich structure of WordNet and using a custom pooling strategy with InfoNCE loss, we can train models that understand the specific meaning of a word in context.

## The Challenge: Polysemy

Polysemy—the capacity for a word to have multiple meanings—is a classic challenge in NLP. Consider the word "bank":
1. "I sat on the river **bank**."
2. "I deposited money at the **bank**."

Standard sentence embeddings pool the entire sentence into a single vector. While this captures the overall semantic meaning, it doesn't explicitly isolate the representation of the *target word* itself.

## The Method: Word-Specific Pooling & Hard Negatives

The new WordNet training example in `sentence-transformers` introduces a novel approach to this problem.

### 1. Word-Targeted Input Format
The model expects inputs in a specific format that highlights the target word:
`'<word>': <text>`

For example:
- `'bank': I sat on the river bank.`
- `'bank': A sloping land (especially the slope beside a body of water).`

### 2. Custom Word Pooling
Instead of averaging all tokens in the sentence (Mean Pooling), the model uses a **WordPooling** layer. This layer identifies the tokens corresponding to the target word (e.g., "bank") and only averages those specific tokens. This forces the embedding to represent the *word in context*, rather than the whole sentence context.

### 3. Inter-Word Negatives (Hard Negatives)
The most crucial part of the training strategy is the use of **inter-word negatives**. 

When training with InfoNCE loss (via `MultipleNegativesRankingLoss`), we typically use other samples in the batch as negatives. However, for word sense disambiguation, the hardest negatives are *other definitions of the same word*.

If the anchor is:
> `'bank': I sat on the river bank.`

And the positive is:
> `'bank': A sloping land (especially the slope beside a body of water).`

The model explicitly provides other definitions of "bank" as negative samples:
> Negative: `'bank': A financial institution that accepts deposits and channels the money into lending activities.`

This forces the model to push apart the representations of "bank" (river) and "bank" (financial), effectively learning distinct subspaces for different word senses.

## Implementation Highlights

The implementation uses a custom `WordSenseTransformer` class that overrides the `tokenize` method to generate a `word_mask`. This mask is then used by the `WordPooling` layer.

```python
class WordPooling(nn.Module):
    def forward(self, features):
        # ...
        # Sum embeddings where word_mask is 1 (target word tokens only)
        sum_embeddings = torch.sum(token_embeddings * word_mask_expanded, dim=1)
        # ...
        return {"sentence_embedding": sentence_embeddings}
```

## Results

By training on the `marksverdhei/wordnet-definitions-en-2021` dataset, the model learns to map usage examples (anchors) to their correct definitions (positives) while distinguishing them from incorrect definitions of the *same word*.

This method is particularly useful for:
- **Word Sense Disambiguation (WSD)**: Determining which sense of a word is used.
- **Lexical Substitution**: Finding synonyms that fit a specific context.
- **Fine-grained Semantic Search**: Retrieving definitions or documents matching a specific nuance of a query term.

Check out the full example code in `examples/sentence_transformer/training/wordnet/train_wordnet_infonce.py` to try it yourself!
