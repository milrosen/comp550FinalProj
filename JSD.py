# The Jensen-Shannon Divergence (JSD) is a symmetric measure of similarity between two probability distributions.
# It ranges from 0 to 1 (in base-2 logarithms):
#
# - JSD = 0:
#   - Indicates the two distributions are identical (perfect overlap in vocabulary usage between languages).
#
# - JSD > 0:
#   - Indicates some divergence between the distributions.
#   - The larger the value, the more dissimilar the distributions are in terms of token usage.
#
# - JSD = 1:
#   - Indicates the two distributions are completely disjoint (no shared tokens at all).
#
# Interpretation in the context of tokenizers:
# - Low JSD (e.g., close to 0):
#   - Suggests high overlap in token representation across languages.
#   - This is beneficial for cross-lingual tasks like translation or retrieval, where shared tokens can help transfer learning.
#
# - High JSD (e.g., close to 1):
#   - Suggests low overlap or distinct token representations between languages.
#   - This may be advantageous for language-specific tasks, as it reduces token ambiguity.

import numpy as np
from scipy.spatial.distance import jensenshannon
from transformers import PreTrainedTokenizerFast
from collections import Counter

def compute_jsd_between_tokenizers(tokenizer_path_1, tokenizer_path_2, corpus_1, corpus_2, max_length=128):
    """
    Computes the Jensen-Shannon Divergence (JSD) between two tokenizers given their respective corpora.

    Args:
        tokenizer_path_1 (str): Path to the first tokenizer JSON file.
        tokenizer_path_2 (str): Path to the second tokenizer JSON file.
        corpus_1 (list of str): Corpus for the first tokenizer.
        corpus_2 (list of str): Corpus for the second tokenizer.
        max_length (int): Maximum length for tokenization (default: 128).

    Returns:
        float: The computed JSD value between the two token distributions.
    """
    # Load tokenizers
    tokenizer_1 = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path_1)
    tokenizer_2 = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path_2)

    # Add padding tokens if not already defined
    if tokenizer_1.pad_token is None:
        tokenizer_1.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer_2.pad_token is None:
        tokenizer_2.add_special_tokens({'pad_token': '[PAD]'})

    # Tokenize corpora
    tokens_1 = tokenizer_1(corpus_1, truncation=True, padding=True, max_length=max_length)["input_ids"]
    tokens_2 = tokenizer_2(corpus_2, truncation=True, padding=True, max_length=max_length)["input_ids"]

    # Count token frequencies
    freq_1 = Counter([token for seq in tokens_1 for token in seq])
    freq_2 = Counter([token for seq in tokens_2 for token in seq])

    # Normalize frequencies
    vocab_size = max(max(freq_1.keys()), max(freq_2.keys())) + 1
    dist_1 = np.zeros(vocab_size)
    dist_2 = np.zeros(vocab_size)

    for token, freq in freq_1.items():
        dist_1[token] = freq / sum(freq_1.values())
    for token, freq in freq_2.items():
        dist_2[token] = freq / sum(freq_2.values())

    # Compute JSD
    jsd = jensenshannon(dist_1, dist_2, base=2)
    return jsd

# Example usage
if __name__ == "__main__":
    tokenizer_path_1 = "./tokenizers/tokenizer-cc-en.json"
    tokenizer_path_2 = "./tokenizers/tokenizer-cc-de.json"
    corpus_1 = ["This is an example text for English."]
    corpus_2 = ["Dies ist ein Beispieltext für Deutsch."]
    jsd_value = compute_jsd_between_tokenizers(tokenizer_path_1, tokenizer_path_2, corpus_1, corpus_2, max_length=10)
    print("Jensen-Shannon Divergence (JSD):", jsd_value)
