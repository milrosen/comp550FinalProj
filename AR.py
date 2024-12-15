# Average Rank (AR) measures how much of the tokenizer's vocabulary is effectively allocated to a specific language.
# It is calculated as a weighted average of token ranks, based on their frequency in a monolingual corpus:
# AR = Σ (Rank_of_Token * Token_Probability)
#
# Interpretation:
# - Low AR:
#   - Indicates that many of the most frequently used tokens in the language are placed in the high-priority range of the vocabulary.
#   - This implies good vocabulary allocation, making the tokenizer efficient for this language.
#
# - High AR:
#   - Indicates that frequently used tokens are lower-priority in the vocabulary (higher ranks).
#   - This may lead to poor performance for tasks involving this language, as its key tokens are less accessible.
#
# Use Case:
# - AR is especially important for multilingual tokenizers:
#   - If AR is high for a language, it suggests underrepresentation in the vocabulary.
#   - Balancing AR across languages ensures fair allocation of vocabulary space, essential for multilingual NLP tasks.

import json
from collections import Counter
from transformers import PreTrainedTokenizerFast

import json
from collections import Counter

def compute_average_rank(tokenizer, corpus):
    """
    Computes the Average Rank (AR) for a tokenizer given a corpus.
    
    Args:
        tokenizer (PreTrainedTokenizerFast): A tokenizer object already prepared.
        corpus (list of str): List of sentences for evaluation.

    Returns:
        float: The computed Average Rank (AR) value.
    """
    # Tokenize corpus
    tokens = tokenizer(corpus, truncation=True, padding=True, max_length=128)

    # Flatten token IDs and count frequencies
    flat_tokens = [token for seq in tokens["input_ids"] for token in seq]
    token_counts = Counter(flat_tokens)

    # Load vocabulary from tokenizer
    vocab = tokenizer.get_vocab()
    
    # Map token IDs to ranks based on frequency
    ranked_vocab = {token_id: rank for rank, (token_id, _) in enumerate(sorted(token_counts.items(), key=lambda x: -x[1]))}

    # Compute AR (Average Rank)
    ar = sum(ranked_vocab.get(token, 0) * freq / len(flat_tokens) for token, freq in token_counts.items())
    return ar
