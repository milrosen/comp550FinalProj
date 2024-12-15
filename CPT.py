
# Characters Per Token (CPT) measures the average number of characters in each token produced by the tokenizer.
# It is calculated as: CPT = Total_Characters / Total_Tokens
#
# Interpretation:
# - Low CPT (e.g., close to 1):
#   - Indicates that the tokenizer splits the text into many short tokens.
#   - This may lead to inefficiency in downstream tasks, as models need to process more tokens for the same text.
#   - However, it can be beneficial for languages with complex morphology, as shorter tokens can capture subword patterns.
#
# - High CPT (e.g., > 5 or 6):
#   - Indicates that the tokenizer produces longer tokens on average.
#   - This may improve efficiency (fewer tokens to process) but risks losing granularity for languages with rich inflections or morphology.
#
# Use Case:
# - An optimal CPT depends on the language and task:
#   - For tasks like machine translation, moderate CPT can help balance between token granularity and computational efficiency.
#   - For languages with compound words (e.g., German), lower CPT may be more suitable.
def compute_characters_per_token(tokenizer, corpus):
    """
    Computes the Characters Per Token (CPT) for a tokenizer given a corpus.

    Args:
        tokenizer (PreTrainedTokenizerFast): A tokenizer object already prepared.
        corpus (list of str): List of sentences for evaluation.

    Returns:
        float: The computed Characters Per Token (CPT) value.
    """
    # Tokenize corpus
    tokens = tokenizer(corpus, truncation=True, padding=True, max_length=128)

    # Compute total characters and tokens
    total_chars = sum(len(text) for text in corpus)
    total_tokens = sum(len(token_seq) for token_seq in tokens["input_ids"])

    # Calculate CPT
    cpt = total_chars / total_tokens
    return cpt
