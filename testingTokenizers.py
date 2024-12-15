from datasets import load_dataset
from AR import compute_average_rank
from CPT import compute_characters_per_token
from transformers import PreTrainedTokenizerFast


def load_and_prepare_tokenizer(tokenizer_path):
    """
    Loads a tokenizer and ensures special tokens are defined.
    Args:
        tokenizer_path (str): Path to the tokenizer JSON file.
    Returns:
        PreTrainedTokenizerFast: The prepared tokenizer.
    """
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # Add special tokens if not already defined
    special_tokens = {"unk_token": "[UNK]", "pad_token": "[PAD]"}
    for token_name, token_value in special_tokens.items():
        if getattr(tokenizer, token_name) is None:
            tokenizer.add_special_tokens({token_name: token_value})
    
    return tokenizer

def load_cc100_corpus(language, num_sentences=10):
    """
    Loads a specified number of sentences from the CC100 dataset for a given language.
    """
    print(f"[INFO] Loading {num_sentences} sentences from the CC100 dataset for language '{language}'...")
    dataset = load_dataset("cc100", lang=language, split="train", streaming=True)
    corpus = [example["text"] for example in dataset.take(num_sentences)]
    print(f"[INFO] Loaded {len(corpus)} sentences.")
    return corpus

def run_ar_and_cpt(tokenizer_path, language, num_sentences=10000):
    """
    Runs both AR and CPT calculations using the CC100 dataset as the corpus.
    """
    # Load the corpus from CC100
    corpus = load_cc100_corpus(language, num_sentences)

    # Load and prepare the tokenizer
    tokenizer = load_and_prepare_tokenizer(tokenizer_path)

    # Compute Average Rank (AR)
    print("\n[INFO] Calculating Average Rank (AR)...")
    ar = compute_average_rank(tokenizer, corpus)
    print(f"Average Rank (AR): {ar:.2f}")

    # Compute Characters Per Token (CPT)
    print("\n[INFO] Calculating Characters Per Token (CPT)...")
    cpt = compute_characters_per_token(tokenizer, corpus)
    print(f"Characters Per Token (CPT): {cpt:.2f}")

if __name__ == "__main__":
    tokenizer_path = "./tokenizers/tokenizer-cc-all.json"
    language = "en"  # Language code
    num_sentences = 10  # Number of sentences to load from CC100

    run_ar_and_cpt(tokenizer_path, language, num_sentences)
