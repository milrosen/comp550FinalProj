import os
import json
import re
from datasets import load_dataset
from AR import compute_average_rank
from CPT import compute_characters_per_token
from transformers import PreTrainedTokenizerFast
from scipy.spatial.distance import jensenshannon
import numpy as np
from collections import Counter

def load_and_prepare_tokenizer(tokenizer_path):
    """
    Loads a tokenizer and ensures special tokens are defined.
    """
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # Add special tokens if not already defined
    special_tokens = {"unk_token": "[UNK]", "pad_token": "[PAD]"}
    for token_name, token_value in special_tokens.items():
        if getattr(tokenizer, token_name) is None:
            tokenizer.add_special_tokens({token_name: token_value})
    
    return tokenizer

def load_cc100_corpus(language, num_sentences=1000):
    """
    Loads a specified number of sentences from the CC100 dataset for a given language.
    """
    print(f"[INFO] Loading {num_sentences} sentences from the CC100 dataset for language '{language}'...")
    try:
        dataset = load_dataset("cc100", lang=language, split="train", streaming=True)
        corpus = [example["text"] for example in dataset.take(num_sentences)]
        print(f"[INFO] Loaded {len(corpus)} sentences.")
        return corpus
    except Exception as e:
        print(f"[ERROR] Could not load corpus for language '{language}': {e}")
        return []

def get_language_for_tokenizer(tokenizer_name):
    """
    Maps tokenizer filenames to their respective languages or corpora.
    """
    language_mapping = {
        "tokenizer-cc-zh-Hans.json": "zh",  # Simplified Chinese
        "tokenizer-cc-northern_brahmic_script_family.json": "hi",  # Hindi
        "tokenizer-cc-greek_script_family.json": "el",  # Greek
        "tokenizer-cc-arabic_script_family.json": "ar",  # Arabic
        "tokenizer-cc-all.json": "en",  # English
        "en-fr-vi-tokenizer-cc.json": "en"  # English for tokmix tokenizer
    }
    return language_mapping.get(tokenizer_name, None)

def run_ar_and_cpt_on_tokenizers(tokenizers_dir, output_file, num_sentences=1000):
    """
    Runs AR and CPT calculations on all tokenizers in the specified directory.
    Matches the tokenizer's language to the appropriate corpus.
    """
    # Prepare results dictionary
    results = {}

    # Iterate through all tokenizer files
    print("\n[INFO] Scanning for tokenizers...")
    for root, _, files in os.walk(tokenizers_dir):
        if "partial" in root:  # Skip the 'partial' folder
            continue

        for file in files:
            if file.endswith(".json"):
                tokenizer_path = os.path.join(root, file)
                language = get_language_for_tokenizer(file) or extract_language_from_filename(file)

                if not language:
                    print(f"[WARNING] Could not determine language for tokenizer: {file}")
                    continue

                print(f"\n[INFO] Processing tokenizer: {file} (Language: {language})")

                # Load the corpus for the detected language
                corpus = load_cc100_corpus(language, num_sentences)
                if not corpus:
                    print(f"[ERROR] Missing corpus for {file}, logging AR and CPT as -1.")
                    results[file] = {
                        "tokenizer_path": tokenizer_path,
                        "language": language,
                        "average_rank": -1,
                        "characters_per_token": -1,
                        "num_sentences": num_sentences
                    }
                    continue

                # Load and prepare the tokenizer
                try:
                    tokenizer = load_and_prepare_tokenizer(tokenizer_path)
                except Exception as e:
                    print(f"[ERROR] Failed to load tokenizer {file}: {e}")
                    results[file] = {
                        "tokenizer_path": tokenizer_path,
                        "language": language,
                        "average_rank": -1,
                        "characters_per_token": -1,
                        "num_sentences": num_sentences
                    }
                    continue

                # Compute AR and CPT
                try:
                    ar = compute_average_rank(tokenizer, corpus)
                    cpt = compute_characters_per_token(tokenizer, corpus)
                    print(f"[INFO] AR: {ar:.2f}, CPT: {cpt:.2f}")
                except Exception as e:
                    print(f"[ERROR] Failed to compute AR/CPT for {file}: {e}")
                    ar, cpt = -1, -1

                # Save results
                results[file] = {
                    "tokenizer_path": tokenizer_path,
                    "language": language,
                    "average_rank": round(ar, 2) if ar != -1 else -1,
                    "characters_per_token": round(cpt, 2) if cpt != -1 else -1,
                    "num_sentences": num_sentences
                }

    # Write results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[INFO] Results saved to {output_file}")

def compute_jsd_between_tokenizers(tokenizer_1, tokenizer_2, corpus_1, corpus_2):
    """
    Computes the Jensen-Shannon Divergence (JSD) between two tokenizers given their respective corpora.
    """
    try:
        # Tokenize corpora
        tokens_1 = tokenizer_1(corpus_1, truncation=True, padding=True, max_length=128)["input_ids"]
        tokens_2 = tokenizer_2(corpus_2, truncation=True, padding=True, max_length=128)["input_ids"]

        # Count token frequencies
        freq_1 = Counter([token for seq in tokens_1 for token in seq])
        freq_2 = Counter([token for seq in tokens_2 for token in seq])

        # Normalize distributions
        vocab_size = max(max(freq_1.keys()), max(freq_2.keys())) + 1
        dist_1 = np.zeros(vocab_size)
        dist_2 = np.zeros(vocab_size)

        for token, freq in freq_1.items():
            dist_1[token] = freq / sum(freq_1.values())
        for token, freq in freq_2.items():
            dist_2[token] = freq / sum(freq_2.values())

        # Compute JSD
        jsd = jensenshannon(dist_1, dist_2, base=2)
        return round(jsd, 4)
    except Exception as e:
        print(f"[ERROR] Failed to compute JSD: {e}")
        return -1

def run_jsd_on_tokenizer_pairs(tokenizers_dir, output_file, num_sentences=1000):
    """
    Runs JSD calculations between all pairs of tokenizers.
    """
    results = {}
    tokenizers = []
    tokenizer_files = []

    # Load tokenizers
    print("\n[INFO] Loading all tokenizers...")
    for root, _, files in os.walk(tokenizers_dir):
        if "partial" in root:  # Skip the 'partial' folder
            continue

        for file in files:
            if file.endswith(".json"):
                tokenizer_path = os.path.join(root, file)
                language = get_language_for_tokenizer(file) or extract_language_from_filename(file)

                if not language:
                    print(f"[WARNING] Could not determine language for tokenizer: {file}")
                    continue

                try:
                    tokenizer = load_and_prepare_tokenizer(tokenizer_path)
                    corpus = load_cc100_corpus(language, num_sentences)
                    if corpus:
                        tokenizers.append((file, tokenizer, corpus))
                        tokenizer_files.append(file)
                except Exception as e:
                    print(f"[ERROR] Failed to load tokenizer {file}: {e}")
                    continue

    # Compute JSD for all pairs of tokenizers
    print("\n[INFO] Calculating JSD for all pairs of tokenizers...")
    for i, (name_1, tokenizer_1, corpus_1) in enumerate(tokenizers):
        for j, (name_2, tokenizer_2, corpus_2) in enumerate(tokenizers):
            if i >= j:
                continue  # Avoid duplicate pairs

            print(f"[INFO] Computing JSD between {name_1} and {name_2}...")
            jsd = compute_jsd_between_tokenizers(tokenizer_1, tokenizer_2, corpus_1, corpus_2)

            # Save results
            pair_key = f"{name_1}__vs__{name_2}"
            results[pair_key] = {
                "tokenizer_1": name_1,
                "tokenizer_2": name_2,
                "jsd": jsd
            }
            print(f"[INFO] JSD: {jsd}")

    # Write results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[INFO] JSD results saved to {output_file}")


def extract_language_from_filename(filename):
    """
    Extracts the language code from a tokenizer filename.
    """
    match = re.search(r"tokenizer-cc-([a-zA-Z_]+)\.json", filename)
    return match.group(1) if match else None

if __name__ == "__main__":
    tokenizers_dir = "./tokenizers"  # Path to the tokenizers directory
    ar_cpt_output_file = "tokenizer_metrics_ar_cpt.json"  # Output JSON file for AR and CPT
    jsd_output_file = "tokenizer_metrics_jsd.json"  # Output JSON file for JSD
    num_sentences = 1000  # Number of sentences to process

    # Run AR and CPT calculations
    #run_ar_and_cpt_on_tokenizers(tokenizers_dir, ar_cpt_output_file, num_sentences)

    # Run JSD calculations
    run_jsd_on_tokenizer_pairs(tokenizers_dir, jsd_output_file, num_sentences)
