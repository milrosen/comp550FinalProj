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
from transformers import AutoTokenizer
from scipy.spatial.distance import jensenshannon

### LOADING METHODS ###
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

def extract_language_from_filename(filename):
    """
    Extracts the language code from a tokenizer filename.
    """
    match = re.search(r"tokenizer-cc-([a-zA-Z_]+)\.json", filename)
    return match.group(1) if match else None


### AR and CPT ###
def run_ar_and_cpt_on_tokenizers(tokenizers_dir, output_file, num_sentences=1000):
    """
    Runs AR and CPT calculations on all tokenizers in the specified directory.
    Matches the tokenizer's language to the appropriate corpus.
    """
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
                    print(f"[ERROR] Missing corpus for language '{language}', logging AR and CPT as -1.")
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
                    print(f"[ERROR] Failed to load tokenizer '{file}': {e}")
                    results[file] = {
                        "tokenizer_path": tokenizer_path,
                        "language": language,
                        "average_rank": -1,
                        "characters_per_token": -1,
                        "num_sentences": num_sentences
                    }
                    continue

                # Check and fix missing `unk_id`
                try:
                    ar = compute_average_rank(tokenizer, corpus)
                    cpt = compute_characters_per_token(tokenizer, corpus)
                    print(f"[INFO] AR: {ar:.2f}, CPT: {cpt:.2f}")
                except Exception as e:
                    if "unk_id is missing" in str(e):
                        print(f"[ERROR] Missing `unk_id` in tokenizer '{file}', attempting to fix...")
                        fixed = fix_tokenizer_json_unk_id(tokenizer_path)
                        if fixed:
                            print(f"[INFO] Retrying with fixed tokenizer '{file}'...")
                            tokenizer = load_and_prepare_tokenizer(tokenizer_path)
                            try:
                                ar = compute_average_rank(tokenizer, corpus)
                                cpt = compute_characters_per_token(tokenizer, corpus)
                                print(f"[INFO] AR: {ar:.2f}, CPT: {cpt:.2f}")
                            except Exception as retry_error:
                                print(f"[ERROR] Retry failed for tokenizer '{file}': {retry_error}")
                                ar, cpt = -1, -1
                        else:
                            print(f"[ERROR] Failed to fix tokenizer '{file}'.")
                            ar, cpt = -1, -1
                    else:
                        print(f"[ERROR] Failed to compute AR/CPT for '{file}': {e}")
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

def run_ar_and_cpt_on_pretrained_tokenizers(pretrained_tokenizers, output_file, num_sentences=1000):
    """
    Runs AR and CPT calculations on pre-trained tokenizers using the specified corpus.
    """
    results = {}

    # Process each pre-trained tokenizer
    for tokenizer_name, language in pretrained_tokenizers.items():
        print(f"\n[INFO] Processing tokenizer: {tokenizer_name} (Language: {language})")

        # Load the corpus
        corpus = load_cc100_corpus(language, num_sentences)
        if not corpus:
            print(f"[ERROR] Missing corpus for language '{language}', logging AR and CPT as -1.")
            results[tokenizer_name] = {
                "average_rank": -1,
                "characters_per_token": -1,
                "num_sentences": num_sentences
            }
            continue

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            # Add padding token for GPT-2 and similar models
            if tokenizer.pad_token is None:
                print(f"[INFO] Adding PAD token to tokenizer '{tokenizer_name}'...")
                tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token or "[PAD]"})
                print(f"[INFO] PAD token added: {tokenizer.pad_token}")

        except Exception as e:
            print(f"[ERROR] Failed to load tokenizer '{tokenizer_name}': {e}")
            results[tokenizer_name] = {
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
            print(f"[ERROR] Failed to compute AR/CPT for '{tokenizer_name}': {e}")
            ar, cpt = -1, -1

        # Save results
        results[tokenizer_name] = {
            "average_rank": round(ar, 2) if ar != -1 else -1,
            "characters_per_token": round(cpt, 2) if cpt != -1 else -1,
            "num_sentences": num_sentences
        }

    # Write results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[INFO] Results saved to {output_file}")

def run_ar_and_cpt_on_selected_tokenizers(output_file, num_sentences=1000):
    """
    Runs AR and CPT calculations on a predefined list of tokenizers.
    Matches the tokenizer's language to the appropriate corpus.
    """
    # Define the selected tokenizers
    selected_tokenizers = [
        {"tokenizer_path": "./tokenizers/tokenizer-cc-all.json", "language": "en"},
        {"tokenizer_path": "./tokenizers/tokenizer-cc-greek_script_family.json", "language": "el"},
        {"tokenizer_path": "./tokenizers/tokenizer-cc-northern_brahmic_script_family.json", "language": "hi"},
        {"tokenizer_path": "./tokenizers/tokmix/en-fr-vi-tokenizer-cc.json", "language": "en"},
        {"tokenizer_path": "./tokenizers/tokenizer-cc-zh-Hans.json", "language": "zh"},
    ]

    results = {}

    # Iterate through the selected tokenizers
    for tokenizer_info in selected_tokenizers:
        tokenizer_path = tokenizer_info["tokenizer_path"]
        language = tokenizer_info["language"]

        print(f"\n[INFO] Processing tokenizer: {tokenizer_path} (Language: {language})")

        # Load the corpus for the detected language
        corpus = load_cc100_corpus(language, num_sentences)
        if not corpus:
            print(f"[ERROR] Missing corpus for language '{language}', logging AR and CPT as -1.")
            results[tokenizer_path] = {
                "tokenizer_path": tokenizer_path,
                "language": language,
                "average_rank": -1,
                "characters_per_token": -1,
                "num_sentences": num_sentences,
            }
            continue

        # Load and prepare the tokenizer
        try:
            tokenizer = load_and_prepare_tokenizer(tokenizer_path)
        except Exception as e:
            print(f"[ERROR] Failed to load tokenizer '{tokenizer_path}': {e}")
            results[tokenizer_path] = {
                "tokenizer_path": tokenizer_path,
                "language": language,
                "average_rank": -1,
                "characters_per_token": -1,
                "num_sentences": num_sentences,
            }
            continue

        # Check and fix missing `unk_id`
        try:
            ar = compute_average_rank(tokenizer, corpus)
            cpt = compute_characters_per_token(tokenizer, corpus)
            print(f"[INFO] AR: {ar:.2f}, CPT: {cpt:.2f}")
        except Exception as e:
            if "unk_id is missing" in str(e):
                print(f"[ERROR] Missing `unk_id` in tokenizer '{tokenizer_path}', attempting to fix...")
                fixed = fix_tokenizer_json_unk_id(tokenizer_path)
                if fixed:
                    print(f"[INFO] Retrying with fixed tokenizer '{tokenizer_path}'...")
                    tokenizer = load_and_prepare_tokenizer(tokenizer_path)
                    try:
                        ar = compute_average_rank(tokenizer, corpus)
                        cpt = compute_characters_per_token(tokenizer, corpus)
                        print(f"[INFO] AR: {ar:.2f}, CPT: {cpt:.2f}")
                    except Exception as retry_error:
                        print(f"[ERROR] Retry failed for tokenizer '{tokenizer_path}': {retry_error}")
                        ar, cpt = -1, -1
                else:
                    print(f"[ERROR] Failed to fix tokenizer '{tokenizer_path}'.")
                    ar, cpt = -1, -1
            else:
                print(f"[ERROR] Failed to compute AR/CPT for '{tokenizer_path}': {e}")
                ar, cpt = -1, -1

        # Save results
        results[tokenizer_path] = {
            "tokenizer_path": tokenizer_path,
            "language": language,
            "average_rank": round(ar, 2) if ar != -1 else -1,
            "characters_per_token": round(cpt, 2) if cpt != -1 else -1,
            "num_sentences": num_sentences,
        }

    # Write results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[INFO] Results saved to {output_file}")


### JSD ###
def compute_jsd_for_language_pair(tokenizer, corpus_1, corpus_2):
    """
    Computes the Jensen-Shannon Divergence (JSD) between token distributions of two corpora using the same tokenizer.
    """
    try:
        # Tokenize both corpora
        tokens_1 = tokenizer(corpus_1, truncation=True, padding=True, max_length=128)["input_ids"]
        tokens_2 = tokenizer(corpus_2, truncation=True, padding=True, max_length=128)["input_ids"]

        # Count token frequencies
        freq_1 = Counter([token for seq in tokens_1 for token in seq])
        freq_2 = Counter([token for seq in tokens_2 for token in seq])

        # Normalize token distributions
        vocab_size = max(max(freq_1.keys(), default=0), max(freq_2.keys(), default=0)) + 1
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

def run_jsd_for_language_pairs_with_local_tokenizers(tokenizers_dir, output_file, language_pairs, num_sentences=1000):
    """
    Runs JSD calculations between pairs of languages using all local tokenizers.
    """
    results = {}

    # Scan through all tokenizer files
    print("\n[INFO] Scanning for local tokenizers...")
    for root, _, files in os.walk(tokenizers_dir):
        if "partial" in root:  # Skip the 'partial' folder
            continue

        for file in files:
            if file.endswith(".json"):
                tokenizer_path = os.path.join(root, file)
                print(f"\n[INFO] Processing tokenizer: {file}")

                # Load and prepare the tokenizer
                try:
                    tokenizer = load_and_prepare_tokenizer(tokenizer_path)
                except Exception as e:
                    print(f"[ERROR] Failed to load tokenizer '{file}': {e}")
                    continue

                # Compute JSD for each language pair
                for lang1, lang2 in language_pairs:
                    print(f"[INFO] Computing JSD for languages '{lang1}' and '{lang2}' using tokenizer '{file}'...")

                    # Load corpora on demand
                    corpus_1 = load_cc100_corpus(lang1, num_sentences)
                    corpus_2 = load_cc100_corpus(lang2, num_sentences)

                    if not corpus_1 or not corpus_2:
                        print(f"[ERROR] Missing corpora for languages '{lang1}' and '{lang2}'.")
                        jsd = -1
                    else:
                        jsd = compute_jsd_for_language_pair(tokenizer, corpus_1, corpus_2)

                    # Save results
                    pair_key = f"{file}__{lang1}__vs__{lang2}"
                    results[pair_key] = {
                        "tokenizer": file,
                        "language_1": lang1,
                        "language_2": lang2,
                        "jsd": jsd
                    }
                    print(f"[INFO] JSD: {jsd}")

    # Write results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[INFO] JSD results saved to {output_file}")



def fix_tokenizer_json_unk_id(tokenizer_path):
    """
    Fixes the tokenizer JSON file by adding `unk_id` to the model's vocab if missing.
    """
    try:
        # Load the tokenizer JSON file
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        # Check and set `unk_id`
        if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
            vocab = tokenizer_data["model"]["vocab"]
            if "unk_id" not in vocab:
                print(f"[INFO] Adding `unk_id` to tokenizer '{tokenizer_path}'...")
                vocab["unk_id"] = 1  # Default `unk_id` value

                # Save the updated tokenizer JSON
                with open(tokenizer_path, "w", encoding="utf-8") as f:
                    json.dump(tokenizer_data, f, indent=4)
                print(f"[INFO] Successfully added `unk_id` to tokenizer '{tokenizer_path}'.")
                return True
            else:
                print(f"[INFO] `unk_id` already exists in tokenizer '{tokenizer_path}'.")
                return True
        else:
            print(f"[ERROR] Invalid tokenizer JSON structure in '{tokenizer_path}'.")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to fix tokenizer '{tokenizer_path}': {e}")
        return False


def run_jsd_for_language_pairs_with_pretrained_tokenizers(pretrained_tokenizers, output_file, language_pairs, num_sentences=1000):
    """
    Runs JSD calculations between pairs of languages using pre-trained tokenizers.
    """
    results = {}

    # Process each pre-trained tokenizer
    print("\n[INFO] Processing pre-trained tokenizers...")
    for tokenizer_name in pretrained_tokenizers:
        print(f"\n[INFO] Processing tokenizer: {tokenizer_name}")

        # Load the tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            # Add padding token if missing (e.g., for GPT-2)
            if tokenizer.pad_token is None:
                print(f"[INFO] Adding PAD token to tokenizer '{tokenizer_name}'...")
                tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token or "[PAD]"})
        except Exception as e:
            print(f"[ERROR] Failed to load tokenizer '{tokenizer_name}': {e}")
            continue

        # Compute JSD for each language pair
        for lang1, lang2 in language_pairs:
            print(f"[INFO] Computing JSD for languages '{lang1}' and '{lang2}' using tokenizer '{tokenizer_name}'...")

            # Load corpora on demand
            corpus_1 = load_cc100_corpus(lang1, num_sentences)
            corpus_2 = load_cc100_corpus(lang2, num_sentences)

            if not corpus_1 or not corpus_2:
                print(f"[ERROR] Missing corpora for languages '{lang1}' and '{lang2}'.")
                jsd = -1
            else:
                # Compute JSD
                try:
                    jsd = compute_jsd_for_language_pair(tokenizer, corpus_1, corpus_2)
                except Exception as e:
                    print(f"[ERROR] Failed to compute JSD for '{tokenizer_name}' with languages '{lang1}' and '{lang2}': {e}")
                    jsd = -1

            # Save results
            pair_key = f"{tokenizer_name}__{lang1}__vs__{lang2}"
            results[pair_key] = {
                "tokenizer": tokenizer_name,
                "language_1": lang1,
                "language_2": lang2,
                "jsd": jsd
            }
            print(f"[INFO] JSD: {jsd}")

    # Write results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[INFO] JSD results saved to {output_file}")



### MAIN ###
if __name__ == "__main__":
    num_sentences = 10000  # Number of sentences to process

    pretrained_tokenizers = {
        "xlm-roberta-base": "en",          # XLM-R tokenizer
        "xlm-mlm-enfr-1024": "en",         # XLM English-French tokenizer
        "xlm-mlm-17-1280": "en",           # XLM multilingual tokenizer
        "bert-base-uncased": "en",         # BERT base tokenizer
        "bert-base-multilingual-cased": "en",  # Multilingual BERT tokenizer
        "gpt2": "en",                      # GPT-2 tokenizer
        "facebook/bart-base": "en"         # BART base tokenizer
    }

    pretrained_tokenizers_jsd = [
        "xlm-roberta-base",             # XLM-R base tokenizer
        "xlm-mlm-enfr-1024",            # XLM English-French tokenizer
        "xlm-mlm-17-1280",              # XLM multilingual tokenizer
        "bert-base-uncased",            # BERT base tokenizer
        "bert-base-multilingual-cased", # Multilingual BERT tokenizer
        "gpt2",                         # GPT-2 tokenizer
        "facebook/bart-base"            # BART tokenizer
    ]

    language_pairs = [
    ("hi", "gu"),  # Hindi vs Gujarati: Indic languages with distinct scripts and regional diversity.
    ("hi", "ur"),  # Hindi vs Urdu: Same linguistic roots but different scripts (Devanagari vs Perso-Arabic).
    ("hi", "ar"),  # Hindi vs Arabic: Indic vs Semitic languages to compare unrelated linguistic families.
    ("hi", "el"),  # Hindi vs Greek: Indic vs Hellenic languages to evaluate historical script diversity.
    ("gu", "mr"),  # Gujarati vs Marathi: Both Indic languages with unique phonetics and writing systems.
    ("gu", "ta"),  # Gujarati vs Tamil: Indic vs Dravidian languages to explore South Asian script variation.
    ("hi", "sw"),  # Hindi vs Swahili: Indic vs Bantu languages to measure geographical and linguistic contrast.
    ("gu", "vi"),  # Gujarati vs Vietnamese: Indic vs Austroasiatic languages, distinct scripts and syntax.
    ("en", "hi"),  # English vs Hindi: Popular multilingual setup for many NLP models.
    ("en", "es"),  # English vs Spanish: Germanic vs Romance languages, both widely spoken.
    ("gu", "ru"),  # Gujarati vs Russian: Indic vs Slavic languages for exploring Cyrillic and Devanagari scripts.
    ("hi", "th"),  # Hindi vs Thai: Indic vs Tai-Kadai languages for contrasting script and tonal structure.
    ("fr", "de"),  # French vs German: Indo-European languages but from different branches (Romance vs Germanic).
    ("hi", "de"),  # Hindi vs German: Indic vs Germanic languages to compare structural and script differences.
    ]

    # Run AR and CPT calculations
    run_ar_and_cpt_on_tokenizers("./tokenizers", "local_tokenizer_metrics_ar_cpt.json", num_sentences)
    #run_ar_and_cpt_on_selected_tokenizers("selected_local_tokenizer_metrics_ar_cpt.json", num_sentences)

    # Run AR and CPT calculations for pre-trained tokenizers
    #run_ar_and_cpt_on_pretrained_tokenizers(pretrained_tokenizers, "pretrained_tokenizer_metrics_ar_cpt.json", num_sentences)

    #output_file = "tokenizer_jsd_language_pairs.json"  # Output JSON file
  
    # Run JSD calculations for all local tokenizers and language pairs
    run_jsd_for_language_pairs_with_local_tokenizers("./tokenizers","local_tokenizer_jsd_language_pairs.json" , language_pairs, num_sentences)
    #run_jsd_for_language_pairs_with_pretrained_tokenizers(pretrained_tokenizers, "pretrained_tokenizer_jsd_language_pairs.json", language_pairs, num_sentences)
