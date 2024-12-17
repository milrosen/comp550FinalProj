from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling, AutoModelForMaskedLM, TrainingArguments, Trainer, BertConfig
from datasets import load_dataset
from datasets import Dataset
import os
from transformers import TrainerCallback
from tabulate import tabulate
import torch

def load_tokenizer(path_to_tokenizer): 
    """
    Loads a pretrained tokenizer and ensures required special tokens are added.
    """
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path_to_tokenizer)
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]"
    })
    return tokenizer

def tokenize_and_limit(dataset, tokenizer, max_samples):
    """
    Tokenizes and limits the dataset to a fixed number of samples.
    """
    tokenized_samples = []
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        tokenized_samples.append(tokenizer(example["text"], truncation=True, padding="max_length", max_length=128))
    return tokenized_samples

def prepare_datasets(languages, tokenizer_path, max_train_samples, max_val_samples):
    """
    Prepares tokenized training and validation datasets.
    """
    dataset = load_dataset("cc100", lang="en", split="train", streaming=True)
    tokenizer = load_tokenizer(tokenizer_path)

    # Split dataset into train and validation subsets
    train_samples = tokenize_and_limit(dataset, tokenizer, max_train_samples)
    val_samples = tokenize_and_limit(dataset, tokenizer, max_val_samples)

    print(f"Train Dataset Size = {len(train_samples)}, Validation Dataset Size = {len(val_samples)}")
    return train_samples, val_samples, tokenizer


def train_mlm(languages, tokenizer_path, output_dir="mlm_model_large_english", max_train_samples=50_000, max_val_samples=5_000):
    """
    Trains a masked language model (MLM) using a custom tokenizer.
    """
    train_dataset, val_dataset, tokenizer = prepare_datasets(
        languages, tokenizer_path, max_train_samples, max_val_samples
    )

    # Model configuration
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512
    )
    model = AutoModelForMaskedLM.from_config(config)

    # Resize embeddings to match tokenizer size
    model.resize_token_embeddings(len(tokenizer))

    # Use Metal backend for acceleration if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Metal backend for training.")
        model.to(device)
    else:
        print("Metal backend not available. Using CPU.")
        device = torch.device("cpu")

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        logging_steps=500,                 # Log more frequently for shorter runs
        per_device_train_batch_size=32,    # Increase batch size for faster training
        gradient_accumulation_steps=1,     # Simulate larger batch size if needed
        num_train_epochs=6,                # Fit training within one hour
        learning_rate=2e-5,                # Keep learning rate low for convergence
        save_steps=5000,                   # Save less frequently
        save_total_limit=2,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train and save model
    trainer.train()
    trainer.save_model(output_dir)
    
if __name__ == '__main__':
    print("Tokenizer file exists:", os.path.exists("../tokenizers/tokenizer-cc-en.json"))
 
    languages = ["en", "fr", "vi"]
    tokenizer_path = "../tokenizers/tokenizer-cc-en.json"

    train_mlm(languages, tokenizer_path)