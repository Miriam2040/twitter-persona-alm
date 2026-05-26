"""
finetune.py — fine-tune GPT-2 on one persona's tweets to create an ALM.

Run:
    python src/finetune.py --persona trump
    python src/finetune.py --persona trump --model gpt2-medium  # better, slower

What it does:
    Continues pretraining GPT-2 on a person's tweets.
    The model learns to predict the next token given previous tokens.
    After training, it has internalized that person's vocabulary, rhythm, and topics.
    We never generate from it — we only use it to MEASURE how surprised it is by new text.

Why GPT-2 and not a bigger model:
    - Runs on CPU/MPS (Apple Silicon) and free Colab GPU
    - 117M params is enough to capture stylistic patterns
    - Fast enough to iterate quickly
    - Can upgrade to gpt2-medium (345M) or phi-3-mini later for the paper
"""

import os
import random
import argparse
import time
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)

MAX_LENGTH = 128    # GPT-2 supports 1024 but tweets are short; 128 covers ~99% of them
EPOCHS     = 1      # 1 epoch for demo; use --epochs 3 for paper-quality model (~65 min)
BATCH_SIZE = 16     # 16 is better MPS utilization than 8 on Apple Silicon
SEED       = 42     # fixed for reproducibility


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"      # Apple Silicon GPU
    return "cpu"


class TweetDataset(Dataset):
    """
    Each tweet is treated as an independent sequence.
    WHY not concatenate tweets into long blocks (common LM trick):
        Perplexity is computed per-tweet at scoring time, so training and scoring
        must use the same unit. If we train on concatenated blocks, the model learns
        cross-tweet context that doesn't exist at scoring time.
    """

    def __init__(self, csv_path: str, tokenizer: GPT2Tokenizer):
        df = pd.read_csv(csv_path)
        self.examples = []

        for text in df["text"].dropna():
            # Append EOS token so the model learns tweet boundaries
            enc = tokenizer(
                text.strip() + tokenizer.eos_token,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze()
            attention_mask = enc["attention_mask"].squeeze()

            # Labels = input_ids for causal LM, but padding positions get -100
            # so they are excluded from the loss computation
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def run(persona: str, model_name: str, epochs: int = EPOCHS, seed: int = SEED):
    # Fix all random seeds for reproducibility
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    print(f"\n=== Fine-tuning: {persona} | model: {model_name} | device: {device} | seed: {seed} ===")

    output_dir = f"models/{persona}"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # GPT-2 has no padding token by default — use EOS as pad
    # WHY: the tokenizer needs a pad token to produce fixed-length batches.
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = TweetDataset(f"data/processed/{persona}_train.csv", tokenizer)
    eval_dataset  = TweetDataset(f"data/processed/{persona}_eval.csv",  tokenizer)
    steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
    total_steps = steps_per_epoch * epochs
    print(f"  Train tweets : {len(train_dataset):,} | Eval tweets: {len(eval_dataset):,}")
    print(f"  Steps/epoch  : {steps_per_epoch:,} | Total steps: {total_steps:,}")
    print(f"  ETA (MPS M5) : ~{total_steps // 120} min  (1.7–2 steps/sec)")
    t0 = time.time()

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=200,
        report_to="none",               # disable W&B / TensorBoard
        fp16=torch.cuda.is_available(), # half precision only on CUDA
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    elapsed = time.time() - t0
    print(f"  Training took: {elapsed/60:.1f} min")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Model saved → {output_dir}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="trump", help="persona name")
    parser.add_argument("--model",   default="gpt2",  help="base model: gpt2 or gpt2-medium")
    parser.add_argument("--epochs",  default=EPOCHS,  type=int, help="training epochs")
    parser.add_argument("--seed",    default=SEED,    type=int, help="random seed")
    args = parser.parse_args()
    run(args.persona, args.model, args.epochs, args.seed)
