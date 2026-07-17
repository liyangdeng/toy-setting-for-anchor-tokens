#!/usr/bin/env python3
"""Train multilingual MLM models for lexical-overlap experiments.

This script is adapted from src/training/train_multilingual_synset.py.
It is strictly adapted to match the baseline hyperparameters and standard
special token treatment.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Standard list of special tokens matching baseline and semantic setups
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def build_tokenizer(corpus_files):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=SPECIAL_TOKENS, min_frequency=1)
    tokenizer.train([str(path) for path in corpus_files], trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def read_lines(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_dataset(lines, tokenizer, max_length):
    # Standard tokenization that automatically adds [CLS] and [SEP] universally
    encodings = tokenizer(lines, truncation=True, max_length=max_length, padding=False)
    return Dataset.from_dict(encodings)


def plot_loss_history(trainer, output_dir):
    history = trainer.state.log_history
    train_records = [r for r in history if 'loss' in r and 'eval_loss' not in r and 'epoch' in r]
    eval_records  = [r for r in history if 'eval_loss' in r and 'epoch' in r]

    train_epochs = [r['epoch'] for r in train_records]
    train_losses = [r['loss']  for r in train_records]
    eval_epochs  = [r['epoch'] for r in eval_records]
    eval_losses  = [r['eval_loss'] for r in eval_records]

    if not train_losses:
        print('Warning: no training-loss records found.')
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_epochs, train_losses, marker='o', label='Training loss')
    if eval_losses:
        ax.plot(eval_epochs, eval_losses, marker='o', label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MLM loss')
    ax.set_title('Lexical-overlap — training and validation loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    
    out_path = Path(output_dir) / "loss_curve.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss plot saved to {out_path}')


def main():
    parser = argparse.ArgumentParser(description="Train MLM for Lexical Overlap & Anchoring Experiment (Baseline Hyperparams)")
    parser.add_argument("--condition", type=str, required=True, help="Unique experimental condition token name")
    parser.add_argument("--corpus_a", type=str, required=True, help="Path to modified language A corpus")
    parser.add_argument("--corpus_b", type=str, required=True, help="Path to modified language B corpus")
    parser.add_argument("--overlap", type=float, required=True, help="Target overlap percentage (e.g. 2.5, 5.0, 7.5, 10.0)")
    parser.add_argument("--strategy", type=str, required=True, choices=["high", "mid", "low", "none"], help="Anchor selection strategy")
    parser.add_argument("--output_dir", type=str, required=True, help="Output tracking and checkpoint directory")
    
    # Hyperparameters aligned with other scripts (train_multilingual_synset and train_punct)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--dev_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading experimental corpora:\n  A: {args.corpus_a}\n  B: {args.corpus_b}")
    sentences_a = read_lines(args.corpus_a)
    sentences_b = read_lines(args.corpus_b)

    tokenizer = build_tokenizer([args.corpus_a, args.corpus_b])
    print(f"Shared Vocabulary Size: {len(tokenizer)}")

    rng = random.Random(args.seed)
    rng.shuffle(sentences_a)
    rng.shuffle(sentences_b)

    dev_len_a = int(len(sentences_a) * args.dev_frac)
    dev_len_b = int(len(sentences_b) * args.dev_frac)

    train_sents = sentences_a[dev_len_a:] + sentences_b[dev_len_b:]
    dev_sents   = sentences_a[:dev_len_a] + sentences_b[:dev_len_b]

    print(f"Data split done: {len(train_sents)} train sentences, {len(dev_sents)} dev sentences.")

    train_ds = prepare_dataset(train_sents, tokenizer, args.max_length)
    dev_ds   = prepare_dataset(dev_sents, tokenizer, args.max_length)

    # Reverted to smaller configuration (4 layers, 128 hidden size) like other scripts
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
    )

    print("Training initialized model...")
    trainer.train()

    plot_loss_history(trainer, args.output_dir)

    train_loss = trainer.evaluate(train_ds)['eval_loss']
    dev_loss   = trainer.evaluate(dev_ds)['eval_loss']

    print(f"\nFinal train perplexity : {math.exp(train_loss):.2f}")
    print(f"Final dev   perplexity : {math.exp(dev_loss):.2f}")

    # Save model and tokenizer
    final_output_path = out_dir / "final"
    trainer.save_model(str(final_output_path))
    tokenizer.save_pretrained(str(final_output_path))

    # Structured metadata
    metadata = {
        "condition": args.condition,
        "corpus_a": args.corpus_a,
        "corpus_b": args.corpus_b,
        "vocab_size": len(tokenizer),
        "epochs": args.epochs,
        "train_sentences": len(train_sents),
        "dev_sentences": len(dev_sents),
        "language_a_sentences": len(sentences_a),
        "language_b_sentences": len(sentences_b),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "mlm_prob": args.mlm_prob,
        "learning_rate": args.lr,
        "warmup_steps": args.warmup_steps,
        "dev_frac": args.dev_frac,
        "seed": args.seed,
        "train_perplexity": math.exp(train_loss),
        "dev_perplexity": math.exp(dev_loss),
        "lexical_overlap_percentage": args.overlap,
        "frequency_strategy": args.strategy,
    }

    (out_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    print(f"Metadata tracking successfully recorded inside: {args.output_dir}")


if __name__ == "__main__":
    main()
