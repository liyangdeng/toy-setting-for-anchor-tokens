#!/usr/bin/env python3
"""Train multilingual MLM models for lexical-overlap & anchoring experiments.

This script is strictly adapted from the format used by the other experiments
(train_semantic_overlap_multilingual, train_punct, train_special).
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
    encodings = tokenizer(lines, truncation=True, max_length=max_length)
    return Dataset.from_dict(encodings)


def plot_loss_history(trainer, output_dir):
    history = trainer.state.log_history
    train_loss = [x['loss'] for x in history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    
    plt.figure(figsize=(8, 5))
    if train_loss:
        plt.plot(train_loss, label='Train Loss')
    if eval_loss:
        plt.plot(eval_loss, label='Eval Loss', linestyle='--')
    plt.xlabel('Epochs / Log Steps')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(output_dir) / "loss_history.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train MLM for Lexical Overlap & Anchoring Experiment")
    parser.add_argument("--condition", type=str, required=True, help="Unique experimental condition token name")
    parser.add_argument("--corpus_a", type=str, required=True, help="Path to modified language A corpus")
    parser.add_argument("--corpus_b", type=str, required=True, help="Path to modified language B corpus")
    parser.add_argument("--overlap", type=float, required=True, help="Target overlap percentage (e.g. 2.5, 5.0, 7.5, 10.0)")
    parser.add_argument("--strategy", type=str, required=True, choices=["high", "mid", "low", "none"], help="Anchor selection strategy")
    parser.add_argument("--output_dir", type=str, required=True, help="Output tracking and checkpoint directory")
    
    # Standard SWP hyperparameters matched with colleagues' setups
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_length", type=int, default=128)
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

    # Standard configuration matching Dufter & Schütze's small setting inside swp_orga
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(config)

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

    train_perplexity = math.exp(train_loss) if train_loss < 20 else float('inf')
    dev_perplexity   = math.exp(dev_loss) if dev_loss < 20 else float('inf')

    print(f"\nFinal train perplexity : {train_perplexity:.2f}")
    print(f"Final dev   perplexity : {dev_perplexity:.2f}")

    # Save tokenizer and model bundle
    final_output_path = out_dir / "final"
    tokenizer.save_pretrained(str(final_output_path))
    model.save_pretrained(str(final_output_path))

    # Structured metadata tailored exactly to match the other pipeline tracking formats
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
        "train_perplexity": train_perplexity,
        "dev_perplexity": dev_perplexity,
        "lexical_overlap_percentage": args.overlap,
        "frequency_strategy": args.strategy,
    }

    (out_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    print(f"Metadata tracking successfully recorded inside: {args.output_dir}")


if __name__ == "__main__":
    main()
