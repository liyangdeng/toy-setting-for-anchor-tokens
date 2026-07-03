#!/usr/bin/env python3
"""Train multilingual MLM models for semantic-overlap experiments.

This script is adapted from src/training/train_multilingual_synset.py. It keeps
the same model, tokenizer, and train/dev split style, while adding experiment
metadata and optional matched per-epoch sampling for semantic-overlap runs.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-semantic-overlap")

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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def load_corpus(path):
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def split_corpus(sentences, dev_frac, seed):
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * dev_frac))
    return shuffled[n_dev:], shuffled[:n_dev]


def sample_epoch(pool, size, rng):
    if size <= len(pool):
        return rng.sample(pool, size)
    return [rng.choice(pool) for _ in range(size)]


def build_train_sentences(train_a, train_b, epochs, epoch_sample_size, seed):
    if epoch_sample_size is None:
        return train_a + train_b, None

    rng = random.Random(seed + 1000)
    per_epoch_a = epoch_sample_size // 2
    per_epoch_b = epoch_sample_size - per_epoch_a
    sampled = []
    epoch_counts = []
    for _epoch in range(epochs):
        epoch_rng = random.Random(rng.randint(0, 2**31 - 1))
        epoch_sents = sample_epoch(train_a, per_epoch_a, epoch_rng)
        epoch_sents += sample_epoch(train_b, per_epoch_b, epoch_rng)
        epoch_rng.shuffle(epoch_sents)
        sampled.extend(epoch_sents)
        epoch_counts.append(len(epoch_sents))
    return sampled, epoch_counts


def tokenize_dataset(sentences, tokenizer, max_length):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)

    dataset = Dataset.from_dict({"text": sentences})
    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def plot_loss_history(trainer, output_dir, steps_per_logical_epoch=None):
    history = trainer.state.log_history
    train_records = [r for r in history if "loss" in r and "eval_loss" not in r and "epoch" in r]
    eval_records = [r for r in history if "eval_loss" in r and "epoch" in r]
    if steps_per_logical_epoch:
        train_epochs = [r.get("step", 0) / steps_per_logical_epoch for r in train_records]
        eval_epochs = [r.get("step", 0) / steps_per_logical_epoch for r in eval_records]
    else:
        train_epochs = [r["epoch"] for r in train_records]
        eval_epochs = [r["epoch"] for r in eval_records]
    train_losses = [r["loss"] for r in train_records]
    eval_losses = [r["eval_loss"] for r in eval_records]

    if not train_losses:
        print("Warning: no training-loss records found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_epochs, train_losses, marker="o", label="Training loss")
    if eval_losses:
        ax.plot(eval_epochs, eval_losses, marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MLM loss")
    ax.set_title("Semantic-overlap multilingual MLM loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = Path(output_dir) / "loss_curve.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Semantic-overlap multilingual MLM training")
    parser.add_argument("--corpus_a", required=True, help="Language A corpus, e.g. CJK")
    parser.add_argument("--corpus_b", required=True, help="Language B corpus, e.g. Hiragana")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--graph_metadata", help="Condition metadata.json from graph sampling")
    parser.add_argument("--corpus_metadata", help="semantic_corpus_metadata.json for this condition")
    parser.add_argument("--condition", help="Condition name, e.g. overlap_050")
    parser.add_argument("--epoch_sample_size", type=int, help="Matched mixed-language sentences per epoch")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--dev_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building tokenizer from both corpora...")
    tokenizer = build_tokenizer([args.corpus_a, args.corpus_b])
    print(f"Vocabulary size: {len(tokenizer)}  (Language A + Language B)")

    print("Loading corpora...")
    sentences_a = load_corpus(args.corpus_a)
    sentences_b = load_corpus(args.corpus_b)
    print(f"  Language A : {len(sentences_a)} sentences")
    print(f"  Language B : {len(sentences_b)} sentences")
    print(f"  Total      : {len(sentences_a) + len(sentences_b)} sentences")

    train_a, dev_a = split_corpus(sentences_a, args.dev_frac, args.seed)
    train_b, dev_b = split_corpus(sentences_b, args.dev_frac, args.seed + 1)
    train_sents, epoch_counts = build_train_sentences(
        train_a=train_a,
        train_b=train_b,
        epochs=args.epochs,
        epoch_sample_size=args.epoch_sample_size,
        seed=args.seed,
    )
    dev_sents = dev_a + dev_b

    (out_dir / "train.txt").write_text("\n".join(train_sents), encoding="utf-8")
    (out_dir / "dev.txt").write_text("\n".join(dev_sents), encoding="utf-8")
    print(f"  Train : {len(train_sents)} | Dev : {len(dev_sents)}")
    if epoch_counts:
        print(f"  Matched epoch sample size: {epoch_counts[0]}")
    print("Example tokenization:", tokenizer.tokenize(train_sents[0]))

    train_ds = tokenize_dataset(train_sents, tokenizer, args.max_length)
    dev_ds = tokenize_dataset(dev_sents, tokenizer, args.max_length)

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
    n_params = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {n_params:,}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
    )
    steps_per_logical_epoch = (
        math.ceil(epoch_counts[0] / args.batch_size)
        if epoch_counts
        else None
    )
    eval_strategy = "steps" if epoch_counts else "epoch"
    eval_steps = steps_per_logical_epoch if epoch_counts else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1 if epoch_counts else args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=eval_strategy,
        save_steps=eval_steps,
        logging_strategy=eval_strategy,
        logging_steps=eval_steps,
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

    print("Training...")
    trainer.train()
    plot_loss_history(trainer, args.output_dir, steps_per_logical_epoch)

    train_loss = trainer.evaluate(train_ds)["eval_loss"]
    dev_loss = trainer.evaluate(dev_ds)["eval_loss"]
    train_perplexity = math.exp(train_loss)
    dev_perplexity = math.exp(dev_loss)
    print(f"\nFinal train perplexity : {train_perplexity:.2f}")
    print(f"Final dev   perplexity : {dev_perplexity:.2f}")

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    graph_metadata = read_json(args.graph_metadata) if args.graph_metadata else None
    corpus_metadata = read_json(args.corpus_metadata) if args.corpus_metadata else None
    metadata = {
        "condition": args.condition,
        "corpus_a": args.corpus_a,
        "corpus_b": args.corpus_b,
        "graph_metadata": args.graph_metadata,
        "corpus_metadata": args.corpus_metadata,
        "vocab_size": len(tokenizer),
        "epochs": args.epochs,
        "effective_trainer_epochs": 1 if epoch_counts else args.epochs,
        "epoch_sample_size": epoch_counts[0] if epoch_counts else None,
        "epoch_counts": epoch_counts,
        "steps_per_logical_epoch": steps_per_logical_epoch,
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
        "semantic_overlap": graph_metadata.get("overlap") if graph_metadata else None,
        "corpus_generation": corpus_metadata,
    }
    (out_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved to {final_dir}")
    print(f"Wrote {out_dir / 'training_metadata.json'}")


if __name__ == "__main__":
    main()
