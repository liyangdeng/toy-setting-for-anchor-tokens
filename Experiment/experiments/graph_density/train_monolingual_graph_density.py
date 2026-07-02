#!/usr/bin/env python3
"""Train a small monolingual BERT-style MLM for graph-density experiments."""

import argparse
import json
import math
import random
from pathlib import Path

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def load_corpus(path):
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_artificial_tokens(paths):
    tokens = []
    seen = set()
    for path in paths:
        dictionary = json.loads(Path(path).read_text(encoding="utf-8"))
        for entry in dictionary.values():
            token = entry["artificial"]
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


def build_fixed_tokenizer(vocab_tokens):
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token in sorted(vocab_tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


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


def build_epoch_sampled_train(train_pool, epochs, epoch_sample_size, seed):
    rng = random.Random(seed + 1000)
    per_epoch = epoch_sample_size or len(train_pool)
    sampled = []
    epoch_counts = []
    for _epoch in range(epochs):
        epoch_rng = random.Random(rng.randint(0, 2**31 - 1))
        epoch_sents = sample_epoch(train_pool, per_epoch, epoch_rng)
        sampled.extend(epoch_sents)
        epoch_counts.append(len(epoch_sents))
    return sampled, epoch_counts


def tokenize_dataset(sentences, tokenizer, max_length):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)

    dataset = Dataset.from_dict({"text": sentences})
    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def plot_loss_history(trainer, output_dir):
    if plt is None:
        print("matplotlib is not installed; skipping loss curve.")
        return
    history = trainer.state.log_history
    train_records = [r for r in history if "loss" in r and "eval_loss" not in r]
    eval_records = [r for r in history if "eval_loss" in r]
    if not train_records:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_records) + 1), [r["loss"] for r in train_records], marker="o", label="Training loss")
    if eval_records:
        ax.plot(range(1, len(eval_records) + 1), [r["eval_loss"] for r in eval_records], marker="o", label="Validation loss")
    ax.set_xlabel("Evaluation boundary")
    ax.set_ylabel("MLM loss")
    ax.set_title("Graph-density monolingual MLM loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Graph-density monolingual MLM training")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fixed_vocab", nargs="+", required=True, help="Artificial dictionary JSON files")
    parser.add_argument("--epoch_sample_size", type=int, help="Sentences sampled per epoch")
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

    tokenizer = build_fixed_tokenizer(load_artificial_tokens(args.fixed_vocab))
    sentences = load_corpus(args.corpus)
    train_pool, dev_sents = split_corpus(sentences, args.dev_frac, args.seed)
    train_sents, epoch_counts = build_epoch_sampled_train(
        train_pool, args.epochs, args.epoch_sample_size, args.seed
    )

    (out_dir / "train_sampled.txt").write_text("\n".join(train_sents), encoding="utf-8")
    (out_dir / "dev.txt").write_text("\n".join(dev_sents), encoding="utf-8")

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
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)
    steps_per_epoch = math.ceil(epoch_counts[0] / args.batch_size)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        eval_strategy="steps",
        eval_steps=steps_per_epoch,
        save_strategy="steps",
        save_steps=steps_per_epoch,
        logging_strategy="steps",
        logging_steps=steps_per_epoch,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds, data_collator=collator)
    trainer.train()
    plot_loss_history(trainer, args.output_dir)

    train_loss = trainer.evaluate(train_ds)["eval_loss"]
    dev_loss = trainer.evaluate(dev_ds)["eval_loss"]
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (out_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "corpus": args.corpus,
                "fixed_vocab": args.fixed_vocab,
                "vocab_size": len(tokenizer),
                "epochs": args.epochs,
                "epoch_sample_size": epoch_counts[0],
                "epoch_counts": epoch_counts,
                "train_sampled_sentences": len(train_sents),
                "dev_sentences": len(dev_sents),
                "steps_per_epoch": steps_per_epoch,
                "train_perplexity": math.exp(train_loss),
                "dev_perplexity": math.exp(dev_loss),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()
