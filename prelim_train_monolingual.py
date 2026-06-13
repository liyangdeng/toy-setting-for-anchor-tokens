"""
Train a small BERT-style MLM on a single language corpus (monolingual baseline).
Last change: 10.06.26

Patched version of Clara's script: 
train/dev split, 
per-epoch evaluation, 
more epochs,
final perplexity report, 
higher LR + warmup,
loss change plot.

The accuracy and stuff come in the evaluation script.

Usage:
    python train_monolingual.py \
        --corpus ../generate_sentences/corpus_hiragana.txt \
        --output_dir ./checkpoints_monolingual
"""

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt

from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from datasets import Dataset


# ── Vocabulary ─────────────────────────────────────────────────────────────────

def build_tokenizer(corpus_files):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=1,                      # keep every artificial token
    )
    tokenizer.train(corpus_files, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_corpus(path):
    lines = Path(path).read_text(encoding='utf-8').strip().split('\n')
    return [l for l in lines if l.strip()]


def split_corpus(sentences, dev_frac, seed):
    """Shuffle (deterministically) and split into train / dev."""
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * dev_frac))
    return shuffled[n_dev:], shuffled[:n_dev]      # (train, dev)


def tokenize_dataset(sentences, tokenizer, max_length=64):
    def tokenize(batch):
        return tokenizer(
            batch['text'],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    dataset = Dataset.from_dict({'text': sentences})
    return dataset.map(tokenize, batched=True, remove_columns=['text'])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Monolingual MLM training (baseline)')
    p.add_argument('--corpus',     default='../generate_sentences/corpus_hiragana.txt')
    p.add_argument('--output_dir', default='./checkpoints_monolingual')
    p.add_argument('--max_length', type=int,   default=64)
    p.add_argument('--epochs',     type=int,   default=60)      # was 10, I think from-scratch needs more
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--mlm_prob',   type=float, default=0.15)
    p.add_argument('--lr',         type=float, default=1e-3)    # was Trainer default 5e-5
    p.add_argument('--warmup_steps', type=int, default=50)
    p.add_argument('--dev_frac',   type=float, default=0.1)     # 10% held out for eval
    p.add_argument('--seed',       type=int,   default=42)
    args = p.parse_args()

    set_seed(args.seed)

    # 1. Build tokenizer
    print('Building tokenizer...')
    tokenizer = build_tokenizer([args.corpus])
    vocab_size = len(tokenizer)
    print(f'Vocabulary size: {vocab_size}')

    # 2. Load, split, and tokenize data
    print('Loading corpus...')
    sentences = load_corpus(args.corpus)
    train_sents, dev_sents = split_corpus(sentences, args.dev_frac, args.seed)
    # Save splits so evaluation always uses the exact same sentences
    split_dir = Path(args.output_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / 'train.txt').write_text('\n'.join(train_sents), encoding='utf-8')
    (split_dir / 'dev.txt').write_text('\n'.join(dev_sents),   encoding='utf-8')
    print(f'Splits saved to {split_dir}')
    print(f'Total: {len(sentences)} | train: {len(train_sents)} | dev: {len(dev_sents)}')

    # Sanity check: confirm tokenization splits into the intended tokens
    print('Example tokenization:', tokenizer.tokenize(train_sents[0]))

    train_ds = tokenize_dataset(train_sents, tokenizer, args.max_length)
    dev_ds   = tokenize_dataset(dev_sents,   tokenizer, args.max_length)

    # 3. Model
    config = BertConfig(
        vocab_size=vocab_size,
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

    # 4. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
    )

    # 5. Train  (eval + save once per epoch; keep the best dev-loss checkpoint)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        seed=args.seed,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
    )

    print('training...')
    trainer.train()

    # 7. Plot loss history

    def plot_loss_history(trainer, output_dir):
        """Plot training and validation loss recorded by Hugging Face Trainer."""

        history = trainer.state.log_history

        train_records = [
            record for record in history
            if 'loss' in record
            and 'eval_loss' not in record
            and 'epoch' in record
        ]

        eval_records = [
            record for record in history
            if 'eval_loss' in record
            and 'epoch' in record
        ]

        train_epochs = [record['epoch'] for record in train_records]
        train_losses = [record['loss'] for record in train_records]

        eval_epochs = [record['epoch'] for record in eval_records]
        eval_losses = [record['eval_loss'] for record in eval_records]

        if not train_losses:
            print('Warning: no training-loss records found.')
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            train_epochs,
            train_losses,
            marker='o',
            label='Training loss',
        )

        if eval_losses:
            ax.plot(
                eval_epochs,
                eval_losses,
                marker='o',
                label='Validation loss',
            )

        ax.set_xlabel('Epoch')
        ax.set_ylabel('MLM loss')
        ax.set_title('Training and validation loss')
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        output_path = Path(output_dir) / 'loss_curve.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f'Loss plot saved to {output_path}')

    plot_loss_history(trainer, args.output_dir)

    # 7. Report perplexity on train and dev
    train_loss = trainer.evaluate(train_ds)['eval_loss']
    dev_loss   = trainer.evaluate(dev_ds)['eval_loss']
    print(f'\nFinal train perplexity: {math.exp(train_loss):.2f}')
    print(f'Final dev   perplexity: {math.exp(dev_loss):.2f}')

    out = Path(args.output_dir) / 'final'
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
