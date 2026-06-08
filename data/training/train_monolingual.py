"""
Train a small BERT-style MLM on a single language corpus (monolingual baseline).

Usage:
    python train_monolingual.py \
        --corpus ../generate_sentences/corpus_hiragana.txt \
        --output_dir ./checkpoints_monolingual
"""

import argparse
from pathlib import Path

from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
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
        min_frequency=1,
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
    p.add_argument('--max_length', type=int, default=64)
    p.add_argument('--epochs',     type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--mlm_prob',   type=float, default=0.15)
    args = p.parse_args()

    # 1. Build tokenizer
    print('Building tokenizer...')
    tokenizer = build_tokenizer([args.corpus])
    vocab_size = len(tokenizer)
    print(f'Vocabulary size: {vocab_size}')

    # 2. Load and tokenize data
    print('Loading corpus...')
    sentences = load_corpus(args.corpus)
    print(f'Total sentences: {len(sentences)}')
    dataset = tokenize_dataset(sentences, tokenizer, args.max_length)

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

    # 5. Train
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_strategy='epoch',
        logging_steps=100,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print('Training...')
    trainer.train()

    out = Path(args.output_dir) / 'final'
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
