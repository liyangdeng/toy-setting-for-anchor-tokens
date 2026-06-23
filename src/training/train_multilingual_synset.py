"""
Train a small BERT-style MLM on two artificial language corpora jointly
(multilingual baseline for the anchor-token hypothesis experiment).

Based on prelim_train_monolingual.py: same architecture, same hyperparameters,
same train/dev split logic — the only difference is that the tokenizer and
dataset are built from both Language A (CJK) and Language B (Hiragana) together,
so the model sees a shared vocabulary and mixed-language sentences.

No Next Sentence Prediction (NSP) — BertForMaskedLM does MLM only.

Usage:
    python train_multilingual_synset.py \
        --corpus_a ~/toy-setting.../corpus/corpus_cjk_synset.txt \
        --corpus_b ~/toy-setting.../corpus/corpus_hiragana_synset.txt \
        --output_dir ~/Desktop/coding/checkpoints_multi_synset
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
    """
    Build a whitespace-split WordLevel tokenizer from all corpus files.
    Vocabulary covers both Language A and Language B tokens.
    """
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


def split_corpus(sentences, dev_frac, seed):
    """Shuffle deterministically and split into train / dev."""
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * dev_frac))
    return shuffled[n_dev:], shuffled[:n_dev]   # (train, dev)


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


# ── Loss plot ──────────────────────────────────────────────────────────────────

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
    ax.set_title('Multilingual — training and validation loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = Path(output_dir) / 'loss_curve.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss plot saved to {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    BASE = Path('/Users/pengyuwen/toy-setting-for-anchor-tokens/data')

    p = argparse.ArgumentParser(description='Joint MLM training on two language corpora')
    p.add_argument('--corpus_a',     default=str(BASE / 'corpus/corpus_cjk_synset.txt'))
    p.add_argument('--corpus_b',     default=str(BASE / 'corpus/corpus_hiragana_synset.txt'))
    p.add_argument('--output_dir',   default=str(Path.home() / 'Desktop/coding/checkpoints_multi_synset'))
    p.add_argument('--max_length',   type=int,   default=64)
    p.add_argument('--epochs',       type=int,   default=60)
    p.add_argument('--batch_size',   type=int,   default=64)
    p.add_argument('--mlm_prob',     type=float, default=0.15)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--warmup_steps', type=int,   default=50)
    p.add_argument('--dev_frac',     type=float, default=0.1)
    p.add_argument('--seed',         type=int,   default=42)
    args = p.parse_args()

    set_seed(args.seed)

    corpus_files = [args.corpus_a, args.corpus_b]

    # 1. Build tokenizer from both languages
    print('Building tokenizer from both corpora...')
    tokenizer = build_tokenizer(corpus_files)
    vocab_size = len(tokenizer)
    print(f'Vocabulary size: {vocab_size}  (Language A + Language B)')

    # 2. Load, mix, split, tokenize
    print('Loading corpora...')
    sentences_a = load_corpus(args.corpus_a)
    sentences_b = load_corpus(args.corpus_b)
    all_sentences = sentences_a + sentences_b
    print(f'  Language A : {len(sentences_a)} sentences')
    print(f'  Language B : {len(sentences_b)} sentences')
    print(f'  Total      : {len(all_sentences)} sentences')

    train_sents, dev_sents = split_corpus(all_sentences, args.dev_frac, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'train.txt').write_text('\n'.join(train_sents), encoding='utf-8')
    (out_dir / 'dev.txt').write_text('\n'.join(dev_sents),   encoding='utf-8')
    print(f'  Train : {len(train_sents)} | Dev : {len(dev_sents)}')

    print('Example tokenization:', tokenizer.tokenize(train_sents[0]))

    train_ds = tokenize_dataset(train_sents, tokenizer, args.max_length)
    dev_ds   = tokenize_dataset(dev_sents,   tokenizer, args.max_length)

    # 3. Model — same architecture as monolingual for fair comparison
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

    # 4. Data collator (MLM only, no NSP)
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

    print('Training...')
    trainer.train()

    # 6. Loss plot
    plot_loss_history(trainer, args.output_dir)

    # 7. Final perplexity
    train_loss = trainer.evaluate(train_ds)['eval_loss']
    dev_loss   = trainer.evaluate(dev_ds)['eval_loss']
    print(f'\nFinal train perplexity : {math.exp(train_loss):.2f}')
    print(f'Final dev   perplexity : {math.exp(dev_loss):.2f}')

    # 8. Save
    out = out_dir / 'final'
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
