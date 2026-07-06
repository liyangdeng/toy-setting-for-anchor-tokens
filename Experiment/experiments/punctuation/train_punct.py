"""
Train for the punctuation experiment 

(!) Special tokens:
Before, we did not use [CLS] and [SEP], now they are
actually inserted, and if we want to do that for structure
experiments, we need to modify the script.

--setting       CJK             Hiragana
  shared        .,  [SEP]       .,  [SEP]
  nopunct       (none)             (none)
  disjoint      *; [END]        ., [SEP] 

usage:
    python train_punct.py --setting shared \
        --corpus_a corpus_cjk_synset.txt \
        --corpus_b corpus_hiragana_synset.txt \
        --output_dir checkpoints_shared
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
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from datasets import Dataset, concatenate_datasets


SEP_BY_SETTING = {
    'shared':   {'a': '[SEP]', 'b': '[SEP]'},   # [SEP] shared
    'disjoint': {'a': '[END]', 'b': '[SEP]'},   # CJK has [SEP] -> [END]
    'nopunct':  {'a': None,    'b': None},      # [SEP] gone
}

# TOKENISER 

def build_tokenizer(corpus_files):
    """
    whitespace-split WordLevel tokenizer

    [END] is reserved here (and marked special):
    gets an id, never gets split, and is never masked by MLM.
    """
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[END]"],
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
        additional_special_tokens=["[END]"],   #protect [END]
    )


def make_post_processor(tokenizer, sep_token):
    """
    prepends [CLS] and appends [SEP]/[END].
    sep_token=None -> "[CLS] $A" (for nopunct)
    """
    cls = "[CLS]"
    cls_id = tokenizer.convert_tokens_to_ids(cls)
    
    if sep_token is None:
        return TemplateProcessing(single=f"{cls} $A",
                                  special_tokens=[(cls, cls_id)])
    
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    
    return TemplateProcessing(
        single=f"{cls} $A {sep_token}",
        special_tokens=[(cls, cls_id), (sep_token, sep_id)],
    )


# DATA

def load_corpus(path):
    lines = Path(path).read_text(encoding='utf-8').strip().split('\n')
    return [l for l in lines if l.strip()]


def split_corpus(rows, dev_frac, seed):
    """Shuffle deterministically and split into train / dev. 
    Works on any rows
    (here: (sentence, lang) tuples) -> identical split across settings
    """
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * dev_frac))
    return shuffled[n_dev:], shuffled[:n_dev]


def tokenize_tagged(tagged_rows, tokenizer, sep_for_lang, max_length=64):

    parts = []
    for lang in ('a', 'b'):
        texts = [s for (s, l) in tagged_rows if l == lang]
        if not texts:
            continue
        tokenizer.backend_tokenizer.post_processor = make_post_processor(
            tokenizer, sep_for_lang[lang]
        )
        
        ds = Dataset.from_dict({'text': texts}).map(
            lambda batch: tokenizer(
                batch['text'], truncation=True, max_length=max_length, padding=False
            ),
            batched=True,
            remove_columns=['text'],
            load_from_cache_file=False,
        )
        parts.append(ds)
    return concatenate_datasets(parts)


# PLOT LOSS

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


# _________________________________________________________________________

def main():

    p = argparse.ArgumentParser(description='Punct experiment')
    p.add_argument('--setting',      choices=list(SEP_BY_SETTING), default='shared')
    p.add_argument('--corpus_a',     default=str('corpus/corpus_cjk_synset.txt'))
    p.add_argument('--corpus_b',     default=str('corpus/corpus_hiragana_synset.txt'))
    p.add_argument('--output_dir',   default=str('checkpoints_punct'))
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
    sep_for_lang = SEP_BY_SETTING[args.setting]
    print(f"Setting '{args.setting}': boundary tokens  A/CJK={sep_for_lang['a']}  "
          f"B/Hiragana={sep_for_lang['b']}  ([CLS] shared in all settings)")

    corpus_files = [args.corpus_a, args.corpus_b]

    # 1. Build tokenizer from both languages
    print('Building tokenizer from both corpora...')
    tokenizer = build_tokenizer(corpus_files)
    vocab_size = len(tokenizer)
    print(f'Vocabulary size: {vocab_size}  (Language A + Language B)')

    # 2. Load, tag by language, mix, split
    print('Loading corpora...')
    sentences_a = load_corpus(args.corpus_a)
    sentences_b = load_corpus(args.corpus_b)
    tagged = [(s, 'a') for s in sentences_a] + [(s, 'b') for s in sentences_b]
    print(f'  Language A : {len(sentences_a)} sentences')
    print(f'  Language B : {len(sentences_b)} sentences')
    print(f'  Total      : {len(tagged)} sentences')

    train_tagged, dev_tagged = split_corpus(tagged, args.dev_frac, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'train.txt').write_text('\n'.join(s for s, _ in train_tagged), encoding='utf-8')
    (out_dir / 'dev.txt').write_text('\n'.join(s for s, _ in dev_tagged),   encoding='utf-8')
    print(f'  Train : {len(train_tagged)} | Dev : {len(dev_tagged)}')

    train_ds = tokenize_tagged(train_tagged, tokenizer, sep_for_lang, args.max_length)
    dev_ds   = tokenize_tagged(dev_tagged,   tokenizer, sep_for_lang, args.max_length)

    # 3. MODEL
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
    tokenizer.backend_tokenizer.post_processor = make_post_processor(tokenizer, sep_for_lang['a'])
    out = out_dir / 'final'
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f'Saved to {out}')

if __name__ == '__main__':
    main()
