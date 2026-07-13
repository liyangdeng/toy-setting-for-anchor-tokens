"""
Train for the special-token experiment

--setting
  shared    [CLS]/[SEP]/[MASK] identical
  none      no [CLS]/[SEP] at all, [MASK]/[PAD] kept (MLM needs them)
  disjoint  CJK: [BEG]/[END]/[HID]  Hiragana: [CLS]/[SEP]/[MASK]

  [CLS],[SEP] -> tokenizer post-processor  -> per-language template
  [MASK]      -> MLM collator              -> per-language mask id
  [PAD],[UNK] -> PAD is attention masked, UNK never fires

usage:
  python train_special.py --setting shared \
      --corpus_a corpus_cjk_synset.txt \
      --corpus_b corpus_hiragana_synset.txt \
      --output_dir checkpoints_special_shared
"""

import argparse
import json
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


# SPECIAL TOKEN PATTERN: a = CJK, b = Hiragana

SPECIAL_SETTINGS = {
    'shared':   {'a': {'cls': '[CLS]', 'sep': '[SEP]', 'mask': '[MASK]'},
                 'b': {'cls': '[CLS]', 'sep': '[SEP]', 'mask': '[MASK]'}},
    'none':     {'a': {'cls': None,    'sep': None,    'mask': '[MASK]'},
                 'b': {'cls': None,    'sep': None,    'mask': '[MASK]'}},
    'disjoint': {'a': {'cls': '[BEG]', 'sep': '[END]', 'mask': '[HID]'},
                 'b': {'cls': '[CLS]', 'sep': '[SEP]', 'mask': '[MASK]'}},
}

ALL_SPECIALS   = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[BEG]', '[END]', '[HID]']
EXTRA_SPECIALS = ['[BEG]', '[END]', '[HID]']   # registered so they're never masked


# TOKENIZER

def build_tokenizer(corpus_files):

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=ALL_SPECIALS, min_frequency=1)
    tokenizer.train(corpus_files, trainer)
    
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
        additional_special_tokens=EXTRA_SPECIALS,
    )


def set_template(tokenizer, cls_tok, sep_tok):

    if cls_tok is None:
        tokenizer.backend_tokenizer.post_processor = TemplateProcessing(single="$A")
        return
    ids = tokenizer.convert_tokens_to_ids
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single=f"{cls_tok} $A {sep_tok}",
        special_tokens=[(cls_tok, ids(cls_tok)), (sep_tok, ids(sep_tok))],
    )


# HANDLE MASK

class PerLanguageMLMCollator(DataCollatorForLanguageModeling):
    """ enable language-specific [MASK] id in each row """

    def __init__(self, *args, mask_ids, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ids = mask_ids    # {'a': id, 'b': id}

    def torch_call(self, examples):
        langs = [ex.pop('lang') for ex in examples]
        batch = super().torch_call(examples)
        default = self.tokenizer.mask_token_id
        for i, lang in enumerate(langs):
            tgt = self.mask_ids[lang]
            if tgt != default:
                row = batch['input_ids'][i]
                row[row == default] = tgt
        return batch


# DATA

def load_corpus(path):
    lines = Path(path).read_text(encoding='utf-8').strip().split('\n')
    return [l for l in lines if l.strip()]


def split_corpus(rows, dev_frac, seed):
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * dev_frac))
    return shuffled[n_dev:], shuffled[:n_dev]


def tokenize_tagged(tagged_rows, tokenizer, policy, max_length=64):
    parts = []
    for lang in ('a', 'b'):
        texts = [s for (s, l) in tagged_rows if l == lang]
        if not texts:
            continue
        set_template(tokenizer, policy[lang]['cls'], policy[lang]['sep'])
        ds = Dataset.from_dict({'text': texts}).map(
            lambda b: tokenizer(b['text'], truncation=True, max_length=max_length, padding=False),
            batched=True, remove_columns=['text'], load_from_cache_file=False)
        ds = ds.add_column('lang', [lang] * len(ds))
        parts.append(ds)
    return concatenate_datasets(parts)


# PLOT LOSS

def plot_loss_history(trainer, output_dir):
    history = trainer.state.log_history
    tr = [(r['epoch'], r['loss']) for r in history if 'loss' in r and 'eval_loss' not in r and 'epoch' in r]
    ev = [(r['epoch'], r['eval_loss']) for r in history if 'eval_loss' in r and 'epoch' in r]
    if not tr:
        print('Warning: no training-loss records found.')
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(*zip(*tr), marker='o', label='Training loss')
    if ev:
        ax.plot(*zip(*ev), marker='o', label='Validation loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MLM loss')
    ax.set_title('Special-token experiment - loss'); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out_path = Path(output_dir) / 'loss_curve.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'Loss plot saved to {out_path}')

# _________________________________________________________________________

def main():

    p = argparse.ArgumentParser(description='Special token experiment')
    p.add_argument('--setting', choices=list(SPECIAL_SETTINGS), default='shared')
    p.add_argument('--corpus_a', default='corpus/corpus_cjk_synset.txt')
    p.add_argument('--corpus_b', default='corpus/corpus_hiragana_synset.txt')
    p.add_argument('--output_dir', default='checkpoints_special')
    p.add_argument('--max_length', type=int, default=64)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--mlm_prob', type=float, default=0.15)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup_steps', type=int, default=50)
    p.add_argument('--dev_frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    policy = SPECIAL_SETTINGS[args.setting]
    print(f"Setting '{args.setting}':")

    tokenizer = build_tokenizer([args.corpus_a, args.corpus_b])
    vocab_size = len(tokenizer)
    print(f'Vocabulary size: {vocab_size}')

    sentences_a = load_corpus(args.corpus_a)
    sentences_b = load_corpus(args.corpus_b)
    tagged = [(s, 'a') for s in sentences_a] + [(s, 'b') for s in sentences_b]
    print(f'  A:{len(sentences_a)}  B:{len(sentences_b)}  total:{len(tagged)}')
    train_tagged, dev_tagged = split_corpus(tagged, args.dev_frac, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'train.txt').write_text('\n'.join(s for s, _ in train_tagged), encoding='utf-8')
    (out_dir / 'dev.txt').write_text('\n'.join(s for s, _ in dev_tagged), encoding='utf-8')
    print(f'  Train:{len(train_tagged)}  Dev:{len(dev_tagged)}')

    for lang, label in (('a', 'CJK'), ('b', 'Hiragana')):
        ex = next((s for s, l in train_tagged if l == lang), None)
        if ex is not None:
            set_template(tokenizer, policy[lang]['cls'], policy[lang]['sep'])
            print(f'  example {label}: {tokenizer.convert_ids_to_tokens(tokenizer(ex)["input_ids"])}')

    train_ds = tokenize_tagged(train_tagged, tokenizer, policy, args.max_length)
    dev_ds   = tokenize_tagged(dev_tagged,   tokenizer, policy, args.max_length)

    config = BertConfig(
        vocab_size=vocab_size, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=4, intermediate_size=512,
        max_position_embeddings=128, pad_token_id=tokenizer.pad_token_id)
    model = BertForMaskedLM(config)
    print(f'Model parameters: {sum(pp.numel() for pp in model.parameters()):,}')

    mask_ids = {l: tokenizer.convert_tokens_to_ids(policy[l]['mask']) for l in ('a', 'b')}
    data_collator = PerLanguageMLMCollator(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob, mask_ids=mask_ids)

    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr, warmup_steps=args.warmup_steps,
        eval_strategy='epoch', save_strategy='epoch', logging_strategy='epoch',
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss',
        seed=args.seed, report_to='none',
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=dev_ds, data_collator=data_collator)
    print('Training...')
    trainer.train()

    plot_loss_history(trainer, args.output_dir)
    train_loss = trainer.evaluate(train_ds)['eval_loss']
    dev_loss   = trainer.evaluate(dev_ds)['eval_loss']
    print(f'\nFinal train perplexity : {math.exp(train_loss):.2f}')
    print(f'Final dev   perplexity : {math.exp(dev_loss):.2f}')

    set_template(tokenizer, policy['a']['cls'], policy['a']['sep'])
    out = out_dir / 'final'
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    (out / 'special_config.json').write_text(
        json.dumps({'setting': args.setting, **policy}, indent=2), encoding='utf-8')
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()