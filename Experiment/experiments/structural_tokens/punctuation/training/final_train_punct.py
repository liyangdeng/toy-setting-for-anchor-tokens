"""
Train for the punctuation experiment.
 
The presense of punctuation is determined by the corpus files created by:
create_corpora_punct.py
 
Make sure to use the correct corpora as --corpus_a / --corpus_b
 
--setting       CJK             Hiragana
  shared        .,              .,  
  none       (none)             (none)
  disjoint      *;              .,          

usage:
    python train_punct.py --setting shared \
        --seed 42 \
        --corpus_a corpus_cjk_synset.txt \
        --corpus_b corpus_hiragana_synset.txt \
        --output_dir punct_checkpoints_shared

alternatively:
    bash run_punct_seeds.sh
 
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
from datasets import Dataset
 
 
PUNCT_BY_SETTING = {
    'shared':   [',', '.'],
    'none':     [],
    'disjoint': [';', '*', ',', '.'],
}
 
 
# TOKENISER
 
def build_tokenizer(corpus_files, punct_tokens=()):
    """
    Whitespace-split WordLevel tokenizer.
    (mo [CLS]/[SEP] anymore!)
    Makes sure punctuation is never masked.
    """
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[MASK]"],
        min_frequency=1,
    )
    tokenizer.train(corpus_files, trainer)
 
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        additional_special_tokens=list(punct_tokens),
    )
    fast.backend_tokenizer.post_processor = TemplateProcessing(single="$A")
 
    return fast
 
 
# DATA
 
def load_corpus(path):
    lines = Path(path).read_text(encoding='utf-8').strip().split('\n')
    return [l for l in lines if l.strip()]
 
 
def split_corpus(rows, dev_frac, seed):
    """ Deterministic shuffle + split
    identical split across settings/seeds
    """
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * dev_frac))
    return shuffled[n_dev:], shuffled[:n_dev]
 
 
def tokenize_sentences(sentences, tokenizer, max_length=64):
 
    ds = Dataset.from_dict({'text': sentences})
    return ds.map(
        lambda b: tokenizer(b['text'], truncation=True, max_length=max_length, padding=False),
        batched=True,
        remove_columns=['text'],
        load_from_cache_file=False,
    )
 
 
# PLOT LOSS
 
def plot_loss_history(trainer, output_dir, setting):
    history = trainer.state.log_history
 
    train_records = [r for r in history if 'loss' in r and 'eval_loss' not in r and 'epoch' in r]
    eval_records  = [r for r in history if 'eval_loss' in r and 'epoch' in r]
 
    train_epochs = [r['epoch'] for r in train_records]
    train_losses = [r['loss']  for r in train_records]
    eval_epochs  = [r['epoch'] for r in eval_records]
    eval_losses  = [r['eval_loss'] for r in eval_records]
 
    if not train_losses:
        print('No loss records found')
        return
 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_epochs, train_losses, marker='o', label='Training loss')
 
    if eval_losses:
        ax.plot(eval_epochs, eval_losses, marker='o', label='Validation loss')
 
    ax.set_xlabel('epoch')
    ax.set_ylabel('MLM loss')
 
    ax.set_title(f"Punctuation, setting '{setting}'")
 
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = Path(output_dir) / 'loss_curve.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
 
    print(f'Loss plot saved to {out_path}')
 
 
# _________________________________________________________________________
 
def main():
 
    p = argparse.ArgumentParser(description='Punctuation experiment (no [CLS]/[SEP])')
    p.add_argument('--setting', choices=['shared', 'none', 'disjoint'])
    p.add_argument('--corpus_a')
    p.add_argument('--corpus_b')
    p.add_argument('--output_dir',   default='punct_checkpoints')
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
    print(f"Final punctuation experiment, setting '{args.setting}'")
 
    tokenizer = build_tokenizer([args.corpus_a, args.corpus_b], PUNCT_BY_SETTING[args.setting])
    vocab_size = len(tokenizer)
 
    print('Loading corpora...')
    sentences_a = load_corpus(args.corpus_a)
    sentences_b = load_corpus(args.corpus_b)
    tagged = [(s, 'a') for s in sentences_a] + [(s, 'b') for s in sentences_b]
    print(f'CJK: {len(sentences_a)} sentences')
    print(f'Hiragana: {len(sentences_b)} sentences')
    print(f'Total: {len(tagged)} sentences')
 
    train_tagged, dev_tagged = split_corpus(tagged, args.dev_frac, args.seed)
 
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'train.txt').write_text('\n'.join(s for s, _ in train_tagged), encoding='utf-8')
    (out_dir / 'dev.txt').write_text('\n'.join(s for s, _ in dev_tagged),   encoding='utf-8')
    print(f'Train : {len(train_tagged)} | Dev : {len(dev_tagged)}')
 
    train_sents = [s for s, _ in train_tagged]
    dev_sents   = [s for s, _ in dev_tagged]
    train_ds = tokenize_sentences(train_sents, tokenizer, args.max_length)
    dev_ds   = tokenize_sentences(dev_sents,   tokenizer, args.max_length)
 
    # MODEL 
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
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f'Model parameters: {n_params:,}')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
    )
 
    # Train
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
 
    # Loss plot
    plot_loss_history(trainer, args.output_dir, args.setting)
 
    # Final perplexity
    train_loss = trainer.evaluate(train_ds)['eval_loss']
    dev_loss   = trainer.evaluate(dev_ds)['eval_loss']
    print(f'\nFinal train perplexity : {math.exp(train_loss):.2f}')
    print(f'Final dev perplexity : {math.exp(dev_loss):.2f}')
 
    out = out_dir / 'final'
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
 
 
if __name__ == '__main__':
    main()