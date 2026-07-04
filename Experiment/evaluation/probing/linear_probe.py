"""
Linear probing for the anchor-token transfer experiment (mask-ENTITY track).

Two sections:
  1. TRAINING -- pasted from train_multilingual_synset.py almost verbatim,
     wrapped in train_multilingual() so this one file can train a bilingual
     model from scratch if you don't already have a checkpoint.
  2. LINEAR PROBE -- the new part. For every mask-entity probe triple:
       - take its A sentence and B sentence (from parallel_corpus_synset.json)
       - replace the TARGET entity token with [MASK] (kept in the sentence,
         not deleted -- avoids leakage since the answer token is gone from
         the input, while the sentence structure the model was trained on
         stays intact)
       - run the frozen model, output_hidden_states=True
       - take the hidden state AT THE [MASK] POSITION, for every layer
     Then, per layer L:
       - fit a linear (logistic regression) classifier on the A vectors,
         label = target concept (the synset id -- shared across A/B, so the
         classifier trained on A can be applied to B)
       - apply that SAME classifier to the B vectors of the SAME probe items
       - accuracy on B = how much of the fact is readable in B's [MASK]
         representation, despite B never having trained on it
     Reports one accuracy curve (layer 0..N) overall and per relation, saves
     a CSV and a PNG plot.

Inputs come from build_probing_corpus.py's output for THIS run:
  --corpus_a / --corpus_b  <- a_training.txt / b_training.txt
  --final_omitted          <- final_omitted.json (post-filter probes)
  --parallel               <- final_omitted_corpus/parallel_corpus_synset.json
                              (the FRESH parallel file built from
                              final_omitted.json, not the old full-corpus one)

Usage (train fresh + probe):
    python linear_probe.py \
        --do_train \
        --corpus_a probing_run/a_training.txt --corpus_b probing_run/b_training.txt \
        --train_output_dir ./checkpoints_treatment \
        --final_omitted probing_run/final_omitted.json \
        --parallel probing_run/final_omitted_corpus/parallel_corpus_synset.json \
        --cjk_dict synset_pos_artificial_cjk_edges_adj_augmented.json \
        --hira_dict synset_pos_artificial_hiragana_edges_adj_augmented.json \
        --out_dir ./probe_results_treatment

Usage (probe an existing checkpoint, e.g. the no-anchor floor model):
    python linear_probe.py \
        --model_dir ./checkpoints_no_anchor/final \
        --final_omitted probing_run/final_omitted.json \
        --parallel probing_run/final_omitted_corpus/parallel_corpus_synset.json \
        --cjk_dict ... --hira_dict ... \
        --out_dir ./probe_results_no_anchor
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
from sklearn.linear_model import LogisticRegression


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRAINING (pasted from train_multilingual_synset.py)
# ══════════════════════════════════════════════════════════════════════════

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


def train_multilingual(corpus_a, corpus_b, output_dir, max_length=64,
                        epochs=60, batch_size=64, mlm_prob=0.15, lr=1e-3,
                        warmup_steps=50, dev_frac=0.1, seed=42):
    """Train a small bilingual BERT-style MLM from scratch. Returns the path
    to the saved final checkpoint (output_dir/final)."""
    set_seed(seed)
    corpus_files = [corpus_a, corpus_b]

    print('Building tokenizer from both corpora...')
    tokenizer = build_tokenizer(corpus_files)
    vocab_size = len(tokenizer)
    print(f'Vocabulary size: {vocab_size}  (Language A + Language B)')

    print('Loading corpora...')
    sentences_a = load_corpus(corpus_a)
    sentences_b = load_corpus(corpus_b)
    all_sentences = sentences_a + sentences_b
    print(f'  Language A : {len(sentences_a)} sentences')
    print(f'  Language B : {len(sentences_b)} sentences')
    print(f'  Total      : {len(all_sentences)} sentences')

    train_sents, dev_sents = split_corpus(all_sentences, dev_frac, seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'train.txt').write_text('\n'.join(train_sents), encoding='utf-8')
    (out_dir / 'dev.txt').write_text('\n'.join(dev_sents),   encoding='utf-8')
    print(f'  Train : {len(train_sents)} | Dev : {len(dev_sents)}')

    train_ds = tokenize_dataset(train_sents, tokenizer, max_length)
    dev_ds   = tokenize_dataset(dev_sents,   tokenizer, max_length)

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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        seed=seed,
        report_to='none',
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=data_collator,
    )

    print('Training...')
    trainer.train()
    plot_loss_history(trainer, output_dir)

    train_loss = trainer.evaluate(train_ds)['eval_loss']
    dev_loss   = trainer.evaluate(dev_ds)['eval_loss']
    print(f'\nFinal train perplexity : {math.exp(train_loss):.2f}')
    print(f'Final dev   perplexity : {math.exp(dev_loss):.2f}')

    out = out_dir / 'final'
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f'Saved to {out}')
    return str(out)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — LINEAR PROBE
# ══════════════════════════════════════════════════════════════════════════

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_and_mask(sentences, target_tok, mask_str):
    """Try each candidate rendering of a triple; return the first sentence
    where target_tok is a standalone word, with it replaced by mask_str.
    Returns None if target_tok never appears as a standalone word."""
    for sent in sentences:
        words = sent.split()
        if target_tok in words:
            words[words.index(target_tok)] = mask_str
            return " ".join(words)
    return None


def build_examples(final_omitted, parallel, cjk_dict, hira_dict, mask_str):
    """
    final_omitted: list of {source, relation, target} -- the surviving
    mask-entity probes from build_probing_corpus.py's top-1 + per-relation
    filtering. Every entry here IS a mask-entity candidate (this pipeline
    doesn't handle mask-relation), so there's no "track" field to check.

    parallel: the FRESH parallel file built from final_omitted.json in
    build_probing_corpus.py's step 8 (final_omitted_corpus/parallel_corpus_
    synset.json) -- NOT the old full-corpus parallel file. Its lang_a
    sentences are exactly what got appended into a_training.txt, so the model
    actually trained on them.

    Builds (a_sentence_masked, b_sentence_masked, concept_label, relation).
    Skips entries whose target token can't be located as a standalone word in
    either language's sentence (should be rare).
    """
    par = {(e["source"], e["relation"], e["target"]): e for e in parallel}
    examples = []
    for p in final_omitted:
        key = (p["source"], p["relation"], p["target"])
        entry = par.get(key)
        if entry is None:
            continue
        a_tok = cjk_dict.get(p["target"])
        b_tok = hira_dict.get(p["target"])
        if a_tok is None or b_tok is None:
            continue
        a_sent = find_and_mask(entry["lang_a"], a_tok, mask_str)
        b_sent = find_and_mask(entry["lang_b"], b_tok, mask_str)
        if a_sent is None or b_sent is None:
            continue
        examples.append({
            "a_sent": a_sent, "b_sent": b_sent,
            "concept": p["target"], "relation": p["relation"],
        })
    return examples


@torch.no_grad()
def extract_all_layers(model, tokenizer, sentences, device, max_length=64,
                        batch_size=32):
    """
    Returns a list (length = num_layers+1) of np.arrays [n_sentences, hidden],
    the hidden state at the [MASK] position, for every layer.
    """
    n_layers = model.config.num_hidden_layers + 1
    out = [[] for _ in range(n_layers)]
    mask_id = tokenizer.mask_token_id

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length).to(device)
        outputs = model(**enc, output_hidden_states=True)
        hs = outputs.hidden_states   # tuple of [batch, seq, hidden], len n_layers

        mask_positions = (enc["input_ids"] == mask_id)
        for row in range(len(batch)):
            pos = mask_positions[row].nonzero()
            pos = pos[0, 0].item() if len(pos) else 0   # fallback: first token
            for L in range(n_layers):
                out[L].append(hs[L][row, pos].cpu().numpy())

    return [np.stack(layer_vecs) for layer_vecs in out]


def run_probe(model_dir, examples, out_dir, seed=42):
    device = pick_device()
    print(f"device: {device}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir).to(device).eval()

    a_sents = [e["a_sent"] for e in examples]
    b_sents = [e["b_sent"] for e in examples]
    concepts = [e["concept"] for e in examples]
    relations = [e["relation"] for e in examples]

    concept_to_id = {c: i for i, c in enumerate(sorted(set(concepts)))}
    y = np.array([concept_to_id[c] for c in concepts])

    print(f"probe items: {len(examples)} | distinct concepts: {len(concept_to_id)}")

    X_a_layers = extract_all_layers(model, tokenizer, a_sents, device)
    X_b_layers = extract_all_layers(model, tokenizer, b_sents, device)
    n_layers = len(X_a_layers)

    rows = []
    per_relation_rows = defaultdict(list)   # layer -> list of (relation, correct)
    for L in range(n_layers):
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X_a_layers[L], y)
        preds = clf.predict(X_b_layers[L])
        correct = (preds == y)
        acc = correct.mean()
        rows.append({"layer": L, "accuracy": float(acc), "n": len(y)})
        print(f"layer {L}: accuracy = {acc:.3f}")

        rel_acc = defaultdict(lambda: [0, 0])
        for rel, c in zip(relations, correct):
            rel_acc[rel][0] += int(c)
            rel_acc[rel][1] += 1
        for rel, (hit, n) in rel_acc.items():
            per_relation_rows[L].append({"relation": rel, "accuracy": hit / n, "n": n})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "layerwise_accuracy.csv", "w") as f:
        f.write("layer,accuracy,n\n")
        for r in rows:
            f.write(f"{r['layer']},{r['accuracy']:.4f},{r['n']}\n")

    with open(out_dir / "layerwise_accuracy_per_relation.csv", "w") as f:
        f.write("layer,relation,accuracy,n\n")
        for L, recs in per_relation_rows.items():
            for r in recs:
                f.write(f"{L},{r['relation']},{r['accuracy']:.4f},{r['n']}\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["layer"] for r in rows], [r["accuracy"] for r in rows],
            marker="o")
    ax.set_xlabel("layer (0 = embedding)")
    ax.set_ylabel("accuracy (A-trained probe on B)")
    ax.set_title("Linear probe: layer-wise transfer accuracy")
    ax.set_xticks([r["layer"] for r in rows])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved -> {out_dir/'layerwise_accuracy.csv'}")
    print(f"saved -> {out_dir/'layerwise_accuracy_per_relation.csv'}")
    print(f"saved -> {out_dir/'layerwise_accuracy.png'}")
    return rows


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    # training (optional)
    p.add_argument("--do_train", action="store_true",
                    help="train a fresh bilingual model before probing")
    p.add_argument("--corpus_a")
    p.add_argument("--corpus_b",
                    help="a_training.txt / b_training.txt from build_probing_corpus.py "
                         "(corpus_a = deprived-A + final_omitted-A; corpus_b = deprived-B)")
    p.add_argument("--train_output_dir", default="./checkpoints_probe_model")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    # probing (always required)
    p.add_argument("--model_dir",
                    help="existing bilingual checkpoint (…/final); ignored if --do_train")
    p.add_argument("--final_omitted", required=True,
                    help="final_omitted.json from build_probing_corpus.py "
                         "(post top-1-filter, post per-relation floor)")
    p.add_argument("--parallel", required=True,
                    help="final_omitted_corpus/parallel_corpus_synset.json -- the FRESH "
                         "parallel file built FROM final_omitted.json (step 8), not the "
                         "old full-corpus parallel_corpus_synset.json")
    p.add_argument("--cjk_dict", required=True)
    p.add_argument("--hira_dict", required=True)
    p.add_argument("--out_dir", default="./probe_results")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.do_train:
        assert args.corpus_a and args.corpus_b, "--do_train needs --corpus_a/--corpus_b"
        model_dir = train_multilingual(
            args.corpus_a, args.corpus_b, args.train_output_dir,
            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        )
    else:
        assert args.model_dir, "pass --model_dir or use --do_train"
        model_dir = args.model_dir

    final_omitted = json.load(open(args.final_omitted))
    parallel = json.load(open(args.parallel))
    cjk_dict = {k: v["artificial"] for k, v in json.load(open(args.cjk_dict)).items()}
    hira_dict = {k: v["artificial"] for k, v in json.load(open(args.hira_dict)).items()}

    tok_probe = PreTrainedTokenizerFast.from_pretrained(model_dir)
    examples = build_examples(final_omitted, parallel, cjk_dict, hira_dict,
                               tok_probe.mask_token)
    print(f"built {len(examples)} usable mask-entity examples "
          f"(of {len(final_omitted)} final_omitted triples)")

    run_probe(model_dir, examples, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
