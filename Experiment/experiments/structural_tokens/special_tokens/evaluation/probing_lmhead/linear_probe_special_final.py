"""
Linear probing for the SPECIAL TOKEN experiment (final version)

Matches final_train_st.py:
  - special_config.json has per-language {cls, sep} only.
    [MASK] is shared across languages -> not a per-language field anymore.
  - Only the tokenizer's post-processor differs across languages
    (arm 'shared': both [CLS] $A [SEP];  arm 'none': both $A;
     arm 'disjoint': CJK is [BEG] $A [END], Hiragana is [CLS] $A [SEP]).

The script switches the tokenizer's post-processor between A- and B-side
extraction so each language is encoded with the wrapping it was trained on.
Mask handling stays language-agnostic (single [MASK]).

Two sections:
  1. TRAINING (pasted from train_multilingual_synset.py) -- kept for
     --do_train, but final_train_st.py is the actual trainer for this
     experiment; use --model_dir to probe an existing checkpoint.
  2. LINEAR PROBE. Per triple: mask the target entity token in A- and
     B-side sentences, extract hidden state at [MASK] per layer, fit
     logistic regression on A vectors, evaluate on B vectors,
     OR-aggregate per triple.

Inputs from build_probing_corpus.py's output for THIS run:
  --model_dir      <- final_train_st.py's output_dir/final
  --final_omitted  <- final_omitted.json (post-filter, tagged "track": "entity")
  --parallel       <- final_omitted_corpus/parallel_corpus_synset.json

Usage:
    python linear_probe_special_final.py \
        --model_dir st_checkpoints_disjoint/final \
        --final_omitted probing_run/final_omitted.json \
        --parallel probing_run/final_omitted_corpus/parallel_corpus_synset.json \
        --cjk_dict synset_pos_artificial_cjk.json \
        --hira_dict synset_pos_artificial_hiragana.json \
        --out_dir probe_st_disjoint \
        --seed 42
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
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from datasets import Dataset
from sklearn.linear_model import LogisticRegression


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 -- TRAINING (kept for --do_train; final_train_st.py is preferred)
# ══════════════════════════════════════════════════════════════════════════

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
        unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )


def load_corpus(path):
    lines = Path(path).read_text(encoding='utf-8').strip().split('\n')
    return [l for l in lines if l.strip()]


def split_corpus(sentences, dev_frac, seed):
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * dev_frac))
    return shuffled[n_dev:], shuffled[:n_dev]


def tokenize_dataset(sentences, tokenizer, max_length=64):
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True,
                          max_length=max_length, padding=False)
    dataset = Dataset.from_dict({'text': sentences})
    return dataset.map(tokenize, batched=True, remove_columns=['text'])


def plot_loss_history(trainer, output_dir):
    history = trainer.state.log_history
    tr = [(r['epoch'], r['loss']) for r in history
          if 'loss' in r and 'eval_loss' not in r and 'epoch' in r]
    ev = [(r['epoch'], r['eval_loss']) for r in history
          if 'eval_loss' in r and 'epoch' in r]
    if not tr:
        print('No training-loss records found.')
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(*zip(*tr), marker='o', label='Training loss')
    if ev:
        ax.plot(*zip(*ev), marker='o', label='Validation loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MLM loss')
    ax.set_title('Multilingual training and validation loss')
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out_path = Path(output_dir) / 'loss_curve.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss plot saved to {out_path}')


def train_multilingual(corpus_a, corpus_b, output_dir, max_length=64,
                        epochs=60, batch_size=64, mlm_prob=0.15, lr=1e-3,
                        warmup_steps=50, dev_frac=0.1, seed=42):
    set_seed(seed)
    corpus_files = [corpus_a, corpus_b]

    print('Building tokenizer from both corpora...')
    tokenizer = build_tokenizer(corpus_files)
    vocab_size = len(tokenizer)
    print(f'Vocabulary size: {vocab_size}')

    print('Loading corpora...')
    sentences_a = load_corpus(corpus_a)
    sentences_b = load_corpus(corpus_b)
    all_sentences = sentences_a + sentences_b
    print(f'  A : {len(sentences_a)} | B : {len(sentences_b)} '
          f'| total : {len(all_sentences)}')

    train_sents, dev_sents = split_corpus(all_sentences, dev_frac, seed)
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'train.txt').write_text('\n'.join(train_sents), encoding='utf-8')
    (out_dir / 'dev.txt').write_text('\n'.join(dev_sents),   encoding='utf-8')

    train_ds = tokenize_dataset(train_sents, tokenizer, max_length)
    dev_ds   = tokenize_dataset(dev_sents,   tokenizer, max_length)

    config = BertConfig(
        vocab_size=vocab_size, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=4, intermediate_size=512,
        max_position_embeddings=128, pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(config)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob,
    )
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=lr, warmup_steps=warmup_steps,
        eval_strategy='epoch', save_strategy='epoch', logging_strategy='epoch',
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model='eval_loss', seed=seed, report_to='none',
    )
    trainer = Trainer(model=model, args=training_args,
                       train_dataset=train_ds, eval_dataset=dev_ds,
                       data_collator=data_collator)
    print('Training...')
    trainer.train()
    plot_loss_history(trainer, output_dir)

    train_loss = trainer.evaluate(train_ds)['eval_loss']
    dev_loss   = trainer.evaluate(dev_ds)['eval_loss']
    print(f'\nFinal train perplexity : {math.exp(train_loss):.2f}')
    print(f'Final dev perplexity : {math.exp(dev_loss):.2f}')

    out = out_dir / 'final'
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f'Saved to {out}')
    return str(out)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 -- LINEAR PROBE
# ══════════════════════════════════════════════════════════════════════════

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_special_config(model_dir):
    """
    Read special_config.json written by final_train_st.py.
    Returns {'a': {cls, sep}, 'b': {cls, sep}} -- no mask field,
    [MASK] is shared across languages in this experiment.
    Warns and returns None if the file is missing.
    """
    cfg_path = Path(model_dir) / "special_config.json"
    if not cfg_path.exists():
        print(f"NO special_config.json at {cfg_path} (!!!)")
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {"a": cfg["a"], "b": cfg["b"]}


def apply_template(tokenizer, cls_tok, sep_tok):
    """
    Same behaviour as final_train_st.py's set_template. Wraps sentences with
    the arm's per-language boundary tokens, or with no wrapping when cls_tok
    is None (the 'none' arm).
    """
    if cls_tok is None:
        tokenizer.backend_tokenizer.post_processor = TemplateProcessing(single="$A")
        return
    ids = tokenizer.convert_tokens_to_ids
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single=f"{cls_tok} $A {sep_tok}",
        special_tokens=[(cls_tok, ids(cls_tok)), (sep_tok, ids(sep_tok))],
    )


def sanity_check_wrapping(tokenizer, policy, mask_str):
    """
    For each language, apply the setting's post-processor and print the
    tokenized stub. Confirms boundary tokens are placed correctly and that
    [MASK] survives as a single token.

    Expected output (disjoint arm):
        CJK       'foo [MASK] bar' -> ['[BEG]', '[UNK]', '[MASK]', '[UNK]', '[END]']
        Hiragana  'foo [MASK] bar' -> ['[CLS]', '[UNK]', '[MASK]', '[UNK]', '[SEP]']
    """
    print("Tokenizer output per language:")
    for lang, label in (("a", "CJK     "), ("b", "Hiragana")):
        apply_template(tokenizer, policy[lang]["cls"], policy[lang]["sep"])
        stub = f"foo {mask_str} bar"
        ids = tokenizer(stub)["input_ids"]
        toks = tokenizer.convert_ids_to_tokens(ids)
        print(f"  {label} '{stub}' -> {toks}")


def find_all_masked(sentences, target_tok, mask_str):
    """
    Every rendering of a triple where target_tok appears as a standalone
    word, each with target_tok replaced by mask_str (order preserved,
    duplicates collapsed). A triple's accuracy is judged by OR across ALL
    of these at eval time (see run_probe), and all of them are used as
    separate training rows under the same label (see build_examples).
    """
    out, seen = [], set()
    for sent in sentences:
        words = sent.split()
        if target_tok not in words:
            continue
        words[words.index(target_tok)] = mask_str
        masked = " ".join(words)
        if masked not in seen:
            seen.add(masked)
            out.append(masked)
    return out


def build_examples(final_omitted, parallel, cjk_dict, hira_dict, mask_str):
    """
    Build per-triple examples with masked A- and B-side sentence renderings.
    mask_str is a single string ([MASK] in this experiment).
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
        a_sents = find_all_masked(entry["lang_a"], a_tok, mask_str)
        b_sents = find_all_masked(entry["lang_b"], b_tok, mask_str)
        if not a_sents or not b_sents:
            continue
        examples.append({
            "a_sents": a_sents, "b_sents": b_sents,
            "concept": p["target"], "relation": p["relation"],
        })
    return examples


@torch.no_grad()
def extract_all_layers(model, tokenizer, sentences, device,
                        max_length=64, batch_size=32):
    """
    For every layer, the hidden state at the [MASK] position.
    Uses tokenizer.mask_token_id -- [MASK] is shared across languages
    in this experiment.
    """
    n_layers = model.config.num_hidden_layers + 1
    out = [[] for _ in range(n_layers)]
    mask_id = tokenizer.mask_token_id

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length).to(device)
        outputs = model(**enc, output_hidden_states=True)
        hs = outputs.hidden_states

        mask_positions = (enc["input_ids"] == mask_id)
        for row in range(len(batch)):
            pos = mask_positions[row].nonzero(as_tuple=True)[0]
            for L in range(n_layers):
                out[L].append(hs[L][row, pos].mean(dim=0).cpu().numpy())

    return [np.stack(layer_vecs) for layer_vecs in out]


def run_probe(model_dir, examples, out_dir, policy=None, seed=42):
    """
    Training uses every triple's every A-sentence rendering as a separate row
    under the same label. Evaluation OR-aggregates per triple: correct if any
    B rendering is classified correctly. Accuracy = hit triples / total triples.

    policy: {'a': {cls, sep}, 'b': {cls, sep}} from load_special_config. The
    tokenizer's post-processor is switched to A's wrapping before A extraction
    and to B's before B extraction. When None, the tokenizer's saved
    post-processor is used unchanged for both languages.
    """
    device = pick_device()
    print(f"device: {device}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir).to(device).eval()

    concepts = [e["concept"] for e in examples]
    concept_to_id = {c: i for i, c in enumerate(sorted(set(concepts)))}
    y_by_triple = np.array([concept_to_id[c] for c in concepts])

    train_sents, train_labels = [], []
    for e in examples:
        label = concept_to_id[e["concept"]]
        for s in e["a_sents"]:
            train_sents.append(s)
            train_labels.append(label)
    y_train = np.array(train_labels)

    eval_sents, eval_triple_idx = [], []
    for ti, e in enumerate(examples):
        for s in e["b_sents"]:
            eval_sents.append(s)
            eval_triple_idx.append(ti)
    eval_triple_idx = np.array(eval_triple_idx)

    n_triples = len(examples)
    print(f"triples: {n_triples} | distinct concepts: {len(concept_to_id)} | "
          f"training rows (A, all renderings): {len(train_sents)} | "
          f"eval rows (B, all renderings): {len(eval_sents)}")

    # CJK extraction -- apply A's template
    if policy is not None:
        apply_template(tokenizer, policy["a"]["cls"], policy["a"]["sep"])
    X_a_layers = extract_all_layers(model, tokenizer, train_sents, device)

    # Hiragana extraction -- apply B's template
    if policy is not None:
        apply_template(tokenizer, policy["b"]["cls"], policy["b"]["sep"])
    X_b_layers = extract_all_layers(model, tokenizer, eval_sents, device)
    n_layers = len(X_a_layers)

    rows = []
    per_relation_rows = defaultdict(list)
    for L in range(n_layers):
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X_a_layers[L], y_train)
        preds = clf.predict(X_b_layers[L])
        row_correct = (preds == y_by_triple[eval_triple_idx])

        triple_hit = np.zeros(n_triples, dtype=bool)
        for correct, ti in zip(row_correct, eval_triple_idx):
            if correct:
                triple_hit[ti] = True
        acc = triple_hit.mean()
        rows.append({"layer": L, "accuracy": float(acc), "n": n_triples})
        print(f"layer {L}: accuracy = {acc:.3f}  ({int(triple_hit.sum())}/{n_triples} triples)")

        rel_acc = defaultdict(lambda: [0, 0])
        for e, hit in zip(examples, triple_hit):
            rel_acc[e["relation"]][0] += int(hit)
            rel_acc[e["relation"]][1] += 1
        for rel, (hit, n) in rel_acc.items():
            per_relation_rows[L].append({"relation": rel, "accuracy": hit / n, "n": n})

    write_probe_outputs(rows, per_relation_rows, out_dir)
    return rows


def write_probe_outputs(rows, per_relation_rows, out_dir):
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
    ax.plot([r["layer"] for r in rows], [r["accuracy"] for r in rows], marker="o")
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


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--do_train", action="store_true",
                    help="train a fresh bilingual model before probing (vanilla, "
                         "not arm-aware -- prefer final_train_st.py + --model_dir)")
    p.add_argument("--corpus_a")
    p.add_argument("--corpus_b")
    p.add_argument("--train_output_dir", default="./checkpoints_probe_model")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--model_dir",
                    help="existing bilingual checkpoint (…/final); ignored if --do_train")
    p.add_argument("--final_omitted", required=True)
    p.add_argument("--parallel", required=True)
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

    final_omitted = [p for p in json.load(open(args.final_omitted))
                     if p.get("track") == "entity"]
    parallel = json.load(open(args.parallel))
    cjk_dict = {k: v["artificial"] for k, v in json.load(open(args.cjk_dict)).items()}
    hira_dict = {k: v["artificial"] for k, v in json.load(open(args.hira_dict)).items()}

    tok_probe = PreTrainedTokenizerFast.from_pretrained(model_dir)
    policy = load_special_config(model_dir)
    if policy is not None:
        print("found special_config.json")
        print(f"CJK       cls={policy['a']['cls']}  sep={policy['a']['sep']}")
        print(f"Hiragana  cls={policy['b']['cls']}  sep={policy['b']['sep']}")
        sanity_check_wrapping(tok_probe, policy, tok_probe.mask_token)

    mask_str = tok_probe.mask_token   # [MASK], shared across languages
    examples = build_examples(final_omitted, parallel, cjk_dict, hira_dict, mask_str)
    print(f"built {len(examples)} usable mask-entity triples "
          f"(of {len(final_omitted)} final_omitted entity-track triples)")
    run_probe(model_dir, examples, args.out_dir, policy=policy, seed=args.seed)


if __name__ == "__main__":
    main()
