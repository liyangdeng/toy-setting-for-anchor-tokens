"""
Linear probing for the anchor-token transfer experiment -- mask-ENTITY and
mask-RELATION tracks (--track).

Two sections:
  1. TRAINING -- pasted from train_multilingual_synset.py almost verbatim,
     wrapped in train_multilingual() so this one file can train a bilingual
     model from scratch if you don't already have a checkpoint.
  2. LINEAR PROBE. For every probe triple:
       - take its A sentence(s) and B sentence(s) (from parallel_corpus_
         synset.json) and mask the relevant span:
           mask-ENTITY:   the single TARGET entity token
           mask-RELATION: the relation-phrase span located by
                           strip_entities_and_match (build_probing_corpus.py)
                           -- 1 token (TRANS branch) or 3 (COP branch)
         (kept as [MASK] in the sentence, not deleted -- avoids leakage since
         the answer token is gone from the input, while the sentence
         structure the model was trained on stays intact)
       - run the frozen model, output_hidden_states=True
       - take the hidden state at the [MASK] position(s), MEAN-POOLED across
         however many there are, for every layer (mask-ENTITY always has
         exactly one, so this is a no-op there; mask-RELATION can have 1 or 3)
     Then, per layer L:
       - fit a linear (logistic regression) classifier on the A vectors.
         Label: mask-ENTITY = target concept id (shared A/B via the synset
         dicts); mask-RELATION = the SPECIFIC template tuple that rendering
         actually realized (the PCFG redraws the VP per sentence, so two
         renderings of the same triple can realize different templates --
         this is a per-rendering label, not a per-triple one like concept id)
       - apply that SAME classifier to the B vectors
       - a TRIPLE counts as correctly-transferred if ANY of its (possibly
         several) B renderings is classified correctly (OR-aggregation --
         see run_probe/run_relation_probe). Final accuracy = hit triples /
         total triples, not per-sentence accuracy. Training likewise uses
         every triple's every A rendering as a separate row under the same
         label, for more context diversity per class.
     Reports one accuracy curve (layer 0..N) overall and per relation, saves
     a CSV and a PNG plot.

Inputs come from build_probing_corpus.py's output for THIS run:
  --corpus_a / --corpus_b  <- a_training.txt / b_training.txt
  --final_omitted          <- final_omitted.json (post-filter probes, each
                              tagged "track": "entity"/"relation" -- pass
                              --track to pick which one this run scores)
  --parallel               <- final_omitted_corpus/parallel_corpus_synset.json
                              (the FRESH parallel file built from
                              final_omitted.json, not the old full-corpus one)
  --gen_script / --grammar <- required for --track relation only, to rebuild
                              the candidate template sets (same as
                              build_probing_corpus.py's build_relation_templates)

Usage (train fresh + probe, mask-ENTITY):
    python linear_probe.py \
        --track entity \
        --do_train \
        --corpus_a probing_run/a_training.txt --corpus_b probing_run/b_training.txt \
        --train_output_dir ./checkpoints_treatment \
        --final_omitted probing_run/final_omitted.json \
        --parallel probing_run/final_omitted_corpus/parallel_corpus_synset.json \
        --cjk_dict synset_pos_artificial_cjk_edges_adj_augmented.json \
        --hira_dict synset_pos_artificial_hiragana_edges_adj_augmented.json \
        --out_dir ./probe_results_treatment

Usage (probe an existing checkpoint, mask-RELATION):
    python linear_probe.py \
        --track relation \
        --model_dir ./checkpoints_no_anchor/final \
        --final_omitted probing_run/final_omitted.json \
        --parallel probing_run/final_omitted_corpus/parallel_corpus_synset.json \
        --gen_script .../v3_generate_sentences.py --grammar .../grammar_templates_adj.py \
        --cjk_dict ... --hira_dict ... \
        --out_dir ./probe_results_no_anchor_relation
"""

import argparse
import json
import math
import random
import sys
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

# mask-RELATION reuses build_probing_corpus.py's template/matching logic
# rather than duplicating it -- keeps the candidate-template definition and
# the entity-span-stripping heuristic in exactly one place. build_probing_corpus/
# is a sibling of this file's parent dir (both live under masked_language_probing/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build_probing_corpus"))
from build_probing_corpus import (
    build_relation_templates,
    translate_templates,
    reverse_translate_templates,
    all_grammar_terminals,
    strip_entities_and_match,
)


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


def find_all_masked(sentences, target_tok, mask_str):
    """Every rendering of a triple where target_tok appears as a standalone
    word, each with target_tok replaced by mask_str (order preserved,
    duplicates collapsed). A triple's accuracy is judged by OR across ALL
    of these at eval time (see run_probe), and all of them are used as
    separate training rows under the same label (see build_examples)."""
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
    final_omitted: mask-ENTITY probes (each tagged "track": "entity" by
    build_probing_corpus.py; callers should already have filtered to just
    this track before calling).

    parallel: the FRESH parallel file built from final_omitted.json in
    build_probing_corpus.py's step 8 (final_omitted_corpus/parallel_corpus_
    synset.json) -- NOT the old full-corpus parallel file. Its lang_a
    sentences are exactly what got appended into a_training.txt, so the model
    actually trained on them.

    Each returned example is ONE TRIPLE, carrying ALL of its usable masked
    A/B sentence renderings (not just one) -- see run_probe for how training
    (every A rendering, same label) and evaluation (OR across a triple's B
    renderings) use them.
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
def extract_all_layers(model, tokenizer, sentences, device, max_length=64,
                        batch_size=32):
    """
    Returns a list (length = num_layers+1) of np.arrays [n_sentences, hidden]:
    for every layer, the MEAN of the hidden states at however many [MASK]
    positions are present in each sentence. mask-ENTITY sentences have
    exactly one mask position, so this is identical to "the mask position's
    vector" there; mask-RELATION sentences can have 1 (TRANS) or 3 (COP)
    mask positions, mean-pooled into the same fixed-size vector so both
    tracks (and both branch lengths within relation track) share one
    classifier per layer.
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
            pos = mask_positions[row].nonzero(as_tuple=True)[0]
            for L in range(n_layers):
                out[L].append(hs[L][row, pos].mean(dim=0).cpu().numpy())

    return [np.stack(layer_vecs) for layer_vecs in out]


def run_probe(model_dir, examples, out_dir, seed=42):
    """
    Training uses EVERY triple's EVERY A-sentence rendering as a separate
    row under that triple's concept label (more context diversity per
    class). Evaluation runs the frozen-on-A classifier on EVERY triple's
    EVERY B-sentence rendering, then OR-aggregates per triple: a triple
    counts as correctly-transferred if ANY of its B renderings is
    classified correctly. Final accuracy = hit triples / total triples
    (not per-sentence accuracy).
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

    X_a_layers = extract_all_layers(model, tokenizer, train_sents, device)
    X_b_layers = extract_all_layers(model, tokenizer, eval_sents, device)
    n_layers = len(X_a_layers)

    rows = []
    per_relation_rows = defaultdict(list)   # layer -> list of (relation, correct)
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

    write_probe_outputs(rows, per_relation_rows, out_dir, title_suffix="")
    return rows


def write_probe_outputs(rows, per_relation_rows, out_dir, title_suffix=""):
    """Shared CSV/plot writer for both run_probe (mask-ENTITY) and
    run_relation_probe (mask-RELATION)."""
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
    ax.set_title(f"Linear probe{title_suffix}: layer-wise transfer accuracy")
    ax.set_xticks([r["layer"] for r in rows])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved -> {out_dir/'layerwise_accuracy.csv'}")
    print(f"saved -> {out_dir/'layerwise_accuracy_per_relation.csv'}")
    print(f"saved -> {out_dir/'layerwise_accuracy.png'}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2b — LINEAR PROBE, mask-RELATION track
# ══════════════════════════════════════════════════════════════════════════

def _mask_span(sentence, start, length, mask_str):
    """Replace `length` contiguous tokens starting at `start` (positions
    computed against the period-stripped word list, matching
    strip_entities_and_match) with `mask_str`, all at once."""
    words = sentence.split()
    period = bool(words) and words[-1] == "."
    if period:
        words = words[:-1]
    masked = words[:start] + [mask_str] * length + words[start + length:]
    if period:
        masked.append(".")
    return " ".join(masked)


def build_relation_examples(final_omitted, parallel, cjk_dict, hira_dict,
                             relation_templates_a, relation_templates_b,
                             reverse_templates_a, reverse_templates_b,
                             grammar_vocab_a, grammar_vocab_b, mask_str):
    """
    final_omitted: mask-RELATION probes (each tagged "track": "relation";
    callers should already have filtered to just this track).

    Unlike mask-entity's concept id, the relation-phrase label is NOT a
    per-triple constant: the PCFG redraws the VP structure independently
    for every emitted sentence, so two renderings of the same triple can
    realize different templates (e.g. one draws the TRANS branch, another
    draws COP).

    Language A and B sentences for the same rendering come from the SAME
    underlying PCFG draw (build_synset_corpus.py translates one shared
    `sent` into both languages), but strip_entities_and_match matches
    against each language's OWN translated token strings -- and those are
    disjoint between languages, so the raw matched tuple isn't a label
    that means the same thing across languages. reverse_templates_a/b
    (from build_probing_corpus.reverse_translate_templates) map each
    language's matched tuple back to the shared English-terminal identity,
    which is what actually gets used as the label -- mirroring mask-
    ENTITY's concept id, which is already language-agnostic by construction.

    Returns a list of per-triple records:
      {"relation": rel,
       "a_rows": [(masked_a_sentence, canonical_template_tuple), ...],
       "b_rows": [(masked_b_sentence, canonical_template_tuple), ...]}
    """
    par = {(e["source"], e["relation"], e["target"]): e for e in parallel}
    examples = []
    for p in final_omitted:
        key = (p["source"], p["relation"], p["target"])
        entry = par.get(key)
        if entry is None:
            continue
        src_a, tgt_a = cjk_dict.get(p["source"]), cjk_dict.get(p["target"])
        src_b, tgt_b = hira_dict.get(p["source"]), hira_dict.get(p["target"])
        if None in (src_a, tgt_a, src_b, tgt_b):
            continue
        templates_a = relation_templates_a.get(p["relation"], {})
        templates_b = relation_templates_b.get(p["relation"], {})
        rev_a = reverse_templates_a.get(p["relation"], {})
        rev_b = reverse_templates_b.get(p["relation"], {})
        if not templates_a or not templates_b:
            continue

        a_rows, seen_a = [], set()
        for s in entry["lang_a"]:
            m = strip_entities_and_match(s, src_a, tgt_a, templates_a, grammar_vocab_a)
            if m is None:
                continue
            start, length, matched = m
            canonical = rev_a.get(matched)
            if canonical is None:
                continue
            masked = _mask_span(s, start, length, mask_str)
            if masked not in seen_a:
                seen_a.add(masked)
                a_rows.append((masked, canonical))

        b_rows, seen_b = [], set()
        for s in entry["lang_b"]:
            m = strip_entities_and_match(s, src_b, tgt_b, templates_b, grammar_vocab_b)
            if m is None:
                continue
            start, length, matched = m
            canonical = rev_b.get(matched)
            if canonical is None:
                continue
            masked = _mask_span(s, start, length, mask_str)
            if masked not in seen_b:
                seen_b.add(masked)
                b_rows.append((masked, canonical))

        if not a_rows or not b_rows:
            continue
        examples.append({"relation": p["relation"], "a_rows": a_rows, "b_rows": b_rows})
    return examples


def run_relation_probe(model_dir, examples, out_dir, seed=42):
    """
    mask-RELATION counterpart to run_probe. Label space is a single GLOBAL
    template_to_id across every relation's every CANONICAL (language-
    agnostic) template identity -- mirrors entity track's concept_to_id.
    NOT the relation name (the classifier's job is to recover the specific
    realized template) and NOT the raw per-language matched tuple from
    strip_entities_and_match (build_relation_examples already converts that
    to the shared English-terminal identity via reverse_translate_templates
    -- see its docstring for why: the same PCFG draw renders to completely
    different, disjoint token strings per language, so the untranslated
    tuple can't be used as a label shared across languages). Per-relation
    numbers are a post-hoc groupby over which relation each triple belongs
    to, exactly like entity track's per_relation_rows.

    Training: every triple's every A-rendering is a row, labeled with THAT
    rendering's own canonical template identity (not a shared per-triple
    label). Evaluation: every triple's every B-rendering is scored against
    its own canonical template identity; a triple counts as hit if ANY of
    its B-renderings is classified correctly (OR-aggregation). Final
    accuracy = hit triples / total triples.
    """
    device = pick_device()
    print(f"device: {device}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir).to(device).eval()

    all_templates = sorted({matched for e in examples
                             for _, matched in e["a_rows"] + e["b_rows"]})
    template_to_id = {t: i for i, t in enumerate(all_templates)}

    train_sents, train_labels = [], []
    for e in examples:
        for sent, matched in e["a_rows"]:
            train_sents.append(sent)
            train_labels.append(template_to_id[matched])
    y_train = np.array(train_labels)

    eval_sents, eval_labels, eval_triple_idx = [], [], []
    for ti, e in enumerate(examples):
        for sent, matched in e["b_rows"]:
            eval_sents.append(sent)
            eval_labels.append(template_to_id[matched])
            eval_triple_idx.append(ti)
    eval_labels = np.array(eval_labels)
    eval_triple_idx = np.array(eval_triple_idx)

    n_triples = len(examples)
    print(f"triples: {n_triples} | distinct templates: {len(template_to_id)} | "
          f"training rows (A, all renderings): {len(train_sents)} | "
          f"eval rows (B, all renderings): {len(eval_sents)}")

    X_a_layers = extract_all_layers(model, tokenizer, train_sents, device)
    X_b_layers = extract_all_layers(model, tokenizer, eval_sents, device)
    n_layers = len(X_a_layers)

    rows = []
    per_relation_rows = defaultdict(list)
    for L in range(n_layers):
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X_a_layers[L], y_train)
        preds = clf.predict(X_b_layers[L])
        row_correct = (preds == eval_labels)

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

    write_probe_outputs(rows, per_relation_rows, out_dir, title_suffix=" (mask-RELATION)")
    return rows


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=["entity", "relation"], default="entity",
                    help="which probe track in final_omitted.json to score "
                         "(entries are tagged \"track\": \"entity\"/\"relation\")")
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
    # --track relation only
    p.add_argument("--gen_script",
                    help="required for --track relation: sentence generator matching your "
                         "current grammar/switches version (same one build_probing_corpus.py "
                         "used), needed to rebuild the candidate template sets")
    p.add_argument("--grammar",
                    help="required for --track relation: grammar_templates_adj.py")
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

    final_omitted = [p for p in json.load(open(args.final_omitted)) if p.get("track") == args.track]
    parallel = json.load(open(args.parallel))
    cjk_dict = {k: v["artificial"] for k, v in json.load(open(args.cjk_dict)).items()}
    hira_dict = {k: v["artificial"] for k, v in json.load(open(args.hira_dict)).items()}

    tok_probe = PreTrainedTokenizerFast.from_pretrained(model_dir)

    if args.track == "entity":
        examples = build_examples(final_omitted, parallel, cjk_dict, hira_dict,
                                   tok_probe.mask_token)
        print(f"built {len(examples)} usable mask-entity triples "
              f"(of {len(final_omitted)} final_omitted entity-track triples)")
        run_probe(model_dir, examples, args.out_dir, seed=args.seed)
    else:
        assert args.gen_script and args.grammar, \
            "--track relation needs --gen_script and --grammar (to rebuild candidate templates)"
        templates = build_relation_templates(args.gen_script, args.grammar)
        templates_a = translate_templates(templates, cjk_dict)
        templates_b = translate_templates(templates, hira_dict)
        reverse_a = reverse_translate_templates(templates, cjk_dict)
        reverse_b = reverse_translate_templates(templates, hira_dict)
        terminals = all_grammar_terminals(templates)
        vocab_a = {cjk_dict[t] for t in terminals if t in cjk_dict}
        vocab_b = {hira_dict[t] for t in terminals if t in hira_dict}

        examples = build_relation_examples(final_omitted, parallel, cjk_dict, hira_dict,
                                            templates_a, templates_b, reverse_a, reverse_b,
                                            vocab_a, vocab_b, tok_probe.mask_token)
        print(f"built {len(examples)} usable mask-relation triples "
              f"(of {len(final_omitted)} final_omitted relation-track triples)")
        run_relation_probe(model_dir, examples, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
