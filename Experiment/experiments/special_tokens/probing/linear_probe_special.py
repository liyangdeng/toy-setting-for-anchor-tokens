"""
Linear probing for the SPECIAL TOKEN* experiment -- mask-ENTITY track.

* theoretically should work with punctuation too. 

Two sections:
  1. TRAINING -- pasted from train_multilingual_synset.py almost verbatim,
     wrapped in train_multilingual() so this one file can train a bilingual
     model from scratch if you don't already have a checkpoint.
  2. LINEAR PROBE. For every probe triple:
       - take its A sentence(s) and B sentence(s) (from parallel_corpus_
         synset.json) and mask the single TARGET entity token (kept as
         [MASK] in the sentence, not deleted -- avoids leakage since the
         answer token is gone from the input, while the sentence structure
         the model was trained on stays intact)
       - run the frozen model, output_hidden_states=True
       - take the hidden state at the [MASK] position, for every layer
     Then, per layer L:
       - fit a linear (logistic regression) classifier on the A vectors,
         label = target concept id (shared A/B via the synset dicts)
       - apply that SAME classifier to the B vectors
       - a TRIPLE counts as correctly-transferred if ANY of its (possibly
         several) B renderings is classified correctly (OR-aggregation --
         see run_probe). Final accuracy = hit triples / total triples, not
         per-sentence accuracy. Training likewise uses every triple's every
         A rendering as a separate row under the same label, for more
         context diversity per class.
     Reports one accuracy curve (layer 0..N) overall and per relation, saves
     a CSV and a PNG plot.

Special-token experiment support:
if there is special_config.json (written by train_special.py), load:
1. find_all_masked uses each language's own mask string (e.g. [HID] for CJK vs [MASK] for Hiragana)
2. extract_all_layers uses each language's own mask id when locating the [MASK] position
3. the post-processor was meant to match the way the model was trained

usage (train fresh + probe):
    python linear_probe.py \
        --do_train \
        --corpus_a probing_run/a_training.txt --corpus_b probing_run/b_training.txt \
        --train_output_dir ./checkpoints_treatment \
        --final_omitted probing_run/final_omitted.json \
        --parallel probing_run/final_omitted_corpus/parallel_corpus_synset.json \
        --cjk_dict synset_pos_artificial_cjk_edges_adj_augmented.json \
        --hira_dict synset_pos_artificial_hiragana_edges_adj_augmented.json \
        --out_dir ./probe_results_treatment

usage (probe an existing checkpoint):
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
from tokenizers.processors import TemplateProcessing
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
 
 
def load_special_config(model_dir):
    """
    read special_config.json written by train_special.py)
    return:
      {'a': {cls,sep,mask}, 'b': {...}} 
    warns if the config does not exist. probe falls back to [MASK]
    and doesn't use the tokenizer's post-processor

    """
    cfg_path = Path(model_dir) / "special_config.json"
    if not cfg_path.exists():
        print(f"NO special_config.json at {cfg_path} (!!!)")
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {"a": cfg["a"], "b": cfg["b"]}
 
 
def apply_template(tokenizer, cls_tok, sep_tok):
    """
    same behaviour as train_special.py's set_template:
    post-processor wraps sentences with start and end tokens is needed
    """
    if cls_tok is None:
        tokenizer.backend_tokenizer.post_processor = TemplateProcessing(single="$A")
        return
    ids = tokenizer.convert_tokens_to_ids
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single=f"{cls_tok} $A {sep_tok}",
        special_tokens=[(cls_tok, ids(cls_tok)), (sep_tok, ids(sep_tok))],
    )
 
 
def sanity_check_wrapping(tokenizer, policy):
    """
    for each language, apply the setting's post-processor and print
    the token sequence produced. confirms that the
    boundary tokens are placed correctly, and each language's mask
    token does not get replaced by [UNK].
 
    output supposed to be:
        A/CJK       'foo [HID] bar' -> ['[BEG]', '[UNK]', '[HID]', '[UNK]', '[END]']
        B/Hiragana  'foo [MASK] bar' -> ['[CLS]', '[UNK]', '[MASK]', '[UNK]', '[SEP]']
    """
    print("Tokenizer output per language:")
    for lang, label in (("a", "CJK"), ("b", "Hiragana")):
        apply_template(tokenizer, policy[lang]["cls"], policy[lang]["sep"])
        mask = policy[lang]["mask"]
        stub = f" {mask} "
        ids = tokenizer(stub)["input_ids"]
        toks = tokenizer.convert_ids_to_tokens(ids)
        print(f"  {label} '{stub}' -> {toks}")
 
 
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
 
 
def build_examples(final_omitted, parallel, cjk_dict, hira_dict,
                    mask_str_a, mask_str_b):
    """
    final_omitted: mask-ENTITY probes (each tagged "track": "entity" by
    build_probing_corpus.py).
 
    parallel: the FRESH parallel file built from final_omitted.json in
    build_probing_corpus.py's step 8 (final_omitted_corpus/parallel_corpus_
    synset.json) -- NOT the old full-corpus parallel file. Its lang_a
    sentences are exactly what got appended into a_training.txt, so the model
    actually trained on them.
 
    mask_str_a / mask_str_b: which token to insert at the masked position
    for each language. Same string for shared/none arms ("[MASK]") and
    different for the disjoint special-token arm ("[HID]" for CJK,
    "[MASK]" for Hiragana).
 
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
        a_sents = find_all_masked(entry["lang_a"], a_tok, mask_str_a)
        b_sents = find_all_masked(entry["lang_b"], b_tok, mask_str_b)
        if not a_sents or not b_sents:
            continue
        examples.append({
            "a_sents": a_sents, "b_sents": b_sents,
            "concept": p["target"], "relation": p["relation"],
        })
    return examples
 
 
@torch.no_grad()
def extract_all_layers(model, tokenizer, sentences, device, mask_id=None,
                        max_length=64, batch_size=32):
    """
    return a list (length = num_layers+1) of np.arrays [n_sentences, hidden]:
    for every layer, the hidden state at the [MASK] position.
 
    mask_id: which token id marks the mask position. 

    works for shared/none and for checkpoints without special_config.json.
    for the special-token disjoint setting, pass the id of [HID] for CJK and [MASK] for Hiragana.
    """
    n_layers = model.config.num_hidden_layers + 1
    out = [[] for _ in range(n_layers)]
    if mask_id is None:
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
    Training uses EVERY triple's EVERY A-sentence rendering as a separate
    row under that triple's concept label (more context diversity per
    class). Evaluation runs the frozen-on-A classifier on EVERY triple's
    EVERY B-sentence rendering, then OR-aggregates per triple: a triple
    counts as correctly-transferred if ANY of its B renderings is
    classified correctly. Final accuracy = hit triples / total triples
    (not per-sentence accuracy).
 
    {'a': {cls,sep,mask}, 'b': {...}} from load_special_config
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
 
    # CJK extraction
    if policy is not None:
        apply_template(tokenizer, policy["a"]["cls"], policy["a"]["sep"])
        mask_id_a = tokenizer.convert_tokens_to_ids(policy["a"]["mask"])
    else:
        mask_id_a = None
    X_a_layers = extract_all_layers(model, tokenizer, train_sents, device,
                                     mask_id=mask_id_a)
 
    # Hiragana extraction
    if policy is not None:
        apply_template(tokenizer, policy["b"]["cls"], policy["b"]["sep"])
        mask_id_b = tokenizer.convert_tokens_to_ids(policy["b"]["mask"])
    else:
        mask_id_b = None
    X_b_layers = extract_all_layers(model, tokenizer, eval_sents, device,
                                     mask_id=mask_id_b)
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
 
    final_omitted = [p for p in json.load(open(args.final_omitted)) if p.get("track") == "entity"]
    parallel = json.load(open(args.parallel))
    cjk_dict = {k: v["artificial"] for k, v in json.load(open(args.cjk_dict)).items()}
    hira_dict = {k: v["artificial"] for k, v in json.load(open(args.hira_dict)).items()}
 
    tok_probe = PreTrainedTokenizerFast.from_pretrained(model_dir)
    policy = load_special_config(model_dir)

    if policy is not None:
        print("found special_config.json")
        print(f"CJK       cls={policy['a']['cls']}  sep={policy['a']['sep']}  mask={policy['a']['mask']}")
        print(f"Hiragana  cls={policy['b']['cls']}  sep={policy['b']['sep']}  mask={policy['b']['mask']}")
        sanity_check_wrapping(tok_probe, policy)
        mask_str_a = policy["a"]["mask"]
        mask_str_b = policy["b"]["mask"]
    else:
        mask_str_a = mask_str_b = tok_probe.mask_token
 
    examples = build_examples(final_omitted, parallel, cjk_dict, hira_dict, mask_str_a, mask_str_b)
    print(f"built {len(examples)} usable mask-entity triples "
          f"(of {len(final_omitted)} final_omitted entity-track triples)")
    run_probe(model_dir, examples, args.out_dir, policy=policy, seed=args.seed)
 
 
if __name__ == "__main__":
    main()