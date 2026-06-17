"""
Evaluate cross-lingual alignment of the jointly trained multilingual MLM.

Test 1 — Word Translation Precision
    For each synset content token, look up its Language B (Hiragana) artificial
    token and its Language A (CJK) artificial token in the joint vocabulary.
    Embed Language B token in isolation (word-embedding lookup, no forward pass).
    Cosine-rank all Language A content-token embeddings as candidates.
    Metric: top-1 and top-5 precision.

Test 2 — Sentence Retrieval Precision
    Sample parallel sentence pairs from parallel_corpus_synset.json.
    Mean-pool word embeddings over Language A and Language B sentences.
    Cosine-rank Language A sentence vectors as candidates for each Language B query.
    Correct match = same triple index.
    Metric: top-1 and top-5 precision.

Usage:
    # Both tests
    python evaluate_synset.py --model ~/Desktop/coding/checkpoints_multi_synset/final

    # Test 1 only
    python evaluate_synset.py --model ... --test 1

    # Test 2 with 1000 sampled pairs
    python evaluate_synset.py --model ... --test 2 --n_sample 1000
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE = Path('/Users/pengyuwen/toy-setting-for-anchor-tokens/data')
DEFAULT_CJK     = BASE / 'semantic_backbones/dict_to_artificial/synset_pos_artificial_cjk.json'
DEFAULT_HIRA    = BASE / 'semantic_backbones/dict_to_artificial/synset_pos_artificial_hiragana.json'
DEFAULT_PARALLEL = BASE / 'corpus/parallel_corpus_synset.json'


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(model_dir, device):
    tok   = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir)
    model.to(device).eval()
    return model, tok


def word_emb_matrix(model):
    """Return static word-embedding matrix as numpy array: (vocab_size, hidden)."""
    return model.bert.embeddings.word_embeddings.weight.detach().cpu().float().numpy()


def cosine_sim(a, b):
    """
    a: (M, H)  b: (N, H)  →  (M, N) cosine-similarity matrix.
    """
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def precision_at_k(sim_matrix, correct_indices, k):
    """sim_matrix (M, N), correct_indices list[int] → scalar P@k."""
    top  = np.argsort(-sim_matrix, axis=1)[:, :k]
    hits = sum(correct_indices[i] in top[i] for i in range(len(correct_indices)))
    return hits / len(correct_indices)


# ── Test 1: Word Translation Precision ────────────────────────────────────────

def test1_word_translation(model, tokenizer, cjk_path, hira_path):
    print('── Test 1: Word Translation Precision ──────────────────────────────')

    with open(cjk_path)  as f: cjk_dict  = json.load(f)
    with open(hira_path) as f: hira_dict = json.load(f)

    vocab = tokenizer.get_vocab()   # token string → id
    emb   = word_emb_matrix(model)  # (V, H)

    # Build aligned pairs for SYNSET content tokens only
    # (grammar terminals are shared structural words — less interesting to translate)
    synset_pairs = []   # (cjk_art, hira_art, concept_key)
    for key, cjk_entry in cjk_dict.items():
        if cjk_entry['source'] != 'synsets':
            continue
        hira_entry = hira_dict[key]
        a_tok = cjk_entry['artificial']
        b_tok = hira_entry['artificial']
        if a_tok in vocab and b_tok in vocab:
            synset_pairs.append((a_tok, b_tok, key))

    print(f'  Synset content-token pairs in joint vocab : {len(synset_pairs)}')

    if not synset_pairs:
        print('  [ERROR] No aligned pairs found — is this a joint multilingual model?')
        return

    # Also build gallery restricted to Language A synset tokens
    a_tokens  = [p[0] for p in synset_pairs]
    b_tokens  = [p[1] for p in synset_pairs]
    a_ids     = np.array([vocab[t] for t in a_tokens])
    b_ids     = np.array([vocab[t] for t in b_tokens])

    a_embs = emb[a_ids]   # (M, H) — Language A (CJK) embeddings
    b_embs = emb[b_ids]   # (M, H) — Language B (Hiragana) embeddings

    # Query: Language B  →  Gallery: Language A (restricted to synset tokens)
    sim = cosine_sim(b_embs, a_embs)   # (M, M)
    correct = list(range(len(synset_pairs)))

    p1 = precision_at_k(sim, correct, k=1)
    p5 = precision_at_k(sim, correct, k=5)

    print(f'  top-1 precision : {p1:.4f}  ({p1*100:.1f}%)')
    print(f'  top-5 precision : {p5:.4f}  ({p5*100:.1f}%)')

    # Diagnostic: show a few wrong top-1 predictions
    top1 = np.argsort(-sim, axis=1)[:, 0]
    wrong = [(synset_pairs[i][2], a_tokens[top1[i]], a_tokens[i])
             for i in range(len(synset_pairs)) if top1[i] != i]
    if wrong:
        print(f'\n  Sample wrong predictions (concept | predicted_A | correct_A):')
        for concept, pred, correct_tok in wrong[:5]:
            print(f'    {concept:<40s}  pred={pred}  gold={correct_tok}')
    print()


# ── Test 2: Sentence Retrieval Precision ──────────────────────────────────────

def sentence_vec(sentence, vocab, emb_matrix):
    """Mean-pool word embeddings (skip OOV tokens)."""
    ids = [vocab[t] for t in sentence.split() if t in vocab]
    if not ids:
        return np.zeros(emb_matrix.shape[1])
    return emb_matrix[ids].mean(axis=0)


def test2_sentence_retrieval(model, tokenizer, parallel_path, n_sample=500, seed=42):
    print('── Test 2: Sentence Retrieval Precision ────────────────────────────')

    with open(parallel_path, encoding='utf-8') as f:
        data = json.load(f)

    rng = random.Random(seed)
    if len(data) > n_sample:
        data = rng.sample(data, n_sample)

    vocab = tokenizer.get_vocab()
    emb   = word_emb_matrix(model)

    a_vecs, b_vecs = [], []
    for rec in data:
        if not rec['lang_a'] or not rec['lang_b']:
            continue
        a_vecs.append(sentence_vec(rec['lang_a'][0], vocab, emb))
        b_vecs.append(sentence_vec(rec['lang_b'][0], vocab, emb))

    a_vecs = np.array(a_vecs)   # (N, H)
    b_vecs = np.array(b_vecs)   # (N, H)

    print(f'  Sentence pairs evaluated : {len(a_vecs)}')

    sim     = cosine_sim(b_vecs, a_vecs)   # (N, N)  B queries → A gallery
    correct = list(range(len(a_vecs)))

    p1 = precision_at_k(sim, correct, k=1)
    p5 = precision_at_k(sim, correct, k=5)

    print(f'  top-1 precision : {p1:.4f}  ({p1*100:.1f}%)')
    print(f'  top-5 precision : {p5:.4f}  ({p5*100:.1f}%)')
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',    required=True,
                   help='Path to trained model directory (e.g. checkpoints_multi_synset/final)')
    p.add_argument('--test',     type=int, choices=[1, 2], default=None,
                   help='Which test to run (default: both)')
    p.add_argument('--cjk',     default=str(DEFAULT_CJK))
    p.add_argument('--hiragana', default=str(DEFAULT_HIRA))
    p.add_argument('--parallel', default=str(DEFAULT_PARALLEL))
    p.add_argument('--n_sample', type=int, default=500,
                   help='Number of sentence pairs for Test 2 (default: 500)')
    p.add_argument('--seed',     type=int, default=42)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Model  : {args.model}')
    print()

    model, tokenizer = load_model(args.model, device)
    print(f'Vocab size     : {len(tokenizer)}')
    print(f'Hidden size    : {model.config.hidden_size}')
    print(f'Num parameters : {sum(p.numel() for p in model.parameters()):,}')
    print()

    run_both = args.test is None

    if run_both or args.test == 1:
        test1_word_translation(model, tokenizer, args.cjk, args.hiragana)

    if run_both or args.test == 2:
        test2_sentence_retrieval(model, tokenizer, args.parallel,
                                 n_sample=args.n_sample, seed=args.seed)


if __name__ == '__main__':
    main()
