#!/usr/bin/env python3
"""
Sentence-retrieval precision (P@1/P@5) for the Lexical Overlap experiment.
Mean-pools content-token hidden states and evaluates cross-lingual matching performance.
"""

import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast


def wrap_ids(tokenizer):
    """Return leading [CLS] and trailing [SEP] token IDs."""
    cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    return [cls_id], [sep_id]


@torch.no_grad()
def sentence_vectors(sentences, tokenizer, model, layer, device, max_length=64, batch_size=128):
    pad_id = tokenizer.pad_token_id
    lead, trail = wrap_ids(tokenizer)
    start = len(lead)                    
    trail_len = len(trail)               
 
    seqs = []
    for s in sentences:
        ids = tokenizer(s, add_special_tokens=False)['input_ids']
        seqs.append((lead + ids + trail)[:max_length])
 
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        L = max(len(s) for s in chunk)
        ids  = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, s in enumerate(chunk):
            ids[j, :len(s)] = torch.tensor(s); attn[j, :len(s)] = 1
        hs = model(input_ids=ids.to(device), attention_mask=attn.to(device)).hidden_states[layer]
        hs = hs.cpu().float().numpy()
        for j, s in enumerate(chunk):
            end = len(s) - trail_len
            out.append(hs[j, start:end].mean(axis=0))
    return np.array(out)


def both_directions(a, b):
    """Computes sentence retrieval P@1 and P@5 averaged over both directions."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    
    sim_ab = np.dot(a_norm, b_norm.T)
    sim_ba = sim_ab.T
    
    n = len(a)
    correct_idx = np.arange(n)
    
    # Direction A -> B
    ranks_ab = np.argsort(-sim_ab, axis=1)
    p1_ab = np.sum(ranks_ab[:, 0] == correct_idx) / n
    p5_ab = np.sum([correct_idx[i] in ranks_ab[i, :5] for i in range(n)]) / n
    
    # Direction B -> A
    ranks_ba = np.argsort(-sim_ba, axis=1)
    p1_ba = np.sum(ranks_ba[:, 0] == correct_idx) / n
    p5_ba = np.sum([correct_idx[i] in ranks_ba[i, :5] for i in range(n)]) / n
    
    return (p1_ab + p1_ba) / 2.0, (p5_ab + p5_ba) / 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help="Path to the trained model bundle directory.")
    p.add_argument('--parallel', type=Path, required=True, help="Path to parallel_corpus_synset.json")
    p.add_argument('--layers', default='0,-1')
    p.add_argument('--all_layers', action='store_true', help='Evaluate every hidden layer')
    p.add_argument('--n_sample', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save', type=Path, default=None, help='Write results to this .txt path')
    args = p.parse_args()
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}\nModel  : {args.model}')
 
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = BertForMaskedLM.from_pretrained(args.model, output_hidden_states=True).to(device).eval()
    
    n_hidden = model.config.num_hidden_layers + 1
    layers = list(range(n_hidden)) if args.all_layers else [int(x) for x in args.layers.split(',')]
    print(f'Vocab size : {len(tokenizer)} | Layers evaluated : {layers}\n')
 
    data = json.loads(args.parallel.read_text(encoding='utf-8'))
    data = [r for r in data if r['lang_a'] and r['lang_b']]
    rng = random.Random(args.seed)
    if len(data) > args.n_sample:
        data = rng.sample(data, args.n_sample)
    a_sents = [r['lang_a'][0] for r in data]
    b_sents = [r['lang_b'][0] for r in data]
    print(f'  Sentence pairs evaluated : {len(a_sents)}')
 
    results = {}
    for layer in layers:
        a_vecs = sentence_vectors(a_sents, tokenizer, model, layer, device, batch_size=128)
        b_vecs = sentence_vectors(b_sents, tokenizer, model, layer, device, batch_size=128)
        
        p1, p5 = both_directions(a_vecs, b_vecs)
        results[layer] = (p1, p5)
        
        print(f'  layer {layer:>3} | P@1: {p1:.4f} | P@5: {p5:.4f}')
 
    if args.save is not None:
        with open(args.save, 'w', encoding='utf-8') as f:
            f.write(f'Sentence-level evaluation, P@1 and P@5\n')
            f.write(f'model  : {args.model}\n')
            f.write(f'pairs  : {len(a_sents)}\n\n')
            f.write(f'{"layer":>6}  {"P@1":>12}  {"P@5":>12}\n')
            for layer, (p1, p5) in results.items():
                f.write(f'{layer:>6}  {p1:>12.4f}  {p5:>12.4f}\n')
        print(f'Saved report to: {args.save}')


if __name__ == '__main__':
    main()