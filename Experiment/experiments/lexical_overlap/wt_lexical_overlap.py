#!/usr/bin/env python3
"""
Track cross-lingual token representation alignment via Word Translation Precision (P@1, P@5)
for Lexical Overlap experiments, making results fully comparable with P@1/P@5 metrics.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast


def wrap_ids(tokenizer):
    """Returns leading [CLS] and trailing [SEP] token IDs."""
    cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    return [cls_id], [sep_id], 1  # 1 is the index of the content token


@torch.no_grad()
def token_vectors(tokens, tokenizer, model, layer, device, batch_size=256):
    lead, trail, content_idx = wrap_ids(tokenizer)
    pad = tokenizer.pad_token_id
    seqs = [lead + tokenizer(t, add_special_tokens=False)['input_ids'] + trail for t in tokens]
 
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        L = max(len(s) for s in chunk)
        ids  = torch.full((len(chunk), L), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, s in enumerate(chunk):
            ids[j, :len(s)] = torch.tensor(s); attn[j, :len(s)] = 1
        hs = model(input_ids=ids.to(device), attention_mask=attn.to(device)).hidden_states[layer]
        out.append(hs[:, content_idx, :].cpu().float().numpy())
    return np.vstack(out)


def both_directions(a, b):
    """Computes P@1 and P@5 averaged over both directions (A->B and B->A)."""
    # Normalize vectors
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    
    # Cosine similarity matrices
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
    
    # Average over both directions
    return (p1_ab + p1_ba) / 2.0, (p5_ab + p5_ba) / 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help="Path to the trained model bundle directory.")
    p.add_argument('--cjk', type=Path, required=True, help="Path to synset_pos_artificial_cjk.json")
    p.add_argument('--hiragana', type=Path, required=True, help="Path to synset_pos_artificial_hiragana.json")
    p.add_argument('--layers', default='0,-1')
    p.add_argument('--all_layers', action='store_true', help='Evaluate every hidden layer')
    p.add_argument('--save', type=Path, default=None, help='Write results to this .txt path')
    args = p.parse_args()
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}\nModel  : {args.model}')
 
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = BertForMaskedLM.from_pretrained(args.model, output_hidden_states=True).to(device).eval()
    
    n_hidden = model.config.num_hidden_layers + 1
    layers = list(range(n_hidden)) if args.all_layers else [int(x) for x in args.layers.split(',')]
    print(f'Vocab size : {len(tokenizer)} | Layers evaluated : {layers}\n')
 
    cjk_dict  = json.loads(args.cjk.read_text(encoding='utf-8'))
    hira_dict = json.loads(args.hiragana.read_text(encoding='utf-8'))
    vocab = tokenizer.get_vocab()
    
    pairs = [(c['artificial'], hira_dict[k]['artificial'])
             for k, c in cjk_dict.items()
             if c['source'] == 'synsets'
             and c['artificial'] in vocab and hira_dict[k]['artificial'] in vocab]
    a_tokens, b_tokens = zip(*pairs)
    print(f'  Synset pairs in joint vocab : {len(pairs)}')
 
    results = {}
    for layer in layers:
        a_vecs = token_vectors(list(a_tokens), tokenizer, model, layer, device)
        b_vecs = token_vectors(list(b_tokens), tokenizer, model, layer, device)
        
        p1, p5 = both_directions(a_vecs, b_vecs)
        results[layer] = (p1, p5)
        
        print(f'  layer {layer:>3} | P@1: {p1:.4f} | P@5: {p5:.4f}')
 
    if args.save is not None:
        with open(args.save, 'w', encoding='utf-8') as f:
            f.write(f'Word translation precision, special tokens\n')
            f.write(f'model  : {args.model}\n')
            f.write(f'pairs  : {len(pairs)}\n\n')
            f.write(f'{"layer":>6}  {"P@1":>12}  {"P@5":>12}\n')
            for layer, (p1, p5) in results.items():
                f.write(f'{layer:>6}  {p1:>12.4f}  {p5:>12.4f}\n')
        print(f'Saved report to: {args.save}')


if __name__ == '__main__':
    main()