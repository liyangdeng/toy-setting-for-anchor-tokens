"""
Word-translation precision for punctuation experiment

needs boundary_config.json as a source for the disjoint [SEP] variant

usage:
    python wt_punct.py \
        --setting shared
        --model checkpoints_multi_synset/final \
        --cjk .../synset_pos_artificial_cjk.json \
        --hiragana .../synset_pos_artificial_hiragana.json \
        --all_layers --save results_shared.tx
"""

import argparse
import json
from pathlib import Path
 
import numpy as np
import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast
 
 
def load_boundary(model_dir, setting):

    if setting == 'shared':
        return {'a': '[SEP]', 'b': '[SEP]'}
    if setting == 'nopunct':
        return {'a': None, 'b': None}
    return json.loads((Path(model_dir) / 'boundary_config.json').read_text(encoding='utf-8'))['boundary']
 
 
def wrap_ids(tokenizer, lang, boundary):

    sep_tok = boundary[lang]
    trailing = [] if sep_tok is None else [tokenizer.convert_tokens_to_ids(sep_tok)]
    return trailing, 1
 
 
@torch.no_grad()
def token_vectors(tokens, lang, tokenizer, model, boundary, layer, device, batch_size=256):

    cls_id = tokenizer.cls_token_id
    pad_id = tokenizer.pad_token_id
    trailing, widx = wrap_ids(tokenizer, lang, boundary)
    seqs = [[cls_id] + tokenizer(t, add_special_tokens=False)['input_ids'] + trailing for t in tokens]
    out = []

    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        L = max(len(s) for s in chunk)
        ids  = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, s in enumerate(chunk):
            ids[j, :len(s)] = torch.tensor(s); attn[j, :len(s)] = 1
        hs = model(input_ids=ids.to(device), attention_mask=attn.to(device)).hidden_states[layer]
        out.append(hs[:, widx, :].cpu().float().numpy())

    return np.vstack(out)
 
 
def cosine_sim(a, b):

    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T
 
 
def precision_at_k(sim, k):

    top = np.argsort(-sim, axis=1)[:, :k]
    return sum(i in top[i] for i in range(len(sim))) / len(sim)
 
 
def both_directions(a_vecs, b_vecs):

    ba = cosine_sim(b_vecs, a_vecs)
    ab = cosine_sim(a_vecs, b_vecs)

    return (0.5 * (precision_at_k(ba, 1) + precision_at_k(ab, 1)),
            0.5 * (precision_at_k(ba, 5) + precision_at_k(ab, 5)))
 
 
def main():

    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--setting', choices=['shared', 'disjoint', 'nopunct'], required=True)
    p.add_argument('--cjk', type=Path, required=True)
    p.add_argument('--hiragana', type=Path, required=True)
    p.add_argument('--layers', default='0,-1')
    p.add_argument('--all_layers', action='store_true',
                   help='report tau at every hidden layer (overrides --layers)')
    p.add_argument('--save', type=Path, default=None, help='write results to this .txt path')
    args = p.parse_args()
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}\nModel  : {args.model}')
 
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = BertForMaskedLM.from_pretrained(args.model, output_hidden_states=True).to(device).eval()
    boundary = load_boundary(args.model, args.setting)
    print(f'  setting={args.setting}  boundary: A/CJK={boundary["a"]}  B/Hiragana={boundary["b"]}')
    n_hidden = model.config.num_hidden_layers + 1
    layers = list(range(n_hidden)) if args.all_layers else [int(x) for x in args.layers.split(',')]
    print(f'Vocab size : {len(tokenizer)} | Hidden : {model.config.hidden_size} '
          f'| Layers evaluated : {layers}\n')
 
    cjk_dict  = json.loads(args.cjk.read_text(encoding='utf-8'))
    hira_dict = json.loads(args.hiragana.read_text(encoding='utf-8'))
    vocab = tokenizer.get_vocab()
    pairs = [(c['artificial'], hira_dict[k]['artificial'])
             for k, c in cjk_dict.items()
             if c['source'] == 'synsets'
             and c['artificial'] in vocab and hira_dict[k]['artificial'] in vocab]
    a_tokens, b_tokens = zip(*pairs)
    print(f'  Synset pairs in joint vocab : {len(pairs)}')
 
    tau = {}
    for layer in layers:
        a_vecs = token_vectors(list(a_tokens), 'a', tokenizer, model, boundary, layer, device)
        b_vecs = token_vectors(list(b_tokens), 'b', tokenizer, model, boundary, layer, device)
        p1, p5 = both_directions(a_vecs, b_vecs)
        tau[layer] = (p1, p5)
        print(f'  layer {layer:>3}  tau P@1 : {p1:.4f}   P@5 : {p5:.4f}')
 
    mean_tau = sum(p1 for p1, _ in tau.values()) / len(tau)
    print(f'  mean tau over layers {list(tau)} : {mean_tau:.4f}')
 
    if args.save is not None:
        with open(args.save, 'w', encoding='utf-8') as f:
            f.write(f'Word tranlsation precision, punctuation\n')
            f.write(f'model  : {args.model}\n')
            f.write(f'pairs  : {len(pairs)}\n')
            f.write(f'layers : {list(tau)}\n\n')
            f.write(f'{"layer":>6}  {"P@1":>8}  {"P@5":>8}\n')
            for layer, (p1, p5) in tau.items():
                f.write(f'{layer:>6}  {p1:>8.4f}  {p5:>8.4f}\n')
            f.write(f'\nmean tau (P@1) over layers {list(tau)} : {mean_tau:.4f}\n')
        print(f'  saved -> {args.save}')
 
 
if __name__ == '__main__':
    main()