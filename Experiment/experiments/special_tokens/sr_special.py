"""
Sentence-retrieval precision (rho) for the special-token experiment.
 
Method (contextual, D&S):
    embed each parallel sentence, mean-pool its content-token hidden states
    (skipping [CLS] if present and the trailing [SEP]/[END] if present),
    cosine-rank against the other language's sentences. Report P@1/P@5 averaged
    over BOTH directions, at layers 0 and last (default).
 
Per-language wrapping read from special_config.json next to the model.
A=CJK, B=Hiragana.
 
Usage:
    python sr_special.py \
        --model checkpoints_special_shared/final \
        --parallel parallel_corpus_synset.json \
        --layers 0,-1  --n_sample 500
"""
 
import argparse
import json
import random
from pathlib import Path
 
import numpy as np
import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast
 
 
def load_special(model_dir, setting):
    """Only 'disjoint' needs the sidecar."""
    if setting == 'shared':
        return {'a': '[CLS]', 'b': '[CLS]'}, {'a': '[SEP]', 'b': '[SEP]'}
    if setting == 'none':
        return {'a': None, 'b': None}, {'a': None, 'b': None}
    cfg = json.loads((Path(model_dir) / 'special_config.json').read_text(encoding='utf-8'))
    return cfg['cls'], cfg['sep']
 
 
def wrap_ids(tokenizer, lang, cls_map, sep_map):
    """Return (leading_ids, trailing_ids). Either may be empty (arm 'none')."""
    cls = cls_map[lang]; sep = sep_map[lang]
    lead  = []  if cls is None else [tokenizer.convert_tokens_to_ids(cls)]
    trail = []  if sep is None else [tokenizer.convert_tokens_to_ids(sep)]
    return lead, trail
 
 
@torch.no_grad()
def sentence_vectors(sentences, lang, tokenizer, model, cls_map, sep_map, layer, device,
                     max_length=64, batch_size=128):
    """Mean-pool content-token hidden states, skipping leading/trailing specials."""
    pad_id = tokenizer.pad_token_id
    lead, trail = wrap_ids(tokenizer, lang, cls_map, sep_map)
    start = len(lead)                    # first content index
    trail_len = len(trail)               # 0 or 1
 
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
    p.add_argument('--setting', choices=['shared', 'none', 'disjoint'], required=True,
                   help="'shared'/'none' need no sidecar; 'disjoint' reads special_config.json")
    p.add_argument('--parallel', type=Path, required=True)
    p.add_argument('--layers', default='0,-1')
    p.add_argument('--all_layers', action='store_true')
    p.add_argument('--n_sample', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save', type=Path, default=None, help='write results to this .txt path')
    args = p.parse_args()
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}\nModel  : {args.model}')
 
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = BertForMaskedLM.from_pretrained(args.model, output_hidden_states=True).to(device).eval()
    cls_map, sep_map = load_special(args.model, args.setting)
    print(f"  setting={args.setting}  cls A={cls_map['a']} B={cls_map['b']}  "
          f"sep A={sep_map['a']} B={sep_map['b']}")
    n_hidden = model.config.num_hidden_layers + 1
    layers = list(range(n_hidden)) if args.all_layers else [int(x) for x in args.layers.split(',')]
    print(f'Vocab size : {len(tokenizer)} | Hidden : {model.config.hidden_size} '
          f'| Layers evaluated : {layers}\n')
 
    data = json.loads(args.parallel.read_text(encoding='utf-8'))
    data = [r for r in data if r['lang_a'] and r['lang_b']]
    rng = random.Random(args.seed)
    if len(data) > args.n_sample:
        data = rng.sample(data, args.n_sample)
    a_sents = [r['lang_a'][0] for r in data]
    b_sents = [r['lang_b'][0] for r in data]
    print(f'  Sentence pairs evaluated : {len(a_sents)}')
 
    rho = {}
    for layer in layers:
        a_vecs = sentence_vectors(a_sents, 'a', tokenizer, model, cls_map, sep_map, layer, device)
        b_vecs = sentence_vectors(b_sents, 'b', tokenizer, model, cls_map, sep_map, layer, device)
        p1, p5 = both_directions(a_vecs, b_vecs)
        rho[layer] = (p1, p5)
        print(f'  layer {layer:>3}  rho P@1 : {p1:.4f}   P@5 : {p5:.4f}')
 
    mean_rho = sum(p1 for p1, _ in rho.values()) / len(rho)
    print(f'  mean rho over layers {list(rho)} : {mean_rho:.4f}')
 
    if args.save is not None:
        with open(args.save, 'w', encoding='utf-8') as f:
            f.write(f'model  : {args.model}\n')
            f.write(f'pairs  : {len(a_sents)}\n')
            f.write(f'layers : {list(rho)}\n\n')
            f.write(f'{"layer":>6}  {"P@1":>8}  {"P@5":>8}\n')
            for layer, (p1, p5) in rho.items():
                f.write(f'{layer:>6}  {p1:>8.4f}  {p5:>8.4f}\n')
            f.write(f'\nmean rho (P@1) over layers {list(rho)} : {mean_rho:.4f}\n')
        print(f'  saved -> {args.save}')
 
 
if __name__ == '__main__':
    main()