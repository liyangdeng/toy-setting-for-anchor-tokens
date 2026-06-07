"""
PCFG sentence generator — 2-switch word order control.

Switch definitions
──────────────────
  s1  S switch     0 = NP VP  (subject before predicate)
                   1 = VP NP  (predicate before subject)

  s2  VP switch    0 = OV     (object before verb)   — TRANS only
                   1 = VO     (verb before object)   — TRANS only

COP sentences use a fixed canonical internal order:
  AUX  [ADJP]  PP  tgt      (e.g. "is located at park")

VP types produced by the grammar
──────────────────────────────────
  TRANS    — transitive verb:       {type, v}
  COP      — copula + preposition:  {type, aux, pp}
  COP_ADJ  — copula + adj + prep:   {type, aux, adjp, pp}
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Grammar parsing ────────────────────────────────────────────────────────────

def parse_grammar(grammar_str):
    rules = defaultdict(list)
    for line in grammar_str.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = re.match(r'(\w+)\s*->\s*(.+?)\s*\[([0-9.]+)\]', line)
        if not m:
            continue
        lhs = m.group(1)
        prob = float(m.group(3))
        rhs = []
        for tok in re.findall(r"'[^']*'|\w+", m.group(2).strip()):
            if tok.startswith("'"):
                rhs.append(('terminal', tok[1:-1]))
            else:
                rhs.append(('nonterminal', tok))
        rules[lhs].append((rhs, prob))
    return dict(rules)


def _weighted_choice(productions):
    probs = [p for _, p in productions]
    rhs, _ = random.choices(productions, weights=probs)[0]
    return rhs


def _sample_leaf(rules, symbol):
    """Sample one terminal from a leaf nonterminal (e.g. V, AUX, ADJP, PP)."""
    rhs = _weighted_choice(rules[symbol])
    return rhs[0][1]


# ── Structured VP sampling ─────────────────────────────────────────────────────

def sample_vp_struct(rules):
    """
    Sample a VP and return a structured dict.

    Returns one of:
        {'type': 'TRANS',   'v': str}
        {'type': 'COP',     'aux': str, 'pp': str}
        {'type': 'COP_ADJ', 'aux': str, 'adjp': str, 'pp': str}
    """
    vp_rhs = _weighted_choice(rules['VP'])
    top_sym = vp_rhs[0][1]           # 'TRANS' or 'COP'

    if top_sym == 'TRANS':
        return {'type': 'TRANS', 'v': _sample_leaf(rules, 'V')}

    cop_rhs = _weighted_choice(rules['COP'])
    cop_syms = [sym for kind, sym in cop_rhs if kind == 'nonterminal']
    aux = _sample_leaf(rules, 'AUX')
    pp  = _sample_leaf(rules, 'PP')

    if 'ADJP' in cop_syms:
        return {'type': 'COP_ADJ', 'aux': aux,
                'adjp': _sample_leaf(rules, 'ADJP'), 'pp': pp}
    return {'type': 'COP', 'aux': aux, 'pp': pp}


# ── 2-switch sentence assembly ─────────────────────────────────────────────────

def apply_switches(src, vp_struct, tgt, s1, s2):
    """
    Assemble a sentence using 2 binary switches.

    s1  S switch   0 = NP VP   1 = VP NP
    s2  VP switch  0 = OV      1 = VO     (TRANS only)

    COP sentences have fixed internal order: AUX [ADJP] PP tgt
    """
    vtype = vp_struct['type']

    if vtype == 'TRANS':
        v = vp_struct['v']
        pred = [v, tgt] if s2 == 1 else [tgt, v]

    elif vtype == 'COP':
        aux = vp_struct['aux']
        pp  = vp_struct['pp']
        pred = [aux, pp, tgt]

    else:  # COP_ADJ
        aux  = vp_struct['aux']
        adjp = vp_struct['adjp']
        pp   = vp_struct['pp']
        pred = [aux, adjp, pp, tgt]

    tokens = ([src] + pred if s1 == 0 else pred + [src])
    return ' '.join(t.replace('_', ' ') for t in tokens)


# ── High-level generation ──────────────────────────────────────────────────────

SWITCH_NAMES = {
    (0, 1): 'SVO',
    (0, 0): 'SOV',
    (1, 1): 'VOS',
    (1, 0): 'OVS',
}


def generate_sentences(edges, parsed_grammars, switches,
                       n_samples=5, max_attempts=50,
                       node_to_lemma=None):
    """
    Generate sentences for all edges under a fixed switch configuration.

    Args:
        edges:           list of edge dicts {source, relation, target, ...}
        parsed_grammars: {relation: rules_dict}
        switches:        (s1, s2) tuple of 0/1
        n_samples:       unique sentences to generate per triple
        max_attempts:    max sampling attempts per triple
        node_to_lemma:   optional {node_id: lemma_str} lookup

    Returns:
        results:  list of {source, relation, target, sentences}
        skipped:  set of relation strings that had no grammar
    """
    s1, s2 = switches
    results = []
    skipped = set()

    def get_lemma(node_id):
        if node_to_lemma and node_id in node_to_lemma:
            return node_to_lemma[node_id]
        return node_id.split('.')[0]

    for edge in edges:
        rel = edge['relation']
        if rel not in parsed_grammars:
            skipped.add(rel)
            continue

        src = get_lemma(edge['source'])
        tgt = get_lemma(edge['target'])
        rules = parsed_grammars[rel]

        sentences = set()
        attempts = 0
        while len(sentences) < n_samples and attempts < max_attempts:
            vp_struct = sample_vp_struct(rules)
            sent = apply_switches(src, vp_struct, tgt, s1, s2)
            sentences.add(sent)
            attempts += 1

        results.append({
            'source':    edge['source'],
            'relation':  edge['relation'],
            'target':    edge['target'],
            'sentences': list(sentences),
        })

    return results, skipped


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description='Generate PCFG sentences with 2 switches')
    p.add_argument('--s1', type=int, default=0, choices=[0, 1],
                   help='S switch: 0=NP VP  1=VP NP')
    p.add_argument('--s2', type=int, default=1, choices=[0, 1],
                   help='VP switch: 0=OV  1=VO  (TRANS only)')
    p.add_argument('--n_samples', type=int, default=5)
    p.add_argument('--edges',   default='../semantic_backbones/kg_with_virtual_adj/edges.json')
    p.add_argument('--synsets', default='../semantic_backbones/kg_with_virtual_adj/synsets.json')
    p.add_argument('--grammar', default='../grammar/grammar_templates.py')
    p.add_argument('--output',  default='generated_sentences.json')
    p.add_argument('--seed',    type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    switches = (args.s1, args.s2)

    import importlib.util
    spec = importlib.util.spec_from_file_location('grammar', args.grammar)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    parsed_grammars = {rel: parse_grammar(g) for rel, g in mod.grammar.items()}

    with open(args.synsets) as f:
        synsets = json.load(f)
    node_to_lemma = {
        s['id']: s['lemmas'][0].replace('_', ' ') if s.get('lemmas') else s['id'].split('.')[0]
        for s in synsets
    }

    with open(args.edges) as f:
        all_edges = json.load(f)
    edges = [e for e in all_edges if e.get('source_type') != 'virtual_adjective']
    print(f'Loaded {len(edges)} edges')

    results, skipped = generate_sentences(
        edges, parsed_grammars, switches,
        n_samples=args.n_samples,
        node_to_lemma=node_to_lemma,
    )

    label = SWITCH_NAMES.get(switches, f's1={args.s1}_s2={args.s2}')
    output = {'switches': {'s1': args.s1, 's2': args.s2}, 'results': results}
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total = sum(len(r['sentences']) for r in results)
    print(f'Config: {label} (s1={args.s1}, s2={args.s2})')
    print(f'Triples: {len(results)}  |  Sentences: {total}')
    if skipped:
        print(f'Skipped: {sorted(skipped)}')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
