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

    top_sym = None
    for kind, sym in vp_rhs:
        if kind == 'nonterminal' and sym in ['TRANS', 'COP']:
            top_sym = sym
            break
            
    if not top_sym:
        top_sym = 'TRANS'

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

# ── Recursive phrase sampling ─────────────────────────────────────────────────

def sample_phrase(rules, symbol):
    """
    Recursively sample a phrase from the grammar starting at the given symbol.
    """
    if symbol not in rules:
        if (symbol.startswith("'") and symbol.endswith("'")) or (symbol.startswith('"') and symbol.endswith('"')):
            return [symbol[1:-1]]
        return [symbol]
        
    rhs = _weighted_choice(rules[symbol])
    tokens = []
    for kind, sym in rhs:
        if kind == 'terminal':
            tokens.append(sym)
        else:
            tokens.extend(sample_phrase(rules, sym))
    return tokens


# ── 2-switch sentence assembly ─────────────────────────────────────────────────

def apply_switches(src, vp_struct, tgt, s1, s2):
    """
    Assemble a sentence using 2 binary switches.

    All tokens (entity synset IDs and grammar terminals) are kept as-is with
    underscores intact, matching the dictionary keys exactly.  No underscore
    replacement is applied: multi-word grammar concepts such as 'more_general'
    or 'belongs_to' remain single tokens, and synset IDs such as
    'abdominal_wall.n.01' likewise stay intact.
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
    return ' '.join(tokens)


# ── High-level generation ──────────────────────────────────────────────────────

SWITCH_NAMES = {
    (0, 1): 'SVO',
    (0, 0): 'SOV',
    (1, 1): 'VOS',
    (1, 0): 'OVS',
}

ALLOWED_EXPANSIONS = {
    "hypernym": "hyponym",
    "hyponym": "hypernym",
    "instance_hypernym": "instance_hyponym",
    "instance_hyponym": "instance_hypernym",

    "part_meronym": "part_holonym",
    "part_holonym": "part_meronym",
    "member_meronym": "member_holonym",
    "member_holonym": "member_meronym",
    "substance_meronym": "substance_holonym",
    "substance_holonym": "substance_meronym",

    "AtLocation": ["hypernym", "hyponym"],
    "UsedFor": ["hypernym", "hyponym"],
    "SimilarTo": "DistinctFrom",
    "Antonym": "SimilarTo",
    "CapableOf": ["hypernym", "hyponym"],

    "HasProperty": ["hypernym", "hyponym"],
    "DistinctFrom": "Antonym",
    "MadeOf": "UsedFor",
    "Desires": "NotDesires",

    "CausesDesire": ["hypernym", "hyponym"],
    "HasFirstSubevent": "HasLastSubevent",
    "HasLastSubevent": "HasFirstSubevent",
}

def generate_sentences(edges, parsed_grammars, switches, n_samples=7, node_to_lemma=None):
    """
    Generate sentences for all edges under a fixed switch configuration.
    """
    s1, s2 = switches
    results = []
    skipped = []

    from collections import defaultdict
    edges_by_source = defaultdict(list)
    for edge in edges:
        edges_by_source[edge['source']].append(edge)


    for edge in edges:
        rel = edge['relation']
        src = edge['source']
        tgt = edge['target']

        if rel not in parsed_grammars:
            skipped.append(rel)
            continue

        rules = parsed_grammars[rel]

        sentences = set()
        attempts = 0
        max_attempts = 50

        # 1. Generate simple sentences
        n_simple = min(5, n_samples) 
        while len(sentences) < n_simple and attempts < max_attempts:
            vp_struct = sample_vp_struct(rules)
            sent = apply_switches(src, vp_struct, tgt, s1, s2)
            sentences.add(sent)
            attempts += 1

        # 2. Try to generate complex sentences by finding sibling edges with the same source
        sibling_edges = [se for se in edges_by_source[src] if se != edge]
        
        if sibling_edges:
            allowed = ALLOWED_EXPANSIONS.get(rel)
            target_relations = [allowed] if isinstance(allowed, str) else (allowed if allowed else [])
          
            same_rel_edges = [se for se in sibling_edges if se['relation'] == rel]
            expansion_edges = [
                se for se in sibling_edges 
                if se['relation'] in target_relations and f"VP_EXP_{se['relation']}" in rules
            ]
            
            while len(sentences) < n_samples and attempts < max_attempts:
                attempts += 1
                
                available_choices = []
                if same_rel_edges:
                    available_choices.append(('same', same_rel_edges))
                if expansion_edges:
                    available_choices.append(('expand', expansion_edges))
                
                if not available_choices:
                    break
                
                choice_type, edges_list = random.choice(available_choices)
                second_edge = random.choice(edges_list)
                
                rel_2 = second_edge['relation']
                tgt_2 = second_edge['target']
                
                # Both edges share the same relation — try to coordinate the two targets with "and"
                if choice_type == 'same':
                    vp_struct_1 = sample_vp_struct(rules)
                    part_1 = apply_switches(src, vp_struct_1, tgt, s1, s2)
                    complex_sent = f"{part_1} and {tgt_2}"
                    sentences.add(complex_sent)
                
                # The second edge's relation has a corresponding expansion rule
                elif choice_type == 'expand':
                    exp_key = f"VP_EXP_{rel_2}"
                    raw_tokens = sample_phrase(rules, exp_key)
                    raw_sentence = " ".join(raw_tokens)
                    
                    complex_sent = f"{src} {raw_sentence}"
                    complex_sent = complex_sent.replace('tgt_2', tgt_2).replace('tgt', tgt)
                    sentences.add(" ".join(complex_sent.split()))

        results.append({
            'source': src,
            'relation': rel,
            'target': tgt,
            'sentences': list(sentences)
        })

    return results, set(skipped)

# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description='Generate PCFG sentences with 2 switches')
    p.add_argument('--s1', type=int, default=0, choices=[0, 1],
                   help='S switch: 0=NP VP  1=VP NP')
    p.add_argument('--s2', type=int, default=1, choices=[0, 1],
                   help='VP switch: 0=OV  1=VO  (TRANS only)')
    p.add_argument('--n_samples', type=int, default=7)
    p.add_argument('--edges',   default='edges.json')
    p.add_argument('--synsets', default='synsets.json')
    p.add_argument('--grammar', default='grammar_templates_extended.py')
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
