"""
PCFG sentence generator - word-order switches (22.06)
+ adjectives (23.06)
+ more adjective variety (24.06)
+ questions (28.06)

last update: 28.06.2026


Switch definitions
-----------------------------------------
  s1  S switch     0 = NP VP   (subject before predicate)
                   1 = VP NP   (predicate before subject)

  s2  VP switch    0 = OV      (object before verb)   - TRANS only
                   1 = VO      (verb before object)   - TRANS only

(new in this version)
  s3  N-A switch   0 = A N     (adjective before noun)
                   1 = N A     (noun before adjective)
                   Only effective when an adjective is actually inserted.

Adjectives in NPs
-----------------------------------------
      NP -> A N [p] | N [1-p]      (normal relations)
      NP -> N [1.0]                (HasProperty / NotHasProperty)

  Adjectives come from HasQuality edges
  (e.g. "guru, HasQuality, wise").
  When the NP rule draws A N AND the noun has HasQuality entries AND the
  adj_when_available fires, the chosen adj is prepended (s3=0) 
  or appended (s3=1) to the noun. 
  Otherwise the NP falls back to just the noun.

    Reminder: "adjectives" that are sampled via "HasProperty" (e.g. "green" 
    in "grass HasProperty green") are never being treated as adjectives. 
    We treat them as nouns. The switch does not apply there since the phrase 
    only consists of 1 word anyways.

    (new in this version)
    Now we sometimes sample sentences that differ only in an adjective.
    E.g., possible:
    "striped bee remains situated within green jar.",
    "small bee remains situated within green jar.",
    "striped bee remains situated within fragile jar.",
    "small bee remains situated within fragile jar.",

This script uses synsets
-----------------------------------------
  All input tokens come from synsets. In the resulting sentences, each and every
  KG entity appears in its synset form, e.g. conceptnet:bee:1509.

VP types produced by the grammar
-----------------------------------------
  TRANS    — transitive verb:       {type, v}
  COP      — copula + preposition:  {type, aux, pp}
  COP_ADJ  — copula + adj + prep:   {type, aux, adjp, pp}

____________________________________________________________________________________

Usage:

python v2_generate_pcfg_sentences_synsets_adj_more.py \
  --edges edges_adj.json \
  --grammar grammar_templates_adj.py \
  --s1 0 --s2 1 --s3 0 \
  --adj_when_available 0.8 \
  --adj_variant_prob 0.3 \
  --max_adj_variants 6 \
  --n_samples 8

Remember to adjust the adjective occurrence rate:
--adj_when_available
how much we sample from an edge:
--n_samples

max_adj_variants = x means: for a noun with multiple adjectives available, 
control how many of them can appear paired with this noun
(thus how many minimal pair sentences can occur).

To create a corpus with no adjectives, choose:
--adj_variant_prob 0.0 --max_adj_variants 0

"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


# ___ Grammar parsing ________________________________________________

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


NO_ADJ_RELATIONS = {'HasProperty', 'NotHasProperty'}


def _weighted_choice(productions):
    probs = [p for _, p in productions]
    rhs, _ = random.choices(productions, weights=probs)[0]
    return rhs


def _sample_leaf(rules, symbol):
    """Sample one terminal from a leaf nonterminal (e.g. V, AUX, ADJP, PP)."""
    rhs = _weighted_choice(rules[symbol])
    return rhs[0][1]


# ___ Structured VP sampling ________________________________________________

def sample_vp_struct(rules):
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


# ___ HasQuality lookup + NP realization ________________________________________________

def build_quality_lookup(edges):
    """
    Build {node_id: [adjective tokens]} from HasQuality edges. Adjective
    tokens are stored verbatim — no stripping of prefixes/suffixes/underscores.
    """
    lookup = defaultdict(list)
    for edge in edges:
        if edge.get('relation') != 'HasQuality':
            continue
        lookup[edge['source']].append(edge['target'])
    return dict(lookup)


def sample_np_form(rules):
    """
    Sample one NP -> ... draw. Returns 'A_N' if the chosen RHS contains an
    A slot, otherwise 'N'. Falls back to 'N' if the grammar declares no NP rule.
    """
    if 'NP' not in rules:
        return 'N'
    rhs = _weighted_choice(rules['NP'])
    syms = {sym for kind, sym in rhs if kind == 'nonterminal'}
    return 'A_N' if 'A' in syms else 'N'


def realize_np(noun_token, node_id, rules, quality_lookup,
               s3=0, adj_when_available=0.8):
    """
    Return the surface form of one NP. Falls back to the bare noun when:
      - the NP rule drew the N-only branch this sample, OR
      - the noun has no HasQuality entries in the lookup, OR
      - adj_when_available fails.

    s3 = 0  ->  A N
    s3 = 1  ->  N A

    """
    if sample_np_form(rules) != 'A_N':
        return noun_token
    adjectives = quality_lookup.get(node_id, [])
    if not adjectives:
        return noun_token
    if random.random() >= adj_when_available:
        return noun_token
    adj = random.choice(adjectives)
    return f"{adj} {noun_token}" if s3 == 0 else f"{noun_token} {adj}"


def expand_adjective_variants(sent, node_ids, quality_lookup, s3=0,
                              max_variants_per_noun=3):
    """
    Generate minimal-pair variants of `sent` by substituting alternative
    adj for any node in `node_ids` whose current adj is realised
    in the sentence. For each such node, up to (max_variants_per_noun - 1)
    alternative adj are randomly chosen from its HasQuality list, in
    addition to the one that's already present.

    If both src and tgt have smapled adjectives with multiple alternatives,
    the result is the cartesian across nodes.

    The returned set always includes the original sentence.
    """
    variants = {sent}
    for node_id in node_ids:
        alternatives = quality_lookup.get(node_id, [])
        if len(alternatives) <= 1:
            continue

        current_adj = None
        for adj in alternatives:
            pattern = f"{adj} {node_id}" if s3 == 0 else f"{node_id} {adj}"
            if pattern in sent:
                current_adj = adj
                break
        if current_adj is None:
            continue 

        others = [a for a in alternatives if a != current_adj]
        random.shuffle(others)
        chosen = others[:max(0, max_variants_per_noun - 1)]
        if not chosen:
            continue
        current_pattern = (f"{current_adj} {node_id}" if s3 == 0
                           else f"{node_id} {current_adj}")
        new_variants = set()
        for v in variants:
            if current_pattern not in v:
                continue
            for other_adj in chosen:
                new_pattern = (f"{other_adj} {node_id}" if s3 == 0
                               else f"{node_id} {other_adj}")
                new_variants.add(v.replace(current_pattern, new_pattern))
        variants.update(new_variants)
    return variants


# ___ Recursive phrase sampling __________________________________________

def sample_phrase(rules, symbol):
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


# ___ Sentence assembly ______________________________________________________

def apply_switches(src, vp_struct, tgt, s1, s2):
    """src and tgt are already-realized NP strings, not bare node IDs."""
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


# ___ High-level generation ___________________________________________________

SWITCH_NAMES = {
    (0, 1): 'SVO',
    (0, 0): 'SOV',
    (1, 1): 'VOS',
    (1, 0): 'OVS',
}

ALLOWED_EXPANSIONS = {
    "hypernym": "hyponym",
    "hyponym": "hypernym",
    "AtLocation": ["hypernym", "hyponym"],
    "UsedFor": ["hypernym", "hyponym"],
    "Antonym": "SimilarTo",
    "CapableOf": ["hypernym", "hyponym"],
    "HasPrerequisite": "HasSubevent",
    "HasProperty": ["hypernym", "hyponym"],
    "DistinctFrom": ["Antonym", "RelatedTo"],
    "HasSubevent": "HasPrerequisite",
    "Causes": ["hypernym", "hyponym"],
    "MadeOf": ["UsedFor", "AtLocation"],
    "ReceivesAction": "RelatedTo",
    "Desires": "CapableOf",
    "NotHasProperty": ["hypernym", "hyponym"],
    "CausesDesire": ["hypernym", "hyponym"],
    "HasFirstSubevent": ["HasLastSubevent", "HasPrerequisite"],
    "HasLastSubevent": ["HasFirstSubevent", "HasPrerequisite"],
    "NotDesires": ["hypernym", "hyponym"]
}


def generate_sentences(edges, parsed_grammars, switches, n_samples=8,
                       quality_lookup=None, s3=0, adj_when_available=0.8,
                       adj_variant_prob=0.3, max_adj_variants=3):
    s1, s2 = switches
    quality_lookup = quality_lookup or {}
    results = []
    skipped = []

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

        # NP realiser for this edge. resamples per call !! 
        # so different sentences for the same edge can independently get 
        # plain or decorated forms.
        # safety belt for HasProperty/NotHasProperty.
        def np(noun_token, node_id, _rel=rel, _rules=rules):
            if _rel in NO_ADJ_RELATIONS:
                return noun_token
            return realize_np(noun_token, node_id, _rules, quality_lookup,
                              s3=s3, adj_when_available=adj_when_available)

        # Sentence emitter. Always adds the sampled sentence.
        # ALSO adds minimal-pair variants substituting
        # alternative adjectives for nodes that have multiple HasQuality
        # entries, up to max_adj_variants total per noun. 

        # Variants are additive !!!(they go beyond the n_samples limit)
        def emit(sent, node_ids, _rel=rel):
            sentences.add(sent)
            if _rel in NO_ADJ_RELATIONS:
                return
            if random.random() >= adj_variant_prob:
                return
            for v in expand_adjective_variants(sent, node_ids, quality_lookup,
                                               s3=s3,
                                               max_variants_per_noun=max_adj_variants):
                sentences.add(v)

        sentences = set()
        attempts = 0
        max_attempts = 50

        # 1. Simple sentences
        n_simple = min(5, n_samples)
        while len(sentences) < n_simple and attempts < max_attempts:
            vp_struct = sample_vp_struct(rules)
            src_np = np(src, src)
            tgt_np = np(tgt, tgt)
            sent = apply_switches(src_np, vp_struct, tgt_np, s1, s2) + "."
            emit(sent, [src, tgt])
            attempts += 1

        # 2. Complex sentences
        sibling_edges = [se for se in edges_by_source[src] if se != edge]

        if sibling_edges:
            allowed = ALLOWED_EXPANSIONS.get(rel)
            target_relations = [allowed] if isinstance(allowed, str) else (allowed if allowed else [])
            expansion_edges = [
                se for se in sibling_edges
                if se['relation'] in target_relations and se['target'] != tgt
                and se['relation'] in parsed_grammars
                and f"VP_EXP_{se['relation']}" in rules
            ]
            same_rel_edges = [se for se in sibling_edges
                              if se['relation'] == rel and se['target'] != tgt]
            same_tgt_edges = [
                se for se in sibling_edges
                if se['target'] == tgt and se['relation'] != rel
                and se['relation'] in parsed_grammars
            ]

            used_edges = []
            while len(sentences) < n_samples and attempts < max_attempts:
                attempts += 1

                available_choices = []
                if same_rel_edges:
                    available_choices.append(('same_rel', same_rel_edges))
                if expansion_edges:
                    available_choices.append(('expand', expansion_edges))
                if same_tgt_edges:
                    available_choices.append(('same_tgt', same_tgt_edges))

                if not available_choices:
                    break

                choice_type, edges_list = random.choice(available_choices)
                second_edge = random.choice(edges_list)
                rel_2 = second_edge['relation']
                tgt_2 = second_edge['target']

                # Same relation - coordinate the two targets with "and".
                if choice_type == 'same_rel':
                    vp_struct_1 = sample_vp_struct(rules)
                    src_np = np(src, src)
                    tgt_np = np(tgt, tgt)
                    tgt_2_np = np(tgt_2, tgt_2)
                    coordinated_tgt = f"{tgt_np} and {tgt_2_np}"
                    complex_sent = apply_switches(src_np, vp_struct_1, coordinated_tgt, s1, s2) + "."
                    complex_sent = complex_sent.replace(" ,", ",")
                    emit(complex_sent, [src, tgt, tgt_2])

                # Same target, different relation - coordinate the predicates.
                elif choice_type == 'same_tgt':
                    rules_2 = parsed_grammars[rel_2]
                    vp_struct_1 = sample_vp_struct(rules)
                    vp_struct_2 = sample_vp_struct(rules_2)
                    pred1 = (vp_struct_1['v'] if vp_struct_1['type'] == 'TRANS'
                             else f"{vp_struct_1['aux']} {vp_struct_1.get('adjp', '')} {vp_struct_1['pp']}")
                    pred2 = (vp_struct_2['v'] if vp_struct_2['type'] == 'TRANS'
                             else f"{vp_struct_2['aux']} {vp_struct_2.get('adjp', '')} {vp_struct_2['pp']}")
                    pred1 = " ".join(pred1.split())
                    pred2 = " ".join(pred2.split())
                    coordinated_pred = f"{pred1} and {pred2}"
                    fake_vp_struct = {'type': 'TRANS', 'v': coordinated_pred}
                    src_np = np(src, src)
                    tgt_np = np(tgt, tgt)
                    complex_sent = apply_switches(src_np, fake_vp_struct, tgt_np, s1, s2) + "."
                    complex_sent = complex_sent.replace(" ,", ",")
                    emit(" ".join(complex_sent.split()), [src, tgt])

                # Expansion via VP_EXP_*
                elif choice_type == 'expand':
                    vp_struct_1 = sample_vp_struct(rules)
                    vp_struct_2 = sample_vp_struct(parsed_grammars[rel_2])
                    # Sample NP forms ONCE and reuse, so the substring strip
                    # below in the s1=1 case matches what's actually in part_1.
                    src_np = np(src, src)
                    tgt_np = np(tgt, tgt)
                    tgt_2_np = np(tgt_2, tgt_2)
                    part_1 = apply_switches(src_np, vp_struct_1, tgt_np, s1, s2)
                    exp_key = f"VP_EXP_{rel_2}"
                    conj_word = ""
                    if exp_key in rules:
                        chosen_rhs = _weighted_choice(rules[exp_key])
                        found_conj_sym = None
                        for kind, sym in chosen_rhs:
                            if kind == 'nonterminal' and sym.startswith('CONJ'):
                                found_conj_sym = sym
                                break
                        if found_conj_sym:
                            conj_word = _sample_leaf(rules, found_conj_sym)
                    part_2_cleaned = apply_switches("", vp_struct_2, tgt_2_np, s1, s2).strip()
                    if s1 == 1:
                        # part_1 ends with src_np; strip exactly that string.
                        part_1_cleaned = part_1.replace(src_np, "").strip()
                        complex_sent = f"{part_1_cleaned} {conj_word} {part_2_cleaned} {src_np}"
                    else:
                        complex_sent = f"{part_1} {conj_word} {part_2_cleaned}"
                    complex_sent = complex_sent.strip() + "."
                    complex_sent = complex_sent.replace(" ,", ",")
                    emit(" ".join(complex_sent.split()), [src, tgt, tgt_2])

                used_edges.append(second_edge)

        results.append({
            'source': src,
            'relation': rel,
            'target': tgt,
            'sentences': list(sentences),
            'expansions': used_edges if sibling_edges else []
        })

    return results, set(skipped)


# ___ CLI entry point _____________________________________________________________

def main():
    import argparse
    p = argparse.ArgumentParser(description='Generate PCFG sentences with switches')
    p.add_argument('--s1', type=int, default=0, choices=[0, 1],
                   help='S switch: 0=NP VP  1=VP NP')
    p.add_argument('--s2', type=int, default=1, choices=[0, 1],
                   help='VP switch: 0=OV  1=VO  (TRANS only)')
    p.add_argument('--s3', type=int, default=0, choices=[0, 1],
                   help='N-A switch: 0=A N  1=N A (only effective when an '
                        'adjective is actually inserted)')
    p.add_argument('--adj_when_available', type=float, default=0.8,
                   help='Given the NP rule allows an A slot AND the noun has '
                        'HasQuality entries, probability of inserting one.')
    p.add_argument('--adj_variant_prob', type=float, default=0.3,
                   help='Probability, per emitted sentence, of also generating '
                        'minimal-pair variants using alternative adjectives. '
                        'Only fires when a noun has multiple HasQuality entries '
                        'AND its adjective is realized in the sentence.')
    p.add_argument('--max_adj_variants', type=int, default=3,
                   help='Max number of adjective variants per noun (including '
                        'the one already in the sentence). Set to 1 to disable.')
    p.add_argument('--n_samples', type=int, default=8)
    p.add_argument('--edges',   default='edges.json')
    p.add_argument('--grammar', default='grammar_templates_adj.py')
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

    with open(args.edges) as f:
        all_edges = json.load(f)

    # HasQuality is only used as adjective lookup source
    quality_lookup = build_quality_lookup(all_edges)
    edges = [
        e for e in all_edges
        if e.get('relation') != 'HasQuality'
    ]
    n_quality = sum(len(v) for v in quality_lookup.values())
    print(f'Loaded {len(edges)} edges  |  HasQuality: {n_quality} adjectives '
          f'across {len(quality_lookup)} nouns')

    results, skipped = generate_sentences(
        edges, parsed_grammars, switches,
        n_samples=args.n_samples,
        quality_lookup=quality_lookup,
        s3=args.s3,
        adj_when_available=args.adj_when_available,
        adj_variant_prob=args.adj_variant_prob,
        max_adj_variants=args.max_adj_variants,
    )

    label = SWITCH_NAMES.get(switches, f's1={args.s1}_s2={args.s2}')
    output = {
        'switches': {'s1': args.s1, 's2': args.s2, 's3': args.s3},
        'results': results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total = sum(len(r['sentences']) for r in results)
    print(f'Config: {label} (s1={args.s1}, s2={args.s2}, s3={args.s3})')
    print(f'Triples: {len(results)}  |  Sentences: {total}')
    if skipped:
        print(f'Skipped: {sorted(skipped)}')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
