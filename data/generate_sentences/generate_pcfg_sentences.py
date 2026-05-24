import json
import random
import re
from collections import defaultdict
from pathlib import Path
from grammar_templates import grammar


# ── Relation grammars ─────────────────────────────────────────────────────────

RELATION_GRAMMARS = grammar

# Relations that share grammar with another
RELATION_ALIASES = {
    "Entails": "entailment",
    "SimilarTo": "similar_to",
}


# ── Grammar parser ────────────────────────────────────────────────────────────

def parse_grammar(grammar_str):
    rules = defaultdict(list)
    for line in grammar_str.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.match(r'(\w+)\s*->\s*(.+?)\s*\[([0-9.]+)\]', line)
        if not match:
            continue
        lhs = match.group(1)
        rhs_str = match.group(2).strip()
        prob = float(match.group(3))
        rhs = []
        for token in re.findall(r"'[^']*'|\w+", rhs_str):
            if token.startswith("'"):
                rhs.append(('terminal', token[1:-1]))
            else:
                rhs.append(('nonterminal', token))
        rules[lhs].append((rhs, prob))
    return dict(rules)


def sample_vp(rules, symbol='VP'):
    productions = rules[symbol]
    probs = [p for _, p in productions]
    chosen_rhs, _ = random.choices(productions, weights=probs)[0]
    result = []
    for kind, tok in chosen_rhs:
        if kind == 'terminal':
            result.append(tok)
        else:
            result.extend(sample_vp(rules, tok))
    return result


# ── Switch mechanism ──────────────────────────────────────────────────────────

SWITCH_NAMES = {
    (0, 0): 'SOV',
    (0, 1): 'SVO',
    (1, 0): 'OVS',
    (1, 1): 'VOS',
}


def apply_switches(src, pred_tokens, tgt, s_switch, vp_switch):
    if vp_switch == 1:
        vp = pred_tokens + [tgt]
    else:
        vp = [tgt] + pred_tokens

    if s_switch == 0:
        tokens = [src] + vp
    else:
        tokens = vp + [src]

    return ' '.join(t.replace('_', ' ') for t in tokens)


# ── Lemma extraction ──────────────────────────────────────────────────────────

def get_lemma(edge, role):
    lemma_key = f'{role}_lemma'
    if lemma_key in edge:
        return edge[lemma_key]
    return edge[role].split('.')[0]


# ── Main generation ───────────────────────────────────────────────────────────

def generate_for_config(edges, parsed_grammars, s_switch, vp_switch,
                        n_samples=3, max_attempts=50):
    skipped = []
    results = []

    for edge in edges:
        relation = RELATION_ALIASES.get(edge['relation'], edge['relation'])
        if relation not in parsed_grammars:
            skipped.append(edge['relation'])
            continue

        src = get_lemma(edge, 'source')
        tgt = get_lemma(edge, 'target')
        rules = parsed_grammars[relation]

        sentences = set()
        attempts = 0
        while len(sentences) < n_samples and attempts < max_attempts:
            pred = sample_vp(rules)
            sentence = apply_switches(src, pred, tgt, s_switch, vp_switch)
            sentences.add(sentence)
            attempts += 1

        results.append({
            'source': edge['source'],
            'relation': edge['relation'],
            'target': edge['target'],
            'sentences': list(sentences),
        })

    return results, skipped


def main():

    with open('combined_synset_edges.json') as f:
        edges = json.load(f)

    parsed_grammars = {
        rel: parse_grammar(g) for rel, g in RELATION_GRAMMARS.items()
    }

    all_results = {}
    for s_switch in (0, 1):
        for vp_switch in (0, 1):
            config = SWITCH_NAMES[(s_switch, vp_switch)]
            results, skipped = generate_for_config(
                edges, parsed_grammars, s_switch, vp_switch
            )
            all_results[config] = results
            total = sum(len(r['sentences']) for r in results)
            print(f'{config}: {len(results)} triples, {total} sentences generated')
            if skipped:
                unique_skipped = set(skipped)
                print(f'  skipped relations: {unique_skipped}')
    print(all_results)



if __name__ == '__main__':
    main()
