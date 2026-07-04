"""
build_different_parallel_corpus.py
Experiment-specific corpus + parallel builder for the Anchor Necessity / Sufficiency setting.

This is an assembled ("集合") version of two existing repo tools:
  - data/generate_sentences/generate_sentences_no_anchors.py  (punctuation-free PCFG generator)
  - data/corpus/build_synset_corpus.py                        (synset -> artificial-token replacer)
Both are imported and reused here — no logic is duplicated.

For a FIXED language A (config 111, CJK) paired against language B at each of the four
structural configs {000, 001, 011, 111} (Hiragana), it writes to ./corpus/:

  langA_cjk_111.txt        language A  (one file, reused across every condition)
  langB_hira_<cfg>.txt     language B  (one per config)
  parallel_<cfg>.json      lang_a = 111/CJK, lang_b = <cfg>/Hiragana, aligned by edge/triple

Language A and B share ZERO tokens (disjoint CJK/Hiragana ranges + the no-anchors
generator strips all punctuation), so every condition is a zero-anchor setting;
only the structural (word-order) similarity between A and B varies, controlled by
the switch config of B:
    B=000 -> 0%   B=001 -> 33%   B=011 -> 67%   B=111 -> 100%  (Hamming to A=111)

Because A and B come from DIFFERENT structural configs, the parallel is built by
pairing the two generations per edge. Same seed keeps the edge order identical, so
results_A[i] and results_B[i] are the same triple. Retrieval evaluation matches at
the triple level, so within-edge sentence order need not coincide.

Run:  python build_different_parallel_corpus.py
"""

import importlib.util
import json
import random
from pathlib import Path

# ── Paths (repo-absolute; edit REPO if the checkout moves) ───────────────────────
REPO = Path('/Users/pengyuwen/toy-setting-for-anchor-tokens')
EDGES   = REPO / 'data/semantic_backbones/edges_adj.json'
GRAMMAR = REPO / 'data/grammar/grammar_templates_adj.py'
DICTS   = REPO / 'data/semantic_backbones/dict_to_artificial/dicts_synset_adj_augmented'
CJK_DICT  = DICTS / 'synset_pos_artificial_cjk_edges_adj_augmented.json'
HIRA_DICT = DICTS / 'synset_pos_artificial_hiragana_edges_adj_augmented.json'

OUT_DIR = Path(__file__).resolve().parent   # script lives inside the corpus/ folder; outputs sit beside it

SEED = 42
LANG_A_CONFIG = (1, 1, 1)                    # fixed reference language A (CJK)
LANG_B_CONFIGS = ['000', '001', '011', '111']  # language B (Hiragana), varying structure


# ── Import the two existing tools ────────────────────────────────────────────────

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

gen = _load_module(REPO / 'data/generate_sentences/generate_sentences_no_anchors.py',
                   'gen_no_anchors')
bld = _load_module(REPO / 'data/corpus/build_synset_corpus.py', 'build_synset_corpus')


def load_grammar(path):
    mod = _load_module(path, 'grammar_mod')
    return {rel: gen.parse_grammar(g) for rel, g in mod.grammar.items()}


def generate_for_config(edges, grammars, quality_lookup, cfg):
    """cfg = (s1, s2, s3). Returns per-edge results with punctuation-free 'sentences'."""
    random.seed(SEED)                        # reset per config so content is identical, only word order differs
    s1, s2, s3 = cfg
    results, _ = gen.generate_sentences(
        edges, grammars, (s1, s2),
        n_samples=8, quality_lookup=quality_lookup, s3=s3,
    )
    return results


def replace_edge_sentences(edge_result, mapping):
    """Replace each sentence with artificial tokens; drop any OOV sentence."""
    out = []
    for s in edge_result['sentences']:
        r = bld.replace_sentence(s, mapping)
        if r is not None:
            out.append(r)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grammars = load_grammar(GRAMMAR)
    all_edges = json.loads(EDGES.read_text(encoding='utf-8'))
    quality_lookup = gen.build_quality_lookup(all_edges)
    edges = [e for e in all_edges if e.get('relation') != 'HasQuality']

    cjk_map  = bld.load_dict(str(CJK_DICT))
    hira_map = bld.load_dict(str(HIRA_DICT))
    print(f'edges: {len(edges)}  |  cjk dict: {len(cjk_map)}  |  hira dict: {len(hira_map)}')

    # Language A: fixed config 111, CJK. Generated once, reused for every condition.
    results_A = generate_for_config(edges, grammars, quality_lookup, LANG_A_CONFIG)
    A_cjk = [replace_edge_sentences(r, cjk_map) for r in results_A]
    langA_lines = [s for edge in A_cjk for s in edge]
    (OUT_DIR / 'langA_cjk_111.txt').write_text('\n'.join(langA_lines), encoding='utf-8')
    print(f'langA_cjk_111.txt : {len(langA_lines)} sentences (config 111)')

    # Language B: each structural config, Hiragana. Corpus + parallel-with-A.
    for cfg_str in LANG_B_CONFIGS:
        cfg = tuple(int(c) for c in cfg_str)
        results_B = generate_for_config(edges, grammars, quality_lookup, cfg)
        B_hira = [replace_edge_sentences(r, hira_map) for r in results_B]

        langB_lines = [s for edge in B_hira for s in edge]
        (OUT_DIR / f'langB_hira_{cfg_str}.txt').write_text('\n'.join(langB_lines), encoding='utf-8')

        # Parallel: pair per edge. results_A / results_B share edge order (same edges + seed).
        parallel = []
        for ra, a_sents, b_sents in zip(results_A, A_cjk, B_hira):
            if not a_sents or not b_sents:
                continue
            parallel.append({
                'source': ra['source'], 'relation': ra['relation'], 'target': ra['target'],
                'lang_a': a_sents, 'lang_b': b_sents,
            })
        (OUT_DIR / f'parallel_{cfg_str}.json').write_text(
            json.dumps(parallel, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'langB_hira_{cfg_str}.txt : {len(langB_lines)} sentences  |  '
              f'parallel_{cfg_str}.json : {len(parallel)} triples')

    print(f'\nDone. Outputs in {OUT_DIR}')


if __name__ == '__main__':
    main()
