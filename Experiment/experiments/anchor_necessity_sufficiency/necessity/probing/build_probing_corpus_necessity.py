"""
Held-out (deprived/omitted) corpus + mono-B leak filter for the Anchor
Necessity setting: mask-ENTITY track only.

Unlike build_probing_corpus.py (which assumes Language A and B share ONE
PCFG derivation, just translated into two token vocabularies),
here A is fixed at switch config 111 and B varies across {000,001,011,111}
-- A and B are INDEPENDENT derivations with different word order, so
sentences are generated separately per language and paired by
(source,relation,target), the same way build_different_parallel_corpus.py
pairs the full (non-held-out) corpus.

Each of the 4 B-configs gets its own mono-B leak filter (a triple that's
"clean" -- i.e. not guessable by a monolingual model alone -- in one word
order isn't necessarily clean in another; per the design discussion, we do
NOT intersect surviving triples across configs, since the metric itself
(transfer accuracy net of what monolingual reasoning explains) is already
well-defined per condition).

Steps 1-2 run once (shared); steps 3-9 run once per B-config.

Usage:
    python build_probing_corpus_necessity.py --stage select
    python build_probing_corpus_necessity.py --stage fastgen
    python build_probing_corpus_necessity.py --stage mono --cfg 000
    python build_probing_corpus_necessity.py --stage filter --cfg 000
    python build_probing_corpus_necessity.py --stage assemble --cfg 000
"""

import argparse
import importlib.util
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast

REPO = Path('/Users/pengyuwen/toy-setting-for-anchor-tokens')
EDGES = REPO / 'data/semantic_backbones/edges_adj.json'
GRAMMAR = REPO / 'data/grammar/grammar_templates_adj.py'
DICTS = REPO / 'data/semantic_backbones/dict_to_artificial/dicts_synset_adj_augmented'
CJK_DICT_PATH = DICTS / 'synset_pos_artificial_cjk_edges_adj_augmented.json'
HIRA_DICT_PATH = DICTS / 'synset_pos_artificial_hiragana_edges_adj_augmented.json'
MONO_SCRIPT = REPO / 'Experiment/training/train_monolingual_synset.py'
GEN_SCRIPT_PATH = REPO / 'data/generate_sentences/generate_sentences_no_anchors.py'
BUILD_CORPUS_PATH = REPO / 'data/corpus/build_synset_corpus.py'
SELECT_SCRIPT = REPO / 'Experiment/evaluation/masked_language_probing/build_probing_corpus/select_probe_triples.py'
BPC_SCRIPT = REPO / 'Experiment/evaluation/masked_language_probing/build_probing_corpus/build_probing_corpus.py'

OUT = REPO / 'Experiment/experiments/anchor_necessity_sufficiency/necessity/probing'
LANG_A_CONFIG = (1, 1, 1)
LANG_B_CONFIGS = ['000', '001', '011', '111']
SEED = 42
MONO_EPOCHS = 60
MIN_SURVIVORS = 40
N_PER_RELATION = 50
MIN_TOTAL = 100


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load_module(GEN_SCRIPT_PATH, 'gen_no_anchors_necessity')
bld = _load_module(BUILD_CORPUS_PATH, 'build_synset_corpus_necessity')
bpc = _load_module(BPC_SCRIPT, 'build_probing_corpus_necessity_reuse')


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_grammar(path):
    mod = _load_module(path, 'grammar_mod_necessity')
    return {rel: gen.parse_grammar(g) for rel, g in mod.grammar.items()}


def generate_for_config(edges, grammars, quality_lookup, cfg):
    random.seed(SEED)
    s1, s2, s3 = cfg
    results, _ = gen.generate_sentences(edges, grammars, (s1, s2), n_samples=8,
                                         quality_lookup=quality_lookup, s3=s3)
    return results


def translate_and_pair(results_a, results_b, cjk_map, hira_map):
    a_lines, b_lines, parallel = [], [], []
    b_by_key = {(r['source'], r['relation'], r['target']): r for r in results_b}
    for ra in results_a:
        key = (ra['source'], ra['relation'], ra['target'])
        rb = b_by_key.get(key)
        if rb is None:
            continue
        a_sents = [s for s in (bld.replace_sentence(s, cjk_map) for s in ra['sentences']) if s]
        b_sents = [s for s in (bld.replace_sentence(s, hira_map) for s in rb['sentences']) if s]
        if not a_sents or not b_sents:
            continue
        a_lines.extend(a_sents)
        b_lines.extend(b_sents)
        parallel.append({'source': ra['source'], 'relation': ra['relation'],
                          'target': ra['target'], 'lang_a': a_sents, 'lang_b': b_sents})
    return a_lines, b_lines, parallel


# ── stage: select (shared, run once) ────────────────────────────────────────

def stage_select():
    OUT.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(SELECT_SCRIPT),
         '--input', str(REPO / 'data/generate_sentences/v3_generated_sentences_adj.json'),
         '--n_per_relation', N_PER_RELATION, '--min_total', MIN_TOTAL, '--seed', SEED,
         '--out', OUT / 'probe_manifest_full.json',
         '--out_omitted', OUT / 'omitted_triples_full.json',
         '--out_deprived', OUT / 'deprived_triples_full.json'])

    manifest = json.load(open(OUT / 'probe_manifest_full.json'))
    entity_keys = {(p['source'], p['relation'], p['target'])
                   for p in manifest['probes'] if p['track'] == 'entity'}
    print(f'entity-track probes: {len(entity_keys)}')

    all_edges = json.load(open(EDGES))
    edges = [e for e in all_edges if e.get('relation') != 'HasQuality']
    omitted = [e for e in edges if (e['source'], e['relation'], e['target']) in entity_keys]
    deprived = [e for e in edges if (e['source'], e['relation'], e['target']) not in entity_keys]

    json.dump(omitted, open(OUT / 'omitted_triples.json', 'w'), indent=2, ensure_ascii=False)
    json.dump(deprived, open(OUT / 'deprived_triples.json', 'w'), indent=2, ensure_ascii=False)
    print(f'omitted (entity-only): {len(omitted)} | deprived: {len(deprived)}')


def run(cmd):
    cmd = [str(c) for c in cmd]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ── stage: fastgen (shared A-side + per-B-config, no training) ─────────────

def stage_fastgen():
    """Generates+translates+pairs deprived and omitted corpora for all 4
    B-configs. A-side sentences are generated once per triple-set (deprived,
    omitted) and reused across the 4 configs."""
    grammars = load_grammar(GRAMMAR)
    all_edges = json.load(open(EDGES))
    quality_lookup = gen.build_quality_lookup(all_edges)
    cjk_map = bld.load_dict(str(CJK_DICT_PATH))
    hira_map = bld.load_dict(str(HIRA_DICT_PATH))

    deprived_edges = json.load(open(OUT / 'deprived_triples.json'))
    omitted_edges = json.load(open(OUT / 'omitted_triples.json'))

    print('Generating Language A (config 111) -- deprived + omitted (shared across configs)...')
    a_deprived = generate_for_config(deprived_edges, grammars, quality_lookup, LANG_A_CONFIG)
    a_omitted = generate_for_config(omitted_edges, grammars, quality_lookup, LANG_A_CONFIG)

    for cfg_str in LANG_B_CONFIGS:
        cfg = tuple(int(c) for c in cfg_str)
        cfg_dir = OUT / f'cfg_{cfg_str}'
        cfg_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n-- config {cfg_str} --')

        b_deprived = generate_for_config(deprived_edges, grammars, quality_lookup, cfg)
        b_omitted = generate_for_config(omitted_edges, grammars, quality_lookup, cfg)

        a_lines, b_lines, parallel = translate_and_pair(a_deprived, b_deprived, cjk_map, hira_map)
        (cfg_dir / 'deprived_a.txt').write_text('\n'.join(a_lines), encoding='utf-8')
        (cfg_dir / 'deprived_b.txt').write_text('\n'.join(b_lines), encoding='utf-8')
        json.dump(parallel, open(cfg_dir / 'deprived_parallel.json', 'w'), ensure_ascii=False)
        print(f'  deprived: A={len(a_lines)} sents, B={len(b_lines)} sents, {len(parallel)} triples')

        a_lines_o, b_lines_o, parallel_o = translate_and_pair(a_omitted, b_omitted, cjk_map, hira_map)
        (cfg_dir / 'omitted_a.txt').write_text('\n'.join(a_lines_o), encoding='utf-8')
        (cfg_dir / 'omitted_b.txt').write_text('\n'.join(b_lines_o), encoding='utf-8')
        json.dump(parallel_o, open(cfg_dir / 'omitted_parallel.json', 'w'), ensure_ascii=False)
        print(f'  omitted:  A={len(a_lines_o)} sents, B={len(b_lines_o)} sents, {len(parallel_o)} triples')


# ── stage: mono (per config, trains) ────────────────────────────────────────

def stage_mono(cfg_str):
    cfg_dir = OUT / f'cfg_{cfg_str}'
    run([sys.executable, str(MONO_SCRIPT),
         '--corpus', cfg_dir / 'deprived_b.txt',
         '--output_dir', cfg_dir / 'mono_b',
         '--epochs', MONO_EPOCHS, '--seed', SEED])


# ── stage: filter (per config, uses trained mono-B, no training) ───────────

def stage_filter(cfg_str):
    cfg_dir = OUT / f'cfg_{cfg_str}'
    hira_map_raw = json.load(open(HIRA_DICT_PATH))
    hira_dict = {k: v['artificial'] for k, v in hira_map_raw.items()}

    device = pick_device()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg_dir / 'mono_b' / 'final')
    model = BertForMaskedLM.from_pretrained(cfg_dir / 'mono_b' / 'final').to(device).eval()

    omitted_edges = json.load(open(OUT / 'omitted_triples.json'))
    parallel = json.load(open(cfg_dir / 'omitted_parallel.json'))
    par = {(e['source'], e['relation'], e['target']): e['lang_b'] for e in parallel}

    survivors, dropped, skipped = [], [], []
    per_rel = defaultdict(lambda: {'n': 0, 'hit': 0})
    for e in omitted_edges:
        key = (e['source'], e['relation'], e['target'])
        gold_tok = hira_dict.get(e['target'])
        sents = par.get(key, [])
        if gold_tok is None or not sents:
            skipped.append(e)
            continue
        hit_any = any(bpc.top1_hit(model, tokenizer, device, s, gold_tok) for s in sents)
        per_rel[e['relation']]['n'] += 1
        per_rel[e['relation']]['hit'] += int(hit_any)
        (dropped if hit_any else survivors).append(e)

    print(f"\n{'relation':18s} {'n':>5s} {'hit':>5s} {'surv':>5s}")
    for rel, d in sorted(per_rel.items(), key=lambda kv: -kv[1]['n']):
        print(f"{rel:18s} {d['n']:5d} {d['hit']:5d} {d['n']-d['hit']:5d}")
    print(f'skipped: {len(skipped)} | dropped (mono-B guessed it): {len(dropped)} | survivors: {len(survivors)}')

    surv_by_rel = Counter(e['relation'] for e in survivors)
    kept_rel = {r for r, n in surv_by_rel.items() if n >= MIN_SURVIVORS}
    final = [e for e in survivors if e['relation'] in kept_rel]
    print(f'per-relation floor ({MIN_SURVIVORS}): kept {len(kept_rel)} relations, final_omitted = {len(final)} triples')

    for e in final:
        e['track'] = 'entity'  # linear_probe.py filters final_omitted.json by this field
    json.dump(final, open(cfg_dir / 'final_omitted.json', 'w'), indent=2, ensure_ascii=False)


# ── stage: assemble (per config, fresh gen for survivors, no training) ─────

def stage_assemble(cfg_str):
    cfg_dir = OUT / f'cfg_{cfg_str}'
    cfg = tuple(int(c) for c in cfg_str)
    grammars = load_grammar(GRAMMAR)
    all_edges = json.load(open(EDGES))
    quality_lookup = gen.build_quality_lookup(all_edges)
    cjk_map = bld.load_dict(str(CJK_DICT_PATH))
    hira_map = bld.load_dict(str(HIRA_DICT_PATH))

    final_edges = json.load(open(cfg_dir / 'final_omitted.json'))
    a_final = generate_for_config(final_edges, grammars, quality_lookup, LANG_A_CONFIG)
    b_final = generate_for_config(final_edges, grammars, quality_lookup, cfg)
    a_lines, b_lines, parallel = translate_and_pair(a_final, b_final, cjk_map, hira_map)
    json.dump(parallel, open(cfg_dir / 'final_omitted_parallel.json', 'w'), ensure_ascii=False)

    deprived_a = (cfg_dir / 'deprived_a.txt').read_text(encoding='utf-8').splitlines()
    a_training = deprived_a + a_lines
    (cfg_dir / 'a_training.txt').write_text('\n'.join(a_training), encoding='utf-8')
    (cfg_dir / 'b_training.txt').write_text((cfg_dir / 'deprived_b.txt').read_text(encoding='utf-8'), encoding='utf-8')
    print(f'a_training: {len(a_training)} sentences | b_training: deprived-B only')
    print(f'final_omitted_parallel: {len(parallel)} triples -> {cfg_dir / "final_omitted_parallel.json"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['select', 'fastgen', 'mono', 'filter', 'assemble'])
    p.add_argument('--cfg', choices=LANG_B_CONFIGS)
    args = p.parse_args()

    if args.stage == 'select':
        stage_select()
    elif args.stage == 'fastgen':
        stage_fastgen()
    elif args.stage == 'mono':
        assert args.cfg, '--stage mono needs --cfg'
        stage_mono(args.cfg)
    elif args.stage == 'filter':
        assert args.cfg, '--stage filter needs --cfg'
        stage_filter(args.cfg)
    elif args.stage == 'assemble':
        assert args.cfg, '--stage assemble needs --cfg'
        stage_assemble(args.cfg)


if __name__ == '__main__':
    main()
