"""
Build artificial language corpora from generated sentences.

Takes generated_sentences.json and applies two dictionaries to produce:
  corpus_cjk.txt       — Language A (CJK tokens)
  corpus_hiragana.txt  — Language B (Hiragana tokens)

Each file has one sentence per line.
Also saves a parallel JSON for evaluation (triple info preserved).
"""

import json
import argparse
from pathlib import Path


def build_replacer(dictionary):
    """
    Build a lookup dict with space-normalised keys (underscores → spaces),
    sorted by number of words descending so longer phrases match first.
    Returns (normalised_dict, sorted_keys).
    """
    norm = {}
    for key, val in dictionary.items():
        norm[key.replace('_', ' ')] = val['artificial']
    sorted_keys = sorted(norm.keys(), key=lambda x: len(x.split()), reverse=True)
    return norm, sorted_keys


def replace_sentence(sentence, norm_dict, sorted_keys):
    """
    Token-by-token greedy phrase matching.
    Splits by space first so we never replace substrings inside words.
    """
    words = sentence.split()
    result = []
    i = 0
    while i < len(words):
        matched = False
        for key in sorted_keys:
            key_words = key.split()
            n = len(key_words)
            if words[i:i + n] == key_words:
                result.append(norm_dict[key])
                i += n
                matched = True
                break
        if not matched:
            result.append(words[i])
            i += 1
    return ' '.join(result)


def check_coverage(sentences, dictionary, sorted_keys, lang_name):
    """Report any words left unreplaced (not in dictionary)."""
    import re
    norm_dict = {k.replace('_', ' '): v['artificial'] for k, v in dictionary.items()}
    missing = set()
    for sent in sentences:
        replaced = replace_sentence(sent, norm_dict, sorted_keys)
        leftover = re.findall(r'[a-zA-Z_]+', replaced)
        missing.update(leftover)
    if missing:
        print(f'[{lang_name}] WARNING: {len(missing)} words not in dictionary:')
        for w in sorted(missing)[:20]:
            print(f'  {w}')
    else:
        print(f'[{lang_name}] Coverage: 100% ✓')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sentences', default='generated_sentences.json')
    p.add_argument('--cjk',      default='/Users/pengyuwen/Downloads/lemma_pos_artificial_cjk.json')
    p.add_argument('--hiragana', default='/Users/pengyuwen/Downloads/lemma_pos_artificial_hiragana.json')
    p.add_argument('--out_dir',  default='.')
    args = p.parse_args()

    with open(args.sentences) as f:
        data = json.load(f)
    with open(args.cjk) as f:
        cjk = json.load(f)
    with open(args.hiragana) as f:
        hira = json.load(f)

    cjk_norm,  cjk_keys  = build_replacer(cjk)
    hira_norm, hira_keys = build_replacer(hira)

    # Collect all raw sentences for coverage check
    all_sents = [s for r in data['results'] for s in r['sentences']]
    check_coverage(all_sents, cjk,  cjk_keys,  'CJK')
    check_coverage(all_sents, hira, hira_keys, 'Hiragana')

    # Build corpora
    out_dir = Path(args.out_dir)
    cjk_lines  = []
    hira_lines = []
    parallel   = []   # for evaluation

    for result in data['results']:
        cjk_sents  = [replace_sentence(s, cjk_norm,  cjk_keys)  for s in result['sentences']]
        hira_sents = [replace_sentence(s, hira_norm, hira_keys) for s in result['sentences']]

        cjk_lines.extend(cjk_sents)
        hira_lines.extend(hira_sents)

        parallel.append({
            'source':   result['source'],
            'relation': result['relation'],
            'target':   result['target'],
            'lang_a':   cjk_sents,
            'lang_b':   hira_sents,
        })

    # Save training corpora (one sentence per line)
    (out_dir / 'corpus_cjk.txt').write_text('\n'.join(cjk_lines), encoding='utf-8')
    (out_dir / 'corpus_hiragana.txt').write_text('\n'.join(hira_lines), encoding='utf-8')

    # Save parallel corpus for evaluation
    with open(out_dir / 'parallel_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(parallel, f, indent=2, ensure_ascii=False)

    print(f'Sentences per language: {len(cjk_lines)}')
    print(f'Saved: corpus_cjk.txt, corpus_hiragana.txt, parallel_corpus.json')


if __name__ == '__main__':
    main()
