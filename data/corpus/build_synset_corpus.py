"""
Build artificial language corpora from synset-based generated sentences.

Reads generated_sentences_synsets.json and applies the synset artificial
mapping dictionaries to produce:
  corpus_cjk_synset.txt       — Language A (CJK tokens)
  corpus_hiragana_synset.txt  — Language B (Hiragana tokens)
  parallel_corpus_synset.json — parallel corpus for evaluation

All tokens (synset IDs and grammar terminals) keep their underscores intact,
matching dictionary keys exactly. Sentences with any OOV token are skipped.

Usage:
    python build_synset_corpus.py \
        --sentences <path>/generated_sentences_synsets.json \
        --cjk      <path>/synset_pos_artificial_cjk.json \
        --hiragana <path>/synset_pos_artificial_hiragana.json \
        --out_dir  ~/Desktop/coding
"""

import argparse
import json
from pathlib import Path


def load_dict(path):
    """Load synset artificial mapping: key -> artificial token string."""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    return {k: v['artificial'] for k, v in raw.items()}

PUNCT_CHARS = {'.', ','}

PUNCT_CHARS = {'.', ',', ';', '!', '?', '(', ')', '[', ']', '{', '}', '"'}


def split_edge_punctuation(raw_tok):
    """
    Split punctuation marks from lexical tokens.

    Keeps internal punctuation intact!
      abdomen.n.01.          -> abdomen.n.01 .
      conceptnet:art:1286.   -> conceptnet:art:1286 .
      abdominal_wall.n.01,   -> abdominal_wall.n.01 ,
      ball-and-socket_joint.n.01. -> ball-and-socket_joint.n.01 .

    """
    leading = []
    trailing = []

    while raw_tok and raw_tok[0] in PUNCT_CHARS:
        leading.append(raw_tok[0])
        raw_tok = raw_tok[1:]

    while raw_tok and raw_tok[-1] in PUNCT_CHARS:
        trailing.append(raw_tok[-1])
        raw_tok = raw_tok[:-1]

    pieces = []
    pieces.extend(leading)

    if raw_tok:
        pieces.append(raw_tok)

    pieces.extend(reversed(trailing))
    return pieces


def tokenize_with_punctuation(sentence):
    """
    Tokenise by whitespace, but separate punctuation from lexical tokens.
    """
    tokens = []

    for raw_tok in sentence.split():
        tokens.extend(split_edge_punctuation(raw_tok))

    return tokens

def replace_sentence(sentence, mapping):
    """
    Replace every whitespace-separated token with its artificial equivalent.
    Returns the replaced sentence string, or None if any token is OOV.

    Punctuation becomes a separate token:
      abdominal_wall.n.01. -> 俘煎 .
      token, token         -> 噤尥 , 俘煎
    """
    tokens = tokenize_with_punctuation(sentence)
    result = []

    for tok in tokens:
        if tok in PUNCT_CHARS:
            result.append(tok)
            continue

        art = mapping.get(tok)

        if art is None:
            return None
        result.append(art)
    return ' '.join(result)


def build_corpora(sentences_path, cjk_mapping, hira_mapping):
    with open(sentences_path, encoding='utf-8') as f:
        data = json.load(f)

    cjk_lines  = []
    hira_lines = []
    parallel   = []

    total = kept = skipped = 0

    for entry in data['results']:
        src = entry['source']
        rel = entry['relation']
        tgt = entry['target']

        lang_a_sents = []
        lang_b_sents = []

        for sent in entry['sentences']:
            total += 1
            a = replace_sentence(sent, cjk_mapping)
            b = replace_sentence(sent, hira_mapping)
            if a is None or b is None:
                skipped += 1
                continue
            kept += 1
            lang_a_sents.append(a)
            lang_b_sents.append(b)

        if lang_a_sents:
            cjk_lines.extend(lang_a_sents)
            hira_lines.extend(lang_b_sents)
            parallel.append({
                'source': src,
                'relation': rel,
                'target': tgt,
                'lang_a': lang_a_sents,
                'lang_b': lang_b_sents,
            })

    return cjk_lines, hira_lines, parallel, total, kept, skipped


def main():
    BASE = Path('data')

    p = argparse.ArgumentParser()
    p.add_argument('--sentences', default=str(BASE / 'generate_sentences/generated_sentences_adj.py'))
    p.add_argument('--cjk',      default=str(BASE / 'semantic_backbones/dict_to_artificial/dicts_synset_adj_augmented/synset_pos_artificial_cjk_edges_adj_augmented.json'))
    p.add_argument('--hiragana', default=str(BASE / 'semantic_backbones/dict_to_artificial/dicts_synset_adj_augmented/synset_pos_artificial_hiragana_edges_adj_augmented.json'))
    p.add_argument('--out_dir',  default=str(Path.home() / 'Desktop/coding'))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading dictionaries...')
    cjk_mapping  = load_dict(args.cjk)
    hira_mapping = load_dict(args.hiragana)
    print(f'  CJK dict      : {len(cjk_mapping)} entries')
    print(f'  Hiragana dict : {len(hira_mapping)} entries')

    print('Building corpora...')
    cjk_lines, hira_lines, parallel, total, kept, skipped = build_corpora(
        args.sentences, cjk_mapping, hira_mapping
    )

    print(f'  Total sentences  : {total}')
    print(f'  Kept             : {kept}  ({kept/total:.1%})')
    print(f'  Skipped (OOV)    : {skipped}')
    print(f'  Parallel triples : {len(parallel)}')

    # Write corpus files
    cjk_path  = out_dir / 'corpus_cjk_synset.txt'
    hira_path = out_dir / 'corpus_hiragana_synset.txt'
    para_path = out_dir / 'parallel_corpus_synset.json'

    cjk_path.write_text('\n'.join(cjk_lines), encoding='utf-8')
    hira_path.write_text('\n'.join(hira_lines), encoding='utf-8')
    with open(para_path, 'w', encoding='utf-8') as f:
        json.dump(parallel, f, indent=2, ensure_ascii=False)

    print(f'\nOutput written to {out_dir}:')
    print(f'  {cjk_path.name}           — {len(cjk_lines)} sentences')
    print(f'  {hira_path.name}  — {len(hira_lines)} sentences')
    print(f'  {para_path.name}')

    # Show a few sample pairs
    print('\nSample (first 2 triples):')
    for entry in parallel[:2]:
        print(f'  [{entry["source"]} --{entry["relation"]}--> {entry["target"]}]')
        print(f'    CJK  : {entry["lang_a"][0]}')
        print(f'    Hira : {entry["lang_b"][0]}')


if __name__ == '__main__':
    main()
