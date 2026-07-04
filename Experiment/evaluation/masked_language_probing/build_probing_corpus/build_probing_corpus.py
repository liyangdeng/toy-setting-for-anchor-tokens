"""
Orchestrator: deprived_triples.json + omitted_triples.json (from
select_probe_triples.py) -> a fresh mono-B reason filter -> final training
corpus for the treatment model's Language A.

This is EXPERIMENT-SPECIFIC and must be re-run from scratch every time you
train a model: the sentence generator is a PCFG, so every run produces
different concrete sentences, and the mono-B model that does the reason
filtering has to be trained fresh on THIS run's deprived corpus. Nothing here
is safely reusable across experiments except select_probe_triples.py's output.

Pipeline:
  1. deprived_triples.json --[gen_script]--> deprived_sentences.json
     omitted_triples.json  --[gen_script]--> omitted_sentences.json
  2. deprived_sentences.json --[build_corpus_script]--> deprived corpus
     (both languages) + deprived parallel file
     omitted_sentences.json  --[build_corpus_script]--> omitted corpus
     (both languages, per-triple) -- only lang_b is needed for filtering
  3. train mono-B on the deprived Hiragana corpus (subprocess:
     train_monolingual_synset.py)
  4. TOP-1 reason filter, split by track using probe_manifest.json (the only
     place that still tags each triple entity-vs-relation -- omitted_triples.json
     is deduped across both tracks):
       - mask-ENTITY: mask the target token in its B sentence(s), ask mono-B.
       - mask-RELATION: locate the relation-phrase span via
         strip_entities_and_match (built from grammar_templates_adj.py, see
         build_relation_templates), mask all of it at once, ask mono-B, check
         whether the joint decode lands in the relation's candidate set.
     Both tracks use the same hit_any direction: any sentence for a triple
     where mono-B gets it right -> drop (mono-B alone can infer it -> not a
     clean cross-lingual probe).
  5. per-relation floor: after the top-1 drop, if a relation's surviving
     count < --min_survivors, drop the WHOLE relation (not enough left for a
     reliable per-relation estimate). Applied separately per track.
  6. write final_omitted.json (surviving triples, each tagged with its track)
  7. regenerate sentences for final_omitted.json FRESH (new PCFG draw, not
     reusing step-1's omitted_sentences.json) --[gen_script]-->
     final_omitted_sentences.json
  8. build corpus for final_omitted_sentences.json (both languages) --
     -[build_corpus_script]--> final_omitted corpus + parallel file (the
     parallel file is what linear_probe.py's --parallel should point to)
  9. concatenate deprived-A corpus (step 2) + final_omitted-A corpus (step 8)
     -> a_training file (Language A training corpus for the treatment model)
     Also copies deprived-B corpus as b_training (Language B is NOT extended
     with final_omitted -- that's the whole point of the deprivation).

--gen_script and --build_corpus_script are REQUIRED, no default: pick the
exact script version matching your current grammar/switches, since the PCFG
output (and therefore what "deprived"/"omitted" concretely render to) depends
on it. --build_corpus_script MUST be a version that tokenizes punctuation
correctly (i.e. build_synset_corpus.py, not the older build_corpus.py) --
v3-generated sentences glue "." and "," onto the adjacent token, and only the
punctuation-aware tokenizer splits them back out before the dictionary
lookup. Passing the wrong one will silently corrupt or drop sentences.

Usage:
    python build_probing_corpus.py \
        --deprived_triples deprived_triples.json \
        --omitted_triples  omitted_triples.json \
        --gen_script          .../v3_generate_sentences.py \
        --build_corpus_script .../build_synset_corpus.py \
        --mono_train_script   .../train_monolingual_synset.py \
        --grammar   .../grammar_templates_adj.py \
        --probe_manifest .../probe_manifest.json \
        --cjk_dict  .../synset_pos_artificial_cjk_edges_adj_augmented.json \
        --hira_dict .../synset_pos_artificial_hiragana_edges_adj_augmented.json \
        --out_dir ./probing_run_1
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast


def run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── mask-RELATION: candidate templates, straight from the grammar ──────────

def build_relation_templates(gen_script, grammar_path):
    """
    Enumerate every relation's possible relation-phrase realizations,
    bucketed by token length, straight from grammar_templates_adj.py (not
    sampled -- exact and complete). Reuses parse_grammar from gen_script so
    this always matches whichever grammar-format version is in play.

      TRANS branch -> length 1: (v,)
      COP   branch -> length 3: (aux, adjp, pp)  if the COP rule has an ADJP
                      slot (every relation in the current grammar does),
                      else length 2: (aux, pp)

    Returns {relation: {length: {english_terminal_tuple, ...}}}.
    """
    gen_mod = _load_module(gen_script, "gen_mod_for_templates")
    grammar_mod = _load_module(grammar_path, "grammar_mod_for_templates")
    parse_grammar = gen_mod.parse_grammar

    templates = {}
    for rel, raw in grammar_mod.grammar.items():
        rules = parse_grammar(raw)
        by_len = defaultdict(set)
        for rhs, _ in rules.get("VP", []):
            nonterms = [sym for kind, sym in rhs if kind == "nonterminal"]
            if "TRANS" in nonterms:
                for v_rhs, _ in rules.get("V", []):
                    by_len[1].add((v_rhs[0][1],))
            if "COP" in nonterms:
                aux_choices = [r[0][1] for r, _ in rules.get("AUX", [])]
                pp_choices = [r[0][1] for r, _ in rules.get("PP", [])]
                cop_has_adjp = any(
                    "ADJP" in [sym for kind, sym in cop_rhs if kind == "nonterminal"]
                    for cop_rhs, _ in rules.get("COP", [])
                )
                if cop_has_adjp:
                    adjp_choices = [r[0][1] for r, _ in rules.get("ADJP", [])]
                    for a in aux_choices:
                        for adj in adjp_choices:
                            for pp in pp_choices:
                                by_len[3].add((a, adj, pp))
                else:
                    for a in aux_choices:
                        for pp in pp_choices:
                            by_len[2].add((a, pp))
        templates[rel] = dict(by_len)
    return templates


def translate_templates(templates, lang_dict):
    """Translate English-terminal candidate tuples into one language's
    artificial tokens (lang_dict: english_terminal -> artificial_token)."""
    out = {}
    for rel, by_len in templates.items():
        out[rel] = {}
        for length, tuples in by_len.items():
            translated = {tuple(lang_dict[t] for t in tup)
                          for tup in tuples if all(t in lang_dict for t in tup)}
            if translated:
                out[rel][length] = translated
    return out


def reverse_translate_templates(templates, lang_dict):
    """Inverse of translate_templates: {relation: {artificial_tuple: english_tuple}}.

    The two languages render the SAME PCFG draw (same sentence, same VP
    choice) into completely different, disjoint token strings -- so after
    strip_entities_and_match matches a rendering in one language's tokens,
    the artificial tuple it returns is NOT a label that means the same
    thing in the other language. This map recovers the language-agnostic
    template identity (the original English terminal tuple) so a caller
    building a classifier label can use ONE shared identity space across
    languages, the same way mask-ENTITY uses concept id (which is already
    language-agnostic, coming straight from the KG) rather than the
    translated target token."""
    out = {}
    for rel, by_len in templates.items():
        out[rel] = {}
        for tuples in by_len.values():
            for tup in tuples:
                if all(t in lang_dict for t in tup):
                    out[rel][tuple(lang_dict[t] for t in tup)] = tup
    return out


def all_grammar_terminals(templates):
    """Flat set of every English terminal used by ANY relation's templates
    -- lets strip_entities_and_match tell apart 'this token is grammar
    vocabulary' from 'this token is an adjective clinging to an entity'."""
    terms = set()
    for by_len in templates.values():
        for tuples in by_len.values():
            for tup in tuples:
                terms.update(tup)
    return terms


def strip_entities_and_match(sentence, src_tok, tgt_tok, templates_for_rel, grammar_vocab):
    """
    Locate src_tok/tgt_tok in `sentence` (each possibly with ONE adjacent
    adjective token -- recognized as "a neighboring token that isn't
    grammar vocabulary"), strip those spans plus the trailing '.', and
    check whether the remaining contiguous run of tokens exactly matches
    one of templates_for_rel's candidates of the same length.

    Word-order switches (s1/s2) can put the relation phrase before,
    between, or after the two entities -- this only cares about "the
    tokens that are neither entity" so it works regardless of order.

    Complex sentences (coordinated targets, VP_EXP expansions, etc.) get
    rejected here: either the leftover isn't one contiguous run, or it is
    but doesn't exactly match any enumerated candidate (which is finite
    and only covers plain simple-sentence realizations).

    Returns (start_index, length, matched_tuple) or None (not usable for
    mask-RELATION -- try another sentence for this triple, or drop it).
    """
    words = sentence.split()
    if words and words[-1] == ".":
        words = words[:-1]

    if src_tok not in words or tgt_tok not in words:
        return None
    i_src = words.index(src_tok)
    i_tgt = words.index(tgt_tok)

    entity_positions = {i_src, i_tgt}
    for i in (i_src, i_tgt):
        for j in (i - 1, i + 1):
            if 0 <= j < len(words) and j not in entity_positions and words[j] not in grammar_vocab:
                entity_positions.add(j)

    leftover_positions = sorted(set(range(len(words))) - entity_positions)
    if not leftover_positions:
        return None
    if leftover_positions != list(range(leftover_positions[0], leftover_positions[-1] + 1)):
        return None  # not contiguous -> coordinated/expanded sentence, skip

    leftover = tuple(words[p] for p in leftover_positions)
    candidates = templates_for_rel.get(len(leftover), set())
    if leftover in candidates:
        return (leftover_positions[0], len(leftover), leftover)
    return None


@torch.no_grad()
def relation_span_hit(model, tokenizer, device, sentence, start, length, candidates_of_len):
    """Mask `length` contiguous positions starting at `start` (positions
    computed by strip_entities_and_match against the period-stripped word
    list) all at once, decode each masked position's top-1 independently
    in a single forward pass, and check whether the joint sequence is one
    of this relation's candidates of that length."""
    words = sentence.split()
    period = bool(words) and words[-1] == "."
    if period:
        words = words[:-1]

    masked = words[:start] + [tokenizer.mask_token] * length + words[start + length:]
    if period:
        masked = masked + ["."]
    enc = tokenizer(" ".join(masked), return_tensors="pt").to(device)
    logits = model(**enc).logits[0]
    mask_positions = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    pred_ids = [logits[pos].argmax().item() for pos in mask_positions]
    pred_toks = tuple(tokenizer.convert_ids_to_tokens(pred_ids))
    return pred_toks in candidates_of_len


def load_manifest_tracks(manifest_path):
    """probe_manifest.json's probes carry a 'track' field that the deduped
    omitted_triples.json/deprived_triples.json edge-lists don't (they're
    deduped across BOTH tracks). Returns {track: {(source,relation,target)}}
    so the two track-specific filters only see their own probes."""
    manifest = json.load(open(manifest_path))
    by_track = defaultdict(set)
    for p in manifest["probes"]:
        by_track[p["track"]].add((p["source"], p["relation"], p["target"]))
    return by_track


# ── stage 1+2: generate sentences, build corpus ─────────────────────────────

def generate_sentences(gen_script, edges_path, grammar, out_path, seed=42):
    run([sys.executable, gen_script,
         "--edges", edges_path, "--grammar", grammar,
         "--output", out_path, "--seed", seed])


def build_corpus(build_corpus_script, sentences_path, cjk_dict, hira_dict, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run([sys.executable, build_corpus_script,
         "--sentences", sentences_path,
         "--cjk", cjk_dict, "--hiragana", hira_dict,
         "--out_dir", out_dir])
    return {
        "corpus_a": str(Path(out_dir) / "corpus_cjk_synset.txt"),
        "corpus_b": str(Path(out_dir) / "corpus_hiragana_synset.txt"),
        "parallel": str(Path(out_dir) / "parallel_corpus_synset.json"),
    }


# ── stage 3: train mono-B ───────────────────────────────────────────────────

def train_mono_b(mono_train_script, corpus_b, output_dir, epochs=60, seed=42):
    run([sys.executable, mono_train_script,
         "--corpus", corpus_b, "--output_dir", output_dir,
         "--epochs", epochs, "--seed", seed])
    return str(Path(output_dir) / "final")


# ── stage 4: top-1 reason filter, mask-ENTITY track ─────────────────────────

def load_probes(omitted_triples_path, hira_dict_path):
    """omitted_triples.json only stores (source, relation, target), deduped
    across BOTH tracks. Callers split it by track using
    load_manifest_tracks(probe_manifest.json) before filtering."""
    edges = json.load(open(omitted_triples_path))
    hira = {k: v["artificial"] for k, v in json.load(open(hira_dict_path)).items()}
    return edges, hira


@torch.no_grad()
def top1_hit(model, tokenizer, device, sentence, gold_tok):
    words = sentence.split()
    if gold_tok not in words:
        return None
    words[words.index(gold_tok)] = tokenizer.mask_token
    enc = tokenizer(" ".join(words), return_tensors="pt").to(device)
    logits = model(**enc).logits[0]
    mpos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0, 0]
    top1 = logits[mpos].argmax().item()
    gid = tokenizer.convert_tokens_to_ids(gold_tok)
    return gid == top1


def filter_top1(mono_b_dir, omitted_edges, hira_dict, omitted_parallel_path,
                 min_survivors):
    device = pick_device()
    print(f"device: {device}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(mono_b_dir)
    model = BertForMaskedLM.from_pretrained(mono_b_dir).to(device).eval()

    par = {(e["source"], e["relation"], e["target"]): e["lang_b"]
           for e in json.load(open(omitted_parallel_path))}

    survivors, dropped_inferable, skipped = [], [], []
    per_rel = defaultdict(lambda: {"n": 0, "hit": 0})

    for e in omitted_edges:
        key = (e["source"], e["relation"], e["target"])
        gold_tok = hira_dict.get(e["target"])
        sents = par.get(key, [])
        if gold_tok is None or not sents:
            skipped.append(e)
            continue
        hit_any = False
        for s in sents:
            r = top1_hit(model, tokenizer, device, s, gold_tok)
            if r:
                hit_any = True
                break
        per_rel[e["relation"]]["n"] += 1
        per_rel[e["relation"]]["hit"] += int(hit_any)
        (dropped_inferable if hit_any else survivors).append(e)

    print(f"\n{'relation':18s} {'n':>5s} {'top1_hit':>9s} {'survivors':>10s}")
    print("-" * 44)
    for rel, d in sorted(per_rel.items(), key=lambda kv: -kv[1]["n"]):
        surv = d["n"] - d["hit"]
        print(f"{rel:18s} {d['n']:5d} {d['hit']:9d} {surv:10d}")
    print(f"skipped (no sentence/token): {len(skipped)}")
    print(f"top-1-inferable (dropped): {len(dropped_inferable)} | "
          f"survivors before relation floor: {len(survivors)}")

    # per-relation floor: drop the WHOLE relation if survivors < min_survivors
    surv_by_rel = Counter(e["relation"] for e in survivors)
    kept_relations = {r for r, n in surv_by_rel.items() if n >= min_survivors}
    dropped_relations = {r for r in surv_by_rel if r not in kept_relations}
    final = [e for e in survivors if e["relation"] in kept_relations]

    print(f"\nper-relation floor (min_survivors={min_survivors}):")
    for r, n in sorted(surv_by_rel.items(), key=lambda kv: -kv[1]):
        status = "KEPT" if r in kept_relations else "DROPPED (whole relation)"
        print(f"  {r:18s} {n:4d} survivors -> {status}")
    print(f"\nfinal_omitted: {len(final)} triples across {len(kept_relations)} relations")

    return final


def filter_relation_top1(mono_b_dir, omitted_edges, relation_templates_b, grammar_vocab_b,
                          hira_dict, omitted_parallel_path, min_survivors):
    """mask-RELATION counterpart to filter_top1 (same hit_any direction and
    per-relation floor). For every omitted candidate, try ALL its B
    sentences: locate + mask the relation-phrase span via
    strip_entities_and_match, and check whether mono-B's joint decode
    lands in the relation's candidate set. Any hit -> mono-B alone can
    already fill it in, so it isn't a clean cross-lingual probe, drop it."""
    device = pick_device()
    print(f"device: {device}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(mono_b_dir)
    model = BertForMaskedLM.from_pretrained(mono_b_dir).to(device).eval()

    par = {(e["source"], e["relation"], e["target"]): e["lang_b"]
           for e in json.load(open(omitted_parallel_path))}

    survivors, dropped_inferable, skipped = [], [], []
    per_rel = defaultdict(lambda: {"n": 0, "hit": 0})

    for e in omitted_edges:
        key = (e["source"], e["relation"], e["target"])
        src_tok = hira_dict.get(e["source"])
        tgt_tok = hira_dict.get(e["target"])
        templates_for_rel = relation_templates_b.get(e["relation"], {})
        sents = par.get(key, [])
        if src_tok is None or tgt_tok is None or not sents or not templates_for_rel:
            skipped.append(e)
            continue

        hit_any = False
        for s in sents:
            match = strip_entities_and_match(s, src_tok, tgt_tok, templates_for_rel, grammar_vocab_b)
            if match is None:
                continue  # complex/coordinated rendering of this triple, skip this sentence
            start, length, _ = match
            if relation_span_hit(model, tokenizer, device, s, start, length,
                                  templates_for_rel[length]):
                hit_any = True
                break
        per_rel[e["relation"]]["n"] += 1
        per_rel[e["relation"]]["hit"] += int(hit_any)
        (dropped_inferable if hit_any else survivors).append(e)

    print(f"\n{'relation':18s} {'n':>5s} {'top1_hit':>9s} {'survivors':>10s}")
    print("-" * 44)
    for rel, d in sorted(per_rel.items(), key=lambda kv: -kv[1]["n"]):
        surv = d["n"] - d["hit"]
        print(f"{rel:18s} {d['n']:5d} {d['hit']:9d} {surv:10d}")
    print(f"skipped (no sentence/no simple rendering/token): {len(skipped)}")
    print(f"top-1-inferable (dropped): {len(dropped_inferable)} | "
          f"survivors before relation floor: {len(survivors)}")

    surv_by_rel = Counter(e["relation"] for e in survivors)
    kept_relations = {r for r, n in surv_by_rel.items() if n >= min_survivors}
    final = [e for e in survivors if e["relation"] in kept_relations]

    print(f"\nper-relation floor (min_survivors={min_survivors}):")
    for r, n in sorted(surv_by_rel.items(), key=lambda kv: -kv[1]):
        status = "KEPT" if r in kept_relations else "DROPPED (whole relation)"
        print(f"  {r:18s} {n:4d} survivors -> {status}")
    print(f"\nfinal_omitted (relation): {len(final)} triples across {len(kept_relations)} relations")

    return final


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deprived_triples", required=True)
    p.add_argument("--omitted_triples", required=True)
    p.add_argument("--probe_manifest", required=True,
                   help="probe_manifest.json from select_probe_triples.py -- "
                        "the only place that still tags each triple's track "
                        "(entity/relation); omitted_triples.json is deduped "
                        "across both tracks and loses that tag")
    p.add_argument("--gen_script", required=True,
                   help="sentence generator matching your current grammar/switches version")
    p.add_argument("--build_corpus_script", required=True,
                   help="MUST be punctuation-aware (e.g. build_synset_corpus.py), "
                        "not the older build_corpus.py")
    p.add_argument("--mono_train_script", required=True)
    p.add_argument("--grammar", required=True)
    p.add_argument("--cjk_dict", required=True)
    p.add_argument("--hira_dict", required=True)
    p.add_argument("--min_survivors", type=int, default=45,
                   help="drop a whole relation if its top-1-filtered survivor "
                        "count falls below this")
    p.add_argument("--mono_epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="./probing_run")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. generate sentences for deprived + omitted
    deprived_sents = out / "deprived_sentences.json"
    omitted_sents = out / "omitted_sentences.json"
    generate_sentences(args.gen_script, args.deprived_triples, args.grammar,
                        deprived_sents, args.seed)
    generate_sentences(args.gen_script, args.omitted_triples, args.grammar,
                        omitted_sents, args.seed)

    # 2. build corpus for each
    deprived_corpus = build_corpus(args.build_corpus_script, deprived_sents,
                                    args.cjk_dict, args.hira_dict, out / "deprived_corpus")
    omitted_corpus = build_corpus(args.build_corpus_script, omitted_sents,
                                   args.cjk_dict, args.hira_dict, out / "omitted_corpus")

    # 3. train mono-B on deprived-B
    mono_b_dir = train_mono_b(args.mono_train_script, deprived_corpus["corpus_b"],
                               out / "mono_b", epochs=args.mono_epochs, seed=args.seed)

    # 4+5. top-1 filter + per-relation floor, split by track (omitted_triples.json
    # is deduped across both tracks, so probe_manifest.json is what tells them apart)
    omitted_edges, hira_dict = load_probes(args.omitted_triples, args.hira_dict)
    by_track = load_manifest_tracks(args.probe_manifest)
    entity_edges = [e for e in omitted_edges
                     if (e["source"], e["relation"], e["target"]) in by_track["entity"]]
    relation_edges = [e for e in omitted_edges
                       if (e["source"], e["relation"], e["target"]) in by_track["relation"]]

    final_entity = filter_top1(mono_b_dir, entity_edges, hira_dict,
                                omitted_corpus["parallel"], args.min_survivors)
    for e in final_entity:
        e["track"] = "entity"

    relation_templates = build_relation_templates(args.gen_script, args.grammar)
    relation_templates_b = translate_templates(relation_templates, hira_dict)
    grammar_vocab_b = {hira_dict[t] for t in all_grammar_terminals(relation_templates)
                        if t in hira_dict}
    final_relation = filter_relation_top1(mono_b_dir, relation_edges, relation_templates_b,
                                           grammar_vocab_b, hira_dict,
                                           omitted_corpus["parallel"], args.min_survivors)
    for e in final_relation:
        e["track"] = "relation"

    final_omitted = final_entity + final_relation

    final_path = out / "final_omitted.json"
    json.dump(final_omitted, open(final_path, "w"), indent=2, ensure_ascii=False)
    print(f"saved -> {final_path}")

    # 6+7. FRESH sentence generation for the surviving final_omitted set
    final_sents = out / "final_omitted_sentences.json"
    generate_sentences(args.gen_script, final_path, args.grammar, final_sents, args.seed)
    final_corpus = build_corpus(args.build_corpus_script, final_sents,
                                args.cjk_dict, args.hira_dict, out / "final_omitted_corpus")

    # 8. assemble A/B training corpora
    a_lines = (Path(deprived_corpus["corpus_a"]).read_text(encoding="utf-8").splitlines()
               + Path(final_corpus["corpus_a"]).read_text(encoding="utf-8").splitlines())
    a_training = out / "a_training.txt"
    a_training.write_text("\n".join(a_lines), encoding="utf-8")

    b_training = out / "b_training.txt"
    b_training.write_text(Path(deprived_corpus["corpus_b"]).read_text(encoding="utf-8"),
                           encoding="utf-8")

    print(f"\na_training (deprived-A + final_omitted-A): {len(a_lines)} sentences -> {a_training}")
    print(f"b_training (deprived-B only)              -> {b_training}")
    print(f"final_omitted parallel (for linear_probe.py --parallel) -> {final_corpus['parallel']}")


if __name__ == "__main__":
    main()
