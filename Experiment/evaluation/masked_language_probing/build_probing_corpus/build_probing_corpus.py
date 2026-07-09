"""
Orchestrator: deprived_triples.json + omitted_triples.json (from
select_probe_triples.py) -> a fresh mono-B reason filter -> final training
corpus for the treatment model's Language A.

mask-ENTITY track only -- mask-RELATION was tried and dropped (mono-B could
already guess the relation phrase for most relations without any
cross-lingual info; see git history for the removed implementation).

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
  4. TOP-1 reason filter: omitted_triples.json is deduped across the
     mask-ENTITY and mask-RELATION tracks that select_probe_triples.py
     computes, so probe_manifest.json is what tells them apart -- only the
     entity-track subset is used here. For every candidate, mask the target
     token in its B sentence(s) and ask mono-B. Any top-1 hit -> drop
     (mono-B alone can infer it -> not a clean cross-lingual probe).
  5. per-relation floor: after the top-1 drop, if a relation's surviving
     count < --min_survivors, drop the WHOLE relation (not enough left for a
     reliable per-relation estimate).
  6. write final_omitted.json (surviving triples, tagged "track": "entity")
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


def load_manifest_tracks(manifest_path):
    """probe_manifest.json's probes carry a 'track' field that the deduped
    omitted_triples.json/deprived_triples.json edge-lists don't (they're
    deduped across BOTH tracks). Returns {track: {(source,relation,target)}}
    so callers can pull out just the entity-track subset."""
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

    # 4+5. top-1 filter + per-relation floor, entity track only (omitted_triples.json
    # is deduped across both tracks, so probe_manifest.json is what tells them apart)
    omitted_edges, hira_dict = load_probes(args.omitted_triples, args.hira_dict)
    by_track = load_manifest_tracks(args.probe_manifest)
    entity_edges = [e for e in omitted_edges
                     if (e["source"], e["relation"], e["target"]) in by_track["entity"]]

    final_omitted = filter_top1(mono_b_dir, entity_edges, hira_dict,
                                 omitted_corpus["parallel"], args.min_survivors)
    for e in final_omitted:
        e["track"] = "entity"

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
