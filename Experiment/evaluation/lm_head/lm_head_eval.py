"""
LM-head evaluation for the anchor-token transfer experiment -- mask-ENTITY.

A fourth evaluation metric next to word translation, sentence retrieval and
linear probing. Instead of training an external classifier on the hidden
states (the probe), it reads the model's OWN masked-LM head: mask the target
entity in a sentence on the omitted side and ask the model to fill it.

Same setup as the probe. Facts are omitted from language B (Hiragana): the
model trained on the A rendering of each probe triple but never on the B
rendering, so filling the entity correctly on the B side can only come from the
A parallel -- i.e. cross-lingual transfer. A = CJK (source), B = Hiragana
(target, where we prompt).

Because the two vocabularies live in one joint output space, a single "correct"
criterion is ambiguous, so we report three, from loose to strict:

  lenient        argmax over TARGET (B) tokens only  == the correct B token
  strict_concept argmax over BOTH vocabularies       -> the correct CONCEPT
                 (the A token OR the B token of that concept counts)
  strict_token   argmax over BOTH vocabularies        == the correct B token
                 (must beat every A token too -- the "true transfer" bar)

strict_token is the pre-registered primary metric for significance; the other
two are reported as robustness (does the conclusion hold under a looser or a
concept-level criterion). Both top-1 and top-3 are reported.

Two diagnostics (descriptive, per sentence, not tested):
  source_token_rate         how often the model fills a SOURCE (A) token at all
  right_concept_wrong_lang  how often it fills the A token of the CORRECT concept
                            (right meaning, wrong language -- "cat is suss")

A triple counts as a hit if ANY of its B renderings hits (OR-aggregation, same
as the probe). Accuracy = hit triples / total triples.

Takes the same inputs as linear_probe.py, so it runs on any experiment's probe
outputs (final_omitted.json + the fresh parallel file + the two synset dicts),
not just one setting.

Usage:
    python lm_head_eval.py \
        --model_dir probing_run/checkpoints/final \
        --final_omitted probing_run/final_omitted.json \
        --parallel probing_run/final_omitted_corpus/parallel_corpus_synset.json \
        --cjk_dict synset_pos_artificial_cjk_edges_adj_augmented.json \
        --hira_dict synset_pos_artificial_hiragana_edges_adj_augmented.json \
        --out_dir probing_run/lm_head_results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_all_masked(sentences, target_tok, mask_str):
    """Every rendering where target_tok is a standalone word, with target_tok
    replaced by mask_str (order preserved, duplicates collapsed)."""
    out, seen = [], set()
    for sent in sentences:
        words = sent.split()
        if target_tok not in words:
            continue
        words[words.index(target_tok)] = mask_str
        masked = " ".join(words)
        if masked not in seen:
            seen.add(masked)
            out.append(masked)
    return out


def build_examples(final_omitted, parallel, cjk_dict, hira_dict, mask_str):
    """One entry per triple, carrying its masked B renderings and the concept.
    Only the B (target) side is needed here, but we mirror the probe's builder."""
    par = {(e["source"], e["relation"], e["target"]): e for e in parallel}
    examples = []
    for p in final_omitted:
        entry = par.get((p["source"], p["relation"], p["target"]))
        if entry is None:
            continue
        b_tok = hira_dict.get(p["target"])
        if b_tok is None or cjk_dict.get(p["target"]) is None:
            continue
        b_sents = find_all_masked(entry["lang_b"], b_tok, mask_str)
        if not b_sents:
            continue
        examples.append({"b_sents": b_sents, "concept": p["target"],
                          "relation": p["relation"]})
    return examples


@torch.no_grad()
def mask_logits(model, tokenizer, sentences, device, max_length=64, batch_size=32):
    """Full-vocab logits at the [MASK] position, one row per sentence."""
    mask_id = tokenizer.mask_token_id
    out = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        logits = model(**enc).logits
        mask_pos = (enc["input_ids"] == mask_id)
        for row in range(len(batch)):
            pos = mask_pos[row].nonzero(as_tuple=True)[0]
            out.append(logits[row, pos[0]].cpu().numpy())
    return out


def evaluate(model_dir, examples, cjk_dict, hira_dict, out_dir, topk=3):
    device = pick_device()
    print(f"device: {device}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir).to(device).eval()
    vocab = tokenizer.get_vocab()

    b_ids = np.array(sorted({vocab[t] for t in hira_dict.values() if t in vocab}))
    a_id_set = {vocab[t] for t in cjk_dict.values() if t in vocab}

    eval_sents, triple_idx = [], []
    for ti, e in enumerate(examples):
        for s in e["b_sents"]:
            eval_sents.append(s)
            triple_idx.append(ti)

    n = len(examples)
    print(f"triples: {n} | eval rows (B renderings): {len(eval_sents)}")
    logits_rows = mask_logits(model, tokenizer, eval_sents, device)

    metrics = ["lenient", "strict_concept", "strict_token"]
    hit1 = {m: np.zeros(n, dtype=bool) for m in metrics}
    hit3 = {m: np.zeros(n, dtype=bool) for m in metrics}
    src_hits = wrong_lang_hits = seen = 0

    for logits, ti in zip(logits_rows, triple_idx):
        concept = examples[ti]["concept"]
        cb = vocab.get(hira_dict.get(concept))
        ca = vocab.get(cjk_dict.get(concept))
        if cb is None:
            continue
        seen += 1

        top_full = np.argsort(-logits)[:topk]
        arg_full = int(top_full[0])
        top_full_set = set(int(x) for x in top_full)

        b_local = np.argsort(-logits[b_ids])[:topk]
        top_b = b_ids[b_local]
        arg_b = int(top_b[0])
        top_b_set = set(int(x) for x in top_b)

        concept_ids = {cb} | ({ca} if ca is not None else set())

        if arg_b == cb:                      hit1["lenient"][ti] = True
        if cb in top_b_set:                  hit3["lenient"][ti] = True
        if arg_full in concept_ids:          hit1["strict_concept"][ti] = True
        if concept_ids & top_full_set:       hit3["strict_concept"][ti] = True
        if arg_full == cb:                   hit1["strict_token"][ti] = True
        if cb in top_full_set:               hit3["strict_token"][ti] = True

        if arg_full in a_id_set:             src_hits += 1
        if ca is not None and arg_full == ca: wrong_lang_hits += 1

    print()
    for m in metrics:
        print(f"{m:<15} top1 = {hit1[m].mean():.4f}  top3 = {hit3[m].mean():.4f}  "
              f"({int(hit1[m].sum())}/{n} triples)")
    src_rate = src_hits / seen if seen else 0.0
    wl_rate = wrong_lang_hits / seen if seen else 0.0
    print(f"\ndiagnostics (per sentence, n={seen}):")
    print(f"  source_token_rate        = {src_rate:.4f}")
    print(f"  right_concept_wrong_lang = {wl_rate:.4f}")

    write_outputs(hit1, hit3, n, src_rate, wl_rate, examples, out_dir)


def write_outputs(hit1, hit3, n, src_rate, wl_rate, examples, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "lm_head_accuracy.csv", "w") as f:
        f.write("metric,top1,top3,n\n")
        for m in ("lenient", "strict_concept", "strict_token"):
            f.write(f"{m},{hit1[m].mean():.4f},{hit3[m].mean():.4f},{n}\n")

    with open(out_dir / "lm_head_diagnostics.csv", "w") as f:
        f.write("diagnostic,rate\n")
        f.write(f"source_token_rate,{src_rate:.4f}\n")
        f.write(f"right_concept_wrong_lang,{wl_rate:.4f}\n")

    # per-triple hits for the primary metric (strict_token, top1) -> significance
    primary = hit1["strict_token"]
    with open(out_dir / "strict_token_hits.csv", "w") as f:
        f.write("id,concept,hit\n")
        for i, e in enumerate(examples):
            f.write(f"{i},{e['concept']},{int(primary[i])}\n")

    print(f"\nsaved -> {out_dir/'lm_head_accuracy.csv'}")
    print(f"saved -> {out_dir/'lm_head_diagnostics.csv'}")
    print(f"saved -> {out_dir/'strict_token_hits.csv'}  "
          f"(primary = strict_token top1: {int(primary.sum())}/{n})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="omitted-model checkpoint (…/final)")
    p.add_argument("--final_omitted", required=True)
    p.add_argument("--parallel", required=True,
                   help="the fresh parallel file built from final_omitted.json")
    p.add_argument("--cjk_dict", required=True)
    p.add_argument("--hira_dict", required=True)
    p.add_argument("--out_dir", default="./lm_head_results")
    args = p.parse_args()

    final_omitted = [p for p in json.load(open(args.final_omitted)) if p.get("track") == "entity"]
    parallel = json.load(open(args.parallel))
    cjk_dict = {k: v["artificial"] for k, v in json.load(open(args.cjk_dict)).items()}
    hira_dict = {k: v["artificial"] for k, v in json.load(open(args.hira_dict)).items()}

    tok = PreTrainedTokenizerFast.from_pretrained(args.model_dir)
    examples = build_examples(final_omitted, parallel, cjk_dict, hira_dict, tok.mask_token)
    print(f"built {len(examples)} usable mask-entity triples "
          f"(of {len(final_omitted)} final_omitted entity-track triples)")
    evaluate(args.model_dir, examples, cjk_dict, hira_dict, args.out_dir)


if __name__ == "__main__":
    main()
