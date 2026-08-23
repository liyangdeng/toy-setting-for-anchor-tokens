"""
LM-head evaluation for the SPECIAL TOKEN experiment -- mask-ENTITY.

Same idea as lm_head_eval.py: mask the target entity in a B (Hiragana)
sentence and read the model's OWN masked-LM head, rather than training an
external probe. The only difference from the generic version is that this
one applies the arm's B-side wrapping before prompting.

Why that matters: final_train_st.py saves the tokenizer with the A-side
template active. For the DISJOINT arm A = CJK is wrapped [BEG] $A [END],
so the saved post-processor would wrap the Hiragana prompts with [BEG]/[END]
too -- out of distribution, since Hiragana was trained with [CLS]/[SEP].
This script reads special_config.json and forces B's template ([CLS]/[SEP]
for disjoint, matching training) before encoding.

For the 'shared' and 'none' arms both languages share a template, so this is
a no-op -- but running this script for all three arms keeps the command
uniform and guarantees correctness for disjoint.

[MASK] is shared across languages in this experiment, so mask handling stays
language-agnostic (single tokenizer.mask_token_id).

Metrics (loose -> strict), same as the generic version:
  lenient        argmax over TARGET (B) tokens only  == correct B token
  strict_concept argmax over BOTH vocabularies       -> correct CONCEPT
  strict_token   argmax over BOTH vocabularies        == correct B token
strict_token top1 is the primary metric. Both top-1 and top-3 reported.

Diagnostics (descriptive, per sentence):
  source_token_rate         how often the model fills a SOURCE (A) token
  right_concept_wrong_lang  fills the A token of the CORRECT concept

Usage:
    python lm_head_eval_special.py \
        --model_dir st_checkpoints_disjoint_seed42/final \
        --final_omitted st_probing_run/final_omitted.json \
        --parallel st_probing_run/final_omitted_corpus/parallel_corpus_synset.json \
        --cjk_dict synset_pos_artificial_cjk.json \
        --hira_dict synset_pos_artificial_hiragana.json \
        --out_dir lmhead_st_disjoint
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_special_config(model_dir):
    """
    Read special_config.json written by final_train_st.py.
    Returns {'a': {cls, sep}, 'b': {cls, sep}}. Warns and returns None if
    missing (then the tokenizer's saved post-processor is used unchanged).
    """
    cfg_path = Path(model_dir) / "special_config.json"
    if not cfg_path.exists():
        print(f"NO special_config.json at {cfg_path} (!!!) -- using saved post-processor")
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {"a": cfg["a"], "b": cfg["b"]}


def apply_template(tokenizer, cls_tok, sep_tok):
    """Same as final_train_st.py's set_template. Wraps as 'cls $A sep', or
    bare '$A' when cls_tok is None (the 'none' arm)."""
    if cls_tok is None:
        tokenizer.backend_tokenizer.post_processor = TemplateProcessing(single="$A")
        return
    ids = tokenizer.convert_tokens_to_ids
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single=f"{cls_tok} $A {sep_tok}",
        special_tokens=[(cls_tok, ids(cls_tok)), (sep_tok, ids(sep_tok))],
    )


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
    Only the B (target) side is prompted here."""
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
    """Full-vocab logits at the [MASK] position, one row per sentence.
    Assumes the tokenizer's post-processor is already set to B's wrapping."""
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


def evaluate(model_dir, examples, cjk_dict, hira_dict, out_dir, policy, topk=3):
    device = pick_device()
    print(f"device: {device}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir).to(device).eval()

    # Force B/Hiragana wrapping before prompting. For disjoint this switches
    # the saved A-side [BEG]/[END] template to Hiragana's [CLS]/[SEP]; for
    # shared/none it's a no-op.
    if policy is not None:
        apply_template(tokenizer, policy["b"]["cls"], policy["b"]["sep"])
        print(f"B/Hiragana wrapping applied: cls={policy['b']['cls']}  sep={policy['b']['sep']}")

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

        if arg_b == cb:                       hit1["lenient"][ti] = True
        if cb in top_b_set:                   hit3["lenient"][ti] = True
        if arg_full in concept_ids:           hit1["strict_concept"][ti] = True
        if concept_ids & top_full_set:        hit3["strict_concept"][ti] = True
        if arg_full == cb:                    hit1["strict_token"][ti] = True
        if cb in top_full_set:                hit3["strict_token"][ti] = True

        if arg_full in a_id_set:              src_hits += 1
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
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    final_omitted = [p for p in json.load(open(args.final_omitted)) if p.get("track") == "entity"]
    parallel = json.load(open(args.parallel))
    cjk_dict = {k: v["artificial"] for k, v in json.load(open(args.cjk_dict)).items()}
    hira_dict = {k: v["artificial"] for k, v in json.load(open(args.hira_dict)).items()}

    tok = PreTrainedTokenizerFast.from_pretrained(args.model_dir)
    policy = load_special_config(args.model_dir)
    examples = build_examples(final_omitted, parallel, cjk_dict, hira_dict, tok.mask_token)
    print(f"built {len(examples)} usable mask-entity triples "
          f"(of {len(final_omitted)} final_omitted entity-track triples)")
    evaluate(args.model_dir, examples, cjk_dict, hira_dict, args.out_dir,
             policy=policy, topk=args.topk)


if __name__ == "__main__":
    main()
