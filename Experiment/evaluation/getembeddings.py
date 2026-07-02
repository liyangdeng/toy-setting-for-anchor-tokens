"""
analyse learned embeddings.

1.  computes avg cosine similarity across vocabulary
        mean, median, standard deviation, minimum
        1, 5, 95, 99 percentile

2.  shows 20 most and least similar pairs

3.  checks k tokens most similar to a specific one
        (!) define the token and k in: show_neighbors("んぢょ", k=10)
            if left as is, it takes a few random ones and "tiger"

4.  old feature:
    computes embeddings' cosine similarity for picked tokens.
        (!) define desired tokens in:
        cherries = ["んぢょ", "ぱい", "つどもぽ", "そあ", "ぷぎ", "さきへま"]
        if left as is, takes "dog, canine, bridge, liquid, tiger, lion"

usage:
    python getembeddings.py \
        --model_dir ./checkpoints_monolingual/final \
        --dev_file ./checkpoints_monolingual/dev.txt
"""

import torch
import torch.nn.functional as F
import random
from transformers import BertForMaskedLM, PreTrainedTokenizerFast
from collections import defaultdict
import argparse
from pathlib import Path

# _________________________________________________________________
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
    description="compute cosine similarity for lvl 4 embeddings")
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True)

    parser.add_argument(
        "--dev_file",
        type=Path,
        required=True)

    return parser.parse_args()


args = parse_args()

MODEL_DIR = args.model_dir
DEV_FILE = args.dev_file

if not MODEL_DIR.is_dir():
    raise FileNotFoundError(
        f"Model directory does not exist: {MODEL_DIR.resolve()}"
    )

if not DEV_FILE.is_file():
    raise FileNotFoundError(
        f"Development file does not exist: {DEV_FILE.resolve()}"
    )

print(f"Model directory: {MODEL_DIR}")
print(f"Development file: {DEV_FILE}")

# _________________________________________________________________

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForMaskedLM.from_pretrained(MODEL_DIR)
model.eval()

with open(DEV_FILE, encoding="utf-8") as f:
    sentences = [l.strip() for l in f if l.strip()]

# collect layer 4 embeddings
embeddings_per_token = defaultdict(list)

with torch.no_grad():
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[4]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        for pos, token in enumerate(tokens):
            if token in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
                continue
            embeddings_per_token[token].append(hidden[0, pos, :])

avg_embeddings = {
    token: F.normalize(torch.stack(vecs).mean(dim=0).unsqueeze(0), dim=-1).squeeze(0)
    for token, vecs in embeddings_per_token.items()
}

vocab_list = list(avg_embeddings.keys())
emb_matrix = torch.stack([avg_embeddings[t] for t in vocab_list])
token_to_idx = {t: i for i, t in enumerate(vocab_list)}

sim_matrix = emb_matrix @ emb_matrix.t() # full pairwise cosine similarity matrix


# 1. Avg similarity across vocabulary ________________________________
V = sim_matrix.size(0)
mask = ~torch.eye(V, dtype=torch.bool)
off_diag = sim_matrix[mask]

mean_sim   = off_diag.mean().item()
median_sim = off_diag.median().item()
std_sim    = off_diag.std().item()
p95_sim    = off_diag.quantile(0.95).item()
p99_sim    = off_diag.quantile(0.99).item()
min_sim    = off_diag.min().item()
p01_sim    = off_diag.quantile(0.01).item()
p05_sim    = off_diag.quantile(0.05).item()

print(f"\n__ Similarity distribution across {V} tokens __")
print(f"  pairs compared : {off_diag.numel():,}")
print(f"  mean           : {mean_sim:.4f}")
print(f"  median         : {median_sim:.4f}")
print(f"  standard dev   : {std_sim:.4f}")
print(f"  minimum        : {min_sim:.4f}")
print(f"  1% of pairs are below  : {p01_sim:.4f}")
print(f"     a score below the 1 percentile → notably dissimilar")
print(f"  5% of pairs are below  : {p05_sim:.4f}")
print(f"     a score below the 5 percentile → meaningfully dissimilar")
print(f"-------------------------------------------------------------------------")
print(f"a score between the 5 and 95 percentiles → unremarkable, just average")
print(f"-------------------------------------------------------------------------")
print(f"  95% of pairs are below : {p95_sim:.4f}")
print(f"     a score above the 95 percentile → meaningfully similar")
print(f"  99% of pairs are below : {p99_sim:.4f}")
print(f"     a score above the 99 percentile → notably similar")
print(f"-------------------------------------------------------------------------")

# 2. Top-n (20) most and least similar pairs ___________________________________
def top_pairs(n=20):
    V = sim_matrix.size(0)
    iu = torch.triu_indices(V, V, offset=1)
    sims = sim_matrix[iu[0], iu[1]]
    
    top_sims, top_flat = sims.topk(n)
    return [
        (vocab_list[iu[0][i].item()], vocab_list[iu[1][i].item()], s.item())
        for i, s in zip(top_flat, top_sims)
    ]

print(f"\n Top 20 most similar pairs in vocabulary")
for t1, t2, sim in top_pairs(20):
    print(f"  {sim:.4f}  {t1}  -  {t2}")

def bottom_pairs(n=20):
    V = sim_matrix.size(0)
    iu = torch.triu_indices(V, V, offset=1)
    sims = sim_matrix[iu[0], iu[1]]
    
    # topk with largest=False gives the smallest values
    bot_sims, bot_flat = sims.topk(n, largest=False)
    return [
        (vocab_list[iu[0][i].item()], vocab_list[iu[1][i].item()], s.item())
        for i, s in zip(bot_flat, bot_sims)
    ]

print(f"\n Top 20 least similar pairs in vocabulary")
for t1, t2, sim in bottom_pairs(20):
    print(f"  {sim:.4f}  {t1}  ↔  {t2}")
print(f"\n ---------------------------------------")

# 3. Top-k (10) most similar tokens for a query ___________________________________
def most_similar(query, k=10):
    if query not in token_to_idx:
        print(f"Token '{query}' not in vocabulary")
        return []
    idx = token_to_idx[query]
    sims = sim_matrix[idx].clone()
    sims[idx] = -float("inf")                # exclude self
    top_sims, top_indices = sims.topk(k)
    return [(vocab_list[i], s.item()) for i, s in zip(top_indices, top_sims)]


def show_neighbors(query, k=10):
    print(f"\nTop {k} neighbours of '{query}':")
    neighbors = most_similar(query, k)
    if not neighbors:
        return
    for token, sim in neighbors:
        # flag scores that exceed 95% as notable
        flag = " *" if sim > p95_sim else ""
        print(f"  {sim:.4f}  {token}{flag}")


# show neighbors of a few random tokens
random.seed(0)
sample = random.sample(vocab_list, min(5, len(vocab_list)))
for query in sample:
    show_neighbors(query, k=10)

# SPECIFIC QUERY
show_neighbors("んぢょ", k=10) # tiger
# show_neighbors("んぢょ", k=10) # dog

# __________________________________________________embedding cherry-picking
# 4. Compute similarity for picked tokens __________________________________

# Example
#             dog    canine   bridge    liquid   tiger  lion
cherries = ["んぢょ", "ぱい", "つどもぽ", "そあ", "ぷぎ", "さきへま"]

#            princess   queen    piglet    cub   liqour    alcohol
# cherries = ["ぁろぶ", "えこめ", "まつも", "きぎぉ", "ぷにと", "ちぜぁこ"]

def similarity(t1, t2):
    if t1 not in avg_embeddings or t2 not in avg_embeddings:
        print(f"Token not found")
        return None
    return (avg_embeddings[t1] * avg_embeddings[t2]).sum().item()

for c1 in cherries:
    for c2 in cherries:
        if c1 < c2:
            print(f"{c1} — {c2}: {similarity(c1, c2):.3f}")