"""
accuracy on a saved checkpoint:
    top1, top5, mean reciprocal rank
gets saved as res_accuracy.txt by default

usage:
    python accuracy.py \
        --model_dir ./checkpoints_monolingual/final \
        --dev_file ./checkpoints_monolingual/dev.txt
"""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import ( BertForMaskedLM, PreTrainedTokenizerFast,DataCollatorForLanguageModeling )
from datasets import Dataset


def mlm_metrics(model, dataloader, device, topk=(1, 5)):
    model.eval()
    correct_by_k = {k: 0 for k in topk}
    rr_sum = 0.0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            masked = labels != -100
            masked_logits = logits[masked]
            gold = labels[masked]

            sorted_ids = masked_logits.argsort(dim=-1, descending=True)
            for k in topk:
                correct_by_k[k] += (sorted_ids[:, :k] == gold.unsqueeze(1)).any(dim=1).sum().item()

            ranks = (sorted_ids == gold.unsqueeze(1)).nonzero()[:, 1].float() + 1
            rr_sum += (1.0 / ranks).sum().item()

            total += masked.sum().item()

    return {
        **{f"top{k}_accuracy": correct_by_k[k] / total for k in topk},
        "mrr": rr_sum / total,
        "n_masked": total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--dev_file",  required=True)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--mlm_prob",   type=float, default=0.15)
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_dir)
    model = BertForMaskedLM.from_pretrained(args.model_dir).to(device)

    sentences = Path(args.dev_file).read_text(encoding="utf-8").strip().split("\n")
    sentences = [s for s in sentences if s.strip()]

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)

    dataset = Dataset.from_dict({"text": sentences})
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # apply masking
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    results = mlm_metrics(model, dataloader, device)

    output_lines = [
        f"Model directory: {args.model_dir}",
        f"Dev file: {args.dev_file}",
        f"Device: {device}",
        f"Dev sentences: {len(sentences)}",
        f"Masked tokens evaluated: {results['n_masked']}",
        f"top-1 accuracy: {results['top1_accuracy']:.4f}",
        f"top-5 accuracy: {results['top5_accuracy']:.4f}",
        f"mrr: {results['mrr']:.4f}",
    ]

    output_text = "\n".join(output_lines)

    print(output_text)

    default_filename = "res_accuracy.txt"

    filename = input(
        f"Enter the desired filename for the results: [{default_filename}]: "
    ).strip()

    if not filename:
        filename = default_filename

    if not filename.lower().endswith(".txt"):
        filename += ".txt"

    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / filename

    output_path.write_text(output_text + "\n", encoding="utf-8")

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()