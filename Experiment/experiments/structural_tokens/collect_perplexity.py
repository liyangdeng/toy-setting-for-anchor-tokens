"""
Collect final dev perplexity from every trained checkpoint into one txt.

usage:
    python collect_perplexity.py --glob '1_PUNCT/punct_checkpoints_*' --out punct_perplexity.txt
    python collect_perplexity.py --glob '1_ST/st_checkpoints_*' --out st_perplexity.txt

    # multiple directories/patterns can also be collected together
    python collect_perplexity.py \
        --glob '1_PUNCT/punct_checkpoints_*' '1_ST/st_checkpoints_*' \
        --out all_perplexity.txt

"""
import argparse
import glob
import json
import math
import os
import re


def final_eval_loss(run_dir):

    candidates = [
        os.path.join(run_dir, 'trainer_state.json'),
        os.path.join(run_dir, 'final', 'trainer_state.json'),
    ]

    candidates.extend(
        glob.glob(os.path.join(run_dir, 'checkpoint-*', 'trainer_state.json'))
    )

    states = []

    for cand in candidates:
        if os.path.isfile(cand):
            state = json.load(open(cand))
            states.append(state)

    if not states:
        return None

    # use the trainer state from the latest checkpoint
    state = max(states, key=lambda s: s.get('global_step', 0))

    evals = [
        r['eval_loss']
        for r in state.get('log_history', [])
        if 'eval_loss' in r
    ]

    if evals:
        return min(evals)

    return None


def parse_name(name):
    m = re.search(r'_(shared|none|disjoint)_seed(\d+)', name)
    return (m.group(1), int(m.group(2))) if m else (name, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = []
    seen = set()

    for pattern in args.glob:
        for d in sorted(glob.glob(pattern)):
            if not os.path.isdir(d):
                continue

            d = os.path.normpath(d)
            if d in seen:
                continue
            seen.add(d)

            setting, seed = parse_name(os.path.basename(d))
            loss = final_eval_loss(d)
            ppl = math.exp(loss) if loss is not None else None
            rows.append((setting, seed, ppl, d))

    order = {'shared': 0, 'none': 1, 'disjoint': 2}
    rows.sort(key=lambda r: (order.get(r[0], 9), r[1], r[3]))

    with open(args.out, 'w') as f:
        f.write(f"{'setting':<10} {'seed':<6} {'dev_ppl':<10} run\n")
        for setting, seed, ppl, d in rows:
            ppl_s = f"{ppl:.2f}" if ppl is not None else "NA"
            f.write(f"{setting:<10} {seed:<6} {ppl_s:<10} {d}\n")

        # per-setting mean +/- std across seeds
        f.write("\n# mean +/- std across seeds\n")
        by_setting = {}
        for setting, seed, ppl, d in rows:
            if ppl is not None:
                by_setting.setdefault(setting, []).append(ppl)
        for setting in sorted(by_setting, key=lambda s: order.get(s, 9)):
            vals = by_setting[setting]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            f.write(f"{setting:<10} n={len(vals)}  {mean:.2f} +/- {std:.2f}\n")

    print(f"wrote {len(rows)} runs -> {args.out}")


if __name__ == '__main__':
    main()
