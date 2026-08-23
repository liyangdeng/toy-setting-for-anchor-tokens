import re
from pathlib import Path

import numpy as np


RESULTS_DIR = Path("additional_sr_punct_results")

settings = ["shared", "none", "disjoint"]
seeds = [42, 43, 44]

results = {
    setting: {"top1": [], "top5": []}
    for setting in settings
}


for setting in settings:
    for seed in seeds:

        path = RESULTS_DIR / f"{setting}_seed{seed}.txt"
        text = path.read_text(encoding="utf-8")

        top1_match = re.search(
            r"top-1 precision\s*:\s*([0-9.]+)",
            text
        )
        top5_match = re.search(
            r"top-5 precision\s*:\s*([0-9.]+)",
            text
        )

        if top1_match is None or top5_match is None:
            raise ValueError(f"Could not parse results from {path}")

        top1 = float(top1_match.group(1))
        top5 = float(top5_match.group(1))

        results[setting]["top1"].append(top1)
        results[setting]["top5"].append(top5)


print("Punctuation SR: mean ± SD across seeds")

for setting in settings:

    top1 = np.array(results[setting]["top1"])
    top5 = np.array(results[setting]["top5"])

    top1_mean = top1.mean()
    top1_sd = top1.std(ddof=1)

    top5_mean = top5.mean()
    top5_sd = top5.std(ddof=1)

    print(f"\n{setting}")
    print(f"  top-1: {top1_mean:.4f} ± {top1_sd:.4f}")
    print(f"  top-5: {top5_mean:.4f} ± {top5_sd:.4f}")

    print(
        "  seeds top-1:",
        ", ".join(f"{x:.4f}" for x in top1)
    )
    print(
        "  seeds top-5:",
        ", ".join(f"{x:.4f}" for x in top5)
    )