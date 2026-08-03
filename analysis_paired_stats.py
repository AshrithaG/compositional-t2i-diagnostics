"""
Paired per-prompt statistics for condition comparisons (no GPU).

Every eval JSON stores per-image scores; conditions share prompts and seeds,
so paired tests are far more powerful than comparing two means. For each
requested pair of eval files this computes, on the INTERSECTION of scored
indices: mean difference, a 10k-resample bootstrap 95% CI, and a sign-flip
permutation p-value.

Usage:
  python analysis_paired_stats.py --field spatial_correct \
      --a eval_out/perturb_none_spatial.json --b eval_out/verify_spatial_retries5.json
  python analysis_paired_stats.py --field avg_color_match \
      --a eval_out/perturb_none_color.json --b eval_out/perturb_attr_swap_color.json
Fields: spatial eval: obj1_present obj2_present spatial_correct combined
        color eval:   avg_presence avg_color_yes_no avg_color_match combined
        detector eval: relation_correct iou_obj1 iou_obj2 followed_planned_relation
"""

import argparse, json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="baseline/control eval JSON")
ap.add_argument("--b", required=True, help="treatment eval JSON")
ap.add_argument("--field", required=True)
ap.add_argument("--resamples", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = random.Random(args.seed)


def load(path):
    data = json.load(open(path))
    return {r["index"]: float(r[args.field])
            for r in data["per_image"]
            if args.field in r and r[args.field] is not None}


A, B = load(args.a), load(args.b)
common = sorted(set(A) & set(B))
diffs = [B[i] - A[i] for i in common]
n = len(diffs)
mean_a = sum(A[i] for i in common) / n
mean_b = sum(B[i] for i in common) / n
mean_d = sum(diffs) / n

boot = []
for _ in range(args.resamples):
    s = [diffs[rng.randrange(n)] for _ in range(n)]
    boot.append(sum(s) / n)
boot.sort()
lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]

extreme = 0
for _ in range(args.resamples):
    s = sum(d if rng.random() < 0.5 else -d for d in diffs)
    if abs(s / n) >= abs(mean_d):
        extreme += 1
p = (extreme + 1) / (args.resamples + 1)

print(f"field={args.field}  n_paired={n}")
print(f"A (control):   {mean_a:.4f}   [{args.a}]")
print(f"B (treatment): {mean_b:.4f}   [{args.b}]")
print(f"paired mean diff (B-A): {mean_d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"sign-flip permutation p = {p:.4f}"
      f"   {'SIGNIFICANT' if p < 0.05 else 'not significant'} at alpha=0.05")
