"""
Experiment: Mention-order control for the left/right planning asymmetry.

R2 found GoT-R1 plans "on the left of" correctly ~100% but "on the right of"
only ~56%. Two competing explanations:
  (a) directional prior: the planner is simply worse at "right"
  (b) mention-order bias (Order-Is-Not-Layout): the first-mentioned entity
      tends to be placed first/left, which AGREES with left-prompts and
      CONFLICTS with right-prompts.
For every asymmetric spatial prompt "X <rel> Y" we also plan its semantically
equivalent inversion "Y <inv-rel> X" (same layout, opposite mention order and
relation word). Comparing planning accuracy across the 2x2 of
(relation word, which object is mentioned first) separates (a) from (b).

Chains only — no image decoding. ~5s/chain: 2 variants x N seeds x ~230
asymmetric prompts ≈ 2-6 GPU-hours depending on --num_seeds.

Usage:
  python exp_order_swap.py --num_seeds 3
Outputs: r3_outputs/order_swap/results.json + printed 2x2 table.
"""

import argparse, json
from pathlib import Path
from tqdm import tqdm
from exp_common import (GotR1, set_seed, load_prompts, verify_chain,
                        parse_spatial_prompt, ASYMMETRIC_RELATIONS, INVERSE_RELATION)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", default="ckpts/GoT-R1-1B")
parser.add_argument("--split", default="val")
parser.add_argument("--prompt_dir", default="T2I-CompBench/examples/dataset")
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--num_seeds", type=int, default=3)
parser.add_argument("--max_prompts", type=int, default=-1)
args = parser.parse_args()

bot = GotR1(args.ckpt_path)
prompts = load_prompts(args.prompt_dir, "spatial", args.split, args.max_prompts)

out = Path("r3_outputs/order_swap")
out.mkdir(parents=True, exist_ok=True)
results_file = out / "results.json"
results = json.load(open(results_file)) if results_file.exists() else []
done = {(r["index"], r["variant"], r["seed_offset"]) for r in results}

tasks = []
for idx, prompt in enumerate(prompts):
    o1, rel, o2 = parse_spatial_prompt(prompt)
    if rel not in ASYMMETRIC_RELATIONS:
        continue
    variants = {
        "original": prompt,                                  # "X rel Y"
        "inverted": f"{o2} {INVERSE_RELATION[rel]} {o1}",    # same layout, order swapped
    }
    for variant, vprompt in variants.items():
        for s in range(args.num_seeds):
            if (idx, variant, s) not in done:
                tasks.append((idx, variant, vprompt, s))

print(f"Asymmetric prompts: {len({t[0] for t in tasks} | {r['index'] for r in results})} "
      f"| chain generations remaining: {len(tasks)}")

for count, (idx, variant, vprompt, s) in enumerate(tqdm(tasks)):
    o1, rel, o2 = parse_spatial_prompt(vprompt)
    prompt_ids = bot.format_prompt(vprompt)
    set_seed(args.seed + s)
    _, chain_text = bot.generate_reasoning_chain(prompt_ids)
    ok, details = verify_chain(chain_text, vprompt, "spatial")
    results.append({
        "index": idx, "variant": variant, "seed_offset": s,
        "prompt": vprompt, "relation": rel,
        "plan_valid": bool(ok), "checkable": details.get("reason") == "spatial_check",
        "reasoning": chain_text,
    })
    if (count + 1) % 25 == 0:
        json.dump(results, open(results_file, "w"), indent=2)
json.dump(results, open(results_file, "w"), indent=2)

# ── 2x2 summary: relation word x mention order ──────────────────────────────
from collections import defaultdict
cell = defaultdict(lambda: [0, 0])   # (relation, variant) -> [correct, checkable]
for r in results:
    if r["checkable"]:
        c = cell[(r["relation"], r["variant"])]
        c[1] += 1
        c[0] += int(r["plan_valid"])

print(f"\n{'relation':20s} {'variant':10s} {'plan acc':>9s} {'n':>5s}")
for (rel, var), (c, n) in sorted(cell.items()):
    print(f"{rel:20s} {var:10s} {c/n:9.1%} {n:5d}")
print("\nReading: if accuracy tracks the RELATION WORD regardless of variant, "
      "it's a directional prior; if 'inverted' right-prompts (object order "
      "swapped) recover accuracy, it's mention-order bias.")
