"""
Experiment: Is degradation predicted by how out-of-distribution the chain is?

Scores every chain in one or more results/chains JSON files with its
teacher-forced NLL under GoT-R1's own planner (conditioned on the prompt).
If per-prompt metric degradation correlates with per-prompt chain NLL, the
co-adaptation interpretation gains direct mechanistic evidence — and the NLL
gap between oracle conditions (alien vs empirical vs the model's own chains)
quantifies the distribution shift itself.

No image generation — forward passes only, minutes not hours.

Usage:
  python exp_chain_nll.py --chains reasoning_chains_spatial.json \
      r3_outputs/oracle_minimal_alien_spatial_val/results.json \
      r3_outputs/oracle_verbose_empirical_spatial_val/results.json \
      r3_outputs/minimal_repair_spatial_val/results.json
  → chain_nll_scores.json  (one record per (source_file, index))

Then correlate: join on index with per-image metric outputs (BLIP-VQA /
UniDet) and report Pearson/Spearman between mean_nll and metric drop.
"""

import argparse, json, os
from tqdm import tqdm
from exp_common import GotR1

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", default="ckpts/GoT-R1-1B")
parser.add_argument("--chains", nargs="+", required=True,
                    help="JSON files: list of {index, prompt, reasoning}")
parser.add_argument("--out", default="chain_nll_scores.json")
args = parser.parse_args()

bot = GotR1(args.ckpt_path)
records = []

for path in args.chains:
    entries = json.load(open(path))
    tag = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
    print(f"Scoring {len(entries)} chains from {path}")
    for e in tqdm(entries):
        chain = e.get("reasoning")
        if not chain:
            continue
        prompt_ids = bot.format_prompt(e["prompt"])
        stats = bot.chain_nll(prompt_ids, chain)
        records.append({"source": tag, "index": e["index"],
                        "prompt": e["prompt"], **stats})

json.dump(records, open(args.out, "w"), indent=2)

# quick per-source summary
from collections import defaultdict
by_src = defaultdict(list)
for r in records:
    by_src[r["source"]].append(r["mean_nll"])
print(f"\n{'source':45s} {'n':>5s} {'mean NLL':>9s}")
for src, vals in by_src.items():
    print(f"{src:45s} {len(vals):5d} {sum(vals)/len(vals):9.3f}")
print(f"\nWrote {args.out}")
