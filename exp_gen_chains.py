"""
Chains-only generation pass: produce the planner's own chains for any split,
saved as results.json ({index, prompt, reasoning}) usable as a --chains_file
by exp_none_control / exp_margin_repair / exp_detector_eval.

Needed for the held-out train-split replication (699 prompts we never used):
no pre-existing chains file covers that split. Chains generated at seed 1000
are IDENTICAL to those exp_minimal_repair generates internally (same seed,
same sampling path), so paired comparisons across conditions remain valid.

Usage:
  python exp_gen_chains.py --subset spatial --split train
Output: r3_outputs/chains_{subset}_{split}/results.json   (~5s/chain, no images)
"""

import argparse
from exp_common import GotR1, set_seed, load_prompts, run_checkpointed

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", default="ckpts/GoT-R1-1B")
parser.add_argument("--subset", choices=["spatial", "color"], required=True)
parser.add_argument("--split", default="train")
parser.add_argument("--prompt_dir", default="T2I-CompBench/examples/dataset")
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--max_prompts", type=int, default=-1)
args = parser.parse_args()

bot = GotR1(args.ckpt_path)
prompts = load_prompts(args.prompt_dir, args.subset, args.split, args.max_prompts)


def work(idx, prompt_text):
    set_seed(args.seed)
    _, chain_text = bot.generate_reasoning_chain(bot.format_prompt(prompt_text))
    return None, {"reasoning": chain_text}


run_checkpointed(f"r3_outputs/chains_{args.subset}_{args.split}", prompts, work)
