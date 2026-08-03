"""
Control condition for the perturbation study: generate images from the SAME
pre-existing chains, through the SAME re-tokenization + generation path as the
perturbed conditions, but with NO perturbation. This is the correct baseline
for all perturb_* comparisons (same machine, same torch, same seed, same code
path) — the February baseline images came from different hardware/library
versions and are not sampling-comparable.

Usage:
  python exp_none_control.py --subset spatial --chains_file reasoning_chains_spatial.json
  python exp_none_control.py --subset color   --chains_file reasoning_chains_color.json
Output: r3_outputs/perturb_none_{subset}/
"""

import argparse, json
from exp_common import GotR1, set_seed, load_prompts, run_checkpointed

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", default="ckpts/GoT-R1-1B")
parser.add_argument("--subset", choices=["spatial", "color"], required=True)
parser.add_argument("--chains_file", required=True)
parser.add_argument("--split", default="val")
parser.add_argument("--prompt_dir", default="T2I-CompBench/examples/dataset")
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--cfg_weight", type=float, default=5.0)
args = parser.parse_args()

chains = {e["index"]: e["reasoning"] for e in json.load(open(args.chains_file))}
bot = GotR1(args.ckpt_path, cfg_weight=args.cfg_weight)
prompts = load_prompts(args.prompt_dir, args.subset, args.split)


def work(idx, prompt_text):
    chain = chains.get(idx)
    if not chain:
        return None, {"skipped": "no_chain"}
    prompt_ids = bot.format_prompt(prompt_text)
    set_seed(args.seed)
    img = bot.generate_image_from_chain(prompt_ids, bot.chain_text_to_ids(chain))
    return img, {"reasoning": chain}


suffix = "" if args.split == "val" else f"_{args.split}"
suffix += "" if args.seed == 1000 else f"_seed{args.seed}"
run_checkpointed(f"r3_outputs/perturb_none_{args.subset}{suffix}", prompts, work)
