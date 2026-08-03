"""
Experiment: Minimal in-distribution plan repair.

The oracle experiment showed that hand-built, logically-correct chains HURT.
This experiment isolates style-vs-content: take the model's OWN chain and,
only when it violates the prompt constraint, apply the smallest possible fix
  - spatial: swap the two bounding-box strings (flips the relation, preserves
    box statistics, phrasing, and everything else)
  - color:   swap the two color words throughout the chain
then decode as usual. If minimal repair HELPS where alien oracles HURT, the
degradation is caused by distribution shift, not by plan editing per se.

Usage:
  python exp_minimal_repair.py --subset spatial
  python exp_minimal_repair.py --subset color
  python exp_minimal_repair.py --subset spatial --split test   # full benchmark

Outputs: r3_outputs/minimal_repair_{subset}_{split}/
  images (all 300, repaired-or-not), results.json with per-prompt fields:
  was_valid, repaired, original_chain, final_chain, verify_details
Compare against baseline and oracle runs on (a) all prompts, (b) the repaired
subset only — (b) is the headline comparison.
"""

import argparse, re
from exp_common import (GotR1, set_seed, load_prompts, run_checkpointed,
                        verify_chain, parse_boxes_from_chain, parse_color_prompt)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", default="ckpts/GoT-R1-1B")
parser.add_argument("--subset", choices=["spatial", "color"], required=True)
parser.add_argument("--split", default="val")
parser.add_argument("--prompt_dir", default="T2I-CompBench/examples/dataset")
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--cfg_weight", type=float, default=5.0)
parser.add_argument("--max_prompts", type=int, default=-1)
args = parser.parse_args()


def repair_spatial(chain_text):
    parsed = parse_boxes_from_chain(chain_text)
    if len(parsed) < 2:
        return chain_text, False
    b1, b2 = parsed[0]["bbox"], parsed[1]["bbox"]
    s1 = f"({b1[0]},{b1[1]}),({b1[2]},{b1[3]})"
    s2 = f"({b2[0]},{b2[1]}),({b2[2]},{b2[3]})"
    if s1 == s2:
        return chain_text, False
    fixed = chain_text.replace(s1, "\x00BOX\x00").replace(s2, s1).replace("\x00BOX\x00", s2)
    return fixed, True


def repair_color(chain_text, prompt):
    """Insert each missing prompt color into its object's span(s).

    The verifier fails a color chain only when a required color word is absent
    from the chain entirely, so swap-based repair can never fix it; the minimal
    content-preserving edit is insertion: '<|obj_start|>banana' ->
    '<|obj_start|>red banana' (dropping any other prompt color already
    prefixed to that object). Falls back to a prepended plan sentence when the
    object span cannot be located.
    """
    pairs = parse_color_prompt(prompt)
    if not pairs or len(pairs) < 2:
        return chain_text, False
    prompt_colors = {c for c, _ in pairs}
    fixed, changed = chain_text, False
    for color, obj in pairs:
        if re.search(rf"\b{color}\b", fixed):
            continue                      # color word already present
        span_re = re.compile(
            rf"(<\|obj_start\|>)([^<]*\b{re.escape(obj)}\b[^<]*)(<\|obj_end\|>)")
        m = span_re.search(fixed)
        if m:
            span = m.group(2)
            for other in prompt_colors - {color}:   # drop wrong prompt-color prefix
                span = re.sub(rf"\b{other}\b\s*", "", span)
            fixed = fixed[:m.start()] + m.group(1) + f"{color} " + span.lstrip() \
                + m.group(3) + fixed[m.end():]
        else:
            fixed = f"The {obj} is {color}. " + fixed
        changed = True
    return fixed, changed


bot = GotR1(args.ckpt_path, cfg_weight=args.cfg_weight)
prompts = load_prompts(args.prompt_dir, args.subset, args.split, args.max_prompts)


def work(idx, prompt_text):
    prompt_ids = bot.format_prompt(prompt_text)

    set_seed(args.seed)
    chain_ids, chain_text = bot.generate_reasoning_chain(prompt_ids)
    was_valid, details = verify_chain(chain_text, prompt_text, args.subset)

    repaired = False
    final_chain = chain_text
    if not was_valid:
        if args.subset == "spatial":
            final_chain, repaired = repair_spatial(chain_text)
        else:
            final_chain, repaired = repair_color(chain_text, prompt_text)
        # confirm repair actually fixed the constraint
        now_valid, _ = verify_chain(final_chain, prompt_text, args.subset)
        if repaired and not now_valid:
            repaired = False          # unfixable (e.g. <2 boxes) — keep original
            final_chain = chain_text

    # Uniform conditioning path: ALL chains (repaired or not) go through the
    # same decode->re-encode route, so the repaired-vs-unrepaired comparison
    # isolates the repair itself rather than a tokenization difference.
    set_seed(args.seed)
    img = bot.generate_image_from_chain(prompt_ids, bot.chain_text_to_ids(final_chain))

    return img, {
        "was_valid": was_valid,
        "repaired": repaired,
        "verify_details": details,
        "reasoning": final_chain,
        "original_chain": chain_text if repaired else None,
    }


suffix = "" if args.seed == 1000 else f"_seed{args.seed}"
results = run_checkpointed(
    f"r3_outputs/minimal_repair_{args.subset}_{args.split}{suffix}", prompts, work)

n_invalid = sum(1 for r in results if not r["was_valid"])
n_repaired = sum(1 for r in results if r["repaired"])
print(f"Invalid plans: {n_invalid}/{len(results)} | Repaired: {n_repaired}")
print("Evaluate with vm_blip_eval.py / run_official_eval.py; compare the "
      "repaired-index subset against the same indices in baseline & oracle runs.")
