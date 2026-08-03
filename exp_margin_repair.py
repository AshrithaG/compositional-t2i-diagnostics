"""
ICLR Phase A: Geometric plan repair — the causal margin dose-response.

Takes the planner's OWN chains (same file as the control condition), keeps
objects and prose untouched, and rewrites ONLY the two bounding boxes so that
their centers satisfy the PROMPT relation with a target normalized margin
along the relation axis (box sizes preserved; off-axis centers preserved).
Sweeping the margin turns the observational margin-accuracy curve into a
causal dose-response, and doubles as a deployable inference-time method
("geometric plan repair").

Analysis note: for prompts whose original plan was VALID, this edit changes
margin only (prose stays consistent) — the clean causal subset. For invalid
plans it also fixes the relation sign (prose may then contradict boxes, as in
minimal repair). results.json records was_valid + original/new margins so the
analysis can split these.

Usage (three conditions of the dose-response):
  python exp_margin_repair.py --target_margin 0.2
  python exp_margin_repair.py --target_margin 0.4
  python exp_margin_repair.py --target_margin 0.6
Outputs: r3_outputs/margin_repair_m{margin}_spatial_{split}[_seed{seed}]/
Symmetric-relation prompts (near/next to/on side of) pass through unmodified
and are recorded with modified=false (internal control).
"""

import argparse, json, re
from exp_common import (GotR1, set_seed, load_prompts, run_checkpointed,
                        parse_spatial_prompt, parse_boxes_from_chain,
                        ASYMMETRIC_RELATIONS)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", default="ckpts/GoT-R1-1B")
parser.add_argument("--split", default="val")
parser.add_argument("--target_margin", type=float, required=True,
                    help="normalized center margin along the relation axis (0-1)")
parser.add_argument("--chains_file", default="reasoning_chains_spatial.json")
parser.add_argument("--prompt_dir", default="T2I-CompBench/examples/dataset")
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--cfg_weight", type=float, default=5.0)
parser.add_argument("--max_prompts", type=int, default=-1)
args = parser.parse_args()

BOX_PAT = re.compile(r'<\|box_start\|>\(\d+,\d+\),\(\d+,\d+\)<\|box_end\|>')
AXIS = {"on the left of": (0, -1), "on the right of": (0, +1),
        "on the top of": (1, -1), "on the bottom of": (1, +1)}
# (axis, sign): sign=-1 means obj1 center must be SMALLER on that axis (left/top)


def retarget_boxes(b1, b2, rel, margin):
    """Move centers to +/- margin/2 around the axis midpoint; keep sizes and
    off-axis centers; clamp into [0,999]."""
    axis, sign = AXIS[rel]
    m = margin * 1000 / 2
    lo, hi = 500 - m, 500 + m
    c1_axis, c2_axis = (lo, hi) if sign < 0 else (hi, lo)

    def rebuild(b, new_axis_center):
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if axis == 0:
            cx = new_axis_center
        else:
            cy = new_axis_center
        nx1 = int(max(0, min(999 - w, cx - w / 2)))
        ny1 = int(max(0, min(999 - h, cy - h / 2)))
        return (nx1, ny1, int(nx1 + w), int(ny1 + h))

    return rebuild(b1, c1_axis), rebuild(b2, c2_axis)


def rewrite_chain(chain, new_boxes):
    it = iter(new_boxes)

    def sub(m):
        try:
            b = next(it)
            return f"<|box_start|>({b[0]},{b[1]}),({b[2]},{b[3]})<|box_end|>"
        except StopIteration:
            return m.group(0)
    return BOX_PAT.sub(sub, chain)


chains = {e["index"]: e["reasoning"] for e in json.load(open(args.chains_file))}
bot = GotR1(args.ckpt_path, cfg_weight=args.cfg_weight)
prompts = load_prompts(args.prompt_dir, "spatial", args.split, args.max_prompts)


def work(idx, prompt_text):
    chain = chains.get(idx)
    if not chain:
        return None, {"skipped": "no_chain"}
    o1, rel, o2 = parse_spatial_prompt(prompt_text)
    parsed = parse_boxes_from_chain(chain)

    modified, was_valid, orig_margin, achieved_margin = False, None, None, None
    final_chain = chain
    if rel in ASYMMETRIC_RELATIONS and len(parsed) >= 2:
        axis, sign = AXIS[rel]
        c1, c2 = parsed[0]["center"], parsed[1]["center"]
        orig_margin = abs(c1[axis] - c2[axis]) / 1000.0
        was_valid = (c1[axis] < c2[axis]) if sign < 0 else (c1[axis] > c2[axis])
        nb1, nb2 = retarget_boxes(parsed[0]["bbox"], parsed[1]["bbox"], rel,
                                  args.target_margin)
        # clamping preserves box sizes, so very large boxes cannot reach large
        # margins — record the ACHIEVED margin; analyses bin on this, not the target
        nc1 = ((nb1[0] + nb1[2]) / 2, (nb1[1] + nb1[3]) / 2)
        nc2 = ((nb2[0] + nb2[2]) / 2, (nb2[1] + nb2[3]) / 2)
        achieved_margin = abs(nc1[axis] - nc2[axis]) / 1000.0
        # remaining boxes (3rd+) untouched
        new_boxes = [nb1, nb2] + [p["bbox"] for p in parsed[2:]]
        final_chain = rewrite_chain(chain, new_boxes)
        modified = True

    set_seed(args.seed)
    img = bot.generate_image_from_chain(
        bot.format_prompt(prompt_text), bot.chain_text_to_ids(final_chain))
    return img, {"reasoning": final_chain, "modified": modified,
                 "was_valid": was_valid, "orig_margin": orig_margin,
                 "achieved_margin": achieved_margin,
                 "target_margin": args.target_margin}


suffix = "" if args.seed == 1000 else f"_seed{args.seed}"
out = f"r3_outputs/margin_repair_m{args.target_margin}_spatial_{args.split}{suffix}"
results = run_checkpointed(out, prompts, work)
n_mod = sum(1 for r in results if r.get("modified"))
n_fix = sum(1 for r in results if r.get("modified") and r.get("was_valid") is False)
print(f"Modified: {n_mod} | of which sign-fixed (originally invalid): {n_fix}")
