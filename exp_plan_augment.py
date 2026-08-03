"""
Phase 7a: Build a plan-format-diversity training set (self-supervised).

For each baseline (prompt, chain, image) triple, emit training pairs where the
TARGET is the model's original image (as VQ tokens, encoded in
train_plan_robust_lora.py) and the CONDITIONING chain is a content-preserving
reformatting of the original chain. The decoder thereby learns to execute the
same plan content regardless of surface form — the hypothesized cure for
planner-decoder co-adaptation.

Augmentation families (content-preserving by construction):
  identity      original chain (keeps the model on-distribution)
  box_jitter    scale/translate both boxes by small factors; rejected if the
                prompt relation would flip
  box_stats     replace boxes with same-relation boxes sampled from the
                empirical pool (as exp_oracle_dial --boxes empirical)
  minimal       terse template with the ORIGINAL boxes (style shift only)
  verbose       templated descriptive paragraph with the original boxes
  reorder       object segments listed in reverse order, text glue adjusted

Usage:
  python exp_plan_augment.py --subset spatial --chains_file reasoning_chains_spatial.json
  python exp_plan_augment.py --subset color   --chains_file reasoning_chains_color.json
Output: plan_augmented_train.jsonl (appends across invocations)
  {"prompt", "chain", "aug_type", "subset", "image_path", "source_index"}
"""

import argparse, json, random, re
from exp_common import (parse_boxes_from_chain, parse_spatial_prompt,
                        parse_color_prompt, verify_chain, BOX_RE)

parser = argparse.ArgumentParser()
parser.add_argument("--subset", choices=["spatial", "color"], required=True)
parser.add_argument("--chains_file", required=True)
parser.add_argument("--image_dir", default=None,
                    help="Directory with baseline images {index}.png "
                         "(default: infer from chains entries' image_path)")
parser.add_argument("--out", default="plan_augmented_train.jsonl")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--jitter_variants", type=int, default=2)
args = parser.parse_args()

rng = random.Random(args.seed)
OBJ = lambda n: f"<|obj_start|>{n}<|obj_end|>"
BOXTOK = lambda b: f"<|box_start|>({b[0]},{b[1]}),({b[2]},{b[3]})<|box_end|>"


def replace_boxes(chain, new_boxes):
    """Substitute the i-th box token with new_boxes[i], preserving all text."""
    it = iter(new_boxes)

    def sub(m):
        try:
            return BOXTOK(next(it))
        except StopIteration:
            return m.group(0)
    return re.sub(BOX_RE.replace("(\\d+)", "\\d+").replace("\\(", "\\(") if False else
                  r'<\|box_start\|>\(\d+,\d+\),\(\d+,\d+\)<\|box_end\|>', sub, chain)


def jitter_box(b, rng):
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    s = rng.uniform(0.8, 1.2)
    dx, dy = rng.randint(-80, 80), rng.randint(-80, 80)
    cx, cy = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
    nw, nh = w * s, h * s
    nx1 = int(max(0, min(999 - nw, cx - nw / 2)))
    ny1 = int(max(0, min(999 - nh, cy - nh / 2)))
    return (nx1, ny1, int(nx1 + nw), int(ny1 + nh))


def make_variants(prompt, chain):
    parsed = parse_boxes_from_chain(chain)
    variants = [("identity", chain)]
    if len(parsed) < 2:
        return variants
    boxes = [p["bbox"] for p in parsed]

    # box_jitter (validity-checked)
    for _ in range(args.jitter_variants):
        cand = replace_boxes(chain, [jitter_box(b, rng) for b in boxes])
        ok, _ = verify_chain(cand, prompt, args.subset)
        if ok:
            variants.append(("box_jitter", cand))

    # style templates with ORIGINAL boxes
    if args.subset == "spatial":
        o1, rel, o2 = parse_spatial_prompt(prompt)
        if rel:
            b1, b2 = boxes[0], boxes[1]
            variants.append(("minimal",
                f"A {OBJ(o1)} {BOXTOK(b1)} {rel} a {OBJ(o2)} {BOXTOK(b2)}.<begin_of_image>"))
            variants.append(("verbose",
                f"A {OBJ(o1)} {BOXTOK(b1)} is positioned {rel} a {OBJ(o2)} {BOXTOK(b2)} "
                f"in a clean, well-lit scene. The {o1} shows its typical shape and texture, "
                f"and the {o2} is clearly visible in its own region of the frame, so the "
                f"layout is unambiguous.<begin_of_image>"))
    else:
        pairs = parse_color_prompt(prompt)
        if pairs:
            (c1, o1), (c2, o2) = pairs
            b1, b2 = boxes[0], boxes[1]
            variants.append(("minimal",
                f"A {OBJ(f'{c1} {o1}')} {BOXTOK(b1)} and a {OBJ(f'{c2} {o2}')} "
                f"{BOXTOK(b2)}.<begin_of_image>"))

    # reorder object segments
    seg_re = r'(<\|obj_start\|>.*?<\|box_end\|>)'
    segs = re.findall(seg_re, chain, re.DOTALL)
    if len(segs) == 2 and segs[0] != segs[1]:
        swapped = chain.replace(segs[0], "\x00A\x00").replace(segs[1], segs[0]) \
                       .replace("\x00A\x00", segs[1])
        ok, _ = verify_chain(swapped, prompt, args.subset)
        # NOTE: reorder swaps which object gets which box — only valid for
        # symmetric relations / when the check still passes
        if ok:
            variants.append(("reorder", swapped))
    return variants


entries = json.load(open(args.chains_file))
n_out = 0
with open(args.out, "a") as f:
    for e in entries:
        chain, prompt = e.get("reasoning"), e["prompt"]
        if not chain:
            continue
        img = e.get("image_path") or (f"{args.image_dir}/{e['index']}.png"
                                      if args.image_dir else None)
        for aug_type, variant in make_variants(prompt, chain):
            f.write(json.dumps({
                "prompt": prompt, "chain": variant, "aug_type": aug_type,
                "subset": args.subset, "image_path": img,
                "source_index": e["index"]}) + "\n")
            n_out += 1
print(f"Wrote {n_out} training pairs to {args.out}")
