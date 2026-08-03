"""
Experiment: Oracle distribution dial (v2 — post-audit).

v2 fixes over v1 (v1 runs are archived, not comparable):
  - objects are de-articled before templating (v1 emitted "A a rabbit")
  - verbose template's second clause is direction-correct mirror phrasing
    (v1 emitted garbled "appears distinctly of" / layout-contradicting text)
  - sample_boxes fallback no longer reuses variables from a rejected loop
  - NEW style "mimic": planner-mimic oracles built by editing a REAL donor
    chain from the planner's own outputs (replace object names + reuse donor
    boxes), matching register/length/token statistics by construction —
    isolates plan CONTENT from plan STYLE.

Styles: minimal | verbose | mimic.  Boxes: alien | empirical (ignored by
mimic, which keeps donor boxes).

Usage:
  python exp_oracle_dial.py --subset spatial --style mimic
  python exp_oracle_dial.py --subset spatial --style verbose --boxes empirical
Outputs: r3_outputs/oracle2_{style}_{boxes}_{subset}_{split}/
"""

import argparse, json, random, re
from exp_common import (GotR1, set_seed, load_prompts, run_checkpointed,
                        parse_spatial_prompt, parse_color_prompt,
                        parse_boxes_from_chain, SPATIAL_RELATIONS, OBJ_RE)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", default="ckpts/GoT-R1-1B")
parser.add_argument("--subset", choices=["spatial", "color"], required=True)
parser.add_argument("--split", default="val")
parser.add_argument("--style", choices=["minimal", "verbose", "mimic"], required=True)
parser.add_argument("--boxes", choices=["alien", "empirical"], default="empirical")
parser.add_argument("--chains_file", default=None,
                    help="Planner chains JSON (default reasoning_chains_{subset}.json); "
                         "source of empirical box stats AND mimic donor chains")
parser.add_argument("--prompt_dir", default="T2I-CompBench/examples/dataset")
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--cfg_weight", type=float, default=5.0)
parser.add_argument("--max_prompts", type=int, default=-1)
args = parser.parse_args()

OBJ = lambda name: f"<|obj_start|>{name}<|obj_end|>"
BOX = lambda b: f"<|box_start|>({b[0]},{b[1]}),({b[2]},{b[3]})<|box_end|>"

def deart(s):
    return re.sub(r"^(a|an|the)\s+", "", s.strip(), flags=re.IGNORECASE)

REL_PHRASE = {
    "on the left of": "is positioned to the left of",
    "on the right of": "is positioned to the right of",
    "on the top of": "is positioned above",
    "on the bottom of": "is positioned below",
    "next to": "stands next to",
    "near": "is placed near",
    "on side of": "is on the side of",
}
# direction-correct description of o2's position relative to o1
MIRROR_PHRASE = {
    "on the left of": "to the right of",
    "on the right of": "to the left of",
    "on the top of": "below",
    "on the bottom of": "above",
    "next to": "beside",
    "near": "near",
    "on side of": "beside",
}

# ── planner chains: empirical box pool + mimic donors ───────────────────────

chains_file = args.chains_file or f"reasoning_chains_{args.subset}.json"
planner_entries = json.load(open(chains_file))

empirical_boxes, donors = [], []
for entry in planner_entries:
    parsed = parse_boxes_from_chain(entry["reasoning"])
    for p in parsed:
        x1, y1, x2, y2 = p["bbox"]
        if x2 > x1 and y2 > y1:
            empirical_boxes.append((x2 - x1, y2 - y1, (x1 + x2) / 2, (y1 + y2) / 2))
    if len(parsed) >= 2:
        donors.append({"index": entry["index"], "chain": entry["reasoning"],
                       "parsed": parsed})
print(f"{chains_file}: {len(empirical_boxes)} boxes, {len(donors)} mimic donors")


def clamp_box(cx, cy, w, h):
    x1 = int(max(0, min(999 - w, cx - w / 2)))
    y1 = int(max(0, min(999 - h, cy - h / 2)))
    return (x1, y1, int(x1 + w), int(y1 + h))


def sample_boxes(rel, rng):
    if args.boxes == "alien":
        pair = ((50, 250, 400, 750), (600, 250, 950, 750))
        if rel in ("on the top of", "on the bottom of"):
            pair = ((250, 30, 750, 450), (250, 550, 750, 970))
        a, b = pair
        if rel in ("on the right of", "on the bottom of"):
            a, b = b, a
        return a, b

    check = SPATIAL_RELATIONS.get(rel, (None, lambda a, b: True))[1]
    last = None
    for _ in range(200):
        w1, h1, cx1, cy1 = rng.choice(empirical_boxes)
        w2, h2, cx2, cy2 = rng.choice(empirical_boxes)
        last = (w1, h1, cx1, cy1, w2, h2, cx2, cy2)
        if check((cx1, cy1), (cx2, cy2)):
            return clamp_box(cx1, cy1, w1, h1), clamp_box(cx2, cy2, w2, h2)
    # fallback: keep the last sampled sizes, force-order the centers
    w1, h1, cx1, cy1, w2, h2, cx2, cy2 = last
    if rel == "on the left of":
        cx1, cx2 = 300, 700
    elif rel == "on the right of":
        cx1, cx2 = 700, 300
    elif rel == "on the top of":
        cy1, cy2 = 300, 700
    elif rel == "on the bottom of":
        cy1, cy2 = 700, 300
    return clamp_box(cx1, cy1, w1, h1), clamp_box(cx2, cy2, w2, h2)


def build_mimic(prompt_text, rng):
    """Edit a real donor chain: swap in target object names (and colors),
    keep the donor's boxes, prose, and length. For spatial prompts the donor
    must already satisfy the target relation with its own boxes."""
    if args.subset == "spatial":
        o1, rel, o2 = parse_spatial_prompt(prompt_text)
        if rel is None:
            return None
        o1, o2 = deart(o1), deart(o2)
        check = SPATIAL_RELATIONS.get(rel, (None, lambda a, b: True))[1]
        pool = [d for d in donors
                if check(d["parsed"][0]["center"], d["parsed"][1]["center"])]
        if not pool:
            return None
        donor = rng.choice(pool)
        new_names = [o1, o2]
    else:
        pairs = parse_color_prompt(prompt_text)
        if not pairs:
            return None
        donor = rng.choice(donors)
        new_names = [f"{c} {deart(o)}" for c, o in pairs]

    chain = donor["chain"]
    old_names = [p["object"] for p in donor["parsed"][:2]]
    # replace the object spans (first two), then donor names in the prose
    for old, new in zip(old_names, new_names):
        chain = chain.replace(f"<|obj_start|>{old}<|obj_end|>",
                              f"<|obj_start|>{new}<|obj_end|>", 1)
    for old, new in zip(old_names, new_names):
        bare_old = deart(re.sub(r"^\W+|\W+$", "", old))
        bare_new = new
        if bare_old and bare_old.lower() != bare_new.lower():
            chain = re.sub(rf"\b{re.escape(bare_old)}\b", bare_new, chain)
    return chain


def build_template(prompt_text, rng):
    if args.subset == "spatial":
        o1, rel, o2 = parse_spatial_prompt(prompt_text)
        if rel is None:
            return None
        o1, o2 = deart(o1), deart(o2)
        b1, b2 = sample_boxes(rel, rng)
        head = f"A {OBJ(o1)} {BOX(b1)} {REL_PHRASE[rel]} a {OBJ(o2)} {BOX(b2)}"
        if args.style == "minimal":
            return head + ".<begin_of_image>"
        return (f"{head}, creating a clear and balanced composition. "
                f"The {o1} is rendered with its characteristic shape, texture, and "
                f"natural proportions, fully visible within its region of the frame. "
                f"The {o2} appears {MIRROR_PHRASE[rel]} the {o1}, with its own "
                f"recognizable form and details. Both objects are set against a "
                f"simple, neutral background with soft, even lighting, and neither "
                f"object occludes the other, so the spatial arrangement is "
                f"unambiguous.<begin_of_image>")

    pairs = parse_color_prompt(prompt_text)
    if not pairs:
        return None
    (c1, o1), (c2, o2) = pairs
    o1, o2 = deart(o1), deart(o2)
    b1, b2 = sample_boxes("on the left of", rng)
    head = (f"A {OBJ(f'{c1} {o1}')} {BOX(b1)} and a {OBJ(f'{c2} {o2}')} {BOX(b2)} "
            f"appear side by side")
    if args.style == "minimal":
        return head + ".<begin_of_image>"
    return (f"{head} in a simple, well-lit scene. The {o1} is uniformly {c1}, its "
            f"surface clearly showing the {c1} coloration across its whole body. "
            f"The {o2} is uniformly {c2}, easily distinguished from the {o1} by "
            f"both its shape and its {c2} color. The two objects are separated in "
            f"the frame, set against a neutral background so each color binds "
            f"unmistakably to its own object.<begin_of_image>")


bot = GotR1(args.ckpt_path, cfg_weight=args.cfg_weight)
prompts = load_prompts(args.prompt_dir, args.subset, args.split, args.max_prompts)

box_tag = "donor" if args.style == "mimic" else args.boxes


def work(idx, prompt_text):
    rng = random.Random(args.seed + idx)
    chain = build_mimic(prompt_text, rng) if args.style == "mimic" \
        else build_template(prompt_text, rng)
    if chain is None:
        return None, {"skipped": "unparseable_or_no_donor"}
    if not chain.rstrip().endswith("<begin_of_image>"):
        chain = chain.rstrip() + "<begin_of_image>"
    prompt_ids = bot.format_prompt(prompt_text)
    set_seed(args.seed)
    img = bot.generate_image_from_chain(prompt_ids, bot.chain_text_to_ids(chain))
    return img, {"reasoning": chain, "style": args.style, "boxes": box_tag}


suffix = "" if args.seed == 1000 else f"_seed{args.seed}"
run_checkpointed(
    f"r3_outputs/oracle2_{args.style}_{box_tag}_{args.subset}_{args.split}{suffix}",
    prompts, work)
