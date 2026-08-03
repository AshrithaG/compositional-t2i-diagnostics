"""
Object-identity-MATCHED plan verification (post-audit re-analysis, no GPU).

The original verify_chain assumes the chain mentions objects in prompt order
(checks parsed[0] vs parsed[1]). If the planner tends to mention the left/top
object first REGARDLESS of prompt order — the very bias under study — a
correct-layout chain for a right-prompt gets scored invalid, and the reported
left/right asymmetry could be an artifact of the metric.

This script re-scores plan validity by MATCHING chain objects to prompt roles
by name (token overlap), then checking the relation with the correct
assignment. It reports, side by side:
  - naive (order-assumed) validity — the original metric
  - matched validity — the corrected metric
  - mention-order stats: how often the chain mentions the prompt's second
    object first, and where the first-mentioned object is placed (left/top?)

Run on:  reasoning_chains JSON files and/or order_swap results.json
  python analysis_matched_verify.py --chains reasoning_chains_spatial.json
  python analysis_matched_verify.py --order_swap r3_outputs/order_swap/results.json
"""

import argparse, json, re
from collections import defaultdict

SPATIAL_RELATIONS = {
    "on the left of": lambda c1, c2: c1[0] < c2[0],
    "on the right of": lambda c1, c2: c1[0] > c2[0],
    "on the top of": lambda c1, c2: c1[1] < c2[1],
    "on the bottom of": lambda c1, c2: c1[1] > c2[1],
}
STOP = {"a", "an", "the", "of", "on", "in"}


def parse_spatial_prompt(prompt):
    for rel in sorted(SPATIAL_RELATIONS, key=len, reverse=True):
        m = re.match(rf"^(.+?)\s+{re.escape(rel)}\s+(.+?)$", prompt, re.IGNORECASE)
        if m:
            return m.group(1).strip(), rel, m.group(2).strip()
    return None, None, None


def parse_chain(chain):
    objs = re.findall(r'<\|obj_start\|>(.*?)<\|obj_end\|>', chain)
    boxes = re.findall(r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>', chain)
    out = []
    for o, b in zip(objs, boxes):
        x1, y1, x2, y2 = map(int, b)
        out.append({"object": o.strip().lower(),
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2)})
    return out


def words(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP}


def match_roles(chain_objs, o1, o2):
    """Assign chain objects to prompt roles by token overlap.
    Returns (idx_for_o1, idx_for_o2) or None if ambiguous/unmatched."""
    w1, w2 = words(o1), words(o2)
    scores = [(i, len(words(co["object"]) & w1), len(words(co["object"]) & w2))
              for i, co in enumerate(chain_objs)]
    c1 = max(scores, key=lambda t: t[1])
    c2 = max(scores, key=lambda t: t[2])
    if c1[1] == 0 or c2[2] == 0 or c1[0] == c2[0]:
        return None
    return c1[0], c2[0]


def analyze(entries, label, variant_key=None):
    agg = defaultdict(lambda: defaultdict(int))
    for e in entries:
        prompt = e["prompt"]
        chain = e.get("reasoning")
        o1, rel, o2 = parse_spatial_prompt(prompt)
        if not chain or rel is None:
            continue
        cobjs = parse_chain(chain)
        if len(cobjs) < 2:
            continue
        key = (rel, e.get(variant_key, "-")) if variant_key else (rel,)
        a = agg[key]
        a["n"] += 1

        check = SPATIAL_RELATIONS[rel]
        a["naive_valid"] += int(check(cobjs[0]["center"], cobjs[1]["center"]))

        m = match_roles(cobjs, o1, o2)
        if m is None:
            a["unmatched"] += 1
        else:
            i1, i2 = m
            a["matched_n"] += 1
            a["matched_valid"] += int(check(cobjs[i1]["center"], cobjs[i2]["center"]))
            a["mentions_o2_first"] += int(i2 < i1)
        # raster stats: is the FIRST-mentioned chain object placed left/top?
        c0, c1_ = cobjs[0]["center"], cobjs[1]["center"]
        a["first_is_left"] += int(c0[0] < c1_[0])
        a["first_is_top"] += int(c0[1] < c1_[1])

    print(f"\n=== {label} ===")
    hdr = f"{'relation':18s} {'var':9s} {'n':>4s} {'naive%':>7s} {'matched%':>9s} " \
          f"{'o2-first%':>10s} {'1st-left%':>10s} {'1st-top%':>9s} {'unmat':>6s}"
    print(hdr)
    for key in sorted(agg):
        a = agg[key]
        rel = key[0]
        var = key[1] if len(key) > 1 else "-"
        mn = a["matched_n"] or 1
        print(f"{rel:18s} {var:9s} {a['n']:4d} "
              f"{100*a['naive_valid']/a['n']:7.1f} "
              f"{100*a['matched_valid']/mn:9.1f} "
              f"{100*a['mentions_o2_first']/mn:10.1f} "
              f"{100*a['first_is_left']/a['n']:10.1f} "
              f"{100*a['first_is_top']/a['n']:9.1f} "
              f"{a['unmatched']:6d}")


ap = argparse.ArgumentParser()
ap.add_argument("--chains", default=None)
ap.add_argument("--order_swap", default=None)
args = ap.parse_args()

if args.chains:
    analyze(json.load(open(args.chains)), f"planner chains: {args.chains}")
if args.order_swap:
    analyze(json.load(open(args.order_swap)),
            f"order-swap: {args.order_swap}", variant_key="variant")
if not args.chains and not args.order_swap:
    ap.error("give --chains and/or --order_swap")
