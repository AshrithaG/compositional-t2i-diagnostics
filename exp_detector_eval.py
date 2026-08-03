"""
Detector-based spatial evaluation + Plan Execution Fidelity (Phase 5.5).

Uses OWLv2 (open-vocabulary detection, ships inside `transformers` — no
detectron2 needed) to evaluate every spatial condition geometrically:

  A. SPATIAL CORRECTNESS (detector-based, BLIP-yes-bias-free):
     detect obj1 and obj2 by name; score the prompt relation from detected
     box centers, mirroring the official UniDet protocol's geometry:
     both objects detected AND centers satisfy the relation -> 1, else 0.
     (Symmetric relations count as correct when both objects are detected.)

  B. PLAN EXECUTION FIDELITY (novel metric):
     when the condition folder's results.json (or a chains JSON) contains the
     conditioning chain, compare each object's PLANNED box (parsed from the
     chain, /1000-normalized) with its DETECTED box: per-object IoU + whether
     the detected-centers relation matches the planned-centers relation.
     This measures "did the decoder execute its own plan?" independent of the
     prompt.

Usage:
  python exp_detector_eval.py --dirs \
      r3_outputs/perturb_none_spatial r3_outputs/perturb_box_swap_spatial \
      r3_outputs/oracle_minimal_empirical_spatial_val ... \
      --out_dir detector_eval
Chains are read per-dir from <dir>/results.json ("reasoning" field); use
--chains_file to supply one for dirs lacking it (e.g., perturb_none uses the
same chains file as generation).
Outputs: detector_eval/<name>.json + detector_eval/summary.md
"""

import argparse, json, os, re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from exp_common import (parse_boxes_from_chain, parse_spatial_prompt,
                        SPATIAL_RELATIONS, ASYMMETRIC_RELATIONS)


def reconstruct_perturbed(chain, ptype, prompt=None):
    """Re-apply a deterministic r3 perturbation so fidelity is scored against
    the chain the decoder ACTUALLY saw (r3 results.json truncates chains)."""
    import re as _re
    if ptype == "chain_ablate":
        return None                      # no plan to be faithful to
    if ptype == "box_remove":
        return None                      # no boxes to compare against
    if ptype == "box_swap":
        parsed = parse_boxes_from_chain(chain)
        if len(parsed) < 2:
            return chain
        b1, b2 = parsed[0]["bbox"], parsed[1]["bbox"]
        s1 = f"({b1[0]},{b1[1]}),({b1[2]},{b1[3]})"
        s2 = f"({b2[0]},{b2[1]}),({b2[2]},{b2[3]})"
        return chain.replace(s1, "\x00B\x00").replace(s2, s1).replace("\x00B\x00", s2)
    if ptype == "chain_shuffle":
        segs = _re.findall(r'(<\|obj_start\|>.*?<\|box_end\|>)', chain, _re.DOTALL)
        if len(segs) >= 2:
            out = chain
            for i, s in enumerate(segs):
                out = out.replace(s, f"\x00S{i}\x00")
            for i, s in enumerate(segs[::-1]):
                out = out.replace(f"\x00S{i}\x00", s)
            return out
        return chain
    return chain                          # 'none' or unknown → unperturbed

parser = argparse.ArgumentParser()
parser.add_argument("--dirs", nargs="+", required=True)
parser.add_argument("--prompt_dir", default="T2I-CompBench/examples/dataset")
parser.add_argument("--chains_file", default=None,
                    help="fallback chains JSON for dirs without reasoning in results.json")
parser.add_argument("--out_dir", default="detector_eval")
parser.add_argument("--det_threshold", type=float, default=0.1)
parser.add_argument("--split", default="val")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device("cuda")

from transformers import Owlv2Processor, Owlv2ForObjectDetection
proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
det = Owlv2ForObjectDetection.from_pretrained(
    "google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16).to(device).eval()
print(f"OWLv2 loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

with open(os.path.join(args.prompt_dir, f"spatial_{args.split}.txt")) as f:
    PROMPTS = [l.strip() for l in f if l.strip()]

fallback_chains = {}
if args.chains_file:
    fallback_chains = {e["index"]: e["reasoning"]
                       for e in json.load(open(args.chains_file))}


@torch.inference_mode()
def detect(img, queries):
    """Best box per query name. Returns {query: (x1,y1,x2,y2,score) or None} in pixels."""
    inputs = proc(text=[queries], images=img, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    out = det(**inputs)
    target_size = torch.tensor([img.size[::-1]], device=device)
    res = proc.post_process_object_detection(
        out, threshold=args.det_threshold, target_sizes=target_size)[0]
    best = {q: None for q in queries}
    for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
        q = queries[label]
        s = float(score)
        if best[q] is None or s > best[q][4]:
            b = [float(v) for v in box]
            best[q] = (b[0], b[1], b[2], b[3], s)
    return best


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def eval_dir(d):
    name = os.path.basename(d.rstrip("/"))
    res_file = os.path.join(d, "results.json")
    chains = dict(fallback_chains)
    if os.path.exists(res_file):
        for e in json.load(open(res_file)):
            if e.get("reasoning"):
                chains[e["index"]] = e["reasoning"]

    # perturbed conditions: r3 saved only truncated chains — reconstruct the
    # ACTUAL conditioning chain from the fallback file + the deterministic op
    m = re.match(r"perturb_(box_swap|box_remove|attr_swap|chain_shuffle|chain_ablate)",
                 name)
    if m:
        ptype = m.group(1)
        chains = {}
        for idx, ch in fallback_chains.items():
            prompt = PROMPTS[idx] if idx < len(PROMPTS) else None
            rc = reconstruct_perturbed(ch, ptype, prompt)
            if rc is not None:
                chains[idx] = rc

    rows = []
    for idx, prompt in enumerate(tqdm(PROMPTS, desc=name)):
        img_path = os.path.join(d, f"{idx}.png")
        o1, rel, o2 = parse_spatial_prompt(prompt)
        if not os.path.exists(img_path) or rel is None:
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        found = detect(img, [o1, o2])
        b1, b2 = found[o1], found[o2]

        # A: detector-based prompt correctness
        both = b1 is not None and b2 is not None
        _, check = SPATIAL_RELATIONS.get(rel, (None, lambda a, b: True))
        rel_ok = bool(both and check(center(b1), center(b2)))

        row = {"index": idx, "prompt": prompt, "relation": rel,
               "obj1_detected": b1 is not None, "obj2_detected": b2 is not None,
               "detected_box1": list(b1[:4]) if b1 else None,
               "detected_box2": list(b2[:4]) if b2 else None,
               "relation_correct": rel_ok}

        # B: plan execution fidelity vs the conditioning chain.
        # followed_planned_relation only for ASYMMETRIC relations — symmetric
        # predicates are constant-true and would auto-score 1.0.
        chain = chains.get(idx)
        if chain and both:
            planned = parse_boxes_from_chain(chain)
            if len(planned) >= 2:
                p1 = [v / 1000 * (W if i % 2 == 0 else H)
                      for i, v in enumerate(planned[0]["bbox"])]
                p2 = [v / 1000 * (W if i % 2 == 0 else H)
                      for i, v in enumerate(planned[1]["bbox"])]
                row["iou_obj1"] = iou(p1, b1[:4])
                row["iou_obj2"] = iou(p2, b2[:4])
                if rel in ASYMMETRIC_RELATIONS:
                    # does the image realize the PLAN's left/right (etc.)
                    # ordering, regardless of the prompt?
                    _, ck = SPATIAL_RELATIONS[rel]
                    planned_says = ck(planned[0]["center"], planned[1]["center"])
                    image_says = ck(center(b1), center(b2))
                    row["followed_planned_relation"] = bool(planned_says == image_says)
        rows.append(row)

    ious = [r["iou_obj1"] for r in rows if "iou_obj1" in r] + \
           [r["iou_obj2"] for r in rows if "iou_obj2" in r]
    followed = [r["followed_planned_relation"] for r in rows
                if "followed_planned_relation" in r]
    summary = {
        "n": len(rows),
        "obj1_det_rate": float(np.mean([r["obj1_detected"] for r in rows])),
        "obj2_det_rate": float(np.mean([r["obj2_detected"] for r in rows])),
        "relation_correct": float(np.mean([r["relation_correct"] for r in rows])),
        "mean_plan_iou": float(np.mean(ious)) if ious else None,
        "followed_planned_relation": float(np.mean(followed)) if followed else None,
        "n_fidelity_scored": len(followed),
    }
    per_rel = defaultdict(list)
    for r in rows:
        per_rel[r["relation"]].append(r["relation_correct"])
    summary["per_relation"] = {k: float(np.mean(v)) for k, v in per_rel.items()}
    json.dump({"dir": d, "summary": summary, "per_image": rows},
              open(os.path.join(args.out_dir, f"{name}.json"), "w"), indent=2)
    return name, summary


rows = []
for d in args.dirs:
    name = os.path.basename(d.rstrip("/"))
    out_file = os.path.join(args.out_dir, f"{name}.json")
    if os.path.exists(out_file):
        print(f"skip {name} (done)")
        rows.append((name, json.load(open(out_file))["summary"]))
        continue
    rows.append(eval_dir(d))

lines = ["| condition | det1 | det2 | relation✓ | plan IoU | followed plan |",
         "|---|---|---|---|---|---|"]
for name, s in rows:
    fp = f"{s['followed_planned_relation']:.4f}" if s["followed_planned_relation"] is not None else "—"
    pi = f"{s['mean_plan_iou']:.4f}" if s["mean_plan_iou"] is not None else "—"
    lines.append(f"| {name} | {s['obj1_det_rate']:.3f} | {s['obj2_det_rate']:.3f} | "
                 f"{s['relation_correct']:.4f} | {pi} | {fp} |")
open(os.path.join(args.out_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nWrote {args.out_dir}/summary.md")
