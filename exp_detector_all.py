"""
P0: Extended detector pass — store ALL OWLv2 detections (not best-per-query)
plus box sizes, per image, for every condition directory given.

This is the data layer for the Phase F / "Command vs. Prior" analyses, all of
which run CPU-only afterward:
  - duplication/count analysis: #detections per object class vs condition
    (does "a pig" yield two pigs? does margin repair reduce duplicates?)
  - size-prior analysis: detected box area vs planned box area per category
    (does the decoder obey planned SIZE, or do semantic priors override it?)

Detections are stored raw (class, score, box) above --det_threshold with NO
best-box selection, alongside the planned boxes parsed from the chain, so any
future matching/analysis policy can be applied without re-running the GPU.

Usage:
  python exp_detector_all.py --chains_file reasoning_chains_spatial.json --dirs \
      r3_outputs/perturb_none_spatial r3_outputs/margin_repair_m0.2_spatial_val \
      r3_outputs/oracle2_minimal_alien_spatial_val r3_outputs/perturb_box_swap_spatial
Outputs: detector_all/<dirname>.json
"""

import argparse, json, os, re
import torch
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dirs", nargs="+", required=True)
parser.add_argument("--prompt_dir", default="T2I-CompBench/examples/dataset")
parser.add_argument("--split", default="val")
parser.add_argument("--chains_file", default=None,
                    help="fallback chains JSON for dirs without reasoning in results.json")
parser.add_argument("--out_dir", default="detector_all")
parser.add_argument("--det_threshold", type=float, default=0.1)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device("cuda")

from transformers import Owlv2Processor, Owlv2ForObjectDetection
proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
det = Owlv2ForObjectDetection.from_pretrained(
    "google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16).to(device).eval()
print(f"OWLv2 loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

BOX_RE = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
OBJ_RE = r'<\|obj_start\|>(.*?)<\|obj_end\|>'
ART = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)

with open(os.path.join(args.prompt_dir, f"spatial_{args.split}.txt")) as f:
    PROMPTS = [l.strip() for l in f if l.strip()]

fallback = {}
if args.chains_file:
    fallback = {e["index"]: e["reasoning"] for e in json.load(open(args.chains_file))}


def planned(chain):
    objs = re.findall(OBJ_RE, chain)
    boxes = re.findall(BOX_RE, chain)
    out = []
    for o, b in zip(objs, boxes):
        x1, y1, x2, y2 = (int(v) for v in b)
        out.append({"object": ART.sub("", o.strip()), "bbox": [x1, y1, x2, y2],
                    "area_frac": max(0, (x2 - x1)) * max(0, (y2 - y1)) / 1e6})
    return out


@torch.inference_mode()
def detect_all(img, queries):
    inputs = proc(text=[queries], images=img, return_tensors="pt").to(device, torch.float16)
    out = det(**inputs)
    res = proc.post_process_object_detection(
        out, threshold=args.det_threshold,
        target_sizes=torch.tensor([img.size[::-1]]))[0]
    W, H = img.size
    dets = []
    for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
        x1, y1, x2, y2 = [float(v) for v in box]
        dets.append({"query": queries[int(label)], "score": float(score),
                     "box": [x1, y1, x2, y2],
                     "area_frac": max(0, (x2 - x1)) * max(0, (y2 - y1)) / (W * H)})
    return dets


for d in args.dirs:
    name = os.path.basename(d.rstrip("/"))
    out_file = os.path.join(args.out_dir, f"{name}.json")
    if os.path.exists(out_file):
        print(f"skip {name} (done)")
        continue
    chains = dict(fallback)
    rj = os.path.join(d, "results.json")
    if os.path.exists(rj):
        for e in json.load(open(rj)):
            if e.get("reasoning"):
                chains[e["index"]] = e["reasoning"]
    rows = []
    for idx, prompt in enumerate(tqdm(PROMPTS, desc=name)):
        img_path = os.path.join(d, f"{idx}.png")
        chain = chains.get(idx)
        if not os.path.exists(img_path) or not chain:
            continue
        plan = planned(chain)
        if not plan:
            continue
        img = Image.open(img_path).convert("RGB")
        queries = sorted({p["object"] for p in plan})
        rows.append({"index": idx, "prompt": prompt, "planned": plan,
                     "detections": detect_all(img, queries)})
    json.dump({"dir": d, "threshold": args.det_threshold, "split": args.split,
               "per_image": rows}, open(out_file, "w"))
    n_det = sum(len(r["detections"]) for r in rows)
    print(f"{name}: {len(rows)} images, {n_det} raw detections stored")
print(f"Done. Analyses (count/size) run CPU-only from {args.out_dir}/ from now on.")
