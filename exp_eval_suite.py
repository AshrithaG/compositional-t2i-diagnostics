"""
Generic diagnostic BLIP-VQA evaluation over any number of condition folders.
Replicates the EXACT question formats and scoring of vm_blip_eval.py (spatial)
and vm_blip_eval_color.py (color) so scores are comparable with all prior runs.

Spatial (per prompt): obj1 presence, obj2 presence, "Is the {obj1} {rel} the
{obj2}?", combined = s1*s2*s3. Color (per color-object pair): presence,
"Is the {obj} {color}?", open-ended "What color is the {obj}?" match;
combined = avg_presence * avg_color_yn.

Usage (subset inferred from dir name containing 'color', or force with =color/=spatial):
  python exp_eval_suite.py --dirs r3_outputs/perturb_none_spatial \
      r3_outputs/perturb_box_swap_spatial r3_outputs/oracle_verbose_empirical_color_val \
      "some_dir=color"
Outputs: eval_out/<dirname>.json per condition + eval_out/summary.md table.
"""

import argparse, json, os, re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--dirs", nargs="+", required=True)
parser.add_argument("--prompt_dir", default="T2I-CompBench/examples/dataset")
parser.add_argument("--split", default="val")
parser.add_argument("--out_dir", default="eval_out")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device("cuda")

from transformers import BlipProcessor, BlipForQuestionAnswering
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base", torch_dtype=torch.float16).to(device).eval()
print(f"BLIP loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

SPATIAL_PATTERNS = [
    "on the left of", "on the right of", "on top of", "on the bottom of",
    "on the side of", "above", "below", "beneath", "under", "near", "next to",
]
COLORS = ["red", "blue", "green", "brown", "black", "white", "yellow", "gold",
          "gray", "grey", "pink", "purple", "silver", "orange"]


def ask_blip(image, question):
    inputs = blip_processor(image, question, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_new_tokens=10)
    answer = blip_processor.decode(output[0], skip_special_tokens=True).strip().lower()
    if answer in ["yes", "true", "correct", "right"]:
        return 1.0, answer
    if answer in ["no", "false", "wrong", "incorrect"]:
        return 0.0, answer
    return 0.5, answer


def ask_blip_open(image, question):
    inputs = blip_processor(image, question, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_new_tokens=10)
    return blip_processor.decode(output[0], skip_special_tokens=True).strip().lower()


def parse_spatial_prompt(prompt):
    pl = prompt.lower()
    for pattern in SPATIAL_PATTERNS:
        if pattern in pl:
            parts = pl.split(pattern)
            if len(parts) == 2:
                return (parts[0].strip().lstrip("a ").strip(), pattern,
                        parts[1].strip().lstrip("a ").strip())
    return None, None, None


def parse_color_prompt(prompt):
    pattern = r'a\s+(' + '|'.join(COLORS) + r')\s+(\w+(?:\s+\w+)?)'
    pairs = []
    for color, obj in re.findall(pattern, prompt.lower()):
        obj = obj.strip()
        if obj in ["and", "or", "with", "the", "in", "on"]:
            continue
        pairs.append({"color": color, "object": obj})
    return pairs


def eval_spatial(img_dir, prompts):
    results = []
    for idx, prompt in enumerate(tqdm(prompts, desc=os.path.basename(img_dir))):
        img_path = os.path.join(img_dir, f"{idx}.png")
        obj1, rel, obj2 = parse_spatial_prompt(prompt)
        if not os.path.exists(img_path) or obj1 is None:
            continue
        img = Image.open(img_path).convert("RGB")
        s1, _ = ask_blip(img, f"Is there a {obj1} in the image?")
        s2, _ = ask_blip(img, f"Is there a {obj2} in the image?")
        s3, _ = ask_blip(img, f"Is the {obj1} {rel} the {obj2}?")
        results.append({"index": idx, "prompt": prompt, "relation": rel,
                        "obj1_present": s1, "obj2_present": s2,
                        "spatial_correct": s3, "combined": s1 * s2 * s3})
    summary = {
        "n": len(results),
        "obj1_presence": float(np.mean([r["obj1_present"] for r in results])),
        "obj2_presence": float(np.mean([r["obj2_present"] for r in results])),
        "spatial_accuracy": float(np.mean([r["spatial_correct"] for r in results])),
        "combined": float(np.mean([r["combined"] for r in results])),
    }
    per_rel = defaultdict(list)
    for r in results:
        per_rel[r["relation"]].append(r["spatial_correct"])
    summary["per_relation"] = {k: float(np.mean(v)) for k, v in per_rel.items()}
    return summary, results


def eval_color(img_dir, prompts):
    results = []
    for idx, prompt in enumerate(tqdm(prompts, desc=os.path.basename(img_dir))):
        img_path = os.path.join(img_dir, f"{idx}.png")
        pairs = parse_color_prompt(prompt)
        if not os.path.exists(img_path) or len(pairs) < 2:
            continue
        img = Image.open(img_path).convert("RGB")
        pres, cyn, cmatch = [], [], []
        for pair in pairs:
            obj, color = pair["object"], pair["color"]
            s, _ = ask_blip(img, f"Is there a {obj} in the image?")
            pres.append(s)
            s, _ = ask_blip(img, f"Is the {obj} {color}?")
            cyn.append(s)
            predicted = ask_blip_open(img, f"What color is the {obj}?")
            cmatch.append(1.0 if color in predicted else 0.0)
        results.append({"index": idx, "prompt": prompt,
                        "avg_presence": float(np.mean(pres)),
                        "avg_color_yes_no": float(np.mean(cyn)),
                        "avg_color_match": float(np.mean(cmatch)),
                        "combined": float(np.mean(pres) * np.mean(cyn))})
    summary = {
        "n": len(results),
        "presence": float(np.mean([r["avg_presence"] for r in results])),
        "color_yes_no": float(np.mean([r["avg_color_yes_no"] for r in results])),
        "color_match": float(np.mean([r["avg_color_match"] for r in results])),
        "combined": float(np.mean([r["combined"] for r in results])),
    }
    return summary, results


prompt_cache = {}
rows = []
for spec in args.dirs:
    if "=" in spec:
        d, subset = spec.split("=")
    else:
        d = spec
        subset = "color" if "color" in os.path.basename(d) else "spatial"
    name = os.path.basename(d.rstrip("/"))
    out_file = os.path.join(args.out_dir, f"{name}.json")
    if os.path.exists(out_file):
        print(f"skip {name} (already evaluated)")
        rows.append((name, subset, json.load(open(out_file))["summary"]))
        continue
    if subset not in prompt_cache:
        with open(os.path.join(args.prompt_dir, f"{subset}_{args.split}.txt")) as f:
            prompt_cache[subset] = [l.strip() for l in f if l.strip()]
    fn = eval_color if subset == "color" else eval_spatial
    summary, per_image = fn(d, prompt_cache[subset])
    json.dump({"dir": d, "subset": subset, "summary": summary,
               "per_image": per_image}, open(out_file, "w"), indent=2)
    rows.append((name, subset, summary))
    print(f"{name}: {json.dumps(summary, indent=1)[:400]}")

# summary table
lines = ["| condition | subset | " +
         "obj1 | obj2 | spatial | combined | presence | color_y/n | color_match |",
         "|---|---|---|---|---|---|---|---|---|"]
for name, subset, s in rows:
    if subset == "spatial":
        lines.append(f"| {name} | spatial | {s['obj1_presence']:.4f} | "
                     f"{s['obj2_presence']:.4f} | {s['spatial_accuracy']:.4f} | "
                     f"{s['combined']:.4f} | — | — | — |")
    else:
        lines.append(f"| {name} | color | — | — | — | {s['combined']:.4f} | "
                     f"{s['presence']:.4f} | {s['color_yes_no']:.4f} | "
                     f"{s['color_match']:.4f} |")
open(os.path.join(args.out_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nWrote {args.out_dir}/summary.md")
